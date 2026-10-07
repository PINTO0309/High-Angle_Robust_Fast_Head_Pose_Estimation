#!/usr/bin/env python3
"""Train YawNet (biternion cos/sin yaw regressor) on data/yawpose.

Checkpoints (runs/<run_name>/):
  - last.pt   : full checkpoint overwritten at the end of every epoch
                (model / optimizer / scaler / scheduler / epoch / best / RNG state).
                --resume restores every parameter and resumes training.
  - best_{maae:.6f}.pt : saved only when the validation MAAE improves (weights and
                metadata only). The old best file is deleted so that only the
                single best one is kept.
  - train_log.jsonl : training log with 1 line per epoch (loss / lr / validation
                metrics / best / elapsed seconds). --resume keeps appending (the
                resume boundary is recorded as resumed=true).
  - result.json : final result summary.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ema import Ema
from yawnet import YawNet, von_mises_loss, von_mises_nll
from yaw_dataset import YawDataset, make_balanced_sampler, angular_error_deg
from val_preview import render_val_preview, select_indices
from vram_presets import get_preset

ROOT = Path(__file__).resolve().parent.parent


def make_lr_fn(schedule: str, warmup_steps: int, total_steps: int,
               decay_steps: int):
    """Return the lr multiplier function (per optimizer step).

    cosine: after warmup, cosine decay 1 → 0 over the whole remaining span.
    wsd   : after warmup, constant at 1.0, then cosine decay 1 → 0 over the last
            decay_steps (decay_steps=0 means no decay = warmup + stable only).
    """
    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if schedule == "cosine":
            p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
        # wsd
        decay_start = total_steps - decay_steps
        if step < decay_start or decay_steps == 0:
            return 1.0
        p = (step - decay_start) / max(1, decay_steps)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return lr_at


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, Any]:
    model.eval()
    errs: list[torch.Tensor] = []
    yaws: list[torch.Tensor] = []
    for x, _, yaw in tqdm(loader, desc="val", dynamic_ncols=True, leave=False):
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model(x.to(device, non_blocking=True))
        errs.append(angular_error_deg(pred.float().cpu(), yaw))
        yaws.append(yaw)
    errs_t = torch.cat(errs)
    yaws_t = torch.cat(yaws)
    per_bin: dict[str, float] = {}
    for b in range(0, 360, 30):
        m = (yaws_t >= b) & (yaws_t < b + 30)
        if m.any():
            per_bin[f"{b:03d}-{b+30:03d}"] = round(errs_t[m].mean().item(), 2)
    return {
        "maae": round(errs_t.mean().item(), 3),
        "median": round(errs_t.median().item(), 3),
        "acc15": round((errs_t <= 15).float().mean().item() * 100, 2),
        "acc30": round((errs_t <= 30).float().mean().item() * 100, 2),
        "per_bin_mae": per_bin,
    }


def epoch_print(line: str, improved: bool) -> None:
    """Print the epoch summary; green when best improved (colored only on a TTY)."""
    if improved and sys.stdout.isatty():
        line = f"\033[32m{line}\033[0m"
    print(line, flush=True)


def rng_states() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_states(s: dict[str, Any]) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    torch.cuda.set_rng_state_all(s["cuda"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64,
                    choices=[64, 96, 128, 192, 224, 320],
                    help="input resolution (192 and above are mainly intended for "
                         "training the distillation teacher)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--vram", type=int, default=8, choices=[8, 16, 96],
                    help="machine VRAM tier (batch/accum resolved in vram_presets.py)")
    ap.add_argument("--batch", type=int, default=0,
                    help="micro batch override (0 = follow preset; accum becomes 1)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--lr-schedule", type=str, default="cosine",
                    choices=["cosine", "wsd"],
                    help="wsd = warmup → constant lr → cosine decay over the last "
                         "decay-epochs (--epochs can be changed on resume)")
    ap.add_argument("--decay-epochs", type=int, default=5,
                    help="decay span of wsd (in epochs); ignored for cosine")
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help="EMA decay rate (0 = disabled; when enabled, eval"
                         " and best saving use the EMA weights; e.g. 0.999)")
    ap.add_argument("--grad-clip", type=float, default=0.0,
                    help="gradient norm clipping (0 = none)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "yawpose"))
    ap.add_argument("--balance", type=str, default="inv",
                    choices=["sqrt", "inv", "none"],
                    help="mode of yaw-bin frequency balanced sampling")
    ap.add_argument("--balance-bin", type=int, default=10,
                    help="bin width of the balanced sampling (degrees)")
    ap.add_argument("--balance-max-ratio", type=float, default=20.0,
                    help="upper bound of the max/min ratio of sample weights")
    ap.add_argument("--kappa-head", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="κ (confidence) head + von Mises NLL"
                         " (--no-kappa-head for the conventional fixed-κ loss)")
    ap.add_argument("--unified", action="store_true",
                    help="train on train + val merged (intentional data leak accepted; "
                         "val metrics become reference values for selection only)")
    ap.add_argument("--resume", action="store_true",
                    help="resume with full state restored from runs/<run_name>/last.pt")
    args = ap.parse_args()

    preset = get_preset("yawnet_small" if args.size <= 128 else "yawnet_320",
                        args.vram)
    if args.batch > 0:                      # no accumulation on explicit override
        micro_batch, grad_accum = args.batch, 1
    else:
        micro_batch, grad_accum = preset.micro_batch, preset.grad_accum
    effective_batch = micro_batch * grad_accum

    device = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    run_name = (f"yawnet_{args.size:03d}{'_unified' if args.unified else ''}"
                f"{('_' + args.tag) if args.tag else ''}")
    run_dir = ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / "last.pt"

    data = Path(args.data)
    train_ds = YawDataset(data, "unified" if args.unified else "train",
                          args.size, train=True)
    val_ds = YawDataset(data, "val", args.size, train=False)
    sampler = make_balanced_sampler(train_ds, bin_deg=args.balance_bin,
                                    mode=args.balance,
                                    max_ratio=args.balance_max_ratio)
    train_ld = DataLoader(train_ds, batch_size=micro_batch, sampler=sampler,
                          num_workers=args.workers, pin_memory=True,
                          drop_last=True, persistent_workers=True)
    # shrink the validation batch with resolution (512 at 320px OOMs on activations)
    val_batch = min(512, max(32, int(512 * (128 / args.size) ** 2)))
    val_ld = DataLoader(val_ds, batch_size=val_batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = YawNet(width=args.width, kappa_head=args.kappa_head).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"run={run_name} params={n_params:,} train={len(train_ds)} val={len(val_ds)} "
          f"vram={args.vram}GB micro_batch={micro_batch} accum={grad_accum} "
          f"(effective {effective_batch})")

    if args.lr_schedule == "wsd" and not (0 <= args.decay_epochs <= args.epochs):
        raise ValueError(f"--decay-epochs must be in 0..epochs: {args.decay_epochs}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = len(train_ld) // grad_accum
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup * steps_per_epoch
    decay_steps = args.decay_epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, make_lr_fn(args.lr_schedule, warmup_steps, total_steps, decay_steps))
    scaler = torch.amp.GradScaler("cuda")
    ema = Ema(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    start_epoch = 0
    best: dict[str, Any] = {"maae": 1e9}
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"--resume given but {last_path} does not exist")
        # load on CPU because the RNG state (ByteTensor) must be restored on CPU.
        # load_state_dict moves the model / optimizer state to the parameter device.
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        # wsd allows changing epochs / decay_epochs because its constant span does
        # not depend on the total epoch count (fixed for cosine)
        check_keys = ["size", "width", "lr", "wd", "kappa", "warmup",
                      "unified", "kappa_head",
                      "balance", "balance_bin", "balance_max_ratio"]
        if args.lr_schedule == "cosine":
            check_keys += ["epochs", "decay_epochs"]
        for k in check_keys:
            # keys absent from old checkpoints (lr_schedule etc.) count as CLI defaults
            ck_val = ck["args"].get(k, ap.get_default(k))
            if getattr(args, k) != ck_val:
                raise ValueError(
                    f"hyperparameter mismatch on resume: {k} "
                    f"(checkpoint={ck_val}, now={getattr(args, k)})")
        # the micro batch may differ across tiers, but the effective batch must match
        ck_eff = ck.get("effective_batch", ck["args"].get("batch"))
        if ck_eff != effective_batch:
            raise ValueError(
                f"effective batch mismatch on resume: checkpoint={ck_eff}, "
                f"now={effective_batch} (cross-tier resume needs same effective batch)")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        # lr_schedule / ema_decay may be switched on resume (applied from current step)
        if args.lr_schedule != ck["args"].get("lr_schedule", "cosine"):
            print(f"note: lr_schedule changed {ck['args'].get('lr_schedule', 'cosine')}"
                  f" → {args.lr_schedule} (new schedule applies from the current step)")
        if args.ema_decay != ck["args"].get("ema_decay", 0.0):
            print(f"note: ema_decay changed {ck['args'].get('ema_decay', 0.0)}"
                  f" → {args.ema_decay}")
        if ema is not None:
            if "ema" in ck:
                ema.load_state_dict(ck["ema"])
                for v in ema.shadow.values():
                    v.data = v.data.to(device)
            else:
                # EMA enabled mid-run: start the EMA from the loaded current weights
                ema = Ema(model, decay=args.ema_decay)
        restore_rng_states(ck["rng"])
        start_epoch = ck["epoch"] + 1
        best = ck["best"]
        print(f"resumed from {last_path} (next epoch {start_epoch}, "
              f"best maae {best['maae']})")

    log_path = run_dir / "train_log.jsonl"
    preview_indices = select_indices(val_ds)
    resumed = args.resume
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        kappa_sum = 0.0
        opt.zero_grad(set_to_none=True)
        bar = tqdm(train_ld, desc=f"ep {epoch}/{args.epochs - 1}",
                   dynamic_ncols=True, leave=False)
        for step, (x, target, _) in enumerate(bar, 1):
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if args.kappa_head:
                with torch.autocast("cuda", dtype=torch.float16):
                    pred, kappa = model.forward_with_kappa(x)
                loss = von_mises_nll(pred.float(), kappa.float(), target)
                kappa_sum += kappa.mean().item()
            else:
                with torch.autocast("cuda", dtype=torch.float16):
                    pred = model(x)
                loss = von_mises_loss(pred.float(), target, kappa=args.kappa)
            scaler.scale(loss / grad_accum).backward()
            running += loss.item()
            if step % grad_accum == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.grad_clip)
                # do not step the scheduler when fp16 scaling skipped the optimizer step
                # (a skip is detected by the scaler shrinking its scale)
                prev_scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if scaler.get_scale() >= prev_scale:
                    sched.step()
                if ema is not None:
                    ema.update(model)
            bar.set_postfix(loss=f"{running / step:.4f}",
                            lr=f"{opt.param_groups[0]['lr']:.2e}")

        # evaluate and save best with the EMA weights (when enabled)
        if ema is not None:
            ema.apply_to(model)
        metrics = evaluate(model, val_ld, device)
        improved = metrics["maae"] < best["maae"]
        if improved:
            best = {**metrics, "epoch": epoch}
            for old in run_dir.glob("best_*.pt"):
                old.unlink()
            torch.save({"model": model.state_dict(), "metrics": metrics,
                        "epoch": epoch, "args": vars(args), "params": n_params,
                        "weights": "ema" if ema is not None else "raw"},
                       run_dir / f"best_{metrics['maae']:.6f}.pt")
            try:
                render_val_preview(model, val_ds, preview_indices,
                                   run_dir / "val_preview_best.png", device,
                                   epoch, metrics["maae"])
            except Exception as e:  # noqa: BLE001  keep training if the preview fails
                print(f"val preview failed: {e}")
        if ema is not None:
            ema.restore(model)

        torch.save({
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": rng_states(),
            "epoch": epoch,
            "best": best,
            "metrics": metrics,
            "args": vars(args),
            "params": n_params,
            "effective_batch": effective_batch,
            **({"ema": ema.state_dict()} if ema is not None else {}),
        }, last_path)

        with open(log_path, "a") as f:
            f.write(json.dumps({
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "epoch": epoch,
                "loss": round(running / len(train_ld), 6),
                "lr": opt.param_groups[0]["lr"],
                **({"kappa_mean": round(kappa_sum / len(train_ld), 3)}
                   if args.kappa_head else {}),
                **metrics,
                "best_maae": best["maae"],
                "best_epoch": best.get("epoch"),
                "elapsed_sec": round(time.time() - t0, 1),
                **({"resumed": True} if resumed and epoch == start_epoch else {}),
            }) + "\n")

        epoch_print(
            f"ep {epoch:3d} loss {running/len(train_ld):.4f} "
            + (f"kappa {kappa_sum/len(train_ld):5.1f} " if args.kappa_head else "")
            + f"maae {metrics['maae']:6.2f} med {metrics['median']:6.2f} "
            f"acc15 {metrics['acc15']:5.1f} acc30 {metrics['acc30']:5.1f} "
            f"(best {best['maae']:.2f}@{best.get('epoch','-')}) "
            f"{time.time()-t0:6.0f}s", improved)

    with open(run_dir / "result.json", "w") as f:
        json.dump({"params": n_params, "best": best, "args": vars(args)}, f, indent=2)
    print("BEST:", json.dumps(best))


if __name__ == "__main__":
    main()
