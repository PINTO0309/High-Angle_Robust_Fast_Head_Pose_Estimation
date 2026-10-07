#!/usr/bin/env python3
"""Train the DINOv3 ViT teacher (320x320, biternion yaw).

Per-VRAM-tier settings (--vram 8|16|96) are separated into scripts/vram_presets.py:
  - The effective batch (micro_batch × grad_accum = 64) is common to all tiers,
    so the training dynamics do not change between tiers and --resume across
    tiers is possible (the resume consistency check uses the effective batch).
  - Only the 8GB tier trains just the last 8 backbone blocks + the head
    (so the AdamW optimizer state fits in memory). 16/96GB train all blocks.

Recipe (follows the HRFFA teacher):
  - epoch 0 keeps the backbone frozen (head only); afterwards the preset range
    is unfrozen
  - differential learning rates: backbone 2e-5 / head 2e-4, bf16, grad_clip 1.0
  - input normalization is ImageNet (a DINOv3 requirement; differs from the
    center05 used by the students)

lr schedule (--lr-schedule):
  - cosine (default): warmup → cosine decay ending at ≈ 0. It depends on the
    total epoch count, so epochs cannot be changed on resume.
  - wsd: warmup → constant lr (peak) → cosine decay → 0 over the last
    --decay-epochs. The constant phase does not depend on the total epoch count,
    so --epochs can be rewritten on resume to extend/shorten training (setting
    epochs = current epoch + decay_epochs enters decay immediately).
    All lengths are measured in optimizer steps and match across VRAM tiers.

Checkpoints (runs/<run_name>/) follow the same convention as train_yawnet.py.
Note that last.pt includes the optimizer state and grows to several GB.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dinov3_yaw import Dinov3YawNet
from ema import Ema
from yawnet import von_mises_loss, von_mises_nll, von_mises_nll_per_sample
from yaw_dataset import YawDataset, make_balanced_sampler, angular_error_deg
from train_yawnet import (epoch_print, make_lr_fn, rng_states,
                          restore_rng_states)
from val_preview import render_val_preview, select_indices
from vram_presets import get_preset

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str,
             amp_dtype: torch.dtype, pitch: bool = False) -> dict[str, Any]:
    """Return validation metrics. "sel" is the best-selection key: equal to maae
    for yaw-only, (maae + pitch_maae) / 2 for the joint pitch teacher. Pitch
    metrics are computed only over rows with an intent pitch (pitch_valid=1)."""
    model.eval()
    errs: list[torch.Tensor] = []
    yaws: list[torch.Tensor] = []
    p_errs: list[torch.Tensor] = []
    p_gts: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="val", dynamic_ncols=True, leave=False):
        x = batch[0].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            if pitch:
                pred, _, pitch_pred, _ = model.forward_full(x)
            else:
                pred = model(x)
        errs.append(angular_error_deg(pred.float().cpu(), batch[2]))
        yaws.append(batch[2])
        if pitch:
            valid = batch[5] > 0.5
            if valid.any():
                p_errs.append(angular_error_deg(
                    pitch_pred.float().cpu()[valid], batch[4][valid]))
                p_gts.append(batch[4][valid])
    errs_t = torch.cat(errs)
    yaws_t = torch.cat(yaws)
    per_bin: dict[str, float] = {}
    for b in range(0, 360, 30):
        m = (yaws_t >= b) & (yaws_t < b + 30)
        if m.any():
            per_bin[f"{b:03d}-{b+30:03d}"] = round(errs_t[m].mean().item(), 2)
    metrics: dict[str, Any] = {
        "maae": round(errs_t.mean().item(), 3),
        "median": round(errs_t.median().item(), 3),
        "acc15": round((errs_t <= 15).float().mean().item() * 100, 2),
        "acc30": round((errs_t <= 30).float().mean().item() * 100, 2),
        "per_bin_mae": per_bin,
    }
    if pitch and p_errs:
        pe = torch.cat(p_errs)
        pg = torch.cat(p_gts)
        p_bin: dict[str, float] = {}
        for b in range(-120, 120, 30):
            m = (pg >= b) & (pg < b + 30)
            if m.any():
                p_bin[f"{b:+04d}-{b+30:+04d}"] = round(pe[m].mean().item(), 2)
        metrics.update({
            "pitch_maae": round(pe.mean().item(), 3),
            "pitch_median": round(pe.median().item(), 3),
            "pitch_acc15": round((pe <= 15).float().mean().item() * 100, 2),
            "pitch_n": int(pe.numel()),
            "pitch_per_bin_mae": p_bin,
        })
        metrics["sel"] = round((metrics["maae"] + metrics["pitch_maae"]) / 2, 3)
    else:
        metrics["sel"] = metrics["maae"]
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vram", type=int, default=8, choices=[8, 16, 96],
                    help="machine VRAM tier (batch size etc. resolved by vram_presets.py)")
    ap.add_argument("--variant", type=str, default="vitl16",
                    choices=["vits16", "vitb16", "vitl16"])
    ap.add_argument("--dinov3-ckpt", type=str, default="",
                    help="DINOv3 pretrained weights .pth (default: the one in HRFFA/ckpts)")
    ap.add_argument("--init-ckpt", type=str, default="",
                    help="warm start from an existing run's best_*.pt (or its run "
                         "directory). If omitted, start from DINOv3 pretrained weights")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--lr-schedule", type=str, default="cosine",
                    choices=["cosine", "wsd"],
                    help="wsd = warmup → constant lr → cosine decay in last decay-epochs")
    ap.add_argument("--decay-epochs", type=int, default=5,
                    help="decay span of wsd (in epochs). Ignored for cosine")
    ap.add_argument("--freeze-epochs", type=int, default=1,
                    help="fully freeze the backbone for this many epochs (head only)")
    ap.add_argument("--kappa-head", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="κ (confidence) head + von Mises NLL (default since v5; "
                         "--no-kappa-head uses the former fixed-κ loss)")
    ap.add_argument("--pitch-head", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="joint yaw+pitch teacher (6 outputs). pitch contributes to the "
                         "loss only for rows with an intent label (s001 is masked). "
                         "Requires the κ head")
    ap.add_argument("--pitch-weight", type=float, default=1.0,
                    help="loss weight λ of the pitch NLL (loss = yaw + λ·pitch)")
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="EMA decay rate (0 disables). Eval and best saving use EMA weights")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "yawpose"))
    ap.add_argument("--balance", type=str, default="inv",
                    choices=["sqrt", "inv", "none"])
    ap.add_argument("--balance-bin", type=int, default=10)
    ap.add_argument("--balance-max-ratio", type=float, default=20.0)
    ap.add_argument("--unified", action="store_true",
                    help="train on the unified train + val set (intentional data leak "
                         "accepted; val metrics become selection-only reference values)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.pitch_head and not args.kappa_head:
        raise SystemExit("--pitch-head requires --kappa-head (fixed 6 outputs)")

    preset = get_preset("dinov3_teacher", args.vram)
    amp_dtype = torch.bfloat16 if preset.amp_dtype == "bf16" else torch.float16
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    run_name = (f"dinov3_{args.variant}_{args.size:03d}"
                f"{'_yp' if args.pitch_head else ''}"
                f"{'_unified' if args.unified else ''}"
                f"{('_' + args.tag) if args.tag else ''}")
    run_dir = ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / "last.pt"

    data = Path(args.data)
    train_ds = YawDataset(data, "unified" if args.unified else "train",
                          args.size, train=True, input_norm="imagenet",
                          with_pitch=args.pitch_head)
    val_ds = YawDataset(data, "val", args.size, train=False,
                        input_norm="imagenet", with_pitch=args.pitch_head)
    sampler = make_balanced_sampler(train_ds, bin_deg=args.balance_bin,
                                    mode=args.balance,
                                    max_ratio=args.balance_max_ratio)
    train_ld = DataLoader(train_ds, batch_size=preset.micro_batch, sampler=sampler,
                          num_workers=preset.workers, pin_memory=True,
                          drop_last=True, persistent_workers=True)
    val_ld = DataLoader(val_ds, batch_size=max(16, preset.micro_batch),
                        shuffle=False, num_workers=preset.workers, pin_memory=True)

    if args.init_ckpt:
        init_path = Path(args.init_ckpt)
        if init_path.is_dir():
            cands = sorted(init_path.glob("best_*.pt"))
            if not cands:
                raise FileNotFoundError(f"no best_*.pt found in {init_path}")
            init_path = cands[0]
        ick = torch.load(init_path, map_location="cpu", weights_only=False)
        if ick.get("model_type") != "dinov3_yaw" or ick.get("variant") != args.variant:
            raise ValueError(
                f"--init-ckpt must be a dinov3_yaw checkpoint of the same variant: "
                f"type={ick.get('model_type')}, variant={ick.get('variant')}")
        model = Dinov3YawNet(args.variant, pretrained=False,
                             kappa_head=args.kappa_head,
                             pitch_head=args.pitch_head)
        sd = ick["model"]
        old_rows = sd["head.3.weight"].shape[0]
        new_rows = model.head[-1].weight.shape[0]
        if old_rows != new_rows:
            # ckpt without κ → model with κ (or vice versa): copy the common rows only
            with torch.no_grad():
                n = min(old_rows, new_rows)
                model.head[-1].weight[:n] = sd["head.3.weight"][:n]
                model.head[-1].bias[:n] = sd["head.3.bias"][:n]
            sd = {k: v for k, v in sd.items()
                  if k not in ("head.3.weight", "head.3.bias")}
            model.load_state_dict(sd, strict=False)
            print(f"warm start: head rows {old_rows}->{new_rows} ({n} common rows copied)")
        else:
            model.load_state_dict(sd)
        model.to(device)
        print(f"warm start: {init_path} (val_maae="
              f"{(ick.get('metrics') or {}).get('maae')})")
    else:
        model = Dinov3YawNet(args.variant,
                             ckpt_path=args.dinov3_ckpt or None,
                             kappa_head=args.kappa_head,
                             pitch_head=args.pitch_head).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    model.set_trainable_blocks(preset.trainable_blocks)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"run={run_name} vram={args.vram}GB micro_batch={preset.micro_batch} "
          f"accum={preset.grad_accum} (effective {preset.effective_batch}) "
          f"amp={preset.amp_dtype}")
    print(f"params total={n_params:,} trainable={n_train:,} "
          f"(blocks={'all' if preset.trainable_blocks is None else preset.trainable_blocks}) "
          f"train={len(train_ds)} val={len(val_ds)}")

    opt = torch.optim.AdamW([
        {"params": model.backbone_parameters(), "lr": args.lr_backbone},
        {"params": model.head_parameters(), "lr": args.lr_head},
    ], weight_decay=args.wd)
    if args.lr_schedule == "wsd" and not (0 <= args.decay_epochs <= args.epochs):
        raise ValueError(f"--decay-epochs must be within 0..epochs: {args.decay_epochs}")
    opt_steps_per_epoch = len(train_ld) // preset.grad_accum
    total_steps = args.epochs * opt_steps_per_epoch
    warmup_steps = args.warmup * opt_steps_per_epoch
    decay_steps = args.decay_epochs * opt_steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, make_lr_fn(args.lr_schedule, warmup_steps, total_steps, decay_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=(preset.amp_dtype == "fp16"))
    ema = Ema(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    start_epoch = 0
    best: dict[str, Any] = {"maae": 1e9, "sel": 1e9}
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"--resume given but {last_path} does not exist")
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        # wsd's constant phase does not depend on the total epoch count, so changing
        # epochs / decay_epochs on resume (extend/shorten/enter decay now) is allowed
        check_keys = ["variant", "size", "lr_backbone", "lr_head", "wd",
                      "kappa", "warmup", "freeze_epochs",
                      "kappa_head", "pitch_head", "pitch_weight",
                      "unified",
                      "balance", "balance_bin", "balance_max_ratio"]
        if args.lr_schedule == "cosine":
            check_keys += ["epochs", "decay_epochs"]
        for k in check_keys:
            # keys absent in old checkpoints (pitch_head etc.) are treated as CLI defaults
            ck_val = ck["args"].get(k, ap.get_default(k))
            if getattr(args, k) != ck_val:
                raise ValueError(
                    f"hyperparameter mismatch on resume: {k} "
                    f"(checkpoint={ck_val}, now={getattr(args, k)})")
        if ck["effective_batch"] != preset.effective_batch:
            raise ValueError(
                f"effective batch mismatch on resume: checkpoint={ck['effective_batch']}, "
                f"now={preset.effective_batch} (cross-tier resume assumes the same "
                f"effective batch)")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        # lr_schedule / ema_decay may be switched on resume (applied from the current step)
        if args.lr_schedule != ck["args"].get("lr_schedule", "cosine"):
            print(f"note: switching lr_schedule {ck['args'].get('lr_schedule', 'cosine')}"
                  f" → {args.lr_schedule} (new schedule applies from the current step)")
        if args.ema_decay != ck["args"].get("ema_decay", 0.0):
            print(f"note: switching ema_decay {ck['args'].get('ema_decay', 0.0)}"
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
        best.setdefault("sel", best.get("maae", 1e9))  # old checkpoint compat
        print(f"resumed from {last_path} (next epoch {start_epoch}, "
              f"best sel {best['sel']})")

    log_path = run_dir / "train_log.jsonl"
    preview_indices = select_indices(val_ds)
    resumed = args.resume
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        # freeze schedule: head only below freeze_epochs, then unfreeze the preset range
        if epoch < args.freeze_epochs:
            model.set_trainable_blocks(0)
        else:
            model.set_trainable_blocks(preset.trainable_blocks)

        model.train()
        running = 0.0
        kappa_sum = 0.0
        pitch_kappa_sum = 0.0
        opt.zero_grad(set_to_none=True)
        bar = tqdm(train_ld, desc=f"ep {epoch}/{args.epochs - 1}",
                   dynamic_ncols=True, leave=False)
        for step, batch in enumerate(bar, 1):
            x = batch[0].to(device, non_blocking=True)
            target = batch[1].to(device, non_blocking=True)
            if args.pitch_head:
                p_target = batch[3].to(device, non_blocking=True)
                p_valid = batch[5].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=amp_dtype):
                    pred, kappa, p_pred, p_kappa = model.forward_full(x)
                loss = von_mises_nll(pred.float(), kappa.float(), target)
                # pitch is averaged only over rows with an intent label (s001 is masked)
                nll_p = von_mises_nll_per_sample(
                    p_pred.float(), p_kappa.float(), p_target)
                loss = loss + args.pitch_weight * (
                    (nll_p * p_valid).sum() / p_valid.sum().clamp(min=1.0))
                kappa_sum += kappa.mean().item()
                pitch_kappa_sum += p_kappa.mean().item()
            elif args.kappa_head:
                with torch.autocast("cuda", dtype=amp_dtype):
                    pred, kappa = model.forward_with_kappa(x)
                loss = von_mises_nll(pred.float(), kappa.float(), target)
                kappa_sum += kappa.mean().item()
            else:
                with torch.autocast("cuda", dtype=amp_dtype):
                    pred = model(x)
                loss = von_mises_loss(pred.float(), target, kappa=args.kappa)
            scaler.scale(loss / preset.grad_accum).backward()
            running += loss.item()
            if step % preset.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    args.grad_clip)
                # when fp16 scaling skipped the optimizer step, do not step the scheduler
                # (with bf16 the scaler is disabled, so no skip occurs and it always steps)
                prev_scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if scaler.get_scale() >= prev_scale:
                    sched.step()
                if ema is not None:
                    ema.update(model)
            bar.set_postfix(loss=f"{running / step:.4f}",
                            lr=f"{opt.param_groups[1]['lr']:.2e}")

        # evaluation and best saving use the EMA weights (when enabled)
        if ema is not None:
            ema.apply_to(model)
        metrics = evaluate(model, val_ld, device, amp_dtype,
                           pitch=args.pitch_head)
        improved = metrics["sel"] < best["sel"]
        if improved:
            best = {**metrics, "epoch": epoch}
            for old in run_dir.glob("best_*.pt"):
                old.unlink()
            # ema.apply_to() is active, so state_dict holds the EMA weights (when enabled)
            torch.save({"model": model.state_dict(), "metrics": metrics,
                        "epoch": epoch, "args": vars(args), "params": n_params,
                        "model_type": "dinov3_yaw", "variant": args.variant,
                        "kappa_head": args.kappa_head,
                        "pitch_head": args.pitch_head,
                        "weights": "ema" if ema is not None else "raw"},
                       run_dir / f"best_{metrics['sel']:.6f}.pt")
            try:
                render_val_preview(model, val_ds, preview_indices,
                                   run_dir / "val_preview_best.png", device,
                                   epoch, metrics["maae"],
                                   pitch_maae=metrics.get("pitch_maae"),
                                   amp_dtype=amp_dtype)
            except Exception as e:  # noqa: BLE001  do not stop training on preview failure
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
            "effective_batch": preset.effective_batch,
            "model_type": "dinov3_yaw",
            "variant": args.variant,
            "kappa_head": args.kappa_head,
            "pitch_head": args.pitch_head,
            **({"ema": ema.state_dict()} if ema is not None else {}),
        }, last_path)

        with open(log_path, "a") as f:
            f.write(json.dumps({
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "epoch": epoch,
                "loss": round(running / len(train_ld), 6),
                "lr_head": opt.param_groups[1]["lr"],
                **({"kappa_mean": round(kappa_sum / len(train_ld), 3)}
                   if args.kappa_head else {}),
                **({"pitch_kappa_mean": round(pitch_kappa_sum / len(train_ld), 3)}
                   if args.pitch_head else {}),
                **metrics,
                "best_maae": best["maae"],
                "best_sel": best["sel"],
                "best_epoch": best.get("epoch"),
                "elapsed_sec": round(time.time() - t0, 1),
                **({"resumed": True} if resumed and epoch == start_epoch else {}),
            }) + "\n")

        # with the joint pitch head, prefix the yaw metrics with y_ so they read as pairs
        yp = "y_" if args.pitch_head else ""
        epoch_print(
            f"ep {epoch:3d} loss {running/len(train_ld):.4f} "
            + (f"{yp}kappa {kappa_sum/len(train_ld):5.1f} "
               if args.kappa_head else "")
            + f"{yp}maae {metrics['maae']:6.2f} {yp}med {metrics['median']:6.2f} "
            f"{yp}acc15 {metrics['acc15']:5.1f} {yp}acc30 {metrics['acc30']:5.1f} "
            + (f"p_maae {metrics.get('pitch_maae', float('nan')):6.2f} "
               f"p_kappa {pitch_kappa_sum/len(train_ld):5.1f} "
               if args.pitch_head else "")
            + f"(best {best['sel']:.2f}@{best.get('epoch','-')}) "
            f"{time.time()-t0:6.0f}s", improved)

    with open(run_dir / "result.json", "w") as f:
        json.dump({"params": n_params, "best": best, "args": vars(args)}, f, indent=2)
    print("BEST:", json.dumps(best))


if __name__ == "__main__":
    main()
