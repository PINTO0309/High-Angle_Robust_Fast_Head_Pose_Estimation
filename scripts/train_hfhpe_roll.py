#!/usr/bin/env python3
"""Train the HFHPE roll branch (biternion cos/sin roll regressor, full 360-degree).

Trains the roll estimation branch of HFHPE (High-Angle Robust Fast Head Pose
Estimation, scripts/hfhpe.py). The model is the existing YawNet used as is
(width 0.5 default, ≈ 0.2M), and the GT is obtained exactly from the full-surround
rotation augmentation of RollDataset (no additional synthetic data is needed).

--arch vitt trains the ViT-T/16 variant and --arch vitl the DINOv3 ViT-L/16 variant
(ceiling for the roll teacher; --size 320 recommended, and the differential lr,
bf16 and backbone freezing for the first epochs follow the same recipe as
train_teacher_dinov3.py). Passing --teacher <roll run> turns training into online
distillation with alpha*KD + beta*GT (the teacher may be any yawnet / vitt / vitl
checkpoint; when the teacher's training resolution is larger than the student's,
warp and degradation are applied once at the teacher resolution and the student
gets a downscaled copy, i.e. a same-condition pair scheme).

--extra-data mixes extra data roots of "derotated real-photo crops" into train
(any root that has <root>/manifest.jsonl or labels_fixed.jsonl and <root>/images/).
The GT is the applied rotation θ, so no pose labels are needed.
val stays the val split of --data (yawpitchpose) to keep metric continuity with
existing runs.

The checkpoint / resume / log conventions are the same as train_yawnet.py
(runs/<run_name>/last.pt / best_{maae:.6f}.pt / train_log.jsonl).
All metrics such as maae are circular roll errors. No balanced sampler is needed
(the rotation θ is generated uniformly).
"""
import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ema import Ema
from roll_dataset import RollDataset
from roll_models import build_roll_model, load_roll_checkpoint
from train_yawnet import (epoch_print, evaluate, make_lr_fn,
                          restore_rng_states, rng_states)
from vram_presets import get_preset
from yawnet import von_mises_loss, von_mises_nll

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64,
                    choices=[64, 96, 128, 224, 320],
                    help="input resolution (roll is a global tilt, so 64 is assumed "
                         "to be enough; 320 recommended for the vitl teacher)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--vram", type=int, default=8, choices=[8, 16, 96])
    ap.add_argument("--batch", type=int, default=0,
                    help="explicit micro batch override (0 = yawnet_small preset)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--arch", type=str, default="yawnet",
                    choices=["yawnet", "vitt", "vitl"],
                    help="model architecture. vitt = ViT-T/16 (initialized from "
                         "ckpts/vitt_distill.pt), vitl = DINOv3 ViT-L/16 (initialized "
                         "from the DINOv3 weights in ckpts). The arch is added to the "
                         "run name")
    ap.add_argument("--variant", type=str, default="vitl16",
                    help="vitl only: DINOv3 variant")
    ap.add_argument("--lr-backbone", type=float, default=2e-5,
                    help="vitl only: lr for the backbone (--lr is for the head; "
                         "2e-4 recommended)")
    ap.add_argument("--freeze-epochs", type=int, default=1,
                    help="vitl only: freeze the backbone for the first N epochs "
                         "(head only)")
    ap.add_argument("--pretrained", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="vitt / vitl only: start from pretrained weights (disabled "
                         "automatically with --resume)")
    ap.add_argument("--teacher", type=str, default="",
                    help="roll checkpoint to distill from (run directory or .pt, "
                         "either yawnet or vitt architecture). When given, training "
                         "becomes online distillation with alpha*KD + beta*GT")
    ap.add_argument("--init-student", type=str, default="",
                    help="initial weights of the trained model (run directory of a "
                         "roll checkpoint or .pt). The architecture (arch / width / "
                         "kappa-head) must match. For vitt this replaces the ImageNet "
                         "initialization. Ignored with --resume (last.pt has priority)")
    ap.add_argument("--teacher-chunk", type=int, default=-1,
                    help="chunk size for the teacher forward (-1 = VRAM preset, "
                         "0 = all at once). An OOM workaround for running a ViT-L "
                         "teacher on small VRAM; does not affect the training result")
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="weight of the distillation loss (only with --teacher)")
    ap.add_argument("--beta", type=float, default=0.3,
                    help="weight of the GT loss (only with --teacher)")
    ap.add_argument("--width", type=float, default=0.5,
                    help="YawNet width multiplier (0.5 ≈ 0.2M parameters)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--lr-schedule", type=str, default="cosine",
                    choices=["cosine", "wsd"],
                    help="wsd = warmup → constant lr → cosine decay over the last "
                         "decay-epochs (--epochs can be changed on resume)")
    ap.add_argument("--decay-epochs", type=int, default=5,
                    help="decay interval of wsd (in epochs). Ignored for cosine")
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help="EMA decay rate (0 = disabled. When enabled, evaluation and "
                         "best saving use the EMA weights. e.g. 0.999)")
    ap.add_argument("--grad-clip", type=float, default=0.0,
                    help="gradient norm clipping (0 = none)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "yawpitchpose"))
    ap.add_argument("--extra-data", action="append", default=[],
                    help="extra image roots mixed into train (repeatable). Reads the "
                         "image field of <root>/manifest.jsonl or labels_fixed.jsonl "
                         "(upright crops assumed). Not mixed into val")
    ap.add_argument("--kappa-head", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--unified", action="store_true",
                    help="train on train + val merged (intentional data leak accepted)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    is_vitl = args.arch == "vitl"

    device = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    arch_tag = "" if args.arch == "yawnet" else f"{args.arch}_"
    run_name = (f"hfhpe_roll_{arch_tag}{args.size:03d}"
                f"{'_unified' if args.unified else ''}"
                f"{('_' + args.tag) if args.tag else ''}")
    run_dir = ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / "last.pt"

    teacher = None
    teacher_size = args.size
    if args.teacher:
        teacher, tck, tpath = load_roll_checkpoint(args.teacher)
        teacher.to(device)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher_size = int(tck["args"].get("size", 64))
        if teacher_size < args.size:
            raise SystemExit(f"teacher resolution {teacher_size} is smaller than "
                             f"the student's {args.size}")
        tm = tck.get("metrics") or {}
        print(f"teacher: {tpath} (arch={tck['args'].get('arch', 'yawnet')}, "
              f"size={teacher_size}, val maae={tm.get('maae')}) "
              f"-> KD alpha={args.alpha} / GT beta={args.beta}")
    # when the teacher is higher-resolution, render at the teacher resolution and
    # pass a downscaled copy to the student
    render_size = teacher_size

    # VRAM preset: dinov3_teacher for vitl training, roll_kd_vitl (chunked teacher
    # forward) for distillation from a vitl teacher, yawnet_small otherwise
    if is_vitl:
        task = "dinov3_teacher"
    elif teacher is not None and tck["args"].get("arch", "yawnet") == "vitl":
        task = "roll_kd_vitl"
    else:
        task = "yawnet_small"
    preset = get_preset(task, args.vram)
    if args.batch > 0:
        micro_batch, grad_accum = args.batch, 1
    else:
        micro_batch, grad_accum = preset.micro_batch, preset.grad_accum
    effective_batch = micro_batch * grad_accum
    teacher_chunk = (args.teacher_chunk if args.teacher_chunk >= 0
                     else preset.teacher_chunk)

    data = Path(args.data)
    base_ds = RollDataset(data, "unified" if args.unified else "train",
                          render_size, train=True)
    def _extra_split(root: Path) -> str:
        """Auto-detect the format of an extra data root.

        Accepts either manifest.jsonl (output of the real-photo crop conversion
        tool) or the yawpitchpose-compatible layout (labels_fixed.jsonl =
        train+val). Since the roll GT is the applied rotation θ, both are
        treated the same way, as a "list of upright crop images".
        """
        if (root / "manifest.jsonl").exists():
            return "manifest"
        if (root / "labels_fixed.jsonl").exists():
            return "labels_fixed"
        raise FileNotFoundError(
            f"{root} has neither manifest.jsonl nor labels_fixed.jsonl")

    extras = [RollDataset(Path(p), _extra_split(Path(p)), render_size,
                          train=True)
              for p in args.extra_data]
    train_ds: torch.utils.data.Dataset = (
        torch.utils.data.ConcatDataset([base_ds, *extras]) if extras else base_ds)
    if extras:
        mix = " + ".join(f"{Path(p).name}={len(e)}"
                         for p, e in zip(args.extra_data, extras))
        print(f"train mix: {data.name}={len(base_ds)} + {mix}")
    val_ds = RollDataset(data, "val", args.size, train=False)
    train_ld = DataLoader(train_ds, batch_size=micro_batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True,
                          drop_last=True, persistent_workers=True)
    # vitl runs ViT-L at 320px, so the validation batch is reduced (same rule as
    # the yaw/pitch teacher)
    val_batch = max(16, micro_batch) if is_vitl else 512
    val_ld = DataLoader(val_ds, batch_size=val_batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model: torch.nn.Module = build_roll_model(
        args.arch, width=args.width, kappa_head=args.kappa_head,
        variant=args.variant,
        pretrained=(args.pretrained and not args.resume
                    and not args.init_student)).to(device)

    if args.init_student and not args.resume:
        ipath = Path(args.init_student)
        if ipath.is_dir():
            cands = sorted(ipath.glob("best_*.pt"))
            if not cands:
                raise FileNotFoundError(f"no best_*.pt found in {ipath}")
            ipath = cands[0]
        ick = torch.load(ipath, map_location="cpu", weights_only=False)
        if ick.get("model_type") not in ("hfhpe_roll", "hyprnet_roll"):
            raise SystemExit(f"--init-student accepts roll checkpoints only: "
                             f"model_type={ick.get('model_type')}")
        i_arch = ick["args"].get("arch", "yawnet")
        if i_arch != args.arch or (
                args.arch == "yawnet"
                and float(ick["args"].get("width", 0.5)) != args.width):
            raise SystemExit(
                f"--init-student architecture mismatch: checkpoint has arch={i_arch} "
                f"width={ick['args'].get('width', 0.5)}, requested arch={args.arch} "
                f"width={args.width}")
        model.load_state_dict(ick["model"])
        im = ick.get("metrics") or {}
        print(f"init-student: warm-start from {ipath} (val maae={im.get('maae')})")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"run={run_name} params={n_params:,} train={len(train_ds)} "
          f"val={len(val_ds)} vram={args.vram}GB micro_batch={micro_batch} "
          f"accum={grad_accum} (effective {effective_batch}) "
          f"val_batch={val_batch}"
          + (f" teacher_chunk={teacher_chunk or 'all'}"
             if teacher is not None else ""))

    if args.lr_schedule == "wsd" and not (0 <= args.decay_epochs <= args.epochs):
        raise ValueError(f"--decay-epochs must be in 0..epochs: {args.decay_epochs}")
    if is_vitl:
        opt = torch.optim.AdamW([
            {"params": model.backbone_parameters(), "lr": args.lr_backbone},
            {"params": model.head_parameters(), "lr": args.lr},
        ], weight_decay=args.wd)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.wd)
    steps_per_epoch = len(train_ld) // grad_accum
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup * steps_per_epoch
    decay_steps = args.decay_epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, make_lr_fn(args.lr_schedule, warmup_steps, total_steps, decay_steps))
    # vitl applies bf16 inside the model, so GradScaler is disabled
    scaler = torch.amp.GradScaler("cuda", enabled=not is_vitl)
    ema = Ema(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    start_epoch = 0
    best: dict[str, Any] = {"maae": 1e9}
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"--resume given but {last_path} does not exist")
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        check_keys = ["size", "width", "lr", "wd", "kappa", "warmup",
                      "unified", "kappa_head", "extra_data",
                      "arch", "teacher", "alpha", "beta", "init_student",
                      "variant", "lr_backbone", "freeze_epochs"]
        if args.lr_schedule == "cosine":
            check_keys += ["epochs", "decay_epochs"]
        for k in check_keys:
            # keys absent from old checkpoints (e.g. lr_schedule) use the CLI default
            ck_val = ck["args"].get(k, ap.get_default(k))
            if getattr(args, k) != ck_val:
                raise ValueError(
                    f"hyperparameter mismatch on resume: {k} "
                    f"(checkpoint={ck_val}, now={getattr(args, k)})")
        if ck["effective_batch"] != effective_batch:
            raise ValueError(
                f"effective batch mismatch on resume: "
                f"checkpoint={ck['effective_batch']}, now={effective_batch}")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        # lr_schedule / ema_decay may be switched on resume (applied from the
        # current step)
        if args.lr_schedule != ck["args"].get("lr_schedule", "cosine"):
            print(f"note: lr_schedule {ck['args'].get('lr_schedule', 'cosine')}"
                  f" → {args.lr_schedule} (switched; new schedule from current step)")
        if args.ema_decay != ck["args"].get("ema_decay", 0.0):
            print(f"note: ema_decay {ck['args'].get('ema_decay', 0.0)}"
                  f" → {args.ema_decay} (switched)")
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
    resumed = args.resume
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        if is_vitl:
            # freeze schedule: head only before freeze_epochs, then the preset range
            model.set_trainable_blocks(
                0 if epoch < args.freeze_epochs else preset.trainable_blocks)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        running = 0.0
        kappa_sum = 0.0
        opt.zero_grad(set_to_none=True)
        bar = tqdm(train_ld, desc=f"ep {epoch}/{args.epochs - 1}",
                   dynamic_ncols=True, leave=False)
        for step, (x, target, _) in enumerate(bar, 1):
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            t_pred = None
            if teacher is not None:
                with torch.no_grad(), \
                        torch.autocast("cuda", dtype=torch.float16):
                    if 0 < teacher_chunk < x.shape[0]:
                        t_pred = torch.cat([
                            teacher(x[i:i + teacher_chunk])
                            for i in range(0, x.shape[0], teacher_chunk)])
                    else:
                        t_pred = teacher(x)
                if render_size != args.size:
                    x = F.interpolate(x, size=(args.size, args.size),
                                      mode="area")
            if args.kappa_head:
                with torch.autocast("cuda", dtype=torch.float16):
                    pred, kappa = model.forward_with_kappa(x)
                loss = von_mises_nll(pred.float(), kappa.float(), target)
                if t_pred is not None:
                    loss = (args.alpha * von_mises_nll(
                                pred.float(), kappa.float(),
                                t_pred.float().detach())
                            + args.beta * loss)
                kappa_sum += kappa.mean().item()
            else:
                with torch.autocast("cuda", dtype=torch.float16):
                    pred = model(x)
                loss = von_mises_loss(pred.float(), target, kappa=args.kappa)
                if t_pred is not None:
                    loss = (args.alpha * von_mises_loss(
                                pred.float(), t_pred.float().detach(),
                                kappa=args.kappa)
                            + args.beta * loss)
            scaler.scale(loss / grad_accum).backward()
            running += loss.item()
            if step % grad_accum == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.grad_clip)
                prev_scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if scaler.get_scale() >= prev_scale:
                    sched.step()
                if ema is not None:
                    ema.update(model)
            bar.set_postfix(loss=f"{running / step:.4f}",
                            lr=f"{opt.param_groups[-1]['lr']:.2e}")

        # evaluation and best saving use the EMA weights (when enabled)
        if ema is not None:
            ema.apply_to(model)
        metrics = evaluate(model, val_ld, device)   # all metrics: circular roll error
        improved = metrics["maae"] < best["maae"]
        if improved:
            best = {**metrics, "epoch": epoch}
            for old in run_dir.glob("best_*.pt"):
                old.unlink()
            torch.save({"model": model.state_dict(), "metrics": metrics,
                        "epoch": epoch, "args": vars(args), "params": n_params,
                        "model_type": "hfhpe_roll",
                        "weights": "ema" if ema is not None else "raw"},
                       run_dir / f"best_{metrics['maae']:.6f}.pt")
        if ema is not None:
            ema.restore(model)
        vram_peak = torch.cuda.max_memory_allocated() / 2 ** 30

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
            "model_type": "hfhpe_roll",
            **({"ema": ema.state_dict()} if ema is not None else {}),
        }, last_path)

        with open(log_path, "a") as f:
            f.write(json.dumps({
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "epoch": epoch,
                "loss": round(running / len(train_ld), 6),
                "lr": opt.param_groups[-1]["lr"],
                **({"kappa_mean": round(kappa_sum / len(train_ld), 3)}
                   if args.kappa_head else {}),
                **metrics,
                "best_maae": best["maae"],
                "best_epoch": best.get("epoch"),
                "elapsed_sec": round(time.time() - t0, 1),
                "vram_peak_gb": round(vram_peak, 2),
                **({"resumed": True} if resumed and epoch == start_epoch else {}),
            }) + "\n")

        epoch_print(
            f"ep {epoch:3d} loss {running/len(train_ld):.4f} "
            + (f"kappa {kappa_sum/len(train_ld):5.1f} " if args.kappa_head else "")
            + f"maae {metrics['maae']:6.2f} med {metrics['median']:6.2f} "
            f"acc15 {metrics['acc15']:5.1f} acc30 {metrics['acc30']:5.1f} "
            f"(best {best['maae']:.2f}@{best.get('epoch','-')}) "
            f"vram {vram_peak:.1f}G {time.time()-t0:6.0f}s", improved)

    with open(run_dir / "result.json", "w") as f:
        json.dump({"params": n_params, "best": best, "args": vars(args)}, f, indent=2)
    print("BEST:", json.dumps(best))


if __name__ == "__main__":
    main()
