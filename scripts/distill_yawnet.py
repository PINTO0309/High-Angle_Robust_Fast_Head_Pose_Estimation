#!/usr/bin/env python3
"""Distillation training: 320x320 teacher → low-resolution student.

Method (same-condition paired distillation):
  - DistillYawDataset returns 2 views of identical content, with the geometric
    transform + degradation (photometric/lowres/blur/erase) applied once on the
    320 side (teacher = degraded 320, student = its downscaled copy; identical
    conditions except for resolution)
  - The teacher runs online inference under no_grad every step (it sees the
    augmented view, so its outputs cannot be precomputed)
  - loss = alpha * vonMises(student, teacher output) + beta * vonMises(student, GT)

The teacher checkpoint is runs/<teacher_run>/best_*.pt from train_yawnet.py
(--teacher accepts either the directory or the .pt file).

Checkpoints (runs/<run_name>/) follow the same convention as train_yawnet.py:
  last.pt (full state, fully restored by --resume) / best_{maae:.6f}.pt (best only) /
  train_log.jsonl / result.json
"""
import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ema import Ema
from yawnet import YawNet, von_mises_loss, von_mises_nll, von_mises_nll_per_sample
from yaw_dataset import (DistillYawDataset, YawDataset, make_balanced_sampler)
from train_yawnet import (epoch_print, evaluate, make_lr_fn, rng_states,
                          restore_rng_states)
from train_teacher_dinov3 import evaluate as evaluate_yp
from val_preview import render_val_preview, select_indices
from vram_presets import get_preset

ROOT = Path(__file__).resolve().parent.parent


def load_teacher(spec: str, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Restore the teacher from the run directory or .pt specified by --teacher.

    The YawNet teacher / DINOv3 teacher is auto-detected from the checkpoint's
    model_type. "norm" in the returned info is the input normalization required
    for the teacher view.
    """
    path = Path(spec)
    if path.is_dir():
        cands = sorted(path.glob("best_*.pt"))
        if not cands:
            raise FileNotFoundError(f"no best_*.pt found in {path}")
        path = cands[0]   # by convention only one best file is ever kept
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model_type = ck.get("model_type", "yawnet")
    if model_type == "dinov3_yaw":
        from dinov3_yaw import Dinov3YawNet, head_flags  # noqa: PLC0415
        # The yaw block is the shared first 3 rows even in a 6-output (yaw+pitch)
        # teacher, so yaw distillation via forward / forward_with_kappa works as is
        kappa_head, pitch_head = head_flags(ck["model"]["head.3.weight"].shape[0])
        model: torch.nn.Module = Dinov3YawNet(ck["variant"], pretrained=False,
                                              kappa_head=kappa_head,
                                              pitch_head=pitch_head)
        norm = "imagenet"                    # required by DINOv3
        width = None
    elif model_type == "vitt_yaw":
        from dinov3_yaw import head_flags  # noqa: PLC0415
        from vitt_yaw import VittYawNet  # noqa: PLC0415
        kappa_head, pitch_head = head_flags(ck["model"]["head.3.weight"].shape[0])
        model = VittYawNet(pretrained=False, kappa_head=kappa_head,
                           pitch_head=pitch_head)
        norm = "center05"
        width = None
    elif model_type == "hgnetv2_yaw":
        from dinov3_yaw import head_flags  # noqa: PLC0415
        from hgnetv2_yaw import HGNetV2Yaw  # noqa: PLC0415
        kappa_head, pitch_head = head_flags(ck["model"]["fc.weight"].shape[0])
        model = HGNetV2Yaw(kappa_head=kappa_head, pitch_head=pitch_head)
        norm = "center05"
        width = None
    else:
        from dinov3_yaw import head_flags  # noqa: PLC0415
        width = float(ck["args"].get("width", 1.0))
        kappa_head, pitch_head = head_flags(ck["model"]["fc.weight"].shape[0])
        model = YawNet(width=width, kappa_head=kappa_head, pitch_head=pitch_head)
        norm = "center05"
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    info = {"path": str(path), "type": model_type, "width": width, "norm": norm,
            "size": ck["args"].get("size"), "metrics": ck.get("metrics")}
    return model, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=str, required=True,
                    help="teacher run directory (runs/yawnet_320_*) or best_*.pt")
    ap.add_argument("--student-size", type=int, default=96, choices=[64, 96, 128])
    ap.add_argument("--teacher-size", type=int, default=320)
    ap.add_argument("--student-arch", type=str, default="yawnet",
                    choices=["yawnet", "vitt", "hgnetv2"],
                    help="student architecture: yawnet (CNN lite 0.77M, scratch) / "
                         "vitt (ViT-T 5.6M, initialized from ckpts/vitt_distill.pt; "
                         "input normalization is folded into center05 at load time) / "
                         "hgnetv2 (regular CNN ~1.9M, PP-HGNetV2 B0 pretrained "
                         "backbone + YawNet-convention head; center05 input, "
                         "out_stride16 gives 4x4 features from 64px)")
    ap.add_argument("--width", type=float, default=1.0,
                    help="student width (yawnet only; ignored for vitt/hgnetv2)")
    ap.add_argument("--backbone-ckpt", type=str,
                    default=str(ROOT / "ckpts" / "PPHGNetV2_B0_stage1.pth"),
                    help="pretrained backbone weights for the hgnetv2 student"
                         " (empty string = scratch init; --init-student /"
                         " --resume take precedence when given)")
    ap.add_argument("--init-student", type=str, default="",
                    help="warm-start from a trained student checkpoint (run "
                         "directory or .pt); counterpart of --init-ckpt in teacher "
                         "training. Same arch and head config only. With --resume, "
                         "last.pt takes precedence and this is effectively ignored")
    ap.add_argument("--alpha", type=float, default=0.7, help="distillation loss weight")
    ap.add_argument("--beta", type=float, default=0.3, help="GT loss weight")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--vram", type=int, default=8, choices=[8, 16, 96],
                    help="machine VRAM tier (batch/accum resolved in vram_presets.py)")
    ap.add_argument("--batch", type=int, default=0,
                    help="micro batch override (0 = follow preset; accum becomes 1)")
    ap.add_argument("--lr", type=float, default=3e-3,
                    help="learning rate (head-side lr when --lr-backbone is set). "
                         "For the vitt student ~2e-4 is recommended (3e-3 breaks "
                         "the pretrained ViT)")
    ap.add_argument("--lr-backbone", type=float, default=0.0,
                    help="separate backbone lr for vitt / hgnetv2 students (0 = single "
                         "lr); kept low so the pretrained backbone is not broken. "
                         "HRFFA proven config: --lr 2e-4 --lr-backbone 2e-5")
    ap.add_argument("--grad-clip", type=float, default=0.0,
                    help="gradient norm clipping (0 = none); 1.0 recommended for vitt")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--lr-schedule", type=str, default="cosine",
                    choices=["cosine", "wsd"],
                    help="wsd = warmup → constant lr → cosine decay over the last "
                         "decay-epochs (--epochs can be changed on resume)")
    ap.add_argument("--decay-epochs", type=int, default=5,
                    help="decay span of wsd (in epochs); ignored for cosine")
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help="student EMA decay rate (0 = disabled; when enabled, eval"
                         " and best saving use the EMA weights; e.g. 0.999)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "yawpose"))
    ap.add_argument("--balance", type=str, default="inv",
                    choices=["sqrt", "inv", "none"])
    ap.add_argument("--balance-bin", type=int, default=10)
    ap.add_argument("--balance-max-ratio", type=float, default=20.0)
    ap.add_argument("--kappa-head", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="student κ (confidence) head + von Mises NLL"
                         " (--no-kappa-head for the conventional fixed-κ loss)")
    ap.add_argument("--pitch-head", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="joint yaw+pitch student (6 outputs). Also requires a "
                         "6-output teacher (dinov3 teacher trained with --pitch-head). "
                         "κ head is mandatory")
    ap.add_argument("--pitch-weight", type=float, default=1.0,
                    help="pitch-term loss weight λ (yaw + λ·pitch for both kd and gt)")
    ap.add_argument("--unified", action="store_true",
                    help="train on train + val merged (intentional data leak accepted; "
                         "val metrics become reference values for selection only)")
    ap.add_argument("--resume", action="store_true",
                    help="resume with full state restored from runs/<run_name>/last.pt")
    args = ap.parse_args()
    if args.pitch_head and not args.kappa_head:
        raise SystemExit("--pitch-head requires --kappa-head (fixed 6 outputs)")

    preset = get_preset("distill", args.vram)
    if args.batch > 0:
        micro_batch, grad_accum = args.batch, 1
    else:
        micro_batch, grad_accum = preset.micro_batch, preset.grad_accum
    effective_batch = micro_batch * grad_accum

    device = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    run_name = (f"{args.student_arch}"
                f"_distill_{args.student_size:03d}"
                f"{'_yp' if args.pitch_head else ''}"
                f"{'_unified' if args.unified else ''}"
                f"{('_' + args.tag) if args.tag else ''}")
    run_dir = ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / "last.pt"

    teacher, teacher_info = load_teacher(args.teacher, device)
    if args.pitch_head and not getattr(teacher, "pitch_head", False):
        raise SystemExit("--pitch-head distillation needs a 6-output yaw+pitch teacher"
                         f" (the given teacher is yaw only: {teacher_info['path']})")
    print(f"teacher: {teacher_info['path']} (type={teacher_info['type']}, "
          f"norm={teacher_info['norm']}, trained@{teacher_info['size']}, "
          f"val={teacher_info['metrics'] and teacher_info['metrics'].get('maae')})")
    print(f"vram={args.vram}GB micro_batch={micro_batch} accum={grad_accum} "
          f"(effective {effective_batch})")

    data = Path(args.data)
    train_ds = DistillYawDataset(data, "unified" if args.unified else "train",
                                 args.student_size,
                                 teacher_size=args.teacher_size, train=True,
                                 teacher_norm=teacher_info["norm"],
                                 with_pitch=args.pitch_head)
    val_ds = YawDataset(data, "val", args.student_size, train=False,
                        with_pitch=args.pitch_head)
    sampler = make_balanced_sampler(train_ds, bin_deg=args.balance_bin,
                                    mode=args.balance,
                                    max_ratio=args.balance_max_ratio)
    train_ld = DataLoader(train_ds, batch_size=micro_batch, sampler=sampler,
                          num_workers=args.workers, pin_memory=True,
                          drop_last=True, persistent_workers=True)
    val_batch = min(512, max(32, int(512 * (128 / args.student_size) ** 2)))
    val_ld = DataLoader(val_ds, batch_size=val_batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    if args.student_arch == "vitt":
        from vitt_yaw import VittYawNet  # noqa: PLC0415
        # --init-student / --resume overwrite all weights in the later load,
        # so skip loading the ImageNet initial weights (ckpts/vitt_distill.pt)
        student: torch.nn.Module = VittYawNet(
            kappa_head=args.kappa_head, pitch_head=args.pitch_head,
            input_norm="center05",
            pretrained=not (args.init_student or args.resume)).to(device)
    elif args.student_arch == "hgnetv2":
        from hgnetv2_yaw import HGNetV2Yaw  # noqa: PLC0415
        # --init-student / --resume overwrite the backbone too in the later load
        student = HGNetV2Yaw(kappa_head=args.kappa_head,
                             pitch_head=args.pitch_head,
                             backbone_ckpt=("" if args.init_student or args.resume
                                            else args.backbone_ckpt)).to(device)
        if not (args.init_student or args.resume):
            print(f"backbone init: {args.backbone_ckpt or 'scratch'}")
    else:
        student = YawNet(width=args.width, kappa_head=args.kappa_head,
                         pitch_head=args.pitch_head).to(device)
    if args.init_student:
        init_path = Path(args.init_student)
        if init_path.is_dir():
            cands = sorted(init_path.glob("best_*.pt"))
            if not cands:
                raise FileNotFoundError(f"no best_*.pt found in {init_path}")
            init_path = cands[0]
        ick = torch.load(init_path, map_location="cpu", weights_only=False)
        want = {"vitt": "vitt_yaw", "hgnetv2": "hgnetv2_yaw",
                "yawnet": "yawnet"}[args.student_arch]
        if ick.get("model_type") != want:
            raise SystemExit(
                f"--init-student accepts only a student checkpoint of the same arch: "
                f"type={ick.get('model_type')} (expected: {want})")
        student.load_state_dict(ick["model"])
        im = ick.get("metrics") or {}
        print(f"warm start: {init_path} "
              f"(val sel={im.get('sel')}, maae={im.get('maae')})")
    n_params = sum(p.numel() for p in student.parameters())
    print(f"run={run_name} student_params={n_params:,} "
          f"train={len(train_ds)} val={len(val_ds)}")

    if args.lr_schedule == "wsd" and not (0 <= args.decay_epochs <= args.epochs):
        raise ValueError(f"--decay-epochs must be in 0..epochs: {args.decay_epochs}")
    if args.lr_backbone > 0:
        if args.student_arch not in ("vitt", "hgnetv2"):
            raise SystemExit("--lr-backbone is supported only with --student-arch "
                             "vitt / hgnetv2")
        head_params = (student.head.parameters() if args.student_arch == "vitt"
                       else student.fc.parameters())
        opt = torch.optim.AdamW([
            {"params": student.backbone.parameters(), "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr},
        ], weight_decay=args.wd)
    else:
        opt = torch.optim.AdamW(student.parameters(), lr=args.lr,
                                weight_decay=args.wd)
    steps_per_epoch = len(train_ld) // grad_accum
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup * steps_per_epoch
    decay_steps = args.decay_epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, make_lr_fn(args.lr_schedule, warmup_steps, total_steps, decay_steps))
    scaler = torch.amp.GradScaler("cuda")
    ema = Ema(student, decay=args.ema_decay) if args.ema_decay > 0 else None

    start_epoch = 0
    best: dict[str, Any] = {"maae": 1e9, "sel": 1e9}
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"--resume given but {last_path} does not exist")
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        check_keys = ["student_size", "teacher_size", "student_arch", "width",
                      "lr", "wd",
                      "kappa", "warmup", "alpha", "beta", "unified",
                      "kappa_head", "pitch_head", "pitch_weight",
                      "lr_backbone", "grad_clip",
                      "balance", "balance_bin", "balance_max_ratio"]
        if args.lr_schedule == "cosine":
            check_keys += ["epochs", "decay_epochs"]
        for k in check_keys:
            # keys absent from old checkpoints (pitch_head etc.) count as CLI defaults
            ck_val = ck["args"].get(k, ap.get_default(k))
            if getattr(args, k) != ck_val:
                raise ValueError(
                    f"hyperparameter mismatch on resume: {k} "
                    f"(checkpoint={ck_val}, now={getattr(args, k)})")
        ck_eff = ck.get("effective_batch", ck["args"].get("batch"))
        if ck_eff != effective_batch:
            raise ValueError(
                f"effective batch mismatch on resume: checkpoint={ck_eff}, "
                f"now={effective_batch} (cross-tier resume needs same effective batch)")
        student.load_state_dict(ck["model"])
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
                ema = Ema(student, decay=args.ema_decay)
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
        student.train()
        run_total = run_kd = run_gt = 0.0
        kappa_sum = 0.0
        pitch_kappa_sum = 0.0
        opt.zero_grad(set_to_none=True)
        bar = tqdm(train_ld, desc=f"ep {epoch}/{args.epochs - 1}",
                   dynamic_ncols=True, leave=False)
        for step, batch in enumerate(bar, 1):
            t_x = batch[0].to(device, non_blocking=True)
            s_x = batch[1].to(device, non_blocking=True)
            target = batch[2].to(device, non_blocking=True)
            if args.pitch_head:
                p_target = batch[4].to(device, non_blocking=True)
                p_valid = batch[6].to(device, non_blocking=True)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    t_yaw, _, t_pitch, _ = teacher.forward_full(t_x)
                with torch.autocast("cuda", dtype=torch.float16):
                    s_yaw, s_ky, s_pitch, s_kp = student.forward_full(s_x)
                s_yaw, s_ky = s_yaw.float(), s_ky.float()
                s_pitch, s_kp = s_pitch.float(), s_kp.float()
                loss_kd = (von_mises_nll(s_yaw, s_ky, t_yaw.float().detach())
                           + args.pitch_weight * von_mises_nll(
                               s_pitch, s_kp, t_pitch.float().detach()))
                # the kd term needs no labels and learns from every row; only the GT
                # pitch term is mask-averaged over rows that have an intent label
                nll_gt_p = von_mises_nll_per_sample(s_pitch, s_kp, p_target)
                loss_gt = (von_mises_nll(s_yaw, s_ky, target)
                           + args.pitch_weight * (nll_gt_p * p_valid).sum()
                           / p_valid.sum().clamp(min=1.0))
                kappa_sum += s_ky.mean().item()
                pitch_kappa_sum += s_kp.mean().item()
            elif args.kappa_head:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    t_pred = teacher(t_x)
                with torch.autocast("cuda", dtype=torch.float16):
                    s_pred, s_kappa = student.forward_with_kappa(s_x)
                s_pred, s_kappa = s_pred.float(), s_kappa.float()
                loss_kd = von_mises_nll(s_pred, s_kappa, t_pred.float().detach())
                loss_gt = von_mises_nll(s_pred, s_kappa, target)
                kappa_sum += s_kappa.mean().item()
            else:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    t_pred = teacher(t_x)
                with torch.autocast("cuda", dtype=torch.float16):
                    s_pred = student(s_x)
                loss_kd = von_mises_loss(s_pred.float(), t_pred.float().detach(),
                                         kappa=args.kappa)
                loss_gt = von_mises_loss(s_pred.float(), target, kappa=args.kappa)
            loss = args.alpha * loss_kd + args.beta * loss_gt
            scaler.scale(loss / grad_accum).backward()
            if step % grad_accum == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(student.parameters(),
                                                   args.grad_clip)
                # do not step the scheduler when fp16 scaling skipped the optimizer step
                prev_scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if scaler.get_scale() >= prev_scale:
                    sched.step()
                if ema is not None:
                    ema.update(student)
            run_total += loss.item()
            run_kd += loss_kd.item()
            run_gt += loss_gt.item()
            bar.set_postfix(loss=f"{run_total / step:.4f}",
                            kd=f"{run_kd / step:.4f}",
                            gt=f"{run_gt / step:.4f}",
                            lr=f"{opt.param_groups[-1]['lr']:.2e}")

        # evaluate and save best with the EMA weights (when enabled)
        if ema is not None:
            ema.apply_to(student)
        if args.pitch_head:
            metrics = evaluate_yp(student, val_ld, device, torch.float16,
                                  pitch=True)
        else:
            metrics = evaluate(student, val_ld, device)
            metrics["sel"] = metrics["maae"]
        improved = metrics["sel"] < best["sel"]
        if improved:
            best = {**metrics, "epoch": epoch}
            for old in run_dir.glob("best_*.pt"):
                old.unlink()
            torch.save({"model": student.state_dict(), "metrics": metrics,
                        "epoch": epoch, "args": vars(args), "params": n_params,
                        "model_type": {"vitt": "vitt_yaw", "hgnetv2": "hgnetv2_yaw",
                                       "yawnet": "yawnet"}[args.student_arch],
                        "pitch_head": args.pitch_head,
                        "weights": "ema" if ema is not None else "raw",
                        "teacher": teacher_info},
                       run_dir / f"best_{metrics['sel']:.6f}.pt")
            try:
                render_val_preview(student, val_ds, preview_indices,
                                   run_dir / "val_preview_best.png", device,
                                   epoch, metrics["maae"],
                                   pitch_maae=metrics.get("pitch_maae"))
            except Exception as e:  # noqa: BLE001  keep training if the preview fails
                print(f"val preview failed: {e}")
        if ema is not None:
            ema.restore(student)

        torch.save({
            "model": student.state_dict(),
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
            "model_type": {"vitt": "vitt_yaw", "hgnetv2": "hgnetv2_yaw",
                           "yawnet": "yawnet"}[args.student_arch],
            "pitch_head": args.pitch_head,
            "teacher": teacher_info,
            **({"ema": ema.state_dict()} if ema is not None else {}),
        }, last_path)

        with open(log_path, "a") as f:
            f.write(json.dumps({
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "epoch": epoch,
                "loss": round(run_total / len(train_ld), 6),
                "loss_kd": round(run_kd / len(train_ld), 6),
                "loss_gt": round(run_gt / len(train_ld), 6),
                **({"kappa_mean": round(kappa_sum / len(train_ld), 3)}
                   if args.kappa_head else {}),
                **({"pitch_kappa_mean": round(pitch_kappa_sum / len(train_ld), 3)}
                   if args.pitch_head else {}),
                "lr": opt.param_groups[-1]["lr"],
                **metrics,
                "best_maae": best["maae"],
                "best_sel": best["sel"],
                "best_epoch": best.get("epoch"),
                "elapsed_sec": round(time.time() - t0, 1),
                **({"resumed": True} if resumed and epoch == start_epoch else {}),
            }) + "\n")

        yp = "y_" if args.pitch_head else ""
        epoch_print(
            f"ep {epoch:3d} loss {run_total/len(train_ld):.4f} "
            f"(kd {run_kd/len(train_ld):.4f} gt {run_gt/len(train_ld):.4f}) "
            + (f"{yp}kappa {kappa_sum/len(train_ld):5.1f} "
               if args.kappa_head else "")
            + f"{yp}maae {metrics['maae']:6.2f} {yp}med {metrics['median']:6.2f} "
            + f"{yp}acc15 {metrics['acc15']:5.1f} {yp}acc30 {metrics['acc30']:5.1f} "
            + (f"p_maae {metrics.get('pitch_maae', float('nan')):6.2f} "
               f"p_kappa {pitch_kappa_sum/len(train_ld):5.1f} "
               if args.pitch_head else "")
            + f"(best {best['sel']:.2f}@{best.get('epoch','-')}) "
            + f"{time.time()-t0:6.0f}s", improved)

    with open(run_dir / "result.json", "w") as f:
        json.dump({"params": n_params, "best": best, "args": vars(args),
                   "teacher": teacher_info}, f, indent=2)
    print("BEST:", json.dumps(best))


if __name__ == "__main__":
    main()
