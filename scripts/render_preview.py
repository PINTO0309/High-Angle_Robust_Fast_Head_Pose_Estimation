#!/usr/bin/env python3
"""CLI that generates the validation preview (3x3) offline from a trained checkpoint.

Produces the same image that the training script writes automatically on each best
update, from any checkpoint. Teacher (dinov3_yaw) / student (YawNet) is detected
automatically from the checkpoint's model_type, and the input normalization and
resolution are restored automatically too (the loader is shared with export_onnx.py).

Usage:
    uv run python scripts/render_preview.py --ckpt runs/dinov3_vitl16_320_teacher_v3
    uv run python scripts/render_preview.py --ckpt runs/yawnet_distill_64_v3

The output defaults to <ckpt folder>/val_preview_best<set>.png (changeable with
--out; it always differs from the val_preview_best.png written by the training
script).
"""
import argparse
import json
from pathlib import Path

import torch

from export_onnx import load_model
from val_preview import (render_val_preview, select_diverse_indices,
                         select_indices)
from yaw_dataset import YawDataset

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="run directory (auto-finds best_*.pt) or a .pt file")
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "yawpose"))
    ap.add_argument("--out", type=str, default="",
                    help="output PNG (default: <ckpt folder>/"
                         "val_preview_best[N].png)")
    ap.add_argument("--set", type=str, default="1", dest="sample_set",
                    help="sample set number. A range like '1-100' writes the"
                         " consecutive sets in one go (the model is loaded once)."
                         " Without a filter: another 9 diverse samples drawn with"
                         " a different random sequence; with a filter: the next 9"
                         " candidates that do not overlap the previous set")
    ap.add_argument("--seed", type=int, default=0,
                    help="random seed for the diverse sample selection. The default 0"
                         " combines with --set into a deterministic seed, so rerunning"
                         " the same --set always reproduces the same 9 images"
                         " (-1 = random every time)")
    ap.add_argument("--filter-image", type=str, default="",
                    help="narrow the candidates with a regex on the image path"
                         " (e.g. mask = mask-wearing s001 images, ^images/s007_)")
    ap.add_argument("--min-abs-pitch", type=float, default=0.0,
                    help="select only from rows with |pitch_deg| >= this value"
                         " (high pitch angle condition; rows without a pitch label"
                         " are excluded)")
    args = ap.parse_args()
    if "-" in args.sample_set:
        lo, hi = (int(v) for v in args.sample_set.split("-", 1))
        if lo < 1 or hi < lo:
            raise SystemExit(f"invalid --set range: {args.sample_set}")
        sets = list(range(lo, hi + 1))
    else:
        sets = [max(1, int(args.sample_set))]
    if (args.filter_image or args.min_abs_pitch > 0) and not args.out:
        raise SystemExit("--out must be given explicitly with --filter-image /"
                         " --min-abs-pitch to avoid overwriting the default name")

    model, ck_path, size, _, norm = load_model(args.ckpt, 0)
    device = "cuda"
    model.to(device)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    epoch = int(ck.get("epoch", -1))
    metrics = ck.get("metrics") or {}
    maae = float(metrics.get("maae", float("nan")))
    amp_dtype = torch.bfloat16 if norm == "imagenet" else torch.float16

    ds = YawDataset(args.data, "val", size, train=False, input_norm=norm)
    use_filter = bool(args.filter_image or args.min_abs_pitch > 0)
    allowed = None
    if use_filter:
        import re  # noqa: PLC0415
        pat = re.compile(args.filter_image) if args.filter_image else None
        allowed = []
        for i, r in enumerate(ds.rows):
            if pat is not None and not pat.search(r["image"]):
                continue
            if args.min_abs_pitch > 0:
                p = r.get("pitch_deg")
                if p is None or abs(float(p)) < args.min_abs_pitch:
                    continue
            allowed.append(i)
        if len(allowed) < 9:
            raise SystemExit(f"only {len(allowed)} candidate rows after filtering")
        print(f"filter: {len(allowed)} candidate rows")
    import numpy as np  # noqa: PLC0415
    accessory: set[str] = set()
    if not use_filter:
        acc_path = Path(args.data) / "accessory_index.json"
        if acc_path.exists():
            accessory = set(json.load(open(acc_path)).keys())
        else:
            print(f"note: {acc_path} not found; the accessory bucket is drawn from"
                  " all rows (can be generated with scripts/build_accessory_index.py)")

    for set_no in sets:
        if use_filter:
            # with a filter: the original deterministic closest-row selection
            # (--set selects the next candidate set)
            exclude: set[int] = set()
            for _ in range(set_no - 1):
                exclude.update(select_indices(ds, exclude=exclude, allowed=allowed))
            indices = select_indices(ds, exclude=exclude, allowed=allowed)
        else:
            # without a filter: diverse samples of full-surround yaw × (pitch0 /
            # looking up / looking down / accessory), picked at random near each
            # target angle
            seed = None if args.seed < 0 else args.seed + set_no - 1
            rng = np.random.default_rng(seed)
            indices = select_diverse_indices(ds, accessory, rng)
        # To avoid clashing with the val_preview_best.png written by the training
        # script, the CLI output is always numbered (val_preview_best1.png), even
        # for --set 1. When --out is combined with a range, the set number is
        # inserted before the extension
        if args.out and len(sets) > 1:
            o = Path(args.out)
            out = o.with_name(f"{o.stem}{set_no}{o.suffix}")
        elif args.out:
            out = Path(args.out)
        else:
            out = ck_path.parent / f"val_preview_best{set_no}.png"
        render_val_preview(model, ds, indices, out, device,
                           epoch, maae, pitch_maae=metrics.get("pitch_maae"),
                           amp_dtype=amp_dtype)
        print(f"saved: {out} (ckpt={ck_path.name}, size={size}, norm={norm}, "
              f"epoch={epoch}, maae={maae})")


if __name__ == "__main__":
    main()
