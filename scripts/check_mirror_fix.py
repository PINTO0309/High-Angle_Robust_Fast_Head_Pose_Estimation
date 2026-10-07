#!/usr/bin/env python3
"""Check adaptation to the mirror fixes (mirror_fix_*) without stopping training.

Extracts from labels_fixed.jsonl the rows that carry the old value yaw_deg_orig_mirror
(s007 oracle fixes + s002..s006 direction detector fixes + individual fixes from the
user's visual checks; both horizontal flips mirror_fix_* and front/back flips
frontback_fix_*), runs inference with the given checkpoint and measures two MAAEs:
  - maae_new : error vs the fixed labels (yaw_deg) → drops as adaptation proceeds
  - maae_old : error vs the pre-fix labels (yaw_deg_orig_mirror) → drifts toward 180°

With a contaminated old teacher, maae_old is small and maae_new is large (before the
crossover). As retraining takes effect the two cross and maae_new << maae_old.

The checkpoint may be either last.pt or best_*.pt (teacher and student are both
auto-detected). To avoid clashing with a last.pt being written, it is read from a
temporary copy.

Usage (against a run in training; 1,433 rows × 320px infer in just under 1 min):
    uv run python scripts/check_mirror_fix.py \
        --ckpt runs/dinov3_vitl16_320_yp_unified_teacher2/last.pt \
        --data data/yawpitchpose
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from distill_yawnet import load_teacher
from yaw_dataset import YawDataset, angular_error_deg

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="last.pt / best_*.pt or its run directory"
                         " (a directory auto-discovers best)")
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "yawpitchpose"))
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    src = Path(args.ckpt)
    if src.is_file():
        # read a temporary copy so it is safe even while training is writing to it
        tmp = Path(tempfile.mkdtemp()) / src.name
        shutil.copy2(src, tmp)
        spec = str(tmp)
    else:
        spec = str(src)
    model, info = load_teacher(spec, "cuda")
    print(f"ckpt: {src} (type={info['type']}, norm={info['norm']})")

    size = 320 if info["norm"] == "imagenet" else int(info.get("size") or 64)
    ds = YawDataset(args.data, "unified", size, train=False,
                    input_norm=info["norm"], with_pitch=True)
    idx = [i for i, r in enumerate(ds.rows) if "yaw_deg_orig_mirror" in r]
    if not idx:
        raise SystemExit("no mirror_fix_v6u rows found (unfixed data?)")
    print(f"mirror-fixed rows: {len(idx)}")

    ld = DataLoader(Subset(ds, idx), batch_size=args.batch, num_workers=8)
    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in ld:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(batch[0].cuda())
            preds.append(pred.float().cpu())
    pred = torch.cat(preds)

    new = torch.tensor([ds.rows[i]["yaw_deg"] for i in idx])
    old = torch.tensor([ds.rows[i]["yaw_deg_orig_mirror"] for i in idx])
    maae_new = angular_error_deg(pred, new).mean().item()
    maae_old = angular_error_deg(pred, old).mean().item()
    print(f"maae vs NEW (fixed) labels: {maae_new:6.2f} deg")
    print(f"maae vs OLD (mirrored)    : {maae_old:6.2f} deg")
    if maae_new < maae_old:
        print("=> adapted to the fixed labels (new < old)")
    else:
        print("=> still on the old (mirrored) label side (new >= old). "
              "Should reverse as retraining progresses")


if __name__ == "__main__":
    main()
