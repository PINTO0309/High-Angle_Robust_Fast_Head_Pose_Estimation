#!/usr/bin/env python3
"""Run the oracle checkpoint on all s007 rows and tally suspected mirror rows.

Generates the oracle json (qa_s007_oracle.json) read by fix_s007_mirror.py.
If the json already exists it is backed up to qa_s007_oracle_prev.json before being
overwritten. When last.pt is given it is read from a temporary copy, so it is safe to
run while training is in progress.

Usage (can run without stopping training):
    uv run python scripts/screen_s007_oracle.py \
        --ckpt runs/dinov3_vitl16_320_yp_unified_teacher2/last.pt \
        --data data/yawpitchpose

To apply the fix afterwards (the fix is not picked up by the running training job;
stop training after applying and restart with --resume):
    uv run python scripts/fix_s007_mirror.py --data data/yawpitchpose

Caution: do not use "MIRROR (already fixed)" in the judgement as grounds for automatic
rollback. An oracle still in training has been measured to mispredict ~0.5% of the
already-fixed rows (2026-09-07: 7 rows misjudged by teacher2 were taken at face value
and rolled back; visual comparison with same-band reference examples from a trusted
source (s005) proved them wrong and they were restored). Left/right is settled by
comparing "which side of the frame the hint of the face / exposed neck appears on"
against reference examples side by side.
"""
import argparse
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from distill_yawnet import load_teacher
from yaw_dataset import YawDataset

ROOT = Path(__file__).resolve().parent.parent


def signed(d: float) -> float:
    return ((d + 180.0) % 360.0) - 180.0


def circ(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="oracle checkpoint (last.pt / best_*.pt / run directory)")
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "yawpitchpose"))
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    src = Path(args.ckpt)
    if src.is_file():
        tmp = Path(tempfile.mkdtemp()) / src.name
        shutil.copy2(src, tmp)
        spec = str(tmp)
    else:
        spec = str(src)
    model, info = load_teacher(spec, "cuda")
    print(f"oracle ckpt: {src} (type={info['type']}, norm={info['norm']})")

    size = 320 if info["norm"] == "imagenet" else int(info.get("size") or 64)
    ds = YawDataset(args.data, "unified", size, train=False,
                    input_norm=info["norm"], with_pitch=True)
    idx = [i for i, r in enumerate(ds.rows) if r["source"] == "synthetic_007"]
    print(f"s007 rows: {len(idx)}")

    ld = DataLoader(Subset(ds, idx), batch_size=args.batch, num_workers=12)
    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in ld:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(batch[0].cuda())
            preds.append(torch.rad2deg(
                torch.atan2(pred[:, 1].float().cpu(),
                            pred[:, 0].float().cpu())) % 360.0)
    pred = torch.cat(preds)

    out = []
    stat: Counter = Counter()
    for k, i in enumerate(idx):
        r = ds.rows[i]
        gt = r["yaw_deg"] % 360.0
        o = pred[k].item()
        out.append({"image": r["image"], "gt": round(gt, 1),
                    "pitch": r.get("pitch_deg"), "oracle": round(o, 1),
                    "already_fixed": "yaw_deg_orig_mirror" in r})
        if not 25.0 <= abs(signed(gt)) <= 155.0:
            stat["undecidable"] += 1
        elif circ(o, (360.0 - gt) % 360.0) <= 30 and circ(o, gt) > 50:
            stat["MIRROR (already fixed)" if "yaw_deg_orig_mirror" in r
                 else "MIRROR (new)"] += 1
        elif circ(o, gt) <= 30:
            stat["gt-support"] += 1
        else:
            stat["ambiguous"] += 1

    dst = Path(args.data) / "qa_s007_oracle.json"
    if dst.exists():
        shutil.move(dst, dst.with_name("qa_s007_oracle_prev.json"))
        print(f"backed up existing json: {dst.with_name('qa_s007_oracle_prev.json')}")
    with open(dst, "w") as f:
        json.dump(out, f)
    print(f"saved: {dst}")
    print("judgement:", dict(stat))
    if any(k.startswith("MIRROR (already fixed)") for k in stat):
        print("warning: MIRROR (already fixed) may be oracle mispredictions. "
              "Do not roll back automatically; confirm by visual comparison with "
              "same-band reference examples from a trusted source")
    print("\nnext: uv run python scripts/fix_s007_mirror.py --data", args.data,
          "\n(after applying, stop training and restart with --resume)")


if __name__ == "__main__":
    main()
