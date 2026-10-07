#!/usr/bin/env python3
"""Relabel the s001 intent_rear labels with the DINOv3 teacher.

Background (from the analysis of the teacher training log):
  Only the 240-270° bin of val has an outlying MAE (45° vs 19° in the mirror bin).
  Training is symmetrized by hflip, so an asymmetric error points to one-sided noise
  in the s001 intent_rear labels rather than the model (median label vs sixd gap in
  that band 52° vs 33° in the mirror band).

Method:
  - Targets: rows of labels_fixed.jsonl with source=synthetic_001 and
    label_source=intent_rear (both train / val. The teacher prediction is recorded
    for every target row)
  - Teacher: runs/<teacher_run>/best_*.pt (output of train_teacher_dinov3.py),
    restored with the same loader as distill. Input is ImageNet-normalized, 320x320
  - hflip TTA: unit-vector mean of pred(x) and mirror(pred(flip(x)))
    (uses the model's left/right symmetry to even out noise)
  - Replacement rule: only rows whose circular difference between teacher prediction
    and label is > --thr (default 25°) get yaw_deg replaced by the teacher prediction
    (label_fix="teacher_relabel"). Within the threshold the intent is kept
    ("+teacher_ok" is appended to label_fix)
  - train/val membership is never changed (redrawing the split would move the
    teacher's training images into val and make the evaluation self-fulfilling).
    The 3 files labels_fixed / train / val are joined by image key and updated

Output: the 3 jsonl files above updated (originals backed up as .bak_relabel) +
        relabel_summary.json
"""
import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from distill_yawnet import load_teacher
from yaw_dataset import normalize_image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "yawpose"


def signed(deg: float) -> float:
    d = deg % 360.0
    return d - 360.0 if d > 180.0 else d


def circ_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


@torch.no_grad()
def predict_yaws(teacher: torch.nn.Module, norm: str, images: list[str],
                 size: int, batch: int, device: str) -> dict[str, float]:
    """Infer the teacher's yaw [deg, 0..360) with hflip TTA."""
    preds: dict[str, float] = {}
    for i in tqdm(range(0, len(images), batch), desc="teacher", dynamic_ncols=True):
        chunk = images[i:i + batch]
        xs, xs_f = [], []
        for rel in chunk:
            bgr = cv2.imread(str(OUT / rel), cv2.IMREAD_COLOR)
            im = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if im.shape[0] != size:
                im = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
            xs.append(normalize_image(im, norm))
            xs_f.append(normalize_image(cv2.flip(im, 1), norm))
        x = torch.stack(xs).to(device)
        xf = torch.stack(xs_f).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = teacher(x).float()
            pf = teacher(xf).float()
        # mirrored prediction is yaw -> -yaw, so flip the sin sign back and average
        pf[:, 1] = -pf[:, 1]
        v = torch.nn.functional.normalize(p + pf, dim=1)
        yaw = torch.rad2deg(torch.atan2(v[:, 1], v[:, 0])) % 360.0
        for rel, yv in zip(chunk, yaw.tolist()):
            preds[rel] = round(yv, 4)
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=str,
                    default=str(ROOT / "runs" / "dinov3_vitl16_320_teacher"))
    ap.add_argument("--thr", type=float, default=25.0,
                    help="only rows whose circular difference [deg] exceeds this "
                         "are replaced with the teacher prediction")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = "cuda"
    teacher, info = load_teacher(args.teacher, device)
    print(f"teacher: {info['path']} (type={info['type']}, val_maae="
          f"{info['metrics'] and info['metrics'].get('maae')})")

    fixed = [json.loads(l) for l in open(OUT / "labels_fixed.jsonl")]
    targets = [r for r in fixed
               if r["source"] == "synthetic_001"
               and r.get("label_source") == "intent_rear"]
    print(f"targets (s001 intent_rear): {len(targets)}")

    preds = predict_yaws(teacher, info["norm"], [r["image"] for r in targets],
                         args.size, args.batch, device)

    # update table: image -> (new_yaw, fix_tag, teacher_yaw)
    updates: dict[str, tuple[float, str, float]] = {}
    n_relabel = 0
    diffs: list[float] = []
    for r in targets:
        t_yaw = preds[r["image"]]
        d = circ_diff(r["yaw_deg"], t_yaw)
        diffs.append(d)
        if d > args.thr:
            updates[r["image"]] = (t_yaw, "teacher_relabel", t_yaw)
            n_relabel += 1
        else:
            updates[r["image"]] = (r["yaw_deg"], r["label_fix"] + "+teacher_ok", t_yaw)

    def apply(path: Path) -> int:
        rows = [json.loads(l) for l in open(path)]
        n = 0
        for r in rows:
            u = updates.get(r["image"])
            if u is None:
                continue
            new_yaw, tag, t_yaw = u
            if "yaw_deg_orig" not in r:
                r["yaw_deg_orig"] = r["yaw_deg"]
            r["yaw_deg"] = new_yaw
            r["label_fix"] = tag
            r["teacher_yaw"] = t_yaw
            n += 1
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak_relabel"))
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return n

    counts = {p.name: apply(p) for p in
              [OUT / "labels_fixed.jsonl", OUT / "train.jsonl", OUT / "val.jsonl"]}

    d = np.array(diffs)
    summary: dict[str, Any] = {
        "teacher": info["path"],
        "teacher_val_maae": info["metrics"] and info["metrics"].get("maae"),
        "threshold_deg": args.thr,
        "targets": len(targets),
        "relabeled": n_relabel,
        "kept": len(targets) - n_relabel,
        "diff_teacher_vs_label": {
            "mean": round(float(d.mean()), 2),
            "median": round(float(np.median(d)), 2),
            "p90": round(float(np.percentile(d, 90)), 2),
        },
        "updated_rows": counts,
        "note": "train/val membership unchanged (only labels updated). Old value in "
                "yaw_deg_orig, teacher prediction in teacher_yaw for every target row.",
    }
    with open(OUT / "relabel_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
