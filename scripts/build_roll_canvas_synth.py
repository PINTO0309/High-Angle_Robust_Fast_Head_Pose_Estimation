#!/usr/bin/env python3
"""Build the wide-canvas version of the SynthYaw data for roll training.

Purpose: when roll_warp rotates a square crop, BORDER_REPLICATE wedges appear in the
four corners and their orientation leaks the applied angle θ (a shortcut). So a canvas
f times (default 1.8) the crop contract (DEIM box + 5% margin, long-side square),
including its surroundings, is re-cropped from the source frame and saved. At training
time RollDataset reads the "canvas" factor and rotates so that the central 1/f = the
crop contract fills the whole output (the four corners are supplied from the ring of
real pixels; the operational framing contract is unchanged).

Covers all 7 sources of yawpitchpose. The train/val split follows yawpitchpose.

Resolving the source frame and the crop contract:
  - s001: data/synthetic_001/batches/<generation_run>/images/<orig_name>.jpg.
    The crop contract is recomputed from head_box_xyxy (source frame coordinates)
    with the same geometry as square_crop_5pct
  - s002/s003: data/synthetic_00N/images/<orig_name>.jpg. Crop contract as above
  - s004..s007: look up data/synthetic_00N/source_runs/<run>/images/<orig_name>.jpg
    from source_run / source_filename in crop_meta.jsonl and use the recorded,
    clamped crop contract head_square_crop_box_xyxy directly
(<orig_name> = the crop name with the sNNN_ prefix removed)

Usage:
    uv run python scripts/build_roll_canvas_synth.py
Output: data/roll_canvas_synth/ (images/ + train.jsonl + val.jsonl +
labels_fixed.jsonl, each row with "canvas": f). Existing images are skipped on rerun.
"""
import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "yawpitchpose"
CROPMETA_SOURCES = {"synthetic_004", "synthetic_005", "synthetic_006",
                    "synthetic_007"}
SAVE_SIDE = 320
MARGIN = 0.05


def resolve_src(row: dict) -> Path:
    """Source frame path for s001-003 (crop contract recomputed from head_box_xyxy)."""
    name = Path(row["image"]).name.split("_", 1)[1]
    src = row["source"]
    if src == "synthetic_001":
        m = re.match(r"^(.+?)_(\d{6}--.+)$", name)
        return ROOT / "data" / src / "batches" / m.group(1) / "images" / name
    return ROOT / "data" / src / "images" / name


def contract_square(box: list[float], W: int, H: int) -> tuple[float, float, float]:
    """Clamped crop contract (cx, cy, side), same geometry as square_crop_5pct."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1 -= w * MARGIN
    x2 += w * MARGIN
    y1 -= h * MARGIN
    y2 += h * MARGIN
    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sx1 = min(max(cx - side / 2, 0), max(W - side, 0))
    sy1 = min(max(cy - side / 2, 0), max(H - side, 0))
    return sx1 + side / 2, sy1 + side / 2, side


def crop_canvas(im: np.ndarray, cx: float, cy: float, side: float) -> tuple[np.ndarray, bool]:
    """Cut a square of size side centered at (cx, cy) (out-of-range: replicate pad)."""
    H, W = im.shape[:2]
    ix1, iy1 = round(cx - side / 2), round(cy - side / 2)
    ix2, iy2 = round(cx + side / 2), round(cy + side / 2)
    pad_l, pad_t = max(0, -ix1), max(0, -iy1)
    pad_r, pad_b = max(0, ix2 - W), max(0, iy2 - H)
    padded = bool(pad_l or pad_t or pad_r or pad_b)
    if padded:
        im = cv2.copyMakeBorder(im, pad_t, pad_b, pad_l, pad_r,
                                cv2.BORDER_REPLICATE)
        ix1 += pad_l
        ix2 += pad_l
        iy1 += pad_t
        iy2 += pad_t
    return im[iy1:iy2, ix1:ix2], padded


def render_canvas(src: str, out_path: str, factor: float,
                  sq: list[float] | None, box: list[float] | None
                  ) -> tuple[float, bool] | None:
    """Cut a canvas factor times the crop contract from the source frame and save it.

    The crop contract is sq (xyxy of the clamped square) if given, otherwise it is
    recomputed from box (the head box). Returns (contract side, whether the area
    outside the frame was replicate-filled). None if the source frame is unreadable.
    """
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    if im is None:
        return None
    H, W = im.shape[:2]
    if sq is None:
        cx, cy, side = contract_square(box, W, H)
    else:
        x1, y1, x2, y2 = sq
        cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1
    canvas, padded = crop_canvas(im, cx, cy, side * factor)
    # same rule as the contract crop: INTER_AREA to shrink, INTER_LANCZOS4 to enlarge
    # (the canvas is smaller than SAVE_SIDE when the head in the source frame is small)
    interp = (cv2.INTER_AREA if SAVE_SIDE < canvas.shape[0]
              else cv2.INTER_LANCZOS4)
    canvas = cv2.resize(canvas, (SAVE_SIDE, SAVE_SIDE), interpolation=interp)
    cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return side, padded


def process_one(task: tuple[dict, str, float, str, list[float] | None]
                ) -> dict | None:
    row, out_path, factor, src, sq = task
    res = render_canvas(src, out_path, factor, sq, row.get("head_box_xyxy"))
    if res is None:
        return None
    side, padded = res
    return {"image": f"images/{Path(out_path).name}",
            "yaw_deg": row["yaw_deg"], "pitch_deg": row["pitch_deg"],
            "source": row["source"], "canvas": factor,
            "contract_side": round(side, 1), "ring_padded": padded}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str,
                    default=str(ROOT / "data" / "roll_canvas_synth"))
    ap.add_argument("--factor", type=float, default=1.8,
                    help="canvas factor f (how many times the crop contract to save)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    # s004-007: resolve the source frame and crop contract from crop_meta.jsonl
    metas: dict[tuple[str, str], dict] = {}
    for src_name in sorted(CROPMETA_SOURCES):
        p = ROOT / "data" / src_name / "crop_meta.jsonl"
        if p.exists():
            for line in open(p):
                m = json.loads(line)
                metas[(src_name, m["filename"])] = m

    for split in ["train", "val"]:
        rows = [json.loads(l) for l in open(SRC / f"{split}.jsonl")]
        if args.limit:
            rows = rows[:args.limit]
        tasks = []
        skipped = miss = 0
        for r in rows:
            op = out / "images" / Path(r["image"]).name
            if op.exists():
                skipped += 1
                continue
            name = Path(r["image"]).name.split("_", 1)[1]
            if r["source"] in CROPMETA_SOURCES:
                m = metas.get((r["source"], name))
                if m is None:
                    miss += 1
                    continue
                src = (ROOT / "data" / r["source"] / "source_runs"
                       / Path(m["source_run"]).name / "images"
                       / m["source_filename"])
                sq = m["head_square_crop_box_xyxy"]
            else:
                src, sq = resolve_src(r), None
            tasks.append((r, str(op), args.factor, str(src), sq))
        print(f"{split}: {len(tasks)} to process / {skipped} skipped (existing)"
              + (f" / no crop_meta {miss}" if miss else ""))
        recs = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, rec in enumerate(ex.map(process_one, tasks, chunksize=64)):
                if rec is not None:
                    recs.append(rec)
                if (i + 1) % 5000 == 0:
                    print(f"  {i + 1}/{len(tasks)}", flush=True)
        # the jsonl is rewritten every time with "all target rows" (so skipped rows
        # can be re-recorded from the existing images, the meta is recomputed from
        # the row)
        by_name = {Path(x["image"]).name: x for x in recs}
        with open(out / f"{split}.jsonl", "w") as f:
            for r in rows:
                name = Path(r["image"]).name
                rec = by_name.get(name)
                if rec is None:
                    if not (out / "images" / name).exists():
                        continue
                    rec = {"image": f"images/{name}",
                           "yaw_deg": r["yaw_deg"],
                           "pitch_deg": r["pitch_deg"],
                           "source": r["source"], "canvas": args.factor}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    rows_all = []
    for split in ["train", "val"]:
        rows_all += [l for l in open(out / f"{split}.jsonl") if l.strip()]
    with open(out / "labels_fixed.jsonl", "w") as f:
        f.writelines(rows_all)
    n_img = len(list((out / "images").glob("*.jpg")))
    print(f"done: labels_fixed {len(rows_all)} rows / images {n_img} -> {out}")


if __name__ == "__main__":
    main()
