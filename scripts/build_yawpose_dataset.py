#!/usr/bin/env python3
"""Build the combined `data/yawpose` dataset for 360-degree yaw regression.

All three sources are cropped from their ORIGINAL full frames:
  - data/synthetic_001/batches/<run>/images/*.jpg  (16,775; yaw = annotations.jsonl angle_deg)
  - data/synthetic_002/images/*.jpg                ( 5,000; yaw = generation_plan.jsonl yaw)
  - data/synthetic_003/images/*.jpg                ( 3,000; yaw = generation_plan.jsonl yaw)

Head boxes come from models/deimv2_wholebody49_boxes_only.onnx (CUDA, head label 7).
Each box is expanded by 5% per side and cropped as a long-side square
(`deim_long_side_square_5pct_per_side`), then saved at exactly 320x320.

Image I/O and resampling use OpenCV only: INTER_AREA when shrinking,
INTER_LANCZOS4 when enlarging (PIL is avoided for numerical accuracy).

The build is resumable: every finished crop appends one line to
data/yawpose/crop_meta.jsonl (including the detected head box) and is skipped
on re-run. The train/val split (9:1, stratified by source x 10-deg yaw bin,
seed 42) is rebuilt at the end from crop_meta.jsonl.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import onnxruntime as ort

Record = dict[str, Any]
WorkItem = tuple[str, Path, Record]

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "yawpose"
IMAGES_OUT = OUT / "images"
META_PATH = OUT / "crop_meta.jsonl"
DEIM_PATH = ROOT / "models" / "deimv2_wholebody49_boxes_only.onnx"

HEAD_LABEL = 7          # verified against auto_qa head_score
HEAD_SCORE_MIN = 0.30
MARGIN = 0.05           # 5% per side
DET_SIZE = 640
BATCH = 4               # postprocessor Gather is memory-hungry on 8GB GPUs
SAVE_SIDE = 320         # every crop is saved at exactly 320x320
SEED = 42


def load_jsonl(path: Path) -> list[Record]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def resize(im: np.ndarray, size: int) -> np.ndarray:
    """INTER_AREA when shrinking, INTER_LANCZOS4 when enlarging."""
    interp = cv2.INTER_AREA if size < im.shape[0] else cv2.INTER_LANCZOS4
    return cv2.resize(im, (size, size), interpolation=interp)


def square_crop_5pct(im: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """Expand box by 5% per side, make a long-side square, crop with edge clamping."""
    H, W = im.shape[:2]
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
    sx2 = min(sx1 + side, W)
    sy2 = min(sy1 + side, H)
    crop = im[round(sy1):round(sy2), round(sx1):round(sx2)]
    ch, cw = crop.shape[:2]
    if cw != ch:  # image smaller than the requested square
        s = max(cw, ch)
        padded = np.zeros((s, s, 3), dtype=crop.dtype)
        oy, ox = (s - ch) // 2, (s - cw) // 2
        padded[oy:oy + ch, ox:ox + cw] = crop
        crop = padded
    return crop


def make_work_list() -> list[WorkItem]:
    """Return [(out_name, full_image_path, meta_dict)] for every source image."""
    work: list[WorkItem] = []

    src = DATA / "synthetic_001"
    for r in load_jsonl(src / "annotations.jsonl"):
        base = Path(r["image"]).name
        full = src / "batches" / r["generation_run"] / "images" / base
        work.append((f"s001_{base}", full, {
            "yaw_deg": round(r["angle_deg"] % 360.0, 4),
            "source": "synthetic_001",
            "label_source": r.get("label_source", ""),
            "generation_run": r.get("generation_run", ""),
        }))

    for name in ["synthetic_002", "synthetic_003"]:
        src = DATA / name
        qa = {r["filename"]: r for r in load_jsonl(src / "auto_qa.jsonl")}
        tag = name.replace("synthetic_", "s")
        for plan in load_jsonl(src / "generation_plan.jsonl"):
            full = src / "images" / plan["filename"]
            work.append((f"{tag}_{plan['filename']}", full, {
                "yaw_deg": round(float(plan["yaw"]) % 360.0, 4),
                "source": name,
                "label_source": "generation_plan",
                "pitch_deg": plan.get("pitch"),
                "cam_deg": plan.get("cam"),
                "qa_head_box": qa.get(plan["filename"], {}).get("head_box_xyxy"),
            }))

    # synthetic_004 / 005: already generated in the yawpose convention (no convention flip
    # needed). Full images live at source_runs/<run>/images/<same file name>; otherwise the
    # absolute source_run path from crop_meta is used. The delivered head_box serves as the
    # fallback when DEIM detects no head
    for name in ["synthetic_004", "synthetic_005", "synthetic_006"]:
        src = DATA / name
        if not src.exists():
            continue
        tag = name.replace("synthetic_00", "s00")
        for r in load_jsonl(src / "crop_meta.jsonl"):
            run = Path(r["source_run"]).name
            full = src / "source_runs" / run / "images" / r["source_filename"]
            if not full.exists():
                full = Path(r["source_run"]) / "images" / r["source_filename"]
            work.append((f"{tag}_{r['source_filename']}", full, {
                "yaw_deg": round(float(r["yaw_yawpose"]) % 360.0, 4),
                "source": name,
                "label_source": f"intent_{tag}",
                "pitch_deg": r.get("pitch"),
                "cam_deg": r.get("cam"),
                "visible_side": r.get("visible_side"),
                "qa_head_box": r.get("head_box_xyxy"),
            }))
    return work


def detect_and_crop(session: ort.InferenceSession, items: list[WorkItem]) -> list[Record]:
    """items: [(out_name, path, meta)]. Returns list of finished record dicts."""
    ims: list[np.ndarray] = []
    for _, path, _ in items:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        ims.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    batch = np.stack([
        cv2.resize(im, (DET_SIZE, DET_SIZE), interpolation=cv2.INTER_AREA)
        .transpose(2, 0, 1).astype(np.float32) / 255.0
        for im in ims
    ])
    out = session.run(None, {"images": batch})[0]  # (N,1240,6) label,x1,y1,x2,y2,score (norm.)
    records: list[Record] = []
    for (out_name, path, meta), im, dets in zip(items, ims, out):
        heads = dets[(dets[:, 0] == HEAD_LABEL) & (dets[:, 5] >= HEAD_SCORE_MIN)]
        qa_box = meta.pop("qa_head_box", None)
        H, W = im.shape[:2]
        if len(heads) > 0:
            best = heads[np.argmax(heads[:, 5])]
            box = [float(best[1]) * W, float(best[2]) * H,
                   float(best[3]) * W, float(best[4]) * H]
            score, box_source = float(best[5]), "deim"
        elif qa_box is not None:
            box, score, box_source = [float(v) for v in qa_box], 0.0, "auto_qa_fallback"
        else:
            print(f"  SKIP (no head): {path.name}", flush=True)
            continue
        crop = square_crop_5pct(im, box)
        native = crop.shape[0]
        crop = resize(crop, SAVE_SIDE)
        cv2.imwrite(str(IMAGES_OUT / out_name),
                    cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        records.append({
            "image": f"images/{out_name}",
            **meta,
            "head_box_xyxy": [round(v, 2) for v in box],
            "head_score": round(score, 4),
            "box_source": box_source,
            "native_crop_side": native,
        })
    return records


def stratified_split(records: list[Record]) -> tuple[list[Record], list[Record]]:
    import random
    rng = random.Random(SEED)
    groups: dict[tuple[str, int], list[Record]] = defaultdict(list)
    for r in records:
        groups[(r["source"], int(r["yaw_deg"] // 10))].append(r)
    train: list[Record] = []
    val: list[Record] = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda r: r["image"])
        rng.shuffle(rows)
        n_val = max(1, round(len(rows) * 0.1)) if len(rows) >= 5 else 0
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main() -> None:
    IMAGES_OUT.mkdir(parents=True, exist_ok=True)
    done: dict[str, Record] = {}
    if META_PATH.exists():
        for r in load_jsonl(META_PATH):
            done[r["image"]] = r
    work = make_work_list()
    todo = [(n, p, m) for n, p, m in work if f"images/{n}" not in done]
    print(f"total={len(work)} done={len(done)} todo={len(todo)}", flush=True)

    if todo:
        session = ort.InferenceSession(
            str(DEIM_PATH), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        assert "CUDAExecutionProvider" in session.get_providers(), "CUDA EP unavailable"
        with open(META_PATH, "a") as meta_f:
            for i in range(0, len(todo), BATCH):
                for rec in detect_and_crop(session, todo[i:i + BATCH]):
                    meta_f.write(json.dumps(rec) + "\n")
                    done[rec["image"]] = rec
                if (i // BATCH) % 100 == 0:
                    meta_f.flush()
                    print(f"  progress {min(i + BATCH, len(todo))}/{len(todo)}", flush=True)

    records = [done[f"images/{n}"] for n, _, _ in work if f"images/{n}" in done]
    train, val = stratified_split(records)
    for split, rows in [("train", train), ("val", val)]:
        with open(OUT / f"{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps({**r, "split": split}) + "\n")

    yaw_bins = Counter(int(r["yaw_deg"] // 30) * 30 for r in records)
    stats = {
        "total": len(records),
        "train": len(train),
        "val": len(val),
        "by_source": dict(Counter(r["source"] for r in records)),
        "by_box_source": dict(Counter(r["box_source"] for r in records)),
        "yaw_hist_30deg": {str(k): yaw_bins[k] for k in sorted(yaw_bins)},
        "crop_rule": "deim_long_side_square_5pct_per_side",
        "resampling": "opencv INTER_AREA (shrink) / INTER_LANCZOS4 (enlarge)",
        "head_label": HEAD_LABEL,
        "seed": SEED,
    }
    with open(OUT / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
