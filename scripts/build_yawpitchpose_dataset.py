#!/usr/bin/env python3
"""Build the combined `data/yawpitchpose` dataset (yaw + pitch regression).

Built on the same principles as data/yawpose:
  - synthetic_007 (9,060) is cropped by us with DEIM from the original full images
    in source_runs/ (deim_long_side_square_5pct_per_side, 320x320, OpenCV only).
    Resumable (crop_meta.jsonl sidecar).
  - synthetic_001..006 inherit the finalized artifacts of data/yawpose:
    the crop images are copied, and the labels_fixed.jsonl rows (teacher relabeled)
    and the train/val membership are carried over as-is (no re-cropping, since they
    are deterministic artifacts of the same crop rule and the same head box).

pitch label (pitch_deg, + = looking up, apparent camera-relative value):
  - s002..s007: generation intent (pitch_source = "intent")
  - s001      : no intent. Weak label recovered from the cam / pitch tokens in
                the file name (pitch_source = "recovered_cam_s001"). Coefficients
                are least-squares fitted in the trusted region of qa_sixd.jsonl
                (frontal yaw ±45°, |sixd_pitch| ≤ 60°) and recorded in stats.json.

train/val: s001..006 inherit the yawpose membership; s007 is stratified 9:1 by
delivery bin (yaw 10° × pitch band, 72 bins) with seed 42.
"""
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from build_yawpose_dataset import (BATCH, DEIM_PATH, DET_SIZE, HEAD_LABEL,
                                   HEAD_SCORE_MIN, SAVE_SIDE, SEED, load_jsonl,
                                   resize, square_crop_5pct)

Record = dict[str, Any]

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SRC_YAWPOSE = DATA / "yawpose"
SRC_S007 = DATA / "synthetic_007"
OUT = DATA / "yawpitchpose"
IMAGES_OUT = OUT / "images"
META_PATH = OUT / "crop_meta.jsonl"     # only the crops newly made for this dataset (s007)

S001_TOKEN = re.compile(r"pan([+-]\d+)_cam([+-]\d+)_pitch([+-]\d+)")


def fit_s001_pitch() -> tuple[float, float, float, dict[str, float]]:
    """Fit apparent s001 pitch ≈ a·cam + b·fn_pitch + c in the sixd trusted region."""
    qa = {q["image"]: q for q in load_jsonl(SRC_YAWPOSE / "qa_sixd.jsonl")}
    rows_a: list[list[float]] = []
    rows_y: list[float] = []
    for r in load_jsonl(SRC_YAWPOSE / "labels_fixed.jsonl"):
        if r["source"] != "synthetic_001":
            continue
        q = qa.get(r["image"])
        if q is None:
            continue
        signed_yaw = ((r["yaw_deg"] + 180.0) % 360.0) - 180.0
        if abs(signed_yaw) > 45.0 or abs(q["sixd_pitch"]) > 60.0:
            continue
        m = S001_TOKEN.search(r["image"])
        if m is None:
            continue
        rows_a.append([float(m.group(2)), float(m.group(3)), 1.0])
        rows_y.append(float(q["sixd_pitch"]))
    a_mat = np.array(rows_a)
    y_vec = np.array(rows_y)
    coef, *_ = np.linalg.lstsq(a_mat, y_vec, rcond=None)
    resid = y_vec - a_mat @ coef
    info = {"n_fit": len(rows_y), "resid_std_deg": round(float(resid.std()), 2),
            "resid_p90_deg": round(float(np.percentile(np.abs(resid), 90)), 2)}
    return float(coef[0]), float(coef[1]), float(coef[2]), info


def crop_s007() -> list[Record]:
    """DEIM-crop s007 from the original full images (resumable). Returns the records."""
    done: dict[str, Record] = {}
    if META_PATH.exists():
        for r in load_jsonl(META_PATH):
            done[r["image"]] = r

    work: list[tuple[str, Path, Record]] = []
    for r in load_jsonl(SRC_S007 / "crop_meta.jsonl"):
        run = Path(r["source_run"]).name
        full = SRC_S007 / "source_runs" / run / "images" / r["source_filename"]
        if not full.exists():
            full = Path(r["source_run"]) / "images" / r["source_filename"]
        work.append((f"s007_{r['source_filename']}", full, {
            "yaw_deg": round(float(r["yaw_yawpose"]) % 360.0, 4),
            "pitch_deg": round(float(r["pitch_yawpose"]), 4),
            "source": "synthetic_007",
            "label_source": "intent_s007",
            "pitch_source": "intent_s007",
            "visible_side": r.get("visible_side"),
            "pitch_band": r.get("pitch_band"),
            "bin": r.get("bin"),
            "qa_head_box": r.get("head_box_xyxy"),
        }))
    todo = [(n, p, m) for n, p, m in work if f"images/{n}" not in done]
    print(f"s007: total={len(work)} done={len(done)} todo={len(todo)}", flush=True)

    if todo:
        session = ort.InferenceSession(
            str(DEIM_PATH),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        assert "CUDAExecutionProvider" in session.get_providers(), "CUDA EP unavailable"
        with open(META_PATH, "a") as meta_f:
            for i in range(0, len(todo), BATCH):
                items = todo[i:i + BATCH]
                ims: list[np.ndarray] = []
                for _, path, _ in items:
                    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise FileNotFoundError(path)
                    ims.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                batch = np.stack([
                    cv2.resize(im, (DET_SIZE, DET_SIZE),
                               interpolation=cv2.INTER_AREA)
                    .transpose(2, 0, 1).astype(np.float32) / 255.0
                    for im in ims
                ])
                out = session.run(None, {"images": batch})[0]
                for (out_name, path, meta), im, dets in zip(items, ims, out):
                    heads = dets[(dets[:, 0] == HEAD_LABEL)
                                 & (dets[:, 5] >= HEAD_SCORE_MIN)]
                    qa_box = meta.pop("qa_head_box", None)
                    h, w = im.shape[:2]
                    if len(heads) > 0:
                        best = heads[np.argmax(heads[:, 5])]
                        box = [float(best[1]) * w, float(best[2]) * h,
                               float(best[3]) * w, float(best[4]) * h]
                        score, box_source = float(best[5]), "deim"
                    elif qa_box is not None:
                        box = [float(v) for v in qa_box]
                        score, box_source = 0.0, "delivery_fallback"
                    else:
                        print(f"  SKIP (no head): {path.name}", flush=True)
                        continue
                    crop = square_crop_5pct(im, box)
                    native = crop.shape[0]
                    crop = resize(crop, SAVE_SIDE)
                    cv2.imwrite(str(IMAGES_OUT / out_name),
                                cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    rec = {"image": f"images/{out_name}", **meta,
                           "head_box_xyxy": [round(v, 2) for v in box],
                           "head_score": round(score, 4),
                           "box_source": box_source,
                           "native_crop_side": native}
                    meta_f.write(json.dumps(rec) + "\n")
                    done[rec["image"]] = rec
                if (i // BATCH) % 100 == 0:
                    meta_f.flush()
                    print(f"  s007 progress {min(i + BATCH, len(todo))}/{len(todo)}",
                          flush=True)
    return [done[f"images/{n}"] for n, _, _ in work if f"images/{n}" in done]


def inherit_yawpose(cam_a: float, cam_b: float, cam_c: float) -> tuple[list[Record],
                                                                       dict[str, str]]:
    """Inherit finalized yawpose rows and fill in pitch. Returns (rows, image->split)."""
    split_of: dict[str, str] = {}
    for name in ("train", "val"):
        for r in load_jsonl(SRC_YAWPOSE / f"{name}.jsonl"):
            split_of[r["image"]] = name

    rows: list[Record] = []
    copied = 0
    for r in load_jsonl(SRC_YAWPOSE / "labels_fixed.jsonl"):
        r = dict(r)
        if r["source"] == "synthetic_001":
            m = S001_TOKEN.search(r["image"])
            if m is None:  # unexpected (verified that every s001 file carries the token)
                continue
            cam, fnp = float(m.group(2)), float(m.group(3))
            r["pitch_deg"] = round(cam_a * cam + cam_b * fnp + cam_c, 2)
            r["pitch_source"] = "recovered_cam_s001"
        else:
            r["pitch_source"] = "intent"
        rows.append(r)
        src = SRC_YAWPOSE / r["image"]
        dst = OUT / r["image"]
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    print(f"inherit: rows={len(rows)} images_copied={copied}", flush=True)
    return rows, split_of


def split_s007(records: list[Record]) -> dict[str, str]:
    """Stratify s007 9:1 by delivery bin (yaw 10° × pitch band). Returns image->split."""
    import random
    rng = random.Random(SEED)
    groups: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        groups[r["bin"]].append(r)
    split_of: dict[str, str] = {}
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda r: r["image"])
        rng.shuffle(rows)
        n_val = max(1, round(len(rows) * 0.1)) if len(rows) >= 5 else 0
        for r in rows[:n_val]:
            split_of[r["image"]] = "val"
        for r in rows[n_val:]:
            split_of[r["image"]] = "train"
    return split_of


def main() -> None:
    IMAGES_OUT.mkdir(parents=True, exist_ok=True)

    cam_a, cam_b, cam_c, fit_info = fit_s001_pitch()
    print(f"s001 pitch calibration: pitch = {cam_a:.4f}*cam + {cam_b:.4f}*fn_pitch "
          f"+ {cam_c:.4f} ({fit_info})", flush=True)

    s007_rows = crop_s007()
    inherited, split_inherit = inherit_yawpose(cam_a, cam_b, cam_c)
    split_s7 = split_s007(s007_rows)

    all_rows = inherited + s007_rows
    with open(OUT / "labels_fixed.jsonl", "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    train: list[Record] = []
    val: list[Record] = []
    for r in all_rows:
        split = split_inherit.get(r["image"]) or split_s7.get(r["image"])
        if split is None:
            raise RuntimeError(f"split not assigned: {r['image']}")
        (train if split == "train" else val).append(r)
    import random
    rng = random.Random(SEED)
    rng.shuffle(train)
    rng.shuffle(val)
    for name, rows in (("train", train), ("val", val)):
        with open(OUT / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps({**r, "split": name}) + "\n")

    yaw_bins = Counter(int(r["yaw_deg"] // 30) * 30 for r in all_rows)
    pit_rows = [r for r in all_rows if r.get("pitch_deg") is not None]
    pit_bins = Counter(int(float(r["pitch_deg"]) // 30) * 30 for r in pit_rows)
    stats = {
        "total": len(all_rows),
        "train": len(train),
        "val": len(val),
        "by_source": dict(Counter(r["source"] for r in all_rows)),
        "by_pitch_source": dict(Counter(r.get("pitch_source", "-")
                                        for r in all_rows)),
        "s007_by_box_source": dict(Counter(r["box_source"] for r in s007_rows)),
        "yaw_hist_30deg": {str(k): yaw_bins[k] for k in sorted(yaw_bins)},
        "pitch_hist_30deg": {str(k): pit_bins[k] for k in sorted(pit_bins)},
        "s001_pitch_calibration": {
            "formula": "pitch_deg = a*cam + b*fn_pitch + c",
            "a_cam": round(cam_a, 4), "b_fn_pitch": round(cam_b, 4),
            "c": round(cam_c, 4), **fit_info,
        },
        "inherited_from": "data/yawpose (labels_fixed + train/val membership, "
                          "s001..s006 crops reused as-is)",
        "crop_rule": "deim_long_side_square_5pct_per_side",
        "resampling": "opencv INTER_AREA (shrink) / INTER_LANCZOS4 (enlarge)",
        "seed": SEED,
    }
    with open(OUT / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
