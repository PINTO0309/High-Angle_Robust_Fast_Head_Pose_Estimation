#!/usr/bin/env python3
"""Determine CAM_YAW_SIGN by measurement.

Images from s002 with shallow pitch (the reliable range of sixdrepnet360) are square-cropped
with a 5% margin from the auto_qa head box and warped with the camera-azimuth homography
(ψ = ±15°); the difference between the sixdrepnet360 yaw readings before and after the warp
measures which way Ry(+ψ) moves yaw.
Expected: mean(delta) ≈ CAM_YAW_SIGN * ψ
"""
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from augment import GeometricParams, apply_geometric

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "sixdrepnet360_1x3x224x224_full.onnx"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def signed(deg: float) -> float:
    d = deg % 360.0
    return d - 360.0 if d > 180.0 else d


def crop5(im: np.ndarray, box: list[float]) -> np.ndarray:
    H, W = im.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1 -= w * .05; x2 += w * .05; y1 -= h * .05; y2 += h * .05
    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sx1 = min(max(cx - side / 2, 0), max(W - side, 0))
    sy1 = min(max(cy - side / 2, 0), max(H - side, 0))
    return im[round(sy1):round(min(sy1 + side, H)), round(sx1):round(min(sx1 + side, W))]


def sixd_yaw(sess: ort.InferenceSession, im224: np.ndarray) -> float:
    a = (im224.astype(np.float32) / 255.0 - MEAN) / STD
    return float(sess.run(None, {"input": a.transpose(2, 0, 1)[None]})[0][0][0])


def main() -> None:
    src = ROOT / "data" / "synthetic_002"
    qa: list[dict[str, Any]] = [json.loads(l) for l in open(src / "auto_qa.jsonl")]
    # pick samples with a shallow pitch intent and a high head score
    plans = {p["filename"]: p for p in
             (json.loads(l) for l in open(src / "generation_plan.jsonl"))}
    cands = [q for q in qa
             if abs(plans[q["filename"]]["pitch"]) <= 15 and q.get("head_score", 0) > 0.9]
    cands = cands[:60]
    print(f"samples: {len(cands)}")

    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    deltas: dict[float, list[float]] = {-15.0: [], 15.0: []}
    for q in cands:
        bgr = cv2.imread(str(src / "images" / q["filename"]), cv2.IMREAD_COLOR)
        im = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        crop = crop5(im, q["head_box_xyxy"])
        base_prm = GeometricParams(out_size=224)
        base_img, _ = apply_geometric(crop, 0.0, base_prm)
        y0 = sixd_yaw(sess, base_img)
        if abs(y0) > 60:  # restrict to the reliable range of sixd
            continue
        for psi in (-15.0, 15.0):
            prm = GeometricParams(out_size=224, cam_yaw_deg=psi)
            img, _ = apply_geometric(crop, 0.0, prm)
            y1 = sixd_yaw(sess, img)
            deltas[psi].append(signed(y1 - y0))

    for psi, ds in deltas.items():
        arr = np.array(ds)
        print(f"psi={psi:+.0f}: n={len(arr)} mean_delta={arr.mean():+.2f} "
              f"median={np.median(arr):+.2f} std={arr.std():.2f}")
    m = np.mean(deltas[15.0]) - np.mean(deltas[-15.0])
    sign = 1.0 if m > 0 else -1.0
    print(f"=> CAM_YAW_SIGN = {sign:+.0f} (spread {m:+.2f} deg over 30 deg)")


if __name__ == "__main__":
    main()
