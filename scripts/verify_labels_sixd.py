#!/usr/bin/env python3
"""Verify yawpose yaw labels with sixdrepnet360 (RGB + ImageNet normalization).

Runs models/sixdrepnet360_1x3x224x224_full.onnx over every crop in
data/yawpose/crop_meta.jsonl and writes data/yawpose/qa_sixd.jsonl with
sixd_yaw/pitch/roll per record, then prints per-source agreement stats.

yaw convention: label yaw_deg in [0,360) -> signed (-180,180] for comparison.
"""
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "yawpose"
MODEL = ROOT / "models" / "sixdrepnet360_1x3x224x224_full.onnx"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def signed(deg: float) -> float:
    d = deg % 360.0
    return d - 360.0 if d > 180.0 else d


def circ_diff(a: float, b: float) -> float:
    return signed(a - b)


def main() -> None:
    rows: list[dict[str, Any]] = [json.loads(l) for l in open(OUT / "crop_meta.jsonl")]
    sess = ort.InferenceSession(str(MODEL),
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("providers:", sess.get_providers(), "rows:", len(rows), flush=True)

    with open(OUT / "qa_sixd.jsonl", "w") as f:
        for i, r in enumerate(rows):
            bgr = cv2.imread(str(OUT / r["image"]), cv2.IMREAD_COLOR)
            im = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (224, 224),
                            interpolation=cv2.INTER_AREA)
            a = (im.astype(np.float32) / 255.0 - MEAN) / STD
            y, p, ro = sess.run(None, {"input": a.transpose(2, 0, 1)[None]})[0][0]
            f.write(json.dumps({
                "image": r["image"], "source": r["source"],
                "yaw_label": r["yaw_deg"],
                "label_source": r.get("label_source", ""),
                "pitch_intent": r.get("pitch_deg"),
                "sixd_yaw": round(float(y), 2),
                "sixd_pitch": round(float(p), 2),
                "sixd_roll": round(float(ro), 2),
            }) + "\n")
            if i % 2000 == 0:
                print(f"  {i}/{len(rows)}", flush=True)

    # ---- analysis ----
    qa = [json.loads(l) for l in open(OUT / "qa_sixd.jsonl")]
    print("\n=== sign agreement (|label|>=20, |sixd|>=10, |sixd_pitch|<45) ===")
    for src in ["synthetic_001", "synthetic_002", "synthetic_003", "synthetic_004", "synthetic_005", "synthetic_006"]:
        sub = [q for q in qa if q["source"] == src]
        lab = np.array([signed(q["yaw_label"]) for q in sub])
        sy = np.array([q["sixd_yaw"] for q in sub])
        sp = np.array([q["sixd_pitch"] for q in sub])
        m = (np.abs(lab) >= 20) & (np.abs(lab) <= 160) & (np.abs(sy) >= 10) & (np.abs(sp) < 45)
        agree = float((np.sign(lab[m]) == np.sign(sy[m])).mean()) if m.any() else float("nan")
        err = np.abs([circ_diff(a, b) for a, b in zip(lab, sy)])
        err_flip = np.abs([circ_diff(-a, b) for a, b in zip(lab, sy)])
        print(f"{src}: n={len(sub)} sign_n={int(m.sum())} agree={agree:.3f} "
              f"circMAE={err.mean():.1f} circMAE_if_flipped={err_flip.mean():.1f}")
        # per label_source
        for ls in sorted({q["label_source"] for q in sub}):
            idx = [i for i, q in enumerate(sub) if q["label_source"] == ls]
            mm = m[idx]
            if mm.sum() >= 20:
                ag = float((np.sign(lab[idx][mm]) == np.sign(sy[idx][mm])).mean())
                print(f"    {ls}: sign_n={int(mm.sum())} agree={ag:.3f}")


if __name__ == "__main__":
    main()
