#!/usr/bin/env python3
"""Feasibility study of rough yaw correction via face alignment (hrffa_vitl_ibug68).

Goal: examine whether a rough yaw derived from the relative positions of both eyes
and the nose tip can correct the sixdrepnet360 yaw in the regions where it is
unstable (extreme pitch, low resolution, near-back). |yaw| > 95° (near-back) is out
of scope; face detectability is assumed to be filtered with DEIMv2 Face (classid=16).

For each target crop (320x320):
  - run DEIMv2 wholebody49 and record the highest Face(16) score
  - estimate the 68-point landmarks and their visibility with hrffa_vitl_ibug68
  - geometric feature: nose-tip offset projected onto the inter-eye vector /
    inter-eye distance (roll invariant)
Results are saved to data/yawpose/study_landmark_yaw.jsonl and a tally is printed.
"""
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "yawpose"
RESULT_PATH = OUT / "study_landmark_yaw.jsonl"

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# iBUG-68: right eye 36-41 / left eye 42-47 / nose tip 30
R_EYE = slice(36, 42)
L_EYE = slice(42, 48)
NOSE_TIP = 30
FACE_LABEL = 16
DET_SIZE = 640


def signed(deg: float) -> float:
    d = deg % 360.0
    return d - 360.0 if d > 180.0 else d


def face_score(deim: ort.InferenceSession, im320: np.ndarray) -> float:
    x = cv2.resize(im320, (DET_SIZE, DET_SIZE), interpolation=cv2.INTER_LINEAR)
    x = x.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    dets = deim.run(None, {"images": x})[0][0]
    faces = dets[dets[:, 0] == FACE_LABEL]
    return float(faces[:, 5].max()) if len(faces) else 0.0


def landmark_features(hrffa: ort.InferenceSession, im320: np.ndarray) -> dict[str, Any]:
    a = (im320.astype(np.float32) / 255.0 - MEAN) / STD
    pts, vis_logits = hrffa.run(None, {"images": a.transpose(2, 0, 1)[None]})
    pts = pts[0]                       # (68,2) normalized [0,1]
    vis = vis_logits[0].argmax(1)      # 0=out of image / 1=occluded / 2=visible
    r_eye = pts[R_EYE].mean(0)
    l_eye = pts[L_EYE].mean(0)
    nose = pts[NOSE_TIP]
    mid = (r_eye + l_eye) / 2
    d = float(np.linalg.norm(l_eye - r_eye))
    u = (l_eye - r_eye) / max(d, 1e-6)     # eye-line direction (makes it roll invariant)
    offset = float(np.dot(nose - mid, u))
    key_vis = vis[[*range(36, 48), NOSE_TIP]]
    return {
        "nose_ratio": offset / max(d, 1e-6),
        "eye_dist": d,
        "key_visible": int((key_vis == 2).sum()),  # number visible out of the 13 points
    }


def main() -> None:
    deim = ort.InferenceSession(str(ROOT / "models" / "deimv2_wholebody49_boxes_only.onnx"),
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    hrffa = ort.InferenceSession(str(ROOT / "models" / "hrffa_vitl_ibug68_1x3x320x320.onnx"),
                                 providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    fixed = {r["image"]: r for r in
             (json.loads(l) for l in open(OUT / "labels_fixed.jsonl"))}
    meta = [json.loads(l) for l in open(OUT / "crop_meta.jsonl")]
    qa = {q["image"]: q for q in (json.loads(l) for l in open(OUT / "qa_sixd.jsonl"))}

    targets: list[dict[str, Any]] = []
    rng = np.random.default_rng(42)
    # (a) calibration/evaluation: s001 operator-verified (all yaw; narrowed to |yaw|<=95 later)
    cal = [r for r in meta if r["source"] == "synthetic_001"
           and r["label_source"] == "intent_operator_promoted"]
    idx = rng.permutation(len(cal))[:4000]
    targets += [{**cal[i], "group": "cal_s001"} for i in idx]
    # (b) all s002/s003 (kept: corrected label; dropped: recorded with yaw_deg=None)
    for r in meta:
        if r["source"] in ("synthetic_002", "synthetic_003"):
            fr = fixed.get(r["image"])
            targets.append({**r,
                            "yaw_fixed": fr["yaw_deg"] if fr else None,
                            "group": "kept" if fr else "dropped"})

    print(f"targets: {len(targets)}", flush=True)
    with open(RESULT_PATH, "w") as f:
        for i, r in enumerate(targets):
            bgr = cv2.imread(str(OUT / r["image"]), cv2.IMREAD_COLOR)
            im = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rec: dict[str, Any] = {
                "image": r["image"], "source": r["source"], "group": r["group"],
                "yaw_label": r["yaw_deg"],
                "yaw_fixed": r.get("yaw_fixed", r["yaw_deg"]),
                "pitch_intent": r.get("pitch_deg"),
                "sixd_yaw": qa[r["image"]]["sixd_yaw"] if r["image"] in qa else None,
                "sixd_pitch": qa[r["image"]]["sixd_pitch"] if r["image"] in qa else None,
                "face_score": round(face_score(deim, im), 4),
            }
            rec.update({k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in landmark_features(hrffa, im).items()})
            f.write(json.dumps(rec) + "\n")
            if i % 1000 == 0:
                f.flush()
                print(f"  {i}/{len(targets)}", flush=True)
    print("done ->", RESULT_PATH, flush=True)


if __name__ == "__main__":
    main()
