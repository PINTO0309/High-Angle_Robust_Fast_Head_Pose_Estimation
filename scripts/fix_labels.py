#!/usr/bin/env python3
"""Fix the yaw label convention and verify signs (based on the qa_sixd.jsonl diagnosis).

Findings of the diagnosis (scripts/verify_labels_sixd.py, data/yawpose/qa_sixd.jsonl):
  - s001 intent labels (operator-validated) agree with sixdrepnet360: sign agreement
    rate 0.99, median circular error 11° → trustworthy as the reference convention.
  - Only the s001 rows with label_source=sixdrepnet360 are stored with every sign
    flipped (median circular error 1.2° after flipping) → convention conversion
    mistake. Fixed by negating the sign.
  - The s002 / s003 generation_plan yaw has the convention itself inverted (for
    |yaw| 30-150°, flipping brings the median from 111° to 20°), and in about 30%
    of the samples the generative model also swapped left and right → invert the
    convention + verify the sign per sample with sixd.
  - s004 was generated in the yawpose convention (no inversion needed). Only
    |yaw| 15-95° is sign-checked against sixd (mismatch → flip); when sixd is
    unreliable, and in the rear band (|yaw|>95°), the intent that passed the
    delivery-side direction QA is kept.

Processing:
  1. Join crop_meta.jsonl + qa_sixd.jsonl + study_landmark_yaw.jsonl and write the
     fixed records to labels_fixed.jsonl
     (yaw_deg holds the fixed value; yaw_deg_orig / label_fix are recorded as well).
  2. For large-angle samples (|yaw|>=AMBIG_MIN_DEG) the sign is verified with sixd
     in the sixd-reliable domain; outside it, it is verified against the sign of an
     approximate yaw from the hrffa_vitl_ibug68 landmark geometry (nose-tip offset
     projected onto the eye line / inter-ocular distance → sin fit)
     (conditions: |yaw|<=95°, DEIMv2 Face(16) score>=0.3, |landmark_yaw|>=8°.
      Feasibility: 99.3% agreement with the sixd-verified sign on kept samples).
     Samples that cannot be verified either way are dropped.
  3. Regenerate train/val (9:1, stratified by source x 10° bin, seed 42).
"""
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from build_yawpose_dataset import OUT, stratified_split

QA_PATH = OUT / "qa_sixd.jsonl"
META_PATH = OUT / "crop_meta.jsonl"
STUDY_PATH = OUT / "study_landmark_yaw.jsonl"
FIXED_PATH = OUT / "labels_fixed.jsonl"

AMBIG_MIN_DEG = 15.0     # below this |yaw| the sign matters little; kept unverified
SIXD_YAW_MIN = 10.0      # min |sixd_yaw| for trusting the sixd sign
SIXD_PITCH_MAX = 60.0    # max |sixd_pitch| for trusting sixd
LM_FACE_MIN = 0.3        # min Face(16) score for using the landmark sign
LM_YAW_MIN = 8.0         # min |landmark_yaw| for trusting the landmark sign
LM_MAX_ABS_YAW = 95.0    # landmark fix upper bound (beyond is near-rear, excluded)


def signed(deg: float) -> float:
    d = deg % 360.0
    return d - 360.0 if d > 180.0 else d


def calibrate_landmark_yaw(study: list[dict[str, Any]]) -> tuple[float, float]:
    """Fit sin(yaw) = S*nose_ratio + B on the operator-validated s001 subset."""
    cal = [r for r in study if r["group"] == "cal_s001"
           and abs(signed(r["yaw_label"])) <= 80 and r["face_score"] >= LM_FACE_MIN]
    ratio = np.array([r["nose_ratio"] for r in cal])
    target = np.sin(np.radians([signed(r["yaw_label"]) for r in cal]))
    a = np.stack([ratio, np.ones(len(cal))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(a, target, rcond=None)
    return float(coef[0]), float(coef[1])


def main() -> None:
    qa: dict[str, dict[str, Any]] = {
        q["image"]: q for q in (json.loads(l) for l in open(QA_PATH))}
    rows: list[dict[str, Any]] = [json.loads(l) for l in open(META_PATH)]

    study_rows: list[dict[str, Any]] = [json.loads(l) for l in open(STUDY_PATH)]
    lm_s, lm_b = calibrate_landmark_yaw(study_rows)
    print(f"landmark calibration: sin(yaw) = {lm_s:.4f} * nose_ratio + {lm_b:.4f}")
    lm: dict[str, float | None] = {}   # image -> landmark yaw (deg) or None
    for r in study_rows:
        if r["face_score"] >= LM_FACE_MIN:
            v = max(-1.0, min(1.0, lm_s * r["nose_ratio"] + lm_b))
            lm[r["image"]] = math.degrees(math.asin(v))
        else:
            lm[r["image"]] = None

    fixed: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for r in rows:
        q = qa.get(r["image"])
        src = r["source"]
        ls = r.get("label_source", "")
        yaw_orig = float(r["yaw_deg"])
        fix = "kept"
        yaw = yaw_orig

        if src == "synthetic_001":
            if ls == "sixdrepnet360":
                yaw = (-yaw_orig) % 360.0          # fix sign flip at storage time
                fix = "negated_storage_bug"
        elif src in ("synthetic_004", "synthetic_005", "synthetic_006"):
            # generated in the yawpose convention (no inversion needed). There is no
            # extreme pitch and the delivery-side direction QA has passed, so only
            # |yaw| 15-95° is sign-checked against sixd (mismatch → flip); when sixd
            # is unreliable, and in the rear band, the intent is kept
            tag = src.replace("synthetic_00", "s00")
            yaw0 = signed(yaw_orig)
            fix = f"kept_intent_{tag}"
            if AMBIG_MIN_DEG <= abs(yaw0) <= LM_MAX_ABS_YAW:
                sixd_ok = (q is not None
                           and abs(q["sixd_yaw"]) >= SIXD_YAW_MIN
                           and abs(q["sixd_pitch"]) < SIXD_PITCH_MAX)
                if sixd_ok:
                    if (q["sixd_yaw"] > 0) != (yaw0 > 0):
                        yaw0 = -yaw0
                        fix = f"{tag}+sixd_flip"
                    else:
                        fix = f"{tag}+sixd_confirm"
                else:
                    fix = f"{tag}_unverified_kept"
            yaw = yaw0 % 360.0
        else:  # synthetic_002 / synthetic_003
            yaw0 = -signed(yaw_orig)               # fix inverted prompt convention
            fix = "negated_convention"
            if abs(yaw0) >= AMBIG_MIN_DEG:
                sixd_ok = (q is not None
                           and abs(q["sixd_yaw"]) >= SIXD_YAW_MIN
                           and abs(q["sixd_pitch"]) < SIXD_PITCH_MAX)
                lm_yaw = lm.get(r["image"])
                lm_ok = (lm_yaw is not None
                         and abs(lm_yaw) >= LM_YAW_MIN
                         and abs(yaw0) <= LM_MAX_ABS_YAW)
                if sixd_ok:
                    if (q["sixd_yaw"] > 0) != (yaw0 > 0):
                        yaw0 = -yaw0               # fix generator's left/right swap
                        fix = "negated_convention+sixd_flip"
                elif lm_ok:
                    if (lm_yaw > 0) != (yaw0 > 0):
                        yaw0 = -yaw0
                        fix = "negated_convention+landmark_flip"
                    else:
                        fix = "negated_convention+landmark_confirm"
                else:
                    stats[f"{src}:dropped_ambiguous"] += 1
                    continue
            yaw = yaw0 % 360.0

        stats[f"{src}:{fix}"] += 1
        fixed.append({**r,
                      "yaw_deg": round(yaw, 4),
                      "yaw_deg_orig": yaw_orig,
                      "label_fix": fix})

    with open(FIXED_PATH, "w") as f:
        for r in fixed:
            f.write(json.dumps(r) + "\n")

    train, val = stratified_split(fixed)
    for split, split_rows in [("train", train), ("val", val)]:
        with open(OUT / f"{split}.jsonl", "w") as f:
            for r in split_rows:
                f.write(json.dumps({**r, "split": split}) + "\n")

    yaw_bins = Counter(int(r["yaw_deg"] // 30) * 30 for r in fixed)
    summary = {
        "total_in": len(rows),
        "total_out": len(fixed),
        "train": len(train),
        "val": len(val),
        "fixes": dict(sorted(stats.items())),
        "yaw_hist_30deg": {str(k): yaw_bins[k] for k in sorted(yaw_bins)},
        "thresholds": {"ambig_min_deg": AMBIG_MIN_DEG,
                       "sixd_yaw_min": SIXD_YAW_MIN,
                       "sixd_pitch_max": SIXD_PITCH_MAX,
                       "lm_face_min": LM_FACE_MIN,
                       "lm_yaw_min": LM_YAW_MIN,
                       "lm_max_abs_yaw": LM_MAX_ABS_YAW},
        "landmark_calibration": {"sin_slope": round(lm_s, 4),
                                 "sin_intercept": round(lm_b, 4)},
    }
    with open(OUT / "label_fix_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
