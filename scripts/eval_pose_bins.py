#!/usr/bin/env python3
"""Validation MAAE of the yaw / pitch body ONNX files per pose bin (the analogue of HRFFA's Table 3).

Each validation crop is resized to the model input (INTER_LINEAR, like the validation
preprocessing of training) and the yaw and pitch MAAE are aggregated per bin of the
ground-truth |yaw| (frontal 0° to the back of the head 180°, left and right folded together)
and of the ground-truth pitch. With several --onnx entries the script prints README-ready
Markdown tables (cells are "yaw / pitch", the last row is the count per bin).

Usage:
    uv run python scripts/eval_pose_bins.py \\
        --onnx vitt-064=runs/vitt_distill_064_yp_unified/hfhpe_vitt_yawpitch_Nx3x64x64.onnx
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
YAW_EDGES = [0, 30, 60, 90, 120, 150, 180]            # |yaw| (absolute value of the signed yaw)
PITCH_EDGES = [-120, -90, -60, -30, 0, 30, 60, 90, 120]


def wrap_abs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def predict(onnx_path: Path, crops: list[np.ndarray], batch: int, cpu: bool) -> np.ndarray:
    import onnxruntime as ort  # noqa: PLC0415
    providers = (["CPUExecutionProvider"] if cpu
                 else ["CUDAExecutionProvider", "CPUExecutionProvider"])
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    size = int(sess.get_inputs()[0].shape[-1])
    bs = 1 if isinstance(sess.get_inputs()[0].shape[0], int) else batch
    meta = sess.get_modelmeta().custom_metadata_map or {}
    imagenet = "mean=" in meta.get("input.images", "")
    mean = np.array([0.485, 0.456, 0.406], np.float32) * 255.0
    std = np.array([0.229, 0.224, 0.225], np.float32) * 255.0
    preds = np.zeros((len(crops), 2))
    for i0 in range(0, len(crops), bs):
        planes = []
        for c in crops[i0:i0 + bs]:
            rgb = cv2.cvtColor(cv2.resize(c, (size, size), interpolation=cv2.INTER_LINEAR),
                               cv2.COLOR_BGR2RGB).astype(np.float32)
            rgb = (rgb - mean) / std if imagenet else rgb / 127.5 - 1.0
            planes.append(rgb.transpose(2, 0, 1))
        cs = sess.run(["cos_sin"], {"images": np.stack(planes)})[0]
        preds[i0:i0 + len(planes), 0] = np.degrees(np.arctan2(cs[:, 1], cs[:, 0]))
        preds[i0:i0 + len(planes), 1] = np.degrees(np.arctan2(cs[:, 3], cs[:, 2]))
    return preds


def table(title: str, keys: list[str], rows: dict[str, list[tuple[float, float]]],
          counts: list[int]) -> str:
    out = [f"#### {title}", "| Model | " + " | ".join(keys) + " |", "|---|" + "--:|" * len(keys)]
    for model, cells in rows.items():
        out.append(f"| {model} | " + " | ".join(f"{y:.2f} / {p:.2f}" for y, p in cells) + " |")
    out.append("| n | " + " | ".join(str(n) for n in counts) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", action="append", required=True,
                    help="body ONNX (name=path sets the row name; repeatable)")
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "yawpitchpose"))
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", type=str, default="", help="result JSON (not written when empty)")
    args = ap.parse_args()

    data = Path(args.data)
    rows = [json.loads(line) for line in open(data / f"{args.split}.jsonl")]
    rows = [r for r in rows if r.get("pitch_deg") is not None]
    crops = [cv2.imread(str(data / r["image"]), cv2.IMREAD_COLOR) for r in rows]
    gt = np.array([[r["yaw_deg"], r["pitch_deg"]] for r in rows], dtype=np.float64)
    abs_yaw = np.abs((gt[:, 0] + 180.0) % 360.0 - 180.0)
    yaw_bin = np.clip(np.digitize(abs_yaw, YAW_EDGES[1:-1]), 0, len(YAW_EDGES) - 2)
    pitch_bin = np.clip(np.digitize(gt[:, 1], PITCH_EDGES[1:-1]), 0, len(PITCH_EDGES) - 2)
    print(f"{args.split}: n={len(rows)}")

    by_yaw: dict[str, list] = {}
    by_pitch: dict[str, list] = {}
    results: dict[str, dict] = {}
    for spec in args.onnx:
        name, _, path = spec.partition("=") if "=" in spec else ("", "", spec)
        p = Path(path)
        name = name or p.parent.name
        e = wrap_abs(predict(p, crops, args.batch, args.cpu), gt)
        by_yaw[name] = [(float(e[yaw_bin == b, 0].mean()), float(e[yaw_bin == b, 1].mean()))
                        for b in range(len(YAW_EDGES) - 1)]
        by_pitch[name] = [(float(e[pitch_bin == b, 0].mean()), float(e[pitch_bin == b, 1].mean()))
                          for b in range(len(PITCH_EDGES) - 1)]
        results[name] = {"by_abs_yaw": by_yaw[name], "by_pitch": by_pitch[name]}
        print(f"  {name:12s} overall yaw {e[:, 0].mean():.2f}  pitch {e[:, 1].mean():.2f}", flush=True)

    yaw_keys = [f"\\|yaw\\|<br>{a}–{b}" for a, b in zip(YAW_EDGES[:-1], YAW_EDGES[1:])]
    pitch_keys = [f"pitch<br>{a}..{b}" for a, b in zip(PITCH_EDGES[:-1], PITCH_EDGES[1:])]
    yaw_n = [int((yaw_bin == b).sum()) for b in range(len(YAW_EDGES) - 1)]
    pitch_n = [int((pitch_bin == b).sum()) for b in range(len(PITCH_EDGES) - 1)]
    print()
    print(table(f"MAAE (deg, yaw / pitch) by |yaw| bin ({args.split})", yaw_keys, by_yaw, yaw_n))
    print()
    print(table(f"MAAE (deg, yaw / pitch) by pitch bin ({args.split})", pitch_keys, by_pitch, pitch_n))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"split": args.split, "n": len(rows), "yaw_edges": YAW_EDGES,
                       "pitch_edges": PITCH_EDGES, "yaw_n": yaw_n, "pitch_n": pitch_n,
                       "results": results}, f, indent=1)
        print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
