#!/usr/bin/env python3
"""Run the integrated HFHPE ONNX on nine images and draw a 3-axis sheet in the AFLW2000-preview look.

Same cell layout as the `--preview` of the evaluation scripts: 224 px cells, the predicted 3 axes
drawn at the center, and a dark banner on top with the name and the gt / pr yaw, pitch and roll.
The images come from a roll-canvas root (data/roll_canvas_synth, rows with canvas=1.8). Each cell
applies a known in-plane rotation θ to the canvas before cutting out the contract frame, so the
roll ground truth is θ itself (the corners are filled from the ring of real pixels, no wedges).
The yaw / pitch ground truth comes from --gt-labels (default: labels_fixed.jsonl of yawpitchpose),
looked up by file name; the roll-canvas jsonl only holds a copy of the labels as of its build time
and may miss later hand corrections.

The selection is deterministic: for each entry of --yaw-targets, the next --pitch-targets and
--rolls values are assigned in turn and the row minimizing |yaw diff| + |pitch diff| is picked
(no image twice, a small penalty for repeating a source). Pass a previous output JSON to --picks
to redraw the same images with the same θ, or to --exclude to pick different images.

Usage:
    uv run python scripts/render_hfhpe_axes_sheet.py \\
        --onnx runs/vitt_distill_064_yp_unified/hfhpe_vitt_1x3x64x64.onnx

Default output: <onnx folder>/hfhpe_axes_sheet_<data name>.png plus a .json of the same name.
"""
import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np

from render_hfhpe_preview import draw_axis
from roll_dataset import roll_warp

ROOT = Path(__file__).resolve().parent.parent
CELL = 224


def _signed(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _circ(a: float, b: float) -> float:
    return abs(_signed(a - b))


def short_name(image: str) -> str:
    """Banner label: source prefix plus the trailing number (e.g. s004 011538)."""
    stem = Path(image).stem
    prefix = stem.split("_")[0]
    nums = re.findall(r"\d{4,}", stem)
    return f"{prefix} {nums[-1]}" if nums else stem[:18]


def load_rows(data: Path, split: str, gt_labels: str) -> list[dict]:
    rows = [json.loads(line) for line in open(data / f"{split}.jsonl")]
    if gt_labels:   # override with the current labels (or fill them in for a manifest)
        lab = {Path(g["image"]).name: g
               for g in (json.loads(line) for line in open(gt_labels))}
        for r in rows:
            g = lab.get(Path(r["image"]).name)
            if g is not None:
                r["yaw_deg"] = g["yaw_deg"]
                r["pitch_deg"] = g.get("pitch_deg")
    return [r for r in rows if r.get("yaw_deg") is not None and r.get("pitch_deg") is not None]


def select(rows: list[dict], yaw_targets: list[float], pitch_targets: list[float],
           rolls: list[float], exclude: set[str] | None = None) -> list[dict]:
    """Per yaw target, pick the row closest to the yaw and pitch targets (no image twice)."""
    used_img, used_src, picks = set(exclude or ()), {}, []
    for i, yt in enumerate(yaw_targets):
        pt = pitch_targets[i % len(pitch_targets)]
        best, best_score = None, float("inf")
        for r in rows:
            if r["image"] in used_img:
                continue
            score = _circ(r["yaw_deg"], yt) + abs(r["pitch_deg"] - pt) + 10.0 * used_src.get(r["source"], 0)
            if score < best_score:
                best, best_score = r, score
        used_img.add(best["image"])
        used_src[best["source"]] = used_src.get(best["source"], 0) + 1
        picks.append({"image": best["image"], "source": best["source"],
                      "yaw_deg": float(best["yaw_deg"]), "pitch_deg": float(best["pitch_deg"]),
                      "canvas": float(best.get("canvas", 1.0)),
                      "roll_deg": float(rolls[i % len(rolls)]),
                      "target": {"yaw": yt, "pitch": pt}})
    return picks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str,
                    default=str(ROOT / "runs" / "vitt_distill_064_yp_unified"
                                / "hfhpe_vitt_1x3x64x64.onnx"),
                    help="integrated HFHPE ONNX (cos_sin (N,6), center05 input)")
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "roll_canvas_synth"),
                    help="roll-canvas root (<root>/<split>.jsonl and images/)")
    ap.add_argument("--split", type=str, default="val",
                    help="jsonl to read (val / train / manifest)")
    ap.add_argument("--gt-labels", type=str,
                    default=str(ROOT / "data" / "yawpitchpose" / "labels_fixed.jsonl"),
                    help="label jsonl the yaw/pitch ground truth is looked up in by file name "
                         "(empty string: use the values of the canvas jsonl as they are)")
    ap.add_argument("--yaw-targets", type=str, default="0,40,80,120,160,200,240,280,320")
    ap.add_argument("--pitch-targets", type=str, default="0,25,-25",
                    help="pitch targets (deg) assigned to the yaw targets in turn")
    ap.add_argument("--rolls", type=str, default="0,-20,20,30,0,-30,-15,15,0",
                    help="in-plane rotation θ per cell (deg, CCW positive; becomes the roll GT)")
    ap.add_argument("--picks", type=str, default="",
                    help="previous output JSON (redraw the same images with the same θ)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="output JSON whose images are excluded from the selection (repeatable)")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    import onnxruntime as ort  # noqa: PLC0415
    onnx_path = Path(args.onnx)
    providers = (["CPUExecutionProvider"] if args.cpu
                 else ["CUDAExecutionProvider", "CPUExecutionProvider"])
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    size = int(sess.get_inputs()[0].shape[-1])
    assert sess.get_outputs()[0].shape[-1] == 6, "pass an integrated ONNX (cos_sin (N,6))"

    data = Path(args.data)
    if args.picks:
        picks = json.load(open(args.picks))["picks"]
    else:
        rows = load_rows(data, args.split, args.gt_labels)
        exclude = {pk["image"] for f in args.exclude for pk in json.load(open(f))["picks"]}
        picks = select(rows,
                       [float(v) for v in args.yaw_targets.split(",")],
                       [float(v) for v in args.pitch_targets.split(",")],
                       [float(v) for v in args.rolls.split(",")], exclude)

    cells = []
    for pk in picks:
        im = cv2.imread(str(data / pk["image"]), cv2.IMREAD_COLOR)
        warped = roll_warp(im, pk["roll_deg"], scale=1.0, tx=0.0, ty=0.0, hflip=False,
                           out_size=size, content_frac=1.0 / pk["canvas"])
        rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        cs = sess.run(["cos_sin"], {"images": rgb.transpose(2, 0, 1)[None]})[0][0]
        pr = [math.degrees(math.atan2(cs[1], cs[0])), math.degrees(math.atan2(cs[3], cs[2])),
              math.degrees(math.atan2(cs[5], cs[4]))]
        pk["pred"] = {"yaw": pr[0], "pitch": pr[1], "roll": pr[2]}
        gy, gp, gr = _signed(pk["yaw_deg"]), pk["pitch_deg"], pk["roll_deg"]
        # same cell composition as the evaluation --preview
        c = cv2.resize(warped, (CELL, CELL))
        draw_axis(c, pr[0], pr[1], pr[2], CELL // 2, CELL // 2, 78, 3, 1.0)
        ov = c.copy()
        cv2.rectangle(ov, (0, 0), (CELL, 42), (0, 0, 0), -1)
        c = cv2.addWeighted(ov, 0.65, c, 0.35, 0)
        cv2.putText(c, short_name(pk["image"]), (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(c, f"gt y{gy:+.0f} p{gp:+.0f} r{gr:+.0f} / "
                       f"pr y{_signed(pr[0]):+.0f} p{pr[1]:+.0f} r{_signed(pr[2]):+.0f}",
                    (4, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(c)
        print(f"{short_name(pk['image']):14s} gt y{gy:+6.1f} p{gp:+6.1f} r{gr:+6.1f} | "
              f"pr y{_signed(pr[0]):+6.1f} p{pr[1]:+6.1f} r{_signed(pr[2]):+6.1f}")

    cols = int(np.ceil(np.sqrt(len(cells))))
    while len(cells) % cols:
        cells.append(np.zeros_like(cells[0]))
    grid = np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])
    out = Path(args.out) if args.out else onnx_path.parent / f"hfhpe_axes_sheet_{data.name}.png"
    cv2.imwrite(str(out), grid)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump({"onnx": str(onnx_path), "data": str(data), "picks": picks}, f, indent=1,
                  ensure_ascii=False)
    print(f"sheet: {out}")


if __name__ == "__main__":
    main()
