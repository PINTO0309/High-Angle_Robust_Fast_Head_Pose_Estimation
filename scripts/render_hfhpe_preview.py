#!/usr/bin/env python3
"""CLI that renders the validation preview (3x3) of the integrated HFHPE ONNX with 3 axes.

Applies a RollDataset-style synthetic rotation θ to the 9 diverse val images (the same
deterministic selection as render_preview.py), runs the HFHPE ONNX (yaw+pitch+roll,
center05 input) and draws the 3 head pose axes in each cell. Instead of 2 rings, one
set of axes shows the 3 angles at once:
  - X axis = red: toward the subject's right cheek (image right when frontal/upright)
  - Y axis = green: toward the chin (image down when frontal/upright)
  - Z axis = blue: toward the nose tip (short projection when frontal, as it points
    at the camera)
Only the prediction axes are drawn; numbers (gt / pred / err) go in the top-left.

GT roll is the synthetic rotation θ itself (val crops are assumed upright). The
yaw/pitch GT is invariant to in-plane rotation, so the label values are kept.
--no-roll disables the rotation (GT roll=0).

Usage:
    uv run python scripts/render_hfhpe_preview.py \
        --onnx runs/vitt_distill_064_yp_unified_v6/hfhpe_vitt_1x3x64x64.onnx \
        --set 1

The output defaults to <onnx folder>/hfhpe_val_preview<set>.png (changeable with --out).
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from roll_dataset import roll_warp
from val_preview import CELL, GRID, _circ_diff, select_diverse_indices
from yaw_dataset import YawDataset

ROOT = Path(__file__).resolve().parent.parent


def _signed(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def draw_axis(img: np.ndarray, yaw: float, pitch: float, roll: float,
              cx: int, cy: int, size: float, thickness: int,
              bright: float) -> None:
    """Draw the 3 axes of the SynthYaw convention (yaw: +=viewer's left / pitch:
    +=looking up / roll: +=image-CCW) with the composition R_img_roll @ R_yaw @ R_pitch.

    Camera coordinates are x=image right, y=image down, z=away from the camera. The
    local basis when frontal and upright is X=(1,0,0) image right, Y=(0,1,0) chin=image
    down, Z=(0,0,-1) nose tip=toward the camera.
    roll is the same "rotation within the image plane" as the RollDataset synthetic
    rotation, so it is applied outermost (leftmost) as a rotation about the camera
    z axis (applying it about the head-local z axis leaves the nose tip unresponsive
    to roll and breaks down at high angles × large roll).
    The extrinsic closed form (HopeNet family) is not used because the pitch
    projection flips for |yaw|>90. bright is the brightness scale of the colors."""
    ps, yw, rl = (math.radians(pitch), math.radians(_signed(yaw)),
                  math.radians(_signed(roll)))
    cy_, sy = math.cos(yw), math.sin(yw)
    cp, sp = math.cos(ps), math.sin(ps)
    cr, sr = math.cos(rl), math.sin(rl)
    # +yaw moves the nose tip to image left (-x) / +pitch moves it to image up (-y) /
    # +roll rotates the whole image CCW (top of the head: 12 o'clock -> 9 o'clock)
    r_yaw = np.array([[cy_, 0, sy], [0, 1, 0], [-sy, 0, cy_]])
    r_pitch = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
    r_img_roll = np.array([[cr, sr, 0], [-sr, cr, 0], [0, 0, 1]])
    rot = r_img_roll @ r_yaw @ r_pitch
    c = int(255 * bright)
    for axis, color in [((1.0, 0.0, 0.0), (0, 0, c)),    # X: red (BGR)
                        ((0.0, 1.0, 0.0), (0, c, 0)),    # Y: green
                        ((0.0, 0.0, -1.0), (c, 0, 0))]:  # Z: blue
        v = rot @ np.array(axis)
        cv2.line(img, (cx, cy), (int(cx + size * v[0]), int(cy + size * v[1])),
                 color, thickness, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str,
                    default=str(ROOT / "runs" / "vitt_distill_064_yp_unified_v6"
                                / "hfhpe_vitt_1x3x64x64.onnx"),
                    help="integrated HFHPE ONNX (cos_sin (N,6), center05 input)")
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "yawpitchpose"))
    ap.add_argument("--set", type=int, default=1, dest="sample_set",
                    help="sample set number (deterministic; same selection as"
                         " render_preview.py)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the diverse sample selection (combined with --set"
                         " as in render_preview.py)")
    ap.add_argument("--no-roll", action="store_true",
                    help="skip the synthetic rotation (raw val images, GT roll=0)")
    ap.add_argument("--accessory-only", action="store_true",
                    help="select the 9 images only from those listed in"
                         " accessory_index.json (masks, sunglasses, etc.); adds _acc"
                         " to the output name")
    ap.add_argument("--out", type=str, default="",
                    help="output PNG (default: <onnx folder>/"
                         "hfhpe_val_preview<set>.png)")
    args = ap.parse_args()

    import onnxruntime as ort  # noqa: PLC0415
    onnx_path = Path(args.onnx)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_size = int(sess.get_inputs()[0].shape[-1])

    data_root = Path(args.data)
    ds = YawDataset(args.data, "val", in_size, train=False, input_norm="center05")

    # the same 9 diverse images as render_preview.py (deterministic)
    acc_path = data_root / "accessory_index.json"
    accessory = set(json.load(open(acc_path)).keys()) if acc_path.exists() else set()
    seed = args.seed + max(1, args.sample_set) - 1
    buckets = ["acc"] * (GRID * GRID) if args.accessory_only else None
    indices = select_diverse_indices(ds, accessory,
                                     np.random.default_rng(seed),
                                     buckets=buckets)

    # synthetic rotation θ (deterministic per set and cell position)
    rng = np.random.default_rng(20260910 + max(1, args.sample_set))
    thetas = [0.0 if args.no_roll else float(rng.uniform(0.0, 360.0))
              for _ in indices]

    cells = []
    errs = {"yaw": [], "pitch": [], "roll": []}
    for i, theta in zip(indices, thetas):
        row = ds.rows[i]
        im = cv2.imread(str(data_root / row["image"]), cv2.IMREAD_COLOR)
        gt_yaw = float(row["yaw_deg"]) % 360.0
        gt_pitch = float(row["pitch_deg"])
        gt_roll = theta

        # inference (apply rotation → RGB → center05)
        small = roll_warp(im, theta, scale=1.0, tx=0.0, ty=0.0, hflip=False,
                          out_size=in_size)
        x = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = (x / 127.5 - 1.0).transpose(2, 0, 1)[None]
        cos_sin = sess.run(["cos_sin"], {"images": x})[0][0]
        pr_yaw = math.degrees(math.atan2(cos_sin[1], cos_sin[0])) % 360.0
        pr_pitch = math.degrees(math.atan2(cos_sin[3], cos_sin[2]))
        pr_roll = math.degrees(math.atan2(cos_sin[5], cos_sin[4]))

        e_yaw = _circ_diff(gt_yaw, pr_yaw)
        e_pitch = _circ_diff(gt_pitch, pr_pitch)
        e_roll = _circ_diff(gt_roll, pr_roll)
        errs["yaw"].append(e_yaw)
        errs["pitch"].append(e_pitch)
        errs["roll"].append(e_roll)

        # display cell (same θ applied at 224px) + predicted 3 axes
        cell = roll_warp(im, theta, scale=1.0, tx=0.0, ty=0.0, hflip=False,
                         out_size=CELL)
        cx = cy = CELL // 2
        draw_axis(cell, pr_yaw, pr_pitch, pr_roll, cx, cy, 78, 3, 1.0)

        lines = [("yaw", gt_yaw, pr_yaw, e_yaw),
                 ("pitch", gt_pitch, pr_pitch, e_pitch),
                 ("roll", _signed(gt_roll), _signed(pr_roll), e_roll)]
        overlay = cell.copy()
        cv2.rectangle(overlay, (0, 0), (CELL, 62), (0, 0, 0), -1)
        cell = cv2.addWeighted(overlay, 0.6, cell, 0.4, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for k, (name, g, pdeg, e) in enumerate(lines):
            cv2.putText(cell, f"{name:<5} {g:6.1f} {pdeg:6.1f} e {e:4.1f}",
                        (4, 16 + 20 * k), font, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)
        cells.append(cell)

    rows = [np.hstack(cells[k * GRID:(k + 1) * GRID]) for k in range(GRID)]
    grid = np.vstack(rows)
    header = np.full((56, grid.shape[1], 3), 32, np.uint8)
    m = {k: float(np.mean(v)) for k, v in errs.items()}
    cv2.putText(header,
                f"HFHPE {onnx_path.name}  mean err: yaw {m['yaw']:.2f} "
                f"pitch {m['pitch']:.2f} roll {m['roll']:.2f} deg (9 cells)",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.putText(header,
                "axes (pred): X=red(right cheek) Y=green(chin) Z=blue(nose) "
                "| text: gt pred err",
                (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
                cv2.LINE_AA)
    suffix = str(max(1, args.sample_set)) + ("_acc" if args.accessory_only else "")
    out = (Path(args.out) if args.out
           else onnx_path.parent / f"hfhpe_val_preview{suffix}.png")
    cv2.imwrite(str(out), np.vstack([header, grid]))
    print(f"saved: {out} (onnx={onnx_path.name}, in={in_size}px, "
          f"mean err yaw {m['yaw']:.2f} / pitch {m['pitch']:.2f} / "
          f"roll {m['roll']:.2f} deg)")


if __name__ == "__main__":
    main()
