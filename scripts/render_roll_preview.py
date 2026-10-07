#!/usr/bin/env python3
"""CLI that renders the validation preview (3x3) of the HFHPE roll branch offline.

Reproduces the deterministic val rotation of RollDataset (θ derived from the row
index, which is the GT as is) and overlays the GT θ (green) and the prediction (red)
on the rotated image as a full-surround dial. The needle is the "direction of the top
of the head": up = 0° (upright), +θ is visually counter-clockwise.

Usage:
    uv run python scripts/render_roll_preview.py --ckpt runs/hfhpe_roll_064_unified

By default the output is <ckpt folder>/val_preview_best<set>.png (changeable with
--out; it always differs from the training-side automatic output
val_preview_best.png).
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from export_hfhpe import load_rollnet
from roll_dataset import RollDataset, roll_warp
from val_preview import CELL, GRID, _circ_diff

ROOT = Path(__file__).resolve().parent.parent


def _roll_needle(img: np.ndarray, center: tuple[int, int], radius: int,
                 roll_deg: float, color: tuple[int, int, int],
                 thickness: int) -> None:
    rad = math.radians(roll_deg)
    # top of the head: 0° = straight up (upright), +θ visually CCW (x right, y down)
    dx, dy = -math.sin(rad), -math.cos(rad)
    tip = (int(center[0] + dx * radius), int(center[1] + dy * radius))
    cv2.arrowedLine(img, center, tip, color, thickness, cv2.LINE_AA, tipLength=0.25)


def _draw_roll_cell(im_bgr: np.ndarray, gt: float, pred: float) -> np.ndarray:
    cell = im_bgr
    radius = 36
    center = (CELL - radius - 8, CELL - radius - 8)
    overlay = cell.copy()
    cv2.circle(overlay, center, radius + 6, (255, 255, 255), -1, cv2.LINE_AA)
    cell = cv2.addWeighted(overlay, 0.55, cell, 0.45, 0)
    cv2.circle(cell, center, radius, (60, 60, 60), 2, cv2.LINE_AA)
    # draw the 0° tick (upright = top of the head up) at the top
    cv2.circle(cell, (center[0], center[1] - radius), 3, (60, 60, 60), -1,
               cv2.LINE_AA)
    _roll_needle(cell, center, radius - 4, gt, (0, 200, 0), 3)     # GT: green
    _roll_needle(cell, center, radius - 4, pred, (0, 0, 255), 2)   # pred: red
    err = _circ_diff(gt, pred)
    lines = [(f"gt   {gt:6.1f}", (0, 200, 0)),
             (f"pred {pred:6.1f}", (0, 0, 255)),
             (f"err  {err:6.1f}", (255, 255, 255))]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    tw = max(cv2.getTextSize(t, font, scale, 1)[0][0] for t, _ in lines)
    overlay = cell.copy()
    cv2.rectangle(overlay, (0, 0), (tw + 28, 18 * len(lines) + 8), (0, 0, 0), -1)
    cell = cv2.addWeighted(overlay, 0.65, cell, 0.35, 0)
    for i, (text, color) in enumerate(lines):
        cv2.putText(cell, text, (6, 18 + 18 * i), font, scale, color, 1,
                    cv2.LINE_AA)
    return cell


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="run directory (best_*.pt auto-detected) or a .pt file")
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "yawpitchpose"))
    ap.add_argument("--out", type=str, default="",
                    help="output PNG (default: <ckpt folder>/"
                         "val_preview_best[N].png)")
    ap.add_argument("--set", type=int, default=1, dest="sample_set",
                    help="sample set number (next 9 candidates that do not overlap "
                         "the previous sets)")
    args = ap.parse_args()

    model, ck_path, size = load_rollnet(args.ckpt)
    device = "cuda"
    model.to(device).eval()
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    epoch = int(ck.get("epoch", -1))
    maae = float((ck.get("metrics") or {}).get("maae", float("nan")))

    ds = RollDataset(args.data, "val", size, train=False)

    # enumerate the deterministic val θ (same formula as RollDataset.__getitem__)
    # without reading images and pick the rows nearest to the target angles (40°
    # steps). --set gives the next candidates after excluding the previous ones
    thetas = np.array([
        float(np.random.default_rng(ds.seed * 1_000_003 + i).uniform(0.0, 360.0))
        for i in range(len(ds))])
    exclude: set[int] = set()
    indices: list[int] = []
    for _ in range(max(1, args.sample_set)):
        indices = []
        for k in range(GRID * GRID):
            target = k * 360.0 / (GRID * GRID)
            d = np.minimum(np.abs(thetas - target), 360.0 - np.abs(thetas - target))
            for i in np.argsort(d):
                if i not in exclude and i not in indices:
                    indices.append(int(i))
                    break
        exclude.update(indices)

    xs = torch.stack([ds[i][0] for i in indices]).to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        pred = model(xs)
    pred = pred.float().cpu()
    pred_deg = (torch.rad2deg(torch.atan2(pred[:, 1], pred[:, 0])) % 360.0).tolist()

    cells = []
    for i, p in zip(indices, pred_deg):
        r = ds.rows[i]
        im = cv2.imread(str(ds.root / r["image"]), cv2.IMREAD_COLOR)
        theta = thetas[i]
        big = roll_warp(im, theta, scale=1.0, tx=0.0, ty=0.0, hflip=False,
                        out_size=CELL)
        cells.append(_draw_roll_cell(big, float(theta), float(p)))
    rows = [np.hstack(cells[k * GRID:(k + 1) * GRID]) for k in range(GRID)]
    grid = np.vstack(rows)
    header = np.full((56, grid.shape[1], 3), 32, np.uint8)
    cv2.putText(header, f"epoch {epoch}  val maae {maae:.2f} deg  "
                "(roll; GT=green, pred=red)",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.putText(header, "needle = top of head: up=0 (upright), +CCW; "
                "val rotation is synthetic (deterministic per row)",
                (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
                cv2.LINE_AA)
    # always numbered, even for --set 1, so that it does not collide with the
    # automatic output on the training side
    suffix = str(max(1, args.sample_set))
    out = (Path(args.out) if args.out
           else ck_path.parent / f"val_preview_best{suffix}.png")
    cv2.imwrite(str(out), np.vstack([header, grid]))
    print(f"saved: {out} (ckpt={ck_path.name}, size={size}, epoch={epoch}, "
          f"maae={maae})")


if __name__ == "__main__":
    main()
