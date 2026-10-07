#!/usr/bin/env python3
"""CLI that renders the validation preview (3x3) of the HFHPE roll branch offline.

From the deterministic val rotation (the same per-row θ as RollDataset), selects the
9 rows closest to target rolls in 40° steps and draws, in each cell, the "image after
applying the θ rotation" and a ring indicator:
  - the ring shows the direction of the top of the head. up = 0° (upright),
    counter-clockwise is + (same as the roll convention)
  - GT = green, prediction = red. Numbers (gt / pred / err) at the top left
Inference runs at the model's training resolution (default 64); the display applies
the same θ at 224px.

Usage:
    uv run python scripts/render_hfhpe_roll_preview.py \
        --ckpt runs/hfhpe_roll_064_unified

By default the output is <ckpt folder>/val_preview_best.png (changeable with --out).
"""
import argparse
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from roll_dataset import RollDataset, roll_warp
from val_preview import _circ_diff
from yawnet import YawNet

ROOT = Path(__file__).resolve().parent.parent
CELL = 224
GRID = 3


def _roll_needle(img: np.ndarray, center: tuple[int, int], radius: int,
                 roll_deg: float, color: tuple[int, int, int],
                 thickness: int) -> None:
    rad = math.radians(roll_deg)
    # top of the head: straight up at roll=0, + counter-clockwise (y-down coordinates)
    dx, dy = -math.sin(rad), -math.cos(rad)
    tip = (int(center[0] + dx * radius), int(center[1] + dy * radius))
    cv2.arrowedLine(img, center, tip, color, thickness, cv2.LINE_AA, tipLength=0.25)


def _draw_cell(im_bgr: np.ndarray, gt: float, pred: float) -> np.ndarray:
    cell = im_bgr
    radius = 36
    center = (CELL - radius - 8, CELL - radius - 8)
    overlay = cell.copy()
    cv2.circle(overlay, center, radius + 6, (255, 255, 255), -1, cv2.LINE_AA)
    cell = cv2.addWeighted(overlay, 0.55, cell, 0.45, 0)
    cv2.circle(cell, center, radius, (60, 60, 60), 2, cv2.LINE_AA)
    # draw the 0° tick (upright = top of the head straight up) at the top
    cv2.circle(cell, (center[0], center[1] - radius), 3, (60, 60, 60), -1, cv2.LINE_AA)
    _roll_needle(cell, center, radius - 4, gt, (0, 200, 0), 3)
    _roll_needle(cell, center, radius - 4, pred, (0, 0, 255), 2)
    err = _circ_diff(gt, pred)
    lines = [(f"gt   {gt:6.1f}", (0, 200, 0)),
             (f"pred {pred:6.1f}", (0, 0, 255)),
             (f"err  {err:6.1f}", (255, 255, 255))]
    font = cv2.FONT_HERSHEY_SIMPLEX
    tw = max(cv2.getTextSize(t, font, 0.5, 1)[0][0] for t, _ in lines)
    overlay = cell.copy()
    cv2.rectangle(overlay, (0, 0), (tw + 28, 18 * len(lines) + 8), (0, 0, 0), -1)
    cell = cv2.addWeighted(overlay, 0.65, cell, 0.35, 0)
    for i, (text, color) in enumerate(lines):
        cv2.putText(cell, text, (6, 18 + 18 * i), font, 0.5, color, 1, cv2.LINE_AA)
    return cell


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="run directory of the roll branch or best_*.pt")
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "yawpitchpose"))
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    path = Path(args.ckpt)
    if path.is_dir():
        path = sorted(path.glob("best_*.pt"))[0]
    ck: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    model = YawNet(width=float(ck["args"].get("width", 0.5)),
                   kappa_head=ck["model"]["fc.weight"].shape[0] == 3)
    model.load_state_dict(ck["model"])
    model.cuda().eval()
    size = int(ck["args"].get("size", 64))
    maae = float((ck.get("metrics") or {}).get("maae", float("nan")))

    ds = RollDataset(args.data, "val", size, train=False)
    # from the deterministic val θ (same formula as RollDataset), pick the rows
    # closest to the 40°-step targets
    thetas = [float(np.random.default_rng(ds.seed * 1_000_003 + i).uniform(0, 360))
              for i in range(len(ds.rows))]
    chosen: list[int] = []
    for k in range(GRID * GRID):
        target = k * 360.0 / (GRID * GRID)
        order = sorted(range(len(thetas)),
                       key=lambda i: _circ_diff(thetas[i], target))
        chosen.append(next(i for i in order if i not in chosen))

    cells: list[np.ndarray] = []
    with torch.no_grad():
        for i in chosen:
            x, _, roll = ds[i]
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(x[None].cuda()).float().cpu()[0]
            pred_deg = math.degrees(math.atan2(float(pred[1]),
                                               float(pred[0]))) % 360.0
            bgr = cv2.imread(str(ds.root / ds.rows[i]["image"]), cv2.IMREAD_COLOR)
            disp = roll_warp(bgr, float(roll), 1.0, 0.0, 0.0, False, CELL)
            cells.append(_draw_cell(disp, float(roll), pred_deg))
    rows = [np.hstack(cells[k * GRID:(k + 1) * GRID]) for k in range(GRID)]
    grid = np.vstack(rows)

    header = np.full((28, grid.shape[1], 3), 32, np.uint8)
    cv2.putText(header,
                f"{path.parent.name}  val roll maae {maae:.2f} deg  "
                f"(GT=green, pred=red; dial: up=0 upright, ccw=+)",
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    out = Path(args.out) if args.out else path.parent / "val_preview_best.png"
    cv2.imwrite(str(out), np.vstack([header, grid]))
    print(f"saved: {out} (ckpt={path.name}, size={size}, maae={maae})")


if __name__ == "__main__":
    main()
