"""Shared module that draws the validation preview image (3x3) on each best update.

Deterministically selects the 9 val rows closest to target yaws in 40° steps (the
same faces every time) and draws the face image and a ring indicator in each cell:
  - The ring is a top-down view (camera at the bottom). Needle: down = 0° (frontal),
    left = +90° (facing viewer's left), up = 180° (back of the head), right = 270°
  - GT = green, prediction = red. Numbers (gt / pred / err) in the top-left
With a pitch-integrated teacher (pitch_head=True), pitch is shown automatically too:
  - A side-view half dial in the bottom-left (right = 0°, up = +90° looking up,
    down = -90° looking down)
  - Numbers use the two-value "yaw / pitch" form. Rows without a pitch GT (yaw-only
    datasets) show "--" and draw only the prediction needle
Output is one image (runs/<run>/val_preview_best.png, overwritten at every best
update).
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch

CELL = 224
GRID = 3


def _circ_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def select_indices(ds, n: int = GRID * GRID,
                   exclude: set[int] | None = None,
                   allowed: list[int] | None = None) -> list[int]:
    """Select the val row closest to each target yaw (360/n steps), no duplicates
    (deterministic).

    Passing already-selected rows via exclude builds another set from the "next
    closest rows" (render_preview.py with --set 2 and later). Passing allowed
    restricts the candidates to those rows (conditional selection such as
    accessory or high pitch angle).
    """
    yaws = [float(r["yaw_deg"]) for r in ds.rows]
    pool = list(range(len(yaws))) if allowed is None else list(allowed)
    banned = set(exclude or ())
    chosen: list[int] = []
    for k in range(n):
        target = k * 360.0 / n
        order = sorted(pool, key=lambda i: _circ_diff(yaws[i], target))
        idx = next(i for i in order if i not in chosen and i not in banned)
        chosen.append(idx)
    return chosen


def select_diverse_indices(ds, accessory: set[str],
                           rng: "np.random.Generator",
                           n: int = GRID * GRID,
                           buckets: list[str] | None = None) -> list[int]:
    """Randomly select rows while keeping full-surround yaw (360/n steps) and mixing
    the conditions evenly.

    The condition buckets of the 9 cells cycle through [any, pitch0, looking up,
    looking down, accessory]:
      pitch0 = |pitch| <= 10 / looking up = pitch >= +25 / looking down = pitch <= -40 /
      accessory = images listed in accessory_index.json.
    Each bucket picks 1 image at random from the candidates within ±20° of the
    target yaw (the 40 closest if that pool is sparse). If a bucket has no
    candidates, it falls back to all rows.
    """
    yaws = [float(r["yaw_deg"]) for r in ds.rows]

    def pitch_of(i: int) -> float | None:
        p = ds.rows[i].get("pitch_deg")
        return None if p is None else float(p)

    def in_bucket(i: int, bucket: str) -> bool:
        if bucket == "any":
            return True
        if bucket == "acc":
            return Path(ds.rows[i]["image"]).name in accessory
        p = pitch_of(i)
        if p is None:
            return False
        return {"pitch0": abs(p) <= 10.0,
                "high": p >= 25.0,
                "low": p <= -40.0}[bucket]

    if buckets is None:
        buckets = ["any", "pitch0", "high", "low", "acc",
                   "pitch0", "high", "low", "acc"][:n]
    chosen: list[int] = []
    for k, bucket in enumerate(buckets):
        target = k * 360.0 / n
        pool = [i for i in range(len(yaws))
                if i not in chosen and in_bucket(i, bucket)]
        if not pool:                      # no match: fall back to all rows
            pool = [i for i in range(len(yaws)) if i not in chosen]
        near = [i for i in pool if _circ_diff(yaws[i], target) <= 20.0]
        if len(near) < 5:
            near = sorted(pool, key=lambda i: _circ_diff(yaws[i], target))[:40]
        chosen.append(int(rng.choice(near)))
    return chosen


def _needle(img: np.ndarray, center: tuple[int, int], radius: int,
            yaw_deg: float, color: tuple[int, int, int], thickness: int) -> None:
    rad = math.radians(yaw_deg)
    # top-down view: screen vector = (-sin(yaw), +cos(yaw)) (x right, y down)
    dx, dy = -math.sin(rad), math.cos(rad)
    tip = (int(center[0] + dx * radius), int(center[1] + dy * radius))
    cv2.arrowedLine(img, center, tip, color, thickness, cv2.LINE_AA, tipLength=0.25)


def _pitch_needle(img: np.ndarray, center: tuple[int, int], radius: int,
                  pitch_deg: float, color: tuple[int, int, int],
                  thickness: int) -> None:
    rad = math.radians(pitch_deg)
    # side view (head faces right): right = 0°, up = +90° (looking up), down = -90°
    dx, dy = math.cos(rad), -math.sin(rad)
    tip = (int(center[0] + dx * radius), int(center[1] + dy * radius))
    cv2.arrowedLine(img, center, tip, color, thickness, cv2.LINE_AA, tipLength=0.25)


def _draw_cell(im_bgr: np.ndarray, gt: float, pred: float,
               gt_pitch: float | None = None,
               pred_pitch: float | None = None) -> np.ndarray:
    cell = cv2.resize(im_bgr, (CELL, CELL), interpolation=cv2.INTER_LINEAR)
    # yaw ring (bottom-right, translucent backing)
    radius = 36
    center = (CELL - radius - 8, CELL - radius - 8)
    overlay = cell.copy()
    cv2.circle(overlay, center, radius + 6, (255, 255, 255), -1, cv2.LINE_AA)
    cell = cv2.addWeighted(overlay, 0.55, cell, 0.45, 0)
    cv2.circle(cell, center, radius, (60, 60, 60), 2, cv2.LINE_AA)
    # put the 0° tick (frontal = camera side) at the bottom
    cv2.circle(cell, (center[0], center[1] + radius), 3, (60, 60, 60), -1, cv2.LINE_AA)
    _needle(cell, center, radius - 4, gt, (0, 200, 0), 3)      # GT: green
    _needle(cell, center, radius - 4, pred, (0, 0, 255), 2)    # pred: red
    err = _circ_diff(gt, pred)

    if pred_pitch is not None:
        # pitch half dial (bottom-left, side view): right = 0°, up = +90°, down = -90°.
        # Only the right half of the arc is drawn, so the center is shifted toward the
        # left edge; for |pitch| > 90° (up to ±118°) the needle extends left of the
        # center, so leave a margin for that
        p_center = (20, CELL - radius - 8)
        overlay = cell.copy()
        cv2.ellipse(overlay, p_center, (radius + 6, radius + 6), 0, -90, 90,
                    (255, 255, 255), -1, cv2.LINE_AA)
        cell = cv2.addWeighted(overlay, 0.55, cell, 0.45, 0)
        cv2.ellipse(cell, p_center, (radius, radius), 0, -90, 90,
                    (60, 60, 60), 2, cv2.LINE_AA)
        cv2.line(cell, (p_center[0], p_center[1] - radius),
                 (p_center[0], p_center[1] + radius), (60, 60, 60), 1, cv2.LINE_AA)
        # put the 0° tick (level) at the right
        cv2.circle(cell, (p_center[0] + radius, p_center[1]), 3,
                   (60, 60, 60), -1, cv2.LINE_AA)
        if gt_pitch is not None:
            _pitch_needle(cell, p_center, radius - 4, gt_pitch, (0, 200, 0), 3)
        _pitch_needle(cell, p_center, radius - 4, pred_pitch, (0, 0, 255), 2)
        p_gt_s = f"{gt_pitch:6.1f}" if gt_pitch is not None else "    --"
        p_err_s = (f"{_circ_diff(gt_pitch, pred_pitch):6.1f}"
                   if gt_pitch is not None else "    --")
        lines = [(f"gt   {gt:6.1f} /{p_gt_s}", (0, 200, 0)),
                 (f"pred {pred:6.1f} /{pred_pitch:6.1f}", (0, 0, 255)),
                 (f"err  {err:6.1f} /{p_err_s}", (255, 255, 255))]
        scale = 0.45
    else:
        lines = [(f"gt   {gt:6.1f}", (0, 200, 0)),
                 (f"pred {pred:6.1f}", (0, 0, 255)),
                 (f"err  {err:6.1f}", (255, 255, 255))]
        scale = 0.5
    # Readability comes from a translucent black backing (a thick putText outline
    # is not used because the glyphs shift right and look doubled)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # width = max measured render width over all lines + right margin (about 2 chars)
    tw = max(cv2.getTextSize(t, font, scale, 1)[0][0] for t, _ in lines)
    overlay = cell.copy()
    cv2.rectangle(overlay, (0, 0), (tw + 28, 18 * len(lines) + 8), (0, 0, 0), -1)
    cell = cv2.addWeighted(overlay, 0.65, cell, 0.35, 0)
    for i, (text, color) in enumerate(lines):
        cv2.putText(cell, text, (6, 18 + 18 * i), font, scale, color, 1,
                    cv2.LINE_AA)
    return cell


@torch.no_grad()
def render_val_preview(model: torch.nn.Module, ds, indices: list[int],
                       out_path: str | Path, device: str, epoch: int,
                       maae: float, pitch_maae: float | None = None,
                       amp_dtype: torch.dtype = torch.float16) -> None:
    """Infer 9 samples and write the 3x3 preview (model is assumed to be in eval mode).

    If model.pitch_head is True (yaw+pitch integrated teacher), pitch is drawn
    automatically as well.
    """
    with_pitch = bool(getattr(model, "pitch_head", False))
    xs = torch.stack([ds[i][0] for i in indices]).to(device)
    with torch.autocast("cuda", dtype=amp_dtype):
        if with_pitch:
            pred, _, p_pred, _ = model.forward_full(xs)
            pred = pred.float().cpu()
            pitch_deg = torch.rad2deg(
                torch.atan2(p_pred.float().cpu()[:, 1],
                            p_pred.float().cpu()[:, 0])).tolist()
        else:
            pred = model(xs).float().cpu()
            pitch_deg = [None] * len(indices)
    pred_deg = (torch.rad2deg(torch.atan2(pred[:, 1], pred[:, 0])) % 360.0).tolist()

    cells: list[np.ndarray] = []
    for i, p, pp in zip(indices, pred_deg, pitch_deg):
        r = ds.rows[i]
        im = cv2.imread(str(ds.root / r["image"]), cv2.IMREAD_COLOR)
        gt_pitch = r.get("pitch_deg")
        cells.append(_draw_cell(
            im, float(r["yaw_deg"]), float(p),
            gt_pitch=None if gt_pitch is None else float(gt_pitch),
            pred_pitch=None if pp is None else float(pp)))
    rows = [np.hstack(cells[k * GRID:(k + 1) * GRID]) for k in range(GRID)]
    grid = np.vstack(rows)

    if with_pitch:
        p_str = f"{pitch_maae:.2f}" if pitch_maae is not None else "-"
        header_lines = [
            f"epoch {epoch}  val y_maae {maae:.2f} deg  p_maae {p_str} deg  "
            f"(GT=green, pred=red; values yaw / pitch)",
            "ring (bottom-right): down=0 front, left=+90  |  "
            "side dial (bottom-left): right=0 level, up=+90 look-up",
        ]
    else:
        header_lines = [f"epoch {epoch}  val maae {maae:.2f} deg  "
                        f"(GT=green, pred=red; ring: down=0 front, left=+90)"]
    header = np.full((8 + 20 * len(header_lines), grid.shape[1], 3), 32, np.uint8)
    for i, line in enumerate(header_lines):
        cv2.putText(header, line, (8, 19 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), np.vstack([header, grid]))
