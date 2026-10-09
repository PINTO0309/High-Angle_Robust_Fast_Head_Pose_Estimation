#!/usr/bin/env python3
"""Overlay the teacher's CLS attention on the nine cells of a render_hfhpe_axes_sheet.py sheet.

Reads the JSON written next to the sheet (images, θ, predictions) and rebuilds the same rotated
cell images. The attention is computed on the image the integrated model's body actually sees,
i.e. the cell derotated by the predicted roll (--no-derotate computes it on the rotated cell
itself), and the map is rotated back by the predicted roll into the cell frame before it is
overlaid. The attention is the last-block CLS → patch attention of the DINOv3 ViT-L teacher
(head mean, normalized by its maximum).

Usage:
    uv run python scripts/render_hfhpe_axes_heatmap.py \\
        --sheet-json runs/vitt_distill_064_yp_unified/hfhpe_axes_sheet_roll_canvas_synth.json \\
        --ckpt runs/dinov3_vitl16_320_yp_unified_teacher

Default output: <sheet-json stem>_attention.png.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from export_onnx import load_model
from render_hfhpe_axes_sheet import CELL, _signed, short_name
from roll_dataset import roll_warp
from yaw_dataset import normalize_image

ROOT = Path(__file__).resolve().parent.parent


class ClsAttention:
    """Wrap compute_attention of the last block and record the CLS → token attention (head mean)."""

    def __init__(self, backbone: torch.nn.Module) -> None:
        self.attn = backbone.blocks[-1].attn
        self.n_prefix = 1 + int(getattr(backbone, "n_storage_tokens", 0))
        self.last: torch.Tensor | None = None
        self._orig = self.attn.compute_attention
        attn = self.attn

        def wrapped(qkv: torch.Tensor, attn_bias=None, rope=None) -> torch.Tensor:
            B, N, _ = qkv.shape
            C = attn.qkv.in_features
            q, k, _ = torch.unbind(qkv.reshape(B, N, 3, attn.num_heads, C // attn.num_heads).transpose(1, 3), 2)
            if rope is not None:
                q, k = attn.apply_rope(q, k, rope)
            logits = (q[:, :, :1].float() @ k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
            self.last = logits.softmax(dim=-1)[:, :, 0].mean(dim=1).detach()   # (B, N)
            return self._orig(qkv, attn_bias=attn_bias, rope=rope)

        self.attn.compute_attention = wrapped

    def patch_map(self, side: int) -> np.ndarray:
        a = self.last[0, self.n_prefix:]
        a = a / a.max().clamp_min(1e-12)
        return a.reshape(side, side).cpu().numpy()

    def close(self) -> None:
        self.attn.compute_attention = self._orig


def overlay(bgr: np.ndarray, heat01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h = cv2.resize(heat01.astype(np.float32), (bgr.shape[1], bgr.shape[0]),
                   interpolation=cv2.INTER_CUBIC)
    color = cv2.applyColorMap((np.clip(h, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 1.0 - alpha, color, alpha, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet-json", type=str,
                    default=str(ROOT / "runs" / "vitt_distill_064_yp_unified"
                                / "hfhpe_axes_sheet_roll_canvas_synth.json"))
    ap.add_argument("--ckpt", type=str,
                    default=str(ROOT / "runs" / "dinov3_vitl16_320_yp_unified_teacher"),
                    help="DINOv3 teacher checkpoint whose attention is drawn (run directory or "
                         ".pt); the students have no CLS attention to show")
    ap.add_argument("--no-derotate", action="store_true",
                    help="compute the attention on the rotated cell image instead of the image "
                         "derotated by the predicted roll")
    ap.add_argument("--alpha", type=float, default=0.45, help="heatmap opacity")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    meta = json.load(open(args.sheet_json))
    data = Path(meta["data"])
    model, ck_path, size, _, norm = load_model(args.ckpt, 0)
    model.cuda().eval()
    cls_attn = ClsAttention(model.backbone)
    print(f"teacher: {ck_path} (size={size}, norm={norm})  derotate={not args.no_derotate}")

    cells = []
    for pk in meta["picks"]:
        canvas = cv2.imread(str(data / pk["image"]), cv2.IMREAD_COLOR)
        theta = float(pk["roll_deg"])
        roll_hat = 0.0 if args.no_derotate else _signed(float(pk["pred"]["roll"]))
        cell_bgr = roll_warp(canvas, theta, 1.0, 0.0, 0.0, False, out_size=size,
                             content_frac=1.0 / pk["canvas"])
        # the image the body sees = the cell rotated back by the predicted roll (one warp)
        body_bgr = roll_warp(canvas, theta - roll_hat, 1.0, 0.0, 0.0, False, out_size=size,
                             content_frac=1.0 / pk["canvas"])
        x = normalize_image(cv2.cvtColor(body_bgr, cv2.COLOR_BGR2RGB), norm)[None].cuda()
        with torch.no_grad():
            model.forward_full(x)                           # records the last-block attention
        heat = cv2.resize(cls_attn.patch_map(size // 16).astype(np.float32), (size, size),
                          interpolation=cv2.INTER_CUBIC)
        if roll_hat:                                        # map of the derotated image -> cell frame
            m = cv2.getRotationMatrix2D((size / 2, size / 2), roll_hat, 1.0)
            heat = cv2.warpAffine(heat, m, (size, size), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        c = overlay(cv2.resize(cell_bgr, (CELL, CELL), interpolation=cv2.INTER_AREA),
                    cv2.resize(heat, (CELL, CELL)), alpha=args.alpha)
        ov = c.copy()
        cv2.rectangle(ov, (0, 0), (CELL, 42), (0, 0, 0), -1)
        c = cv2.addWeighted(ov, 0.65, c, 0.35, 0)
        pr = pk["pred"]
        cv2.putText(c, short_name(pk["image"]), (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(c, f"CLS attn | pr y{_signed(pr['yaw']):+.0f} p{pr['pitch']:+.0f} "
                       f"r{_signed(pr['roll']):+.0f}",
                    (4, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(c)
        print(f"  {short_name(pk['image']):14s} theta {theta:+6.1f}  roll_hat {roll_hat:+6.1f}")
    cls_attn.close()

    cols = int(np.ceil(np.sqrt(len(cells))))
    while len(cells) % cols:
        cells.append(np.zeros_like(cells[0]))
    grid = np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])
    out = Path(args.out) if args.out else Path(args.sheet_json).with_name(
        Path(args.sheet_json).stem + "_attention.png")
    cv2.imwrite(str(out), grid)
    print(f"sheet: {out}")


if __name__ == "__main__":
    main()
