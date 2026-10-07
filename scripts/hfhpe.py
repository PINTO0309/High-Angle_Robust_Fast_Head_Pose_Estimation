"""HFHPE (High-Angle Robust Fast Head Pose Estimation): one-pass integrated model.

Internal roll estimation → in-graph derotation → frozen yaw/pitch body, all in a
single forward / single ONNX graph.

Structure:
  input x (N,3,S,S)  (center05-normalized: x/127.5 - 1)
    ├─ downscale to roll_size with F.interpolate → roll branch (YawNet reused)
    │    → roll unit vector (cos_r, sin_r) and κ_r
    ├─ build the rotation matrix directly from cos/sin (no atan2) and inverse-rotate
    │  the input by -θ̂ with GridSample (derotation; border padding = edge replication)
    └─ re-normalize the derotated image for the body (center05 → ImageNet etc.)
         → frozen yaw/pitch body (forward_full) → yaw / pitch unit vectors and κ

Output: cos_sin (N,6) = [cos_y, sin_y, cos_p, sin_p, cos_r, sin_r],
        kappa (N,3) = [κ_y, κ_p, κ_r].
The roll output is the raw estimate before correction (the detected tilt).

κ_roll gating (gate_mode, default "none" = disabled):
  To prevent derotation by a wrong angle from breaking the body on frames with low
  κ_r (uncertain roll), only the rotation used for derotation is pulled toward the
  identity rotation by a gate coefficient g ∈ [0,1].
    g:  "sigmoid" -> g = sigmoid(a·(κ_r − τ)) (effectively hard for large a)
        "relu"    -> g = clip(a·relu(κ_r − τ), 0, 1) (piecewise linear; robust to
                     quantization)
    application (vector blend): c' = 1 + g·(c−1), s' = g·s → re-normalize with eps
  Built only from Sub/Mul/Add/Sigmoid/Relu/Clip/Sqrt/Div, without Where / Greater /
  Cast (in consideration of operator support on edge EPs / NPUs). The roll and κ_r
  in the cos_sin output remain the raw pre-gate values (downstream does not depend
  on whether gating is used).

The roll convention is the same as scripts/roll_dataset.py (counter-clockwise = +).
The rotation direction is numerically verified by tests: feeding an image warped by
roll_warp(+θ) together with the GT (cosθ, sinθ) recovers the original image
(sampling matrix of _upright = [[c, s], [-s, c]]).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from yaw_dataset import NORMS


def _base_grid(size: int) -> torch.Tensor:
    """Normalized pixel-center grid for align_corners=False: (1, size, size, 1, 2).

    To avoid leaving a batch-dependent Reshape in the graph, the grid is held from
    the start in the spatial shape grid_sample expects, and the rotation is applied
    by a broadcast MatMul.
    """
    coords = (torch.arange(size, dtype=torch.float32) * 2.0 + 1.0) / size - 1.0
    ys, xs = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack([xs, ys], dim=-1).reshape(1, size, size, 1, 2)


class HFHPE(nn.Module):
    """One-pass composition of the roll branch + frozen yaw/pitch body.

    body is a 6-output model with forward_full(x) -> (yaw_unit, kappa_y, pitch_unit,
    kappa_p) (Dinov3YawNet(pitch_head=True); a future integrated student uses the
    same API).
    """

    def __init__(self, rollnet: nn.Module, body: nn.Module,
                 body_norm: str = "imagenet", body_size: int = 320,
                 roll_size: int = 64, gate_mode: str = "none",
                 gate_tau: float = 20.0, gate_a: float = 0.5) -> None:
        super().__init__()
        if gate_mode not in ("none", "sigmoid", "relu"):
            raise ValueError(f"gate_mode must be none/sigmoid/relu: {gate_mode}")
        self.rollnet = rollnet
        self.body = body
        self.roll_size = roll_size
        self.body_size = body_size
        self.gate_mode = gate_mode
        self.gate_tau = float(gate_tau)
        self.gate_a = float(gate_a)
        self.register_buffer("grid0", _base_grid(body_size), persistent=False)
        # center05 -> body_norm re-normalization (per-channel affine transform)
        mean, std = NORMS[body_norm]
        scale = 0.5 / std
        shift = (0.5 - mean) / std
        self.register_buffer(
            "norm_scale", torch.tensor(scale, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            "norm_shift", torch.tensor(shift, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False)

    def _upright(self, x: torch.Tensor, cos_r: torch.Tensor,
                 sin_r: torch.Tensor) -> torch.Tensor:
        """Rotate back by the estimated roll: input coords = R(+θ̂) @ output coords."""
        rot = torch.stack([
            torch.stack([cos_r, sin_r], dim=-1),
            torch.stack([-sin_r, cos_r], dim=-1),
        ], dim=-2)                                   # (N, 2, 2)
        rot = rot.unsqueeze(1).unsqueeze(1)          # (N, 1, 1, 2, 2)
        # (1,H,W,1,2) @ (N,1,1,2,2)^T -> (N,H,W,1,2). The batch dim grows by
        # broadcasting, so no N-dependent Reshape appears in the graph
        grid = torch.matmul(self.grid0, rot.transpose(-1, -2)).squeeze(-2)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border",
                             align_corners=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # With size=, the Resize sizes constant hard-codes batch dim 1 and breaks the
        # N-batch conversion, so export with scale_factor (scales=[1,1,s,s]).
        # When a student body (e.g. 64px) matches roll_size, skip the downscale entirely
        if self.roll_size == self.body_size:
            xr = x
        else:
            xr = F.interpolate(x, scale_factor=self.roll_size / self.body_size,
                               mode="bilinear", align_corners=False,
                               recompute_scale_factor=False)
        roll_unit, kappa_r = self.rollnet.forward_with_kappa(xr)
        cos_r, sin_r = roll_unit[:, 0], roll_unit[:, 1]
        if self.gate_mode != "none":
            # at low κ_r confidence, pull the derotation rotation toward identity
            # (arithmetic only, no Where)
            if self.gate_mode == "sigmoid":
                g = torch.sigmoid(self.gate_a * (kappa_r - self.gate_tau))
            else:                          # "relu": piecewise-linear pseudo step
                g = torch.clamp(self.gate_a * torch.relu(kappa_r - self.gate_tau),
                                0.0, 1.0)
            cg = 1.0 + g * (cos_r - 1.0)   # blend of (c,s) and the identity (1,0)
            sg = g * sin_r
            n = torch.sqrt(cg * cg + sg * sg + 1e-12)
            cos_r, sin_r = cg / n, sg / n
        xu = self._upright(x, cos_r, sin_r)
        xb = xu * self.norm_scale + self.norm_shift
        yaw_unit, kappa_y, pitch_unit, kappa_p = self.body.forward_full(xb)
        cos_sin = torch.cat([yaw_unit, pitch_unit, roll_unit], dim=1)
        kappa = torch.stack([kappa_y, kappa_p, kappa_r], dim=1)
        return cos_sin, kappa
