#!/usr/bin/env python3
"""HGNetV2Yaw: yaw/pitch student with a PP-HGNetV2 B0 backbone (regular CNN variant).

Compared with the from-scratch MBConv YawNet (lite variant, 0.77M), this is the
regular variant with ImageNet-pretrained PP-HGNetV2 B0 (1.85M) as the backbone
(~1.9M in total). The input is the same center05-normalized 64px as YawNet, and
the head follows the same convention (GAP -> dropout -> FC; output layout and the
forward-family API are identical to YawNet).

For 64px input, out_stride=16 (the stage3 downsample is made stride 1) keeps a
4x4 final feature map. Trained with distill_yawnet.py --student-arch hgnetv2;
the checkpoint model_type is "hgnetv2_yaw".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pphgnetv2 import PPHGNetV2B0


class HGNetV2Yaw(nn.Module):
    """Same output/API convention as YawNet (forward returns (N,2) unit vectors)."""

    def __init__(self, dropout: float = 0.2, kappa_head: bool = False,
                 pitch_head: bool = False, out_stride: int = 16,
                 backbone_ckpt: str = "") -> None:
        super().__init__()
        if pitch_head and not kappa_head:
            raise ValueError("pitch_head=True requires kappa_head=True (fixed 6 outputs)")
        self.kappa_head = kappa_head
        self.pitch_head = pitch_head
        self.backbone = PPHGNetV2B0(out_stride=out_stride)
        if backbone_ckpt:
            self.backbone.load_pretrained(backbone_ckpt)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.backbone.out_channels,
                            6 if pitch_head else (3 if kappa_head else 2))
        if kappa_head:
            # softplus(1.85) ≈ 2.0 = start from the same initial κ as YawNet
            with torch.no_grad():
                self.fc.bias[2] = 1.85
                if pitch_head:
                    self.fc.bias[5] = 1.85

    def _head_out(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._head_out(x)
        return F.normalize(out[:, :2], dim=1)

    def forward_with_kappa(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.kappa_head:
            raise RuntimeError("forward_with_kappa cannot be used with kappa_head=False")
        out = self._head_out(x)
        return (F.normalize(out[:, :2], dim=1),
                F.softplus(out[:, 2]).clamp(1e-3, 100.0))

    def forward_full(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                     torch.Tensor, torch.Tensor]:
        if not self.pitch_head:
            raise RuntimeError("forward_full cannot be used with pitch_head=False")
        out = self._head_out(x)
        return (F.normalize(out[:, :2], dim=1),
                F.softplus(out[:, 2]).clamp(1e-3, 100.0),
                F.normalize(out[:, 3:5], dim=1),
                F.softplus(out[:, 5]).clamp(1e-3, 100.0))
