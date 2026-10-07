"""ViT-T student (vitt_distill backbone + biternion yaw/pitch head).

The backbone is a minimal vitt_distill-compatible ViT-T/16 (embed 192 / depth 12 /
heads 3, DINOv3-style axis-separated 2D RoPE) ported from src/hrffa/model/vit_tiny.py
of HRFFA (High-Angle_Robust_Fast_FaceAlignment, a repository by the same author).
Thanks to RoPE, changing the input resolution (64/96/128 etc., multiples of 16)
needs no positional embedding interpolation.

Initial weights: ckpts/vitt_distill.pt (same search order as dinov3_yaw.find_ckpt:
repository ckpts/ → HRFFA/ckpts). The weights were distilled assuming ImageNet
normalization, but following the HRFFA method they are converted losslessly to
center05 (((x/255) - 0.5) / 0.5) by exact folding into the patch embed conv
(the input contract is the same as the other students).

The head has the same structure and layout as Dinov3YawNet:
cat(cls, patch mean) 384 → MLP → 2 / 3 / 6 outputs
([cos_y, sin_y, κ_y, cos_p, sin_p, κ_p], head_flags compatible).
forward / forward_with_kappa / forward_full share the same API, so it drops
straight into distillation, export_onnx, val_preview and the HFHPE body.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from dinov3_yaw import find_ckpt, head_flags  # noqa: F401  (head_flags is re-exported)

VITT_CKPT_NAME = "vitt_distill.pt"
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
NORM_CONSTANTS = {
    "imagenet": (IMAGE_MEAN, IMAGE_STD),
    "center05": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}


class Rope2D(nn.Module):
    """Axis-separated 2D RoPE. periods is a buffer loaded from the checkpoint."""

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        assert head_dim % 4 == 0
        self.head_dim = head_dim
        self.register_buffer("periods", torch.empty(head_dim // 4),
                             persistent=True)
        self._cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def forward(self, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (h, w, str(self.periods.device))
        if key not in self._cache:
            dd = {"device": self.periods.device, "dtype": torch.float32}
            ch = torch.arange(0.5, h, **dd) / h
            cw = torch.arange(0.5, w, **dd) / w
            coords = torch.stack(torch.meshgrid(ch, cw, indexing="ij"),
                                 dim=-1).flatten(0, 1)          # (hw, 2)
            coords = 2.0 * coords - 1.0
            ang = 2 * math.pi * coords[:, :, None] / self.periods.float()[None, None]
            ang = ang.flatten(1, 2).tile(2)                      # (hw, D_head)
            self._cache[key] = (torch.sin(ang), torch.cos(ang))
        return self._cache[key]


def _rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def _rope_apply(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    dt = x.dtype
    x = x.float()
    return ((x * cos) + (_rope_rotate_half(x) * sin)).to(dt)


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor],
                prefix: int) -> torch.Tensor:
        b, n, c = x.shape
        # split qkv with 4-D tensors only (leaves no 5-D Reshape in the ONNX graph; the
        # channel order is [3][H][Dh], so splitting the last axis in 3 → transpose is
        # numerically identical)
        q, k, v = (t.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
                   for t in self.qkv(x).split(c, dim=-1))   # (B, heads, N, Dh)
        sin, cos = rope
        q = torch.cat([q[:, :, :prefix], _rope_apply(q[:, :, prefix:], sin, cos)], dim=2)
        k = torch.cat([k[:, :, :prefix], _rope_apply(k[:, :, prefix:], sin, cos)], dim=2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor],
                prefix: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope, prefix)
        return x + self.mlp(self.norm2(x))


class VitTiny(nn.Module):
    """Minimal vitt_distill-compatible ViT. forward returns (patch BCHW, cls BC)."""

    def __init__(self, embed_dim: int = 192, depth: int = 12, num_heads: int = 3,
                 patch_size: int = 16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = nn.ModuleDict(
            {"proj": nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)})
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.rope_embed = Rope2D(embed_dim // num_heads)
        self.blocks = nn.ModuleList(
            [_Block(embed_dim, num_heads) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.patch_embed["proj"](x)                  # (B, C, h, w)
        b, c, h, w = z.shape
        tokens = z.flatten(2).transpose(1, 2)            # (B, hw, C)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        rope = self.rope_embed(h, w)
        for blk in self.blocks:
            tokens = blk(tokens, rope, prefix=1)
        cls_out, patch = tokens[:, 0], tokens[:, 1:]
        patch = patch.transpose(1, 2).reshape(b, c, h, w)
        return patch, cls_out

    def load_flat_ckpt(self, path: Path, input_norm: str = "center05") -> None:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd.get("state_dict", sd))
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"vitt checkpoint structure mismatch: "
                               f"missing={list(missing)} unexpected={list(unexpected)}")
        if input_norm != "imagenet":
            self._fold_input_norm(input_norm)

    @torch.no_grad()
    def _fold_input_norm(self, input_norm: str) -> None:
        """Exactly convert pretrained weights from ImageNet to another input normalization.

        z_imagenet = a_c * z_new + b_c (per-channel affine) is folded into the weight
        scale and bias correction of the patch embed conv (lossless; the HRFFA
        history/026 method).
        """
        mean_new, std_new = NORM_CONSTANTS[input_norm]
        conv = self.patch_embed["proj"]
        for c in range(3):
            a = std_new[c] / IMAGE_STD[c]
            b = (mean_new[c] - IMAGE_MEAN[c]) / IMAGE_STD[c]
            conv.bias += conv.weight[:, c].sum(dim=(1, 2)) * b
            conv.weight[:, c] *= a


class VittYawNet(nn.Module):
    """vitt_distill backbone + biternion yaw (+pitch) head.

    Output layout and API are identical to Dinov3YawNet / YawNet (6 outputs):
    forward() = yaw unit vector, forward_with_kappa() = (yaw, κ_y),
    forward_full() = (yaw, κ_y, pitch, κ_p). The input contract is center05
    normalization (student convention; the pretrained weights are fold-converted
    at load time).
    """

    def __init__(self, ckpt_path: str | Path | None = None,
                 head_hidden: int = 256, dropout: float = 0.1,
                 pretrained: bool = True,
                 kappa_head: bool = False, pitch_head: bool = False,
                 input_norm: str = "center05") -> None:
        super().__init__()
        if pitch_head and not kappa_head:
            raise ValueError("pitch_head=True requires kappa_head=True (fixed 6 outputs)")
        self.kappa_head = kappa_head
        self.pitch_head = pitch_head
        self.input_norm = input_norm
        self.backbone = VitTiny()
        if pretrained:
            path = Path(ckpt_path) if ckpt_path else find_ckpt(VITT_CKPT_NAME)
            self.backbone.load_flat_ckpt(path, input_norm=input_norm)
        dim = self.backbone.embed_dim
        out_dim = 6 if pitch_head else (3 if kappa_head else 2)
        self.head = nn.Sequential(
            nn.Linear(2 * dim, head_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, out_dim))
        if kappa_head:
            with torch.no_grad():
                self.head[-1].bias[2] = 1.85
                if pitch_head:
                    self.head[-1].bias[5] = 1.85

    def _head_out(self, x: torch.Tensor) -> torch.Tensor:
        patch, cls = self.backbone(x)
        feat = torch.cat([cls, patch.mean(dim=(2, 3))], dim=1)
        return self.head(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._head_out(x)
        return F.normalize(out[:, :2].float(), dim=1, eps=1e-6)

    def forward_with_kappa(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self._head_out(x)
        unit = F.normalize(out[:, :2].float(), dim=1, eps=1e-6)
        if not self.kappa_head:
            return unit, torch.full_like(out[:, 0], 2.0).float()
        kappa = F.softplus(out[:, 2].float()).clamp(1e-3, 100.0)
        return unit, kappa

    def forward_full(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                     torch.Tensor, torch.Tensor]:
        if not self.pitch_head:
            raise RuntimeError("forward_full() is only for models with pitch_head=True")
        out = self._head_out(x)
        yaw_unit = F.normalize(out[:, :2].float(), dim=1, eps=1e-6)
        kappa_y = F.softplus(out[:, 2].float()).clamp(1e-3, 100.0)
        pitch_unit = F.normalize(out[:, 3:5].float(), dim=1, eps=1e-6)
        kappa_p = F.softplus(out[:, 5].float()).clamp(1e-3, 100.0)
        return yaw_unit, kappa_y, pitch_unit, kappa_p


if __name__ == "__main__":
    m = VittYawNet(kappa_head=True, pitch_head=True).eval()
    n = sum(p.numel() for p in m.parameters())
    print(f"params: {n:,}")
    for s in (64, 96, 128):
        with torch.no_grad():
            yu, ky, pu, kp = m.forward_full(torch.randn(2, 3, s, s))
        print(s, yu.shape, pu.shape, float(yu.norm(dim=1).mean()))
