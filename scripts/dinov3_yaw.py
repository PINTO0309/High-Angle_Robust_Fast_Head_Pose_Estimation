"""Yaw teacher model: DINOv3 backbone + biternion head.

Follows the approach of HRFFA (High-Angle_Robust_Fast_FaceAlignment):
  - the official DINOv3 implementation is git-cloned into the torch.hub cache
    location and imported at runtime (for license reasons neither the code nor
    the weights are included in this repository; the teacher is not deployed either)
  - the weights are given directly as a local .pth (default: vitl16 in HRFFA/ckpts)
  - input normalization is ImageNet (required by the DINOv3 hub backbone)

Head: concat CLS token + patch mean → MLP → 2-dim → L2 normalize (biternion).
set_trainable_blocks(k) can restrict training to "the last k blocks + the head only"
(a measure to fit the AdamW optimizer state on the 8GB tier).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_DINOV3_GIT = "https://github.com/facebookresearch/dinov3"

# variant -> (hub function name, default ckpt file name, embedding dim)
DINOV3_VARIANTS: dict[str, tuple[str, str, int]] = {
    "vits16": ("dinov3_vits16", "dinov3_vits16_pretrain_lvd1689m-08c60483.pth", 384),
    "vitb16": ("dinov3_vitb16", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", 768),
    "vitl16": ("dinov3_vitl16", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", 1024),
}

# weight search order: ckpts/ at the repository root (gitignored, for carrying to
# other machines) → HRFFA's ckpts/ (local environment of this machine)
CKPT_SEARCH_DIRS = (
    Path(__file__).resolve().parent.parent / "ckpts",
    Path("/home/b920405/git/High-Angle_Robust_Fast_FaceAlignment/ckpts"),
)


def find_ckpt(ckpt_name: str) -> Path:
    for d in CKPT_SEARCH_DIRS:
        if (d / ckpt_name).exists():
            return d / ckpt_name
    raise FileNotFoundError(
        f"DINOv3 weights not found: {ckpt_name} (searched: "
        f"{', '.join(str(d) for d in CKPT_SEARCH_DIRS)}; "
        f"can be given explicitly with --dinov3-ckpt)")


def _ensure_hub_code() -> None:
    """Ensure the official DINOv3 code is in the torch.hub cache and on the import path."""
    target = Path(torch.hub.get_dir()) / "facebookresearch_dinov3_main"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", _DINOV3_GIT, str(target)],
                       check=True)
    if str(target) not in sys.path:
        sys.path.insert(0, str(target))


def head_flags(out_rows: int) -> tuple[bool, bool]:
    """Recover (kappa_head, pitch_head) from the row count of the head's last layer.

    2 = plain yaw / 3 = yaw + κ (since v5) / 6 = yaw + pitch + κ×2 (joint teacher).
    Structure recovery from a checkpoint (export / distillation / warm start)
    treats this function as the source of truth.
    """
    if out_rows == 2:
        return False, False
    if out_rows == 3:
        return True, False
    if out_rows == 6:
        return True, True
    raise ValueError(f"unknown head output row count: {out_rows} (only 2/3/6 supported)")


class Dinov3YawNet(nn.Module):
    """DINOv3 backbone + biternion yaw (+pitch) head. Output: (N, 2) unit vectors.

    With kappa_head=True (since v5) the last layer has 3 outputs, the third being
    the von Mises concentration κ (confidence). forward() still returns only the
    yaw unit vector (backward compatible); use forward_with_kappa() when κ is
    needed.

    With pitch_head=True (joint teacher) there are 6 outputs
    [cos_y, sin_y, κ_y, cos_p, sin_p, κ_p]. The first 3 rows share the layout of
    the 3-output version, so warm start from existing checkpoints (common-row
    transplant) and the yaw-side API (forward / forward_with_kappa) keep working
    as is. All outputs including pitch are obtained with forward_full().
    """

    def __init__(self, variant: str = "vitl16",
                 ckpt_path: str | Path | None = None,
                 head_hidden: int = 512, dropout: float = 0.1,
                 pretrained: bool = True,
                 kappa_head: bool = False,
                 pitch_head: bool = False) -> None:
        super().__init__()
        if pitch_head and not kappa_head:
            raise ValueError("pitch_head=True requires kappa_head=True (fixed 6 outputs)")
        hub_fn, ckpt_name, dim = DINOV3_VARIANTS[variant]
        self.variant = variant
        self.embed_dim = dim
        self.kappa_head = kappa_head
        self.pitch_head = pitch_head
        _ensure_hub_code()
        from dinov3.hub import backbones  # noqa: PLC0415
        if pretrained:
            path = Path(ckpt_path) if ckpt_path else find_ckpt(ckpt_name)
            if not path.exists():
                raise FileNotFoundError(f"DINOv3 weights not found: {path}")
            self.backbone = getattr(backbones, hub_fn)(
                pretrained=True, weights=str(path))
        else:  # when the distillation side loads a state_dict (build the structure only)
            self.backbone = getattr(backbones, hub_fn)(pretrained=False)
        out_dim = 6 if pitch_head else (3 if kappa_head else 2)
        self.head = nn.Sequential(
            nn.Linear(2 * dim, head_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, out_dim))
        if kappa_head:
            # softplus(1.85) ≈ 2.0 = same initial value as the former fixed kappa
            with torch.no_grad():
                self.head[-1].bias[2] = 1.85
                if pitch_head:
                    self.head[-1].bias[5] = 1.85

    def set_trainable_blocks(self, k: int | None) -> None:
        """Set the trainable backbone range: None=all blocks / 0=frozen / N=last N blocks.

        The head is always trained. norm (the final LayerNorm) is trained if k>0.
        """
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if k is None:
            for p in self.backbone.parameters():
                p.requires_grad_(True)
        elif k > 0:
            blocks = self.backbone.blocks
            for blk in blocks[-k:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            if hasattr(self.backbone, "norm") and self.backbone.norm is not None:
                for p in self.backbone.norm.parameters():
                    p.requires_grad_(True)
        for p in self.head.parameters():
            p.requires_grad_(True)

    def backbone_parameters(self) -> list[nn.Parameter]:
        return list(self.backbone.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def _head_out(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.get_intermediate_layers(
            x, n=1, reshape=True, return_class_token=True)
        patch, cls = feats[0]                       # (B,C,h,w), (B,C)
        feat = torch.cat([cls, patch.mean(dim=(2, 3))], dim=1)
        return self.head(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._head_out(x)
        return F.normalize(out[:, :2].float(), dim=1, eps=1e-6)

    def forward_with_kappa(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (unit vector, κ). κ is softplus-positive and clipped to [1e-3, 100]."""
        out = self._head_out(x)
        unit = F.normalize(out[:, :2].float(), dim=1, eps=1e-6)
        if not self.kappa_head:
            return unit, torch.full_like(out[:, 0], 2.0).float()
        kappa = F.softplus(out[:, 2].float()).clamp(1e-3, 100.0)
        return unit, kappa

    def forward_full(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                     torch.Tensor, torch.Tensor]:
        """Return (yaw unit vector, κ_y, pitch unit vector, κ_p) (6-output teacher only)."""
        if not self.pitch_head:
            raise RuntimeError("forward_full() is only for teachers with pitch_head=True")
        out = self._head_out(x)
        yaw_unit = F.normalize(out[:, :2].float(), dim=1, eps=1e-6)
        kappa_y = F.softplus(out[:, 2].float()).clamp(1e-3, 100.0)
        pitch_unit = F.normalize(out[:, 3:5].float(), dim=1, eps=1e-6)
        kappa_p = F.softplus(out[:, 5].float()).clamp(1e-3, 100.0)
        return yaw_unit, kappa_y, pitch_unit, kappa_p


if __name__ == "__main__":
    m = Dinov3YawNet("vitl16")
    n_all = sum(p.numel() for p in m.parameters())
    m.set_trainable_blocks(8)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"params total={n_all:,} trainable(last8+head)={n_train:,}")
    with torch.no_grad():
        y = m(torch.randn(1, 3, 320, 320))
    print("out:", y.shape, float(y.norm()))
