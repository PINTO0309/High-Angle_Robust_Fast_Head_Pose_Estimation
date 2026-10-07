"""Build roll-branch models and restore checkpoints (training, distillation and export).

arch:
  - yawnet : YawNet (MBConv CNN, variable width). For deployment
  - vitt   : ViT-T/16 (center05 input, ImageNet initialization)
  - vitl   : DINOv3 ViT-L/16 (ceiling for the roll teacher. Dinov3RollNet)

All share the center05 input (x/127.5 - 1), and the forward() /
forward_with_kappa() convention is also the same (unit vector (cos, sin) and κ).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from yaw_dataset import NORMS
from yawnet import YawNet

ROLL_MODEL_TYPES = ("hfhpe_roll", "hyprnet_roll", "hypr_roll", "rollnet")


class Dinov3RollNet(nn.Module):
    """DINOv3 backbone + biternion roll head (center05 input).

    DINOv3 assumes ImageNet normalization and bf16, so the linear center05 →
    ImageNet transform and the bf16 autocast are done inside the model. Callers
    (RollDataset / evaluate / the teacher path of distillation) can treat it like
    the other archs.
    """

    def __init__(self, variant: str = "vitl16", kappa_head: bool = True,
                 pretrained: bool = True) -> None:
        super().__init__()
        from dinov3_yaw import Dinov3YawNet  # noqa: PLC0415
        self.net = Dinov3YawNet(variant, pretrained=pretrained,
                                kappa_head=kappa_head, pitch_head=False)
        self.variant = variant
        self.kappa_head = kappa_head
        mean, std = NORMS["imagenet"]
        self.register_buffer(
            "norm_scale", torch.tensor(0.5 / std, dtype=torch.float32)
            .view(1, 3, 1, 1), persistent=False)
        self.register_buffer(
            "norm_shift", torch.tensor((0.5 - mean) / std, dtype=torch.float32)
            .view(1, 3, 1, 1), persistent=False)

    def set_trainable_blocks(self, k: int | None) -> None:
        self.net.set_trainable_blocks(k)

    def backbone_parameters(self) -> list[nn.Parameter]:
        return self.net.backbone_parameters()

    def head_parameters(self) -> list[nn.Parameter]:
        return self.net.head_parameters()

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() * self.norm_scale + self.norm_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            return self.net(self._prep(x))

    def forward_with_kappa(self, x: torch.Tensor
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            return self.net.forward_with_kappa(self._prep(x))


def build_roll_model(arch: str, *, width: float = 0.5, kappa_head: bool = True,
                     variant: str = "vitl16",
                     pretrained: bool = True) -> nn.Module:
    if arch == "vitl":
        return Dinov3RollNet(variant, kappa_head=kappa_head,
                             pretrained=pretrained)
    if arch == "vitt":
        from vitt_yaw import VittYawNet  # noqa: PLC0415
        return VittYawNet(kappa_head=kappa_head, pitch_head=False,
                          input_norm="center05", pretrained=pretrained)
    if arch == "yawnet":
        return YawNet(width=width, kappa_head=kappa_head)
    raise ValueError(f"unknown roll arch: {arch}")


def resolve_ckpt(spec: str | Path) -> Path:
    path = Path(spec)
    if path.is_dir():
        cands = sorted(path.glob("best_*.pt"))
        if not cands:
            raise FileNotFoundError(f"no best_*.pt found in {path}")
        path = cands[0]
    return path


def load_roll_checkpoint(spec: str | Path
                         ) -> tuple[nn.Module, dict[str, Any], Path]:
    """Restore (model in eval mode, checkpoint dict, path) from a roll checkpoint."""
    path = resolve_ckpt(spec)
    ck: dict[str, Any] = torch.load(path, map_location="cpu",
                                    weights_only=False)
    if ck.get("model_type") not in ROLL_MODEL_TYPES:
        raise SystemExit(f"not a roll checkpoint: {path} "
                         f"(model_type={ck.get('model_type')})")
    a = ck["args"]
    arch = a.get("arch", "yawnet")
    sd = ck["model"]
    if arch == "vitl":
        kh = sd["net.head.3.weight"].shape[0] == 3
    elif arch == "vitt":
        kh = sd["head.3.weight"].shape[0] == 3
    else:
        kh = sd["fc.weight"].shape[0] == 3
    model = build_roll_model(arch, width=float(a.get("width", 0.5)),
                             kappa_head=kh,
                             variant=a.get("variant", "vitl16"),
                             pretrained=False)
    model.load_state_dict(sd)
    model.eval()
    return model, ck, path
