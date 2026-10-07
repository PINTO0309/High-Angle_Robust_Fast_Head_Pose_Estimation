#!/usr/bin/env python3
"""PP-HGNetV2 B0 backbone (PyTorch implementation).

Module names and structure are matched to the weight keys so that
ckpts/PPHGNetV2_B0_stage1.pth (a PyTorch port of the PaddleClas PP-HGNetV2 B0
state_dict, use_lab=True, no classification head, 1.86M params) loads as is with
strict=True.

Architecture (B0):
  stem: stem1(3->16, 3x3 s2) -> [maxpool | stem2a->stem2b] concat
        -> stem3(32->16, 3x3 s2) -> stem4(16->16, 1x1)   ... stride 4
  stage0: HGV2 block x1 (16 -> mid16 -> 64,  k3, light=False)
  stage1: ds(dw s2) + block x1 (64 -> mid32 -> 256, k3, light=False)
  stage2: ds(dw s2) + block x2 (256/512 -> mid64 -> 512, k5, light=True)
  stage3: ds(dw s2) + block x1 (512 -> mid128 -> 1024, k5, light=True)

LAB (Learnable Affine Block) is attached only to ConvBNAct with activation
(not to conv1 (1x1, no act) nor to downsample).

out_stride can turn the downsample of the trailing stages into stride 1:
  32 (original) / 16 (stage3 at s1) / 8 (stage2,3 at s1).
For 64px input, out_stride=16 yields a 4x4 final feature map (=stride16).
Only the depthwise stride changes, so the pretrained weights can be used as is.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableAffineBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, stride: int = 1,
                 groups: int = 1, padding: int | None = None,
                 use_act: bool = True) -> None:
        super().__init__()
        if padding is None:
            padding = (kernel - 1) // 2
        self.conv = nn.Conv2d(cin, cout, kernel, stride, padding,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.use_act = use_act
        if use_act:
            self.lab = LearnableAffineBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        if self.use_act:
            x = self.lab(F.relu(x))
        return x


class LightConvBNAct(nn.Module):
    """1x1 (no act) -> depthwise kxk (with act)."""

    def __init__(self, cin: int, cout: int, kernel: int) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(cin, cout, 1, use_act=False)
        self.conv2 = ConvBNAct(cout, cout, kernel, groups=cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class StemBlock(nn.Module):
    def __init__(self, cin: int = 3, mid: int = 16, cout: int = 16) -> None:
        super().__init__()
        self.stem1 = ConvBNAct(cin, mid, 3, 2)
        self.stem2a = ConvBNAct(mid, mid // 2, 2, padding=0)
        self.stem2b = ConvBNAct(mid // 2, mid, 2, padding=0)
        self.stem3 = ConvBNAct(mid * 2, mid, 3, 2)
        self.stem4 = ConvBNAct(mid, cout, 1)
        self.pool = nn.MaxPool2d(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        return self.stem4(self.stem3(x))


class HGV2Block(nn.Module):
    def __init__(self, cin: int, mid: int, cout: int, kernel: int,
                 layer_num: int, light_block: bool, identity: bool) -> None:
        super().__init__()
        self.identity = identity
        layers: list[nn.Module] = []
        for i in range(layer_num):
            c = cin if i == 0 else mid
            layers.append(LightConvBNAct(c, mid, kernel) if light_block
                          else ConvBNAct(c, mid, kernel))
        self.layers = nn.ModuleList(layers)
        total = cin + mid * layer_num
        self.aggregation = nn.ModuleList([
            ConvBNAct(total, cout // 2, 1),
            ConvBNAct(cout // 2, cout, 1),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        outs = [x]
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        x = torch.cat(outs, dim=1)
        x = self.aggregation[1](self.aggregation[0](x))
        return x + identity if self.identity else x


class HGV2Stage(nn.Module):
    def __init__(self, cin: int, mid: int, cout: int, blocks: int,
                 downsample: bool, light_block: bool, kernel: int,
                 layer_num: int, stride: int = 2) -> None:
        super().__init__()
        if downsample:
            self.downsample = ConvBNAct(cin, cin, 3, stride, groups=cin,
                                        use_act=False)
        else:
            self.downsample = nn.Identity()
        self.blocks = nn.ModuleList([
            HGV2Block(cin if i == 0 else cout, mid, cout, kernel, layer_num,
                      light_block, identity=(i > 0))
            for i in range(blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


# B0: (in, mid, out, blocks, downsample, light_block, kernel, layer_num)
_B0_STAGES = [
    (16, 16, 64, 1, False, False, 3, 3),
    (64, 32, 256, 1, True, False, 3, 3),
    (256, 64, 512, 2, True, True, 5, 3),
    (512, 128, 1024, 1, True, True, 5, 3),
]


class PPHGNetV2B0(nn.Module):
    """Output: (N, 1024, S/out_stride, S/out_stride)."""

    out_channels = 1024

    def __init__(self, out_stride: int = 32) -> None:
        super().__init__()
        if out_stride not in (8, 16, 32):
            raise ValueError(f"out_stride must be 8/16/32: {out_stride}")
        # make the downsample of stage index 2, 3 stride 1 to keep resolution
        s1_from = {32: 99, 16: 3, 8: 2}[out_stride]
        self.stem = StemBlock()
        self.stages = nn.ModuleList([
            HGV2Stage(*cfg, stride=1 if i >= s1_from else 2)
            for i, cfg in enumerate(_B0_STAGES)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def load_pretrained(self, path: str | Path) -> None:
        sd = torch.load(str(path), map_location="cpu", weights_only=False)
        self.load_state_dict(sd, strict=True)
