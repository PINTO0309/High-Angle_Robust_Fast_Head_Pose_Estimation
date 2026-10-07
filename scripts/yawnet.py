"""YawNet: lightweight biternion yaw regressor (cos/sin output, L2-normalized).

Fully convolutional MBConv backbone with SE and SiLU, global average pooling,
so the same architecture serves 64x64 / 96x96 / 128x128 inputs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):
    def __init__(self, ch: int, r: int = 4) -> None:
        super().__init__()
        hidden = max(8, ch // r)
        self.fc1 = nn.Conv2d(ch, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.silu(self.fc1(s))
        return x * torch.sigmoid(self.fc2(s))


class MBConv(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, expand: int = 3,
                 use_se: bool = False) -> None:
        super().__init__()
        mid = cin * expand
        self.use_res = stride == 1 and cin == cout
        layers: list[nn.Module] = []
        if expand != 1:
            layers += [nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.SiLU(inplace=True)]
        layers += [
            nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False),
            nn.BatchNorm2d(mid), nn.SiLU(inplace=True),
        ]
        if use_se:
            layers.append(SE(mid))
        layers += [nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout)]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return x + out if self.use_res else out


class YawNet(nn.Module):
    """Output: (N, 2) unit vector (cos(yaw), sin(yaw)).

    With kappa_head=True the final FC has 3 outputs and the 3rd is the von Mises
    concentration κ (confidence). forward() still returns only the unit vector
    (backward compatible); use forward_with_kappa() when κ is needed (same
    convention as the teacher Dinov3YawNet).

    With pitch_head=True (yaw+pitch integrated student) there are 6 outputs
    [cos_y, sin_y, κ_y, cos_p, sin_p, κ_p]. The first 3 rows share the layout of the
    3-output version, so the yaw-side API is unchanged; all outputs are obtained via
    forward_full() (same convention as the teacher).
    """

    def __init__(self, width: float = 1.0, dropout: float = 0.2,
                 kappa_head: bool = False, pitch_head: bool = False) -> None:
        super().__init__()
        if pitch_head and not kappa_head:
            raise ValueError("pitch_head=True requires kappa_head=True (fixed 6 outputs)")
        self.kappa_head = kappa_head
        self.pitch_head = pitch_head
        def c(ch: int) -> int:
            return max(8, int(round(ch * width / 8)) * 8)

        self.stem = nn.Sequential(
            nn.Conv2d(3, c(24), 3, 2, 1, bias=False), nn.BatchNorm2d(c(24)), nn.SiLU(inplace=True))
        self.stages = nn.Sequential(
            MBConv(c(24), c(32), stride=2, expand=2),
            MBConv(c(32), c(32), stride=1, expand=2),
            MBConv(c(32), c(64), stride=2, expand=3),
            MBConv(c(64), c(64), stride=1, expand=3),
            MBConv(c(64), c(64), stride=1, expand=3),
            MBConv(c(64), c(96), stride=2, expand=3, use_se=True),
            MBConv(c(96), c(96), stride=1, expand=3, use_se=True),
            MBConv(c(96), c(96), stride=1, expand=3, use_se=True),
            MBConv(c(96), c(160), stride=2, expand=3, use_se=True),
            MBConv(c(160), c(160), stride=1, expand=3, use_se=True),
        )
        self.head_conv = nn.Sequential(
            nn.Conv2d(c(160), c(256), 1, bias=False), nn.BatchNorm2d(c(256)), nn.SiLU(inplace=True))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c(256), 6 if pitch_head else (3 if kappa_head else 2))
        if kappa_head:
            # softplus(1.85) ≈ 2.0: start from the same initial value as the old fixed kappa
            with torch.no_grad():
                self.fc.bias[2] = 1.85
                if pitch_head:
                    self.fc.bias[5] = 1.85

    def _head_out(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.head_conv(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._head_out(x)
        if self.kappa_head:
            out = out[:, :2]
        return F.normalize(out, dim=1, eps=1e-6)

    def forward_with_kappa(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (unit vector, κ). κ is positive via softplus, clipped to [1e-3, 100]."""
        out = self._head_out(x)
        unit = F.normalize(out[:, :2], dim=1, eps=1e-6)
        if not self.kappa_head:
            return unit, torch.full_like(out[:, 0], 2.0)
        kappa = F.softplus(out[:, 2]).clamp(1e-3, 100.0)
        return unit, kappa

    def forward_full(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                     torch.Tensor, torch.Tensor]:
        """Return (yaw unit vector, κ_y, pitch unit vector, κ_p) (6-output models only)."""
        if not self.pitch_head:
            raise RuntimeError("forward_full() is only for pitch_head=True students")
        out = self._head_out(x)
        yaw_unit = F.normalize(out[:, :2], dim=1, eps=1e-6)
        kappa_y = F.softplus(out[:, 2]).clamp(1e-3, 100.0)
        pitch_unit = F.normalize(out[:, 3:5], dim=1, eps=1e-6)
        kappa_p = F.softplus(out[:, 5]).clamp(1e-3, 100.0)
        return yaw_unit, kappa_y, pitch_unit, kappa_p


def von_mises_loss(pred_unit: torch.Tensor, target_cos_sin: torch.Tensor,
                   kappa: float = 2.0) -> torch.Tensor:
    """Beyer et al. biternion von Mises loss: 1 - exp(kappa*(cos(delta)-1))."""
    cos_delta = (pred_unit * target_cos_sin).sum(dim=1)
    return (1.0 - torch.exp(kappa * (cos_delta - 1.0))).mean()


def von_mises_nll(pred_unit: torch.Tensor, kappa: torch.Tensor,
                  target_cos_sin: torch.Tensor) -> torch.Tensor:
    """von Mises negative log-likelihood with learned κ (constant term log 2π omitted).

    NLL = -κ cosΔ + log I0(κ) = κ(1 - cosΔ) + log I0e(κ)
    (I0e is the exponentially scaled Bessel function. Hard samples get a lower κ and
    are automatically down-weighted)

    Note: since this is the NLL of a continuous distribution, **the loss becomes
    negative** as predictions get accurate and κ rises (the density sharpens). This
    is normal behavior, not a bug (same as a Gaussian NLL going negative as σ
    shrinks). Its definition differs from the fixed-κ von_mises_loss, so the values
    cannot be compared directly; compare quality with maae instead.
    """
    return von_mises_nll_per_sample(pred_unit, kappa, target_cos_sin).mean()


def von_mises_nll_per_sample(pred_unit: torch.Tensor, kappa: torch.Tensor,
                             target_cos_sin: torch.Tensor) -> torch.Tensor:
    """Per-sample values (N,) of von_mises_nll. Decomposed version so the caller can
    take a masked mean (e.g. excluding weakly labeled pitch rows)."""
    cos_delta = (pred_unit * target_cos_sin).sum(dim=1)
    return kappa * (1.0 - cos_delta) + torch.log(torch.special.i0e(kappa))


if __name__ == "__main__":
    m = YawNet()
    n = sum(p.numel() for p in m.parameters())
    print(f"params: {n:,}")
    for s in (64, 96, 128):
        y = m(torch.randn(2, 3, s, s))
        print(s, y.shape, y.norm(dim=1))
