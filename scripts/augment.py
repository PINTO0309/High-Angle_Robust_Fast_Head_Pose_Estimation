"""Yaw-regression augmentation core (yaw-only adaptation of HRFFA D4 geometric augmentation).

Follows the design of HRFFA (High-Angle_Robust_Fast_FaceAlignment):
all geometric augmentations are composed into a single 3x3 homography T and the image is
warped only once (the resize is part of the warp, so the interpolation of the training
preprocessing is standardized to INTER_LINEAR).

Since yaw is the only label, the GT is updated as follows
(convention: image x right, y down, camera z forward; yaw follows the sixdrepnet360
convention and the synthetic data is assumed to have roll ≈ 0):
  - **Horizontal flip**: R' = M R M (M = diag(-1,1,1)) is equivalent to negating yaw / roll
      → yaw' = -yaw (exact).
  - **Camera azimuth rotation** (ψ): pure camera-rotation warp H = K Ry(ψ) K^{-1}.
      HRFFA updates the full rotation-matrix GT exactly as R' = Ry(ψ) R, but yaw in this
      task follows the appearance-based convention of the 6DRepNet family, and a pure camera
      rotation of an already cropped head does not change how the head looks (no newly
      visible surface), so yaw is almost invariant.
      Measurements (scripts/verify_cam_yaw_sign.py, sixdrepnet360, ψ=±15°, focal_ratio 1.2)
      show the reading moves by +0.166·ψ, so we correct as yaw' = yaw + CAM_YAW_COEF·ψ.
  - **Camera elevation rotation** (φ): R' = Rx(φ) R. The Euler-extracted yaw changes slightly
      where |yaw| is large, but |φ| is kept small and yaw is approximated as invariant.
  - **Small-angle roll** (in-plane rotation θ): R' = Rz(θ) R. Full-360 roll changes the
      Euler-extracted yaw substantially, so it is not used. For small |θ|, yaw is
      approximated as invariant.
  - Scale and translation do not change yaw.

photometric / motion blur / random erase are ported from the HRFFA data/dataset.py
implementation (all leave the yaw GT unchanged).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# Measured coefficient of the effect of a Ry(+ψ) warp on the appearance-based yaw
# (scripts/verify_cam_yaw_sign.py: mean_delta ≈ ±2.5° at ψ=±15° → 2.49/15)
CAM_YAW_COEF: float = 0.166

# Coefficient of the effect of a Rx(+φ) warp on the appearance-based pitch.
# Provisional value: symmetry assumption from the yaw measurement (CAM_YAW_COEF). Before
# pitch training starts in earnest, measure it with a pitch version of
# verify_cam_yaw_sign.py and update this value.
# (+φ in _rx sends the crop content downward = apparent looking-down = negative pitch, so
#  the sign is set negative. Both coefficient and sign must be verified when measured)
CAM_PITCH_COEF: float = -0.166


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


@dataclass
class GeometricParams:
    """Geometric augmentation parameters for one sample (already sampled values)."""
    out_size: int = 128
    roll_deg: float = 0.0        # small in-plane rotation (yaw approx. invariant)
    cam_pitch_deg: float = 0.0   # camera elevation (yaw approx. invariant)
    cam_yaw_deg: float = 0.0     # camera azimuth (yaw' = yaw + CAM_YAW_COEF * ψ)
    scale: float = 1.0
    tx: float = 0.0              # translation as a fraction of the output size
    ty: float = 0.0
    hflip: bool = False
    focal_ratio: float = 1.2


@dataclass
class GeometricPolicy:
    """Sampling ranges (training settings). Input is already a square crop with 5% margin."""
    out_size: int = 128
    roll_deg: float = 8.0
    roll_prob: float = 0.3
    cam_pitch_deg: float = 12.0
    cam_yaw_deg: float = 10.0
    cam_prob: float = 0.3
    scale_range: tuple[float, float] = (0.85, 1.1)
    translate: float = 0.05
    hflip_prob: float = 0.5
    focal_ratio: float = 1.2

    def sample(self, rng: np.random.Generator) -> GeometricParams:
        use_roll = rng.random() < self.roll_prob
        use_cam = rng.random() < self.cam_prob
        return GeometricParams(
            out_size=self.out_size,
            roll_deg=float(rng.uniform(-self.roll_deg, self.roll_deg)) if use_roll else 0.0,
            cam_pitch_deg=(float(rng.uniform(-self.cam_pitch_deg, self.cam_pitch_deg))
                           if use_cam else 0.0),
            cam_yaw_deg=(float(rng.uniform(-self.cam_yaw_deg, self.cam_yaw_deg))
                         if use_cam else 0.0),
            scale=float(rng.uniform(*self.scale_range)),
            tx=float(rng.uniform(-self.translate, self.translate)),
            ty=float(rng.uniform(-self.translate, self.translate)),
            hflip=bool(rng.random() < self.hflip_prob),
            focal_ratio=self.focal_ratio,
        )


def crop_affine(w: int, h: int, p: GeometricParams) -> np.ndarray:
    """Similarity transform (3x3) mapping the whole input image (= head crop with 5% margin)
    onto an out_size square.

    Rotation, scale and translation are also composed here, relative to the crop center.
    """
    cx, cy = w / 2, h / 2
    side = max(w, h)
    s = p.out_size / side * p.scale
    theta = math.radians(p.roll_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    half = p.out_size / 2
    T = np.array([
        [s * cos_t, -s * sin_t, 0.0],
        [s * sin_t, s * cos_t, 0.0],
        [0.0, 0.0, 1.0],
    ])
    T[0, 2] = half + p.tx * p.out_size - (T[0, 0] * cx + T[0, 1] * cy)
    T[1, 2] = half + p.ty * p.out_size - (T[1, 0] * cx + T[1, 1] * cy)
    return T


def camera_homography(p: GeometricParams) -> np.ndarray:
    """Homography H of a pure camera rotation (output crop coordinate system)."""
    phi = math.radians(p.cam_pitch_deg)
    psi = math.radians(p.cam_yaw_deg)
    R_cam = _ry(psi) @ _rx(phi)
    f = p.focal_ratio * p.out_size
    c = p.out_size / 2
    K = np.array([[f, 0, c], [0, f, c], [0, 0, 1.0]])
    return K @ R_cam @ np.linalg.inv(K)


def flip_matrix(out_size: int) -> np.ndarray:
    return np.array([[-1.0, 0.0, out_size - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def apply_geometric_yp(image: np.ndarray, yaw_deg: float, pitch_deg: float,
                       p: GeometricParams) -> tuple[np.ndarray, float, float]:
    """Apply geometric augmentation; return (output image, updated yaw_deg, updated pitch_deg).

    GT update for pitch (+ = looking up):
      - hflip / scale / translation / small-angle roll: pitch invariant
      - camera azimuth rotation ψ: pitch approx. invariant (symmetric to yaw being approx.
        invariant under cam_pitch)
      - camera elevation rotation φ: pitch' = pitch + CAM_PITCH_COEF · φ (provisional
        coefficient, needs measurement)
    The image is warped only once (shared with the yaw version).
    """
    h, w = image.shape[:2]
    T = crop_affine(w, h, p)
    yaw = yaw_deg
    pitch = pitch_deg

    if abs(p.cam_pitch_deg) > 1e-9 or abs(p.cam_yaw_deg) > 1e-9:
        H = camera_homography(p)
        T = H @ T
        # The camera rotation has a translation component of order f·tanψ, so put the crop
        # center back at the output center (a crop-window translation; no effect on yaw GT)
        m = T @ np.array([w / 2, h / 2, 1.0])
        mx, my = m[0] / m[2], m[1] / m[2]
        half = p.out_size / 2
        recenter = np.array([[1.0, 0.0, half + p.tx * p.out_size - mx],
                             [0.0, 1.0, half + p.ty * p.out_size - my],
                             [0.0, 0.0, 1.0]])
        T = recenter @ T
        yaw = yaw + CAM_YAW_COEF * p.cam_yaw_deg
        pitch = pitch + CAM_PITCH_COEF * p.cam_pitch_deg

    if p.hflip:
        T = flip_matrix(p.out_size) @ T
        yaw = -yaw  # pitch is invariant under horizontal flip

    out = cv2.warpPerspective(
        image, T, (p.out_size, p.out_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, yaw % 360.0, pitch


def apply_geometric(image: np.ndarray, yaw_deg: float,
                    p: GeometricParams) -> tuple[np.ndarray, float]:
    """Legacy yaw-only API: geometric augmentation returning (output image, updated yaw_deg)."""
    out, yaw, _ = apply_geometric_yp(image, yaw_deg, 0.0, p)
    return out, yaw


# ---------------------------------------------------------------------------
# photometric family (ported from HRFFA data/dataset.py; yaw GT unchanged)
# ---------------------------------------------------------------------------

def photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Lightweight photometric augmentation (RGB uint8 → RGB uint8)."""
    x = img.astype(np.float32)
    if rng.random() < 0.8:  # brightness, contrast, gamma
        x = x * rng.uniform(0.6, 1.4) + rng.uniform(-25, 25)
        x = 255.0 * (x.clip(0, 255) / 255.0) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.2:  # grayscale conversion
        g = cv2.cvtColor(x.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        x = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB).astype(np.float32)
    if rng.random() < 0.3:  # Gaussian noise
        x = x + rng.normal(0, rng.uniform(3, 12), x.shape)
    if rng.random() < 0.3:  # blur
        k = int(rng.choice([3, 5]))
        x = cv2.GaussianBlur(x, (k, k), 0)
    x = x.clip(0, 255).astype(np.uint8)
    if rng.random() < 0.3:  # JPEG degradation
        q = int(rng.integers(35, 85))
        _, enc = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, q])
        x = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return x


def motion_blur(img: np.ndarray, rng: np.random.Generator,
                max_frac: float = 0.06) -> np.ndarray:
    """Linear motion blur (length = 1% to max_frac of side, uniform angle). GT unchanged."""
    h, w = img.shape[:2]
    length = int(max(3, rng.uniform(0.01, max_frac) * max(h, w)))
    length += (length + 1) % 2
    k = np.zeros((length, length), np.float32)
    c = length // 2
    th = rng.uniform(0, np.pi)
    for t in np.linspace(-c, c, length * 4):
        x, y = int(round(c + t * np.cos(th))), int(round(c + t * np.sin(th)))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1.0
    k /= max(k.sum(), 1.0)
    return cv2.filter2D(img, -1, k)


def random_erase(img: np.ndarray, rng: np.random.Generator,
                 n_max: int = 2) -> np.ndarray:
    """Occlusion augmentation erasing rectangles with noise / mean color (yaw GT unchanged;
    intended to learn orientation estimation even under synthetic occlusion)."""
    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(int(rng.integers(1, n_max + 1))):
        ew = int(w * rng.uniform(0.10, 0.28))
        eh = int(h * rng.uniform(0.10, 0.28))
        x0 = int(rng.integers(0, max(w - ew, 1)))
        y0 = int(rng.integers(0, max(h - eh, 1)))
        if rng.random() < 0.5:
            out[y0:y0 + eh, x0:x0 + ew] = rng.integers(
                0, 256, (eh, ew, 3), dtype=np.uint8)
        else:
            out[y0:y0 + eh, x0:x0 + ew] = out[y0:y0 + eh, x0:x0 + ew].mean(
                axis=(0, 1), keepdims=True).astype(np.uint8)
    return out


def lowres_jitter(img: np.ndarray, rng: np.random.Generator,
                  min_side: int = 20) -> np.ndarray:
    """Shrink → enlarge to mimic a low-resolution (surveillance camera) domain. GT unchanged.
    Interpolation is standardized to INTER_LINEAR to match the training preprocessing policy."""
    h = img.shape[0]
    lo = int(rng.integers(min_side, max(min_side + 1, h)))
    small = cv2.resize(img, (lo, lo), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (h, h), interpolation=cv2.INTER_LINEAR)
