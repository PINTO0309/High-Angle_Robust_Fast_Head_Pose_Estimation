"""HFHPE roll branch training Dataset (exact GT from full-surround rotation augmentation).

roll convention: positive when the top of the head tilts counter-clockwise as seen
by the viewer in the image, taken mod 360 into [0, 360).
The roll of the generation intent is 0 for all sources, so **the applied image
rotation θ is directly the GT** (the natural head tilt of the source images, on the
order of ±15°, is assumed to be 0 and accepted as the label-noise floor. Initial
labeling of the frontal sources with sixd_roll is a possible future improvement).

train: θ ~ U[0,360) + hflip / scale / translation are composed into one affine and
warped once (INTER_LINEAR, BORDER_REPLICATE; the missing corners are filled by
replication, under the same conditions for training and inference). Photometric
augmentations reuse scripts/augment.py.
val: θ is assigned deterministically from the row index (uniform over the full
surround, reproducible).

Returns (image, target_cos_sin, roll_deg), the same form as YawDataset
(train_yawnet.evaluate / angular_error_deg can be reused as is).
"""
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from augment import lowres_jitter, motion_blur, photometric, random_erase
from yaw_dataset import normalize_image


def roll_warp(im: np.ndarray, theta_deg: float, scale: float, tx: float,
              ty: float, hflip: bool, out_size: int,
              content_frac: float = 1.0) -> np.ndarray:
    """One-pass warp: hflip → CCW θ rotation + scale → center on output + translation.

    content_frac < 1 is for "wide canvas" inputs: the central content_frac of the
    image (the crop contract) is enlarged to fill the whole output, and the corners
    needed by the rotation are supplied from the ring outside the crop contract
    (real pixels). This avoids BORDER_REPLICATE wedges and prevents the shortcut of
    reading the rotation angle from the wedges (for data with canvas factor f,
    pass content_frac = 1/f).
    """
    h, w = im.shape[:2]
    side = max(h, w)
    s = out_size / (side * content_frac) * scale
    th = math.radians(theta_deg)
    c, sn = math.cos(th), math.sin(th)
    # with y-down image coordinates, a "visually CCW" +θ rotation is [[c, s], [-s, c]]
    t = np.array([[s * c, s * sn, 0.0],
                  [-s * sn, s * c, 0.0],
                  [0.0, 0.0, 1.0]])
    if hflip:
        f = np.array([[-1.0, 0.0, w - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        t = t @ f
    m = t @ np.array([w / 2, h / 2, 1.0])
    half = out_size / 2
    t[0, 2] += half + tx * out_size - m[0]
    t[1, 2] += half + ty * out_size - m[1]
    return cv2.warpAffine(im, t[:2], (out_size, out_size),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class RollDataset(Dataset):
    """Roll regression Dataset. GT = applied rotation θ (biternion cos/sin target)."""

    def __init__(self, root: str | Path, split: str, size: int, train: bool = False,
                 motion_blur_prob: float = 0.2,
                 erase_prob: float = 0.25,
                 lowres_prob: float = 0.35,
                 input_norm: str = "center05",
                 seed: int = 42) -> None:
        self.root = Path(root)
        self.size = size
        self.train = train
        self.motion_blur_prob = motion_blur_prob
        self.erase_prob = erase_prob
        self.lowres_prob = lowres_prob
        self.input_norm = input_norm
        self.seed = seed
        files = ["train.jsonl", "val.jsonl"] if split == "unified" else [f"{split}.jsonl"]
        self.rows: list[dict[str, Any]] = []
        for name in files:
            with open(self.root / name) as f:
                self.rows += [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = self.rows[idx]
        bgr = cv2.imread(str(self.root / r["image"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(self.root / r["image"])
        im = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # for wide-canvas rows ("canvas": f) the central 1/f is the crop contract
        cf = 1.0 / float(r.get("canvas", 1.0))

        if self.train:
            rng = np.random.default_rng(None)
            theta = float(rng.uniform(0.0, 360.0))
            im = roll_warp(im, theta,
                           scale=float(rng.uniform(0.85, 1.1)),
                           tx=float(rng.uniform(-0.05, 0.05)),
                           ty=float(rng.uniform(-0.05, 0.05)),
                           hflip=bool(rng.random() < 0.5),
                           out_size=self.size, content_frac=cf)
            if rng.random() < self.lowres_prob:
                im = lowres_jitter(im, rng)
            im = photometric(im, rng)
            if rng.random() < self.motion_blur_prob:
                im = motion_blur(im, rng)
            if rng.random() < self.erase_prob:
                im = random_erase(im, rng)
        else:
            # deterministic rotation (fixed per row, uniform over the full surround);
            # plain rotation only, no augmentation
            rng = np.random.default_rng(self.seed * 1_000_003 + idx)
            theta = float(rng.uniform(0.0, 360.0))
            im = roll_warp(im, theta, scale=1.0, tx=0.0, ty=0.0,
                           hflip=False, out_size=self.size, content_frac=cf)

        x = normalize_image(im, self.input_norm)
        rad = math.radians(theta)
        target = torch.tensor([math.cos(rad), math.sin(rad)], dtype=torch.float32)
        return x, target, torch.tensor(theta, dtype=torch.float32)
