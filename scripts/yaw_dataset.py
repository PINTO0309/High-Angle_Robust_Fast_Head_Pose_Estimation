"""Dataset / sampler for yawpose (360-degree yaw, biternion targets).

Augmentation is consolidated in scripts/augment.py (yaw-only adaptation of HRFFA D4):
  - geometric: composed into a single 3x3 matrix and warped only once
    (small-angle roll / camera-rotation homography / flip / scale / translation.
     The resize is also part of the warp and standardized to INTER_LINEAR)
  - photometric / motion blur / random erase / low-res jitter (all ported from HRFFA)
At evaluation time only a deterministic INTER_LINEAR resize is applied.
"""
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from augment import (GeometricPolicy, apply_geometric_yp,
                     lowres_jitter, motion_blur, photometric, random_erase)

# Default is center05 normalization: ((x/255) - 0.5) / 0.5 → range [-1, 1].
# Only the DINOv3 teacher (train_teacher_dinov3.py / teacher view in distillation) uses
# ImageNet normalization (required by the DINOv3 hub backbone). The sixdrepnet / hrffa
# validation scripts likewise use ImageNet.
NORM_MEAN = 0.5
NORM_STD = 0.5
NORMS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "center05": (np.array([0.5, 0.5, 0.5], np.float32),
                 np.array([0.5, 0.5, 0.5], np.float32)),
    "imagenet": (np.array([0.485, 0.456, 0.406], np.float32),
                 np.array([0.229, 0.224, 0.225], np.float32)),
}


def normalize_image(im: np.ndarray, norm: str) -> torch.Tensor:
    mean, std = NORMS[norm]
    arr = (im.astype(np.float32) / 255.0 - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


class YawDataset(Dataset):
    """Dataset for yawpose. By default returns (image, yaw_target, yaw_deg).

    With with_pitch=True (for training the joint yaw+pitch teacher) it returns the 6-tuple
    (image, yaw_target, yaw_deg, pitch_target, pitch_deg, pitch_valid).
    Rows without pitch_deg (the generation intent), i.e. s001, get pitch_valid=0 and
    are masked on the loss side.
    """

    def __init__(self, root: str | Path, split: str, size: int, train: bool = False,
                 policy: GeometricPolicy | None = None,
                 motion_blur_prob: float = 0.2,
                 erase_prob: float = 0.25,
                 lowres_prob: float = 0.35,
                 input_norm: str = "center05",
                 with_pitch: bool = False,
                 seed: int = 0) -> None:
        self.root = Path(root)
        self.size = size
        self.train = train
        self.input_norm = input_norm
        self.policy = policy or GeometricPolicy(out_size=size)
        self.policy.out_size = size
        self.motion_blur_prob = motion_blur_prob
        self.erase_prob = erase_prob
        self.lowres_prob = lowres_prob
        self.with_pitch = with_pitch
        self.seed = seed
        # split="unified" merges train + val (intentionally accepting the data leak;
        # only for uses that do not hold out val, such as the final production training)
        files = ["train.jsonl", "val.jsonl"] if split == "unified" else [f"{split}.jsonl"]
        self.rows: list[dict[str, Any]] = []
        for name in files:
            with open(self.root / name) as f:
                self.rows += [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def bin_weights(self, bin_deg: int = 10, mode: str = "inv",
                    max_ratio: float = 20.0) -> torch.Tensor:
        """Per-sample weights based on yaw-bin frequency (for balanced sampling).

        mode:
          - "inv" : 1/count       — flatten bins completely (default; assumes the uniform
                                    full-surround distribution after the synthetic_004
                                    reinforcement. Rare bins are clipped by max_ratio to
                                    limit repetition)
          - "sqrt": 1/sqrt(count) — partial flattening
          - "none": equal weight for all samples (natural distribution as is)
        max_ratio: upper bound on the max/min weight ratio. Clipped to prevent repeating
        the same images through over-concentration on extremely rare bins (e.g. the
        back-of-the-head band).
        """
        if mode == "none":
            return torch.ones(len(self.rows), dtype=torch.double)
        if mode not in ("inv", "sqrt"):
            raise ValueError(f"unknown balance mode: {mode}")
        bins = [int(r["yaw_deg"] % 360.0 // bin_deg) for r in self.rows]
        counts = Counter(bins)
        exponent = 1.0 if mode == "inv" else 0.5
        w = np.array([1.0 / (counts[b] ** exponent) for b in bins])
        w = np.minimum(w, w.min() * max_ratio)
        return torch.from_numpy(w)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        r = self.rows[idx]
        bgr = cv2.imread(str(self.root / r["image"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(self.root / r["image"])
        im = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        yaw = float(r["yaw_deg"])
        raw_pitch = r.get("pitch_deg")
        pitch = float(raw_pitch) if raw_pitch is not None else 0.0

        if self.train:
            rng = np.random.default_rng(None)
            im, yaw, pitch = apply_geometric_yp(im, yaw, pitch,
                                                self.policy.sample(rng))
            if rng.random() < self.lowres_prob:
                im = lowres_jitter(im, rng)
            im = photometric(im, rng)
            if rng.random() < self.motion_blur_prob:
                im = motion_blur(im, rng)
            if rng.random() < self.erase_prob:
                im = random_erase(im, rng)
        else:
            im = cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        x = normalize_image(im, self.input_norm)
        rad = math.radians(yaw)
        target = torch.tensor([math.cos(rad), math.sin(rad)], dtype=torch.float32)
        if not self.with_pitch:
            return x, target, torch.tensor(yaw, dtype=torch.float32)
        prad = math.radians(pitch)
        pitch_target = torch.tensor([math.cos(prad), math.sin(prad)],
                                    dtype=torch.float32)
        return (x, target, torch.tensor(yaw, dtype=torch.float32),
                pitch_target, torch.tensor(pitch, dtype=torch.float32),
                torch.tensor(0.0 if raw_pitch is None else 1.0,
                             dtype=torch.float32))


class DistillYawDataset(YawDataset):
    """Same-condition pair Dataset for distillation.

    For each sample, the geometric transform + degradations (photometric / lowres_jitter /
    motion_blur / random_erase) are all applied once at teacher_size (default 320), and
    two views of identical content are returned:
      - teacher view: the teacher_size image after degradation
      - student view: the same image merely downscaled to size
    Teacher and student share exactly the same conditions except resolution (same geometric
    transform, same degradations), and the yaw label transform (sign flip on flip,
    camera-rotation correction) also matches exactly.

    Returns: (teacher_x, student_x, target_cos_sin, yaw_deg).
    With with_pitch=True (joint yaw+pitch distillation) it becomes a 7-tuple with
    (..., pitch_target, pitch_deg, pitch_valid) appended.
    """

    def __init__(self, root: str | Path, split: str, size: int,
                 teacher_size: int = 320, train: bool = False,
                 policy: GeometricPolicy | None = None,
                 motion_blur_prob: float = 0.2,
                 erase_prob: float = 0.25,
                 lowres_prob: float = 0.35,
                 teacher_norm: str = "center05",
                 input_norm: str = "center05",
                 with_pitch: bool = False,
                 seed: int = 0) -> None:
        super().__init__(root, split, size, train=train, policy=policy,
                         motion_blur_prob=motion_blur_prob,
                         erase_prob=erase_prob, lowres_prob=lowres_prob,
                         input_norm=input_norm, with_pitch=with_pitch, seed=seed)
        self.teacher_size = teacher_size
        self.teacher_norm = teacher_norm      # specify "imagenet" for the DINOv3 teacher
        self.policy.out_size = teacher_size   # one geometric transform at teacher resolution

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        r = self.rows[idx]
        bgr = cv2.imread(str(self.root / r["image"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(self.root / r["image"])
        im = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        yaw = float(r["yaw_deg"])
        raw_pitch = r.get("pitch_deg")
        pitch = float(raw_pitch) if raw_pitch is not None else 0.0

        if self.train:
            rng = np.random.default_rng(None)
            t_img, yaw, pitch = apply_geometric_yp(im, yaw, pitch,
                                                   self.policy.sample(rng))
            # degradations are applied once at teacher resolution; the student sees the
            # downscaled version (teacher and student share all conditions except resolution)
            if rng.random() < self.lowres_prob:
                t_img = lowres_jitter(t_img, rng)
            t_img = photometric(t_img, rng)
            if rng.random() < self.motion_blur_prob:
                t_img = motion_blur(t_img, rng)
            if rng.random() < self.erase_prob:
                t_img = random_erase(t_img, rng)
            s_img = cv2.resize(t_img, (self.size, self.size),
                               interpolation=cv2.INTER_LINEAR)
        else:
            t_img = cv2.resize(im, (self.teacher_size, self.teacher_size),
                               interpolation=cv2.INTER_LINEAR)
            s_img = cv2.resize(im, (self.size, self.size),
                               interpolation=cv2.INTER_LINEAR)

        rad = math.radians(yaw)
        target = torch.tensor([math.cos(rad), math.sin(rad)], dtype=torch.float32)
        base = (normalize_image(t_img, self.teacher_norm),
                normalize_image(s_img, self.input_norm),
                target, torch.tensor(yaw, dtype=torch.float32))
        if not self.with_pitch:
            return base
        prad = math.radians(pitch)
        pitch_target = torch.tensor([math.cos(prad), math.sin(prad)],
                                    dtype=torch.float32)
        return (*base, pitch_target, torch.tensor(pitch, dtype=torch.float32),
                torch.tensor(0.0 if raw_pitch is None else 1.0,
                             dtype=torch.float32))


def make_balanced_sampler(ds: YawDataset, bin_deg: int = 10, mode: str = "inv",
                          max_ratio: float = 20.0) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that balances yaw-bin frequencies.

    Random numbers come from torch's global generator (no dedicated generator is passed).
    Since the train_yawnet.py checkpoint saves/restores torch's global RNG state, this keeps
    the sampling sequence deterministic after --resume as well.
    """
    w = ds.bin_weights(bin_deg=bin_deg, mode=mode, max_ratio=max_ratio)
    return WeightedRandomSampler(w, num_samples=len(ds), replacement=True)


def angular_error_deg(pred_unit: torch.Tensor, yaw_deg: torch.Tensor) -> torch.Tensor:
    """Absolute angular error in degrees between predicted unit vector and gt yaw."""
    pred = torch.rad2deg(torch.atan2(pred_unit[:, 1], pred_unit[:, 0])) % 360.0
    diff = (pred - yaw_deg) % 360.0
    return torch.minimum(diff, 360.0 - diff)
