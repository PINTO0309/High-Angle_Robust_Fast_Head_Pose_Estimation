"""Training presets per VRAM tier.

Design policy:
  - The effective batch (micro_batch × grad_accum) is the same across all tiers
    for a given task. This keeps the training dynamics (lr schedule, gradient
    noise) independent of the machine and makes --resume across tiers safe (the
    resume consistency check uses the effective batch).
  - Tiers with less VRAM lower micro_batch and compensate with grad_accum.
  - Only the 8GB tier of dinov3_teacher unfreezes just the later blocks, because
    the optimizer state (about 12 byte/param for all parameters with AdamW) does
    not fit.

Tasks:
  - yawnet_small : train_yawnet.py with size <= 128 (effective 256)
  - yawnet_320   : train_yawnet.py with size >= 192 (for the teacher, effective 128)
  - dinov3_teacher : train_teacher_dinov3.py (ViT-L 320px, effective 64)
  - distill      : distill_yawnet.py (effective 128)
  - roll_kd_vitl : train_hfhpe_roll.py when distilling from the ViT-L (320px) teacher
                   (the student's effective 256 is the same as yawnet_small; only
                   the teacher forward is split into chunks of teacher_chunk
                   images. The teacher runs in eval / no_grad and has no
                   batch-dependent layers, so splitting does not change outputs)
"""
from __future__ import annotations

from dataclasses import dataclass

VRAM_TIERS = (8, 16, 96)


@dataclass(frozen=True)
class VramPreset:
    micro_batch: int
    grad_accum: int
    amp_dtype: str = "fp16"          # "fp16" | "bf16"
    trainable_blocks: int | None = None  # dinov3 only: None=all blocks, N=last N blocks
    workers: int = 16
    teacher_chunk: int = 0           # distill: teacher forward chunk (0 = no split)

    @property
    def effective_batch(self) -> int:
        return self.micro_batch * self.grad_accum


PRESETS: dict[tuple[str, int], VramPreset] = {
    # train_yawnet.py, size <= 128 (effective 256)
    ("yawnet_small", 8):  VramPreset(256, 1),
    ("yawnet_small", 16): VramPreset(256, 1),
    ("yawnet_small", 96): VramPreset(256, 1),
    # train_yawnet.py, size >= 192 (effective 128)
    ("yawnet_320", 8):  VramPreset(32, 4),
    ("yawnet_320", 16): VramPreset(64, 2),
    ("yawnet_320", 96): VramPreset(128, 1),
    # train_teacher_dinov3.py (ViT-L 320px, effective 64, bf16)
    ("dinov3_teacher", 8):  VramPreset(8, 8, amp_dtype="bf16", trainable_blocks=8),
    ("dinov3_teacher", 16): VramPreset(16, 4, amp_dtype="bf16", trainable_blocks=None),
    ("dinov3_teacher", 96): VramPreset(64, 1, amp_dtype="bf16", trainable_blocks=None),
    # distill_yawnet.py (effective 128; teacher forward is no_grad and fairly light)
    ("distill", 8):  VramPreset(64, 2),
    ("distill", 16): VramPreset(128, 1),
    ("distill", 96): VramPreset(128, 1),
    # train_hfhpe_roll.py, distillation from ViT-L teacher (effective 256. Measured
    # on 8GB: teacher forward 5.5GB for all 256 images at once, 1.7GB in chunks of 32)
    ("roll_kd_vitl", 8):  VramPreset(256, 1, teacher_chunk=32),
    ("roll_kd_vitl", 16): VramPreset(256, 1, teacher_chunk=64),
    ("roll_kd_vitl", 96): VramPreset(256, 1, teacher_chunk=0),
}


def get_preset(task: str, vram: int) -> VramPreset:
    key = (task, vram)
    if key not in PRESETS:
        raise KeyError(f"unknown preset: task={task} vram={vram} "
                       f"(vram must be one of {VRAM_TIERS})")
    return PRESETS[key]
