# HFHPE: High-Angle Robust Fast Head Pose Estimation

![GitHub](https://img.shields.io/github/license/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation?color=2BAF2B)

A training, distillation and ONNX deployment pipeline for **one-pass yaw / pitch / roll head-pose estimation that stays robust at extreme head poses and under heavy occlusion while remaining light enough for CPU inference**.

- **Head pose on whole-head crops.** HFHPE runs on head boxes produced by an object detector. The whole head is localized stably regardless of its orientation, so the pipeline keeps estimating through profiles, strong back-tilt, the back of the head, and heavy occlusion.
- **Full-surround output.** Yaw covers the full 0–360° circle (including the back of the head), pitch covers −120…+120° (beyond +90° = back-tilt), and roll covers the full 360° of in-plane rotation. Every angle is regressed as a biternion (unit `cos/sin`) with a von Mises concentration κ as a per-angle confidence.
- **One-pass integrated graph.** A tiny roll branch estimates the in-plane rotation, the crop is derotated inside the ONNX graph (GridSample), and the yaw/pitch body then runs on the upright crop — one input, one run, six trigonometric outputs plus three confidences.
- The teacher is a DINOv3 ViT-L/16 (320×320); the students are a ViT-T/16, a PP-HGNetV2-B0 based CNN and a from-scratch MBConv CNN (all 64×64 input). Students are trained by online distillation from the teacher and exported to verified ONNX graphs.
- Training data is **purely synthetic by gpt-image-2** for yaw/pitch (the SynthYawPitchPose dataset: 51,194 whole-head crops, 18.7k of them with large accessories such as masks, sunglasses and hats; every label was reviewed by eye and 34.8k of them were corrected by hand); the roll branch additionally adapts to real photographs whose roll ground truth is generated mechanically by rotation — no external pose label is ever used as a training target.
- **Roll is decoupled from yaw/pitch, by design.** Synthesizing a head at a prescribed yaw and pitch while keeping the composition intact is not difficult once the right generation conditions are met, but roll is hard to express, and a roll twist skews the directions along which yaw and pitch are estimated. Only yaw and pitch are therefore prescribed strictly: every yaw/pitch image is generated upright (roll assumed to be 0°), and roll is learned on its own from upright crops rotated in-plane by a known angle, which makes the roll ground truth exact.
- **Occlusion diversity is generated, not only erased.** On top of the strictly prescribed yaw and pitch, the dataset emphasizes the diversity of head occlusion by accessories that are likely to occur in the real world. It contains a large number of pose–accessory combinations that hardly exist in public datasets, so that occlusion robustness does not rest solely on simplistic, unrealistic preprocessing such as Random Erasing or Cutout.

## 1. Results at a glance

Validation MAAE (mean absolute angular error, degrees, lower is better) of the current generation (all four bodies retrained). Params and GMACs are counted on the exported ONNX graphs (`scripts/count_macs_onnx.py`; GFLOPs = 2 × GMACs). CPU ms = average latency of the batch-1 graph on an i9-10900K with onnxruntime 1.22 CPU EP, measured with [sit4onnx](https://github.com/PINTO0309/sit4onnx).

**Yaw / pitch body alone** (`<run>_kappa_*.onnx`, `hfhpe_dinov3_yawpitch_*.onnx` for the teacher; no roll, input assumed upright):

| Body | Backbone | Input | Params<br>(M) | GMACs<br>GFLOPs | CPU<br>(ms) | MAAE<br>yaw | <br>pitch |
|---|---|--:|--:|--:|--:|--:|--:|
| dinov3-320 | ViT-L/16 | 320x320 | 304.2 | 130.7<br>261.4 | 352 | 0.29 | 0.23 |
| vitt-064 | ViT-T/16 | 64x64 | 5.6 | 0.094<br>0.188 | 1.8 | 0.80 | 0.70 |
| hgnetv2-064 | PP-HGNetV2-B0 | 64x64 | 1.9 | 0.039<br>0.078 | 0.7 | 2.60 | 2.13 |
| yawnet-064 | MBConv | 64x64 | 0.76 | 0.013<br>0.026 | 0.7 | 5.64 | 7.55 |

**Integrated one-pass HFHPE** (`hfhpe_*.onnx` = roll branch + in-graph derotation + the body above; the shared roll branch adds 0.77M params):

| Model | Backbone | Input | Params<br>(M) | GMACs<br>GFLOPs | CPU<br>(ms) | MAAE<br>yaw | <br>pitch | <br>roll |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| hfhpe_dinov3 | ViT-L/16 | 320x320 | 305.1 | 130.7<br>261.4 | 354 | 0.29 | 0.23 | 3.29 |
| hfhpe_vitt | ViT-T/16 | 64x64 | 6.4 | 0.107<br>0.214 | 2.5 | 0.80 | 0.70 | 3.29 |
| hfhpe_hgnetv2 | PP-HGNetV2-B0 | 64x64 | 2.6 | 0.052<br>0.104 | 1.4 | 2.60 | 2.13 | 3.29 |
| hfhpe_yawnet | MBConv | 64x64 | 1.5 | 0.026<br>0.051 | 1.4 | 5.64 | 7.55 | 3.29 |

**How to read these numbers (important):** the models are trained with `--unified` (train + validation merged on purpose), so the yaw/pitch values above measure how well each model fits the training distribution and **must not be compared with published benchmark results**. They are selection metrics, useful for comparing the rows against each other. The yaw/pitch columns are the same in both tables because the roll branch does not change the body; the integrated table adds the roll estimate, the derotation and their cost. The roll column is the shared roll branch — a YawNet w1.0 (0.77M, 64×64) distilled online from a DINOv3 ViT-L/16 roll teacher trained at 320×320 — evaluated on **wedge-free canvas rotations** of the validation crops, an honest protocol in which the four corners are filled with real context pixels instead of border replication.

## 2. Features

- **SynthYawPitchPose dataset tooling**: JSONL label format (`data/yawpitchpose/{train,val}.jsonl` + 320px whole-head crops) covering seven synthetic generation sources, full-surround yaw, pitch −120…+120°, and large-accessory occlusion (masks, sunglasses, hats). Includes the audit tooling used to review every label by eye and correct 34.8k of them (mirror fixes, high-pitch frontal audit, per-cell visual sheets).
- **Angle regression that cannot wrap around**: every angle is a biternion (unit `cos/sin`) trained with a von Mises NLL; a κ head per angle outputs the concentration = confidence. No discontinuity at 0/360° and no gimbal handling in the loss.
- **Online teacher → student distillation** (`scripts/distill_yawnet.py`): the teacher (DINOv3 ViT-L/16 @320) is run on the fly on the same crops as the student (KD 0.7 / GT 0.3), with WSD learning-rate schedule, EMA, inverse-frequency yaw balancing and warm-start (`--init-student`) between generations.
- **Roll branch with label-free real-image adaptation** (`scripts/train_hfhpe_roll.py`): roll GT is the exact synthetic rotation θ applied at load time, so any upright crop is a training sample. Real photographs are re-cropped with the same DEIMv2 detector as deployment and mixed in via `--extra-data` — their pose labels are never used as targets. This removed the catastrophic ~180° roll failures on real photos. Supports ViT-T (`--arch vitt`) and DINOv3 ViT-L/16 (`--arch vitl`, 320 px) roll teachers, online KD (`--teacher`; with a higher-resolution teacher the student receives an area-downscaled copy of the same augmented view) and warm-start (`--init-student`). The deployed branch is a YawNet w1.0 distilled from the ViT-L teacher.
- **Wedge-free canvas rotation** (`scripts/build_roll_canvas_synth.py` + the `canvas` field of `roll_dataset.py`): rotating a square crop normally leaves border-replication wedges in the corners whose orientation leaks the applied angle (a shortcut the model learns to read). The canvas build stores a 1.8× context canvas around the identical crop contract; at train time the central contract region is rotated onto the full output while the corners are filled from the real surrounding ring. θ=0 reproduces the deployed crop exactly, so the runtime contract is unchanged.
- **One-pass HFHPE composition** (`scripts/export_hfhpe.py`): roll estimate → GridSample derotation → normalization conversion → yaw/pitch body, fused into a single ONNX graph.
- **κ-gated derotation (optional)** (`--roll-gate sigmoid|relu`): when the roll confidence κ_roll falls below a threshold τ, the in-graph derotation is blended toward the identity rotation, preventing a wrong large-angle derotation from corrupting the yaw/pitch body. Built from Sub/Mul/Sigmoid/Relu/Clip/Sqrt only — no Where/Greater/Cast — for edge-runtime compatibility. The `cos_sin` / `kappa` outputs always stay the raw estimates.
- **Verified ONNX export** (`scripts/export_onnx.py`): onnxslim → onnxsim → graph canonicalization → torch/onnxruntime parity → fixed batch-1 → N-batch conversion with batch 1/2/3 agreement checks → machine audit → I/O-contract metadata injection (readable in Netron).
- **Preview instruments**: 3×3 validation preview sheets (yaw ring + pitch dial) and 3-axis overlay previews for the integrated ONNX model.

## 3. Models and the ONNX I/O contract

| | Teacher dinov3-320 | Student vitt-064 | Student hgnetv2-064 | Student yawnet-064 |
|---|---|---|---|---|
| Backbone | DINOv3 ViT-L/16 (weights not vendored) | ViT-T/16 (own implementation, initialized from `ckpts/vitt_distill.pt`) | PP-HGNetV2-B0, final stage stride-1 (own implementation, initialized from `ckpts/PPHGNetV2_B0_stage1.pth`) | MBConv CNN (own implementation, from scratch) |
| Head | 6 outputs: yaw/pitch biternion + κ each | same | same | same |
| Input normalization | ImageNet mean/std | center05 `x/127.5 − 1` | center05 | center05 |
| Roll branch (shared) | YawNet w1.0, 0.77M params, center05, 64×64, distilled from a DINOv3 ViT-L/16 roll teacher (320×320) | | | |

Integrated HFHPE ONNX (`hfhpe_*.onnx`; the yaw/pitch-only exports `*_1x3xSxS.onnx` / `*_kappa_*.onnx` omit the roll fields). The `hfhpe_*_rollgate_*.onnx` variants embed the κ-gated derotation (τ=20, sigmoid) with the exact same I/O contract; the gate settings are also recorded in the ONNX metadata (`roll_gate`):

```
# RGB, center05 normalization x/127.5 - 1 (also for the dinov3 variant:
# the ImageNet conversion happens inside the graph). S = 64 (students) / 320 (dinov3)
input   images   float32 [N, 3, S, S]
# unit-vector pairs; angle_deg = degrees(atan2(sin, cos))
output  cos_sin  float32 [N, 6]   # [cos_yaw, sin_yaw, cos_pitch, sin_pitch, cos_roll, sin_roll]
# von Mises concentration = confidence per angle (~1e-3 ... 100)
output  kappa    float32 [N, 3]   # [kappa_yaw, kappa_pitch, kappa_roll]
```

Angle convention (also written into every ONNX file as metadata):

- **yaw**: 0° = facing the camera, +90° = turned to the viewer's left, 180° = back of the head, 270° = viewer's right (mod 360).
- **pitch**: positive = looking up, negative = seen from above; range about −120…+120°, values beyond +90° describe the back-tilt band.
- **roll**: positive = image counter-clockwise (top of the head moves toward the viewer's left), full 360°.

Crop contract (identical in training, evaluation and deployment): DEIMv2-Wholebody49 head box (class 7; score ≥ 0.30 when building the training and evaluation crops, ≥ 0.70 by default in the demo) → expand each side by 5% → square of the longer side, clamped at the image border → resize to S×S. Two files are written per export: fixed batch-1 `*_1x3xSxS.onnx` and `*_Nx3xSxS.onnx` with a symbolic batch axis, verified to agree at batch 1/2/3.

## 4. Setup

Prerequisites: Linux, Python 3.13 (pinned in `.python-version`), [uv](https://docs.astral.sh/uv/), and an NVIDIA GPU (torch built for CUDA 12.8; an 8 GB GPU is enough for inference and export, full training targets a 96 GB class GPU). Dependencies are pinned with `==` in `pyproject.toml` and locked in `uv.lock`.

```bash
git clone https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation.git
cd High-Angle_Robust_Fast_Head_Pose_Estimation
uv sync --frozen                                   # default: onnxruntime-gpu 1.22.0 (the validated toolchain)
uv sync --frozen --no-group ort --group tensorrt   # alternative: onnxruntime-gpu 1.26.0 (TensorRT EP, BF16)
# NOTE: a plain `uv run` re-syncs the DEFAULT groups; keep the same group flags on `uv run`
# or use `uv run --no-sync` after the manual sync.
```

### 4.1 Weights (`ckpts/`, `models/` — none of them are bundled)

| File | Purpose | Source / license |
|---|---|---|
| `ckpts/dinov3_vitl16_pretrain_lvd1689m-*.pth` | Teacher backbone | Distributed by Meta's [DINOv3](https://github.com/facebookresearch/dinov3) (DINOv3 License, no redistribution) |
| `ckpts/vitt_distill.pt` | Initial weights of the ViT-T/16 student | ImageNet-pretrained weights distributed with DEIMv2 (Apache-2.0) |
| `ckpts/PPHGNetV2_B0_stage1.pth` | Initial backbone weights of the CNN student | ImageNet-pretrained weights distributed with DEIMv2 (Apache-2.0) |
| `models/deimv2_wholebody49_boxes_only.onnx` | Head detection for dataset building, the crop contract and evaluation | [DEIMv2-Wholebody49](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/488_DEIMv2-Wholebody49) (Apache-2.0) |

### 4.2 Data (`data/`, not bundled)

- `data/yawpitchpose/` — the SynthYawPitchPose dataset: `labels_fixed.jsonl` / `train.jsonl` / `val.jsonl` plus 320px whole-head crops under `images/`. Fully synthetic (seven generation sources), not redistributed.
- Roll-adaptation crops (optional, `--extra-data` of the roll training): any directory of upright whole-head crops with a `manifest.jsonl` (`{"image": "images/<name>.jpg", ...}` per line). Real photo collections can be converted with the preprocessing tooling in `scripts/`; their pose labels are never used as training targets.

## 5. Training, evaluation and export

```bash
# Teacher (DINOv3 ViT-L/16 @320, yaw+pitch, WSD, EMA; ~7 h / 90 epochs on a 96 GB GPU)
uv run python scripts/train_teacher_dinov3.py \
--vram 96 --variant vitl16 --size 320 \
--pitch-head --unified \
--lr-schedule wsd --epochs 90 --decay-epochs 10 \
--ema-decay 0.999 --grad-clip 1.0 \
--data data/yawpitchpose --tag teacher

# Students (online distillation from the teacher; --init-student warm-starts from a previous run)
## ViT-T @64
uv run python scripts/distill_yawnet.py \
--teacher runs/dinov3_vitl16_320_yp_unified_teacher \
--student-arch vitt --student-size 64 --pitch-head --unified \
--vram 96 --lr-schedule wsd --epochs 75 --decay-epochs 20 --ema-decay 0.999 \
--data data/yawpitchpose --tag v1
## PP-HGNetV2-B0 CNN @64
uv run python scripts/distill_yawnet.py \
--teacher runs/dinov3_vitl16_320_yp_unified_teacher \
--student-arch hgnetv2 --student-size 64 --pitch-head --unified \
--vram 96 --lr 2e-4 --lr-backbone 2e-5 --grad-clip 1.0 \
--lr-schedule wsd --epochs 400 --decay-epochs 100 --ema-decay 0.999 \
--data data/yawpitchpose --tag v1
## MBConv CNN @64 (from scratch)
uv run python scripts/distill_yawnet.py \
--teacher runs/dinov3_vitl16_320_yp_unified_teacher \
--student-arch yawnet --student-size 64 --pitch-head --unified \
--vram 96 --lr-schedule wsd --epochs 285 --decay-epochs 20 --ema-decay 0.999 \
--data data/yawpitchpose --tag v1

# Roll branch, step 0: build the wedge-free 1.8x canvas dataset from the
# SynthYawPitchPose sources (theta=0 reproduces the deployed crop exactly)
uv run python scripts/build_roll_canvas_synth.py

# Roll branch, step 1: DINOv3 ViT-L/16 roll teacher at 320 px (roll GT = the
# synthetic rotation applied at load time; --extra-data mixes in the
# label-free real-image crops of section 4.2). --vram 8 fits an 8 GB GPU
# (last 8 blocks fine-tuned, micro-batch 8 x accumulation 8)
uv run python scripts/train_hfhpe_roll.py \
--arch vitl --size 320 --lr 2e-4 \
--unified --vram 96 \
--lr-schedule wsd --epochs 90 --decay-epochs 10 \
--ema-decay 0.999 --grad-clip 1.0 \
--data data/roll_canvas_synth \
--extra-data data/real_upright_crops \
--tag canvas

# Roll branch, step 2: deployed student (YawNet w1.0, 0.77M, 64 px) distilled
# online from the ViT-L teacher (alpha*KD + beta*GT). The augmented view is
# rendered once at the teacher resolution and area-downscaled for the student
uv run python scripts/train_hfhpe_roll.py \
--width 1.0 \
--teacher runs/hfhpe_roll_vitl_320_unified_canvas \
--unified --vram 96 \
--lr-schedule wsd --epochs 300 --decay-epochs 100 \
--ema-decay 0.999 --grad-clip 1.0 \
--data data/roll_canvas_synth \
--extra-data data/real_upright_crops \
--tag kd_vitl_canvas_w100

# ONNX: yaw/pitch body (fixed batch-1 + N-batch, parity/audit/metadata included)
uv run python scripts/export_onnx.py --ckpt runs/vitt_distill_064_yp_unified_v9

uv run python scripts/export_onnx.py --ckpt runs/vitt_distill_064_yp_unified_v9 \
--with-kappa \
--nbatch-atol 1e-3

# ONNX: integrated one-pass HFHPE (roll + derotation + body in one graph)
uv run python scripts/export_hfhpe.py \
--roll-ckpt runs/hfhpe_roll_064_unified_kd_vitl_canvas_w100 \
--body-ckpt runs/vitt_distill_064_yp_unified_v9 \
--kappa-tol 1e-2 \
--nbatch-atol 1e-3

# ONNX: same, with the kappa-gated derotation baked in (stem gains _rollgate)
uv run python scripts/export_hfhpe.py \
--roll-ckpt runs/hfhpe_roll_064_unified_kd_vitl_canvas_w100 \
--body-ckpt runs/vitt_distill_064_yp_unified_v9 \
--roll-gate sigmoid --roll-gate-tau 20 \
--kappa-tol 1e-2 \
--nbatch-atol 1e-3

# Validation preview sheets (3x3; deterministic per --set)
uv run python scripts/render_preview.py \
--ckpt runs/dinov3_vitl16_320_yp_unified_teacher --data data/yawpitchpose --set 1

# 3-axis overlay preview of the integrated ONNX (X=red right cheek, Y=green chin, Z=blue nose)
uv run python scripts/render_hfhpe_preview.py \
--onnx runs/vitt_distill_064_yp_unified_v9/hfhpe_vitt_1x3x64x64.onnx --set 1
```

Training logs are written to `runs/<run>/train_log.jsonl` (validation metrics per epoch); the best checkpoint is `runs/<run>/best_<val>.pt` and the resume checkpoint is `last.pt` (`--resume`).

Both checkpoints also store the full argument list (dataset roots, teacher / initial-weight paths, hyperparameters) and, for distilled students, the teacher's path and metrics. Before sharing a checkpoint, run `scripts/sanitize_checkpoint.py runs/<run>` to write a copy under `runs/public/<run>/` that keeps only the weights and the keys the loaders need; `export_onnx.py --ckpt runs/public/<run>` then produces identically named ONNX files.

## 6. ONNX demo

Two-stage inference demo (adapted from the [HRFFA](https://github.com/PINTO0309/High-Angle_Robust_Fast_FaceAlignment) demo; no face alignment or YawNet stage): DEIMv2-Wholebody49 head detection → integrated HFHPE ONNX on every head crop. It draws the head boxes, a 3-axis pose overlay per head (X = red right cheek, Y = green chin, Z = blue nose), the per-head yaw/pitch/roll text, and a top-right yaw ring for the largest head (0 = facing the camera, 90 = viewer's left).

```bash
# images
uv run python demo/demo_hfhpe_onnx.py -i images_dir -o output_dir
# video file / camera 0
uv run python demo/demo_hfhpe_onnx.py -v input.mp4 -o output_dir
uv run python demo/demo_hfhpe_onnx.py -v 0 -o output_dir
# TensorRT EP (engine cache under models/trt_cache)
uv run --no-group ort --group tensorrt python demo/demo_hfhpe_onnx.py -v 0 -o output_dir -d tensorrt
# swap the pose model
uv run python demo/demo_hfhpe_onnx.py -pm models/hfhpe_hgnetv2_Nx3x64x64.onnx -v 0 -o output_dir
# N-batch graph of the default model (all heads of a frame in one pass)
uv run python demo/demo_hfhpe_onnx.py -pm models/hfhpe_vitt_rollgate_Nx3x64x64.onnx -v 0 -o output_dir
# without the kappa gate (same I/O; always derotates by the estimated roll)
uv run python demo/demo_hfhpe_onnx.py -pm models/hfhpe_vitt_Nx3x64x64.onnx -v 0 -o output_dir
```

Defaults: detector `models/deimv2_dinov3_s_wholebody49_boxes_only.onnx`, pose `models/hfhpe_vitt_rollgate_1x3x64x64.onnx` (the κ-gated ViT-T model, fixed batch-1 graph: heads are processed one by one; N-batch graphs are also accepted and run all heads of a frame in one pass). The crop contract of section 3 (5% per-side expansion, long-side square clamped at the border, resize to S×S) is applied as-is to every head with score ≥ 0.70 (`--head_score_threshold`). Input normalization is detected per model: the pose model from its ONNX metadata (center05, or ImageNet for the dinov3 yaw/pitch-only graph), the detector from its graph (ImageNet mean/std for DINOv3-based detectors, plain /255 for HGNetV2-based ones; `--detector_norm div255|imagenet` overrides it — the training crops were built with `div255`). Keys on video / camera input: ESC = quit, `b` = head boxes, `a` = 3-axis overlay, `t` = angle text, `r` = yaw ring. `--save_raw_predictions` writes one JSON per image / frame (bbox, score, yaw/pitch/roll, κ); `--yaw_smooth_tau 0.15` smooths the ring display with a circular EMA (display only).

## 7. Repository layout

```
scripts/
  # models
  dinov3_yaw.py vitt_yaw.py hgnetv2_yaw.py pphgnetv2.py yawnet.py hfhpe.py
  # data pipeline
  build_yawpose_dataset.py build_yawpitchpose_dataset.py yaw_dataset.py
  build_roll_canvas_synth.py roll_dataset.py augment.py
  # training / distillation
  train_teacher_dinov3.py distill_yawnet.py train_yawnet.py train_hfhpe_roll.py
  ema.py vram_presets.py
  # export (onnxslim/onnxsim -> parity -> N-batch -> audit -> metadata)
  export_onnx.py export_hfhpe.py add_onnx_metadata.py count_macs_onnx.py
  sanitize_checkpoint.py       distribution copy of a checkpoint without the training-environment details
  # evaluation / previews
  eval_*.py val_preview.py render_preview.py render_hfhpe_preview.py
  render_roll_preview.py render_hfhpe_roll_preview.py plot_*.py
  # label auditing (mirror fixes, screening, verification)
  fix_*.py screen_*.py verify_*.py check_mirror_fix.py relabel_rear_teacher.py
demo/
  demo_hfhpe_onnx.py           two-stage ONNX demo (head detection -> yaw/pitch/roll)
ckpts/ models/ data/ runs/   weights, detector, datasets and training outputs (not tracked by git)
```

## 8. License

- Code: [MIT License](LICENSE) (Copyright (c) 2026 Katsuya Hyodo).
- Weights: DINOv3 (Meta, DINOv3 License), vitt_distill and PP-HGNetV2 (DEIMv2, Apache-2.0) are neither bundled nor redistributed. Check the derived-work terms of these sources before distributing trained HFHPE weights or ONNX files.

## 9. Citation and acknowledgements

### 9.1 Citing this repository

```bibtex
@software{hyodo2026hfhpe,
  author = {Katsuya Hyodo},
  title  = {{HFHPE}: High-Angle Robust Fast Head Pose Estimation},
  url    = {https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation},
  year   = {2026},
}
```

### 9.2 Backbones, pretrained weights and tools

**DINOv3** — teacher backbone (ViT-L/16); weights distributed under the DINOv3 License. Paper: https://arxiv.org/abs/2508.10104 · Code: https://github.com/facebookresearch/dinov3

```bibtex
@misc{simeoni2025dinov3,
  title         = {{DINOv3}},
  author        = {Sim{\'e}oni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{\"e}l and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and J{\'e}gou, Herv{\'e} and Labatut, Patrick and Bojanowski, Piotr},
  year          = {2025},
  eprint        = {2508.10104},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2508.10104}
}
```

**DEIMv2** — source of the ImageNet-pretrained PP-HGNetV2-B0 and ViT-T weights used by the students (Apache-2.0). Paper: https://arxiv.org/abs/2509.20787 · Code: https://github.com/Intellindust-AI-Lab/DEIMv2

```bibtex
@article{huang2025deimv2,
  title   = {Real-Time Object Detection Meets {DINOv3}},
  author  = {Huang, Shihua and Hou, Yongjie and Liu, Longfei and Yu, Xuanlong and Shen, Xi},
  journal = {arXiv preprint arXiv:2509.20787},
  year    = {2025},
  url     = {https://arxiv.org/abs/2509.20787}
}
```

**DEIMv2-Wholebody49** — whole-head detection for dataset building, the crop contract and the evaluation harness (Apache-2.0). https://github.com/PINTO0309/PINTO_model_zoo/tree/main/488_DEIMv2-Wholebody49

```bibtex
@software{DEIMv2-Wholebody49,
  author = {Katsuya Hyodo},
  title  = {Unified multi-task model for detection, pose estimation, and instance segmentation. 49 classes.},
  url    = {https://github.com/PINTO0309/PINTO_model_zoo/tree/main/488_DEIMv2-Wholebody49},
  year   = {2026},
  month  = {05},
  doi    = {10.5281/zenodo.10229410}
}
```

**6DRepNet360** — reference head-pose model used by the dataset audit tooling, not by the deployed graphs (code MIT; weights not bundled). The yaw convention of SynthYawPitchPose (0° = facing the camera, +90° = viewer's left) follows its output convention, and its ONNX export (`models/sixdrepnet360_1x3x224x224_full.onnx`, PINTO_model_zoo #423) was used to check the sign of the synthetic yaw labels (`verify_labels_sixd.py`, `fix_labels.py`) and to measure the camera-yaw augmentation coefficient (`verify_cam_yaw_sign.py`). Paper: https://doi.org/10.1109/TIP.2024.3378180 · Code: https://github.com/thohemp/6DRepNet360 · ONNX: https://github.com/PINTO0309/PINTO_model_zoo/tree/main/423_6DRepNet360

```bibtex
@article{hempel2024toward,
  author  = {Hempel, Thorsten and Abdelrahman, Ahmed A. and Al-Hamadi, Ayoub},
  title   = {Toward Robust and Unconstrained Full Range of Rotation Head Pose Estimation},
  journal = {IEEE Transactions on Image Processing},
  year    = {2024},
  volume  = {33},
  pages   = {2377--2387},
  doi     = {10.1109/TIP.2024.3378180}
}
```

**HRFFA** — the author's whole-head face-alignment project (MIT). HFHPE reuses its single-warp geometric augmentation design (`augment.py`), its export procedure (simplify → parity → N-batch → audit) and its two-stage ONNX demo as the base of `demo/demo_hfhpe_onnx.py`; the HRFFA ViT-L 68-landmark model (`hrffa_vitl_ibug68`) also served as a second reference for the sign of large-yaw labels during dataset auditing (`study_landmark_yaw.py`, `fix_labels.py`). Code: https://github.com/PINTO0309/High-Angle_Robust_Fast_FaceAlignment

```bibtex
@software{hyodo2026hrffa,
  author    = {Katsuya Hyodo},
  title     = {HRFFA: High-Angle Robust Fast FaceAlignment},
  month     = {aug},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22161811},
  url       = {https://doi.org/10.5281/zenodo.22161811},
}
```

### 9.3 Methods referenced (reimplemented from the papers; no code was copied)

**BiternionNet** — the biternion output representation (unit `cos/sin` regression) and the von Mises loss used for every angle. The architectures are our own. Paper: https://lucasb.eyer.be/academic/biternions/biternions_gcpr15.pdf · Original implementation (MIT): https://github.com/lucasb-eyer/BiternionNet

```bibtex
@inproceedings{beyer2015biternion,
  title     = {Biternion Nets: Continuous Head Pose Regression from Discrete Training Labels},
  author    = {Beyer, Lucas and Hermans, Alexander and Leibe, Bastian},
  booktitle = {German Conference on Pattern Recognition (GCPR)},
  year      = {2015},
  url       = {https://lucasb.eyer.be/academic/biternions/biternions_gcpr15.pdf}
}
```

### 9.4 Acknowledgements

- Meta AI for DINOv3 and the DEIMv2 authors for the ViT-T and PP-HGNetV2 weights (originally from PaddlePaddle's PaddleClas / PaddleDetection).
- The SynthYawPitchPose generation pools were produced with image-generation APIs; the synthetic data is not redistributed.
- The ONNX post-processing know-how (batch-axis preservation, N-batch derivation, parity-first verification) builds on the author's PersonViT export pipeline and the [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo) conventions.
