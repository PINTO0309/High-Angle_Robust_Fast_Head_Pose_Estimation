"""HFHPE ONNX inference demo (head detection → yaw/pitch/roll estimation on head crops → overlay).

Two-stage layout based on demo/demo_hrffa_onnx.py in the HRFFA repository; both models can be
swapped via command-line arguments (the Face Alignment / YawNet stages are not used):
  1. DEIMv2-Wholebody49 (ONNX): resize the image directly to 640×640 (BGR→RGB, /255, no
     normalization) and get `label_xyxy_score` [1, Q, 6] = (class, x1, y1, x2, y2, score).
     Coordinates are normalized to [0,1]. Only the class 7 = head boxes are used (default
     score ≥ 0.70; the training-data crops were built with ≥ 0.30).
  2. HFHPE (integrated ONNX): crop with the crop contract shared by training, evaluation and
     deployment (expand the head bbox by 5% per side → long-side square, clamp at the image
     border → resize to S×S), normalize as RGB, center05 (`x/127.5 − 1`) and get `cos_sin` [N, 6]
     and `kappa` [N, 3]. Angles are recovered as angle_deg = degrees(atan2(sin, cos)):
       - yaw:   0 = facing the camera, +90 = viewer's left, 180 = back of the head,
                270 = viewer's right (mod 360)
       - pitch: positive = looking up, negative = looking down (about −120…+120; beyond +90 is
                the backward-tilt band)
       - roll:  positive = image counter-clockwise (full 360)
     κ is the von Mises concentration (per-angle confidence). Models with an N batch axis infer
     all heads of a frame in one call. Standalone yaw/pitch-only graphs (cos_sin [N, 4]) are also
     accepted; roll is then None (0 in the axis overlay). The input normalization is detected
     automatically from the ONNX metadata (`input.images`): ImageNet normalization if mean/std
     is recorded (standalone dinov3 yaw/pitch graphs), otherwise center05.

Only predictions are drawn: head bbox (green), the 3 axes at the head center (X=red right cheek /
Y=green chin / Z=blue nose tip), the angle text above the bbox, and a yaw ring for the largest
head in the top-right corner (0 = down = facing the camera, 90 = left).

Usage (default models):
  python demo/demo_hfhpe_onnx.py -i images_dir -o output_dir
  python demo/demo_hfhpe_onnx.py -v 0 -o output_dir               # camera 0
  python demo/demo_hfhpe_onnx.py -v input.mp4 -o output_dir -d cuda
  python demo/demo_hfhpe_onnx.py -pm models/hfhpe_hgnetv2_Nx3x64x64.onnx -i images -o out

Keys (video / camera): ESC = quit, b = toggle the head bboxes, a = toggle the 3 axes,
t = toggle the angle text, r = toggle the yaw ring.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

try:  # load CUDA / cuDNN first so that onnxruntime-gpu can see them (when running from a venv)
    import torch  # noqa: F401
except ImportError:
    pass
import onnxruntime as ort

DEFAULT_DETECTOR = "models/deimv2_dinov3_s_wholebody49_boxes_only.onnx"
DEFAULT_POSE = "models/hfhpe_vitt_1x3x64x64.onnx"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HEAD_CLASS_ID = 7
DRAW_COLOR = (0, 255, 0)  # BGR. bbox, ring sector and text
RING_COLOR = (230, 230, 230)  # ring ticks (UI element)


@dataclass
class HeadBox:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


@dataclass
class HeadResult:
    box: HeadBox
    yaw_deg: float               # 0 = facing the camera, +90 = viewer's left (mod 360)
    pitch_deg: float             # positive = looking up (signed)
    roll_deg: float | None       # positive = image-CCW (signed). None for yaw/pitch-only graphs
    kappa: np.ndarray | None     # von Mises concentration of [yaw, pitch(, roll)]


# ---------------------------------------------------------------------------
# onnxruntime
# ---------------------------------------------------------------------------
@dataclass
class TrtSettings:
    """TensorRT EP precision and engine cache location (decided once at startup)."""
    precision: str          # fp16 / bf16 / int8
    cache_dir: Path
    ort_version: str
    trt_version: str
    compute_capability: str


def _ort_version_tuple() -> tuple[int, int]:
    major, minor = (int(v) for v in ort.__version__.split(".")[:2])
    return major, minor


def _tensorrt_version() -> str:
    """TensorRT version via libnvinfer getInferLibVersion() (no tensorrt Python package needed)."""
    import ctypes
    import ctypes.util
    for name in ("nvinfer", "nvinfer.so.10", "nvinfer.so.8"):
        path = ctypes.util.find_library(name) if "." not in name else f"lib{name}"
        if not path:
            continue
        try:
            lib = ctypes.CDLL(path)
            lib.getInferLibVersion.restype = ctypes.c_int32
            v = int(lib.getInferLibVersion())
            return f"{v // 10000}.{(v // 100) % 100}.{v % 100}"
        except OSError:
            continue
    return "unknown"


def _compute_capability() -> str:
    """GPU compute capability (e.g. "86"); "unknown" when torch is not available."""
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f"{major}{minor}"
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def resolve_trt_precision(inference_type: str) -> str:
    """`auto`: bf16 with onnxruntime >= 1.25 (has trt_bf16_enable) on Ampere or newer, else fp16.

    An explicit choice is respected, but bf16 is not accepted by onnxruntime < 1.25, so that
    case raises an error.
    """
    version = _ort_version_tuple()
    bf16_available = version >= (1, 25)
    cc = _compute_capability()
    bf16_hw = cc == "unknown" or int(cc) >= 80
    if inference_type == "auto":
        if not bf16_available:
            print(f"onnxruntime-gpu >= 1.25.0 is recommended for TensorRT (BF16 via trt_bf16_enable); installed {ort.__version__}, "
                  "using fp16. Install the tensorrt dependency group and keep its flags on uv run: "
                  "uv sync --frozen --no-group ort --group tensorrt && uv run --no-group ort --group tensorrt python ...")
            return "fp16"
        if not bf16_hw:
            print(f"GPU compute capability {cc} does not support BF16 in TensorRT (needs >= 8.0); using fp16")
            return "fp16"
        return "bf16"
    if inference_type == "bf16" and not bf16_available:
        raise RuntimeError(f"--inference_type bf16 needs onnxruntime-gpu >= 1.25 (installed {ort.__version__}); "
                           "install the tensorrt dependency group and run with the same flags: "
                           "uv run --no-group ort --group tensorrt python ...")
    return inference_type


def prepare_trt_cache(cache_root: Path, precision: str) -> TrtSettings:
    """Keep engine caches in a separate directory per onnxruntime version / TensorRT version /
    precision / GPU, and delete caches built with another onnxruntime version (engines must
    always be rebuilt when the version changes)."""
    import shutil
    ort_version = ort.__version__
    trt_version = _tensorrt_version()
    cc = _compute_capability()
    cache_root.mkdir(parents=True, exist_ok=True)
    prefix = f"ort-{ort_version}_"
    for entry in sorted(cache_root.iterdir()):
        if entry.is_dir() and entry.name.startswith("ort-") and not entry.name.startswith(prefix):
            shutil.rmtree(entry)
            print(f"Removed stale TensorRT engine cache built with another onnxruntime version: {entry}")
    cache_dir = cache_root / f"{prefix}trt-{trt_version}_{precision}_sm{cc}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return TrtSettings(precision=precision, cache_dir=cache_dir, ort_version=ort_version,
                       trt_version=trt_version, compute_capability=cc)


def build_providers(device: str | None, trt: TrtSettings | None = None) -> list:
    """Build execution providers by the same rules as the reference demo (DEIMv2-Wholebody49)."""
    available = set(ort.get_available_providers())
    requested = (device or "").lower()
    if requested.startswith("cuda"):
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider is not available in this onnxruntime build.")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if requested == "tensorrt":
        if "TensorrtExecutionProvider" not in available:
            raise RuntimeError("TensorrtExecutionProvider is not available in this onnxruntime build.")
        if trt is None:
            raise ValueError("TensorRT settings are required for the tensorrt device")
        if trt.precision == "fp16":
            type_params = {"trt_fp16_enable": True}
        elif trt.precision == "bf16":
            type_params = {"trt_bf16_enable": True}
        elif trt.precision == "int8":
            type_params = {"trt_fp16_enable": True, "trt_int8_enable": True,
                           "trt_int8_calibration_table_name": "calibration.flatbuffers"}
        else:
            raise ValueError(f"Unsupported inference type for TensorRT: {trt.precision}")
        providers: list = [(
            "TensorrtExecutionProvider",
            {"trt_engine_cache_enable": True,
             "trt_engine_cache_path": str(trt.cache_dir),
             "trt_timing_cache_enable": True,
             "trt_timing_cache_path": str(trt.cache_dir),
             "trt_op_types_to_exclude": "NonMaxSuppression,NonZero,RoiAlign"} | type_params,
        )]
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers
    if requested and requested != "cpu":
        raise ValueError(f"Unsupported device: {device}. Use cpu, cuda, cuda:0, or tensorrt.")
    if device is None and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


_ORT_LOG_LEVELS = {"verbose": 0, "info": 1, "warning": 2, "error": 3, "fatal": 4}


def set_ort_log_level(level: str) -> None:
    """Set the severity of the onnxruntime global (Default) logger.

    The `[W:onnxruntime:Default, tensorrt_execution_provider…]` warnings that the TensorRT EP
    emits during engine build (Int64 binding, timing cache not created yet, etc.) cannot be
    suppressed by the session's log_severity_level; they are controlled here. Default: error
    (no warnings).
    """
    ort.set_default_logger_severity(_ORT_LOG_LEVELS[level])


def make_session(model: Path | bytes, providers: list) -> ort.InferenceSession:
    """Create a session from a model file or from serialized (e.g. pruned) model bytes."""
    if isinstance(model, Path):
        if not model.exists():
            raise FileNotFoundError(f"Model file not found: {model}")
        model = str(model)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(model, sess_options=so, providers=providers)


# ---------------------------------------------------------------------------
# Stage 1: DEIMv2-Wholebody49 (head detection)
# ---------------------------------------------------------------------------
DETECTOR_OUTPUT = "label_xyxy_score"


def load_onnx_model(model_path: Path):
    """onnx.ModelProto of `model_path`, or None when the onnx package is missing or the file cannot
    be parsed (the callers then fall back to the file name / the unmodified file)."""
    try:
        import onnx  # noqa: PLC0415
        return onnx.load(str(model_path))
    except Exception:
        return None


def detect_detector_norm(model, model_path: Path) -> str:
    """Infer the detector input normalization from the ONNX graph ("imagenet" / "div255").

    DINOv3 (ViT) based DEIMv2 models are trained with ImageNet mean/std normalization, while
    HGNetV2 based ones are trained with division by 255 only. The normalization is not embedded
    in the graph, so the model is treated as ViT based when a Conv fed directly by the input is a
    patch embedding (stride 8 or more). If the graph is not available (`model` is None), the
    decision is based on whether the file name contains "dinov3".
    """
    if model is None:
        return "imagenet" if "dinov3" in model_path.name.lower() else "div255"
    graph = model.graph
    input_name = graph.input[0].name
    for node in graph.node:
        if node.op_type == "Conv" and input_name in node.input:
            strides = next((list(a.ints) for a in node.attribute if a.name == "strides"), [1, 1])
            if max(strides) >= 8:
                return "imagenet"
    return "div255"


def _referenced_names(graph) -> set[str]:
    """Tensor names read by the nodes of `graph`, including those inside control-flow subgraphs."""
    names: set[str] = set()
    for node in graph.node:
        names.update(node.input)
        for attr in node.attribute:
            for sub in ([attr.g] if attr.HasField("g") else []) + list(attr.graphs):
                names.update(_referenced_names(sub))
    return names


def prune_graph_to_outputs(model, output_names: Sequence[str]) -> int:
    """Remove (in place) the nodes, initializers and value_info that do not feed `output_names`,
    which must be existing graph outputs. Returns the number of removed nodes."""
    graph = model.graph
    missing = set(output_names) - {o.name for o in graph.output}
    if missing:
        raise ValueError(f"not graph outputs: {sorted(missing)}")
    producer = {name: node for node in graph.node for name in node.output}
    needed: set[int] = set()
    stack = list(output_names)
    while stack:
        node = producer.get(stack.pop())
        if node is None or id(node) in needed:
            continue
        needed.add(id(node))
        stack.extend(node.input)
        for attr in node.attribute:  # outer-scope tensors read inside If / Loop / Scan bodies
            for sub in ([attr.g] if attr.HasField("g") else []) + list(attr.graphs):
                stack.extend(_referenced_names(sub))
    kept = [node for node in graph.node if id(node) in needed]
    removed = len(graph.node) - len(kept)
    if removed == 0:
        return 0
    del graph.node[:]
    graph.node.extend(kept)
    used = _referenced_names(graph) | set(output_names)
    produced = {name for node in kept for name in node.output}
    for field, keep in ((graph.initializer, lambda t: t.name in used),
                        (graph.value_info, lambda v: v.name in used or v.name in produced),
                        (graph.output, lambda o: o.name in output_names)):
        items = [x for x in field if keep(x)]
        del field[:]
        field.extend(items)
    return removed


def prune_detector_model(model, model_path: Path) -> bytes | Path:
    """Serialized detector graph reduced to the nodes that feed `label_xyxy_score`, or the file
    path when nothing has to be removed (or the graph could not be loaded).

    The DEIMv2 `*_boxes_only.onnx` files were derived from the instance-segmentation graphs by
    deleting the `masks` output only: the mask branch (mask_embed_head / mask_feature_head →
    NonZero → Gather → Einsum 'mc,mchw->mhw') is still in the graph and onnxruntime executes it on
    every frame. The Einsum runs over the top-K rows of class 0 (body); on a frame without any
    such row — e.g. a nearly black low-light frame — it receives zero-size tensors, which makes the
    CUDA EP fail with CUBLAS_STATUS_INVALID_VALUE and the CPU EP crash with a segmentation fault.
    Pruning removes the branch (and about 34 MB of its weights); label_xyxy_score is unchanged.
    """
    if model is None:
        return model_path
    try:
        removed = prune_graph_to_outputs(model, [DETECTOR_OUTPUT])
    except ValueError:  # the output is missing: let onnxruntime report it on the original file
        return model_path
    return model.SerializeToString() if removed else model_path


class HeadDetector:
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def __init__(self, model_path: Path, providers: list, score_threshold: float,
                 input_norm: str = "auto"):
        model = load_onnx_model(model_path)
        self.input_norm = (detect_detector_norm(model, model_path) if input_norm == "auto"
                           else input_norm)
        self.session = make_session(prune_detector_model(model, model_path), providers)
        del model
        self.providers = self.session.get_providers()
        self.score_threshold = score_threshold
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        if len(shape) != 4 or not all(isinstance(v, int) for v in shape[2:4]):
            raise ValueError(f"Detector input must be [N, 3, H, W] with fixed H/W, got {shape}")
        self.input_hw = (int(shape[2]), int(shape[3]))
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.last_inference_time = 0.0

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = self.input_hw
        resized = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = rgb.transpose(2, 0, 1).astype(np.float32)[None] / 255.0
        if self.input_norm == "imagenet":
            x = (x - self.IMAGENET_MEAN) / self.IMAGENET_STD
        return x

    def __call__(self, image_bgr: np.ndarray) -> list[HeadBox]:
        img_h, img_w = image_bgr.shape[:2]
        feed = {self.input_name: self.preprocess(image_bgr)}
        if "orig_target_sizes" in self.input_names:
            feed["orig_target_sizes"] = np.array([[img_w, img_h]], dtype=np.float32)
        t0 = time.perf_counter()
        (pred,) = self.session.run([DETECTOR_OUTPUT], feed)
        self.last_inference_time = time.perf_counter() - t0
        pred = pred[0]  # [Q, 6]
        keep = (pred[:, 0].astype(np.int64) == HEAD_CLASS_ID) & (pred[:, 5] >= self.score_threshold)
        heads: list[HeadBox] = []
        for _, x1, y1, x2, y2, score in pred[keep]:
            if "orig_target_sizes" not in self.input_names:  # normalized coordinates → pixels
                x1, x2 = x1 * img_w, x2 * img_w
                y1, y2 = y1 * img_h, y2 * img_h
            x1, x2 = float(np.clip(x1, 0, img_w - 1)), float(np.clip(x2, 0, img_w - 1))
            y1, y2 = float(np.clip(y1, 0, img_h - 1)), float(np.clip(y2, 0, img_h - 1))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            heads.append(HeadBox(x1, y1, x2, y2, float(score)))
        heads.sort(key=lambda b: -b.score)
        return heads


# ---------------------------------------------------------------------------
# Stage 2: HFHPE (yaw/pitch/roll of the head crops)
# ---------------------------------------------------------------------------
def square_crop(image: np.ndarray, box: HeadBox, margin: float) -> np.ndarray:
    """Crop contract (shared by training, evaluation and deployment): expand each side by margin
    → long-side square → clamp at the image border. If the image is smaller than the square,
    zero-pad it to a square."""
    H, W = image.shape[:2]
    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
    w, h = x2 - x1, y2 - y1
    x1 -= w * margin
    x2 += w * margin
    y1 -= h * margin
    y2 += h * margin
    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sx1 = min(max(cx - side / 2, 0), max(W - side, 0))
    sy1 = min(max(cy - side / 2, 0), max(H - side, 0))
    sx2 = min(sx1 + side, W)
    sy2 = min(sy1 + side, H)
    crop = image[round(sy1):round(sy2), round(sx1):round(sx2)]
    ch, cw = crop.shape[:2]
    if cw != ch:  # image smaller than the requested square
        s = max(cw, ch)
        padded = np.zeros((s, s, 3), dtype=crop.dtype)
        oy, ox = (s - ch) // 2, (s - cw) // 2
        padded[oy:oy + ch, ox:ox + cw] = crop
        crop = padded
    return crop


def resize_square(image: np.ndarray, size: int) -> np.ndarray:
    """INTER_AREA when shrinking, INTER_LANCZOS4 when enlarging (same rule as the dataset build)."""
    interp = cv2.INTER_AREA if size < image.shape[0] else cv2.INTER_LANCZOS4
    return cv2.resize(image, (size, size), interpolation=interp)


class HeadPoseEstimator:
    """HFHPE integrated ONNX: `images` [N, 3, S, S] (RGB, center05) → `cos_sin` [N, 6] + `kappa` [N, 3].

    Standalone yaw/pitch-only graphs (cos_sin [N, 4], kappa [N, 2] or absent) are also accepted.
    The input normalization is decided from the metadata written at export time (graphs with
    ImageNet normalization record mean/std in `input.images`; without that record, center05).
    """

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0

    def __init__(self, model_path: Path, providers: list, crop_margin: float):
        self.session = make_session(model_path, providers)
        self.providers = self.session.get_providers()
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        if len(shape) != 4 or not isinstance(shape[2], int) or shape[2] != shape[3]:
            raise ValueError(f"Pose model input must be [N, 3, S, S] with fixed S, got {shape}")
        self.size = int(shape[2])
        self.dynamic_batch = not isinstance(shape[0], int)
        output_names = [o.name for o in self.session.get_outputs()]
        if "cos_sin" not in output_names:
            raise ValueError(f"Pose model must output `cos_sin`, got {output_names}")
        self.has_kappa = "kappa" in output_names
        self.output_names = ["cos_sin"] + (["kappa"] if self.has_kappa else [])
        meta = self.session.get_modelmeta().custom_metadata_map or {}
        self.input_norm = ("imagenet" if "mean=" in meta.get("input.images", "")
                           else "center05")
        self.crop_margin = crop_margin
        self.last_inference_time = 0.0
        self._nonfinite_warned = False

    def preprocess(self, image_bgr: np.ndarray, box: HeadBox) -> np.ndarray:
        crop = resize_square(square_crop(image_bgr, box, self.crop_margin), self.size)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.input_norm == "imagenet":
            return ((rgb - self.IMAGENET_MEAN) / self.IMAGENET_STD).transpose(2, 0, 1)
        return (rgb / 127.5 - 1.0).transpose(2, 0, 1)

    def _run(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        outs = self.session.run(self.output_names, {self.input_name: batch})
        return outs[0], (outs[1] if self.has_kappa else None)

    def __call__(self, image_bgr: np.ndarray, heads: Sequence[HeadBox]) -> list[HeadResult]:
        if not heads:
            self.last_inference_time = 0.0
            return []
        batch = np.stack([self.preprocess(image_bgr, box) for box in heads]).astype(np.float32)
        t0 = time.perf_counter()
        if self.dynamic_batch:
            cos_sin, kappa = self._run(batch)
        else:
            outs = [self._run(batch[i:i + 1]) for i in range(len(batch))]
            cos_sin = np.concatenate([o[0] for o in outs])
            kappa = np.concatenate([o[1] for o in outs]) if self.has_kappa else None
        self.last_inference_time = time.perf_counter() - t0

        results: list[HeadResult] = []
        for i, box in enumerate(heads):
            cs = cos_sin[i]
            if not np.all(np.isfinite(cs)):  # EP numerical breakdown (e.g. TensorRT fp16 overflow)
                if not self._nonfinite_warned:
                    self._nonfinite_warned = True
                    print("Pose model returned non-finite cos_sin; skipping those heads. "
                          "With -d tensorrt this is usually an FP16 overflow (e.g. the dinov3 320px graph): "
                          "use the BF16 path (uv sync --frozen --no-group ort --group tensorrt && "
                          "uv run --no-sync ... -d tensorrt) or fall back to -d cuda.")
                continue
            yaw = math.degrees(math.atan2(float(cs[1]), float(cs[0]))) % 360.0
            pitch = math.degrees(math.atan2(float(cs[3]), float(cs[2])))
            roll = math.degrees(math.atan2(float(cs[5]), float(cs[4]))) if len(cs) >= 6 else None
            results.append(HeadResult(box=box, yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                                      kappa=None if kappa is None else kappa[i]))
        return results


# ---------------------------------------------------------------------------
# Drawing and saving
# ---------------------------------------------------------------------------
def _signed(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def draw_axis(img: np.ndarray, yaw: float, pitch: float, roll: float,
              cx: int, cy: int, size: float, thickness: int) -> None:
    """Draw the 3 axes of the HFHPE convention (yaw: += viewer's left / pitch: += looking up /
    roll: += image-CCW) with the composite R_img_roll @ R_yaw @ R_pitch (same as
    scripts/render_hfhpe_preview.py).

    Camera coordinates: x = image right, y = image down, z = depth. The local basis for a frontal,
    upright head is X=(1,0,0) image right (subject's right cheek), Y=(0,1,0) chin = image down,
    Z=(0,0,-1) nose tip = towards the camera. roll is a rotation within the image plane, so it is
    applied outermost (leftmost) as a rotation about the camera z axis. The extrinsic closed form
    (HopeNet style) is not used because the pitch projection flips for |yaw|>90."""
    ps, yw, rl = (math.radians(pitch), math.radians(_signed(yaw)),
                  math.radians(_signed(roll)))
    cy_, sy = math.cos(yw), math.sin(yw)
    cp, sp = math.cos(ps), math.sin(ps)
    cr, sr = math.cos(rl), math.sin(rl)
    r_yaw = np.array([[cy_, 0, sy], [0, 1, 0], [-sy, 0, cy_]])
    r_pitch = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
    r_img_roll = np.array([[cr, sr, 0], [-sr, cr, 0], [0, 0, 1]])
    rot = r_img_roll @ r_yaw @ r_pitch
    for axis, color in [((1.0, 0.0, 0.0), (0, 0, 255)),    # X: red (BGR)
                        ((0.0, 1.0, 0.0), (0, 255, 0)),    # Y: green
                        ((0.0, 0.0, -1.0), (255, 0, 0))]:  # Z: blue
        v = rot @ np.array(axis)
        cv2.line(img, (cx, cy), (int(cx + size * v[0]), int(cy + size * v[1])),
                 color, thickness, cv2.LINE_AA)


def draw_results(image: np.ndarray, results: Sequence[HeadResult], draw_bbox: bool,
                 draw_axes: bool, draw_text: bool) -> np.ndarray:
    out = image.copy()
    for r in results:
        side = max(r.box.x2 - r.box.x1, r.box.y2 - r.box.y1)
        thickness = max(1, int(round(side / 160.0)))
        if draw_bbox:
            cv2.rectangle(out, (int(round(r.box.x1)), int(round(r.box.y1))),
                          (int(round(r.box.x2)), int(round(r.box.y2))), DRAW_COLOR, thickness, cv2.LINE_AA)
        if draw_axes:
            cx = int(round((r.box.x1 + r.box.x2) / 2.0))
            cy = int(round((r.box.y1 + r.box.y2) / 2.0))
            draw_axis(out, r.yaw_deg, r.pitch_deg, r.roll_deg or 0.0,
                      cx, cy, 0.45 * side, max(2, thickness))
        if draw_text:
            roll_txt = "" if r.roll_deg is None else f" r{_signed(r.roll_deg):+6.1f}"
            text = f"y{r.yaw_deg:6.1f} p{r.pitch_deg:+6.1f}{roll_txt}"
            x, y = int(round(r.box.x1)), int(round(r.box.y1)) - 6
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            if y - th - 2 < 0:  # move inside when the bbox touches the top edge of the frame
                y = int(round(r.box.y1)) + th + 6
            cv2.rectangle(out, (x - 2, y - th - 2), (x + tw + 2, y + base), (0, 0, 0), -1)
            cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _yaw_vector(deg: float) -> tuple[float, float]:
    """yaw [deg] → unit vector (dx, dy) in image coordinates.
    0 = down (facing the camera), 90 = left, 180 = up, 270 = right."""
    rad = math.radians(deg)
    return -math.sin(rad), math.cos(rad)


def _box_iou(a: HeadBox, b: HeadBox) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union > 0.0 else 0.0


class YawSmoother:
    """Circular EMA for the yaw ring (display only; not applied to yaw_deg in the JSON).

    Averaging angles directly breaks at the 0/360 wrap-around, so the unit vectors (cos θ, sin θ)
    are blended with α = 1 − exp(−Δt/τ) and converted back to an angle with atan2. When the
    "largest head" tracked by the ring switches to another head (IoU with the previous bbox < 0.3),
    the state is discarded and the ring follows immediately.
    """

    def __init__(self, tau: float):
        self.tau = tau
        self.vec: np.ndarray | None = None
        self.box: HeadBox | None = None
        self.t: float | None = None

    def update(self, box: HeadBox, deg: float) -> float:
        now = time.perf_counter()
        v = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))], dtype=np.float64)
        if self.vec is None or self.box is None or self.t is None or _box_iou(self.box, box) < 0.3:
            self.vec, self.box, self.t = v, box, now
            return deg
        alpha = 1.0 if self.tau <= 0.0 else 1.0 - float(np.exp(-max(now - self.t, 0.0) / self.tau))
        w = (1.0 - alpha) * self.vec + alpha * v
        n = float(np.hypot(w[0], w[1]))
        self.vec = v if n < 1e-6 else w / n
        self.box, self.t = box, now
        return float(np.degrees(np.arctan2(self.vec[1], self.vec[0])) % 360.0)


def draw_yaw_ring(image: np.ndarray, results: Sequence[HeadResult],
                  ring_deg: float | None = None) -> None:
    """Draw a ring in the top-right corner showing the yaw of the largest head (HFHPE convention:
    0 = down = facing the camera, 90 = left).

    If ring_deg is given, the sector and the center number use that angle (smoothed display angle).
    """
    if not results:
        return
    h, w = image.shape[:2]
    primary = max(results, key=lambda r: (r.box.x2 - r.box.x1) * (r.box.y2 - r.box.y1))
    radius = max(28, int(round(min(w, h) * 0.09)))
    band = max(6, radius // 4)
    margin = band + 20  # keep the outer labels (radius + band + 9) inside the frame
    center = (w - radius - margin, radius + margin)
    # ring band (semi-transparent base + ticks)
    overlay = image.copy()
    cv2.circle(overlay, center, radius, (40, 40, 40), band, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    cv2.circle(image, center, radius, RING_COLOR, 1, cv2.LINE_AA)
    cv2.circle(image, center, radius - band, RING_COLOR, 1, cv2.LINE_AA)
    for theta, label in ((0.0, "0"), (90.0, "90"), (180.0, "180"), (270.0, "270")):
        dx, dy = _yaw_vector(theta)
        p1 = (int(round(center[0] + dx * (radius - band))), int(round(center[1] + dy * (radius - band))))
        p2 = (int(round(center[0] + dx * (radius + band // 2 + 2))), int(round(center[1] + dy * (radius + band // 2 + 2))))
        cv2.line(image, p1, p2, RING_COLOR, 1, cv2.LINE_AA)
        tx, ty = center[0] + dx * (radius + band + 9), center[1] + dy * (radius + band + 9)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.putText(image, label, (int(round(tx - tw / 2)), int(round(ty + th / 2))), cv2.FONT_HERSHEY_SIMPLEX, 0.35, RING_COLOR, 1, cv2.LINE_AA)
    deg = primary.yaw_deg if ring_deg is None else ring_deg
    # OpenCV ellipse angles run from +x towards +y (down); φ = 90 + θ gives 0 = down, 90 = left
    phi = 90.0 + deg
    cv2.ellipse(image, center, (radius, radius), 0.0, phi - 9.0, phi + 9.0, DRAW_COLOR, band, cv2.LINE_AA)
    text = f"{deg:.0f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(image, text, (center[0] - tw // 2, center[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DRAW_COLOR, 1, cv2.LINE_AA)


def put_text(image: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.7, thickness: int = 2) -> None:
    """Draw white text once on a black band.

    The "thick white text with thin red text on top as an outline" approach looks misaligned with
    OpenCV 5 text rendering because the glyph advance changes with the line thickness (the same
    string comes out 138 px and 146 px wide), so the two strings appear shifted.
    """
    x, y = org
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(image, (x - 4, y - th - 4), (x + tw + 4, y + base + 2), (0, 0, 0), -1)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def results_to_records(results: Sequence[HeadResult]) -> list[dict]:
    return [{
        "head_bbox": [round(r.box.x1, 2), round(r.box.y1, 2), round(r.box.x2, 2), round(r.box.y2, 2)],
        "score": round(r.box.score, 4),
        "yaw_deg": round(r.yaw_deg, 2),
        "pitch_deg": round(r.pitch_deg, 2),
        "roll_deg": None if r.roll_deg is None else round(_signed(r.roll_deg), 2),
        "kappa": None if r.kappa is None else [round(float(k), 3) for k in r.kappa],
    } for r in results]


def save_records(output_dir: Path, stem: str, results: Sequence[HeadResult]) -> None:
    (output_dir / f"{stem}.json").write_text(
        json.dumps({"heads": results_to_records(results)}, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
class Pipeline:
    def __init__(self, args: argparse.Namespace):
        set_ort_log_level(args.ort_log_level)
        det_path, pose_path = Path(args.detector_model), Path(args.pose_model)
        self.trt: TrtSettings | None = None
        if (args.device or "").lower() == "tensorrt":
            self.trt = prepare_trt_cache(Path(args.trt_cache_dir), resolve_trt_precision(args.inference_type))
            print(f"TensorRT: precision {self.trt.precision} (onnxruntime {self.trt.ort_version}, TensorRT {self.trt.trt_version}, "
                  f"sm{self.trt.compute_capability}), engine cache {self.trt.cache_dir}")
        providers = build_providers(args.device, self.trt)
        self.detector = HeadDetector(det_path, providers, args.head_score_threshold,
                                     args.detector_norm)
        self.estimator = HeadPoseEstimator(pose_path, providers, args.crop_margin)
        self.draw_bbox = not args.disable_bbox
        self.draw_axes = not args.disable_axes
        self.draw_text = not args.disable_angle_text
        self.draw_ring = not args.disable_yaw_ring
        self.yaw_smooth_tau = args.yaw_smooth_tau
        self.ring_smoother: YawSmoother | None = None  # set by process_video (video / camera only)
        self.warm_up()

    def warm_up(self) -> None:
        """Run both sessions once on dummy input so that first-run kernel selection and memory
        allocation are excluded from the timings."""
        h, w = self.detector.input_hw
        self.detector(np.zeros((h, w, 3), dtype=np.uint8))
        s = self.estimator.size
        self.estimator(np.zeros((s, s, 3), dtype=np.uint8), [HeadBox(0.0, 0.0, float(s - 1), float(s - 1), 1.0)])

    def run(self, image: np.ndarray) -> tuple[np.ndarray, list[HeadResult], float, float]:
        t0 = time.perf_counter()
        heads = self.detector(image)
        results = self.estimator(image, heads)
        infer_ms = (self.detector.last_inference_time + self.estimator.last_inference_time) * 1000.0
        rendered = draw_results(image, results, self.draw_bbox, self.draw_axes, self.draw_text)
        if self.draw_ring and results:
            ring_deg = None
            if self.ring_smoother is not None:
                primary = max(results, key=lambda r: (r.box.x2 - r.box.x1) * (r.box.y2 - r.box.y1))
                ring_deg = self.ring_smoother.update(primary.box, primary.yaw_deg)
            draw_yaw_ring(rendered, results, ring_deg=ring_deg)
        total_ms = (time.perf_counter() - t0) * 1000.0
        return rendered, results, infer_ms, total_ms


def list_image_paths(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def is_int(value: str) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


def safe_fps(fps: float) -> float:
    """Frame rate reported by the source, or 30 when the backend does not know it."""
    return fps if fps and math.isfinite(fps) and fps > 0 else 30.0


def create_video_writer(output_dir: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    return cv2.VideoWriter(str(output_dir / "output.mp4"), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))


class RealTimePacer:
    """How many file frames to write for a captured camera frame so that the constant-frame-rate
    recording plays back in real time.

    A webcam in low light extends its exposure time (auto exposure) and delivers far fewer frames
    per second than its nominal CAP_PROP_FPS (e.g. 5 instead of 30), and the processing loop itself
    may run slower than the camera. Writing one file frame per captured frame then gives a
    fast-forwarded video. The pacer compares the wall-clock time of each captured frame with the
    file clock: it repeats the frame when the camera falls behind and skips it when frames arrive
    faster than the file rate.
    """

    def __init__(self, fps: float):
        self.fps = fps
        self.t_first: float | None = None
        self.written = 0

    def __call__(self, t: float) -> int:
        if self.t_first is None:
            self.t_first = t
        due = int((t - self.t_first) * self.fps + 0.5) + 1  # frames the file should hold at time t
        count = max(0, due - self.written)
        self.written += count
        return count


def process_images(pipeline: Pipeline, images_dir: Path, output_dir: Path, save_raw: bool) -> None:
    paths = list_image_paths(images_dir)
    if not paths:
        raise FileNotFoundError(f"No image files found in {images_dir}")
    print(f"Processing {len(paths)} images from {images_dir}")
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        rendered, results, infer_ms, total_ms = pipeline.run(image)
        cv2.imwrite(str(output_dir / path.name), rendered)
        if save_raw:
            save_records(output_dir, path.stem, results)
        print(f"{path.name}: heads={len(results)} infer={infer_ms:.1f} ms total={total_ms:.1f} ms")


def process_video(pipeline: Pipeline, video: str, output_dir: Path, save_raw: bool,
                  disable_video_writer: bool, disable_imshow: bool) -> None:
    is_camera = is_int(video)
    cap = cv2.VideoCapture(int(video) if is_camera else video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {video}")
    if is_camera:  # camera input is fixed to VGA (640×480)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"Processing video source: {video}")
    if pipeline.yaw_smooth_tau > 0.0:
        pipeline.ring_smoother = YawSmoother(pipeline.yaw_smooth_tau)
    writer = None
    pacer: RealTimePacer | None = None  # camera input: keep the recording in real time
    frame_index = 0
    t_first = t_frame = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_frame = time.perf_counter()  # ≈ capture time of this frame
            if frame_index == 0:
                t_first = t_frame
            frame_index += 1
            rendered, results, infer_ms, total_ms = pipeline.run(frame)
            put_text(rendered, f"infer: {infer_ms:.2f} ms", (10, 30))
            put_text(rendered, f"total: {total_ms:.2f} ms", (10, 58))
            put_text(rendered, f"heads: {len(results)}", (10, 86))
            if writer is None and not disable_video_writer:
                fps = safe_fps(cap.get(cv2.CAP_PROP_FPS))
                writer = create_video_writer(output_dir, rendered.shape[1], rendered.shape[0], fps)
                if is_camera:
                    pacer = RealTimePacer(fps)
            if writer is not None:
                for _ in range(pacer(t_frame) if pacer is not None else 1):
                    writer.write(rendered)
            if save_raw:
                save_records(output_dir, f"{frame_index:08d}", results)
            if not disable_imshow:
                cv2.imshow("HFHPE ONNX demo", rendered)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                if key == ord("b"):
                    pipeline.draw_bbox = not pipeline.draw_bbox
                elif key == ord("a"):
                    pipeline.draw_axes = not pipeline.draw_axes
                elif key == ord("t"):
                    pipeline.draw_text = not pipeline.draw_text
                elif key == ord("r"):
                    pipeline.draw_ring = not pipeline.draw_ring
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not disable_imshow:
            cv2.destroyAllWindows()
        if pacer is not None and frame_index > 1:
            elapsed = t_frame - t_first
            print(f"Camera: {frame_index} frames captured in {elapsed:.1f} s ({(frame_index - 1) / elapsed:.1f} fps); "
                  f"{pacer.written} frames written at {pacer.fps:g} fps ({pacer.written / pacer.fps:.1f} s)")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="HFHPE ONNX demo: DEIMv2-Wholebody49 head detection followed by HFHPE yaw/pitch/roll estimation on each head crop.")
    ap.add_argument("-dm", "--detector_model", type=str, default=DEFAULT_DETECTOR,
                    help="DEIMv2-Wholebody49 ONNX (outputs label_xyxy_score; class 7 = head)")
    ap.add_argument("-pm", "--pose_model", type=str, default=DEFAULT_POSE,
                    help="HFHPE ONNX (images [N,3,S,S] -> cos_sin, kappa); fixed-batch and N-batch graphs are both accepted")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("-i", "--images_dir", type=str, help="directory of input images")
    src.add_argument("-v", "--video", type=str, help="video file path or camera index")
    ap.add_argument("-o", "--output_dir", type=str, default="output")
    ap.add_argument("-d", "--device", type=str, default=None, help="cpu, cuda, cuda:0 or tensorrt (default: cuda if available)")
    ap.add_argument("--inference_type", type=str, choices=["auto", "fp16", "bf16", "int8"], default="auto",
                    help="TensorRT precision; auto = bf16 with onnxruntime-gpu >= 1.25 on Ampere or newer GPUs, otherwise fp16")
    ap.add_argument("--ort_log_level", type=str, choices=list(_ORT_LOG_LEVELS), default="error",
                    help="onnxruntime global log level (default: error, which hides the TensorRT engine-build warnings)")
    ap.add_argument("--trt_cache_dir", type=str, default="models/trt_cache",
                    help="TensorRT engine cache root; engines are kept per onnxruntime / TensorRT version, precision and GPU, "
                         "and caches built with another onnxruntime version are deleted at startup")
    ap.add_argument("--detector_norm", type=str, default="auto",
                    choices=["auto", "div255", "imagenet"],
                    help="detector input normalization: auto = ImageNet mean/std for DINOv3-based "
                         "detectors and plain /255 for HGNetV2-based ones (detected from the graph). "
                         "The training crops were built with div255; pass it to reproduce them exactly")
    ap.add_argument("--head_score_threshold", type=float, default=0.35,
                    help="head detection score threshold (the training crops were built with 0.30)")
    ap.add_argument("--crop_margin", type=float, default=0.05,
                    help="head-bbox expansion ratio per side of the square crop (crop-contract value: 0.05)")
    ap.add_argument("--yaw_smooth_tau", type=float, default=0.0,
                    help="circular-EMA time constant in seconds for the yaw ring on video / camera input (display only; default 0 = no smoothing, e.g. 0.15 to enable)")
    ap.add_argument("--disable_bbox", action="store_true", help="do not draw head boxes")
    ap.add_argument("--disable_axes", action="store_true", help="do not draw the 3-axis pose overlay (X=red right cheek, Y=green chin, Z=blue nose)")
    ap.add_argument("--disable_angle_text", action="store_true", help="do not draw the per-head yaw/pitch/roll text")
    ap.add_argument("--disable_yaw_ring", action="store_true", help="do not draw the top-right yaw ring (the angles are still estimated and written to the JSON)")
    ap.add_argument("--disable_video_writer", action="store_true")
    ap.add_argument("--disable_imshow", action="store_true", help="do not open a window for video / camera input")
    ap.add_argument("--save_raw_predictions", action="store_true", help="save head boxes, angles and kappa as JSON")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(args)
    print(f"Detector: {args.detector_model} (input {pipeline.detector.input_hw[1]}x{pipeline.detector.input_hw[0]}, "
          f"input norm {pipeline.detector.input_norm}, "
          f"providers {pipeline.detector.providers})")
    print(f"Pose: {args.pose_model} (input {pipeline.estimator.size}x{pipeline.estimator.size}, "
          f"{'N-batch' if pipeline.estimator.dynamic_batch else 'batch 1'}, "
          f"{'with' if pipeline.estimator.has_kappa else 'no'} kappa, "
          f"input norm {pipeline.estimator.input_norm}, crop margin {pipeline.estimator.crop_margin}, "
          f"providers {pipeline.estimator.providers}; "
          "yaw 0 = facing the camera, 90 = viewer's left; pitch + = up; roll + = image CCW)")
    print(f"Output directory: {output_dir}")
    if args.images_dir is not None:
        images_dir = Path(args.images_dir)
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {images_dir}")
        process_images(pipeline, images_dir, output_dir, args.save_raw_predictions)
    else:
        process_video(pipeline, args.video, output_dir, args.save_raw_predictions,
                      args.disable_video_writer, args.disable_imshow)


if __name__ == "__main__":
    main()
