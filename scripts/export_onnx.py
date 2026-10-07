#!/usr/bin/env python3
"""ONNX export CLI for HFHPE models (DINOv3 teacher / YawNet student).

Teacher (dinov3_yaw) / student (YawNet) is detected automatically from the
checkpoint's model_type. The validation pipeline follows the onnx-export-optimize
skill (HRFFA know-how):
  torch export (batch 1, opset 17, TorchScript)
    → simplify_onnx.py (onnxslim [no Gemm fusion] → onnxsim [constant folding only,
       subprocess + fallback] → fixed-graph canonicalization + rank-5 qkv to 4-D
       → ORT parity [graph optimization OFF])
    → torch vs ORT parity
    → fixed_to_nbatch.py (fixed batch-1 → N, with batch 1/2/3 parity check)
    → audit_onnx.py (mechanical check of invariants)

I/O contract (always `images` (N,3,S,S) → output `cos_sin` unit vector):
  - DINOv3 teacher: ImageNet-normalized RGB (x/255 → (x - mean) / std)
  - YawNet student: center05-normalized RGB (x/127.5 - 1)

--with-kappa (κ-head checkpoints only): adds the κ output
(κ = von Mises concentration = confidence; appends _kappa to the file name).

The output shape depends on the checkpoint type:
  - yaw only (student / 3-output teacher): `cos_sin` (N,2), plus `kappa` (N) with κ
  - yaw+pitch integrated teacher (6 outputs, stem is yawpitch): `cos_sin` (N,4) =
    [cos_yaw, sin_yaw, cos_pitch, sin_pitch] (first and second halves are each
    unit vectors), plus `kappa` (N,2) = [κ_yaw, κ_pitch] with κ

Usage:
    uv run python scripts/export_onnx.py --ckpt runs/dinov3_vitl16_320_teacher_v3
    uv run python scripts/export_onnx.py --ckpt runs/yawnet_distill_64_v3

--ckpt is a directory (best_*.pt is discovered automatically) or a .pt file.
Output goes by default to the same folder as the given weights (--out-dir overrides).
File names: teacher = dinov3_<variant>_yaw_{1,N}x3xSxS.onnx (overwritten on re-export),
student = <run name>_{1,N}x3xSxS.onnx.
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SKILL = Path.home() / ".claude" / "skills" / "onnx-export-optimize" / "scripts"


class KappaWrapper(torch.nn.Module):
    """Thin wrapper to export with the 2 outputs (cos_sin, kappa)."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward_with_kappa(x)


class YawPitchWrapper(torch.nn.Module):
    """Wrapper to export a 6-output teacher as cos_sin (N,4) [+ kappa (N,2)].

    cos_sin = [cos_yaw, sin_yaw, cos_pitch, sin_pitch] (first and second halves
    are each unit vectors), kappa = [kappa_yaw, kappa_pitch].
    """

    def __init__(self, model: torch.nn.Module, with_kappa: bool) -> None:
        super().__init__()
        self.model = model
        self.with_kappa = with_kappa

    def forward(self, x: torch.Tensor
                ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        yaw_unit, kappa_y, pitch_unit, kappa_p = self.model.forward_full(x)
        cos_sin = torch.cat([yaw_unit, pitch_unit], dim=1)
        if not self.with_kappa:
            return cos_sin
        return cos_sin, torch.stack([kappa_y, kappa_p], dim=1)


def load_model(spec: str, size_override: int) -> tuple[torch.nn.Module, Path, int, str, str]:
    """Restore (model, ckpt_path, size, stem, input_norm) from a checkpoint."""
    path = Path(spec)
    if path.is_dir():
        cands = sorted(path.glob("best_*.pt"))
        if not cands:
            raise FileNotFoundError(f"no best_*.pt found in {path}")
        path = cands[0]
    ck: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    args = ck["args"]
    if ck.get("model_type") == "dinov3_yaw":
        from dinov3_yaw import Dinov3YawNet, head_flags  # noqa: PLC0415
        kappa_head, pitch_head = head_flags(ck["model"]["head.3.weight"].shape[0])
        model: torch.nn.Module = Dinov3YawNet(
            ck["variant"], pretrained=False,
            kappa_head=kappa_head, pitch_head=pitch_head)
        size = int(args.get("size", 320))
        stem = f"dinov3_{ck['variant']}_{'yawpitch' if pitch_head else 'yaw'}"
        norm = "imagenet"
    elif ck.get("model_type") == "vitt_yaw":
        from dinov3_yaw import head_flags  # noqa: PLC0415
        from vitt_yaw import VittYawNet  # noqa: PLC0415
        kappa_head, pitch_head = head_flags(ck["model"]["head.3.weight"].shape[0])
        model = VittYawNet(pretrained=False, kappa_head=kappa_head,
                           pitch_head=pitch_head)
        size = int(args.get("student_size") or args.get("size"))
        stem = path.parent.name if path.parent != ROOT else path.stem
        norm = "center05"
    elif ck.get("model_type") == "hgnetv2_yaw":
        from dinov3_yaw import head_flags  # noqa: PLC0415
        from hgnetv2_yaw import HGNetV2Yaw  # noqa: PLC0415
        kappa_head, pitch_head = head_flags(ck["model"]["fc.weight"].shape[0])
        model = HGNetV2Yaw(kappa_head=kappa_head, pitch_head=pitch_head)
        size = int(args.get("student_size") or args.get("size"))
        stem = path.parent.name if path.parent != ROOT else path.stem
        norm = "center05"
    else:
        from dinov3_yaw import head_flags  # noqa: PLC0415
        from yawnet import YawNet  # noqa: PLC0415
        kappa_head, pitch_head = head_flags(ck["model"]["fc.weight"].shape[0])
        model = YawNet(width=float(args.get("width", 1.0)),
                       kappa_head=kappa_head, pitch_head=pitch_head)
        size = int(args.get("student_size") or args.get("size"))
        stem = path.parent.name if path.parent != ROOT else path.stem
        norm = "center05"
    model.load_state_dict(ck["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, path, (size_override or size), stem, norm


def run(cmd: list[str]) -> None:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="run directory (auto-discovers best_*.pt) or a .pt file")
    ap.add_argument("--size", type=int, default=0,
                    help="input resolution override (0 = checkpoint training resolution)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--out-dir", type=str, default="",
                    help="output directory (default: same folder as the given weights)")
    ap.add_argument("--with-kappa", action="store_true",
                    help="include κ (confidence) as the 2nd output (κ-head teachers only)")
    ap.add_argument("--nbatch-atol", type=float, default=1e-4,
                    help="fixed→N-batch batch 1/2/3 validation tolerance. κ outputs "
                         "range to ~100, so relaxing to 1e-3 is harmless in relative error")
    args = ap.parse_args()

    model, ck_path, size, stem, norm = load_model(args.ckpt, args.size)
    n_params = sum(p.numel() for p in model.parameters())
    if getattr(model, "pitch_head", False):
        # yaw+pitch integrated teacher: cos_sin (N,4) [+ kappa (N,2)]
        export_model: torch.nn.Module = YawPitchWrapper(
            model, with_kappa=args.with_kappa).eval()
        output_names = ["cos_sin", "kappa"] if args.with_kappa else ["cos_sin"]
        if args.with_kappa:
            stem = f"{stem}_kappa"
    elif args.with_kappa:
        if not getattr(model, "kappa_head", False):
            raise SystemExit("--with-kappa is only supported for κ-head checkpoints "
                             "(this checkpoint has no κ output)")
        export_model = KappaWrapper(model).eval()
        output_names = ["cos_sin", "kappa"]
        stem = f"{stem}_kappa"
    else:
        export_model = model
        output_names = ["cos_sin"]
    out_dir = Path(args.out_dir) if args.out_dir else ck_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / f"{stem}_1x3x{size}x{size}_raw.onnx"
    fixed = out_dir / f"{stem}_1x3x{size}x{size}.onnx"
    nbatch = out_dir / f"{stem}_Nx3x{size}x{size}.onnx"

    print(f"ckpt: {ck_path} (params={n_params:,}, size={size}, input_norm={norm}, "
          f"outputs={output_names})")
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        export_model, (dummy,), str(raw),
        opset_version=args.opset,
        input_names=["images"],
        output_names=output_names,
        dynamo=False,
    )
    print(f"raw export: {raw} ({raw.stat().st_size/1e6:.1f} MB)")

    # onnxslim → onnxsim → canonicalize → ORT parity check vs raw (generic skill script)
    run([sys.executable, str(SKILL / "simplify_onnx.py"), str(raw), str(fixed)])

    # torch vs ORT parity (graph optimization OFF)
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(fixed), so, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        out = export_model(dummy)
    refs = [o.numpy() for o in (out if isinstance(out, tuple) else (out,))]
    gots = sess.run(None, {"images": dummy.numpy()})
    for name, ref, got in zip(output_names, refs, gots):
        tol = 5e-4 if name == "cos_sin" else 1e-3   # kappa is unnormalized, up to 100
        err = float(np.abs(ref - got).max())
        print(f"parity {name}: torch vs ORT max_err={err:.2e} (tol {tol:.0e})")
        assert err < tol, f"parity failed for {name}: {err:.2e}"

    # fixed batch-1 → N-batch conversion (with batch 1/2/3 numerical parity check) + audit
    run([sys.executable, str(SKILL / "fixed_to_nbatch.py"), str(fixed), str(nbatch),
         "--atol", str(args.nbatch_atol)])
    run([sys.executable, str(SKILL / "audit_onnx.py"), str(fixed),
         "--pair", str(nbatch), "--check", "--atol", str(args.nbatch_atol)])

    # inject the I/O contract metadata (visible in Netron etc.)
    from add_onnx_metadata import add_metadata  # noqa: PLC0415
    for p in (fixed, nbatch):
        add_metadata(p, norm=norm)
    print(f"metadata: input/output contract injected (norm={norm})")

    raw.unlink(missing_ok=True)
    print(f"\nartifacts:\n  fixed : {fixed} ({fixed.stat().st_size/1e6:.1f} MB)\n"
          f"  nbatch: {nbatch} ({nbatch.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
