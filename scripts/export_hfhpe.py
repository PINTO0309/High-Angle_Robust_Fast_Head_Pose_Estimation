#!/usr/bin/env python3
"""ONNX export CLI for the HFHPE (High-Angle Robust Fast Head Pose Estimation)
one-pass integrated model.

Composes a roll-branch checkpoint (train_hfhpe_roll.py) and a 6-output yaw/pitch
body checkpoint (train_teacher_dinov3.py --pitch-head etc.) inside the graph, and
runs the onnx-export-optimize skill's validation pipeline (same as export_onnx.py)
in one go.

Input contract: `images` (N,3,S,S), **center05 normalization** (x/127.5 - 1).
Roll estimation → GridSample derotation → body re-normalization all run in the graph.
Output: `cos_sin` (N,6) = [cos_yaw, sin_yaw, cos_pitch, sin_pitch, cos_roll,
sin_roll] (each pair is a unit vector), `kappa` (N,3) = [κ_yaw, κ_pitch, κ_roll].

Usage:
    uv run python scripts/export_hfhpe.py \
        --roll-ckpt runs/hfhpe_roll_064 \
        --body-ckpt runs/dinov3_vitl16_320_yp_unified_teacher

Output goes by default to the same folder as the body checkpoint. The stem is
`hfhpe_<body arch name>` (e.g. hfhpe_vitt, hfhpe_dinov3; --stem overrides it).
The gate mode and τ, a are recorded under `roll_gate` in metadata_props.
"""
import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from export_onnx import SKILL, load_model, run
from hfhpe import HFHPE
from roll_models import load_roll_checkpoint

ROOT = Path(__file__).resolve().parent.parent


def load_rollnet(spec: str) -> tuple[torch.nn.Module, Path, int]:
    """Restore (model, path, size) from a train_hfhpe_roll.py checkpoint."""
    model, ck, path = load_roll_checkpoint(spec)
    return model, path, int(ck["args"].get("size", 64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll-ckpt", type=str, required=True,
                    help="run directory or .pt of the roll branch (train_hfhpe_roll.py)")
    ap.add_argument("--body-ckpt", type=str, required=True,
                    help="run directory or .pt of the 6-output yaw/pitch body")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--out-dir", type=str, default="",
                    help="output directory (default: same folder as the body checkpoint)")
    ap.add_argument("--kappa-tol", type=float, default=1e-3,
                    help="torch vs ORT parity tolerance (absolute) for the κ output. "
                         "κ ranges to ~100, so relaxing to the 1e-2 level for the "
                         "GridSample+ViT composition is harmless in relative error")
    ap.add_argument("--cos-tol", type=float, default=5e-4,
                    help="parity tolerance for the cos_sin output. A randn dummy can "
                         "push the ViT into out-of-distribution activations and expose "
                         "fp32 kernel ordering differences (if real-image parity is at "
                         "the 1e-6 level, relaxing to 2e-3 is harmless)")
    ap.add_argument("--nbatch-atol", type=float, default=1e-4,
                    help="fixed→N-batch batch 1/2/3 validation tolerance")
    ap.add_argument("--stem", type=str, default="",
                    help="output file name stem (default hfhpe_<body arch name>; "
                         "e.g. hfhpe_vitt → hfhpe_vitt_1x3x64x64.onnx; "
                         "_rollgate is appended when --roll-gate is enabled)")
    ap.add_argument("--roll-gate", type=str, default="none",
                    choices=["none", "sigmoid", "relu"],
                    help="κ_roll gating: at low confidence, pull the in-graph "
                         "derotation toward the identity rotation (arithmetic only, "
                         "no Where). sigmoid = g=σ(a(κ−τ)) (effectively hard for "
                         "large a), relu = g=clip(a·relu(κ−τ),0,1) (piecewise "
                         "linear, quantization-friendly)")
    ap.add_argument("--roll-gate-tau", type=float, default=20.0,
                    help="gate threshold τ (in κ units; rule of thumb σ≈1/√κ rad. In "
                         "practice, derive it from the κ-error curve on val)")
    ap.add_argument("--roll-gate-a", type=float, default=0.0,
                    help="gate steepness a (0 = per-mode default: sigmoid 0.5 / "
                         "relu 1e6). For sigmoid, a large a is equivalent to hard")
    args = ap.parse_args()

    if args.roll_gate != "none" and args.roll_gate_a <= 0:
        args.roll_gate_a = 0.5 if args.roll_gate == "sigmoid" else 1e6

    rollnet, roll_path, roll_size = load_rollnet(args.roll_ckpt)
    body, body_path, size, body_stem, body_norm = load_model(args.body_ckpt, 0)
    if not getattr(body, "pitch_head", False):
        raise SystemExit("--body-ckpt must be a 6-output (yaw+pitch integrated) checkpoint")
    model = HFHPE(rollnet, body, body_norm=body_norm,
                  body_size=size, roll_size=roll_size,
                  gate_mode=args.roll_gate, gate_tau=args.roll_gate_tau,
                  gate_a=args.roll_gate_a).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())

    # default stem is hfhpe_<body arch name> (first token of the run name, e.g. vitt/dinov3)
    stem = args.stem or f"hfhpe_{body_stem.removesuffix('_yawpitch').split('_')[0]}"
    if args.roll_gate != "none" and not args.stem:
        stem += "_rollgate"    # do not overwrite the artifacts of the ungated version
    out_dir = Path(args.out_dir) if args.out_dir else body_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / f"{stem}_1x3x{size}x{size}_raw.onnx"
    fixed = out_dir / f"{stem}_1x3x{size}x{size}.onnx"
    nbatch = out_dir / f"{stem}_Nx3x{size}x{size}.onnx"

    gate_desc = ("none" if args.roll_gate == "none" else
                 f"{args.roll_gate} (tau={args.roll_gate_tau}, a={args.roll_gate_a:g})")
    print(f"roll: {roll_path} (size={roll_size})\n"
          f"body: {body_path} (size={size}, norm={body_norm})\n"
          f"onepass params={n_params:,} input=center05 "
          f"outputs=[cos_sin (N,6), kappa (N,3)] roll_gate={gate_desc}")
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        model, (dummy,), str(raw),
        opset_version=args.opset,
        input_names=["images"],
        output_names=["cos_sin", "kappa"],
        dynamo=False,
    )
    print(f"raw export: {raw} ({raw.stat().st_size/1e6:.1f} MB)")

    run([sys.executable, str(SKILL / "simplify_onnx.py"), str(raw), str(fixed)])

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(fixed), so, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        refs = [o.numpy() for o in model(dummy)]
    gots = sess.run(None, {"images": dummy.numpy()})
    for name, ref, got in zip(["cos_sin", "kappa"], refs, gots):
        tol = args.cos_tol if name == "cos_sin" else args.kappa_tol
        err = float(np.abs(ref - got).max())
        print(f"parity {name}: torch vs ORT max_err={err:.2e} (tol {tol:.0e})")
        assert err < tol, f"parity failed for {name}: {err:.2e}"

    run([sys.executable, str(SKILL / "fixed_to_nbatch.py"), str(fixed), str(nbatch),
         "--atol", str(args.nbatch_atol)])
    run([sys.executable, str(SKILL / "audit_onnx.py"), str(fixed),
         "--pair", str(nbatch), "--check", "--atol", str(args.nbatch_atol)])

    # inject the I/O contract metadata (HFHPE input is always center05). roll_gate keeps
    # the gate mode and τ, a so the variant is identifiable beyond the file name (_rollgate)
    from add_onnx_metadata import add_metadata  # noqa: PLC0415
    if args.roll_gate == "none":
        gate_meta = "none; in-graph derotation by the predicted roll is always applied"
    else:
        g_expr = ("sigmoid(a*(kappa_roll-tau))" if args.roll_gate == "sigmoid"
                  else "clip(a*relu(kappa_roll-tau),0,1)")
        gate_meta = (f"{args.roll_gate}; tau={args.roll_gate_tau:g}, a={args.roll_gate_a:g}; "
                     f"g = {g_expr}; derotation vector = normalize(g*(cos,sin)+(1-g)*(1,0)) "
                     "(g->0: identity); the roll output itself is not gated")
    for p in (fixed, nbatch):
        add_metadata(p, norm="center05", extra={"roll_gate": gate_meta})
    print(f"metadata: input/output contract injected (norm=center05, roll_gate={gate_desc})")

    raw.unlink(missing_ok=True)
    print(f"\nartifacts:\n  fixed : {fixed} ({fixed.stat().st_size/1e6:.1f} MB)\n"
          f"  nbatch: {nbatch} ({nbatch.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
