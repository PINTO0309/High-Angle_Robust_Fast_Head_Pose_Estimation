#!/usr/bin/env python3
"""Write a distribution copy of a checkpoint with the training-environment details removed.

The training scripts store, next to the weights, the full argument list (`args`) and,
for distilled students, a `teacher` dict. Together they record dataset roots, extra data
roots, teacher / initial-weight run names, absolute paths of the training machine and all
hyperparameters in plain text. This script writes a copy that keeps only the keys the
loaders read (export_onnx.py / export_hfhpe.py / roll_models.py /
distill_yawnet.py --init-student). The original file is left untouched.

Kept:
  - top level: model, model_type, variant, kappa_head, pitch_head, params, weights,
    epoch, metrics (drop with --drop-metrics)
  - args: size, student_size, student_arch, arch, width, variant, kappa_head, pitch_head
Dropped: every other key (data / extra_data / teacher / init_student / init_ckpt /
backbone_ckpt / dinov3_ckpt / tag / training hyperparameters / optimizer state ...).

After writing, the script checks that (1) no path-like string remains, (2) the file loads
with weights_only=True and (3) the matching loader (load_model / load_roll_checkpoint) can
rebuild the model. The default output is runs/public/<run name>/<same file name>: the
directory keeps the run name, so `--ckpt runs/public/<run>` exports ONNX files with the
same stem and `--init-student runs/public/<run>` works as before.

Usage:
    uv run python scripts/sanitize_checkpoint.py runs/vitt_distill_064_yp_unified_v9 \\
        runs/hfhpe_roll_064_unified_kd_vitl_canvas_w100 [--out-dir DIR] [--drop-metrics]
"""
import argparse
import re
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent

KEEP_TOP = ("model", "model_type", "variant", "kappa_head", "pitch_head", "params",
            "weights", "epoch", "metrics", "args")
KEEP_ARGS = ("size", "student_size", "student_arch", "arch", "width", "variant",
             "kappa_head", "pitch_head")
# refuse to write when any remaining string value still looks like a path (keys are not checked)
FORBIDDEN = re.compile(r"data/|runs/|ckpts|/root|/home/|/mnt/|[A-Za-z]:\\|\.pth?$", re.IGNORECASE)


def resolve(spec: str) -> Path:
    path = Path(spec)
    if path.is_dir():
        cands = sorted(path.glob("best_*.pt"))
        if not cands:
            raise FileNotFoundError(f"no best_*.pt found in {path}")
        return cands[0]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def iter_strings(obj: Any, prefix: str = ""):
    """Yield (location, value) for every string inside nested dicts / lists (tensors excluded)."""
    if isinstance(obj, str):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_strings(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from iter_strings(v, f"{prefix}[{i}]")


def sanitize(ck: dict[str, Any], drop_metrics: bool) -> tuple[dict[str, Any], list[str], list[str]]:
    """Return the filtered dict plus the dropped top-level keys and dropped args keys."""
    keep_top = [k for k in KEEP_TOP if not (drop_metrics and k == "metrics")]
    clean: dict[str, Any] = {k: ck[k] for k in keep_top if k in ck}
    dropped_top = [k for k in ck if k not in clean]
    dropped_args: list[str] = []
    if isinstance(ck.get("args"), dict):
        args = ck["args"]
        clean["args"] = {k: args[k] for k in KEEP_ARGS if k in args}
        dropped_args = [k for k in args if k not in clean["args"]]
    elif "args" in ck:   # argparse.Namespace or similar
        args = vars(ck["args"])
        clean["args"] = {k: args[k] for k in KEEP_ARGS if k in args}
        dropped_args = [k for k in args if k not in clean["args"]]
    return clean, dropped_top, dropped_args


def check_loadable(path: Path, model_type: str) -> str:
    """Rebuild the model with the loader that matches the checkpoint type; return a summary."""
    sys.path.insert(0, str(ROOT / "scripts"))
    if model_type == "hfhpe_roll":
        from roll_models import load_roll_checkpoint  # noqa: PLC0415
        model, ck, _ = load_roll_checkpoint(str(path))
        return (f"load_roll_checkpoint OK (arch={ck['args'].get('arch', 'yawnet')}, "
                f"size={ck['args'].get('size')})")
    from export_onnx import load_model  # noqa: PLC0415
    model, _, size, stem, norm = load_model(str(path), 0)
    return f"load_model OK (size={size}, norm={norm}, stem={stem})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+",
                    help="run directories (best_*.pt is picked) or .pt files")
    ap.add_argument("--out-dir", type=str, default=str(ROOT / "runs" / "public"),
                    help="output root; a <run name>/ directory is created inside it "
                         "(default runs/public)")
    ap.add_argument("--drop-metrics", action="store_true",
                    help="also drop the validation metrics")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the load checks after writing")
    args = ap.parse_args()

    for spec in args.ckpts:
        src = resolve(spec)
        out_dir = Path(args.out_dir) / src.parent.name   # keep the run name (ONNX stem)
        out = out_dir / src.name
        if out.resolve() == src.resolve():
            raise SystemExit(f"output path equals the input: {out}")
        ck = torch.load(src, map_location="cpu", weights_only=False)
        clean, dropped_top, dropped_args = sanitize(ck, args.drop_metrics)

        bad = [(where, val) for where, val in iter_strings({k: v for k, v in clean.items()
                                                             if k != "model"})
               if FORBIDDEN.search(val)]
        if bad:
            for where, val in bad:
                print(f"  remaining: {where} = {val!r}")
            raise SystemExit(f"{src}: path-like strings remain, nothing written")

        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(clean, out)
        reloaded = torch.load(out, map_location="cpu", weights_only=True)   # safe-load check
        n_tensors = len(reloaded["model"])
        print(f"{src} -> {out}")
        print(f"  size {src.stat().st_size/1e6:.1f} MB -> {out.stat().st_size/1e6:.1f} MB, "
              f"tensors {n_tensors}, model_type={clean.get('model_type')}")
        print(f"  kept top-level: {list(clean.keys())}")
        print(f"  kept args     : {clean.get('args')}")
        print(f"  dropped top-level: {dropped_top}")
        print(f"  dropped args     : {dropped_args}")
        if not args.no_check:
            print(f"  check: {check_loadable(out, str(clean.get('model_type')))}")


if __name__ == "__main__":
    main()
