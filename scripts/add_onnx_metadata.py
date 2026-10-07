#!/usr/bin/env python3
"""Inject a description of the I/O contract into the ONNX metadata_props.

The model kind (yaw / yaw+pitch / yaw+pitch+roll, with or without κ) is detected
automatically from the graph output shapes, and a concise English contract
description readable in Netron etc. is written. Existing keys with the same name
are overwritten. The input normalization is inferred from the file name (dinov3 →
imagenet, otherwise center05) and can be overridden explicitly with --norm.

export_onnx.py / export_hfhpe.py call this module's add_metadata() at the end of
the export, so manual runs are normally unnecessary. To retrofit existing files:
    uv run python scripts/add_onnx_metadata.py runs/<run>/*.onnx
"""
import argparse
from pathlib import Path

import onnx

ANGLE_CONVENTION = (
    "degrees; yaw: 0=frontal (facing camera), +90=turned to viewer's left, "
    "180=back of head, 270=viewer's right; pitch: 0=level, positive=looking "
    "up, negative=looking down; roll: 0=upright, positive=head rotates CCW "
    "in the image")

NORM_DESC = {
    "center05": "float32 RGB (N,3,{S},{S}); normalize: x/127.5 - 1 "
                "(x = pixel value 0..255)",
    "imagenet": "float32 RGB (N,3,{S},{S}); normalize: (x/255 - mean)/std, "
                "mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]",
}

COS_SIN_DESC = {
    2: "(N,2) [cos_yaw, sin_yaw]; unit vector; "
       "yaw_deg = degrees(atan2(sin, cos))",
    4: "(N,4) [cos_yaw, sin_yaw, cos_pitch, sin_pitch]; each pair is a unit "
       "vector; angle_deg = degrees(atan2(sin, cos))",
    6: "(N,6) [cos_yaw, sin_yaw, cos_pitch, sin_pitch, cos_roll, sin_roll]; "
       "each pair is a unit vector; angle_deg = degrees(atan2(sin, cos))",
}

KAPPA_DESC = {
    1: "(N) kappa_yaw; von Mises concentration (higher = more confident)",
    2: "(N,2) [kappa_yaw, kappa_pitch]; von Mises concentration "
       "(higher = more confident)",
    3: "(N,3) [kappa_yaw, kappa_pitch, kappa_roll]; von Mises concentration "
       "(higher = more confident)",
}


def _last_dim(vi: onnx.ValueInfoProto) -> int:
    dims = vi.type.tensor_type.shape.dim
    return int(dims[-1].dim_value) if dims and dims[-1].dim_value else 1


def add_metadata(path: Path, norm: str = "",
                 extra: dict[str, str] | None = None) -> dict[str, str]:
    """Write the contract metadata into the ONNX at path and return what was written.

    extra holds caller-specific additions (e.g. roll_gate from export_hfhpe.py).
    """
    model = onnx.load(str(path))
    outs = {o.name: o for o in model.graph.output}
    inp = model.graph.input[0]
    size = int(inp.type.tensor_type.shape.dim[-1].dim_value)
    if not norm:
        norm = "imagenet" if "dinov3" in path.name else "center05"

    meta = {"input.images": NORM_DESC[norm].format(S=size),
            "angle_convention": ANGLE_CONVENTION}
    if "cos_sin" in outs:
        meta["output.cos_sin"] = COS_SIN_DESC[_last_dim(outs["cos_sin"])]
    if "kappa" in outs:
        meta["output.kappa"] = KAPPA_DESC[_last_dim(outs["kappa"])]
    if extra:
        meta.update(extra)

    existing = {p.key: p for p in model.metadata_props}
    for k, v in meta.items():
        if k in existing:
            existing[k].value = v
        else:
            model.metadata_props.add(key=k, value=v)
    onnx.save(model, str(path))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", type=Path, nargs="+", help="target .onnx file(s)")
    ap.add_argument("--norm", type=str, default="",
                    choices=["", "center05", "imagenet"],
                    help="input normalization (default: inferred from the file name)")
    args = ap.parse_args()
    for path in args.models:
        meta = add_metadata(path, args.norm)
        print(f"{path.name}:")
        for k, v in meta.items():
            print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
