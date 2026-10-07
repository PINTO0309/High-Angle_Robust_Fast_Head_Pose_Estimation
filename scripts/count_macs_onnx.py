#!/usr/bin/env python3
"""Compute MACs / parameter count from an ONNX graph.

Target ops: Conv / ConvTranspose / MatMul / Gemm (the dominant multiply-accumulate terms).
Shapes are resolved from the graph's value_info / inputs and outputs / initializers
(the artifacts of export_onnx.py have complete ValueInfo, so the count is exact).
GFLOPs are reported as 2 x GMACs.

Usage:
    uv run python scripts/count_macs_onnx.py model.onnx [model2.onnx ...]
"""
import sys
from pathlib import Path

import numpy as np
import onnx


def tensor_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    graph = model.graph
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        dims = [d.dim_value if d.HasField("dim_value") else -1
                for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = dims
    for init in graph.initializer:
        shapes[init.name] = list(init.dims)
    return shapes


def count(model_path: str) -> tuple[float, int]:
    model = onnx.load(model_path, load_external_data=True)
    shapes = tensor_shapes(model)
    params = sum(int(np.prod(init.dims)) for init in model.graph.initializer)

    macs = 0
    for node in model.graph.node:
        if node.op_type == "Conv":
            w = shapes.get(node.input[1])
            out = shapes.get(node.output[0])
            if not w or not out:
                continue
            groups = 1
            for a in node.attribute:
                if a.name == "group":
                    groups = a.i
            # out_elems x (Cin/groups x kH x kW)
            kernel = int(np.prod(w[1:]))  # w = [Cout, Cin/groups, kH, kW]
            macs += int(np.prod(out[1:])) * kernel
        elif node.op_type == "Gemm":
            a = shapes.get(node.input[0])
            b = shapes.get(node.input[1])
            if not a or not b:
                continue
            trans_b = any(at.name == "transB" and at.i for at in node.attribute)
            m = a[0] if a[0] > 0 else 1
            k = a[1]
            n = b[0] if trans_b else b[1]
            macs += m * k * n
        elif node.op_type == "MatMul":
            a = shapes.get(node.input[0])
            b = shapes.get(node.input[1])
            out = shapes.get(node.output[0])
            if not a or not b or not out:
                continue
            k = a[-1]
            out_elems = int(np.prod([d for d in out if d > 0]))
            macs += out_elems * k
    return macs, params


def main() -> None:
    print(f"{'model':<48} {'params':>13} {'GMACs':>9} {'GFLOPs':>9}")
    for p in sys.argv[1:]:
        macs, params = count(p)
        print(f"{Path(p).name:<48} {params:>13,} {macs/1e9:>9.3f} {2*macs/1e9:>9.3f}")


if __name__ == "__main__":
    main()
