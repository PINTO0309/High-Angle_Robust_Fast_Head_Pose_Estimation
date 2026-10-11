#!/usr/bin/env python3
"""Style-shift evaluation of the yaw / pitch body ONNX files (the degradation set of HRFFA's Table 7).

Each validation crop is resized to the model input (INTER_LINEAR, like the validation
preprocessing of training) and then degraded in crop pixels before the yaw / pitch MAAE is
measured: clean / mblur9 / mblur21 (linear motion blur of 9 / 21 px at 30°) / warm / cool
(channel gains B,G,R = 0.8,1.0,1.2 and 1.2,1.0,0.8) / gamma0.6 / gamma1.6 / gray / jpeg30
(JPEG quality 30). Because the degradations are applied at the input resolution, their
strength depends on the input size (a 21 px blur on a 64 px input corresponds to about
105 px at 320); this matches the HRFFA protocol.

With several --onnx entries the script prints README-ready Markdown tables (value and the
change against clean in percent).

Usage:
    uv run python scripts/eval_style_shift.py \\
        --onnx vitt-064=runs/vitt_distill_064_yp_unified/hfhpe_vitt_yawpitch_Nx3x64x64.onnx \\
        --onnx hgnetv2-064=runs/hgnetv2_distill_064_yp_unified/hfhpe_hgnetv2_yawpitch_Nx3x64x64.onnx
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def style_shifts() -> list[tuple[str, object]]:
    def gains(b: float, g: float, r: float):
        def f(img):
            x = img.astype(np.float32) * np.array([b, g, r], np.float32)
            return x.clip(0, 255).astype(np.uint8)
        return f

    def gamma(g: float):
        def f(img):
            return (255.0 * (img.astype(np.float32) / 255.0) ** g).astype(np.uint8)
        return f

    def gray(img):
        return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    def jpeg30(img):
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 30])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    def mblur(length: int, angle_deg: float = 30.0):
        k = np.zeros((length, length), np.float32)
        c = length // 2
        th = np.deg2rad(angle_deg)
        for t in np.linspace(-c, c, length * 4):
            x, y = int(round(c + t * np.cos(th))), int(round(c + t * np.sin(th)))
            if 0 <= x < length and 0 <= y < length:
                k[y, x] = 1.0
        k /= max(k.sum(), 1.0)

        def f(img):
            return cv2.filter2D(img, -1, k)
        return f

    return [("clean", None), ("mblur9", mblur(9)), ("mblur21", mblur(21)),
            ("warm", gains(0.80, 1.00, 1.20)), ("cool", gains(1.20, 1.00, 0.80)),
            ("gamma0.6", gamma(0.6)), ("gamma1.6", gamma(1.6)),
            ("gray", gray), ("jpeg30", jpeg30)]


def wrap_abs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def evaluate(onnx_path: Path, crops: list[np.ndarray], gt: np.ndarray, batch: int,
             cpu: bool) -> dict[str, dict[str, float]]:
    import onnxruntime as ort  # noqa: PLC0415
    providers = (["CPUExecutionProvider"] if cpu
                 else ["CUDAExecutionProvider", "CPUExecutionProvider"])
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    size = int(sess.get_inputs()[0].shape[-1])
    fixed1 = isinstance(sess.get_inputs()[0].shape[0], int)
    bs = 1 if fixed1 else batch
    meta = sess.get_modelmeta().custom_metadata_map or {}
    imagenet = "mean=" in meta.get("input.images", "")
    mean = np.array([0.485, 0.456, 0.406], np.float32) * 255.0
    std = np.array([0.229, 0.224, 0.225], np.float32) * 255.0
    base = [cv2.resize(c, (size, size), interpolation=cv2.INTER_LINEAR) for c in crops]
    out: dict[str, dict[str, float]] = {}
    for name, fn in style_shifts():
        preds = np.zeros((len(base), 2))
        for i0 in range(0, len(base), bs):
            planes = []
            for im in base[i0:i0 + bs]:
                bgr = im if fn is None else fn(im)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
                rgb = (rgb - mean) / std if imagenet else rgb / 127.5 - 1.0
                planes.append(rgb.transpose(2, 0, 1))
            cs = sess.run(["cos_sin"], {"images": np.stack(planes)})[0]
            preds[i0:i0 + len(planes), 0] = np.degrees(np.arctan2(cs[:, 1], cs[:, 0]))
            preds[i0:i0 + len(planes), 1] = np.degrees(np.arctan2(cs[:, 3], cs[:, 2]))
        e = wrap_abs(preds, gt)
        out[name] = {"yaw": float(e[:, 0].mean()), "pitch": float(e[:, 1].mean())}
        print(f"  {onnx_path.stem:44s} {name:9s} yaw {out[name]['yaw']:6.2f}  "
              f"pitch {out[name]['pitch']:6.2f}", flush=True)
    return out


def cell(v: float, base: float) -> str:
    return f"{v:.2f}" if base == v else f"{v:.2f} ({(v / base - 1) * 100:+.1f}%)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", action="append", required=True,
                    help="body ONNX (cos_sin (N,4) or wider); name=path sets the row name; "
                         "repeatable")
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "yawpitchpose"))
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N rows (0 = all)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", type=str, default="", help="result JSON (not written when empty)")
    args = ap.parse_args()

    data = Path(args.data)
    rows = [json.loads(line) for line in open(data / f"{args.split}.jsonl")]
    rows = [r for r in rows if r.get("pitch_deg") is not None]
    if args.limit:
        rows = rows[:args.limit]
    crops = [cv2.imread(str(data / r["image"]), cv2.IMREAD_COLOR) for r in rows]
    gt = np.array([[r["yaw_deg"], r["pitch_deg"]] for r in rows], dtype=np.float64)
    print(f"{args.split}: n={len(rows)}")

    results: dict[str, dict] = {}
    for spec in args.onnx:
        name, _, path = spec.partition("=") if "=" in spec else ("", "", spec)
        p = Path(path)
        name = name or p.parent.name
        results[name] = evaluate(p, crops, gt, args.batch, args.cpu)

    names = [n for n, _ in style_shifts()]
    for metric in ("yaw", "pitch"):
        print(f"\n#### style-shift: {metric} MAAE (deg) under image degradations "
              f"(in parentheses: degradation vs clean, {args.split} n={len(rows)})")
        print("| Model | " + " | ".join(names) + " |")
        print("|---|" + "--:|" * len(names))
        for model, res in results.items():
            base = res["clean"][metric]
            print(f"| {model} | " + " | ".join(cell(res[n][metric], base) for n in names) + " |")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"split": args.split, "n": len(rows), "results": results}, f, indent=1)
        print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
