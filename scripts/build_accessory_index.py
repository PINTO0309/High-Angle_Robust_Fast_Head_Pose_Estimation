#!/usr/bin/env python3
"""Generate the accessory-wearing index data/<dataset>/accessory_index.json.

Picks up the images wearing accessories from each source's generation metadata
and writes a dict of {image name (file name directly under images/): description}:
  - synthetic_001      : mask_description in annotations.jsonl (mask worn)
  - synthetic_004..007 : headwear / accessory in generation_plan.jsonl
    (other than "none")
  - synthetic_002/003  : no accessory information in the metadata (not covered)

Referenced by the diverse-sample selection (accessory slot) of render_preview.py.
Usage:
    uv run python scripts/build_accessory_index.py --data data/yawpitchpose
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "yawpitchpose"))
    args = ap.parse_args()

    index: dict[str, str] = {}

    src = ROOT / "data" / "synthetic_001"
    for r in load_jsonl(src / "annotations.jsonl"):
        desc = r.get("mask_description")
        if desc in (None, "", "none"):
            continue
        index[f"s001_{Path(r['image']).name}"] = f"mask: {desc}"

    for i in (4, 5, 6, 7):
        src = ROOT / "data" / f"synthetic_00{i}"
        if not src.exists():
            continue
        for r in load_jsonl(src / "generation_plan.jsonl"):
            parts = [f"{k}: {r[k]}" for k in ("headwear", "accessory")
                     if r.get(k) not in (None, "", "none")]
            if parts:
                index[f"s00{i}_{r['filename']}"] = "; ".join(parts)

    out = Path(args.data) / "accessory_index.json"
    with open(out, "w") as f:
        json.dump(index, f, ensure_ascii=False)
    print(f"saved: {out} ({len(index):,} images with accessories)")


if __name__ == "__main__":
    main()
