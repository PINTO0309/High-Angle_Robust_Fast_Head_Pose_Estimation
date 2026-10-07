#!/usr/bin/env python3
"""Verify left/right of the dataset's current yaw with the auto_qa direction detector.

Generates the suspect list (qa_<tag>_direction_suspects.json) read by
fix_mirror_by_direction.py. The decision is made against the dataset's current
yaw_deg rather than the generation intent, so already-fixed rows return to the
"match" side and a rerun leaves only newly missed rows (idempotent).

The direction convention is subject-side (the subject's own left/right):
  - yawpose yaw 0..180 (viewer-left = subject turned right) → right_* is correct
  - yawpose yaw 180..360 → left_* is correct
The decidable band is |signed(yaw)| 25°-170°. Back of the head ±10° (170-190) is
excluded because the detector's verdict is inherently ambiguous there too (a band
where even a flip costs <20° of error, so little real harm).
Verdicts of only front/back (no left/right) are treated as neutral and not counted.

This signal is independent of the s007 oracle method (screen_s007_oracle.py), so it
can also be used on sources the oracle was trained on (s002..s006). If a row that was
already flip-fixed comes out as a mismatch, only a warning is printed (no automatic
rollback; confirm visually).

Usage:
    uv run python scripts/screen_mirror_by_direction.py \
        --qa data/synthetic_005/auto_qa.jsonl \
        --source synthetic_005 --prefix s005_
Then:
    uv run python scripts/fix_mirror_by_direction.py \
        --suspects data/yawpitchpose/qa_s005_direction_suspects.json \
        --source synthetic_005 --prefix s005_
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def signed(d: float) -> float:
    return ((d + 180.0) % 360.0) - 180.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", type=str, required=True,
                    help="the source's auto_qa.jsonl")
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "yawpitchpose"))
    ap.add_argument("--source", type=str, required=True,
                    help="source name in labels (e.g. synthetic_005)")
    ap.add_argument("--prefix", type=str, required=True,
                    help="image name prefix within the dataset (e.g. s005_)")
    ap.add_argument("--band", type=float, nargs=2, default=(25.0, 170.0),
                    help="[min max] of the decidable band |signed(yaw)|")
    args = ap.parse_args()

    data = Path(args.data)
    tag = args.prefix.rstrip("_")

    qa = {}
    for l in open(args.qa):
        r = json.loads(l)
        if r.get("image_valid", True) and r.get("direction"):
            qa[r["filename"]] = r

    split_of = {}
    for name in ("train", "val"):
        for l in open(data / f"{name}.jsonl"):
            r = json.loads(l)
            split_of[r["image"]] = name

    stat: Counter = Counter()
    suspects = []
    fixed_mismatch = []
    for l in open(data / "labels_fixed.jsonl"):
        row = json.loads(l)
        if row["source"] != args.source:
            continue
        fn = row["image"].split("/")[-1]
        fn = fn[len(args.prefix):] if fn.startswith(args.prefix) else fn
        r = qa.get(fn)
        if r is None:
            stat["no qa"] += 1
            continue
        y = float(row["yaw_deg"]) % 360.0
        s = signed(y)
        if not args.band[0] <= abs(s) <= args.band[1]:
            stat["out of band (near frontal/back)"] += 1
            continue
        d = r["direction"]
        det = ("right" if d.startswith("right")
               else "left" if d.startswith("left") else None)
        if det is None:
            stat["neutral (front/back)"] += 1
            continue
        exp = "right" if s > 0 else "left"
        if det == exp:
            stat["match"] += 1
            continue
        item = {"file": fn, "current_yaw": y, "direction": d,
                "score": float(r.get("direction_score") or 0),
                "split": split_of.get(row["image"], "?"),
                "already_fixed": "yaw_deg_orig_mirror" in row}
        if item["already_fixed"]:
            stat["mismatch (already fixed)"] += 1
            fixed_mismatch.append(item)
        else:
            stat["mismatch (new)"] += 1
            suspects.append(item)

    print(f"{args.source}: {dict(stat)}")
    if fixed_mismatch:
        print(f"warning: {len(fixed_mismatch)} already flip-fixed rows disagree with "
              "the detector. Do not roll back automatically; confirm visually:")
        for it in fixed_mismatch[:10]:
            print(f"  {it['file']} yaw={it['current_yaw']} "
                  f"det={it['direction']}({it['score']:.2f})")

    out = data / f"qa_{tag}_direction_suspects.json"
    if out.exists():
        prev = out.with_name(out.stem + "_prev.json")
        out.rename(prev)
        print(f"backed up existing json: {prev}")
    with open(out, "w") as f:
        json.dump(suspects, f, indent=1)
    print(f"saved: {len(suspects)} suspects -> {out}")
    if suspects:
        print("next: uv run python scripts/fix_mirror_by_direction.py "
              f"--suspects {out} --source {args.source} --prefix {args.prefix}")


if __name__ == "__main__":
    main()
