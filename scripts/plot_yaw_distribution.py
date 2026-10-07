#!/usr/bin/env python3
"""Visualize the yaw distribution of the yawpose dataset.

Aggregates data/yawpose/labels_fixed.jsonl (finalized labels) into 10° bins and
writes a per-source stacked bar chart, a polar chart, and an aggregate JSON:
  - data/yawpose/yaw_hist_bar.png
  - data/yawpose/yaw_hist_polar.png
  - data/yawpose/yaw_distribution.json (per-bin total / per-source / per-split
    counts and summary statistics)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "yawpose"
BIN_DEG = 10
SOURCES = ["synthetic_001", "synthetic_002", "synthetic_003", "synthetic_004", "synthetic_005", "synthetic_006"]
COLORS = {"synthetic_001": "#4878cf", "synthetic_002": "#e8a33d",
          "synthetic_003": "#6acc65", "synthetic_004": "#d65f5f",
          "synthetic_005": "#9467bd", "synthetic_006": "#8c564b"}


def load_counts() -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    n_bins = 360 // BIN_DEG
    counts = {s: np.zeros(n_bins, dtype=int) for s in SOURCES}
    split_counts: dict[str, np.ndarray] = {}
    with open(OUT / "labels_fixed.jsonl") as f:
        for line in f:
            r = json.loads(line)
            counts[r["source"]][int(r["yaw_deg"] % 360.0 // BIN_DEG)] += 1
    for split in ("train", "val"):
        split_counts[split] = np.zeros(n_bins, dtype=int)
        with open(OUT / f"{split}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                split_counts[split][int(r["yaw_deg"] % 360.0 // BIN_DEG)] += 1
    edges = np.arange(0, 360, BIN_DEG)
    return edges, counts, split_counts


def write_json(edges: np.ndarray, counts: dict[str, np.ndarray],
               split_counts: dict[str, np.ndarray], total: int) -> None:
    total_bins = sum(counts.values())
    bins = [{
        "yaw_from": int(lo),
        "yaw_to": int(lo) + BIN_DEG,
        "total": int(total_bins[i]),
        "by_source": {s: int(counts[s][i]) for s in SOURCES},
        "by_split": {sp: int(split_counts[sp][i]) for sp in split_counts},
    } for i, lo in enumerate(edges)]
    nonzero = total_bins[total_bins > 0]
    summary = {
        "total": total,
        "bin_deg": BIN_DEG,
        "by_source": {s: int(counts[s].sum()) for s in SOURCES},
        "by_split": {sp: int(c.sum()) for sp, c in split_counts.items()},
        "min_bin": {"yaw_from": int(edges[int(np.argmin(total_bins))]),
                    "count": int(total_bins.min())},
        "max_bin": {"yaw_from": int(edges[int(np.argmax(total_bins))]),
                    "count": int(total_bins.max())},
        "mean_per_bin": round(float(total_bins.mean()), 1),
        "imbalance_max_over_min_nonzero": round(float(total_bins.max() / nonzero.min()), 1),
        "source_file": "labels_fixed.jsonl",
    }
    with open(OUT / "yaw_distribution.json", "w") as f:
        json.dump({"summary": summary, "bins": bins}, f, indent=2, ensure_ascii=False)


def plot_bar(edges: np.ndarray, counts: dict[str, np.ndarray], total: int) -> None:
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    bottom = np.zeros(len(edges))
    for s in SOURCES:
        ax.bar(edges, counts[s], width=BIN_DEG * 0.9, align="edge",
               bottom=bottom, label=s, color=COLORS[s], edgecolor="none")
        bottom += counts[s]
    ax.set_xlabel("yaw (deg)  [0 = frontal, +90 = facing viewer-left]")
    ax.set_ylabel("samples")
    ax.set_title(f"yawpose yaw distribution ({BIN_DEG}\N{DEGREE SIGN} bins, n={total:,})")
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "yaw_hist_bar.png")
    plt.close(fig)


def plot_polar(edges: np.ndarray, counts: dict[str, np.ndarray], total: int) -> None:
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")   # 0° = top (frontal)
    ax.set_theta_direction(-1)        # angles increase clockwise
    theta = np.radians(edges + BIN_DEG / 2)
    width = np.radians(BIN_DEG) * 0.95
    bottom = np.zeros(len(edges))
    for s in SOURCES:
        ax.bar(theta, counts[s], width=width, bottom=bottom,
               label=s, color=COLORS[s], edgecolor="none", alpha=0.95)
        bottom += counts[s]
    ax.set_xticks(np.radians(np.arange(0, 360, 30)))
    ax.set_xticklabels([f"{d}\N{DEGREE SIGN}" for d in range(0, 360, 30)])
    ax.set_title(f"yawpose yaw distribution (polar, {BIN_DEG}\N{DEGREE SIGN} bins, n={total:,})",
                 pad=20)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.1, -0.1))
    fig.tight_layout()
    fig.savefig(OUT / "yaw_hist_polar.png")
    plt.close(fig)


def main() -> None:
    edges, counts, split_counts = load_counts()
    total = int(sum(c.sum() for c in counts.values()))
    plot_bar(edges, counts, total)
    plot_polar(edges, counts, total)
    write_json(edges, counts, split_counts, total)
    print(f"saved: {OUT/'yaw_hist_bar.png'}")
    print(f"saved: {OUT/'yaw_hist_polar.png'}")
    print(f"saved: {OUT/'yaw_distribution.json'}")


if __name__ == "__main__":
    main()
