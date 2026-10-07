#!/usr/bin/env python3
"""Render the distributions of the yawpitchpose dataset as one image.

Three panels:
  1. yaw distribution (10° bins, stacked by source)
  2. pitch distribution (10° bins, stacked by source; s001 is a weak label
     recovered from the file-name cam token)
  3. yaw × pitch 2D coverage (10° × 10° cells, log color)

Output: data/yawpitchpose/distribution.png
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "yawpitchpose"
SOURCES = [f"synthetic_00{i}" for i in range(1, 8)]
COLORS = {"synthetic_001": "#4878cf", "synthetic_002": "#e8a33d",
          "synthetic_003": "#6acc65", "synthetic_004": "#d65f5f",
          "synthetic_005": "#9467bd", "synthetic_006": "#8c564b",
          "synthetic_007": "#e377c2"}


def stacked_hist(ax: plt.Axes, values: dict[str, np.ndarray], bins: np.ndarray,
                 title: str, xlabel: str) -> None:
    bottom = np.zeros(len(bins) - 1)
    width = (bins[1] - bins[0]) * 0.9
    for s in SOURCES:
        h, _ = np.histogram(values[s], bins=bins)
        ax.bar(bins[:-1], h, width=width, align="edge", bottom=bottom,
               color=COLORS[s], label=s, edgecolor="none")
        bottom += h
    n = int(sum(len(v) for v in values.values()))
    ax.set_title(f"{title} (n={n:,})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("samples")
    ax.grid(axis="y", alpha=0.3)


def main() -> None:
    rows = [json.loads(l) for l in open(OUT / "labels_fixed.jsonl")]

    yaw: dict[str, list[float]] = defaultdict(list)
    pitch: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        yaw[r["source"]].append(r["yaw_deg"] % 360.0)
        pitch[r["source"]].append(float(r["pitch_deg"]))
    yaw_a = {s: np.array(yaw[s]) for s in SOURCES}
    pitch_a = {s: np.array(pitch[s]) for s in SOURCES}

    fig, axes = plt.subplots(3, 1, figsize=(14, 15), dpi=150,
                             gridspec_kw={"height_ratios": [1, 1, 1.35]})

    stacked_hist(axes[0], yaw_a, np.arange(0, 361, 10),
                 "yaw (labels_fixed)",
                 "yaw (deg)  [0 = frontal, +90 = facing viewer-left]")
    axes[0].set_xlim(0, 360)
    axes[0].set_xticks(np.arange(0, 361, 30))
    axes[0].set_title(axes[0].get_title(), pad=34)
    axes[0].legend(ncol=len(SOURCES), loc="lower center",
                   bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9.5,
                   columnspacing=1.1, handlelength=1.3)

    stacked_hist(axes[1], pitch_a, np.arange(-120, 121, 10),
                 "pitch (intent for s002-s007, filename-cam recovery for s001)",
                 "pitch (deg)  [+ = looking up, - = viewed from above]")
    axes[1].set_xlim(-120, 120)
    axes[1].set_xticks(np.arange(-120, 121, 30))

    # yaw x pitch 2D coverage (all sources combined, 10x10 degree cells)
    ax = axes[2]
    all_yaw = np.concatenate([yaw_a[s] for s in SOURCES])
    all_pitch = np.concatenate([pitch_a[s] for s in SOURCES])
    y_edges = np.arange(0, 361, 10)
    p_edges = np.arange(-120, 121, 10)
    grid, _, _ = np.histogram2d(all_yaw, all_pitch, bins=[y_edges, p_edges])
    masked = np.ma.masked_equal(grid.T, 0)   # zero cells in white to make the "holes" visible
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    mesh = ax.pcolormesh(y_edges, p_edges, masked, cmap=cmap,
                         norm=LogNorm(vmin=1, vmax=masked.max()))
    ax.set_title(f"yaw \N{MULTIPLICATION SIGN} pitch coverage "
                 f"(10\N{DEGREE SIGN}\N{MULTIPLICATION SIGN}"
                 f"10\N{DEGREE SIGN} cells, log color; white = empty)")
    ax.set_xlabel("yaw (deg)  [0 = frontal, +90 = facing viewer-left]")
    ax.set_ylabel("pitch (deg)  [+ = up]")
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.set_yticks(np.arange(-120, 121, 30))
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.colorbar(mesh, ax=ax, pad=0.01, label="samples / cell")

    fig.suptitle(f"yawpitchpose yaw / pitch distributions (n={len(rows):,})",
                 y=0.997, fontsize=13.5)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = OUT / "distribution.png"
    fig.savefig(path)
    plt.close(fig)
    print("saved:", path)


if __name__ == "__main__":
    main()
