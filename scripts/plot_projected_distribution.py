#!/usr/bin/env python3
"""Plot the projected yaw distribution if augmented with synthetic_004 as a bar chart.

Reads data/yawpose/augmentation_plan.json (current counts and recommended additions)
and stacks the planned additions (synthetic_004 planned) on the current per-source stack.

Output: data/yawpose/yaw_hist_bar_projected.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "yawpose"
SOURCES = ["synthetic_001", "synthetic_002", "synthetic_003", "synthetic_004", "synthetic_005", "synthetic_006"]
COLORS = {"synthetic_001": "#4878cf", "synthetic_002": "#e8a33d",
          "synthetic_003": "#6acc65", "synthetic_004": "#b04a4a",
          "synthetic_005": "#9467bd", "synthetic_006": "#8c564b"}
PLANNED_COLOR = "#d65f5f"


def main() -> None:
    plan = json.load(open(OUT / "augmentation_plan.json"))
    target_name = plan["summary"]["recommended_target"]
    target = plan["summary"]["targets_per_bin"][target_name]
    bin_deg = plan["summary"]["bin_deg"]

    bins = plan["bins"]
    edges = np.array([b["yaw_from"] for b in bins])
    add = np.array([b["additional_needed"][target_name] for b in bins])
    cur_by_src = {s: np.array([b["current_by_source"][s] for b in bins]) for s in SOURCES}
    projected_total = int(sum(b["current"] for b in bins) + add.sum())

    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    bottom = np.zeros(len(bins))
    for s in SOURCES:
        ax.bar(edges, cur_by_src[s], width=bin_deg * 0.9, align="edge",
               bottom=bottom, label=s, color=COLORS[s], edgecolor="none")
        bottom += cur_by_src[s]
    ax.bar(edges, add, width=bin_deg * 0.9, align="edge", bottom=bottom,
           label="planned (to target floor)", color=PLANNED_COLOR,
           edgecolor="white", hatch="//", linewidth=0)
    ax.axhline(target, color="black", linestyle="--", linewidth=1,
               label=f"target floor = {target}/bin")
    ax.set_xlabel("yaw (deg)  [0 = frontal, +90 = facing viewer-left]")
    ax.set_ylabel("samples")
    ax.set_title("yawpose projected yaw distribution after planned additions "
                 f"({bin_deg}\N{DEGREE SIGN} bins, n={projected_total:,} = "
                 f"{plan['summary']['current_total']:,} + "
                 f"{int(add.sum()):,} planned)", pad=50)
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.grid(axis="y", alpha=0.3)
    # legend overlaps the bars, so place it in 2 rows above the plot area (under the title).
    # matplotlib fills column-major, so reorder the entries to read row by row
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    desired = [*SOURCES, "planned (to target floor)", f"target floor = {target}/bin"]
    ncol, nrow = 4, 2
    order = [desired[r * ncol + c] for c in range(ncol) for r in range(nrow)]
    ax.legend([by_label[l] for l in order], order, ncol=ncol,
              loc="lower center", bbox_to_anchor=(0.5, 1.0),
              frameon=False, fontsize=9, columnspacing=1.2, handlelength=1.4)
    fig.tight_layout()
    path = OUT / "yaw_hist_bar_projected.png"
    fig.savefig(path)
    plt.close(fig)
    print("saved:", path)


if __name__ == "__main__":
    main()
