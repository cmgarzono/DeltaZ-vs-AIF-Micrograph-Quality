r"""Fig. 13 -- Axial redundancy across the eight photographic conditions,
focus window 7. Candidate count K is common to all conditions (set by the
axial grid); effective evidence count N_eff, normalized redundancy R_Z,
and mean adjacent evidence overlap show the condition-dependent response
of focal evidence as delta changes. Frozen, unmodified numbers from
results/sampling_fusion/sampling_fusion_raw.csv -- no new metric."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import csv
import matplotlib.pyplot as plt
from common import ROOT, save, panel_label

WINDOW = 7
CONDS = ["H_LF", "H_COARSE", "H_MIXED", "H_FINE",
         "P_LF", "P_COARSE", "P_MIXED", "P_FINE"]
COLORS = dict(zip(CONDS, plt.cm.tab10.colors[:8]))

METRICS = [
    ("K", "candidate count K"),
    ("N_eff", r"effective evidence count N$_{eff}$"),
    ("R_Z", r"normalized redundancy R$_Z$"),
    ("adjacent_overlap_mean", "mean adjacent evidence overlap"),
]


def main():
    by_cond = {c: [] for c in CONDS}
    with open(ROOT / "results" / "sampling_fusion" / "sampling_fusion_raw.csv") as f:
        for row in csv.DictReader(f):
            if row["condition"] in by_cond and int(row["window"]) == WINDOW:
                by_cond[row["condition"]].append(row)
    for c in CONDS:
        by_cond[c].sort(key=lambda r: float(r["delta"]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    for mi, (metric, label) in enumerate(METRICS):
        ax = axes.flat[mi]
        for cond in CONDS:
            rows = by_cond[cond]
            deltas = [float(r["delta"]) for r in rows]
            vals = [float(r[metric]) for r in rows]
            ax.plot(deltas, vals, color=COLORS[cond], lw=0.9, alpha=0.85, label=cond)
        if metric == "K":
            ax.set_yscale("log")
        ax.set_xlabel("δ")
        ax.set_title(label, fontsize=10)
        panel_label(ax, "abcd"[mi], x=-0.10, y=1.08)

    axes.flat[0].legend(fontsize=7, ncol=2, loc="upper right")
    fig.suptitle(f"Axial redundancy across the eight photographic conditions (window={WINDOW})",
                 fontsize=12)

    save(fig, "Figure13", dpi=1050)


if __name__ == "__main__":
    main()
