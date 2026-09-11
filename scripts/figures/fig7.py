r"""Fig. 7 -- Sampling, fusion, and total MSE as functions of normalized
axial spacing for the eight photographic conditions, focus window 7.
Sampling MSE compares I_or with I*, fusion MSE compares I_F with I_or,
and total MSE compares I_F with I*. The vertical dotted line marks
delta = 0.5. Frozen, unmodified numbers from
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


def main():
    by_cond = {c: [] for c in CONDS}
    with open(ROOT / "results" / "sampling_fusion" / "sampling_fusion_raw.csv") as f:
        for row in csv.DictReader(f):
            if row["condition"] in by_cond and int(row["window"]) == WINDOW:
                by_cond[row["condition"]].append(row)
    for c in CONDS:
        by_cond[c].sort(key=lambda r: float(r["delta"]))

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)

    for i, cond in enumerate(CONDS):
        ax = axes.flat[i]
        rows = by_cond[cond]
        deltas = [float(r["delta"]) for r in rows]
        sampling_mse = [float(r["sampling_mse"]) for r in rows]
        fusion_mse = [float(r["fusion_mse"]) for r in rows]
        total_mse = [float(r["total_mse"]) for r in rows]

        ax.plot(deltas, sampling_mse, color="#1f77b4", lw=1.2, label="sampling MSE")
        ax.plot(deltas, fusion_mse, color="#ff7f0e", lw=1.2, label="fusion MSE")
        ax.plot(deltas, total_mse, color="#d62728", lw=1.2, ls="--", label="total MSE")
        ax.axvline(0.5, color="gray", ls=":", lw=0.9, alpha=0.6)
        ax.set_yscale("log")
        ax.set_title(cond, fontsize=10)
        if i >= 4:
            ax.set_xlabel("δ")
        if i % 4 == 0:
            ax.set_ylabel("MSE (log scale)")
        panel_label(ax, "abcdefgh"[i], x=-0.14, y=1.12)

    axes.flat[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Sampling, fusion, and total MSE vs. axial spacing (window={WINDOW})",
                 fontsize=12)

    save(fig, "Figure7", dpi=1050)


if __name__ == "__main__":
    main()
