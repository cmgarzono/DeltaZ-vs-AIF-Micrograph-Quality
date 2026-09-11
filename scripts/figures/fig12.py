r"""Fig. 12 -- Dense axial sampling: quantitative redundancy summary.
Condition H_MIXED, window=7, frozen
results/sampling_fusion/sampling_fusion_raw.csv, full delta range. Panel
(a): candidate count K vs. effective evidence count N_eff. Panel (b):
normalized redundancy R_Z vs. final total image quality (MSE). Dotted
vertical lines mark the three regimes shown in Fig. 11 (delta=0.50, 0.15,
0.05). Frozen, unmodified numbers -- no new metric."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import csv
import matplotlib.pyplot as plt
from common import ROOT, save, panel_label

COND = "H_MIXED"
WINDOW = 7
DENSITIES = [("moderate", 0.50), ("dense", 0.15), ("very_dense", 0.05)]


def main():
    rows = []
    with open(ROOT / "results" / "sampling_fusion" / "sampling_fusion_raw.csv") as f:
        for row in csv.DictReader(f):
            if row["condition"] == COND and int(row["window"]) == WINDOW:
                rows.append(row)
    rows.sort(key=lambda r: float(r["delta"]))
    deltas = [float(r["delta"]) for r in rows]
    K = [float(r["K"]) for r in rows]
    Neff = [float(r["N_eff"]) for r in rows]
    RZ = [float(r["R_Z"]) for r in rows]
    total_mse = [float(r["total_mse"]) for r in rows]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

    ax.plot(deltas, K, label="K (candidate count)", color="#555")
    ax.plot(deltas, Neff, label=r"N$_{eff}$ (effective evidence)", color="#1f77b4")
    for _, d in DENSITIES:
        ax.axvline(d, color="gray", ls=":", lw=0.9, alpha=0.6)
    ax.set_xlabel("δ"); ax.set_ylabel("count"); ax.set_xscale("log")
    ax.legend(fontsize=9, loc="upper right")
    panel_label(ax, "a", x=-0.12, y=1.10)
    ax.set_title(f"{COND}, window={WINDOW}", fontsize=11, pad=12)

    ax2.plot(deltas, RZ, color="#d62728", label=r"R$_Z$ (redundancy)")
    ax2b = ax2.twinx()
    ax2b.plot(deltas, total_mse, color="#2ca02c", label="total MSE (quality)")
    ax2b.set_yscale("log")
    for _, d in DENSITIES:
        ax2.axvline(d, color="gray", ls=":", lw=0.9, alpha=0.6)
    ax2.set_xlabel("δ"); ax2.set_ylabel(r"R$_Z$", color="#d62728")
    ax2b.set_ylabel("total MSE", color="#2ca02c")
    ax2.set_xscale("log")
    panel_label(ax2, "b", x=-0.12, y=1.10)
    ax2.set_title(f"{COND}, window={WINDOW}", fontsize=11, pad=12)

    save(fig, "Figure12", dpi=1050)


if __name__ == "__main__":
    main()
