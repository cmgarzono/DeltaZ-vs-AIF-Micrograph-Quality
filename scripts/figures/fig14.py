r"""Fig. 14 -- Noise and operator-dependent multiplicity. Frozen H_MIXED
400x400 crop, sigma=0.02. Photographic grid: rows = {single, mean,
max-focus w=7}, columns = {N=1, N=32, N=128}, clean reference shown once,
LARGE panels. Quantitative curves (full frozen N range, mean over the 100
frozen replicates): MSE, Laplacian variance, DCT high-frequency fraction,
for single/mean/max-focus(w=7), from
results/multiplicity_noise/multiplicity_noise_raw.csv."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from common import ROOT, VEB, save, panel_label, imshow_gray

SIGMA = 0.02
NS_SHOWN = [1, 32, 128]
RULES = [("single", "single"), ("mean", "mean"), ("maxfocus_w7", "max-focus (w=7)")]


def main():
    N9 = VEB / "09_noise_multiplicity"
    fig = plt.figure(figsize=(13, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.0])
    gs_img = gs[0].subgridspec(3, 4, wspace=0.06, hspace=0.22)

    clean = mpimg.imread(N9 / "clean_reference.png")
    ax = fig.add_subplot(gs_img[:, 0])
    ax.imshow(clean, cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("I$_{clean}$", fontsize=11)
    panel_label(ax, "a")

    for ri, (rk, rlabel) in enumerate(RULES):
        for ci, N in enumerate(NS_SHOWN):
            f = N9 / f"{rk}_sigma0.02_N{N:03d}_rep0.png"
            ax = fig.add_subplot(gs_img[ri, ci + 1])
            img = mpimg.imread(f)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"N={N}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(rlabel, fontsize=10, fontweight="bold")

    # quantitative curves
    s4rows = []
    with open(ROOT / "results" / "multiplicity_noise" / "multiplicity_noise_raw.csv") as f:
        for row in csv.DictReader(f):
            if abs(float(row["sigma"]) - SIGMA) < 1e-9:
                s4rows.append(row)

    def agg(rule, window=None):
        by_n = defaultdict(list)
        for r in s4rows:
            if r["rule"] != rule:
                continue
            if window is not None and r["window"] != str(window):
                continue
            by_n[int(r["N"])].append(r)
        Ns_sorted = sorted(by_n)
        mse = [np.mean([float(x["MSE"]) for x in by_n[n]]) for n in Ns_sorted]
        lap = [np.mean([float(x["laplacian_variance"]) for x in by_n[n]]) for n in Ns_sorted]
        dct = [np.mean([float(x["dct_hf_energy"]) for x in by_n[n]]) for n in Ns_sorted]
        return Ns_sorted, mse, lap, dct

    series = {"single": agg("single"), "mean": agg("mean"), "max-focus w=7": agg("maxfocus", 7)}
    metrics = [("MSE", 1, "log"), ("Laplacian variance", 2, "linear"), ("DCT high-freq fraction", 3, "linear")]
    letters = "bcd"
    gs_q = gs[1].subgridspec(1, 3, wspace=0.3)
    for mi, (mname, midx, yscale) in enumerate(metrics):
        ax = fig.add_subplot(gs_q[0, mi])
        for label, (Ns_, mse, lap, dct) in series.items():
            y = [mse, lap, dct][mi]
            ax.plot(Ns_, y, marker="o", ms=4, label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale(yscale)
        ax.set_xlabel("N"); ax.set_ylabel(mname)
        ax.legend(fontsize=8)
        panel_label(ax, letters[mi])
        ax.set_title(f"{mname}  (σ={SIGMA})", fontsize=10)

    save(fig, "Figure14")


if __name__ == "__main__":
    main()
