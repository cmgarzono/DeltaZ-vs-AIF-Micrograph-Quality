r"""Fig. 9 -- Axial-selection comparison (all 4 cases: H_LF and
P_FINE, delta=1.00 and 0.10, window=7, ROI=hard_step -- same cases as
Fig. 8). For each case: oracle selection z(k_or), fusion selection z(k_F),
and axial-selection discrepancy |z(k_F)-z(k_or)|.

Selection quantities are converted from raw candidate INDEX (0..K-1,
whose meaning depends on K = candidate count, which differs between
delta=1.00 [K=9] and delta=0.10 [K=81] -- so index maps are NOT
comparable across delta) to actual focal-plane depth z, in DOF (a
physical quantity, directly comparable across conditions AND delta).
z_sub = master lattice positions selected by master_lattice_indices(delta)
(frozen, deterministic, same function used to build the raw index maps in
outputs/manuscript_assets/06_selection_maps).

Two color scales, each pooled/auto-scaled ONCE across everything it
covers (documented, not per-panel):
  - selection z(k_or)/z(k_F): one shared linear scale, vmin/vmax = the
    pooled min/max of z(k_or) and z(k_F) over all 4 displayed cases.
  - axial-selection discrepancy: one shared linear scale, vmin=0,
    vmax = the pooled max of |z(k_F)-z(k_or)| over all 4 displayed cases.
No per-case or per-column independent autoscaling."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import matplotlib.pyplot as plt
from common import ROOT, VEB, save, panel_label, imshow_gray, ROIS, crop, dstr

sys.path.insert(0, str(ROOT / "src"))
from deltaz_aif.analysis.sampling_fusion import build_master_lattice, master_lattice_indices

CONDS = ["H_LF", "P_FINE"]
DELTAS = [1.00, 0.10]
WINDOW = 7
ROI_NAME = "hard_step"


def find(sec, kind, cond, delta):
    p = list((VEB / sec / "window7").glob(f"{kind}_{cond}_delta{dstr(delta)}_K*_w{WINDOW}.npy"))[0]
    return np.load(p)


def main():
    roi = ROIS[ROI_NAME]
    zm = build_master_lattice(-4.0, 4.0)

    # ---- pass 1: compute z(k_or), z(k_F), |z(k_F)-z(k_or)| for all 4 cases,
    # and the two pooled common scales ----
    cases = {}
    sel_pool_min, sel_pool_max = np.inf, -np.inf
    disc_pool_max = -np.inf
    for cond in CONDS:
        for delta in DELTAS:
            idx = master_lattice_indices(delta)
            z_sub = zm[idx]
            k_or_idx = crop(find("06_selection_maps", "sel_k_or", cond, delta), roi).astype(int)
            k_f_idx = crop(find("06_selection_maps", "sel_k_F", cond, delta), roi).astype(int)
            z_or = z_sub[k_or_idx]
            z_f = z_sub[k_f_idx]
            disc = np.abs(z_f - z_or)
            cases[(cond, delta)] = (z_or, z_f, disc)
            sel_pool_min = min(sel_pool_min, float(z_or.min()), float(z_f.min()))
            sel_pool_max = max(sel_pool_max, float(z_or.max()), float(z_f.max()))
            disc_pool_max = max(disc_pool_max, float(disc.max()))

    SEL_VMIN, SEL_VMAX = sel_pool_min, sel_pool_max
    DISC_VMIN, DISC_VMAX = 0.0, disc_pool_max
    print(f"selection z common scale: [{SEL_VMIN:.3f}, {SEL_VMAX:.3f}] DOF (pooled)")
    print(f"axial-selection discrepancy common scale: [{DISC_VMIN:.3f}, {DISC_VMAX:.3f}] DOF (pooled)")

    # ---- pass 2: plot ----
    n_rows = len(CONDS) * len(DELTAS)
    fig = plt.figure(figsize=(9, 11), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, 3)

    col_titles = ["oracle selection  z(k_or)", "fusion selection  z(k_F)",
                  "axial-selection discrepancy  |z(k_F)-z(k_or)|"]

    row = 0
    sel_axes, disc_axes = [], []
    for cond in CONDS:
        for delta in DELTAS:
            z_or, z_f, disc = cases[(cond, delta)]
            ax = fig.add_subplot(gs[row, 0])
            imshow_gray(ax, z_or, vmin=SEL_VMIN, vmax=SEL_VMAX, cmap="viridis")
            sel_axes.append(ax)
            ax.set_ylabel(f"{cond}\nδ={delta:.2f}", fontsize=9, fontweight="bold")
            if row == 0:
                ax.set_title(col_titles[0], fontsize=9.5)
            panel_label(ax, "abcd"[row], x=-0.3, y=1.16 if row == 0 else 1.05)

            ax = fig.add_subplot(gs[row, 1])
            imshow_gray(ax, z_f, vmin=SEL_VMIN, vmax=SEL_VMAX, cmap="viridis")
            sel_axes.append(ax)
            if row == 0:
                ax.set_title(col_titles[1], fontsize=9.5)

            ax = fig.add_subplot(gs[row, 2])
            imshow_gray(ax, disc, vmin=DISC_VMIN, vmax=DISC_VMAX, cmap="inferno")
            disc_axes.append(ax)
            if row == 0:
                ax.set_title(col_titles[2], fontsize=9.5)

            row += 1

    cb1 = fig.colorbar(sel_axes[0].images[0], ax=sel_axes, shrink=0.8, aspect=35, pad=0.02, location="left")
    cb1.set_label("selected focal-plane depth z (DOF)", fontsize=8.5)
    cb1.ax.tick_params(labelsize=7.5)
    cb1.ax.yaxis.set_label_position("left")
    cb1.ax.yaxis.set_ticks_position("left")

    cb2 = fig.colorbar(disc_axes[0].images[0], ax=disc_axes, shrink=0.8, aspect=35, pad=0.02, location="right")
    cb2.set_label("axial-selection discrepancy (DOF)", fontsize=8.5)
    cb2.ax.tick_params(labelsize=7.5)

    save(fig, "Figure9", dpi=1050)


if __name__ == "__main__":
    main()
