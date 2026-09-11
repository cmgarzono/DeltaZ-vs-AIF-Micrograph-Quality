r"""Fig. 8 -- Sampling limitation and fusion discrepancy.
Conditions H_LF and P_FINE, delta = 1.00 and 0.10, window=7,
ROI=hard_step. Discrepancy maps share ONE common linear scale (0-0.3) and
ONE shared colorbar, rendered at 3x the figure set's default dpi. The
axial-selection companion panel (same 4 cases) is Fig. 9, in fig9.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import matplotlib.pyplot as plt
from common import VEB, save, panel_label, imshow_gray, ROIS, crop, load_npy, dstr

CONDS = ["H_LF", "P_FINE"]
DELTAS = [1.00, 0.10]
WINDOW = 7
ROI_NAME = "hard_step"
ERR_VMIN, ERR_VMAX = 0.0, 0.3

COL_TITLES = ["I*", "I_or", "I_F", "|I_or-I*|", "|I_F-I_or|", "|I_F-I*|"]


def find(sec, kind, cond, delta):
    p = list((VEB / sec / "window7").glob(f"{kind}_{cond}_delta{dstr(delta)}_K*_w{WINDOW}.npy"))[0]
    return np.load(p)


def main():
    roi = ROIS[ROI_NAME]
    n_rows = len(CONDS) * len(DELTAS)

    fig = plt.figure(figsize=(14.5, 11), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, 6)

    row = 0
    for cond in CONDS:
        for delta in DELTAS:
            istar = load_npy(f"01_photographic_conditions/Istar_{cond}_roi_{ROI_NAME}.npy")
            i_or = crop(find("03_oracle_information_space", "Ior", cond, delta), roi)
            i_f = crop(find("04_fused_edof_space", "IF", cond, delta), roi)
            e_s = crop(find("05_error_maps", "err_sampling", cond, delta), roi)
            e_fus = crop(find("05_error_maps", "err_fusion", cond, delta), roi)
            e_tot = crop(find("05_error_maps", "err_total", cond, delta), roi)

            panels = [istar, i_or, i_f, e_s, e_fus, e_tot]
            for ci, arr in enumerate(panels):
                ax = fig.add_subplot(gs[row, ci])
                if ci < 3:
                    imshow_gray(ax, arr)
                else:
                    imshow_gray(ax, arr, vmin=ERR_VMIN, vmax=ERR_VMAX, cmap="inferno")
                    ax.images[0].set_clim(ERR_VMIN, ERR_VMAX)
                if row == 0:
                    ax.set_title(COL_TITLES[ci], fontsize=10)
                if ci == 0:
                    ax.set_ylabel(f"{cond}\nδ={delta:.2f}", fontsize=9, fontweight="bold")
            panel_label(fig.axes[-6], "abcd"[row], x=-0.28, y=1.18 if row == 0 else 1.05)
            row += 1

    evid_axes = [a for a in fig.axes if a.images and tuple(a.images[0].get_clim()) == (ERR_VMIN, ERR_VMAX)]
    cb = fig.colorbar(evid_axes[0].images[0], ax=evid_axes, shrink=0.75, aspect=30, pad=0.01, location="right")
    cb.set_label("discrepancy (common linear scale, 0–0.3)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    save(fig, "Figure8", dpi=1050)


if __name__ == "__main__":
    main()
