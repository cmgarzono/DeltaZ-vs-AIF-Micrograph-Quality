r"""Fig. 5 -- Through-focus image formation.
Two conditions, H_MIXED and P_MIXED, full-frame M1 only (no ROI crops, no
markers, no insets). Nine systematic master-lattice positions in 1 DOF
steps: z = -4, -3, -2, -1, 0, +1, +2, +3, +4 DOF. Each condition is its
own 3x3 block, with a dedicated label row above it (part of the grid, not
free text) so labels do not overlap the images."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import matplotlib.pyplot as plt
from common import ROOT, save, panel_label, imshow_gray

sys.path.insert(0, str(ROOT / "src"))
from deltaz_aif.analysis.sampling_fusion import N_MASTER, build_master_lattice

CONDS = ["H_MIXED", "P_MIXED"]
J_POS = list(range(0, 801, 100))  # z = -4, -3, ..., +4 (9 positions, step 1 DOF)


def main():
    zm = build_master_lattice(-4.0, 4.0)

    fig = plt.figure(figsize=(11, 9.6), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1])

    for bi, cond in enumerate(CONDS):
        mm = np.memmap(ROOT / "results" / "sampling_fusion" / "cache" / f"frames_{cond}.f64",
                        dtype=np.float64, mode="r", shape=(N_MASTER, 1080, 1920))
        block = outer[bi].subgridspec(4, 3, height_ratios=[0.16, 1, 1, 1], hspace=0.12, wspace=0.04)

        ax_lab = fig.add_subplot(block[0, :])
        ax_lab.axis("off")
        ax_lab.text(0.0, 0.0, cond, fontsize=15, fontweight="bold", ha="left", va="bottom")

        for i, j in enumerate(J_POS):
            r, c = divmod(i, 3)
            z = float(zm[j])
            frame = np.array(mm[j])
            ax = fig.add_subplot(block[1 + r, c])
            imshow_gray(ax, frame)
            ax.set_box_aspect(1080 / 1920)
            ax.set_title(f"z = {z:+.0f} DOF", fontsize=11, pad=6)
            if i == 0:
                panel_label(ax, "a" if bi == 0 else "j", x=-0.04, y=1.20)

        del mm

    save(fig, "Figure5", dpi=700)


if __name__ == "__main__":
    main()
