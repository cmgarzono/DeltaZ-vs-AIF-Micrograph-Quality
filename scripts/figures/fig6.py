r"""Fig. 6 -- THE central photographic figure: axial densification in
information space (I_or) and EDOF image space (I_F).
Four conditions spanning the spectral-content range and both H/P
organizations: H_LF, P_COARSE, H_MIXED, P_FINE. delta = 1.00, 0.50, 0.30,
0.15, 0.05; window = 7; ROI = protrusion (widened so the pyramid occupies
~50% of the crop area).
delta=0.30 is not in the pre-exported outputs/manuscript_assets evidence
bank (which covers {2.00,1.50,1.00,0.75,0.50,0.35,0.25,0.15,0.10,0.05}) --
it is computed here directly from the frozen master-lattice render cache
via the exact same frozen functions (master_lattice_indices,
nearest_plane_index, fuse_stack from sampling_fusion.py), deterministic, no
new simulation. All other deltas reuse the pre-exported evidence-bank
arrays.
Sources: outputs/manuscript_assets/03_oracle_information_space/window7,
04_fused_edof_space/window7, 01_photographic_conditions (I*),
results/sampling_fusion/cache/frames_<COND>.f64 (for delta=0.30 only)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import matplotlib.pyplot as plt
from common import ROOT, VEB, save, panel_label, imshow_gray, load_npy, dstr

sys.path.insert(0, str(ROOT / "src"))
from deltaz_aif.analysis.sampling_fusion import (
    N_MASTER, build_master_lattice, master_lattice_indices,
    nearest_plane_index, fuse_stack,
)

CONDS = ["H_LF", "P_COARSE", "H_MIXED", "P_FINE"]
DELTAS = [1.00, 0.50, 0.30, 0.15, 0.05]
WINDOW = 7
ROI_NAME = "protrusion"
# Widened protrusion ROI (same center as outputs/manuscript_assets's ROI
# [549:610, 989:1050], 61x61 -> 75x75) so the pyramid occupies ~50% of the
# crop area instead of ~75%.
ROI = [542, 617, 982, 1057]

_computed_cache = {}


def crop(arr, roi):
    y0, y1, x0, x1 = roi
    return arr[y0:y1, x0:x1]


def compute_delta030(cond):
    """Deterministic recomputation via the frozen S3 machinery -- identical
    method to outputs/manuscript_assets/03_oracle_information_space and
    04_fused_edof_space, just for a delta not already exported there."""
    key = (cond, 0.30)
    if key in _computed_cache:
        return _computed_cache[key]
    h = np.load(ROOT / "data" / "specimen" / "height_map.npy")
    zm = build_master_lattice(-4.0, 4.0)
    mm = np.memmap(ROOT / "results" / "sampling_fusion" / "cache" / f"frames_{cond}.f64",
                    dtype=np.float64, mode="r", shape=(N_MASTER, 1080, 1920))
    idx = master_lattice_indices(0.30)
    z_sub = zm[idx]
    sub_frames = np.array(mm[idx])
    k_or = nearest_plane_index(h, z_sub)
    i_or = np.take_along_axis(sub_frames, k_or[None, :, :], axis=0)[0]
    _, i_f = fuse_stack(sub_frames, WINDOW)
    _computed_cache[key] = (i_or, i_f)
    del mm
    return i_or, i_f


def get_ior_if(cond, delta):
    if abs(delta - 0.30) < 1e-9:
        return compute_delta030(cond)
    m_or = list((VEB / "03_oracle_information_space" / "window7").glob(
        f"Ior_{cond}_delta{dstr(delta)}_K*_w{WINDOW}.npy"))[0]
    m_f = list((VEB / "04_fused_edof_space" / "window7").glob(
        f"IF_{cond}_delta{dstr(delta)}_K*_w{WINDOW}.npy"))[0]
    return np.load(m_or), np.load(m_f)


def main():
    n_blocks = len(CONDS)
    height_ratios = []
    for i in range(n_blocks):
        height_ratios += [1, 1]
        if i < n_blocks - 1:
            height_ratios += [0.10]
    fig = plt.figure(figsize=(12.5, 12.8), constrained_layout=True)
    gs = fig.add_gridspec(len(height_ratios), 1 + len(DELTAS), height_ratios=height_ratios,
                           hspace=0.0, wspace=0.05)

    letters = "abcd"
    row_cursor = 0
    for bi, cond in enumerate(CONDS):
        row_or, row_f = row_cursor, row_cursor + 1
        row_cursor += 2
        if bi < n_blocks - 1:
            row_cursor += 1  # skip spacer row

        istar = load_npy(f"01_photographic_conditions/Istar_{cond}_full.npy")
        istar_c = crop(istar, ROI)
        ax = fig.add_subplot(gs[row_or, 0])
        imshow_gray(ax, istar_c)
        if bi == 0:
            ax.set_title("I*", fontsize=10)
        ax.set_ylabel("I_or", fontsize=9)
        panel_label(ax, letters[bi], x=-0.32, y=1.12 if bi == 0 else 1.02)

        ax2 = fig.add_subplot(gs[row_f, 0])
        imshow_gray(ax2, istar_c)
        ax2.set_ylabel("I_F", fontsize=9)

        for di, delta in enumerate(DELTAS):
            i_or_full, i_f_full = get_ior_if(cond, delta)
            i_or = crop(i_or_full, ROI)
            i_f = crop(i_f_full, ROI)

            ax = fig.add_subplot(gs[row_or, 1 + di])
            imshow_gray(ax, i_or)
            if bi == 0:
                ax.set_title(f"δ={delta:.2f}", fontsize=10)

            ax = fig.add_subplot(gs[row_f, 1 + di])
            imshow_gray(ax, i_f)

        y_top = 1.0 - (row_or) / len(height_ratios)
        y_bot = 1.0 - (row_or + 2) / len(height_ratios)
        fig.text(0.005, (y_top + y_bot) / 2, cond, rotation=90, va="center", ha="left",
                  fontsize=10, fontweight="bold")

    save(fig, "Figure6", dpi=1050)


if __name__ == "__main__":
    main()
