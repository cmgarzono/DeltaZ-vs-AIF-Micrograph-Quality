r"""Fig. 11 -- Dense axial sampling: candidate images and local
focus-evidence maps. Condition H_MIXED, window=7, ROI=protrusion, three
regimes (delta=0.50 moderate, 0.15 dense, 0.05 very dense), rendered at 3x
the figure set's default dpi. Candidate images: frozen PNGs from
outputs/manuscript_assets/07_redundancy_evidence (common 0..1 grayscale
display, unchanged). Focus-evidence maps: recomputed directly from the
frozen master-lattice render cache using the EXACT SAME frozen function
(deltaz_aif.analysis.sampling_fusion.laplacian_energy) and the EXACT SAME
candidate selection as outputs/manuscript_assets/07_redundancy_evidence --
bit-identical arrays -- with ONE common linear display scale (vmin=0,
vmax=global max across all displayed evidence maps) and one shared
colorbar, so weak and strong evidence stay honestly comparable across
densities. Quantitative panels (K/N_eff, R_Z/quality) moved to part 2."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from common import ROOT, VEB, SPECIMEN_DIR, save, panel_label

sys.path.insert(0, str(ROOT / "src"))
from deltaz_aif.analysis.sampling_fusion import (
    N_MASTER, build_master_lattice, master_lattice_indices, laplacian_energy,
)

COND = "H_MIXED"
DENSITIES = [("moderate", 0.50), ("dense", 0.15), ("very_dense", 0.05)]
WINDOW = 7
ROI = json.loads((VEB / "10_manifest" / "roi_definitions.json").read_text())["protrusion"]


def dstr(d):
    return f"{d:.2f}".replace(".", "p")


def local_candidates(zm, mm, delta, roi_apex_z):
    """EXACT same selection rule as scripts/assets_04_redundancy.py: 3-4
    consecutive master-lattice candidates nearest the ROI's tallest point."""
    idx = master_lattice_indices(delta)
    z_idx = zm[idx]
    mid = int(np.argmin(np.abs(z_idx - roi_apex_z)))
    local_idx = idx[max(0, mid - 2):mid + 2]
    y0, y1, x0, x1 = ROI
    crops = [np.array(mm[int(i)])[y0:y1, x0:x1] for i in local_idx]
    return local_idx, np.stack(crops, axis=0)


def main():
    R = VEB / "07_redundancy_evidence"
    h = np.load(SPECIMEN_DIR / "height_map.npy")
    y0, y1, x0, x1 = ROI
    roi_apex_z = float(h[y0:y1, x0:x1].max())
    zm = build_master_lattice(-4.0, 4.0)
    mm = np.memmap(ROOT / "results" / "sampling_fusion" / "cache" / f"frames_{COND}.f64",
                    dtype=np.float64, mode="r", shape=(N_MASTER, 1080, 1920))

    # ---- recompute evidence arrays for all 3 densities, find the common scale ----
    evidence_by_density = {}
    all_max = 0.0
    for label, delta in DENSITIES:
        local_idx, crops = local_candidates(zm, mm, delta, roi_apex_z)
        E = np.stack([laplacian_energy(crops[k], WINDOW) for k in range(crops.shape[0])], axis=0)
        evidence_by_density[label] = (local_idx, E)
        all_max = max(all_max, float(E.max()))

    EVID_VMIN, EVID_VMAX = 0.0, all_max

    fig = plt.figure(figsize=(12.5, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(2 * len(DENSITIES), 4, height_ratios=[1, 1] * len(DENSITIES))

    letters = "abc"
    evid_im = None
    for ri, (label, delta) in enumerate(DENSITIES):
        tag = f"{COND}_{label}_delta{dstr(delta)}_roiprotrusion"
        cand_files = sorted(R.glob(f"cand_{tag}_n*.png"))
        local_idx, E = evidence_by_density[label]

        row_img, row_evid = 2 * ri, 2 * ri + 1
        for ci, cf in enumerate(cand_files[:4]):
            ax = fig.add_subplot(gs[row_img, ci])
            img = mpimg.imread(cf)
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            z = cf.stem.split("_z")[-1]
            ax.set_title(f"z={z}", fontsize=9, pad=8)
            if ci == 0:
                panel_label(ax, letters[ri], x=-0.25, y=1.22)
                ax.set_ylabel(f"{label}\nδ={delta:.2f}\ncandidate", fontsize=8.5)

        for ci in range(E.shape[0]):
            ax = fig.add_subplot(gs[row_evid, ci])
            evid_im = ax.imshow(E[ci], vmin=EVID_VMIN, vmax=EVID_VMAX, cmap="magma")
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel("evidence\n(Laplacian\nenergy)", fontsize=8.5)
        for ci in range(E.shape[0], 4):
            ax = fig.add_subplot(gs[row_evid, ci]); ax.axis("off")
        for ci in range(len(cand_files[:4]), 4):
            ax = fig.add_subplot(gs[row_img, ci]); ax.axis("off")

    evid_axes = [a for a in fig.axes if a.images and a.images[0].get_clim() == (EVID_VMIN, EVID_VMAX)]
    cb = fig.colorbar(evid_im, ax=evid_axes, shrink=0.7, aspect=25, pad=0.015, location="right")
    cb.set_label("local focus evidence (Laplacian energy, common linear scale)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    save(fig, "Figure11", dpi=1050)


if __name__ == "__main__":
    main()
