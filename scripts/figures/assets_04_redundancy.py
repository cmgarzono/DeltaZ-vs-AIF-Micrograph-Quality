r"""Section G -- visual evidence of axial redundancy. Fixed geometry ROI
(protrusion, from get_rois()), representative condition H_MIXED, three
axial densities (moderate/dense/very-dense). Reuses:
  - raw candidate frames from results/sampling_fusion/cache/frames_H_MIXED.f64 (frozen)
  - the frozen laplacian_energy() definition from sampling_fusion (no new metric)
  - I_or/I_F crops already exported in Sections D (from assets_03_densification,
    which must have completed for H_MIXED before this script runs)
  - the frozen numeric redundancy summary in results/sampling_fusion/sampling_fusion_redundancy_summary.csv
    / sampling_fusion_raw.csv (adjacent_overlap_mean, N_eff, R_Z) -- copied for reference,
    NOT recomputed."""
from __future__ import annotations

import csv
import numpy as np

from assets_common import (
    OUT, load_height_map, frames_memmap, z_master, get_rois, crop,
    master_lattice_indices, save_gray, save_npy, manifest_row, ROOT,
)
from deltaz_aif.analysis.sampling_fusion import laplacian_energy

SEC = OUT / "07_redundancy_evidence"
COND = "H_MIXED"
WINDOW = 7
DELTAS = {"moderate": 0.50, "dense": 0.15, "very_dense": 0.05}
ROI_NAME = "protrusion"


def dstr(d):
    return f"{d:.2f}".replace(".", "p")


def main():
    rois = get_rois()
    roi = rois[ROI_NAME]
    zm = z_master()
    mm = frames_memmap(COND)

    # Reference frozen redundancy numbers for this condition/window (no new metric).
    ref_rows = []
    with open(ROOT / "results" / "sampling_fusion" / "sampling_fusion_raw.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["condition"] == COND and int(row["window"]) == WINDOW and \
               round(float(row["delta"]), 2) in DELTAS.values():
                ref_rows.append(row)
    (SEC / "frozen_redundancy_reference_H_MIXED_w7.csv").parent.mkdir(parents=True, exist_ok=True)
    with open(SEC / "frozen_redundancy_reference_H_MIXED_w7.csv", "w", newline="") as f:
        if ref_rows:
            w = csv.DictWriter(f, fieldnames=["delta", "K", "N_eff", "R_Z", "adjacent_overlap_mean"])
            w.writeheader()
            for row in ref_rows:
                w.writerow({k: row[k] for k in ["delta", "K", "N_eff", "R_Z", "adjacent_overlap_mean"]})
    manifest_row(filename="07_redundancy_evidence/frozen_redundancy_reference_H_MIXED_w7.csv",
                 role="Reference frozen redundancy numbers (N_eff, R_Z, overlap) for the 3 selected deltas",
                 source_frozen_file="results/sampling_fusion/sampling_fusion_raw.csv", direct_or_derived="direct_frozen_reuse",
                 photographic_condition=COND, image_type="redundancy_reference_table")

    h = load_height_map()
    roi_apex_z = float(np.max(crop(h, roi)))  # focus the local window near the ROI's tallest point

    for label, delta in DELTAS.items():
        idx = master_lattice_indices(delta)
        K = len(idx)
        z_idx = zm[idx]
        mid = int(np.argmin(np.abs(z_idx - roi_apex_z)))  # near-focus window, not geometric middle
        local_idx = idx[max(0, mid - 2):mid + 2]  # up to 4 consecutive candidates near ROI focus

        crops = []
        for i in local_idx:
            frame = np.array(mm[int(i)])
            c = crop(frame, roi)
            crops.append(c)
        crops = np.stack(crops, axis=0)

        # Evidence fields on the cropped mini-stack (frozen laplacian_energy
        # definition; edge effects at the crop boundary are expected and
        # documented -- this is a LOCAL illustrative recomputation over a
        # small ROI, not a new metric).
        E = np.stack([laplacian_energy(crops[k], WINDOW) for k in range(len(local_idx))], axis=0)
        k_local = np.argmax(E, axis=0)

        tag = f"{COND}_{label}_delta{dstr(delta)}_roi{ROI_NAME}"
        for n, (i, c, e) in enumerate(zip(local_idx, crops, E)):
            z = float(zm[int(i)])
            p = SEC / f"cand_{tag}_n{n}_z{z:+.3f}.png"
            save_gray(c, p)
            manifest_row(filename=str(p.relative_to(OUT)),
                         role=f"Neighboring candidate frame crop, {label} density, delta={delta}, z={z:+.3f}",
                         source_frozen_file=f"results/sampling_fusion/cache/frames_{COND}.f64", direct_or_derived="deterministic_visualization",
                         photographic_condition=COND, delta=delta, roi_name=ROI_NAME, roi_coords=str(roi),
                         image_type="redundancy_candidate_crop", source_script="scripts/assets_04_redundancy.py")

            pe = SEC / f"evid_{tag}_n{n}_z{z:+.3f}.png"
            save_gray(e, pe, vmin=0.0, vmax=float(E.max()) if E.max() > 0 else 1.0, cmap="magma")
            manifest_row(filename=str(pe.relative_to(OUT)),
                         role=f"Local Laplacian-energy focus-evidence map, {label} density, z={z:+.3f}",
                         source_frozen_file=f"results/sampling_fusion/cache/frames_{COND}.f64", direct_or_derived="deterministic_visualization",
                         photographic_condition=COND, delta=delta, roi_name=ROI_NAME, roi_coords=str(roi),
                         focus_window=WINDOW, image_type="redundancy_evidence_map", source_script="scripts/assets_04_redundancy.py")

        p = SEC / f"selidx_local_{tag}.png"
        save_gray(k_local.astype(np.float64), p, vmin=0, vmax=max(1, len(local_idx) - 1), cmap="viridis")
        manifest_row(filename=str(p.relative_to(OUT)),
                     role=f"Locally-selected candidate index (within the {len(local_idx)}-frame local window), {label} density",
                     source_frozen_file=f"results/sampling_fusion/cache/frames_{COND}.f64", direct_or_derived="deterministic_visualization",
                     photographic_condition=COND, delta=delta, roi_name=ROI_NAME, roi_coords=str(roi),
                     focus_window=WINDOW, image_type="redundancy_local_selection", source_script="scripts/assets_04_redundancy.py")

        # Local I_or / I_F crops, reused from Section D exports if present.
        for kind, sec_name in [("Ior", "03_oracle_information_space"), ("IF", "04_fused_edof_space")]:
            src = OUT / sec_name / "window7" / f"{kind}_{COND}_delta{dstr(delta)}_K{K}_w{WINDOW}.npy"
            if src.exists():
                arr = np.load(src)
                c = crop(arr, roi)
                p = SEC / f"{kind}_{tag}.png"
                save_gray(c, p)
                manifest_row(filename=str(p.relative_to(OUT)), role=f"Local {kind} crop, {label} density, delta={delta}",
                             source_frozen_file=str(src.relative_to(OUT)), direct_or_derived="deterministic_visualization",
                             photographic_condition=COND, delta=delta, roi_name=ROI_NAME, roi_coords=str(roi),
                             focus_window=WINDOW, image_type=f"redundancy_local_{kind}", source_script="scripts/assets_04_redundancy.py")
            else:
                print(f"  [skip] {src} not yet available (run assets_03_densification for {COND} first)")

    print("Section G (redundancy evidence) done.")


if __name__ == "__main__":
    main()
