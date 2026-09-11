r"""Sections D, E, F -- axial densification (I_or, I_F), registered error
maps, and axial-selection maps, for all 8 photographic conditions at 10
frozen delta values, window=7. Computed EXACTLY via the frozen S3 pipeline
functions (master_lattice_indices, nearest_plane_index, fuse_stack)
applied to the frozen master render cache -- no new simulation, no
recomputation of the through-focus renderer itself (frames are read from
cache, not re-rendered).

I*  = fixed photographic oracle (data/specimen/conditions/oracle_<COND>.npy)
I_or = np.take_along_axis(sub_frames, k_or, axis=0)  [S3 sampling oracle]
I_F  = fuse_stack(sub_frames, window=7)[1]           [S3 practical fusion]
"""
from __future__ import annotations

import time
import numpy as np

from assets_common import (
    OUT, CONDITIONS, load_height_map, load_oracle, frames_memmap, z_master,
    master_lattice_indices, nearest_plane_index, fuse_stack,
    save_gray, save_npy, manifest_row,
)

SEC_OR = OUT / "03_oracle_information_space" / "window7"
SEC_F = OUT / "04_fused_edof_space" / "window7"
SEC_ERR = OUT / "05_error_maps" / "window7"
SEC_SEL = OUT / "06_selection_maps" / "window7"

DELTAS = [2.00, 1.50, 1.00, 0.75, 0.50, 0.35, 0.25, 0.15, 0.10, 0.05]
WINDOW = 7


def dstr(d):
    return f"{d:.2f}".replace(".", "p")


def main(conditions=None):
    conditions = conditions or CONDITIONS
    h = load_height_map()
    zm = z_master()

    for cond in conditions:
        t0 = time.time()
        I_star = load_oracle(cond)
        mm = frames_memmap(cond)
        for delta in DELTAS:
            idx = master_lattice_indices(delta)
            z_sub = zm[idx]
            K = len(idx)
            sub_frames = np.array(mm[idx])  # (K,H,W) materialized

            k_or = nearest_plane_index(h, z_sub)
            I_or = np.take_along_axis(sub_frames, k_or[None, :, :], axis=0)[0]
            k_F, I_F = fuse_stack(sub_frames, WINDOW)

            sampling_err = np.abs(I_or - I_star)
            fusion_err = np.abs(I_F - I_or)
            total_err = np.abs(I_F - I_star)

            tag = f"{cond}_delta{dstr(delta)}_K{K}_w{WINDOW}"

            # --- D: I_or, I_F ---
            p = SEC_OR / f"Ior_{tag}.png"
            save_gray(I_or, p)
            save_npy(I_or, SEC_OR / f"Ior_{tag}.npy")
            manifest_row(filename=str(p.relative_to(OUT)), role=f"I_or sampled information space, {cond}, delta={delta}",
                         source_frozen_file=f"results/sampling_fusion/cache/frames_{cond}.f64", direct_or_derived="deterministic_visualization",
                         photographic_condition=cond, delta=delta, focus_window=WINDOW, image_type="I_or",
                         display_normalization="vmin=0,vmax=1,gray", source_script="scripts/assets_03_densification.py")

            p = SEC_F / f"IF_{tag}.png"
            save_gray(I_F, p)
            save_npy(I_F, SEC_F / f"IF_{tag}.npy")
            manifest_row(filename=str(p.relative_to(OUT)), role=f"I_F reconstructed EDOF image, {cond}, delta={delta}",
                         source_frozen_file=f"results/sampling_fusion/cache/frames_{cond}.f64", direct_or_derived="deterministic_visualization",
                         photographic_condition=cond, delta=delta, focus_window=WINDOW, image_type="I_F",
                         display_normalization="vmin=0,vmax=1,gray", source_script="scripts/assets_03_densification.py")

            # --- E: error maps (fixed 0..0.3 display scale for comparability) ---
            for name, arr, sec, role in [
                ("sampling", sampling_err, SEC_ERR, "sampling discrepancy |I_or - I*|"),
                ("fusion", fusion_err, SEC_ERR, "fusion discrepancy |I_F - I_or|"),
                ("total", total_err, SEC_ERR, "total discrepancy |I_F - I*|"),
            ]:
                p = sec / f"err_{name}_{tag}.png"
                save_gray(arr, p, vmin=0.0, vmax=0.3, cmap="inferno")
                save_npy(arr, sec / f"err_{name}_{tag}.npy")
                manifest_row(filename=str(p.relative_to(OUT)), role=f"{role}, {cond}, delta={delta}",
                             source_frozen_file=f"results/sampling_fusion/cache/frames_{cond}.f64;data/specimen/conditions/oracle_{cond}.npy",
                             direct_or_derived="deterministic_visualization", photographic_condition=cond, delta=delta,
                             focus_window=WINDOW, image_type=f"error_{name}",
                             display_normalization="vmin=0,vmax=0.3,inferno (fixed across all E exports)",
                             source_script="scripts/assets_03_densification.py")

            # --- F: selection maps ---
            axial_sel_err = np.abs(z_sub[k_F] - z_sub[k_or])
            for name, arr, cmap, vmax, role in [
                ("k_or", k_or.astype(np.float64), "viridis", K - 1, "oracle selected axial index map"),
                ("k_F", k_F.astype(np.float64), "viridis", K - 1, "fusion selected axial index map"),
                ("axial_sel_err", axial_sel_err, "inferno", None, "axial-selection error |z_sub[k_F]-z_sub[k_or]|"),
            ]:
                p = SEC_SEL / f"sel_{name}_{tag}.png"
                vm = float(np.max(arr)) if vmax is None else vmax
                save_gray(arr, p, vmin=0.0, vmax=vm, cmap=cmap)
                save_npy(arr, SEC_SEL / f"sel_{name}_{tag}.npy")
                manifest_row(filename=str(p.relative_to(OUT)), role=f"{role}, {cond}, delta={delta}",
                             source_frozen_file=f"results/sampling_fusion/cache/frames_{cond}.f64", direct_or_derived="deterministic_visualization",
                             photographic_condition=cond, delta=delta, focus_window=WINDOW, image_type=f"selection_{name}",
                             display_normalization=f"vmin=0,vmax={vm}", source_script="scripts/assets_03_densification.py")

            del sub_frames, I_or, I_F, k_or, k_F, sampling_err, fusion_err, total_err, axial_sel_err
        del mm
        print(f"  {cond} done in {time.time()-t0:.1f}s")

    print("Sections D/E/F (densification, error maps, selection maps) done.")


if __name__ == "__main__":
    import sys
    conds = sys.argv[1:] if len(sys.argv) > 1 else None
    main(conds)
