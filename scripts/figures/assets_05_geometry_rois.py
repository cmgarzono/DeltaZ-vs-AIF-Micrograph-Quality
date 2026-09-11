r"""Section H -- geometry-defined ROI series. Crops the already-exported
Section D/E/F full-frame arrays (I*, I_or, I_F, error maps, selection maps)
at the 4 fixed geometry ROIs, for a representative subset of conditions and
deltas, keeping exact registration (same crop box applied to every image
type/delta). Does NOT recompute anything -- pure cropping of frozen-derived
arrays already on disk from assets_01_conditions.py / assets_03_densification.py."""
from __future__ import annotations

import numpy as np
from assets_common import OUT, load_oracle, get_rois, crop, save_gray, save_npy, manifest_row

SEC = OUT / "08_geometry_rois"
CONDS = ["H_MIXED", "P_MIXED"]
DELTAS = [1.00, 0.50, 0.15]
WINDOW = 7


def dstr(d):
    return f"{d:.2f}".replace(".", "p")


def main():
    rois = get_rois()
    for cond in CONDS:
        I_star = load_oracle(cond)
        for roi_name, roi in rois.items():
            for delta in DELTAS:
                tag = f"{cond}_{roi_name}_delta{dstr(delta)}"

                istar_c = crop(I_star, roi)
                p = SEC / f"Istar_{tag}.png"
                save_gray(istar_c, p)
                manifest_row(filename=str(p.relative_to(OUT)), role=f"I* ROI crop, {cond}, {roi_name}, delta={delta}",
                             source_frozen_file=f"data/specimen/conditions/oracle_{cond}.npy",
                             direct_or_derived="deterministic_visualization", photographic_condition=cond,
                             delta=delta, roi_name=roi_name, roi_coords=str(roi), image_type="roi_I_star",
                             source_script="scripts/assets_05_geometry_rois.py")

                K_guess = None
                for kind, sec_name, dispkw in [
                    ("Ior", "03_oracle_information_space", dict()),
                    ("IF", "04_fused_edof_space", dict()),
                ]:
                    base = OUT / sec_name / "window7"
                    matches = list(base.glob(f"{kind}_{cond}_delta{dstr(delta)}_K*_w{WINDOW}.npy"))
                    if not matches:
                        continue
                    arr = np.load(matches[0])
                    c = crop(arr, roi)
                    p = SEC / f"{kind}_{tag}.png"
                    save_gray(c, p)
                    manifest_row(filename=str(p.relative_to(OUT)), role=f"{kind} ROI crop, {cond}, {roi_name}, delta={delta}",
                                 source_frozen_file=str(matches[0].relative_to(OUT)), direct_or_derived="deterministic_visualization",
                                 photographic_condition=cond, delta=delta, roi_name=roi_name, roi_coords=str(roi),
                                 focus_window=WINDOW, image_type=f"roi_{kind}", source_script="scripts/assets_05_geometry_rois.py")

                for kind, disp in [("sampling", (0, 0.3, "inferno")), ("fusion", (0, 0.3, "inferno")), ("total", (0, 0.3, "inferno"))]:
                    base = OUT / "05_error_maps" / "window7"
                    matches = list(base.glob(f"err_{kind}_{cond}_delta{dstr(delta)}_K*_w{WINDOW}.npy"))
                    if not matches:
                        continue
                    arr = np.load(matches[0])
                    c = crop(arr, roi)
                    p = SEC / f"err_{kind}_{tag}.png"
                    save_gray(c, p, vmin=disp[0], vmax=disp[1], cmap=disp[2])
                    manifest_row(filename=str(p.relative_to(OUT)), role=f"{kind} discrepancy ROI crop, {cond}, {roi_name}, delta={delta}",
                                 source_frozen_file=str(matches[0].relative_to(OUT)), direct_or_derived="deterministic_visualization",
                                 photographic_condition=cond, delta=delta, roi_name=roi_name, roi_coords=str(roi),
                                 focus_window=WINDOW, image_type=f"roi_err_{kind}",
                                 display_normalization=f"vmin={disp[0]},vmax={disp[1]},{disp[2]}", source_script="scripts/assets_05_geometry_rois.py")

                for kind in ["k_or", "k_F", "axial_sel_err"]:
                    base = OUT / "06_selection_maps" / "window7"
                    matches = list(base.glob(f"sel_{kind}_{cond}_delta{dstr(delta)}_K*_w{WINDOW}.npy"))
                    if not matches:
                        continue
                    arr = np.load(matches[0])
                    c = crop(arr, roi)
                    p = SEC / f"sel_{kind}_{tag}.png"
                    vmax = float(np.max(c)) if c.max() > 0 else 1.0
                    save_gray(c, p, vmin=0, vmax=vmax, cmap="viridis" if kind != "axial_sel_err" else "inferno")
                    manifest_row(filename=str(p.relative_to(OUT)), role=f"{kind} ROI crop, {cond}, {roi_name}, delta={delta}",
                                 source_frozen_file=str(matches[0].relative_to(OUT)), direct_or_derived="deterministic_visualization",
                                 photographic_condition=cond, delta=delta, roi_name=roi_name, roi_coords=str(roi),
                                 focus_window=WINDOW, image_type=f"roi_sel_{kind}", source_script="scripts/assets_05_geometry_rois.py")
        print(f"  {cond} done.")
    print("Section H (geometry ROI series) done.")


if __name__ == "__main__":
    main()
