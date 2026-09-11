r"""Section C -- through-focus image evidence. Reads directly from the
frozen master-lattice render cache (results/sampling_fusion/cache/frames_<COND>.f64,
memmap of the shared render_frame_shared output, one entry per master
lattice position z_j = -4 + 0.01*j, j=0..800). No new rendering -- these
frames are already frozen S3-cache outputs.

Reproducible frozen z positions used (9 positions spanning the full domain,
j = 0,100,...,800 -> z = -4,-3,...,+3,+4 DOF exactly, NOT chosen for
visual drama):"""
from __future__ import annotations

import numpy as np
from assets_common import OUT, frames_memmap, z_master, get_rois, crop, save_gray, save_npy, manifest_row

SEC = OUT / "02_through_focus"
SELECTED_CONDITIONS = ["H_LF", "H_MIXED", "H_FINE", "P_MIXED"]  # 1 LF, 1 MIXED, 1 FINE, 1 P
J_POSITIONS = list(range(0, 801, 100))  # 9 positions, z = -4..+4 in steps of 1 DOF


def main():
    zm = z_master()
    rois = get_rois()
    for cond in SELECTED_CONDITIONS:
        mm = frames_memmap(cond)
        for j in J_POSITIONS:
            z = float(zm[j])
            frame = np.array(mm[j])  # materialize slice from memmap
            p = SEC / f"TF_{cond}_j{j:03d}_z{z:+.2f}_full.png"
            save_gray(frame, p)
            manifest_row(filename=str(p.relative_to(OUT)),
                         role=f"Through-focus frame, {cond}, z={z:+.2f} DOF, full frame",
                         source_frozen_file=f"results/sampling_fusion/cache/frames_{cond}.f64",
                         direct_or_derived="direct_frozen_reuse", photographic_condition=cond,
                         z_dof=z, image_type="through_focus_frame",
                         display_normalization="vmin=0,vmax=1,gray")
            for roi_name, roi in rois.items():
                c = crop(frame, roi)
                pc = SEC / f"TF_{cond}_j{j:03d}_z{z:+.2f}_roi_{roi_name}.png"
                save_gray(c, pc)
                manifest_row(filename=str(pc.relative_to(OUT)),
                             role=f"Through-focus frame, {cond}, z={z:+.2f} DOF, ROI {roi_name}",
                             source_frozen_file=f"results/sampling_fusion/cache/frames_{cond}.f64",
                             direct_or_derived="deterministic_visualization", photographic_condition=cond,
                             z_dof=z, roi_name=roi_name, roi_coords=str(roi),
                             image_type="through_focus_frame_crop",
                             display_normalization="vmin=0,vmax=1,gray",
                             source_script="scripts/assets_02_through_focus.py")
        del mm
        print(f"  {cond} done ({len(J_POSITIONS)} positions).")

    # Document the exact lattice positions used, for auditability.
    doc = SEC / "z_positions_used.txt"
    doc.write_text("\n".join(f"j={j} -> z={float(zm[j]):+.4f} DOF" for j in J_POSITIONS))
    print("Section C (through-focus) done.")


if __name__ == "__main__":
    main()
