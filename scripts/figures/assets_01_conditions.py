r"""Section B -- all 8 fixed photographic conditions (I*) + fixed
geometry-based crops. Source: data/specimen/conditions/oracle_*.npy.
No new texture generated."""
from __future__ import annotations

from assets_common import OUT, CONDITIONS, load_oracle, save_gray, save_npy, get_rois, crop, manifest_row

SEC = OUT / "01_photographic_conditions"


def main():
    rois = get_rois()
    for cond in CONDITIONS:
        I_star = load_oracle(cond)
        p = SEC / f"Istar_{cond}_full.png"
        save_gray(I_star, p)
        save_npy(I_star, SEC / f"Istar_{cond}_full.npy")
        manifest_row(filename=str(p.relative_to(OUT)), role=f"I* ideal all-in-focus reference, {cond}, full frame",
                     source_frozen_file=f"data/specimen/conditions/oracle_{cond}.npy",
                     direct_or_derived="direct_frozen_reuse", photographic_condition=cond,
                     image_type="I_star", display_normalization="vmin=0,vmax=1,gray")

        for roi_name, roi in rois.items():
            c = crop(I_star, roi)
            pc = SEC / f"Istar_{cond}_roi_{roi_name}.png"
            save_gray(c, pc)
            save_npy(c, SEC / f"Istar_{cond}_roi_{roi_name}.npy")
            manifest_row(filename=str(pc.relative_to(OUT)),
                         role=f"I* {cond}, geometry ROI crop ({roi_name})",
                         source_frozen_file=f"data/specimen/conditions/oracle_{cond}.npy",
                         direct_or_derived="deterministic_visualization", photographic_condition=cond,
                         roi_name=roi_name, roi_coords=str(roi), image_type="I_star_crop",
                         display_normalization="vmin=0,vmax=1,gray", source_script="scripts/assets_01_conditions.py")
    print("Section B (photographic conditions) done.")


if __name__ == "__main__":
    main()
