r"""Section A -- physical specimen evidence. Source: frozen
data/specimen/height_map.npy + results/sampling_fusion/cache/regional_masks.npz
(the masks actually used by the frozen S3 pipeline -- NOT the stale
percentile-based masks in data/specimen/masks_definition.json).
Reuses existing frozen renders where they already exist (A1-A4-ish); adds
only what is missing (region map from the CURRENT frozen masks, detail
crops, height profiles)."""
from __future__ import annotations

import shutil
import numpy as np
import matplotlib.pyplot as plt

from assets_common import (
    OUT, SPECIMEN_DIR, load_height_map, load_regional_masks, save_gray, save_npy,
    largest_component_bbox, bbox_of_mask, manifest_row,
)

SEC = OUT / "00_specimen"


def reuse(src_name, dst_name, role):
    src = SPECIMEN_DIR / src_name
    if not src.exists():
        print(f"  MISSING frozen asset: {src}")
        return
    dst = SEC / dst_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest_row(filename=str(dst.relative_to(OUT)), role=role,
                 source_frozen_file=str(src.relative_to(SPECIMEN_DIR.parent.parent.parent)),
                 direct_or_derived="direct_frozen_reuse", condition="NA", image_type="specimen")


def main():
    h = load_height_map()
    masks = load_regional_masks()

    # --- A1-A4: reuse existing frozen full-map / 3D renders verbatim ---
    reuse("01_height_map_grayscale_full.png", "A1_height_map_grayscale_full.png",
          "Full M1 height map [-4,+4] DOF, grayscale")
    reuse("02_height_map_falsecolor_full.png", "A1b_height_map_falsecolor_full.png",
          "Full M1 height map [-4,+4] DOF, false color")
    reuse("03_3d_global_view.png", "A2_oblique_3d_global_view.png",
          "Oblique 3D rendering of full physical surface")
    reuse("04_3d_central_platform.png", "A3_3d_central_platform_topview_like.png",
          "3D rendering, central platform (top-view-like)")
    reuse("05_3d_outer_protrusions.png", "A5_3d_outer_protrusions.png",
          "3D detail, outer protrusions region")
    reuse("06_3d_hardstep_region.png", "A6_3d_hardstep_region.png",
          "3D detail, hard-step region")
    reuse("07_3d_ramp_region.png", "A7_3d_ramp_region.png",
          "3D detail, smooth-ramp region")

    save_npy(h, SEC / "A0_height_map.npy")
    manifest_row(filename="00_specimen/A0_height_map.npy", role="M1 height map source array",
                 source_frozen_file="data/specimen/height_map.npy",
                 direct_or_derived="direct_frozen_reuse", image_type="height_array")

    # --- A4: geometry-defined region map from CURRENT frozen masks ---
    H, W = h.shape
    region_map = np.zeros((H, W), dtype=np.uint8)
    region_map[masks["plateau"]] = 1
    region_map[masks["ramp"]] = 2
    region_map[masks["hard_step"]] = 3
    region_map[masks["protrusion"]] = 4
    colors = np.array([
        [0.10, 0.10, 0.10],  # 0 unclassified/background
        [0.20, 0.55, 0.90],  # 1 plateau
        [0.95, 0.75, 0.15],  # 2 ramp
        [0.90, 0.20, 0.20],  # 3 hard_step
        [0.20, 0.85, 0.30],  # 4 protrusion
    ])
    rgb = colors[region_map]
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=200)
    ax.imshow(rgb)
    ax.set_title("M1 geometry-defined regions (frozen ground-truth masks: "
                 "plateau / ramp / hard-step / protrusion)")
    ax.axis("off")
    from matplotlib.patches import Patch
    labels = ["background/unclassified", "plateau", "ramp", "hard_step", "protrusion"]
    handles = [Patch(color=colors[i], label=labels[i]) for i in range(5)]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    p = SEC / "A4_geometry_region_map_current_frozen_masks.png"
    fig.savefig(p)
    plt.close(fig)
    manifest_row(filename=str(p.relative_to(OUT)), role="Geometry-defined region map (current frozen masks)",
                 source_frozen_file="results/sampling_fusion/cache/regional_masks.npz",
                 direct_or_derived="deterministic_visualization", image_type="region_map",
                 source_script="scripts/assets_00_specimen.py")

    # --- A5/A6/A7: large 2D detail crops (height map, not 3D render) ---
    def detail(mask, name, title, pad):
        y0, y1, x0, x1 = largest_component_bbox(mask, pad=pad) if name == "pyramid" else bbox_of_mask(mask, pad=pad)
        crop_h = h[y0:y1, x0:x1]
        fig, ax = plt.subplots(figsize=(8, 8 * crop_h.shape[0] / max(1, crop_h.shape[1])), dpi=200)
        im = ax.imshow(crop_h, cmap="viridis", vmin=-4, vmax=4)
        ax.set_title(f"{title}  (rows {y0}:{y1}, cols {x0}:{x1})")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, label="height (DOF)")
        fig.tight_layout()
        p = SEC / f"A_detail_{name}.png"
        fig.savefig(p)
        plt.close(fig)
        save_npy(crop_h, SEC / f"A_detail_{name}.npy")
        manifest_row(filename=str(p.relative_to(OUT)), role=title,
                     source_frozen_file="data/specimen/height_map.npy",
                     direct_or_derived="deterministic_visualization",
                     roi_name=name, roi_coords=f"{y0}:{y1},{x0}:{x1}", image_type="height_detail",
                     source_script="scripts/assets_00_specimen.py")

    detail(masks["protrusion"], "pyramid", "A5: large detail, one representative square pyramid", 25)
    detail(masks["hard_step"], "hard_step", "A6: large detail, hard-step region", 40)
    detail(masks["ramp"], "ramp", "A7: large detail, smooth-ramp region", 40)

    # --- A8: height profiles crossing important structures ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=150)
    # Horizontal profile through a pyramid row (protrusion mask row with most True)
    prot_rows = masks["protrusion"].sum(axis=1)
    y_star = int(np.argmax(prot_rows))
    axes[0].plot(h[y_star, :])
    axes[0].set_title(f"A8a: horizontal height profile at row y={y_star} "
                       f"(crosses hard-step, ramp, protrusions)")
    axes[0].set_xlabel("x (px)"); axes[0].set_ylabel("height (DOF)")
    axes[0].grid(alpha=0.3)

    prot_cols = masks["protrusion"].sum(axis=0)
    x_star = int(np.argmax(prot_cols))
    axes[1].plot(h[:, x_star])
    axes[1].set_title(f"A8b: vertical height profile at col x={x_star} "
                       f"(crosses hard-step, ramp, protrusions)")
    axes[1].set_xlabel("y (px)"); axes[1].set_ylabel("height (DOF)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p = SEC / "A8_height_profiles.png"
    fig.savefig(p)
    plt.close(fig)
    manifest_row(filename=str(p.relative_to(OUT)), role="Height profiles crossing key structures",
                 source_frozen_file="data/specimen/height_map.npy",
                 direct_or_derived="deterministic_visualization", image_type="height_profile",
                 source_script="scripts/assets_00_specimen.py")

    print("Section A (specimen) done.")


if __name__ == "__main__":
    main()
