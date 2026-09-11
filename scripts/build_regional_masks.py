r"""
Reconstruct the frozen Specimen ground-truth regional masks in an
environment where the external canonical 5-DOF source archive is not
present, using ONLY:
  - data/specimen/height_map.npy (the frozen, already-scaled 8-DOF
    height map, bit-identical to Specimen().height_map)
  - the geometry constants recorded in data/specimen/specimen_stats.json
    (outer/inner rectangle, r_pixels=5.0,
    r_dof=0.2, pyramid_characteristic_semisize_r=3.08,
    pyramid_characteristic_full_width_r=6.16, pyramid_count=27)

This does NOT change the mask construction algorithm in
src/deltaz_aif/specimen/geometry.py in any way -- it only supplies the
(height_map, metadata) inputs that Specimen's constructor would otherwise
read from the external archive. The archive stores h_5dof; Specimen
computes height_map = 1.6 * h_5dof internally, so we invert that once
(h_5dof = cached / 1.6) to satisfy the constructor, then verify the
reconstructed height_map matches the frozen cached array to float64
precision before trusting the derived masks.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

SPECIMEN_DIR = ROOT / "data" / "specimen"
CACHE_DIR = ROOT / "results" / "sampling_fusion" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STATS = json.loads((SPECIMEN_DIR / "specimen_stats.json").read_text())
GEOM = STATS["geometry"]

METADATA = {
    "geometry": {
        "outer_rectangle_px": GEOM["outer_rectangle_px"],
        "inner_rectangle_px": GEOM["inner_rectangle_px"],
        "pyramid_count": GEOM["pyramid_count_declared"],
        "pyramid_characteristic_semisize_r": 3.08,
        "pyramid_characteristic_full_width_r": 6.16,
        "indentation_count": GEOM["indentation_count_declared"],
    },
    "r_pixels": 5.0,
    "r_dof": 0.2,
}


def build_and_cache_masks():
    from deltaz_aif.specimen.geometry import Specimen

    cached_h = np.load(SPECIMEN_DIR / "height_map.npy").astype(np.float64)
    h_5dof = cached_h / 1.6

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        h5_path = td / "height_map.npy"
        meta_path = td / "metadata.json"
        np.save(h5_path, h_5dof)
        meta_path.write_text(json.dumps(METADATA))

        m1 = Specimen(canonical_height_map_path=str(h5_path), canonical_metadata_path=str(meta_path))

    max_diff = float(np.max(np.abs(m1.height_map - cached_h)))
    print(f"Reconstructed height_map max abs diff vs frozen cache: {max_diff:.3e}")
    assert max_diff < 1e-9, "Reconstructed height map does not match frozen cache"

    masks = {
        "plateau": m1.plateau_mask,
        "ramp": m1.ramp_mask,
        "hard_step": m1.hard_step_mask,
        "protrusion": m1.protrusion_mask,
    }
    for a, na in masks.items():
        for b, nb in masks.items():
            if a < b:
                overlap = int(np.sum(na & nb))
                assert overlap == 0, f"{a} and {b} overlap by {overlap} px"
    assert m1.protrusion_count == 27, f"protrusion_count={m1.protrusion_count}, expected 27"
    print(f"protrusion_count = {m1.protrusion_count} (expected 27) -- OK")
    print("All 4 masks mutually exclusive -- OK")

    np.savez(
        CACHE_DIR / "regional_masks.npz",
        plateau=masks["plateau"], ramp=masks["ramp"],
        hard_step=masks["hard_step"], protrusion=masks["protrusion"],
    )
    print(f"Cached masks -> {CACHE_DIR / 'regional_masks.npz'}")
    return masks


if __name__ == "__main__":
    build_and_cache_masks()
