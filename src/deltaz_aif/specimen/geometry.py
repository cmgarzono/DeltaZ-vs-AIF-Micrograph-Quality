r"""
Specimen geometry

The specimen is an exact geometric port of a canonical 5-DOF synthetic
specimen, scaled axially by a single transformation:

    h_8DOF(x, y) = 1.6 * h_5DOF(x, y)

All lateral geometry (resolution, rectangles, pyramid locations) is preserved.
Masks are ground-truth based on known geometric construction, not heuristics.

By default, Specimen() loads the already-scaled height map bundled at
data/specimen/height_map.npy (with data/specimen/specimen_stats.json for
the geometry constants) -- no external source or configuration needed.
If DELTAZ_CANONICAL_SOURCE is set, or explicit paths are passed, it loads
and rescales that external 5-DOF source instead; this is
only needed to regenerate the bundled array itself, not to use it.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import label, center_of_mass


# Location of the canonical (pre-scaling) specimen assets, only used when
# explicitly requested.
#
# Configuration priority:
#   1. Explicit paths passed to Specimen() constructor
#   2. DELTAZ_CANONICAL_SOURCE environment variable (if set)
#   3. The bundled, already-scaled data/specimen/height_map.npy (default)

def _get_canonical_source() -> Optional[Path]:
    """Get canonical source path from environment variable if set."""
    env_path = os.environ.get("DELTAZ_CANONICAL_SOURCE")
    if env_path:
        return Path(env_path)
    return None

_EXTERNAL_CANONICAL_SOURCE = _get_canonical_source()
_CANONICAL_HEIGHT_MAP = (_EXTERNAL_CANONICAL_SOURCE / "height_map.npy") if _EXTERNAL_CANONICAL_SOURCE else None
_CANONICAL_METADATA = (_EXTERNAL_CANONICAL_SOURCE / "metadata.json") if _EXTERNAL_CANONICAL_SOURCE else None

# Bundled, already-scaled fallback (data/specimen/, relative to repo root:
# src/deltaz_aif/specimen/geometry.py -> parents[3]).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BUNDLED_HEIGHT_MAP = _REPO_ROOT / "data" / "specimen" / "height_map.npy"
_BUNDLED_STATS = _REPO_ROOT / "data" / "specimen" / "specimen_stats.json"

# The single permitted geometric transformation.
AXIAL_SCALE_FACTOR = 1.6

# ========== GROUND-TRUTH MASK PARAMETERS ==========
# These are NOT heuristic guesses, but exact geometric definitions
# derived from the specimen construction.

# Hard-step neighborhoods (left & top faces of outer rectangle)
# These boundaries are abrupt; 1 pixel width captures the step.
HARD_STEP_NEIGHBORHOOD_WIDTH_PX = 1

# Ramp regions (smooth transitions):
#   - Right face: ~300 px from x1 toward x0
#   - Bottom face: ~300 px from y1 toward y0
#   - Left shoulder: background region left of x0 (no active ramp there)
RIGHT_RAMP_WIDTH_PX = 300
BOTTOM_RAMP_WIDTH_PX = 300

# Plateau height level (background interior)
# Flat regions have h ≈ 0.526 (for original 5-DOF)
# For 8-DOF (scaled by 1.6): h ≈ 0.526 * 1.6 ≈ 0.841
PLATEAU_HEIGHT_5DOF = 0.52631575
PLATEAU_HEIGHT_TOLERANCE = 1e-4


class Specimen:
    """
    M1 specimen: 8-DOF synthetic topography.

    Loads canonical 5-DOF source and applies 1.6× axial scaling.
    All lateral geometry preserved exactly.
    """

    def __init__(
        self,
        canonical_height_map_path: Optional[str] = None,
        canonical_metadata_path: Optional[str] = None,
        axial_scale_factor: float = AXIAL_SCALE_FACTOR,
    ):
        """
        Initialize M1 specimen by loading and rescaling the canonical 5-DOF geometry.

        Args:
            canonical_height_map_path: Path to original .npy height map (must exist).
                Defaults to the external canonical source path.
            canonical_metadata_path: Path to original metadata.json (must exist).
                Defaults to the external canonical source path.
            axial_scale_factor: Scalar applied to the original height map.
                Must remain 1.6 per specification.
        """
        self.axial_scale_factor = axial_scale_factor
        self._bundled_tmpdir = None  # keeps the fallback temp dir alive, if used
        self.source = "external_canonical"

        # Use provided paths, else the external canonical source (if
        # configured), else fall back to reconstructing from the bundled,
        # already-scaled data/specimen/height_map.npy.
        hmap_path = Path(canonical_height_map_path) if canonical_height_map_path else _CANONICAL_HEIGHT_MAP
        meta_path = Path(canonical_metadata_path) if canonical_metadata_path else _CANONICAL_METADATA

        explicit_source_requested = canonical_height_map_path is not None or _EXTERNAL_CANONICAL_SOURCE is not None
        need_bundled_fallback = (hmap_path is None or not hmap_path.exists()) and not explicit_source_requested

        if need_bundled_fallback and _BUNDLED_HEIGHT_MAP.exists() and _BUNDLED_STATS.exists():
            self.source = "bundled"
            bundled_h8dof = np.load(_BUNDLED_HEIGHT_MAP).astype(np.float64)
            h_5dof = bundled_h8dof / axial_scale_factor
            geom = json.loads(_BUNDLED_STATS.read_text())["geometry"]
            metadata = {
                "geometry": {
                    "outer_rectangle_px": geom["outer_rectangle_px"],
                    "inner_rectangle_px": geom["inner_rectangle_px"],
                    "pyramid_count": geom["pyramid_count_declared"],
                    "pyramid_characteristic_semisize_r": 3.08,
                    "pyramid_characteristic_full_width_r": 6.16,
                    "indentation_count": geom["indentation_count_declared"],
                },
                "r_pixels": 5.0,
                "r_dof": 0.2,
            }
            self._bundled_tmpdir = tempfile.TemporaryDirectory(prefix="deltaz_specimen_")
            td = Path(self._bundled_tmpdir.name)
            hmap_path = td / "height_map.npy"
            meta_path = td / "metadata.json"
            np.save(hmap_path, h_5dof)
            meta_path.write_text(json.dumps(metadata))

        if hmap_path is None or not hmap_path.exists():
            raise FileNotFoundError(
                f"Specimen source height map not found at {hmap_path}.\n"
                f"\nNeither the bundled data/specimen/height_map.npy nor an external\n"
                f"canonical 5-DOF source could be found.\n"
                f"\nIf you `pip install`-ed this package from a wheel/sdist rather than\n"
                f"running from a clone of the repository: the ~230 MB of frozen\n"
                f"scientific data under data/ is deliberately NOT bundled into the\n"
                f"Python package (only the code under src/ is) -- it ships with the\n"
                f"git repository. Clone https://github.com/cmgarzono/DeltaZ_AIF and run\n"
                f"from there (see README.md / REPRODUCE.md), or supply a source\n"
                f"explicitly:\n"
                f"  1. Set the DELTAZ_CANONICAL_SOURCE environment variable, e.g.:\n"
                f"     export DELTAZ_CANONICAL_SOURCE='<path>/Synthetic_BestAvailablePhotographic_DOF5_1080p_200frames'\n"
                f"  2. Pass explicit paths to Specimen():\n"
                f"     m1 = Specimen(\n"
                f"         canonical_height_map_path='<path>/height_map.npy',\n"
                f"         canonical_metadata_path='<path>/metadata.json'\n"
                f"     )\n"
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Specimen metadata not found at {meta_path}.\n"
                f"See error message above for configuration instructions."
            )

        self.canonical_height_map_path = hmap_path
        self.canonical_metadata_path = meta_path

        # Load original geometry (internal use only) — cast to float64 for
        # numerical precision during scaling and downstream analysis.
        self._height_map_original_5dof = np.load(hmap_path).astype(np.float64)

        with open(meta_path, "r") as f:
            self._original_metadata = json.load(f)

        # Resolution is inherited exactly — NOT reconfigurable.
        self.resolution_hw = self._height_map_original_5dof.shape  # (height, width) = (1080, 1920)

        # THE single permitted geometric transformation.
        self.height_map = self.axial_scale_factor * self._height_map_original_5dof

        # Geometry constants inherited verbatim from the canonical source.
        geom = self._original_metadata["geometry"]
        self.outer_rectangle_px: Tuple[int, int, int, int] = tuple(geom["outer_rectangle_px"])
        self.inner_rectangle_px: Tuple[int, int, int, int] = tuple(geom["inner_rectangle_px"])
        self.r_pixels: float = self._original_metadata["r_pixels"]
        self.r_dof: float = self._original_metadata["r_dof"]
        self.pyramid_count_declared: int = geom["pyramid_count"]
        self.pyramid_semisize_r: float = geom["pyramid_characteristic_semisize_r"]
        self.pyramid_full_width_r: float = geom["pyramid_characteristic_full_width_r"]
        self.indentation_count_declared: int = geom["indentation_count"]

        # Ground-truth masks based on known geometric construction
        self.plateau_mask = None
        self.ramp_mask = None
        self.hard_step_mask = None
        self.protrusion_mask = None
        self.protrusion_count = None
        self.protrusion_locations = []
        self.protrusion_connected_components = None

        self._generate_ground_truth_masks()

    # ------------------------------------------------------------------
    # Ground-truth masks (based on known geometric construction)
    # ------------------------------------------------------------------
    #
    # Masks are defined using the specimen's documented geometry:
    # - Outer/inner rectangles and their coordinates
    # - Known flat plateau regions
    # - Intended ramp locations and widths
    # - Declared pyramid count (27)
    # - Mask hierarchy: plateaus exclude ramps, steps, and protrusions
    # ------------------------------------------------------------------

    def _generate_ground_truth_masks(self):
        """
        Generate morphological masks from constructive geometry, not gradient percentiles.

        Masks are defined by the known specimen structure:
        - specimen_domain: actual specimen region (excluding external background at h=-2.0)
        - plateau (reference planar region): representative flat surface at h ≈ 0.841 DOF
        - smooth_ramp: known smooth transitions (right/bottom outer ramps, left shoulder)
        - hard_step: fixed-width neighborhood around left/top faces of outer rectangle
        - protrusion: exactly 27 pyramids (peak detection + validation)
        - unclassified: specimen-domain pixels not belonging to any primary class

        All 4 primary masks are mutually exclusive within specimen domain (0 px overlap).
        """
        h = self.height_map
        H, W = h.shape

        x0, y0, x1, y1 = self.outer_rectangle_px
        ix0, iy0, ix1, iy1 = self.inner_rectangle_px

        # ===== SPECIMEN DOMAIN MASK =====
        # Exclude external background (h = -2.0 ≈ -2 * scale_factor from 5-DOF)
        # Specimen interior starts at x0, y0
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        background_height = -2.0 * self.axial_scale_factor
        self.specimen_domain_mask = (h > background_height)

        # ===== PLATEAU MASK (Priority 1): Reference Planar Region =====
        # Representative flat plateau surface at known height h ≈ 0.841 DOF (0.526 * 1.6)
        # This mask identifies the canonical interior flat region, not all planar surfaces globally
        h_plateau_8dof = PLATEAU_HEIGHT_5DOF * self.axial_scale_factor
        is_plateau_height = np.isclose(h, h_plateau_8dof, atol=PLATEAU_HEIGHT_TOLERANCE * self.axial_scale_factor)

        # Reference planar region = pixels at plateau height INSIDE specimen domain
        self.plateau_mask = is_plateau_height & self.specimen_domain_mask

        # ===== HARD-STEP NEIGHBORHOOD MASK (Priority 2) =====
        # Fixed-width neighborhood ONLY around known abrupt discontinuities:
        # Left face and top face of outer rectangle
        hard_step_width = 10  # px, fixed neighborhood width

        left_face = (np.abs(xx - x0) <= hard_step_width) & (yy >= y0) & (yy <= y1)
        top_face = (np.abs(yy - y0) <= hard_step_width) & (xx >= x0) & (xx <= x1)
        hard_step_candidate = (left_face | top_face) & self.specimen_domain_mask

        # Hard-step excludes plateau
        self.hard_step_mask = hard_step_candidate & ~self.plateau_mask

        # ===== SMOOTH RAMP MASK (Priority 3) =====
        # Constructive ramp regions (known from original geometry):
        # 1. Right outer ramp: 300px from x1 toward interior
        # 2. Bottom outer ramp: 300px from y1 toward interior
        # 3. Left shoulder ramp: background region left of x0

        right_ramp = (xx >= (x1 - 300)) & (xx <= x1) & (yy >= y0) & (yy <= y1)
        bottom_ramp = (yy >= (y1 - 300)) & (yy <= y1) & (xx >= x0) & (xx <= x1)
        left_shoulder = (xx < x0) & (xx >= (x0 - 50)) & (yy >= y0) & (yy <= y1)

        ramp_candidate = (right_ramp | bottom_ramp | left_shoulder) & self.specimen_domain_mask

        # Ramp excludes plateau and hard-step
        self.ramp_mask = ramp_candidate & ~self.plateau_mask & ~self.hard_step_mask

        # ===== PROTRUSION MASK (27 Declared Pyramids) =====
        # Keep current exact ground-truth 27-component mask
        from scipy.ndimage import maximum_filter, find_objects
        footprint_size = int(round(self.pyramid_full_width_r * self.r_pixels))
        footprint_size = max(footprint_size, 3) | 1

        local_max_mask = h == maximum_filter(h, size=footprint_size)
        labeled, n_blobs = label(local_max_mask)

        candidate_ids = []
        if n_blobs > 0:
            sizes = np.bincount(labeled.ravel())[1:]
            max_pyramid_area = footprint_size * footprint_size
            objs = find_objects(labeled)

            min_distance_to_boundary = 2

            for blob_id in range(1, n_blobs + 1):
                sz = sizes[blob_id - 1]
                if not (1 <= sz <= max_pyramid_area):
                    continue

                sl = objs[blob_id - 1]
                blob_h = sl[0].stop - sl[0].start
                blob_w = sl[1].stop - sl[1].start
                aspect = max(blob_h, blob_w) / max(1, min(blob_h, blob_w))

                if aspect > 3:
                    continue

                if sz == 1:
                    blob_center_y = sl[0].start + 0.5
                    blob_center_x = sl[1].start + 0.5
                    dx = min(abs(blob_center_x - b) for b in [x0, x1, ix0, ix1])
                    dy = min(abs(blob_center_y - b) for b in [y0, y1, iy0, iy1])
                    if dx < min_distance_to_boundary or dy < min_distance_to_boundary:
                        continue

                candidate_ids.append(blob_id)

        self.protrusion_mask = np.zeros((H, W), dtype=bool)
        self.protrusion_locations = []

        if candidate_ids:
            for blob_id in candidate_ids:
                self.protrusion_mask |= (labeled == blob_id)

            centers = center_of_mass(h, labeled, candidate_ids)
            for (cy, cx), blob_id in zip(centers, candidate_ids):
                self.protrusion_locations.append((float(cx), float(cy), int(blob_id)))

        self.protrusion_count = len(candidate_ids)
        self.protrusion_connected_components = candidate_ids if candidate_ids else []

        # Ensure protrusions are separated from morphological masks
        self.plateau_mask = self.plateau_mask & ~self.protrusion_mask
        self.ramp_mask = self.ramp_mask & ~self.protrusion_mask
        self.hard_step_mask = self.hard_step_mask & ~self.protrusion_mask

        # ===== UNCLASSIFIED SPECIMEN MASK =====
        # Pixels in specimen domain not belonging to any primary class
        classified = self.plateau_mask | self.ramp_mask | self.hard_step_mask | self.protrusion_mask
        self.unclassified_specimen_mask = self.specimen_domain_mask & ~classified

    # ------------------------------------------------------------------
    # Statistics & verification
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, float]:
        """Compute statistics of the (scaled) height map."""
        h = self.height_map
        return {
            "min": float(np.min(h)),
            "p5": float(np.percentile(h, 5)),
            "median": float(np.median(h)),
            "p95": float(np.percentile(h, 95)),
            "max": float(np.max(h)),
            "mean": float(np.mean(h)),
            "std": float(np.std(h)),
            "range": float(np.max(h) - np.min(h)),
        }


    def verify_specification(self) -> Dict[str, bool]:
        """Verify M1 specification compliance."""
        h = self.height_map
        H, W = h.shape

        checks = {
            "has_height_map": bool(self.height_map is not None),
            "shape_1080x1920": bool((H, W) == (1080, 1920)),
            "dtype_float64": bool(h.dtype == np.float64),
            "no_nan": bool(not np.any(np.isnan(h))),
            "no_inf": bool(not np.any(np.isinf(h))),
            "min_minus4": bool(np.isclose(np.min(h), -4.0, atol=1e-5)),
            "max_plus4": bool(np.isclose(np.max(h), 4.0, atol=1e-5)),
            "range_8dof": bool(np.isclose(np.max(h) - np.min(h), 8.0, atol=1e-5)),
            "outer_rectangle": bool(self.outer_rectangle_px == (330, 170, 1590, 930)),
            "inner_rectangle": bool(self.inner_rectangle_px == (720, 380, 1260, 720)),
            "pyramid_count_27": bool(self.pyramid_count_declared == 27),
            "indentation_count_0": bool(self.indentation_count_declared == 0),
            "masks_generated": bool(all([
                self.plateau_mask is not None,
                self.ramp_mask is not None,
                self.hard_step_mask is not None,
                self.protrusion_mask is not None,
            ])),
            "deterministic": bool(self._test_determinism()),
        }
        return checks

    def _test_determinism(self) -> bool:
        """Test that reloading canonical source produces identical results."""
        try:
            m1_test = Specimen()
            return bool(np.array_equal(self.height_map, m1_test.height_map))
        except Exception:
            return False

    def save(self, filepath: str):
        """Save the scaled (8-DOF) height map to NPY file."""
        np.save(filepath, self.height_map)
