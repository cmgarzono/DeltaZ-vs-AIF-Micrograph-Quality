r"""
Photographic model: combines frozen M1 geometry, texture synthesis, and
local shape-shading into the ideal all-in-focus photographic oracle.

Two texture-organization modes (Factor A):
    H - homogeneous:      one texture field, spatially uniform statistics.
    P - plateau-gradual:  two realizations of the SAME spectral regime,
                           blended across a smooth (Gaussian) spatial mask
                           keyed to the frozen inner-rectangle geometry.
                           No hard albedo edges; independent of Factor B.

Four spatial-content regimes (Factor B): LF, COARSE, MIXED, FINE.
Factor A and Factor B are orthogonal: for a fixed spatial-content regime,
H and P use the identical texture synthesis parameters (same regime),
differing only in spatial organization.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .illumination import compute_shape_shading
from .texture import blend_two_textures, smooth_region_blend_mask, synthesize_texture

ORGANIZATIONS = ("H", "P")
SPATIAL_CONTENTS = ("LF", "COARSE", "MIXED", "FINE")

# Fixed default seeds per configuration: deterministic and documented.
# (Any seed may be overridden explicitly via PhotographicModel(seed=...).)
_DEFAULT_SEED_BASE = 20260818  # date-derived, arbitrary but fixed & documented

GATE2_CONFIGURATIONS = tuple(
    f"{org}_{content}" for org in ORGANIZATIONS for content in SPATIAL_CONTENTS
)

# P-mode blend transition width (px). Large relative to hard-step width (10px)
# so the transition is unambiguously gradual, never a hard edge.
_P_BLEND_SIGMA_PX = 180.0

# P-mode uses two orientations of the same regime (orthogonality preserved:
# spectral content is identical, only orientation/realization differs).
_P_ORIENTATION_A_DEG = 45.0
_P_ORIENTATION_B_DEG = -20.0
_H_ORIENTATION_DEG = 45.0


@dataclass(frozen=True)
class PhotographicConfig:
    organization: str  # "H" or "P"
    spatial_content: str  # "LF" | "COARSE" | "MIXED" | "FINE"
    seed: int

    @property
    def name(self) -> str:
        return f"{self.organization}_{self.spatial_content}"


def _config_seed(organization: str, spatial_content: str) -> int:
    """Deterministic default seed derived from configuration identity."""
    idx = ORGANIZATIONS.index(organization) * len(SPATIAL_CONTENTS) + SPATIAL_CONTENTS.index(spatial_content)
    return _DEFAULT_SEED_BASE + idx


class PhotographicModel:
    """
    Ideal all-in-focus photographic oracle generator for a frozen M1 specimen.

    Does not read or write M1 geometry, height map, or specimen masks other
    than as read-only inputs. Texture is generated once per (configuration,
    seed) and is fixed thereafter -- callers must reuse the same instance
    (or the same seed) across focal planes rather than resynthesizing texture.
    """

    def __init__(self, m1_specimen, organization: str, spatial_content: str, seed: int = None):
        if organization not in ORGANIZATIONS:
            raise ValueError(f"organization must be one of {ORGANIZATIONS}")
        if spatial_content not in SPATIAL_CONTENTS:
            raise ValueError(f"spatial_content must be one of {SPATIAL_CONTENTS}")

        self.m1 = m1_specimen
        self.organization = organization
        self.spatial_content = spatial_content
        self.seed = seed if seed is not None else _config_seed(organization, spatial_content)
        self.config = PhotographicConfig(organization, spatial_content, self.seed)

        self._texture = None  # A(x, y), fixed once computed
        self._shading = None
        self._dilated_protrusion_mask = None
        self._oracle = None

    @property
    def shape(self) -> Tuple[int, int]:
        return self.m1.height_map.shape

    # ------------------------------------------------------------------
    # Texture A(x, y) -- fixed for this (configuration, seed)
    # ------------------------------------------------------------------
    def texture(self) -> np.ndarray:
        """Return the fixed specimen texture A(x, y). Cached after first call."""
        if self._texture is not None:
            return self._texture

        if self.organization == "H":
            tex = synthesize_texture(
                self.shape, seed=self.seed, regime=self.spatial_content,
                orientation_deg=_H_ORIENTATION_DEG,
            )
        else:  # "P" — plateau-gradual: same regime, two realizations, smooth blend
            tex_a = synthesize_texture(
                self.shape, seed=self.seed, regime=self.spatial_content,
                orientation_deg=_P_ORIENTATION_A_DEG,
            )
            tex_b = synthesize_texture(
                self.shape, seed=self.seed + 100000, regime=self.spatial_content,
                orientation_deg=_P_ORIENTATION_B_DEG,
            )
            blend_mask = smooth_region_blend_mask(
                self.shape, self.m1.inner_rectangle_px, sigma_px=_P_BLEND_SIGMA_PX
            )
            tex = blend_two_textures(tex_a, tex_b, blend_mask)

        self._texture = tex
        return tex

    # ------------------------------------------------------------------
    # Illumination / local shape response
    # ------------------------------------------------------------------
    def shading(self) -> np.ndarray:
        """Return the multiplicative shading field (1.0 outside protrusions)."""
        if self._shading is not None:
            return self._shading
        shading, dilated_mask = compute_shape_shading(
            self.m1.height_map, self.m1.protrusion_mask, self.m1.r_pixels
        )
        self._shading = shading
        self._dilated_protrusion_mask = dilated_mask
        return shading

    @property
    def dilated_protrusion_mask(self) -> np.ndarray:
        if self._dilated_protrusion_mask is None:
            self.shading()
        return self._dilated_protrusion_mask

    # ------------------------------------------------------------------
    # Ideal all-in-focus oracle I*(x, y)
    # ------------------------------------------------------------------
    def all_in_focus_oracle(self) -> np.ndarray:
        """
        Ideal all-in-focus photographic appearance: texture modulated by
        local shape-shading. No noise, no defocus. Deterministic and cached.
        """
        if self._oracle is not None:
            return self._oracle

        tex = self.texture()
        shade = self.shading()
        oracle = np.clip(tex * shade, 0.0, 1.0)
        self._oracle = oracle
        return oracle

    def statistics(self) -> Dict[str, float]:
        oracle = self.all_in_focus_oracle()
        return {
            "mean_intensity": float(oracle.mean()),
            "std_intensity": float(oracle.std()),
        }
