"""
Baseline photographic model: shared rendering primitives used by the
photographic-condition generators.

Pipeline:
    1. Extract detail_height (pyramids only, no macro steps) from the
       specimen height map
    2. Render photographic base using detail_height normals:
       - Multi-light Lambertian + specular + occlusion + cast shadow
       - Percentile normalization + gamma
    3. Add texture (deterministic sine-wave composition):
       - H: uniform, amplitude 7.5 gray levels
       - P: 5-region oriented, amplitude 14.0 gray levels
    4. Clip to [26, 214] uint8

Critical: normals computed from detail_height (pyramid-only), NOT from
full height map. This prevents macro-step edges from appearing as lines.
"""

import numpy as np
from math import pi
from scipy.ndimage import gaussian_filter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from deltaz_aif.specimen import Specimen


# ── Constants ────────────────────────────────────────────────────────────

WIDTH = 1920
HEIGHT = 1080
R_PIXELS = 5.0

# Pyramid footprint half-width (~3.1 R * 5 px/R ≈ 15.4 px)
PYRAMID_HALF_WIDTH_PX = 16

# Analytic pyramid apex amplitude (DOF), used for detail_height reconstruction.
# Median elevation of the 19 well-formed (non height-ceiling-clipped) declared
# pyramids above their own local plateau, measured via perimeter-ring median
# on the frozen specimen height map. Applied uniformly to all 27 declared pyramids.
PYRAMID_ANALYTIC_AMPLITUDE_DOF = 1.16

DEFAULT_SEED = 20260818


# ── Step 1: Extract detail_height ────────────────────────────────────────

def _extract_detail_height(m1):
    """
    Build pyramid-only height detail via ANALYTIC reconstruction.

    detail_height contains ONLY the pyramid protrusion geometry, with zero
    everywhere else. Rather than reading each pyramid's relief out of the
    frozen height_map (Specimen), each of the 27 declared pyramids is
    stamped as an ideal square-base pyramid (Chebyshev/L-infinity cone) of
    fixed half-width and fixed amplitude, centered at its known, frozen
    (cx, cy) location.

    Why analytic reconstruction instead of extraction-from-height-map:
    Comparing every one of the 27 declared locations against the canonical
    reference all_in_focus oracle shows ALL 27 rendered as identical, crisp,
    sharp-apex square pyramids -- including the locations where extraction
    from the real height_map fails for two independent, verified
    reasons:
      1. Eight declared pyramids sit on the elevated inner-rectangle region
         where local height meets exactly the frozen +4.0 DOF axial-range
         ceiling. Reading their relief out of height_map reproduces
         that ceiling clip as a flat-topped frustum (a visible nested-square
         "double outline"), which the canonical reference does not show.
      2. At least one declared pyramid sits at the very corner of a small
         plateau immediately adjacent to a hard-step/ramp; any footprint- or
         ring-based baseline estimate there is inherently fighting real
         adjacent macro terrain and yields a shallower, partial silhouette,
         which the canonical reference also does not show.
    Both are genuine properties of the frozen specimen height map, not extraction
    bugs -- but the canonical photographic MODEL clearly does not derive
    pyramid shading from that raw data either; it stamps a consistent ideal
    pyramid at each declared location. This matches the explicit guidance
    to reconstruct pyramids analytically from known parameters rather than
    extract-and-clean from height_map.

    height_map itself is never read or modified here -- only the frozen,
    fixed (cx, cy) center list from m1.protrusion_locations is used.
    Through-focus behavior (Phase 3+) still uses the real height_map;
    this function only supplies local shading detail for the all-in-focus
    photographic oracle.

    Amplitude and half-width are fixed constants representative of the
    frozen specimen (median of the well-formed, non-clipped declared
    pyramids' real elevation above their local plateau, per the specimen geometry):
    ~1.16 DOF, ~16 px half-width -- consistent with the canonical reference's
    uniform pyramid appearance across all 27 locations.

    Returns:
        float32 array (1080, 1920), zero outside the 27 pyramid footprints
    """
    h = m1.height_map
    H, W = h.shape
    r = PYRAMID_HALF_WIDTH_PX
    amplitude = PYRAMID_ANALYTIC_AMPLITUDE_DOF

    detail = np.zeros((H, W), dtype=np.float32)

    for cx, cy, _ in m1.protrusion_locations:
        cx_i, cy_i = int(round(cx)), int(round(cy))

        y0 = max(0, cy_i - r - 1)
        y1 = min(H, cy_i + r + 2)
        x0 = max(0, cx_i - r - 1)
        x1 = min(W, cx_i + r + 2)

        yy, xx = np.mgrid[y0:y1, x0:x1]
        cheby = np.maximum(np.abs(yy - cy_i), np.abs(xx - cx_i)).astype(np.float32)

        # Ideal square-base pyramid: perfectly linear falloff from apex
        # (amplitude, at the center) to 0 at Chebyshev distance r. Constant
        # slope per face-half produces the flat-shaded triangular facets
        # (uniform light/dark diagonal split) seen in the canonical
        # reference, rather than a rounded/gradient cone.
        pyramid = amplitude * np.clip(1.0 - cheby / r, 0.0, 1.0)

        detail[y0:y1, x0:x1] = np.maximum(detail[y0:y1, x0:x1], pyramid)

    return detail


# ── Step 2: Render photographic base ─────────────────────────────────────

def _render_photographic_base(detail_height):
    """
    Render the untextured all-in-focus base using the canonical recipe.

    Multi-light Lambertian illumination + specular + occlusion + cast shadow,
    all computed from detail_height (pyramid-only normals).

    Returns:
        uint8 array (1080, 1920)
    """
    # Smooth detail for clean normals (no aliasing)
    render_height = gaussian_filter(detail_height.astype(np.float32),
                                     sigma=0.45, mode="nearest")

    # Gradients scaled by R_PIXELS
    gy, gx = np.gradient(render_height)
    dzdx = gx * R_PIXELS
    dzdy = gy * R_PIXELS

    # Surface normals with gain
    normal_gain = 1.35
    normal_x = -normal_gain * dzdx
    normal_y = -normal_gain * dzdy
    normal_z = np.ones_like(render_height, dtype=np.float32)
    normal_norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    nx = normal_x / normal_norm
    ny = normal_y / normal_norm
    nz = normal_z / normal_norm

    # Multi-light Lambertian illumination
    light_vectors = [
        np.array([-0.78, -0.54, 0.90], dtype=np.float32),
        np.array([-0.60, -0.72, 0.92], dtype=np.float32),
        np.array([-0.92, -0.30, 0.84], dtype=np.float32),
        np.array([0.30,  0.18, 1.00], dtype=np.float32),
    ]
    light_weights = [0.46, 0.24, 0.18, 0.12]

    lambert = np.zeros_like(render_height, dtype=np.float32)
    for light_vec, weight in zip(light_vectors, light_weights):
        lv = light_vec / np.linalg.norm(light_vec)
        lambert += weight * np.clip(
            nx * lv[0] + ny * lv[1] + nz * lv[2], 0.0, 1.0
        )

    # Specular highlight (soft, from primary light)
    view = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    light0 = light_vectors[0] / np.linalg.norm(light_vectors[0])
    half_vec = light0 + view
    half_vec = half_vec / np.linalg.norm(half_vec)
    specular = np.clip(
        nx * half_vec[0] + ny * half_vec[1] + nz * half_vec[2], 0.0, 1.0
    ) ** 18

    # Occlusion from local slopes
    slope = np.sqrt(dzdx**2 + dzdy**2)
    occlusion = np.clip(
        gaussian_filter(np.clip(slope * 0.22, 0.0, 1.0),
                        sigma=1.2, mode="reflect"),
        0.0, 0.28
    )

    # Cast shadows (shifted slope fields)
    shadow_source = np.clip(slope * 0.22, 0.0, 1.0)
    cast_shadow = np.zeros_like(render_height, dtype=np.float32)
    for shift, weight in ((3, 0.035), (7, 0.025), (13, 0.015)):
        cast_shadow += weight * np.roll(
            np.roll(shadow_source, shift, axis=0), shift, axis=1
        )
    cast_shadow = gaussian_filter(
        np.clip(cast_shadow, 0.0, 0.12), sigma=3.2, mode="reflect"
    )

    # Compose photographic image
    material = np.full(render_height.shape, 0.58, dtype=np.float32)
    photographic = material * (0.40 + 0.62 * lambert)
    photographic *= 1.0 - 0.07 * occlusion
    photographic *= 1.0 - 0.08 * cast_shadow
    photographic += 0.035 * specular

    # Percentile normalization + gamma
    low, high = np.percentile(photographic, [0.2, 99.8])
    photographic = np.clip(
        (photographic - low) / max(high - low, 1e-6), 0.0, 1.0
    )
    base = 38.0 + 162.0 * (photographic ** 0.92)
    base = np.rint(np.clip(base, 26.0, 214.0)).astype(np.uint8)

    return base


# ── Step 3a: H texture (homogeneous) ────────────────────────────────────

def _make_h_texture():
    """
    Exact homogeneous texture from canonical recipe.

    5 sine-wave components. phase = 0.07. Smoothed sigma=0.55.
    Returns zero-mean unit-std texture field.
    """
    y = np.linspace(-1.0, 1.0, HEIGHT, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, WIDTH, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    xpix = (xx + 1.0) * 0.5 * WIDTH
    ypix = (yy + 1.0) * 0.5 * HEIGHT

    texture = (
        0.58 * np.sin(2 * pi * (xpix / 38.0 + ypix / 143.0 + 0.07))
        + 0.42 * np.sin(2 * pi * (-xpix / 91.0 + ypix / 46.0 + 0.37 + 0.07))
        + 0.28 * np.sin(2 * pi * (xpix / 17.0 - ypix / 29.0 + 0.11 + 0.07))
        + 0.18 * np.sin(2 * pi * (xpix / 71.0 + 0.23 + 0.07))
        + 0.16 * np.sin(2 * pi * (ypix / 63.0 + 0.61 + 0.07))
    )
    texture = gaussian_filter(texture, sigma=0.55, mode="reflect")
    texture = (texture - texture.mean()) / texture.std()
    return texture


# ── Step 3b: P texture (plateau-gradual) ────────────────────────────────

def _rect_hard(xpix, ypix, x0, y0, x1, y1):
    """Hard rectangular mask in pixel coordinates."""
    return ((xpix >= x0) & (xpix < x1) & (ypix >= y0) & (ypix < y1)).astype(np.float32)


def _oriented_texture(xx, yy, angle, scale, phase):
    """
    Single oriented texture patch (4 sine-wave components rotated by angle).
    Returns zero-mean unit-std field.
    """
    xpix = (xx + 1.0) * 0.5 * WIDTH
    ypix = (yy + 1.0) * 0.5 * HEIGHT
    ca = np.cos(angle)
    sa = np.sin(angle)
    u = ca * xpix + sa * ypix
    v = -sa * xpix + ca * ypix

    texture = (
        0.70 * np.sin(2 * pi * (u / (24.0 * scale) + phase))
        + 0.36 * np.sin(2 * pi * (v / (58.0 * scale) + 0.27 + phase))
        + 0.24 * np.sin(2 * pi * ((u + 0.55 * v) / (39.0 * scale) + 0.63 + phase))
        + 0.13 * np.sin(2 * pi * ((u - 0.85 * v) / (16.0 * scale) + 0.41 + phase))
    )
    texture = gaussian_filter(texture, sigma=0.45, mode="reflect")
    return (texture - texture.mean()) / texture.std()


def _make_p_texture():
    """
    Exact plateau-gradual texture from canonical recipe.

    5 regions with different orientations, blended with sigma=18px smoothing.
    Returns zero-mean unit-std blended field.
    """
    y = np.linspace(-1.0, 1.0, HEIGHT, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, WIDTH, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    xpix = (xx + 1.0) * 0.5 * WIDTH
    ypix = (yy + 1.0) * 0.5 * HEIGHT

    # Region masks (pixel coordinates, exact canonical values)
    outer = _rect_hard(xpix, ypix, 330, 170, 1590, 930)
    inner = _rect_hard(xpix, ypix, 720, 380, 1260, 720)
    left = _rect_hard(xpix, ypix, 330, 170, 480, 930) * outer
    top = _rect_hard(xpix, ypix, 330, 170, 1590, 285) * outer
    right = _rect_hard(xpix, ypix, 1290, 170, 1590, 930) * outer
    bottom = _rect_hard(xpix, ypix, 330, 740, 1590, 930) * outer

    main_outer = outer * (1.0 - np.maximum.reduce([inner, left, top, right, bottom]))
    outside = 1.0 - outer
    left_top = np.maximum(left, top)
    right_bottom = np.maximum(right, bottom)

    # Raw weights
    raw_weights = [outside, main_outer, inner, left_top, right_bottom]

    # Smooth for gradual transitions (sigma=18px)
    weights = [gaussian_filter(w, sigma=18.0, mode="reflect") for w in raw_weights]
    total = sum(weights)
    total = np.maximum(total, 1e-8)
    weights = [w / total for w in weights]

    # Per-region oriented textures (exact canonical angles/scales/phases)
    textures = [
        _oriented_texture(xx, yy, angle=0.02,  scale=1.25, phase=0.13),  # outside
        _oriented_texture(xx, yy, angle=0.72,  scale=0.86, phase=0.31),  # main_outer
        _oriented_texture(xx, yy, angle=-0.48, scale=1.08, phase=0.53),  # inner
        _oriented_texture(xx, yy, angle=1.22,  scale=0.72, phase=0.79),  # left_top
        _oriented_texture(xx, yy, angle=-0.95, scale=1.48, phase=1.03),  # right_bottom
    ]

    # Blend
    blended = sum(w * t for w, t in zip(weights, textures))
    blended = (blended - blended.mean()) / blended.std()
    return blended


# ── Step 4: Full oracle generation ──────────────────────────────────────

class BaselinePhotographicModel:
    """
    Exact port of canonical photographic rendering.

    H_BASE: base + 7.5 * homogeneous_texture
    P_BASE: base + 14.0 * plateau_gradual_texture

    Where base = multi-light Lambertian render from detail_height (pyramids only).
    """

    def __init__(self, m1, mode='H', seed=DEFAULT_SEED):
        if mode not in ('H', 'P'):
            raise ValueError(f"mode must be 'H' or 'P', got {mode!r}")
        self.m1 = m1
        self.mode = mode
        self.seed = seed
        self._oracle = None

    def generate_oracle(self):
        """
        Generate all-in-focus photographic oracle.

        Returns:
            float64 array (1080, 1920), values in [0, 1]
        """
        if self._oracle is not None:
            return self._oracle

        # Extract pyramid-only detail height
        detail_height = _extract_detail_height(self.m1)

        # Render photographic base (multi-light shading from detail_height)
        base = _render_photographic_base(detail_height)

        # Generate texture
        if self.mode == 'H':
            texture = _make_h_texture()
            amplitude = 7.5
        else:
            texture = _make_p_texture()
            amplitude = 14.0

        # Additive combination (exact canonical recipe)
        textured = base.astype(np.float32) + amplitude * texture
        textured = np.rint(np.clip(textured, 26.0, 214.0)).astype(np.uint8)

        # Convert to float64 [0, 1]
        self._oracle = textured.astype(np.float64) / 255.0
        return self._oracle
