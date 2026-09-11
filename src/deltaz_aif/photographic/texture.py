r"""
Spectral texture synthesis.

Generates deterministic, band-limited, oriented (anisotropic) textures via
FFT filtering of white Gaussian noise. Four controlled spatial-content
regimes are supported (LF, COARSE, MIXED, FINE), forming an ordered
low-to-high frequency progression while remaining locally informative
(no regime collapses to a near-flat / textureless field).

Textures are photographic-looking broadband fields, not periodic charts:
filters are smooth (Gaussian radial bands or 1/f^p power-law), and the
white-noise phase is random per seed, so no exact periodicity is present.
"""

from typing import Tuple

import numpy as np

# Spatial-content regime definitions.
# Frequencies are expressed as a fraction of the Nyquist frequency (0.5 cycles/px).
REGIMES = ("LF", "COARSE", "MIXED", "FINE")

_REGIME_PARAMS = {
    # Gaussian radial band: (center, sigma), plus a broadband floor mix-in
    # fraction (of amplitude) so the regime is never textureless.
    "LF": dict(kind="band", center=0.018, sigma=0.014, floor_mix=0.16),
    "COARSE": dict(kind="band", center=0.055, sigma=0.028, floor_mix=0.08),
    # MIXED is a literal broadband combination of coarse + mid + fine bands
    # (not a global power law), guaranteeing it sits between COARSE and FINE
    # as an intermediate photographic regime.
    "MIXED": dict(
        kind="mix",
        bands=[(0.055, 0.028, 0.32), (0.11, 0.05, 0.32), (0.20, 0.07, 0.36)],
        floor_mix=0.05,
    ),
    "FINE": dict(kind="band", center=0.20, sigma=0.07, floor_mix=0.05),
}


def _radial_filter(R: np.ndarray, regime: str) -> np.ndarray:
    """Build the radial magnitude filter (unnormalized) for a spatial-content regime."""
    params = _REGIME_PARAMS[regime]
    if params["kind"] == "band":
        center, sigma = params["center"], params["sigma"]
        band = np.exp(-0.5 * ((R - center) / sigma) ** 2)
        # Broadband floor: gentle 1/f^0.9 tail mixed in so the regime retains
        # residual structure across scales (esp. important for LF).
        with np.errstate(divide="ignore"):
            floor = 1.0 / np.maximum(R, 1e-4) ** 0.9
        floor = floor / floor.max()
        mix = params["floor_mix"]
        filt = (1.0 - mix) * band + mix * floor
        return filt
    elif params["kind"] == "mix":
        combined = np.zeros_like(R)
        for center, sigma, weight in params["bands"]:
            combined += weight * np.exp(-0.5 * ((R - center) / sigma) ** 2)
        with np.errstate(divide="ignore"):
            floor = 1.0 / np.maximum(R, 1e-4) ** 0.9
        floor = floor / floor.max()
        mix = params["floor_mix"]
        filt = (1.0 - mix) * combined + mix * floor
        return filt
    else:
        raise ValueError(f"Unknown regime kind for {regime}")


def _orientation_filter(THETA: np.ndarray, orientation_deg: float, anisotropy: float) -> np.ndarray:
    """Directional weighting producing an oriented (hatched) texture character."""
    theta0 = np.deg2rad(orientation_deg)
    w = 1.0 + anisotropy * np.cos(2.0 * (THETA - theta0))
    return np.clip(w, 0.0, None)


def synthesize_texture(
    shape: Tuple[int, int],
    seed: int,
    regime: str,
    orientation_deg: float = 45.0,
    anisotropy: float = 0.45,
    mean_level: float = 0.55,
    contrast: float = 0.09,
) -> np.ndarray:
    """
    Synthesize one deterministic, band-limited texture field A(x, y).

    Args:
        shape: (H, W) output resolution.
        seed: RNG seed; identical seed + regime + orientation -> identical texture.
        regime: one of REGIMES ("LF", "COARSE", "MIXED", "FINE").
        orientation_deg: dominant hatch orientation in degrees.
        anisotropy: 0 = isotropic, >0 = increasingly oriented/hatched.
        mean_level: target mean intensity in [0, 1].
        contrast: target intensity standard deviation.

    Returns:
        float64 array of shape (H, W), values in [0, 1].
    """
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")

    H, W = shape
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((H, W))

    fy = np.fft.fftfreq(H)
    fx = np.fft.fftfreq(W)
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX**2 + FY**2)
    THETA = np.arctan2(FY, FX)

    radial = _radial_filter(R, regime)
    orient = _orientation_filter(THETA, orientation_deg, anisotropy)
    filt = radial * orient
    filt[0, 0] = 0.0  # remove DC (mean handled separately)

    F = np.fft.fft2(white)
    tex = np.real(np.fft.ifft2(F * filt))

    tex_std = tex.std()
    if tex_std < 1e-12:
        tex_std = 1.0
    tex = (tex - tex.mean()) / tex_std

    image = mean_level + contrast * tex
    return np.clip(image, 0.0, 1.0)


def blend_two_textures(
    tex_a: np.ndarray,
    tex_b: np.ndarray,
    blend_mask: np.ndarray,
) -> np.ndarray:
    """
    Spatially blend two texture fields using a smooth (already-blurred,
    values in [0, 1]) blend mask. blend_mask == 1 -> fully tex_b,
    blend_mask == 0 -> fully tex_a. No hard edges are introduced here;
    the caller is responsible for supplying a pre-smoothed mask.
    """
    return tex_a * (1.0 - blend_mask) + tex_b * blend_mask


def smooth_region_blend_mask(
    shape: Tuple[int, int],
    rect_px: Tuple[int, int, int, int],
    sigma_px: float,
) -> np.ndarray:
    """
    Build a smooth [0, 1] spatial blend mask that is ~1 inside rect_px and
    ~0 outside, with a gradual (Gaussian-blurred) transition of width
    ~sigma_px. Used for the P (plateau-gradual) texture-organization mode.
    """
    from scipy.ndimage import gaussian_filter

    H, W = shape
    x0, y0, x1, y1 = rect_px
    mask = np.zeros((H, W), dtype=np.float64)
    mask[y0:y1, x0:x1] = 1.0
    mask = gaussian_filter(mask, sigma=sigma_px)
    lo, hi = mask.min(), mask.max()
    if hi - lo > 1e-9:
        mask = (mask - lo) / (hi - lo)
    return mask
