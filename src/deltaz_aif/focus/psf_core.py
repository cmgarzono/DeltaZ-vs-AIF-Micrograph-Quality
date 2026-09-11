r"""
Through-focus mathematical core (effective phenomenological Gaussian PSF).

Scope: ONLY the isolated defocus -> sigma -> MTF mapping. No height map,
no depth-layer rendering, no specimen geometry, no through-focus stacks,
no noise, no focus fusion -- those live in
deltaz_aif.rendering.depth_layer_renderer and deltaz_aif.analysis.

Model
-----
Defocus variable:      d = h - z          (NOT computed in this module)
PSF (phenomenological): MTF(f; sigma) = exp(-2 * pi^2 * sigma^2 * f^2)
Reference frequency:    f_ref = 0.032 cycles/pixel
Calibration condition:  MTF(f_ref, sigma(|d| = 0.5 DOF)) = 0.5
                        => sigma(0.5) ~= 5.86 px
                        => sigma(d)  ~= 11.71 * |d|  px   (fixed, uncapped)

The slope (px per DOF) is derived analytically from the calibration
condition rather than hardcoded.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

F_REF: float = 0.032  # cycles/pixel, reference spatial frequency for DOF definition
CALIBRATION_D: float = 0.5  # DOF units at which MTF(f_ref, sigma) is pinned
CALIBRATION_MTF: float = 0.5  # target MTF value at CALIBRATION_D


def _calibrated_slope(
    f_ref: float = F_REF,
    target_d: float = CALIBRATION_D,
    target_mtf: float = CALIBRATION_MTF,
) -> float:
    """
    Analytically derive the sigma-per-DOF slope from the calibration
    condition MTF(f_ref, sigma(target_d)) = target_mtf, i.e.:

        exp(-2 pi^2 sigma^2 f_ref^2) = target_mtf
        sigma = sqrt( -ln(target_mtf) / (2 pi^2 f_ref^2) )
        slope = sigma / target_d
    """
    sigma_at_target = np.sqrt(-np.log(target_mtf) / (2.0 * np.pi**2 * f_ref**2))
    return float(sigma_at_target / target_d)


# sigma(d) ~= SIGMA_PER_DOF * |d|  [px].  Evaluates to ~11.7108, matching the
# fixed 11.71 value.
SIGMA_PER_DOF: float = _calibrated_slope()

# sigma at the |d| = 0.5 DOF calibration point, for reference/reporting.
SIGMA_AT_HALF_DOF: float = SIGMA_PER_DOF * CALIBRATION_D


def sigma_from_defocus(d):
    """
    Map axial defocus d (in DOF units) to Gaussian PSF sigma (in pixels).

    sigma(d) = SIGMA_PER_DOF * |d|

    Symmetric in the sign of d. No cap is applied.

    Parameters
    ----------
    d : scalar or array_like
        Axial defocus in DOF units. d = h - z is NOT computed here; the
        caller supplies it.

    Returns
    -------
    sigma : same shape as d (float64), Gaussian PSF sigma in pixels.
    """
    d_arr = np.asarray(d, dtype=np.float64)
    return SIGMA_PER_DOF * np.abs(d_arr)


def mtf_gaussian(f, sigma):
    """
    Analytical MTF of a Gaussian PSF with standard deviation sigma (px),
    evaluated at spatial frequency f (cycles/pixel):

        MTF(f; sigma) = exp(-2 * pi^2 * sigma^2 * f^2)

    Parameters
    ----------
    f : scalar or array_like
        Spatial frequency, cycles/pixel.
    sigma : scalar or array_like
        Gaussian PSF standard deviation, pixels. Broadcastable against f.

    Returns
    -------
    mtf : float64, same broadcast shape as (f, sigma).
    """
    f_arr = np.asarray(f, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    return np.exp(-2.0 * np.pi**2 * sigma_arr**2 * f_arr**2)


def mtf_from_defocus(d, f=F_REF):
    """Convenience: analytical MTF(f, sigma(d)) directly from defocus d."""
    return mtf_gaussian(f, sigma_from_defocus(d))
