r"""
Numerical helpers for the Gaussian-PSF defocus core.

These functions generate a pure sinusoidal test signal at f_ref, apply the
1-D Gaussian blur implied by sigma_from_defocus(d) with reflective boundary
conditions, and measure the resulting contrast. They compare
`psf_core.sigma_from_defocus` / `psf_core.mtf_gaussian` with a direct
numerical measurement; they do not touch height_map, geometry, or any
depth-layer rendering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .psf_core import F_REF, mtf_gaussian, sigma_from_defocus

# ---------------------------------------------------------------------------
# Sinusoidal test-signal generation
# ---------------------------------------------------------------------------

# f_ref = 0.032 = 4/125 cycles/pixel exactly -> a period of 125 samples
# contains exactly 4 cycles. Using a sample count that is a multiple of 125
# gives an exact integer number of cycles, eliminating spectral leakage from
# window truncation without needing any tapering.
_PERIOD_SAMPLES = 125  # exact period multiplier s.t. f_ref * _PERIOD_SAMPLES = 4 (integer cycles)


def make_sinusoid(f_ref: float = F_REF, n_periods: int = 128, amplitude: float = 1.0) -> np.ndarray:
    """
    Pure sinusoid at spatial frequency f_ref, zero-mean (no unnecessary DC),
    spanning an exact integer number of cycles to avoid windowing error.
    """
    n_samples = _PERIOD_SAMPLES * n_periods
    n = np.arange(n_samples, dtype=np.float64)
    return amplitude * np.sin(2.0 * np.pi * f_ref * n)


def measure_contrast(signal: np.ndarray, f_ref: float = F_REF, crop_periods: int = 2) -> float:
    """
    Measure the amplitude of the f_ref component of `signal` via direct
    quadrature (matched-filter) projection onto sin/cos at f_ref, after
    cropping `crop_periods` full periods from each edge to exclude any
    reflective-boundary transient from the contrast estimate.

    This is robust to phase and does not rely on peak-picking, so it is
    insensitive to isolated boundary artifacts as long as they are cropped.
    """
    crop = _PERIOD_SAMPLES * crop_periods
    core = signal[crop:-crop]
    n = np.arange(crop, crop + core.size, dtype=np.float64)
    cos_ref = np.cos(2.0 * np.pi * f_ref * n)
    sin_ref = np.sin(2.0 * np.pi * f_ref * n)
    a = 2.0 / core.size * np.dot(core, cos_ref)
    b = 2.0 / core.size * np.dot(core, sin_ref)
    return float(np.hypot(a, b))


def blurred_contrast(d: float, f_ref: float = F_REF, n_periods: int = 128) -> float:
    """Apply sigma_from_defocus(d) Gaussian blur (reflective boundary) to a
    pure f_ref sinusoid and return the measured contrast amplitude."""
    signal = make_sinusoid(f_ref=f_ref, n_periods=n_periods)
    sigma = float(sigma_from_defocus(d))
    if sigma <= 0.0:
        blurred = signal
    else:
        blurred = gaussian_filter1d(signal, sigma=sigma, mode="reflect")
    return measure_contrast(blurred, f_ref=f_ref)


@dataclass
class CalibrationRow:
    d: float
    sigma_px: float
    mtf_analytical: float
    contrast_ratio_numeric: float
    abs_error: float
    rel_error: float


def run_calibration_table(d_values, f_ref: float = F_REF, n_periods: int = 128) -> list[CalibrationRow]:
    """Build the d / sigma / analytical MTF / numeric contrast-ratio table."""
    c0 = blurred_contrast(0.0, f_ref=f_ref, n_periods=n_periods)
    rows = []
    for d in d_values:
        sigma = float(sigma_from_defocus(d))
        mtf_a = float(mtf_gaussian(f_ref, sigma))
        c_d = blurred_contrast(d, f_ref=f_ref, n_periods=n_periods)
        ratio = c_d / c0
        abs_err = abs(ratio - mtf_a)
        rel_err = abs_err / mtf_a if mtf_a > 0 else float("nan")
        rows.append(CalibrationRow(d, sigma, mtf_a, ratio, abs_err, rel_err))
    return rows


# ---------------------------------------------------------------------------
# Boundary-condition check
# ---------------------------------------------------------------------------

def boundary_condition_report(sigma: float, n_samples: int = 512, value: float = 1.0) -> dict:
    """
    Confirm reflective-boundary Gaussian blur of a CONSTANT non-negative
    signal introduces no black edges / spurious intensity loss (which a
    zero-padded convolution would produce).
    """
    signal = np.full(n_samples, value, dtype=np.float64)
    blurred = gaussian_filter1d(signal, sigma=sigma, mode="reflect")
    return {
        "min_value": float(blurred.min()),
        "max_value": float(blurred.max()),
        "edge_first": float(blurred[0]),
        "edge_last": float(blurred[-1]),
        "mean_value": float(blurred.mean()),
        "expected_value": value,
        "max_abs_deviation": float(np.max(np.abs(blurred - value))),
    }
