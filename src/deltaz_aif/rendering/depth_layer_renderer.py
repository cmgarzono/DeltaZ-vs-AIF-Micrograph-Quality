r"""
Depth-layer through-focus renderer.

Implements a spatially-varying through-focus rendering strategy:

    depth-layer rendering
    + linear-hat axial weights
    + normalized masked convolution

    I_k(x,y) = [ sum_j G_{sigma_j} * (I* w_j) ] / [ sum_j G_{sigma_j} * w_j ]

where:
    I*        = oracle all-in-focus image (fixed texture; never regenerated
                between planes).
    h(x,y)    = height_map -- the ONLY source of axial depth for defocus.
    d(x,y;k)  = h(x,y) - z_k
    sigma_j   = sigma_from_defocus(layer_height_j - z_k)   [psf_core, unmodified]
    w_j(x,y)  = linear-hat axial interpolation weight of pixel (x,y) with
                respect to depth layer j.

Properties enforced by construction:
    w_j(x,y) >= 0
    sum_j w_j(x,y) == 1              (partition of unity)
    at most 2 layers are active (nonzero weight) per pixel

Boundary condition: reflective (matches psf_core's `psf_validation` boundary).
No noise, no texture regeneration, no focus fusion, no artificial sigma cap.

Scope: this module implements the single-plane depth-layer renderer. The
four experiments (axial coverage, spectral evidence, sampling/fusion,
multiplicity/noise) live in deltaz_aif.analysis.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from deltaz_aif.focus.psf_core import sigma_from_defocus

__all__ = [
    "build_depth_layers",
    "compute_hat_weights",
    "gaussian_blur_reflect",
    "render_depth_layer",
]


# ---------------------------------------------------------------------------
# A. Depth layer construction
# ---------------------------------------------------------------------------


def build_depth_layers(height_map: np.ndarray, delta_h: float) -> np.ndarray:
    """
    Build a uniform grid of depth-layer heights that fully covers the real
    range of `height_map` (NOT an assumed [-4, +4] DOF range -- the actual
    min/max of the array are measured and used).

    Parameters
    ----------
    height_map : array_like
        height_map, the sole source of axial depth (DOF units).
    delta_h : float
        Axial layer spacing (DOF units), e.g. 0.125 or 0.0625.

    Returns
    -------
    layers : np.ndarray, shape (n_layers,), float64, ascending, uniform
        spacing `delta_h`. Guaranteed layers[0] <= h.min() and
        layers[-1] >= h.max().
    """
    if delta_h <= 0.0:
        raise ValueError(f"delta_h must be > 0, got {delta_h}")

    h = np.asarray(height_map, dtype=np.float64)
    h_min = float(h.min())
    h_max = float(h.max())

    if h_max <= h_min:
        # Degenerate flat height map: still need >= 2 layers to bracket it.
        return np.array([h_min, h_min + delta_h], dtype=np.float64)

    # Small negative epsilon guards against floating-point noise pushing an
    # exact multiple of delta_h up into one extra unnecessary layer.
    n_intervals = int(np.ceil((h_max - h_min) / delta_h - 1e-9))
    n_intervals = max(n_intervals, 1)
    layers = h_min + delta_h * np.arange(n_intervals + 1, dtype=np.float64)

    # Guard against any residual float rounding: top layer must cover h_max.
    if layers[-1] < h_max:
        layers = np.append(layers, layers[-1] + delta_h)

    return layers


# ---------------------------------------------------------------------------
# B. Linear-hat axial weights (sparse two-layer representation)
# ---------------------------------------------------------------------------


def compute_hat_weights(height_map: np.ndarray, layers: np.ndarray):
    """
    Compute linear-hat axial interpolation weights for every pixel, against
    the two neighboring depth layers only.

    This deliberately avoids materializing a dense (H, W, n_layers) tensor:
    each pixel is represented by (j_lo, frac) where the two active layers
    are `j_lo` (weight `1 - frac`) and `j_lo + 1` (weight `frac`).

    Parameters
    ----------
    height_map : array_like, shape (H, W)
    layers : array_like, shape (n_layers,), ascending, uniform spacing.

    Returns
    -------
    j_lo : np.ndarray[int64], shape (H, W)
        Index of the lower neighboring layer for each pixel, in
        [0, n_layers - 2].
    frac : np.ndarray[float64], shape (H, W)
        Weight assigned to the UPPER layer (j_lo + 1), in [0, 1].
        Weight of the lower layer j_lo is (1 - frac).
    """
    h = np.asarray(height_map, dtype=np.float64)
    layers = np.asarray(layers, dtype=np.float64)
    n_layers = layers.shape[0]
    if n_layers < 2:
        raise ValueError("Need at least 2 layers to interpolate between.")

    delta_h = layers[1] - layers[0]

    # Exact boundary treatment: clip height to the covered range first so
    # that h == layers[0] or h == layers[-1] resolve exactly (frac 0 or 1),
    # rather than falling outside due to float noise.
    h_clipped = np.clip(h, layers[0], layers[-1])

    idx = (h_clipped - layers[0]) / delta_h
    j_lo = np.floor(idx).astype(np.int64)
    j_lo = np.clip(j_lo, 0, n_layers - 2)

    frac = idx - j_lo
    # Clip strictly to [0, 1] to absorb floating-point residue at bin edges
    # (e.g. h exactly on a layer boundary, or top-of-range clipping giving
    # idx == n_layers - 1 while j_lo is clipped to n_layers - 2 -> frac ~ 1).
    frac = np.clip(frac, 0.0, 1.0)

    return j_lo, frac


# ---------------------------------------------------------------------------
# C. Gaussian blur with reflective boundary, exact sigma=0 identity
# ---------------------------------------------------------------------------


def gaussian_blur_reflect(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian blur with reflective boundary condition. sigma == 0 is handled
    as an explicit exact identity (never routed through a filter routine
    that could behave ambiguously at sigma=0).
    """
    sigma = float(sigma)
    if sigma <= 0.0:
        return np.array(image, dtype=np.float64, copy=True)
    return ndimage.gaussian_filter(
        np.asarray(image, dtype=np.float64), sigma=sigma, mode="reflect"
    )


# ---------------------------------------------------------------------------
# D. Normalized masked convolution -- single-plane render
# ---------------------------------------------------------------------------


def render_depth_layer(
    oracle: np.ndarray,
    height_map: np.ndarray,
    z: float,
    layers: np.ndarray,
    eps: float = 1e-8,
    sigma_fn=sigma_from_defocus,
):
    """
    Render the through-focus image at focal plane z via depth-layer
    rendering + linear-hat weights + normalized masked convolution:

        I_k = [ sum_j G_{sigma_j} * (I* w_j) ] / [ sum_j G_{sigma_j} * w_j ]

    Implementation processes one depth layer at a time (a single (H, W)
    weight field per iteration), never a dense (H, W, n_layers) tensor.
    Only layers with at least one nonzero-weight pixel are processed.

    Parameters
    ----------
    oracle : array_like, shape (H, W)
        All-in-focus oracle image I* (fixed texture).
    height_map : array_like, shape (H, W)
        height_map -- the sole source of axial depth.
    z : float
        Focal plane position, DOF units.
    layers : array_like, shape (n_layers,)
        Depth layer heights, from `build_depth_layers`.
    eps : float
        Numerical floor applied ONLY to protect against division by values
        near zero; never alters valid (non-near-zero) denominator regions.
    sigma_fn : callable
        sigma(d) mapping, defaults to `sigma_from_defocus`
        (exclusively; not re-derived or modified here).

    Returns
    -------
    I_k : np.ndarray, shape (H, W), float64
    denominator : np.ndarray, shape (H, W), float64
        Returned for stability reporting (min/max should be checked > 0
        over the useful image region).
    """
    oracle = np.asarray(oracle, dtype=np.float64)
    layers = np.asarray(layers, dtype=np.float64)
    n_layers = layers.shape[0]

    j_lo, frac = compute_hat_weights(height_map, layers)

    numerator = np.zeros_like(oracle, dtype=np.float64)
    denominator = np.zeros_like(oracle, dtype=np.float64)

    present_lo = np.unique(j_lo)
    active_layers = sorted(set(present_lo.tolist()) | set((present_lo + 1).tolist()))
    active_layers = [j for j in active_layers if 0 <= j < n_layers]

    for j in active_layers:
        w_j = np.zeros_like(oracle, dtype=np.float64)

        lower_mask = j_lo == j        # pixels where j is the LOWER neighbor
        upper_mask = j_lo == (j - 1)  # pixels where j is the UPPER neighbor

        if np.any(lower_mask):
            w_j[lower_mask] = 1.0 - frac[lower_mask]
        if np.any(upper_mask):
            w_j[upper_mask] = frac[upper_mask]

        if not np.any(w_j):
            continue

        sigma_j = float(sigma_fn(layers[j] - z))

        numerator += gaussian_blur_reflect(oracle * w_j, sigma_j)
        denominator += gaussian_blur_reflect(w_j, sigma_j)

    safe_denominator = np.where(denominator > eps, denominator, eps)
    I_k = numerator / safe_denominator

    return I_k, denominator
