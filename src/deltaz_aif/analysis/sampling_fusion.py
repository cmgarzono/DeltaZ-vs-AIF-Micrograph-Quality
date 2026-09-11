r"""
S3: Photographic manifestation of axial sampling.

Reuses, WITHOUT MODIFICATION:
  - M1 geometry (height_map) and its frozen ground-truth masks
    (src/deltaz_aif/specimen/geometry.py)
  - The calibrated deterministic through-focus renderer
    (src/deltaz_aif/rendering/depth_layer_renderer.py,
     src/deltaz_aif/focus/psf_core.py)
  - The 8 frozen photographic-condition oracles
    (data/specimen/conditions/oracle_*.npy)

This module implements ONLY the S3-specific machinery that did not exist
before: the candidate focal-plane grid, the shared-denominator stack
renderer, focus fusion (local Laplacian energy), image-quality / focus-
selection / redundancy metrics, and regional aggregation.

IMPORTANT — two different "delta" quantities exist and must not be
confused:
  * TF depth-layer spacing (delta_h = 0.0625 DOF): the FROZEN internal
    resolution of the through-focus renderer's depth-layer grid. Constant
    for the whole S3 run, independent of the axial sampling variable.
  * S3 axial sampling spacing (delta = Delta_z / DOF, 0.01..3.20): the
    variable under study -- the spacing between CANDIDATE FOCAL PLANES
    (camera focus positions) at which frames are captured/rendered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity as _ssim

from deltaz_aif.rendering.depth_layer_renderer import (
    build_depth_layers,
    compute_hat_weights,
    gaussian_blur_reflect,
)
from deltaz_aif.focus.psf_core import sigma_from_defocus

# ---------------------------------------------------------------------------
# Frozen renderer configuration (must match the calibrated renderer exactly)
# ---------------------------------------------------------------------------
TF_DEPTH_LAYER_SPACING = 0.0625  # DOF
DENOM_EPS = 1e-8

# ---------------------------------------------------------------------------
# 1. Candidate focal-plane grid (S3-specific; did not exist before)
# ---------------------------------------------------------------------------


def build_candidate_grid(h_min: float, h_max: float, delta: float) -> np.ndarray:
    """
    Candidate focal-plane grid policy (documented, unambiguous, applied
    identically to all 8 photographic conditions):

      * first candidate z = h_min (domain minimum, exactly)
      * candidate spacing = delta, uniform, EXCEPT the final interval
      * last candidate z = h_max (domain maximum, exactly) -- always
        included even if (h_max - h_min) is not an integer multiple of
        delta. The residual interval (< delta) is absorbed entirely into
        the LAST step (no grid offset, no centering).
      * K = number of candidates = ceil((h_max-h_min)/delta - eps) + 1,
        i.e. K-1 uniform steps of size `delta` plus one shorter closing
        step when the domain width is not an exact multiple of delta.
      * endpoints are always included.

    This mirrors the clamping convention already used by the frozen
    `build_depth_layers` for internal consistency, applied here to
    the outer (S3) sampling grid instead of the inner depth-layer grid.
    """
    if delta <= 0.0:
        raise ValueError(f"delta must be > 0, got {delta}")
    width = h_max - h_min
    if width <= 0.0:
        return np.array([h_min, h_max], dtype=np.float64)

    n_steps = int(np.ceil(width / delta - 1e-9))
    n_steps = max(n_steps, 1)
    z = h_min + delta * np.arange(n_steps, dtype=np.float64)
    z = np.append(z, h_max)
    # Guard: if the last uniform step already landed essentially on h_max,
    # drop the duplicate rather than emitting a near-zero final interval.
    if z[-1] - z[-2] < 1e-9:
        z = z[:-1]
    return z


# ---------------------------------------------------------------------------
# 1b. Master axial lattice (S3-ACCELERATE): all 320 candidate grids are
# subsets of a single 801-point lattice z_j = -4.0 + 0.01*j, j=0..800.
# This section does NOT change any science -- it only expresses the
# existing `build_candidate_grid` policy as index selection into a shared
# lattice, so that rendering can be de-duplicated across delta values.
# ---------------------------------------------------------------------------

MASTER_STEP = 0.01  # DOF, matches the domain width (8.0) / this step -> 800 intervals
N_MASTER = 801


def build_master_lattice(h_min: float = -4.0, h_max: float = 4.0) -> np.ndarray:
    """z_j = h_min + MASTER_STEP * j, j = 0..800 (801 points, exact)."""
    width = h_max - h_min
    n_intervals = int(round(width / MASTER_STEP))
    z = h_min + MASTER_STEP * np.arange(n_intervals + 1, dtype=np.float64)
    return z


def master_lattice_indices(delta: float, n_master: int = N_MASTER) -> np.ndarray:
    """
    Index-space equivalent of `build_candidate_grid(h_min, h_max, delta)`,
    for delta values that are exact multiples of MASTER_STEP (i.e. all 320
    S3 delta values 0.01..3.20). Returns indices into `build_master_lattice()`
    such that z_master[master_lattice_indices(delta)] == build_candidate_grid(
    h_min, h_max, delta) exactly (see test_master_lattice_matches_candidate_grid).
    """
    m = int(round(delta / MASTER_STEP))
    if m <= 0:
        raise ValueError(f"delta must be a positive multiple of {MASTER_STEP}, got {delta}")
    n_top = n_master - 1  # index of h_max, i.e. 800
    n_steps = int(np.ceil(n_top / m - 1e-9))
    n_steps = max(n_steps, 1)
    idx = m * np.arange(n_steps, dtype=np.int64)
    idx = np.append(idx, n_top)
    if idx[-1] == idx[-2]:
        idx = idx[:-1]
    return idx


# ---------------------------------------------------------------------------
# 2. Shared-denominator through-focus stack renderer
# ---------------------------------------------------------------------------


@dataclass
class DepthLayerCache:
    """Height-map/layer-derived quantities that are independent of z, of
    delta, and of photographic condition -- computed exactly once per S3
    run and reused for every candidate plane of every condition."""

    layers: np.ndarray
    j_lo: np.ndarray
    frac: np.ndarray

    @classmethod
    def build(cls, height_map: np.ndarray) -> "DepthLayerCache":
        layers = build_depth_layers(height_map, TF_DEPTH_LAYER_SPACING)
        j_lo, frac = compute_hat_weights(height_map, layers)
        return cls(layers=layers, j_lo=j_lo, frac=frac)

    def active_layers(self) -> List[int]:
        present_lo = np.unique(self.j_lo)
        n_layers = self.layers.shape[0]
        active = sorted(set(present_lo.tolist()) | set((present_lo + 1).tolist()))
        return [j for j in active if 0 <= j < n_layers]

    def hat_weight_field(self, j: int) -> np.ndarray:
        w_j = np.zeros_like(self.frac, dtype=np.float64)
        lower_mask = self.j_lo == j
        upper_mask = self.j_lo == (j - 1)
        if np.any(lower_mask):
            w_j[lower_mask] = 1.0 - self.frac[lower_mask]
        if np.any(upper_mask):
            w_j[upper_mask] = self.frac[upper_mask]
        return w_j


def render_frame_shared(
    cache: DepthLayerCache,
    oracles: Dict[str, np.ndarray],
    z: float,
) -> Dict[str, np.ndarray]:
    """
    Render the through-focus frame at focal plane z for every photographic
    condition in `oracles`, sharing the (condition-independent) blurred
    denominator across all conditions. Mathematically IDENTICAL to calling
    the frozen `render_depth_layer` once per condition -- only the
    (H,W)-array-level Gaussian blur of the hat-weight denominator field is
    de-duplicated across conditions; the renderer's math is unchanged.

    Returns {condition_name: I_k array (H, W) float64}.
    """
    active = cache.active_layers()
    shape = cache.frac.shape

    numerators = {name: np.zeros(shape, dtype=np.float64) for name in oracles}
    denominator = np.zeros(shape, dtype=np.float64)

    for j in active:
        w_j = cache.hat_weight_field(j)
        if not np.any(w_j):
            continue
        sigma_j = float(sigma_from_defocus(cache.layers[j] - z))
        denominator += gaussian_blur_reflect(w_j, sigma_j)
        for name, oracle in oracles.items():
            numerators[name] += gaussian_blur_reflect(oracle * w_j, sigma_j)

    safe_denom = np.where(denominator > DENOM_EPS, denominator, DENOM_EPS)
    return {name: numerators[name] / safe_denom for name in oracles}


# ---------------------------------------------------------------------------
# 3. Stack oracle (sampling-only reference)
# ---------------------------------------------------------------------------


def nearest_plane_index(height_map: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    """k_or(x,y) = argmin_k |height_map(x,y) - z_grid[k]|. Uses TRUE height
    only -- this is the oracle, not the fusion algorithm."""
    h = height_map[..., None]
    diffs = np.abs(h - z_grid[None, None, :])
    return np.argmin(diffs, axis=-1)


# ---------------------------------------------------------------------------
# 4. Focus fusion -- local Laplacian energy
# ---------------------------------------------------------------------------

LAPLACIAN_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


def laplacian_energy(frame: np.ndarray, window: int) -> np.ndarray:
    """
    E_w(x,y) = local MEAN of Laplacian(I)^2 over a `window`x`window` box.
    The local mean keeps the response comparable across window sizes.
    """
    L = ndimage.correlate(frame, LAPLACIAN_KERNEL, mode="reflect")
    L2 = L * L
    return ndimage.uniform_filter(L2, size=window, mode="reflect")


def fuse_stack(stack: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    stack: (K, H, W) frames for one (condition, delta).
    Returns (k_F map (H,W) int, I_F fused image (H,W)).
    Ties: deterministic -- argmax keeps the FIRST (lowest-index / lowest-z)
    maximizer, matching numpy's argmax convention.
    Does NOT access true height at any point.
    """
    K = stack.shape[0]
    E = np.stack([laplacian_energy(stack[k], window) for k in range(K)], axis=0)
    k_F = np.argmax(E, axis=0)
    I_F = np.take_along_axis(stack, k_F[None, :, :], axis=0)[0]
    return k_F, I_F


# ---------------------------------------------------------------------------
# 5. Image-quality metrics
# ---------------------------------------------------------------------------

DATA_RANGE = 255.0  # oracles are stored as 0..255 float (uint8-valued)


def image_metrics(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> Dict[str, float]:
    if mask is not None:
        av = a[mask]
        bv = b[mask]
    else:
        av = a.ravel()
        bv = b.ravel()
    mse = float(np.mean((av - bv) ** 2))
    psnr = float("inf") if mse == 0 else 10.0 * np.log10((DATA_RANGE ** 2) / mse)
    if mask is not None:
        # SSIM needs a 2D field; compute globally then restrict to mask.
        ssim_full = _ssim(a, b, data_range=DATA_RANGE)
        ssim_val = float(ssim_full)  # documented limitation: SSIM is global (see report)
    else:
        ssim_val = float(_ssim(a, b, data_range=DATA_RANGE))
    return {"mse": mse, "psnr": psnr, "ssim": ssim_val}


# ---------------------------------------------------------------------------
# 6. Focus-selection metrics
# ---------------------------------------------------------------------------


def focus_selection_metrics(
    k_sel: np.ndarray, height_map: np.ndarray, z_grid: np.ndarray, mask: np.ndarray
) -> Dict[str, float]:
    z_sel = z_grid[k_sel]
    eps_z = np.abs(z_sel - height_map)
    vals = eps_z[mask]
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95)),
    }


def focus_index_agreement(k_F: np.ndarray, k_or: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(k_F[mask] == k_or[mask]))


# ---------------------------------------------------------------------------
# 7. Axial redundancy (pre-fusion evidence fields)
# ---------------------------------------------------------------------------


def cosine_similarity_fields(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    av = a[mask].ravel()
    bv = b[mask].ravel()
    na = np.linalg.norm(av)
    nb = np.linalg.norm(bv)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def axial_redundancy(E_stack: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """E_stack: (K,H,W) pre-fusion evidence fields for ONE window size."""
    K = E_stack.shape[0]
    overlaps = [cosine_similarity_fields(E_stack[k], E_stack[k + 1], mask) for k in range(K - 1)]
    overlaps = np.array(overlaps, dtype=np.float64)
    n_eff = 1.0 + float(np.sum(1.0 - overlaps))
    r_z = 1.0 - n_eff / K
    return {
        "K": K,
        "N_eff": n_eff,
        "R_Z": r_z,
        "adjacent_overlap_mean": float(np.mean(overlaps)) if K > 1 else 1.0,
    }
