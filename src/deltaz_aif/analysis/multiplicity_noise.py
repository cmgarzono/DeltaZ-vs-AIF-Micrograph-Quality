r"""
Multiplicity + additive-noise experiment.

Studies how the number of available noisy observations (N) and the noise
level (sigma) affect the quality and stability of three simple combination
rules (single-frame, mean, max-focus), on ONE frozen representative
photographic condition/crop taken from the H_MIXED all-in-focus oracle.
This module does not touch the axial-coverage, spectral-evidence, or
sampling/fusion experiments, the through-focus renderer, or the specimen
geometry, and does not re-run the axial-spacing sweep.

Reuses, WITHOUT MODIFICATION:
  - The local-Laplacian-energy focus-evidence definition from
    `deltaz_aif.analysis.sampling_fusion.laplacian_energy` / `LAPLACIAN_KERNEL`.

Convention: all images here are float64 in normalized [0, 1] intensity
(NOT 0..255, unlike `sampling_fusion`'s `image_metrics`/`DATA_RANGE`). This
is deliberate and is enforced at the I_clean loading step.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.fft import dctn

from deltaz_aif.analysis.sampling_fusion import LAPLACIAN_KERNEL

# ---------------------------------------------------------------------------
# 0. Experimental grid
# ---------------------------------------------------------------------------

SIGMA_LEVELS = (0.00, 0.01, 0.02, 0.03, 0.04)
N_LEVELS = (1, 2, 4, 8, 16, 32, 64, 128)
WINDOW_LEVELS = (3, 7)
RULES = ("single", "mean", "maxfocus")
N_REPLICATES = 100

# Representative condition + crop: frozen H_MIXED oracle, top-right
# 400x400 corner of the outer rectangle -- the only one of the four
# outer-rectangle-corner candidate windows that simultaneously contains
# all four specimen regional mask types (plateau, ramp, hard_step,
# protrusion). Chosen from geometry alone, before any result from this
# experiment was computed.
ORACLE_CONDITION = "H_MIXED"
CROP_Y0, CROP_Y1 = 170, 570
CROP_X0, CROP_X1 = 1190, 1590

DATA_RANGE = 1.0  # normalized intensity

# ---------------------------------------------------------------------------
# 1. I_clean
# ---------------------------------------------------------------------------


def load_i_clean(oracle_path: str) -> np.ndarray:
    """
    Load the fixed H_MIXED all-in-focus photographic oracle and extract the
    fixed representative crop. No blur/defocus is added here. The oracle is
    already stored as float64 in [0, 1].
    """
    full = np.load(oracle_path)
    crop = full[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1].astype(np.float64)
    if crop.min() < -1e-9 or crop.max() > 1.0 + 1e-9:
        raise ValueError(
            f"I_clean out of expected [0,1] normalized range: "
            f"[{crop.min()}, {crop.max()}]"
        )
    return np.clip(crop, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 2. Deterministic seed policy
# ---------------------------------------------------------------------------


def condition_seed(sigma: float, N: int, replicate: int) -> int:
    """
    Deterministic seed for the (sigma, N, replicate) noisy-stack generation.
    Independent of `rule` and `window` BY DESIGN -- the same underlying
    noisy stack is reused across single/mean/max-focus(w=3)/max-focus(w=7)
    for a given (sigma, N, replicate), giving a paired comparison.
    """
    key = f"s4|sigma={sigma:.6f}|N={N:d}|rep={replicate:d}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def generate_noisy_stack(i_clean: np.ndarray, sigma: float, N: int, seed: int) -> np.ndarray:
    """
    Generate N independent realizations of additive Gaussian noise on
    i_clean, clipped to [0,1]. Each of the N frames uses
    an independently spawned child stream from a single SeedSequence, so
    reruns from the same seed are bit-reproducible and different frames
    within the stack are statistically independent.
    """
    H, W = i_clean.shape
    stack = np.empty((N, H, W), dtype=np.float64)
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(N)
    for k in range(N):
        rng = np.random.default_rng(child_seeds[k])
        if sigma > 0.0:
            eps = rng.normal(loc=0.0, scale=sigma, size=(H, W))
            stack[k] = np.clip(i_clean + eps, 0.0, 1.0)
        else:
            stack[k] = i_clean
    return stack


def clipping_fraction(stack: np.ndarray, i_clean: np.ndarray, sigma: float) -> float:
    """Fraction of pixel-observations that hit the [0,1] clip boundary."""
    if sigma <= 0.0:
        return 0.0
    # Reconstruct raw (unclipped) values is not possible post hoc without
    # storing epsilon; instead measure fraction of output pixels sitting
    # exactly at 0.0 or 1.0, which is the operationally relevant quantity.
    at_bound = (stack <= 0.0) | (stack >= 1.0)
    return float(np.mean(at_bound))


# ---------------------------------------------------------------------------
# 3. Combination rules
# ---------------------------------------------------------------------------


def rule_single(stack: np.ndarray) -> np.ndarray:
    """SINGLE-FRAME: deterministically the first realization."""
    return stack[0]


def rule_mean(stack: np.ndarray) -> np.ndarray:
    """MEAN: arithmetic pixel-wise mean of the N observations."""
    return stack.mean(axis=0)


def laplacian_energy(frame: np.ndarray, window: int) -> np.ndarray:
    """Local mean of Laplacian(I)^2 over a window x window box. Identical
    definition to S3's `deltaz_aif.analysis.sampling_fusion.laplacian_energy`
    (re-implemented here, not imported by call, only to keep S4 fully
    self-contained at the array-op level; the kernel is imported, not
    redefined)."""
    L = ndimage.correlate(frame, LAPLACIAN_KERNEL, mode="reflect")
    L2 = L * L
    return ndimage.uniform_filter(L2, size=window, mode="reflect")


def rule_maxfocus(stack: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    MAX-FOCUS: per-pixel selection of the observation with highest local
    Laplacian-energy evidence. Returns (I_out, k_map).
    Ties: numpy argmax convention (first/lowest-index maximizer), matching
    the sampling-fusion rule.
    """
    K = stack.shape[0]
    E = np.stack([laplacian_energy(stack[k], window) for k in range(K)], axis=0)
    k_map = np.argmax(E, axis=0)
    I_out = np.take_along_axis(stack, k_map[None, :, :], axis=0)[0]
    return I_out, k_map


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------


def mse_psnr(i_out: np.ndarray, i_clean: np.ndarray) -> Tuple[float, float]:
    mse = float(np.mean((i_out - i_clean) ** 2))
    if mse <= 0.0:
        psnr = float("inf")
    else:
        psnr = 10.0 * np.log10((DATA_RANGE ** 2) / mse)
    return mse, psnr


def laplacian_variance(i_out: np.ndarray) -> float:
    """Var[Laplacian(I_out)] -- global variance of the raw Laplacian field
    (distinct from the local-mean-squared focus-evidence field used by
    max-focus selection)."""
    L = ndimage.correlate(i_out, LAPLACIAN_KERNEL, mode="reflect")
    return float(np.var(L))


# DCT-HF definition: uses a 2D type-II orthonormal DCT. "High frequency"
# is defined as the region where
# the normalized radial DCT frequency r = sqrt((u/H)^2 + (v/W)^2) / sqrt(2)
# exceeds a fixed threshold of 0.5 (the upper half of the normalized radial
# range, index-count-independent so it is stable across image sizes).
# The metric reported is the FRACTION of total DCT spectral energy in that
# region (dimensionless, reproducible, comparable across images/sizes).
_DCT_HF_THRESHOLD = 0.5
_dct_hf_mask_cache: Dict[Tuple[int, int], np.ndarray] = {}


def _dct_hf_mask(shape: Tuple[int, int]) -> np.ndarray:
    if shape not in _dct_hf_mask_cache:
        H, W = shape
        u = np.arange(H)[:, None] / H
        v = np.arange(W)[None, :] / W
        r = np.sqrt(u ** 2 + v ** 2) / np.sqrt(2.0)
        _dct_hf_mask_cache[shape] = r > _DCT_HF_THRESHOLD
    return _dct_hf_mask_cache[shape]


def dct_hf_energy(i_out: np.ndarray) -> float:
    C = dctn(i_out, type=2, norm="ortho")
    total = float(np.sum(C ** 2))
    if total <= 0.0:
        return 0.0
    mask = _dct_hf_mask(i_out.shape)
    hf = float(np.sum(C[mask] ** 2))
    return hf / total


def selection_entropy(k_map: np.ndarray, N: int) -> Tuple[float, float]:
    """
    H_sel = -sum p_j log(p_j) over observed indices j in [0, N).
    H_sel_norm = H_sel / log(N) for N > 1, else 0.0.
    """
    counts = np.bincount(k_map.ravel(), minlength=N).astype(np.float64)
    total = counts.sum()
    p = counts[counts > 0] / total
    H_sel = float(-np.sum(p * np.log(p)))
    if N > 1:
        H_sel_norm = H_sel / np.log(N)
    else:
        H_sel_norm = 0.0
    return H_sel, H_sel_norm


# ---------------------------------------------------------------------------
# 5. Row assembly
# ---------------------------------------------------------------------------


@dataclass
class S4Row:
    sigma: float
    N: int
    rule: str
    window: Optional[int]
    replicate: int
    seed: int
    mse: float
    psnr: float
    laplacian_variance: float
    dct_hf_energy: float
    selection_entropy: Optional[float]
    selection_entropy_normalized: Optional[float]
    clipping_fraction: float

    def as_dict(self) -> Dict:
        return {
            "sigma": self.sigma,
            "N": self.N,
            "rule": self.rule,
            "window": self.window if self.window is not None else "NA",
            "replicate": self.replicate,
            "seed": self.seed,
            "MSE": self.mse,
            "PSNR": self.psnr,
            "laplacian_variance": self.laplacian_variance,
            "dct_hf_energy": self.dct_hf_energy,
            "selection_entropy": self.selection_entropy if self.selection_entropy is not None else "NA",
            "selection_entropy_normalized": (
                self.selection_entropy_normalized if self.selection_entropy_normalized is not None else "NA"
            ),
            "clipping_fraction": self.clipping_fraction,
        }


def run_condition(i_clean: np.ndarray, sigma: float, N: int, replicate: int) -> list:
    """
    Generate ONE noisy stack for (sigma, N, replicate) and apply ALL rules
    to it (paired design). Returns a list of S4Row.
    """
    seed = condition_seed(sigma, N, replicate)
    stack = generate_noisy_stack(i_clean, sigma, N, seed)
    clip_frac = clipping_fraction(stack, i_clean, sigma)
    rows = []

    # SINGLE
    out = rule_single(stack)
    mse, psnr = mse_psnr(out, i_clean)
    rows.append(S4Row(
        sigma=sigma, N=N, rule="single", window=None, replicate=replicate, seed=seed,
        mse=mse, psnr=psnr, laplacian_variance=laplacian_variance(out),
        dct_hf_energy=dct_hf_energy(out),
        selection_entropy=None, selection_entropy_normalized=None,
        clipping_fraction=clip_frac,
    ))

    # MEAN
    out = rule_mean(stack)
    mse, psnr = mse_psnr(out, i_clean)
    rows.append(S4Row(
        sigma=sigma, N=N, rule="mean", window=None, replicate=replicate, seed=seed,
        mse=mse, psnr=psnr, laplacian_variance=laplacian_variance(out),
        dct_hf_energy=dct_hf_energy(out),
        selection_entropy=None, selection_entropy_normalized=None,
        clipping_fraction=clip_frac,
    ))

    # MAX-FOCUS, both windows
    for w in WINDOW_LEVELS:
        out, k_map = rule_maxfocus(stack, w)
        mse, psnr = mse_psnr(out, i_clean)
        H_sel, H_sel_norm = selection_entropy(k_map, N)
        rows.append(S4Row(
            sigma=sigma, N=N, rule="maxfocus", window=w, replicate=replicate, seed=seed,
            mse=mse, psnr=psnr, laplacian_variance=laplacian_variance(out),
            dct_hf_energy=dct_hf_energy(out),
            selection_entropy=H_sel, selection_entropy_normalized=H_sel_norm,
            clipping_fraction=clip_frac,
        ))

    return rows
