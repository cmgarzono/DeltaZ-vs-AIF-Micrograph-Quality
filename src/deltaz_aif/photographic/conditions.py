"""
Procedural generators for the 8 photographic conditions.

The H and P texture generators use deterministic frequency scaling to
produce four spectral regimes (LF, COARSE, MIXED, FINE). The generated
oracles preserve the specimen geometry, baseline photographic rendering,
texture amplitudes, region masks, phase choices, gamma, clipping, and
fixed intensity range used by the manuscript simulations.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from deltaz_aif.photographic.baseline_renderer import WIDTH, HEIGHT
pi = np.pi

# ─── Regime targets (design targets, not exact constants) ──────────────
TARGET_CENTROIDS = {
    'LF':     0.014,
    'COARSE': 0.022,
    'MIXED':  0.032,
    'FINE':   0.050,
}
REGIME_ORDER = ['LF', 'COARSE', 'MIXED', 'FINE']


# ─── H texture: direct frequency scaling ────────────────────────────────

def make_h_texture_scaled(s):
    """
    H texture (5 deterministic sine components), spatial frequency of
    every component multiplied by s. s=1.0 reproduces the exact fixed
    _make_h_texture() baseline. Phases, relative weights, and the fixed
    smoothing (sigma=0.55) are all preserved unchanged -- only frequency
    (period -> period/s) is scaled.
    """
    y = np.linspace(-1.0, 1.0, HEIGHT, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, WIDTH, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    xpix = (xx + 1.0) * 0.5 * WIDTH
    ypix = (yy + 1.0) * 0.5 * HEIGHT

    texture = (
        0.58 * np.sin(2 * pi * (s * xpix / 38.0 + s * ypix / 143.0 + 0.07))
        + 0.42 * np.sin(2 * pi * (-s * xpix / 91.0 + s * ypix / 46.0 + 0.37 + 0.07))
        + 0.28 * np.sin(2 * pi * (s * xpix / 17.0 - s * ypix / 29.0 + 0.11 + 0.07))
        + 0.18 * np.sin(2 * pi * (s * xpix / 71.0 + 0.23 + 0.07))
        + 0.16 * np.sin(2 * pi * (s * ypix / 63.0 + 0.61 + 0.07))
    )
    texture = gaussian_filter(texture, sigma=0.55, mode="reflect")
    texture = (texture - texture.mean()) / texture.std()
    return texture


def h_component_frequencies(s):
    """Analytic normalized frequency (Nyquist=1.0) of H's 5 deterministic
    components at scale s, paired with their (pre-normalization) weights."""
    terms = [
        (38.0, 143.0, 0.58),
        (91.0, 46.0, 0.42),
        (17.0, 29.0, 0.28),
        (71.0, None, 0.18),
        (None, 63.0, 0.16),
    ]
    freqs = []
    for lx, ly, w in terms:
        fx = (s / lx) if lx else 0.0
        fy = (s / ly) if ly else 0.0
        mag = np.sqrt(fx ** 2 + fy ** 2)
        freqs.append((mag / 0.5, w))
    return freqs


# ─── P texture: direct frequency scaling ────────────────────────────────

def _rect_hard(xpix, ypix, x0, y0, x1, y1):
    return ((xpix >= x0) & (xpix < x1) & (ypix >= y0) & (ypix < y1)).astype(np.float32)


def _oriented_texture_scaled(xx, yy, angle, region_scale, phase, s):
    """
    Single region's oriented texture (4 sine components), with an
    ADDITIONAL uniform frequency scale s layered on top of the region's
    own existing relative scale factor (region_scale). Orientation,
    phase, and per-region relative period ratios are all preserved
    unchanged -- s only rescales the overall spatial frequency.
    """
    xpix = (xx + 1.0) * 0.5 * WIDTH
    ypix = (yy + 1.0) * 0.5 * HEIGHT
    ca = np.cos(angle)
    sa = np.sin(angle)
    u = ca * xpix + sa * ypix
    v = -sa * xpix + ca * ypix

    texture = (
        0.70 * np.sin(2 * pi * (s * u / (24.0 * region_scale) + phase))
        + 0.36 * np.sin(2 * pi * (s * v / (58.0 * region_scale) + 0.27 + phase))
        + 0.24 * np.sin(2 * pi * (s * (u + 0.55 * v) / (39.0 * region_scale) + 0.63 + phase))
        + 0.13 * np.sin(2 * pi * (s * (u - 0.85 * v) / (16.0 * region_scale) + 0.41 + phase))
    )
    texture = gaussian_filter(texture, sigma=0.45, mode="reflect")
    return (texture - texture.mean()) / texture.std()


def make_p_texture_scaled(s):
    """
    P texture (5 regions, each an oriented 4-sine field), with spatial
    frequency of every regional component multiplied by the SAME s. All
    region masks, blending (sigma=18px), orientations, phases, and
    per-region RELATIVE scale differences are preserved unchanged.
    """
    y = np.linspace(-1.0, 1.0, HEIGHT, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, WIDTH, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    xpix = (xx + 1.0) * 0.5 * WIDTH
    ypix = (yy + 1.0) * 0.5 * HEIGHT

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

    raw_weights = [outside, main_outer, inner, left_top, right_bottom]
    weights = [gaussian_filter(w, sigma=18.0, mode="reflect") for w in raw_weights]
    total = sum(weights)
    total = np.maximum(total, 1e-8)
    weights = [w / total for w in weights]

    textures = [
        _oriented_texture_scaled(xx, yy, angle=0.02, region_scale=1.25, phase=0.13, s=s),
        _oriented_texture_scaled(xx, yy, angle=0.72, region_scale=0.86, phase=0.31, s=s),
        _oriented_texture_scaled(xx, yy, angle=-0.48, region_scale=1.08, phase=0.53, s=s),
        _oriented_texture_scaled(xx, yy, angle=1.22, region_scale=0.72, phase=0.79, s=s),
        _oriented_texture_scaled(xx, yy, angle=-0.95, region_scale=1.48, phase=1.03, s=s),
    ]

    blended = sum(w * t for w, t in zip(weights, textures))
    blended = (blended - blended.mean()) / blended.std()
    return blended


def p_component_frequencies(s):
    """Per-region analytic normalized frequency + weight list at scale s."""
    region_scales = {
        'outside': 1.25, 'main_outer': 0.86, 'inner': 1.08,
        'left_top': 0.72, 'right_bottom': 1.48,
    }
    base_periods_weights = [(24.0, 0.70), (58.0, 0.36), (39.0, 0.24), (16.0, 0.13)]
    out = {}
    for region, rs in region_scales.items():
        freqs = []
        for base_period, w in base_periods_weights:
            period = base_period * rs / s
            freq_norm = (1.0 / period) / 0.5
            freqs.append((freq_norm, w))
        out[region] = freqs
    return out


