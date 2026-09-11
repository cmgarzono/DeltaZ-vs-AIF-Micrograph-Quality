r"""
Local shape-shading (epi-illumination response).

Invariant: macro height discontinuities
(rectangle steps, ramps) must NOT produce artificial intensity edges.
Only local protrusions (pyramids) receive genuine Lambertian-like shading
from their real surface-orientation change. This is enforced structurally:
the surface-normal gradient is computed everywhere but the resulting shading
modulation is masked to zero outside a small dilation of the protrusion mask.
"""

from typing import Tuple

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

# Fixed epi-illumination light direction (oblique, from upper-left),
# expressed as a unit vector (Lx, Ly, Lz) in a right-handed x-right, y-down,
# z-up convention consistent with the height map's DOF axis.
LIGHT_DIRECTION = np.array([-0.45, -0.45, 0.77])
LIGHT_DIRECTION = LIGHT_DIRECTION / np.linalg.norm(LIGHT_DIRECTION)

# Dilation radius (px) applied to the protrusion mask before shading is
# permitted, giving each pyramid's facets a few extra pixels of halo while
# staying far more localized than any macro-step footprint.
PROTRUSION_SHADING_DILATION_PX = 4

# Shading strength (multiplicative modulation amplitude around 1.0).
SHADING_STRENGTH = 0.35


def protrusion_shading_mask(protrusion_mask: np.ndarray, dilation_px: int = PROTRUSION_SHADING_DILATION_PX) -> np.ndarray:
    """Dilate the fixed protrusion mask slightly to capture full pyramid facets."""
    structure = np.ones((3, 3), dtype=bool)
    dilated = protrusion_mask.copy()
    for _ in range(dilation_px):
        dilated = binary_dilation(dilated, structure=structure)
    return dilated


def compute_shape_shading(
    height_map: np.ndarray,
    protrusion_mask: np.ndarray,
    r_pixels: float,
    light_direction: np.ndarray = LIGHT_DIRECTION,
    shading_strength: float = SHADING_STRENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a multiplicative shading field confined to protrusions.

    Args:
        height_map: frozen M1 height map (DOF units).
        protrusion_mask: fixed specimen protrusion mask (boolean).
        r_pixels: lateral scale (px per R unit), used to convert the DOF
            gradient into a comparable pixel-scale surface slope.
        light_direction: unit vector (Lx, Ly, Lz).
        shading_strength: modulation amplitude.

    Returns:
        (shading_field, dilated_protrusion_mask):
            shading_field: float64 array, multiplicative factor centered at 1.0,
                equal to exactly 1.0 (no modulation) outside the dilated
                protrusion mask.
            dilated_protrusion_mask: the mask actually used to gate shading.
    """
    dilated_mask = protrusion_shading_mask(protrusion_mask)

    # Mild smoothing so the normal estimate reflects facet orientation rather
    # than single-pixel noise, without leaking into neighboring macro regions.
    h_smooth = gaussian_filter(height_map, sigma=0.6)
    gy, gx = np.gradient(h_smooth)

    # Surface normal from height gradient: n = normalize(-dz/dx, -dz/dy, 1)
    nz = np.ones_like(h_smooth)
    norm = np.sqrt(gx**2 + gy**2 + nz**2)
    nx, ny, nz = -gx / norm, -gy / norm, nz / norm

    dot = nx * light_direction[0] + ny * light_direction[1] + nz * light_direction[2]
    dot = np.clip(dot, -1.0, 1.0)

    # Modulation only where a real local surface-orientation change exists
    # (i.e., inside the dilated protrusion mask). Elsewhere: exactly 1.0.
    shading = np.ones_like(h_smooth)
    modulation = 1.0 + shading_strength * (dot - dot[dilated_mask].mean() if dilated_mask.any() else 0.0)
    shading[dilated_mask] = np.clip(modulation[dilated_mask], 0.4, 1.8)

    return shading, dilated_mask
