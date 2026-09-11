"""
Photographic appearance model.

Implements the deterministic ideal all-in-focus photographic oracle for the
fixed M1 geometry: texture synthesis (H/P organization x LF/COARSE/MIXED/FINE
spatial content) and local shape-shading (epi-illumination response
confined to protrusions).

This package does not modify M1 geometry, the height map, or the
morphological masks. It reads them as fixed inputs.
"""

from .model import PhotographicModel, GATE2_CONFIGURATIONS

__all__ = ["PhotographicModel", "GATE2_CONFIGURATIONS"]
