r"""
Shared utilities for building outputs/manuscript_assets/.

Inputs are fixed arrays under data/specimen/, results/sampling_fusion/,
and results/multiplicity_noise/. Intermediate assets are written under
outputs/manuscript_assets/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from deltaz_aif.analysis.sampling_fusion import (  # noqa: E402
    build_master_lattice,
    master_lattice_indices,
    nearest_plane_index,
    fuse_stack,
    N_MASTER,
)

SPECIMEN_DIR = ROOT / "data" / "specimen"
CONDITIONS_DIR = ROOT / "data" / "specimen" / "conditions"
S3_CACHE = ROOT / "results" / "sampling_fusion" / "cache"
S4_DIR = ROOT / "results" / "multiplicity_noise"

OUT = ROOT / "outputs/manuscript_assets"
CONDITIONS = ["H_LF", "H_COARSE", "H_MIXED", "H_FINE", "P_LF", "P_COARSE", "P_MIXED", "P_FINE"]


def manifest_row(**kwargs):
    """Compatibility hook for asset scripts; no tracked manifest is written."""
    return None


def load_height_map() -> np.ndarray:
    return np.load(SPECIMEN_DIR / "height_map.npy").astype(np.float64)


def load_regional_masks() -> dict:
    d = np.load(S3_CACHE / "regional_masks.npz")
    return {k: d[k] for k in d.files}


def load_oracle(cond: str) -> np.ndarray:
    return np.load(CONDITIONS_DIR / f"oracle_{cond}.npy").astype(np.float64)


def frames_memmap(cond: str, shape=(1080, 1920)) -> np.memmap:
    p = S3_CACHE / f"frames_{cond}.f64"
    return np.memmap(p, dtype=np.float64, mode="r", shape=(N_MASTER,) + shape)


def z_master() -> np.ndarray:
    return build_master_lattice(-4.0, 4.0)


def save_gray(arr: np.ndarray, path: Path, vmin=0.0, vmax=1.0, cmap="gray"):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(path), arr, vmin=vmin, vmax=vmax, cmap=cmap)


def save_npy(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr)


def bbox_of_mask(mask: np.ndarray, pad: int = 0):
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min() - pad, ys.max() + 1 + pad, xs.min() - pad, xs.max() + 1 + pad
    H, W = mask.shape
    return max(0, y0), min(H, y1), max(0, x0), min(W, x1)


def largest_component_bbox(mask: np.ndarray, pad: int = 20):
    from scipy.ndimage import label, find_objects
    lab, n = label(mask)
    if n == 0:
        return bbox_of_mask(mask, pad)
    sizes = np.bincount(lab.ravel())[1:]
    biggest = 1 + int(np.argmax(sizes))
    sl = find_objects(lab)[biggest - 1]
    H, W = mask.shape
    y0, y1 = max(0, sl[0].start - pad), min(H, sl[0].stop + pad)
    x0, x1 = max(0, sl[1].start - pad), min(W, sl[1].stop + pad)
    return y0, y1, x0, x1


# ---------------------------------------------------------------------------
# Fixed, geometry-defined ROIs (documented pixel coordinates; frozen for the
# whole evidence bank once computed the first time from the frozen masks).
# ---------------------------------------------------------------------------
_ROI_CACHE_PATH = OUT / "10_manifest" / "roi_definitions.json"


def get_rois() -> dict:
    if _ROI_CACHE_PATH.exists():
        return json.loads(_ROI_CACHE_PATH.read_text())
    masks = load_regional_masks()
    rois = {}
    # Protrusion: bbox of one representative single pyramid (largest connected
    # component is one 27th of the protrusion mask -> a single pyramid + pad).
    rois["protrusion"] = list(int(v) for v in largest_component_bbox(masks["protrusion"], pad=25))
    # Hard-step: bbox of the top-face hard-step band, cropped to a 400x400-ish
    # window around its horizontal midpoint (mask itself is a thin band -> use
    # bbox with generous pad so the crop actually shows step context).
    hs_bbox = bbox_of_mask(masks["hard_step"], pad=0)
    y0, y1, x0, x1 = hs_bbox
    cx = (x0 + x1) // 2
    rois["hard_step"] = [int(y0) - 20, int(y0) + 300, int(cx) - 200, int(cx) + 200]
    # Ramp: right-face ramp band, windowed similarly.
    rp_bbox = bbox_of_mask(masks["ramp"], pad=0)
    y0, y1, x0, x1 = rp_bbox
    cy = (y0 + y1) // 2
    rois["ramp"] = [int(cy) - 200, int(cy) + 200, int(x1) - 320, int(x1) + 20]
    # Plateau: a 400x400 interior window centered in the plateau mask's bbox.
    pl_bbox = bbox_of_mask(masks["plateau"], pad=0)
    y0, y1, x0, x1 = pl_bbox
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    rois["plateau"] = [int(cy) - 200, int(cy) + 200, int(cx) - 200, int(cx) + 200]

    H, W = masks["plateau"].shape
    for k, (a, b, c, d) in list(rois.items()):
        rois[k] = [max(0, a), min(H, b), max(0, c), min(W, d)]

    _ROI_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ROI_CACHE_PATH.write_text(json.dumps(rois, indent=2))
    return rois


def crop(arr: np.ndarray, roi):
    y0, y1, x0, x1 = roi
    return arr[y0:y1, x0:x1]
