r"""Shared helpers for the final publication figures (figures/).

Strict rule: every pixel here is either (a) loaded verbatim from a frozen
source file, or (b) a crop/composite of arrays already produced in
outputs/manuscript_assets/, or (c) a plot of numbers taken directly from the
frozen results/*.csv files. No new simulation, no new noise draw, no
texture regeneration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

VEB = ROOT / "outputs/manuscript_assets"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CONDITIONS_DIR = ROOT / "data" / "specimen" / "conditions"
SPECIMEN_DIR = ROOT / "data" / "specimen"
RESULTS = ROOT / "results"

# Consistent typography across all 8 figures.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
    "axes.linewidth": 0.6,
    "mathtext.fontset": "dejavusans",
})

PANEL_LABEL_KW = dict(fontsize=11, fontweight="bold", va="top", ha="left")


def panel_label(ax, letter, x=-0.02, y=1.06):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, **PANEL_LABEL_KW)


def imshow_gray(ax, arr, vmin=0.0, vmax=1.0, title=None, cmap="gray", aspect=None,
                 title_fontsize=8, title_pad=8):
    ax.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.5); s.set_color("#888")
    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=title_pad)


def save(fig, name, dpi=350):
    png = OUT / f"{name}.png"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {png.name}")


def load_npy(rel_path):
    p = VEB / rel_path
    if not p.exists():
        raise FileNotFoundError(p)
    return np.load(p)


def dstr(d):
    return f"{d:.2f}".replace(".", "p")


def find_one(pattern_dir, pattern):
    matches = list(pattern_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"{pattern_dir}/{pattern}")
    return matches[0]


ROIS = {
    "protrusion": [549, 610, 989, 1050],
    "hard_step": [130, 470, 560, 960],
    "ramp": [356, 756, 1271, 1611],
    "plateau": [306, 706, 684, 1084],
}
# (loaded from outputs/manuscript_assets/10_manifest/roi_definitions.json at
# import time below, to guarantee exact consistency with the evidence bank)
import json
_roi_path = VEB / "10_manifest" / "roi_definitions.json"
if _roi_path.exists():
    ROIS = json.loads(_roi_path.read_text())


def crop(arr, roi):
    y0, y1, x0, x1 = roi
    return arr[y0:y1, x0:x1]


def autocrop_whitespace(img_arr, pad=6, thresh=250):
    """Trim near-white margins from a reused RGB(A) PNG so it fills its
    subplot cell without excess blank space."""
    a = img_arr[..., :3] if img_arr.ndim == 3 else img_arr
    mask = np.any(a < thresh / 255.0, axis=-1) if a.max() <= 1.0 else np.any(a < thresh, axis=-1)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return img_arr
    y0, y1 = max(0, ys.min() - pad), min(a.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(a.shape[1], xs.max() + pad)
    return img_arr[y0:y1, x0:x1]
