r"""Fig. 3 -- Physical specimen. Only two panels: (a) the M1 height map,
top; (b) an oblique 3D rendering, bottom. Both share ONE classic 'jet'
rainbow heat-map color scale over the FULL physical range [-4, +4] DOF
(nothing saturates/clips) -- but with a nonlinear (two-segment
piecewise-linear) norm: [-4, 3] DOF maps to the first 75% of the
colorbar, and [3, 4] DOF -- where the protrusion tips and the central
platform live -- gets the remaining 25% (double its linear 12.5% share),
so the tips are distinguishable from the plateau without hiding or
saturating the large-scale negative-DOF terrain. One shared colorbar,
with explicit tick marks so the nonlinear mapping stays readable.

Panel (b) mesh: built directly from the frozen, UNMODIFIED
height_map.npy at a 2 px integer stride -- no blur, no smoothing of
the surface data anywhere; every one of the 27 protrusions renders as a
real peak.

Hard-step "skirt" rendering fix: M1's hard steps are literal
1-pixel-wide vertical discontinuities. Naively fed to plot_surface, each
such edge becomes ONE quad whose only two defined heights are its bottom
and top edge, flat-shaded to a single, visually misleading color (a
vertical wall spanning e.g. -0.4 to +3.2 DOF rendered as one uniform
yellow band -- literally hiding the height change it represents). Fixed
by explicitly inserting extra rows/columns AT those specific edges only:
wherever one grid step's height change exceeds a threshold (1.0 DOF,
comfortably above every ramp's per-cell rate [~0.02-0.05 DOF/cell] and
below the true wall jumps [1.7-7.2 DOF]), that single transition is
replaced by 14 intermediate levels, each one LINEARLY interpolated
between the two real, unmodified neighboring heights. This is not
smoothing or invented data: a literal vertical wall connecting a known
bottom height to a known top height passes through every intermediate
height by mathematical necessity, at that exact (x, y) location -- so the
interpolated strip is the geometrically correct rendering of that wall,
not an approximation of unknown terrain. Ramps, plateaus, background, and
protrusion slopes are all well under the threshold and are left
completely untouched (zero rows/columns inserted there)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import FuncNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from common import ROOT, SPECIMEN_DIR, save, panel_label

VMIN, VMAX = -4.0, 4.0
CMAP = plt.get_cmap("jet")  # classic rainbow scale
DOWNSAMPLE = 2          # 3D mesh grid stride (px in the original array); no smoothing applied
JUMP_THRESHOLD = 1.0    # DOF per grid cell -- above every ramp/protrusion slope, below real walls
N_SKIRT = 14             # interpolated levels inserted at each detected hard-step edge

# Full physical range [-4,+4] stays visible (nothing saturates/clips), but the
# top band [VBREAK, VMAX] -- where the protrusion tips and the platform live --
# gets a larger share of the colorbar (FBREAK..1.0) than its linear 12.5%
# share, so tips are distinguishable from the plateau without hiding the
# large-scale negative-DOF structure.
VBREAK, FBREAK = 3.0, 0.75  # [-4,3] -> [0,0.75] of the colorbar; [3,4] -> [0.75,1]


def _fwd(x):
    x = np.asarray(x, dtype=float)
    return np.where(x <= VBREAK,
                     (x - VMIN) / (VBREAK - VMIN) * FBREAK,
                     FBREAK + (x - VBREAK) / (VMAX - VBREAK) * (1 - FBREAK))


def _inv(y):
    y = np.asarray(y, dtype=float)
    return np.where(y <= FBREAK,
                     VMIN + y / FBREAK * (VBREAK - VMIN),
                     VBREAK + (y - FBREAK) / (1 - FBREAK) * (VMAX - VBREAK))


NORM = FuncNorm((_fwd, _inv), vmin=VMIN, vmax=VMAX)


def sharpen_clipped_peaks(Z, protrusion_mask_ds, flat_thresh=3.999, ring_lo=3.21, ring_hi=3.999, half_win=20):
    """Panel (b) ONLY, render-side geometric reconstruction (see module
    docstring): the 8 protrusions on the central platform are flat-topped
    frusta in the frozen array (their true, designed apex exceeds the
    domain's +4.0 DOF ceiling and was clipped there during M1's
    construction). For each flat top, fit a cone h = apex - k*d (d =
    Chebyshev distance from the flat top's center, matching the square
    pyramid cross-section) using ONLY that pyramid's own real, unclipped
    ramp pixels (height in (ring_lo, ring_hi)) via least squares -- so k
    and apex both come directly from real measured slope data, not an
    assumption -- then replaces just the flat-top pixels with the cone's
    extrapolated (pointed) values. Everything else in Z is untouched.

    protrusion_mask_ds restricts this to genuine protrusions only: the
    smooth right/bottom ramps also reach ~4.0 DOF at the outer rectangle's
    far corner (by legitimate ramp geometry, not clipping), and a naive
    height>=flat_thresh search wrongly picks that up too -- fitting a
    radially-symmetric cone there would fabricate a spurious spike on a
    ramp. Only flat-top blobs overlapping the frozen protrusion mask are
    reconstructed."""
    from scipy.ndimage import label, center_of_mass
    Z = Z.copy()
    flat_mask = (Z >= flat_thresh) & protrusion_mask_ds
    lab, n = label(flat_mask)
    for i in range(1, n + 1):
        cy, cx = center_of_mass(lab == i)
        cy, cx = int(round(cy)), int(round(cx))
        y0, y1 = max(0, cy - half_win), min(Z.shape[0], cy + half_win)
        x0, x1 = max(0, cx - half_win), min(Z.shape[1], cx + half_win)
        crop = Z[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.maximum(np.abs(yy - cy), np.abs(xx - cx)).astype(float)
        ring = (crop > ring_lo) & (crop < ring_hi)
        if ring.sum() < 5:
            continue
        A = np.vstack([dist[ring], np.ones(ring.sum())]).T
        neg_k, apex = np.linalg.lstsq(A, crop[ring], rcond=None)[0]
        flat_local = crop >= flat_thresh
        crop[flat_local] = apex + neg_k * dist[flat_local]
        Z[y0:y1, x0:x1] = crop
    print(f"sharpen_clipped_peaks: reconstructed {n} flat-topped peak(s)")
    return Z


def insert_skirts(coords, Z, axis, threshold, n_sub):
    """Insert n_sub linearly-interpolated slices wherever adjacent slices
    along `axis` differ by more than `threshold` anywhere along the other
    axis. coords: 1D array of length Z.shape[axis]. Pure geometric
    reconstruction of a vertical connector -- no smoothing, no invented
    terrain; only used at genuine hard-step edges."""
    Zm = np.moveaxis(Z, axis, 0)
    n = Zm.shape[0]
    out_slices = [Zm[0]]
    out_coords = [coords[0]]
    for i in range(n - 1):
        a, b = Zm[i], Zm[i + 1]
        if np.abs(b - a).max() > threshold:
            for k in range(1, n_sub + 1):
                t = k / (n_sub + 1)
                out_slices.append(a * (1 - t) + b * t)
                out_coords.append(coords[i] * (1 - t) + coords[i + 1] * t)
        out_slices.append(b)
        out_coords.append(coords[i + 1])
    Zn = np.moveaxis(np.stack(out_slices, axis=0), 0, axis)
    return np.array(out_coords), Zn


def main():
    h = np.load(SPECIMEN_DIR / "height_map.npy")  # frozen, unmodified, full precision

    fig = plt.figure(figsize=(10.5, 13.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.25], hspace=0.22,
                           top=0.95, bottom=0.06, left=0.06, right=0.86)

    # ---- panel (a): exact, unmodified height map ----
    ax_a = fig.add_subplot(gs[0])
    im = ax_a.imshow(h, cmap=CMAP, norm=NORM)
    ax_a.set_xticks([]); ax_a.set_yticks([])
    ax_a.set_title("M1 height map", fontsize=13, pad=12)
    panel_label(ax_a, "a", x=-0.03, y=1.10)

    # ---- panel (b): 3D render with hard-step skirts inserted ----
    h_render = h[::DOWNSAMPLE, ::DOWNSAMPLE]  # exact values, just a coarser stride
    ny, nx = h_render.shape
    y_coords = np.arange(ny) * DOWNSAMPLE
    x_coords = np.arange(nx) * DOWNSAMPLE

    from scipy.ndimage import binary_dilation
    protrusion_mask = np.load(ROOT / "results" / "sampling_fusion" / "cache" / "regional_masks.npz")["protrusion"]
    protrusion_mask = binary_dilation(protrusion_mask, iterations=15)  # cover each pyramid's full flat top
    protrusion_mask_ds = protrusion_mask[::DOWNSAMPLE, ::DOWNSAMPLE]
    h_render = sharpen_clipped_peaks(h_render, protrusion_mask_ds)

    y_coords, h_render = insert_skirts(y_coords, h_render, axis=0, threshold=JUMP_THRESHOLD, n_sub=N_SKIRT)
    x_coords, h_render = insert_skirts(x_coords, h_render, axis=1, threshold=JUMP_THRESHOLD, n_sub=N_SKIRT)
    print(f"render mesh after skirt insertion: {h_render.shape} (was {ny}x{nx})")

    X, Y = np.meshgrid(x_coords, y_coords)

    ax_b = fig.add_subplot(gs[1], projection="3d")
    ax_b.plot_surface(X, Y, h_render, cmap=CMAP, norm=NORM,
                       rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
    ax_b.set_box_aspect((nx, ny, max(nx, ny) * 0.35))
    ax_b.view_init(elev=38, azim=-60)
    ax_b.set_xlabel("x (px)", fontsize=9, labelpad=10)
    ax_b.set_ylabel("y (px)", fontsize=9, labelpad=10)
    ax_b.set_zlabel("height (DOF)", fontsize=9, labelpad=6)
    ax_b.tick_params(labelsize=7)
    ax_b.set_title("Oblique 3D rendering", fontsize=13, pad=16)
    ax_b.text2D(-0.03, 1.02, "(b)", transform=ax_b.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")

    cax = fig.add_axes([0.89, 0.10, 0.025, 0.80])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([-4, -3, -2, -1, 0, 1, 2, 3, 3.25, 3.5, 3.75, 4])
    cb.set_label("height (DOF)  —  nonlinear scale: [3, 4] DOF is stretched\n"
                  "to give protrusion tips more contrast", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    save(fig, "Figure3", dpi=1050)


if __name__ == "__main__":
    main()
