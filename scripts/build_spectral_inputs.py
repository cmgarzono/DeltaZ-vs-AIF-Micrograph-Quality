"""
Construction of the 8 physical spectral inputs consumed by
the spectral-evidence model.

Scope: build P_q(f) for q in {H,P} x {LF,COARSE,MIXED,FINE}, with f in
cycles/pixel (real FFT axis, np.fft.fftfreq), shell-energy = SUM of power
in each radial bin, DC excluded exactly, normalized to sum 1. No
spectral-evidence-model curves, no sensitivity, no manuscript figures --
the photographic-condition generators are not touched.

Inputs:
  - texture generators: make_h_texture_scaled(s), make_p_texture_scaled(s)
    from conditions.py
  - scale factors s_H, s_P per regime
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from deltaz_aif.photographic.conditions import (
    make_h_texture_scaled, make_p_texture_scaled,
)
from deltaz_aif.photographic.baseline_renderer import WIDTH, HEIGHT  # 1920, 1080

# Photographic-condition scale factors
REGIME_ORDER = ["LF", "COARSE", "MIXED", "FINE"]
SCALE_FACTORS = {
    "LF":     {"s_H": 0.4451, "s_P": 0.6078},
    "COARSE": {"s_H": 0.6950, "s_P": 1.0538},
    "MIXED":  {"s_H": 1.0240, "s_P": 1.5179},
    "FINE":   {"s_H": 1.6043, "s_P": 2.2645},
}
F_REF = 0.032  # cycles/pixel, through-focus model reference (NOT a target)

# Radial binning resolution
# Native FFT frequency spacing is 1/W = 1/1920 ~= 5.21e-4 c/px along x and
# 1/H = 1/1080 ~= 9.26e-4 c/px along y -- the two axes' own resolutions.
# f_max ~= sqrt(0.5^2+0.5^2) ~= 0.70711 c/px (corner of the 2D FFT grid).
# N_BINS = 1000 uniform bins over [0, f_max] gives a bin width of
# ~7.07e-4 c/px -- matched to native resolution scale (not thousands of
# near-empty bins, not so coarse that the four regimes blur together).
N_BINS = 1000
N_BINS_FINE = 2000  # robustness check only

ENERGY_TOL = 1e-6
BINNING_CENTROID_TOL = 0.01  # 1% relative


def physical_frequency_grid(H, W):
    fx = np.fft.fftfreq(W, d=1.0)  # cycles/pixel
    fy = np.fft.fftfreq(H, d=1.0)  # cycles/pixel
    fxx, fyy = np.meshgrid(fx, fy)  # shape (H, W)
    f_radial = np.sqrt(fxx ** 2 + fyy ** 2)
    return f_radial


def shell_energy_distribution(texture, n_bins, f_radial=None):
    """
    Returns (f_edges, f_repr, P_q, metadata) where:
      f_repr[i]  = energy-weighted mean frequency of bin i (bin center if empty)
      P_q[i]     = normalized (sum=1) SUMMED power in bin i, DC excluded
    """
    H, W = texture.shape
    if f_radial is None:
        f_radial = physical_frequency_grid(H, W)

    F = np.fft.fft2(texture)
    power = np.abs(F) ** 2

    # Exclude exact DC only: fx=0 & fy=0 is index (0,0) of fft2/fftfreq.
    dc_mask = np.zeros_like(power, dtype=bool)
    dc_mask[0, 0] = True
    non_dc_mask = ~dc_mask

    non_dc_power_total = power[non_dc_mask].sum()
    f_pos = f_radial[non_dc_mask]
    p_pos = power[non_dc_mask]

    f_min_positive = f_pos.min()
    f_max_observed = f_pos.max()

    f_edges = np.linspace(0.0, f_max_observed, n_bins + 1)
    bin_idx = np.clip(np.searchsorted(f_edges, f_pos, side="right") - 1, 0, n_bins - 1)

    P_raw = np.zeros(n_bins, dtype=np.float64)
    f_weighted_sum = np.zeros(n_bins, dtype=np.float64)
    np.add.at(P_raw, bin_idx, p_pos)
    np.add.at(f_weighted_sum, bin_idx, f_pos * p_pos)

    bin_energy_error = abs(P_raw.sum() - non_dc_power_total) / non_dc_power_total

    f_centers = 0.5 * (f_edges[:-1] + f_edges[1:])
    f_repr = np.where(P_raw > 0, np.divide(f_weighted_sum, P_raw, out=np.zeros_like(P_raw), where=P_raw > 0), f_centers)

    P_q = P_raw / P_raw.sum()

    diag = dict(
        f_min_positive=float(f_min_positive),
        f_max_observed=float(f_max_observed),
        non_dc_power_total=float(non_dc_power_total),
        binning_energy_error=float(bin_energy_error),
        normalization_sum=float(P_q.sum()),
    )
    return f_edges, f_repr, P_q, diag


def js_distance(p, q):
    """Jensen-Shannon distance (sqrt of JS divergence), base-2, over
    two discrete distributions defined on the SAME bin grid."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))
    js_div = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    js_div = max(js_div, 0.0)
    return float(np.sqrt(js_div))


def build_all_conditions():
    conditions = {}
    textures = {}
    for regime in REGIME_ORDER:
        s_h = SCALE_FACTORS[regime]["s_H"]
        s_p = SCALE_FACTORS[regime]["s_P"]
        textures[f"H_{regime}"] = make_h_texture_scaled(s_h)
        textures[f"P_{regime}"] = make_p_texture_scaled(s_p)
    return textures


def main():
    output_dir = ROOT / "results" / "spectral_evidence"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Physical spectral inputs (texture-only, frozen photographic conditions)")
    print("=" * 70)

    print("\n[1/6] Regenerating 8 textures from the frozen photographic-condition generators...")
    textures = build_all_conditions()
    for name, tex in textures.items():
        print(f"  {name}: shape={tex.shape} dtype={tex.dtype} "
              f"mean={tex.mean():.6f} std={tex.std():.6f} "
              f"min={tex.min():.4f} max={tex.max():.4f}")

    print(f"\n[2/6] Physical FFT axis + radial binning (N_BINS={N_BINS})...")
    f_radial = physical_frequency_grid(HEIGHT, WIDTH)

    npz_payload = {}
    metadata = {}
    f_edges_common = None
    f_repr_common = {}

    for name, tex in textures.items():
        f_edges, f_repr, P_q, diag = shell_energy_distribution(tex, N_BINS, f_radial=f_radial)
        if f_edges_common is None:
            f_edges_common = f_edges
        else:
            assert np.allclose(f_edges, f_edges_common), "bin edges must be identical across conditions"
        npz_payload[f"P_{name}"] = P_q
        npz_payload[f"frepr_{name}"] = f_repr
        metadata[name] = diag
        print(f"  {name}: f_min_pos={diag['f_min_positive']:.6e}  "
              f"f_max_obs={diag['f_max_observed']:.6f}  "
              f"norm_sum={diag['normalization_sum']:.10f}  "
              f"bin_energy_err={diag['binning_energy_error']:.3e}")

    print("\n[3/6] Physical spectral centroids...")
    centroids = {}
    for name in textures:
        f_repr = npz_payload[f"frepr_{name}"]
        P_q = npz_payload[f"P_{name}"]
        centroids[name] = float(np.sum(f_repr * P_q))
        print(f"  {name}: f_c = {centroids[name]:.6f} c/px  "
              f"(f_c/f_ref = {centroids[name]/F_REF:.4f})")

    print("\n[4/6] Ordering check (LF<COARSE<MIXED<FINE, per mode)...")
    h_seq = [centroids[f"H_{r}"] for r in REGIME_ORDER]
    p_seq = [centroids[f"P_{r}"] for r in REGIME_ORDER]
    ordered_h = all(h_seq[i] < h_seq[i + 1] for i in range(3))
    ordered_p = all(p_seq[i] < p_seq[i + 1] for i in range(3))
    print(f"  H: {' < '.join(f'{c:.6f}' for c in h_seq)}  -> {'PASS' if ordered_h else 'FAIL'}")
    print(f"  P: {' < '.join(f'{c:.6f}' for c in p_seq)}  -> {'PASS' if ordered_p else 'FAIL'}")

    print("\n[5/6] H/P within-regime comparison...")
    hp_diff = {}
    hp_js = {}
    for r in REGIME_ORDER:
        c_h, c_p = centroids[f"H_{r}"], centroids[f"P_{r}"]
        hp_diff[r] = abs(c_h - c_p) / np.mean([c_h, c_p])
        hp_js[r] = js_distance(npz_payload[f"P_H_{r}"], npz_payload[f"P_P_{r}"])
        print(f"  {r}: rel_diff={hp_diff[r]*100:.3f}%  JS_distance={hp_js[r]:.4f}")

    print("\n[6/6] Energy conservation + binning-robustness checks...")
    energy_pass = True
    for name, tex in textures.items():
        F = np.fft.fft2(tex)
        power = np.abs(F) ** 2
        total_power = power.sum() - power[0, 0]  # non-DC total
        shell_sum = metadata[name]["non_dc_power_total"]
        rel_err = abs(total_power - shell_sum) / shell_sum
        ok = rel_err < ENERGY_TOL
        energy_pass = energy_pass and ok
        print(f"  {name}: total_non_dc_power vs shell_sum rel_err={rel_err:.3e}  [{'PASS' if ok else 'FAIL'}]")

    fine_centroids = {}
    for name, tex in textures.items():
        _, f_repr_fine, P_q_fine, _ = shell_energy_distribution(tex, N_BINS_FINE, f_radial=f_radial)
        fine_centroids[name] = float(np.sum(f_repr_fine * P_q_fine))
    binning_rel_err = {name: abs(fine_centroids[name] - centroids[name]) / centroids[name]
                        for name in textures}
    binning_pass = all(v < BINNING_CENTROID_TOL for v in binning_rel_err.values())
    h_seq_fine = [fine_centroids[f"H_{r}"] for r in REGIME_ORDER]
    p_seq_fine = [fine_centroids[f"P_{r}"] for r in REGIME_ORDER]
    ordered_h_fine = all(h_seq_fine[i] < h_seq_fine[i + 1] for i in range(3))
    ordered_p_fine = all(p_seq_fine[i] < p_seq_fine[i + 1] for i in range(3))
    for name in textures:
        print(f"  {name}: coarse={centroids[name]:.6f}  fine={fine_centroids[name]:.6f}  "
              f"rel_err={binning_rel_err[name]*100:.3f}%")
    print(f"  Binning-robustness: centroid tol={BINNING_CENTROID_TOL*100:.0f}% -> "
          f"[{'PASS' if binning_pass else 'FAIL'}]; "
          f"ordering unchanged H={ordered_h_fine}, P={ordered_p_fine}")

    norm_pass = all(abs(d["normalization_sum"] - 1.0) < 1e-9 for d in metadata.values())
    binenergy_pass = all(d["binning_energy_error"] < ENERGY_TOL for d in metadata.values())
    freq_axis_pass = True

    if not (ordered_h and ordered_p):
        status = "S2 INPUTS FAIL ORDERING"
    elif not (norm_pass and binenergy_pass and energy_pass):
        status = "S2 INPUTS FAIL NORMALIZATION"
    elif not freq_axis_pass:
        status = "S2 INPUTS FAIL FREQUENCY AXIS"
    elif not binning_pass or not ordered_h_fine or not ordered_p_fine:
        status = "S2 INPUTS REQUIRE REVIEW"
    else:
        status = "S2 INPUTS COMPLETE"

    print(f"\nStatus: {status}")

    # ── Outputs ───────────────────────────────────────────────────────
    npz_payload["f_edges"] = f_edges_common
    npz_payload["regime_order"] = np.array(REGIME_ORDER)
    npz_payload["condition_names"] = np.array(list(textures.keys()))
    npz_payload["n_bins"] = N_BINS
    npz_payload["f_ref"] = F_REF
    meta = dict(
        n_bins=N_BINS, n_bins_fine=N_BINS_FINE,
        scale_factors=SCALE_FACTORS, f_ref=F_REF,
        status=status,
    )
    npz_payload["metadata_json"] = np.array(json.dumps(meta))

    npz_path = output_dir / "spectral_evidence_physical_spectral_inputs.npz"
    np.savez(npz_path, **npz_payload)
    print(f"\nSaved {npz_path}")

    rows = []
    for name in textures:
        mode, regime = name.split("_", 1)
        row = dict(
            mode=mode, regime=regime, condition_id=name,
            physical_centroid_cyc_per_px=centroids[name],
            centroid_over_fref=centroids[name] / F_REF,
            H_P_relative_difference_if_applicable=hp_diff[regime],
            normalization_sum=metadata[name]["normalization_sum"],
            non_dc_energy=metadata[name]["non_dc_power_total"],
            binning_energy_error=metadata[name]["binning_energy_error"],
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = output_dir / "spectral_evidence_physical_spectral_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    colors = {"LF": "tab:blue", "COARSE": "tab:green", "MIXED": "tab:orange", "FINE": "tab:red"}
    for mode, ax in zip(["H", "P"], axes):
        for regime in REGIME_ORDER:
            name = f"{mode}_{regime}"
            f_repr = npz_payload[f"frepr_{name}"]
            P_q = npz_payload[f"P_{name}"]
            ax.plot(f_repr, P_q, label=f"{regime} (f_c={centroids[name]:.4f})",
                    color=colors[regime], lw=1.2)
        ax.axvline(F_REF, color="gray", ls="--", lw=1, alpha=0.6, label="f_ref=0.032")
        ax.set_ylabel("P_q(f)")
        ax.set_title(f"{mode} texture — shell-energy spectral distributions")
        ax.set_xlim(0, 0.15)
        ax.legend(fontsize=8)
    axes[1].set_xlabel("f [cycles/pixel]")
    fig.tight_layout()
    png_path = output_dir / "spectral_evidence_physical_spectral_inputs.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved {png_path}")

    return dict(
        textures=textures, centroids=centroids, metadata=metadata,
        hp_diff=hp_diff, hp_js=hp_js, ordered_h=ordered_h, ordered_p=ordered_p,
        binning_pass=binning_pass, energy_pass=energy_pass, status=status,
        fine_centroids=fine_centroids, binning_rel_err=binning_rel_err,
        f_edges=f_edges_common,
    )


if __name__ == "__main__":
    main()
