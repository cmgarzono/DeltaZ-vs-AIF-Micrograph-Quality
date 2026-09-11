#!/usr/bin/env python
r"""
Analytical spectral mechanism for axial evidence.

Runs the S2 experiment in one pass: spectral descriptors, constant-width
control, frequency-dependent derivative check, E/M results, CSV outputs,
final figures (panels C-F), and the combined S1+S2 six-panel figure when
the S1 delta-domain CSV is available.

This model does NOT use H/P textures, the physical specimen, photographic oracles,
topography, fusion, or noise. See
src/deltaz_aif/analysis/spectral_evidence.py for the frozen model.

If either internal consistency check fails, this script stops before
generating final figures.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deltaz_aif.analysis.axial_coverage import ALPHA, C_bar_delta, S_delta
from deltaz_aif.analysis import spectral_evidence as s2

RESULTS_DIR = REPO_ROOT / "results" / "spectral_evidence"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = REPO_ROOT / "outputs" / "spectral_evidence"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DELTA_MIN, DELTA_MAX, N_DELTA = 0.001, 4.8, 4800
GATE_A_TOL = 1e-6
GATE_B_ABS_TOL = 5e-3
GATE_B_REL_TOL = 5e-2  # only where M is not near-zero

FIG_WIDTH, FIG_HEIGHT = 7.0, 5.0
COLORS = {"LF": "#1f77b4", "COARSE": "#2ca02c", "MIXED": "#ff7f0e", "FINE": "#d62728"}
LABEL_SIZE, TITLE_SIZE, TICK_SIZE, LEGEND_SIZE = 11, 12, 10, 9


def delta_grid():
    return np.linspace(DELTA_MIN, DELTA_MAX, N_DELTA)


# ---------------------------------------------------------------------------
# 1. Spectral definitions / descriptors
# ---------------------------------------------------------------------------


def write_spectral_definitions(grid: s2.SpectralGrid, desc: dict[str, s2.SpectralDescriptors]) -> pd.DataFrame:
    rows = []
    for r in s2.REGIMES:
        d = desc[r]
        rows.append(
            {
                "regime": r,
                "eta_g": s2.ETA_G[r],
                "f_g": d.f_g,
                "sigma_ln": s2.SIGMA_LN,
                "f_min_numerical": s2.F_MIN,
                "f_max_numerical": s2.F_MAX,
                "truncated_mass_before_renorm": grid.Z[r],
                "integral_normalization_after_renorm": d.integral_normalization,
                "centroid": d.centroid,
                "median_frequency": d.median_frequency,
                "rms_frequency": d.rms_frequency,
                "bandwidth": d.bandwidth,
                "energy_fraction_above_0p25": d.energy_fraction_above_0p25,
                "energy_fraction_above_0p40": d.energy_fraction_above_0p40,
            }
        )
    df = pd.DataFrame(rows)
    path = RESULTS_DIR / "spectral_evidence_spectral_definitions.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {path}")
    return df


# ---------------------------------------------------------------------------
# 2. Constant-width control
# ---------------------------------------------------------------------------


def run_gate_a(grid: s2.SpectralGrid, d: np.ndarray) -> bool:
    print("\n=== Constant-width control ===")
    ref_E = C_bar_delta(d)
    ref_M = S_delta(d)
    max_diff_E = 0.0
    max_diff_M = 0.0
    for r in s2.REGIMES:
        e_cw = s2.E_CW(d, r, grid)
        m_cw = s2.M_CW(d, r, grid)
        de = float(np.max(np.abs(e_cw - ref_E)))
        dm = float(np.max(np.abs(m_cw - ref_M)))
        max_diff_E = max(max_diff_E, de)
        max_diff_M = max(max_diff_M, dm)
        print(f"  {r}: max|E_CW-C_bar_delta|={de:.3e}  max|M_CW-S_delta|={dm:.3e}")

    e05 = float(s2.E_CW(np.array([0.5]), "LF", grid)[0])
    m05 = float(s2.M_CW(np.array([0.5]), "LF", grid)[0])
    e05_target, m05_target = 0.945121, 0.208449
    print(f"  E_CW(0.5)={e05:.6f} (target ~{e05_target}, diff={abs(e05 - e05_target):.2e})")
    print(f"  M_CW(0.5)={m05:.6f} (target ~{m05_target}, diff={abs(m05 - m05_target):.2e})")

    passed = (
        max_diff_E < GATE_A_TOL
        and max_diff_M < GATE_A_TOL
        and abs(e05 - e05_target) < 1e-3
        and abs(m05 - m05_target) < 1e-3
    )
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


# ---------------------------------------------------------------------------
# 3. Frequency-dependent model and derivative check
# ---------------------------------------------------------------------------


def run_fd_model(grid: s2.SpectralGrid, d: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for r in s2.REGIMES:
        out[r] = {"E": s2.E_FD(d, r, grid), "M": s2.M_FD(d, r, grid)}
    return out


def run_gate_b(grid: s2.SpectralGrid) -> tuple[bool, pd.DataFrame]:
    print("\n=== Frequency-dependent derivative check ===")
    # Subset away from grid edges so central differences stay in-domain.
    d_val = np.linspace(0.02, 4.78, 200)
    rows = []
    all_pass = True
    for r in s2.REGIMES:
        m_analytic = s2.M_FD(d_val, r, grid)
        m_numeric = s2.M_FD_numerical_derivative(d_val, r, grid, h=1e-4)
        abs_err = np.abs(m_analytic - m_numeric)
        max_abs = float(np.max(abs_err))
        rms = float(np.sqrt(np.mean(abs_err**2)))
        nonzero = np.abs(m_analytic) > 1e-3
        rel_err = np.abs(abs_err[nonzero] / m_analytic[nonzero])
        rep_rel = float(np.max(rel_err)) if rel_err.size else 0.0
        ok = max_abs < GATE_B_ABS_TOL and rep_rel < GATE_B_REL_TOL
        all_pass = all_pass and ok
        rows.append(
            {"regime": r, "max_abs_error": max_abs, "rms_error": rms, "representative_rel_error": rep_rel, "pass": ok}
        )
        print(f"  {r}: max_abs_err={max_abs:.3e}  rms_err={rms:.3e}  rep_rel_err={rep_rel:.3e}  {'PASS' if ok else 'FAIL'}")
    print(f"  Result: {'PASS' if all_pass else 'FAIL'}")
    return all_pass, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Main results / CSV outputs
# ---------------------------------------------------------------------------


def write_curves_csv(d: np.ndarray, grid: s2.SpectralGrid, fd: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    frames = []
    ref_E_cw = C_bar_delta(d)
    ref_M_cw = S_delta(d)
    for r in s2.REGIMES:
        frames.append(pd.DataFrame({"delta": d, "model": "CW", "regime": r, "E": ref_E_cw, "M": ref_M_cw}))
        frames.append(pd.DataFrame({"delta": d, "model": "FD", "regime": r, "E": fd[r]["E"], "M": fd[r]["M"]}))
    df = pd.concat(frames, ignore_index=True)
    path = RESULTS_DIR / "spectral_evidence_curves.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df)} rows)")
    return df


def compute_summary(
    d: np.ndarray, grid: s2.SpectralGrid, fd: dict[str, dict[str, np.ndarray]], desc: dict[str, s2.SpectralDescriptors]
) -> tuple[pd.DataFrame, dict]:
    rows = []
    idx05 = int(np.argmin(np.abs(d - 0.5)))
    sensmax = {}
    for r in s2.REGIMES:
        sm = s2.locate_fd_sensitivity_maximum_from_grid(r, d, fd[r]["M"])
        sensmax[r] = sm
        dsc = desc[r]
        rows.append(
            {
                "regime": r,
                "f_g": dsc.f_g,
                "centroid": dsc.centroid,
                "median_frequency": dsc.median_frequency,
                "rms_frequency": dsc.rms_frequency,
                "bandwidth": dsc.bandwidth,
                "E_at_delta_0p5": float(fd[r]["E"][idx05]),
                "M_at_delta_0p5": float(fd[r]["M"][idx05]),
                "delta_Mmax": sm.delta_max,
                "Mmax": sm.M_max,
            }
        )
    df = pd.DataFrame(rows)
    path = RESULTS_DIR / "spectral_evidence_summary.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {path}")
    return df, sensmax


def compute_global_summary(d: np.ndarray, fd: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    E_stack = np.vstack([fd[r]["E"] for r in s2.REGIMES])  # (4, N_DELTA)
    spread = E_stack.max(axis=0) - E_stack.min(axis=0)
    i_max = int(np.argmax(spread))

    # Ordering at delta=0.5 and crossing detection across the whole grid.
    idx05 = int(np.argmin(np.abs(d - 0.5)))
    order05 = sorted(s2.REGIMES, key=lambda r: fd[r]["E"][idx05])

    crossings = []
    regimes = s2.REGIMES
    for i in range(len(regimes)):
        for j in range(i + 1, len(regimes)):
            r1, r2 = regimes[i], regimes[j]
            diff = fd[r1]["E"] - fd[r2]["E"]
            sign = np.sign(diff)
            change = np.where(np.diff(sign) != 0)[0]
            for k in change:
                crossings.append(
                    {
                        "regime_1": r1,
                        "regime_2": r2,
                        "delta_crossing": float((d[k] + d[k + 1]) / 2.0),
                    }
                )

    global_row = {
        "delta_spread_max": float(spread[i_max]),
        "delta_at_max_spread": float(d[i_max]),
        "ordering_at_delta_0p5": " > ".join(reversed(order05)),
        "n_crossings": len(crossings),
    }
    df_global = pd.DataFrame([global_row])
    df_crossings = pd.DataFrame(crossings) if crossings else pd.DataFrame(columns=["regime_1", "regime_2", "delta_crossing"])

    return df_global, df_crossings, spread


# ---------------------------------------------------------------------------
# 5. Figures (panels C, D, E, F)
# ---------------------------------------------------------------------------


def mark_delta_half(ax, x_fn, y_fn):
    x = 0.5
    y = float(y_fn(np.array([x]))[0]) if callable(y_fn) else y_fn
    ax.plot([x], [y], "o", color="gray", markersize=6, zorder=5)
    ax.axvline(x, color="gray", ls="--", lw=1.0, alpha=0.6, zorder=1)


def plot_panel_C(d, grid):
    e = s2.E_CW(d, "LF", grid)  # all four coincide by construction
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.plot(d, e, color="#1f77b4", lw=2.0, zorder=3)
    ax.axvline(0.5, color="gray", ls="--", lw=1.0, alpha=0.6, zorder=1)
    e05 = float(np.interp(0.5, d, e))
    ax.plot([0.5], [e05], "o", color="gray", markersize=6, zorder=5)
    ax.text(0.55, 0.15, "all four spectral conditions coincident", fontsize=8, color="#555555", transform=ax.transData)
    ax.set_xlabel(r"$\delta = \Delta z / \mathrm{DOF}$", fontsize=LABEL_SIZE)
    ax.set_ylabel("Constant-width evidence, E(δ)", fontsize=LABEL_SIZE)
    ax.set_title("Panel C: Constant-Width Evidence", fontsize=TITLE_SIZE)
    ax.set_xlim(0, DELTA_MAX)
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(alpha=0.25, linestyle=":", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "spectral_evidence_panel_C_constant_width_evidence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote outputs/spectral_evidence/spectral_evidence_panel_C_constant_width_evidence.png")


def plot_panel_D(d, grid, sens_max_s1_delta):
    m = s2.M_CW(d, "LF", grid)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.plot(d, m, color="#2ca02c", lw=2.0, zorder=3)
    ax.axvline(0.5, color="gray", ls="--", lw=1.0, alpha=0.6, zorder=1)
    m05 = float(np.interp(0.5, d, m))
    ax.plot([0.5], [m05], "o", color="gray", markersize=6, zorder=5)
    ax.axvline(sens_max_s1_delta, color="black", ls=":", lw=1.0, alpha=0.6, zorder=1)
    ax.text(0.55, m05 - 0.03, "all four spectral conditions coincident", fontsize=8, color="#555555")
    ax.set_xlabel(r"$\delta = \Delta z / \mathrm{DOF}$", fontsize=LABEL_SIZE)
    ax.set_ylabel("Constant-width sensitivity, M(δ)", fontsize=LABEL_SIZE)
    ax.set_title("Panel D: Constant-Width Sensitivity", fontsize=TITLE_SIZE)
    ax.set_xlim(0, DELTA_MAX)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(alpha=0.25, linestyle=":", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "spectral_evidence_panel_D_constant_width_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote outputs/spectral_evidence/spectral_evidence_panel_D_constant_width_sensitivity.png")


def plot_panel_E(d, fd):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    for r in s2.REGIMES:
        ax.plot(d, fd[r]["E"], color=COLORS[r], lw=1.8, label=r, zorder=3)
    ax.axvline(0.5, color="gray", ls="--", lw=1.0, alpha=0.6, zorder=1)
    ax.set_xlabel(r"$\delta = \Delta z / \mathrm{DOF}$", fontsize=LABEL_SIZE)
    ax.set_ylabel("Frequency-dependent evidence, E(δ)", fontsize=LABEL_SIZE)
    ax.set_title("Panel E: Frequency-Dependent Evidence", fontsize=TITLE_SIZE)
    ax.set_xlim(0, DELTA_MAX)
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(loc="upper right", fontsize=LEGEND_SIZE, framealpha=0.95)
    ax.grid(alpha=0.25, linestyle=":", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "spectral_evidence_panel_E_frequency_dependent_evidence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote outputs/spectral_evidence/spectral_evidence_panel_E_frequency_dependent_evidence.png")


def plot_panel_F(d, fd, sensmax):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    for r in s2.REGIMES:
        ax.plot(d, fd[r]["M"], color=COLORS[r], lw=1.8, label=r, zorder=3)
    ax.axvline(0.5, color="gray", ls="--", lw=1.0, alpha=0.6, zorder=1)
    for r in s2.REGIMES:
        sm = sensmax[r]
        ax.plot([sm.delta_max], [sm.M_max], marker="^", color=COLORS[r], markersize=5, zorder=5)
    ax.set_xlabel(r"$\delta = \Delta z / \mathrm{DOF}$", fontsize=LABEL_SIZE)
    ax.set_ylabel("Frequency-dependent sensitivity, M(δ)", fontsize=LABEL_SIZE)
    ax.set_title("Panel F: Frequency-Dependent Sensitivity", fontsize=TITLE_SIZE)
    ax.set_xlim(0, DELTA_MAX)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(loc="upper right", fontsize=LEGEND_SIZE, framealpha=0.95)
    ax.grid(alpha=0.25, linestyle=":", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "spectral_evidence_panel_F_frequency_dependent_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote outputs/spectral_evidence/spectral_evidence_panel_F_frequency_dependent_sensitivity.png")


def plot_six_panel(d, grid, fd, sensmax, sens_max_s1_delta):
    """Reconstruct S1 panels (a,b) from S1 CSV data + S2 panels (c-f) into
    one six-panel figure, only if S1 delta-domain CSVs are available.
    """
    s1_full = REPO_ROOT / "results" / "axial_coverage" / "axial_coverage_curve_delta.csv"
    if not s1_full.exists():
        print("axial_coverage_curve_delta.csv not found -- skipping six-panel figure.")
        return
    df_s1 = pd.read_csv(s1_full)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    ax_a, ax_b, ax_c = axes[0]
    ax_d, ax_e, ax_f = axes[1]

    # (a) S1 coverage
    ax_a.plot(df_s1["delta"], df_s1["C_bar"], color="#1f77b4", lw=1.8)
    ax_a.axvline(0.5, color="gray", ls="--", lw=1.0, alpha=0.6)
    ax_a.set_title("(a) S1 coverage", fontsize=TITLE_SIZE)
    ax_a.set_xlabel(r"$\delta$", fontsize=LABEL_SIZE)
    ax_a.set_ylabel("C(δ)", fontsize=LABEL_SIZE)
    ax_a.set_xlim(0, DELTA_MAX)
    ax_a.set_ylim(0, 1.02)
    ax_a.grid(alpha=0.25, linestyle=":", linewidth=0.7)

    # (b) S1 sensitivity
    ax_b.plot(df_s1["delta"], df_s1["S_delta"], color="#2ca02c", lw=1.8)
    ax_b.axvline(0.5, color="gray", ls="--", lw=1.0, alpha=0.6)
    ax_b.axvline(sens_max_s1_delta, color="black", ls=":", lw=1.0, alpha=0.6)
    ax_b.set_title("(b) S1 sensitivity", fontsize=TITLE_SIZE)
    ax_b.set_xlabel(r"$\delta$", fontsize=LABEL_SIZE)
    ax_b.set_ylabel("S(δ)", fontsize=LABEL_SIZE)
    ax_b.set_xlim(0, DELTA_MAX)
    ax_b.grid(alpha=0.25, linestyle=":", linewidth=0.7)

    # (c) S2 CW evidence
    e_cw = s2.E_CW(d, "LF", grid)
    ax_c.plot(d, e_cw, color="#1f77b4", lw=1.8)
    ax_c.axvline(0.5, color="gray", ls="--", lw=1.0, alpha=0.6)
    ax_c.set_title("(c) S2 constant-width evidence", fontsize=TITLE_SIZE)
    ax_c.set_xlabel(r"$\delta$", fontsize=LABEL_SIZE)
    ax_c.set_ylabel("E(δ)", fontsize=LABEL_SIZE)
    ax_c.set_xlim(0, DELTA_MAX)
    ax_c.set_ylim(0, 1.02)
    ax_c.grid(alpha=0.25, linestyle=":", linewidth=0.7)

    # (d) S2 CW sensitivity
    m_cw = s2.M_CW(d, "LF", grid)
    ax_d.plot(d, m_cw, color="#2ca02c", lw=1.8)
    ax_d.axvline(0.5, color="gray", ls="--", lw=1.0, alpha=0.6)
    ax_d.set_title("(d) S2 constant-width sensitivity", fontsize=TITLE_SIZE)
    ax_d.set_xlabel(r"$\delta$", fontsize=LABEL_SIZE)
    ax_d.set_ylabel("M(δ)", fontsize=LABEL_SIZE)
    ax_d.set_xlim(0, DELTA_MAX)
    ax_d.grid(alpha=0.25, linestyle=":", linewidth=0.7)

    # (e) S2 FD evidence
    for r in s2.REGIMES:
        ax_e.plot(d, fd[r]["E"], color=COLORS[r], lw=1.5, label=r)
    ax_e.axvline(0.5, color="gray", ls="--", lw=1.0, alpha=0.6)
    ax_e.set_title("(e) S2 frequency-dependent evidence", fontsize=TITLE_SIZE)
    ax_e.set_xlabel(r"$\delta$", fontsize=LABEL_SIZE)
    ax_e.set_ylabel("E(δ)", fontsize=LABEL_SIZE)
    ax_e.set_xlim(0, DELTA_MAX)
    ax_e.set_ylim(0, 1.02)
    ax_e.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_e.grid(alpha=0.25, linestyle=":", linewidth=0.7)

    # (f) S2 FD sensitivity
    for r in s2.REGIMES:
        ax_f.plot(d, fd[r]["M"], color=COLORS[r], lw=1.5, label=r)
    ax_f.axvline(0.5, color="gray", ls="--", lw=1.0, alpha=0.6)
    ax_f.set_title("(f) S2 frequency-dependent sensitivity", fontsize=TITLE_SIZE)
    ax_f.set_xlabel(r"$\delta$", fontsize=LABEL_SIZE)
    ax_f.set_ylabel("M(δ)", fontsize=LABEL_SIZE)
    ax_f.set_xlim(0, DELTA_MAX)
    ax_f.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_f.grid(alpha=0.25, linestyle=":", linewidth=0.7)

    for ax in axes.flat:
        ax.tick_params(labelsize=TICK_SIZE)

    fig.tight_layout()
    out = FIG_DIR / "axial_coverage_spectral_evidence_six_panel.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    print("Analytical spectral mechanism for axial evidence")
    print("Not used: H/P textures, specimen geometry, oracles, topography, fusion, noise.\n")

    d = delta_grid()
    print(f"delta grid: {len(d)} points over [{DELTA_MIN}, {DELTA_MAX}]")

    grid = s2.build_spectral_grid()
    desc = s2.compute_descriptors(grid)
    write_spectral_definitions(grid, desc)

    for r in s2.REGIMES:
        print(f"  {r}: f_g={desc[r].f_g:.4f}  centroid={desc[r].centroid:.6f}  "
              f"median={desc[r].median_frequency:.6f}  rms={desc[r].rms_frequency:.6f}  "
              f"bandwidth={desc[r].bandwidth:.6f}")

    gate_a_pass = run_gate_a(grid, d)
    if not gate_a_pass:
        print("\nConstant-width control failed.")
        sys.exit(1)

    fd = run_fd_model(grid, d)

    gate_b_pass, df_gate_b = run_gate_b(grid)
    if not gate_b_pass:
        print("\nFrequency-dependent sensitivity check failed.")
        sys.exit(1)

    write_curves_csv(d, grid, fd)
    df_summary, sensmax = compute_summary(d, grid, fd, desc)
    print("\nSummary (delta=0.5, sensitivity maxima):")
    print(df_summary.to_string(index=False))

    df_global, df_crossings, spread = compute_global_summary(d, fd)
    print("\nGlobal summary:")
    print(df_global.to_string(index=False))
    if len(df_crossings):
        print("\nCrossings:")
        print(df_crossings.to_string(index=False))
    else:
        print("\nNo crossings detected between any regime pair over the tested domain.")

    # S1 sensitivity-maximum landmark (delta domain) for panel D annotation
    from deltaz_aif.analysis.axial_coverage import locate_sensitivity_maximum_delta

    s1_smax = locate_sensitivity_maximum_delta()

    print("\n=== Generating figures ===")
    plot_panel_C(d, grid)
    plot_panel_D(d, grid, s1_smax.delta_max)
    plot_panel_E(d, fd)
    plot_panel_F(d, fd, sensmax)
    plot_six_panel(d, grid, fd, sensmax, s1_smax.delta_max)

    print("\nS2 complete.")


if __name__ == "__main__":
    main()
