#!/usr/bin/env python
r"""
S1: analytical axial coverage model -- full curve and figures.

Specimen-independent, renderer-independent. No M1, no height_map, no
photographic-condition oracles, no through-focus renderer, no PSF/renderer
fusion, no noise.

Outputs:
    results/axial_coverage/axial_coverage_curve.csv
    results/axial_coverage/axial_coverage_curve_delta.csv
    results/axial_coverage/axial_coverage_summary_points.csv
    results/axial_coverage/axial_coverage_summary_points_delta.csv
    outputs/axial_coverage/axial_coverage_curve.png
    outputs/axial_coverage/axial_coverage_sensitivity.png
    outputs/axial_coverage/axial_coverage_fine_asymptotic.png
    outputs/axial_coverage/axial_coverage_coarse_asymptotic.png

This script does not execute S2, S3, S4, M1, oracles, fusion, or noise
experiments.
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

from deltaz_aif.analysis.axial_coverage import (
    DOF_LANDMARK,
    ALPHA,
    SQRT_PI,
    C_bar,
    C_bar_delta,
    C_bar_numerical_integration,
    S_analytical,
    S_delta,
    S_numerical_derivative,
    coarse_asymptotic_coverage,
    coverage_deficit,
    fine_asymptotic_deficit,
    locate_sensitivity_maximum,
    validate_against_numerical_integration,
)

S_MIN, S_MAX = 0.001, 8.0
N_GRID = 20_001  # dense uniform grid; ~4e-4 spacing in s_z

FIG_DIR = REPO_ROOT / "outputs" / "axial_coverage"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = REPO_ROOT / "results" / "axial_coverage"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_full_curve():
    s = np.linspace(S_MIN, S_MAX, N_GRID)
    c = C_bar(s)
    deficit = coverage_deficit(s)
    S_ana = S_analytical(s)
    S_num = S_numerical_derivative(s, h=1e-5)
    fine_approx = fine_asymptotic_deficit(s)
    coarse_approx = coarse_asymptotic_coverage(s)

    df = pd.DataFrame(
        {
            "s_z": s,
            "C_bar": c,
            "coverage_deficit": deficit,
            "S_analytical": S_ana,
            "S_numeric": S_num,
            "derivative_abs_error": np.abs(S_ana - S_num),
            "fine_asymptotic_deficit": fine_approx,
            "coarse_asymptotic_coverage": coarse_approx,
        }
    )
    return df


def build_summary_points(smax_result):
    s_points = sorted(
        set(
            [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, smax_result.s_z_max, 2.0, 4.0, 8.0]
        )
    )
    rows = []
    for s in s_points:
        c = float(C_bar(s))
        deficit = float(coverage_deficit(s))
        S_ana = float(S_analytical(s))
        S_num = float(S_numerical_derivative(np.array([s]), h=1e-5)[0])
        rows.append(
            {
                "s_z": s,
                "C_bar": c,
                "coverage_deficit": deficit,
                "S_analytical": S_ana,
                "S_numeric": S_num,
                "derivative_abs_error": abs(S_ana - S_num),
                "fine_asymptotic_deficit": float(fine_asymptotic_deficit(s)),
                "coarse_asymptotic_coverage": float(coarse_asymptotic_coverage(s)),
            }
        )
    return pd.DataFrame(rows)


def to_delta_curve(df_full):
    delta = df_full["s_z"].to_numpy(dtype=np.float64) / ALPHA
    return pd.DataFrame(
        {
            "delta": delta,
            "s_z": df_full["s_z"],
            "C_bar": C_bar_delta(delta),
            "coverage_deficit": 1.0 - C_bar_delta(delta),
            "S_delta": S_delta(delta),
        }
    )


def to_delta_summary(df_summary):
    delta = df_summary["s_z"].to_numpy(dtype=np.float64) / ALPHA
    out = df_summary.copy()
    out.insert(0, "delta", delta)
    out["C_bar_delta"] = C_bar_delta(delta)
    out["S_delta"] = S_delta(delta)
    return out


def build_analytical_vs_numeric():
    sample = [0.001, 0.01, 0.05, 0.1, 0.25, 0.496, 0.5, 1.0, 1.936, 2.0, 4.0, 8.0]
    rows = validate_against_numerical_integration(np.array(sample))
    df = pd.DataFrame(
        [
            {
                "s_z": r.s_z,
                "analytical": r.analytical,
                "numerical": r.numerical,
                "abs_error": r.abs_error,
                "rel_error": r.rel_error,
            }
            for r in rows
        ]
    )
    return df


def plot_coverage(df_full, df_vs_num):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df_full["s_z"], df_full["C_bar"], color="#1f77b4", lw=2, label="C_bar(s_z) analytical")
    ax.scatter(
        df_vs_num["s_z"],
        df_vs_num["numerical"],
        color="#d62728",
        s=28,
        zorder=5,
        label="Numerical integration",
    )
    ax.axvline(DOF_LANDMARK, color="gray", ls="--", lw=1.2, label="s_z = 0.5 reference")
    ax.set_xlabel("s_z (normalized axial spacing)")
    ax.set_ylabel("C_bar(s_z)")
    ax.set_title("S1-A: Mean analytical axial coverage")
    ax.set_xlim(0, S_MAX)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "axial_coverage_curve.png", dpi=200)
    plt.close(fig)


def plot_sensitivity(df_full, smax_result):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df_full["s_z"], df_full["S_analytical"], color="#2ca02c", lw=2, label="S(s_z) analytical")
    # sparse numeric-derivative overlay for legibility
    sparse = df_full.iloc[::400]
    ax.scatter(sparse["s_z"], sparse["S_numeric"], color="#ff7f0e", s=14, zorder=5, label="Numerical derivative")
    ax.axvline(DOF_LANDMARK, color="gray", ls="--", lw=1.2, label="s_z = 0.5 reference")
    ax.axvline(smax_result.s_z_max, color="black", ls=":", lw=1.2, label=f"argmax S at s_z={smax_result.s_z_max:.3f}")
    ax.set_xlabel("s_z (normalized axial spacing)")
    ax.set_ylabel("S(s_z) = -dC_bar/ds_z")
    ax.set_title("S1-B: Marginal sensitivity of axial coverage")
    ax.set_xlim(0, S_MAX)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "axial_coverage_sensitivity.png", dpi=200)
    plt.close(fig)


def plot_fine_asymptotic(df_full):
    mask = df_full["s_z"] <= 1.0
    sub = df_full[mask]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sub["s_z"], sub["coverage_deficit"], color="#1f77b4", lw=2, label="1 - C_bar(s_z) (exact)")
    ax.plot(sub["s_z"], sub["fine_asymptotic_deficit"], color="#d62728", lw=1.6, ls="--", label="s_z^2 / 12 (asymptotic form)")
    ax.axvline(DOF_LANDMARK, color="gray", ls="--", lw=1.0, label="s_z = 0.5 reference")
    ax.set_xlabel("s_z (normalized axial spacing)")
    ax.set_ylabel("Coverage deficit")
    ax.set_title("S1-C: Fine-sampling asymptotic form")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "axial_coverage_fine_asymptotic.png", dpi=200)
    plt.close(fig)


def plot_coarse_asymptotic(df_full):
    mask = df_full["s_z"] >= 1.5
    sub = df_full[mask]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sub["s_z"], sub["C_bar"], color="#1f77b4", lw=2, label="C_bar(s_z) (exact)")
    ax.plot(sub["s_z"], sub["coarse_asymptotic_coverage"], color="#d62728", lw=1.6, ls="--", label="sqrt(pi)/s_z (asymptotic form)")
    ax.set_xlabel("s_z (normalized axial spacing)")
    ax.set_ylabel("Coverage")
    ax.set_title("S1-D: Coarse-sampling asymptotic form")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "axial_coverage_coarse_asymptotic.png", dpi=200)
    plt.close(fig)


def main():
    print(f"S1: grid resolution = {N_GRID} points over s_z in [{S_MIN}, {S_MAX}] "
          f"(spacing ~ {(S_MAX - S_MIN) / (N_GRID - 1):.6e})")

    df_full = build_full_curve()
    full_csv = RESULTS_DIR / "axial_coverage_curve.csv"
    df_full.to_csv(full_csv, index=False)
    print(f"Wrote {full_csv} ({len(df_full)} rows)")

    df_delta = to_delta_curve(df_full)
    delta_csv = RESULTS_DIR / "axial_coverage_curve_delta.csv"
    df_delta.to_csv(delta_csv, index=False)
    print(f"Wrote {delta_csv} ({len(df_delta)} rows)")

    smax_result = locate_sensitivity_maximum(S_MIN, S_MAX, n=2_000_001)
    print(
        f"Max sensitivity: s_z,max={smax_result.s_z_max:.6f}, "
        f"S_max={smax_result.S_max:.6f}, C_bar(s_z,max)={smax_result.C_bar_at_max:.6f}, "
        f"deficit={smax_result.coverage_deficit_at_max:.6f}"
    )

    df_summary = build_summary_points(smax_result)
    summary_csv = RESULTS_DIR / "axial_coverage_summary_points.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"Wrote {summary_csv}")
    print(df_summary.to_string(index=False))

    df_summary_delta = to_delta_summary(df_summary)
    summary_delta_csv = RESULTS_DIR / "axial_coverage_summary_points_delta.csv"
    df_summary_delta.to_csv(summary_delta_csv, index=False)
    print(f"Wrote {summary_delta_csv}")

    df_vs_num = build_analytical_vs_numeric()
    print(df_vs_num.to_string(index=False))

    # Landmark sanity checks
    landmarks = {0.05: 0.99979, 0.496: 0.97987, 8.0: 0.22156}
    print("\nLandmark sanity checks:")
    for s, expected in landmarks.items():
        val = float(C_bar(s))
        print(f"  C_bar({s}) = {val:.6f}  (expected ~{expected}, diff={abs(val - expected):.2e})")

    # Asymptotic checks
    s_fine = 1e-3
    fine_ratio = float(coverage_deficit(s_fine)) / s_fine**2
    print(f"\nFine-limit check: (1-C_bar)/s_z^2 at s_z={s_fine} -> {fine_ratio:.6f} (target 1/12={1/12:.6f})")

    s_coarse = 8.0
    coarse_product = s_coarse * float(C_bar(s_coarse))
    print(f"Coarse-limit check: s_z*C_bar at s_z={s_coarse} -> {coarse_product:.6f} (target sqrt(pi)={SQRT_PI:.6f})")

    plot_coverage(df_full, df_vs_num)
    plot_sensitivity(df_full, smax_result)
    plot_fine_asymptotic(df_full)
    plot_coarse_asymptotic(df_full)
    print(f"\nWrote figures to {FIG_DIR}")

    print("\nAxial-coverage model complete (analytical, specimen-independent -- "
          "no photographic model, oracles, fusion, or noise involved).")


if __name__ == "__main__":
    main()
