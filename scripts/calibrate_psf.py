#!/usr/bin/env python
r"""
Generate the calibration table and figure for the through-focus
Gaussian-PSF core (sigma_from_defocus / mtf_gaussian).

Outputs:
    results/calibration/psf_calibration_table.csv
    outputs/calibration/psf_calibration_curve.png

Scope: numerical calibration only. No specimen, no height map, no stacks.
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

from deltaz_aif.focus.psf_core import F_REF, SIGMA_PER_DOF, mtf_gaussian
from deltaz_aif.focus.psf_validation import boundary_condition_report, run_calibration_table

D_VALUES = [0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0]


def main():
    rows = run_calibration_table(D_VALUES)
    df = pd.DataFrame(
        [
            {
                "d (DOF)": r.d,
                "sigma (px)": r.sigma_px,
                "MTF_analytical": r.mtf_analytical,
                "C(d)/C(0) numeric": r.contrast_ratio_numeric,
                "abs_error": r.abs_error,
                "rel_error": r.rel_error,
            }
            for r in sorted(rows, key=lambda r: r.d)
        ]
    )
    csv_path = REPO_ROOT / "results" / "calibration" / "psf_calibration_table.csv"
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSIGMA_PER_DOF = {SIGMA_PER_DOF:.6f} px/DOF")
    print(f"F_REF = {F_REF} cycles/px")

    # Boundary condition check summary
    print("\nBoundary condition (reflective, constant signal value=1.0):")
    for sigma in (1.0, SIGMA_PER_DOF * 0.5, SIGMA_PER_DOF * 1.0):
        rep = boundary_condition_report(sigma=sigma)
        print(f"  sigma={sigma:6.3f}px  min={rep['min_value']:.9f}  "
              f"max_abs_dev={rep['max_abs_deviation']:.2e}")

    # ---- Figure ----
    d_dense = np.linspace(-1.5, 1.5, 400)
    sigma_dense = SIGMA_PER_DOF * np.abs(d_dense)
    mtf_dense = mtf_gaussian(F_REF, sigma_dense)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(d_dense, mtf_dense, "-", color="#1f77b4", lw=2, label="Analytical MTF(f_ref, sigma(d))")
    ax.plot(
        df["d (DOF)"], df["C(d)/C(0) numeric"], "o", color="#d62728",
        ms=7, mfc="none", mew=2, label="Numeric contrast ratio C(d)/C(0)",
    )
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.7)
    ax.axvline(0.5, color="gray", ls="--", lw=1, alpha=0.7)
    ax.axvline(-0.5, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Defocus d (DOF units)")
    ax.set_ylabel("Contrast response at f_ref = 0.032 cyc/px")
    ax.set_title("Gaussian PSF calibration -- analytical vs numeric")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig_path = REPO_ROOT / "outputs" / "calibration" / "psf_calibration_curve.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")
    print(f"Table saved: {csv_path}")


if __name__ == "__main__":
    main()
