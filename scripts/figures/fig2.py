r"""Fig. 2 -- Analytical focal evidence and sensitivity. Regenerated
(not a flat-image copy) from the same frozen data and plotting logic as
the frozen S2 model in `deltaz_aif.analysis.spectral_evidence` and
`results/axial_coverage/axial_coverage_curve_delta.csv`."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import ROOT, OUT, save

sys.path.insert(0, str(ROOT / "src"))
from deltaz_aif.analysis import spectral_evidence as s2

DELTA_MIN, DELTA_MAX, N_DELTA = 0.001, 4.8, 4800
COLORS = {"LF": "#1f77b4", "COARSE": "#2ca02c", "MIXED": "#ff7f0e", "FINE": "#d62728"}


def main():
    d = np.linspace(DELTA_MIN, DELTA_MAX, N_DELTA)
    grid = s2.build_spectral_grid()

    fd = {}
    for r in s2.REGIMES:
        fd[r] = {"E": s2.E_FD(d, r, grid), "M": s2.M_FD(d, r, grid)}

    df_s1 = pd.read_csv(ROOT / "results" / "axial_coverage" / "axial_coverage_curve_delta.csv")
    sens_max_s1_delta = float(df_s1.loc[df_s1["S_delta"].idxmax(), "delta"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    ax_a, ax_b, ax_c = axes[0]
    ax_d, ax_e, ax_f = axes[1]

    ax_a.plot(df_s1["delta"], df_s1["C_bar"], color="#1f77b4", lw=1.8)
    ax_a.axvline(0.5, color="gray", ls="--", lw=1.0, alpha=0.6)
    ax_a.set_title("(a) Mean focal coverage")
    ax_a.set_xlabel(r"$\delta = \Delta z / \mathrm{DOF}$")
    ax_a.set_ylabel(r"$\bar{C}(\delta)$")
    ax_a.set_xlim(0, DELTA_MAX); ax_a.set_ylim(0, 1.02)
    ax_a.grid(alpha=0.25, linestyle=":", linewidth=0.7)

    ax_b.plot(df_s1["delta"], df_s1["S_delta"], color="#2ca02c", lw=1.8)
    ax_b.axvline(0.5, color="gray", ls="--", lw=1.0, alpha=0.6)
    ax_b.axvline(sens_max_s1_delta, color="black", ls=":", lw=1.0, alpha=0.6)
    ax_b.set_title("(b) Marginal sensitivity")
    ax_b.set_xlabel(r"$\delta = \Delta z / \mathrm{DOF}$")
    ax_b.set_ylabel(r"$S(\delta)$")
    ax_b.set_xlim(0, DELTA_MAX)
    ax_b.grid(alpha=0.25, linestyle=":", linewidth=0.7)

    e_cw = s2.E_CW(d, "LF", grid)
    ax_c.plot(d, e_cw, color="#1f77b4", lw=1.8)
    ax_c.axvline(0.5, color="gray", ls="--", lw=1.0, alpha=0.6)
    ax_c.set_title("(c) Constant-width evidence")
    ax_c.set_xlabel(r"$\delta = \Delta z / \mathrm{DOF}$")
    ax_c.set_ylabel(r"$E(\delta)$")
    ax_c.set_xlim(0, DELTA_MAX); ax_c.set_ylim(0, 1.02)
    ax_c.grid(alpha=0.25, linestyle=":", linewidth=0.7)

    m_cw = s2.M_CW(d, "LF", grid)
    ax_d.plot(d, m_cw, color="#2ca02c", lw=1.8)
    ax_d.axvline(0.5, color="gray", ls="--", lw=1.0, alpha=0.6)
    ax_d.set_title("(d) Constant-width sensitivity")
    ax_d.set_xlabel(r"$\delta = \Delta z / \mathrm{DOF}$")
    ax_d.set_ylabel(r"$M(\delta)$")
    ax_d.set_xlim(0, DELTA_MAX)
    ax_d.grid(alpha=0.25, linestyle=":", linewidth=0.7)

    for r in s2.REGIMES:
        ax_e.plot(d, fd[r]["E"], color=COLORS[r], lw=1.5, label=r)
    ax_e.axvline(0.5, color="gray", ls="--", lw=1.0, alpha=0.6)
    ax_e.set_title("(e) Frequency-dependent evidence")
    ax_e.set_xlabel(r"$\delta = \Delta z / \mathrm{DOF}$")
    ax_e.set_ylabel(r"$E(\delta)$")
    ax_e.set_xlim(0, DELTA_MAX); ax_e.set_ylim(0, 1.02)
    ax_e.legend(loc="upper right", framealpha=0.9)
    ax_e.grid(alpha=0.25, linestyle=":", linewidth=0.7)

    for r in s2.REGIMES:
        ax_f.plot(d, fd[r]["M"], color=COLORS[r], lw=1.5, label=r)
    ax_f.axvline(0.5, color="gray", ls="--", lw=1.0, alpha=0.6)
    ax_f.set_title("(f) Frequency-dependent sensitivity")
    ax_f.set_xlabel(r"$\delta = \Delta z / \mathrm{DOF}$")
    ax_f.set_ylabel(r"$M(\delta)$")
    ax_f.set_xlim(0, DELTA_MAX)
    ax_f.legend(loc="upper right", framealpha=0.9)
    ax_f.grid(alpha=0.25, linestyle=":", linewidth=0.7)

    fig.tight_layout()
    save(fig, "Figure2")


if __name__ == "__main__":
    main()
