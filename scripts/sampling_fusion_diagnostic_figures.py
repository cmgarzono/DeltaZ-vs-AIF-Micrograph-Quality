#!/usr/bin/env python
"""S3 publication figures: S3-A (error decomposition), S3-B (focus selection),
S3-C (axial redundancy), S3-D (topographic regions). Reads only sampling_fusion_raw.csv."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
S3_DIR = ROOT / "results" / "sampling_fusion"
FIG_DIR = ROOT / "outputs" / "sampling_fusion"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(S3_DIR / "sampling_fusion_raw.csv")

CONDS_H = ["H_LF", "H_COARSE", "H_MIXED", "H_FINE"]
CONDS_P = ["P_LF", "P_COARSE", "P_MIXED", "P_FINE"]
ALL_CONDS = CONDS_H + CONDS_P
COLORS = dict(zip(ALL_CONDS, plt.cm.tab10(np.linspace(0, 1, 8))))

# ----------------------------------------------------------------------
# S3-A: sampling / fusion / total error vs delta, small multiples (window=7)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
for i, cond in enumerate(ALL_CONDS):
    ax = axes.flat[i]
    sub = df[(df.condition == cond) & (df.window == 7)].sort_values("delta")
    ax.plot(sub.delta, sub.sampling_mse, label="sampling MSE", color="tab:blue", lw=1.2)
    ax.plot(sub.delta, sub.fusion_mse, label="fusion MSE", color="tab:orange", lw=1.2)
    ax.plot(sub.delta, sub.total_mse, label="total MSE", color="tab:red", lw=1.2, ls="--")
    ax.axvline(0.5, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.set_title(cond, fontsize=10)
    ax.set_yscale("log")
    if i >= 4:
        ax.set_xlabel("delta (DOF)")
    if i % 4 == 0:
        ax.set_ylabel("MSE (log scale)")
axes.flat[0].legend(fontsize=8, loc="upper left")
fig.suptitle("S3-A: Error Decomposition (Sampling / Fusion / Total) vs Axial Spacing, window=7", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG_DIR / "S3-A_error_decomposition.png", dpi=150)
plt.close(fig)
print("[OK] S3-A_error_decomposition.png")

# ----------------------------------------------------------------------
# S3-B: focus-index agreement + axial-selection error vs delta, 3 windows
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
window_colors = {3: "tab:blue", 7: "tab:green", 15: "tab:red"}
for i, cond_group, title in [(0, CONDS_H, "H family"), (1, CONDS_P, "P family")]:
    ax_top = axes[0, i]
    ax_bot = axes[1, i]
    for w in [3, 7, 15]:
        sub = df[(df.condition.isin(cond_group)) & (df.window == w)].groupby("delta").agg(
            focus_index_agreement=("focus_index_agreement", "mean"),
            axial_error_kF_mean=("axial_error_kF_mean", "mean"),
        ).reset_index()
        ax_top.plot(sub.delta, sub.focus_index_agreement, color=window_colors[w], lw=1.2, label=f"w={w}")
        ax_bot.plot(sub.delta, sub.axial_error_kF_mean, color=window_colors[w], lw=1.2, label=f"w={w}")
    ax_top.set_title(f"{title}: focus-index agreement", fontsize=10)
    ax_bot.set_title(f"{title}: axial-selection error (mean)", fontsize=10)
    ax_bot.set_xlabel("delta (DOF)")
    if i == 0:
        ax_top.set_ylabel("agreement fraction")
        ax_bot.set_ylabel("|z_F - h| (DOF)")
axes[0, 0].legend(fontsize=8)
fig.suptitle("S3-B: Focus Selection Performance vs Axial Spacing (condition-family mean)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIG_DIR / "S3-B_focus_selection.png", dpi=150)
plt.close(fig)
print("[OK] S3-B_focus_selection.png")

# ----------------------------------------------------------------------
# S3-C: K, N_eff, R_Z, adjacent_overlap vs delta (condition-family mean, window=7)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
metrics_c = [("K", "candidate count K"), ("N_eff", "effective independent evidence N_eff"),
             ("R_Z", "axial redundancy R_Z"), ("adjacent_overlap_mean", "mean adjacent overlap")]
for ax, (metric, label) in zip(axes.flat, metrics_c):
    for cond in ALL_CONDS:
        sub = df[(df.condition == cond) & (df.window == 7)].sort_values("delta")
        ax.plot(sub.delta, sub[metric], color=COLORS[cond], lw=0.9, alpha=0.8, label=cond)
    ax.set_title(label, fontsize=10)
    if metric == "K":
        ax.set_yscale("log")
    ax.set_xlabel("delta (DOF)")
axes.flat[0].legend(fontsize=6, ncol=2, loc="upper right")
fig.suptitle("S3-C: Axial Redundancy vs Axial Spacing (window=7)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIG_DIR / "S3-C_axial_redundancy.png", dpi=150)
plt.close(fig)
print("[OK] S3-C_axial_redundancy.png")

# ----------------------------------------------------------------------
# S3-D: regional total MSE + axial error vs delta (condition-family mean, window=7)
# ----------------------------------------------------------------------
regions = ["plateau", "ramp", "hard_step", "protrusion"]
region_colors = {"plateau": "tab:blue", "ramp": "tab:green", "hard_step": "tab:red", "protrusion": "tab:purple"}
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
mean_over_conds = df[df.window == 7].groupby("delta").agg(
    **{f"region_{r}_mse": (f"region_{r}_mse", "mean") for r in regions},
    **{f"region_{r}_axial_error_mean": (f"region_{r}_axial_error_mean", "mean") for r in regions},
).reset_index()
for r in regions:
    axes[0].plot(mean_over_conds.delta, mean_over_conds[f"region_{r}_mse"], color=region_colors[r], label=r, lw=1.3)
    axes[1].plot(mean_over_conds.delta, mean_over_conds[f"region_{r}_axial_error_mean"], color=region_colors[r], label=r, lw=1.3)
axes[0].set_yscale("log")
axes[0].set_title("Regional total MSE vs delta")
axes[0].set_xlabel("delta (DOF)")
axes[1].set_title("Regional axial-selection error vs delta")
axes[1].set_xlabel("delta (DOF)")
axes[0].legend(fontsize=9)
fig.suptitle("S3-D: Topographic Region Behavior (all-condition mean, window=7)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIG_DIR / "S3-D_topographic_regions.png", dpi=150)
plt.close(fig)
print("[OK] S3-D_topographic_regions.png")

print("\n[FIGURES] All 4 figures written to figures/s3/")
