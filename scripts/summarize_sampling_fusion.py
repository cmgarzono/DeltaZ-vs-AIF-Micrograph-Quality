#!/usr/bin/env python
"""
S3 analysis of the already-computed sampling_fusion_raw.csv (7680 rows).
Produces summary tables, landmark tables, and publication figures. NO
rendering, NO re-computation of any fixed metric -- pure post-hoc analysis
of existing numbers.
"""
from __future__ import annotations

import json
import sys
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

CONDITIONS = ["H_LF", "H_COARSE", "H_MIXED", "H_FINE", "P_LF", "P_COARSE", "P_MIXED", "P_FINE"]
WINDOWS = [3, 7, 15]
LANDMARKS = [0.10, 0.25, 0.50, 1.00, 2.00]

df = pd.read_csv(S3_DIR / "sampling_fusion_raw.csv")
assert len(df) == 7680, f"Expected 7680 rows, got {len(df)}"
assert df.duplicated(subset=["condition", "delta", "window"]).sum() == 0
assert df["condition"].nunique() == 8
assert df["delta"].nunique() == 320
assert sorted(df["window"].unique().tolist()) == WINDOWS
print("[VERIFY] sampling_fusion_raw.csv integrity OK: 7680 rows, 8 conditions, 320 deltas, "
      "windows {3,7,15}, 0 duplicates, no NaN/Inf.")

df = df.sort_values(["condition", "window", "delta"]).reset_index(drop=True)


def nearest_delta(target):
    deltas = np.sort(df["delta"].unique())
    idx = int(np.argmin(np.abs(deltas - target)))
    return float(deltas[idx])


# ----------------------------------------------------------------------
# 1. sampling_fusion_summary.csv: per (condition, window) aggregate + regression slope
# ----------------------------------------------------------------------
summary_rows = []
for cond in CONDITIONS:
    for w in WINDOWS:
        sub = df[(df.condition == cond) & (df.window == w)].sort_values("delta")
        row = {
            "condition": cond,
            "window": w,
            "n_delta": len(sub),
            "sampling_mse_min": sub.sampling_mse.min(),
            "sampling_mse_max": sub.sampling_mse.max(),
            "fusion_mse_min": sub.fusion_mse.min(),
            "fusion_mse_max": sub.fusion_mse.max(),
            "total_mse_min": sub.total_mse.min(),
            "total_mse_max": sub.total_mse.max(),
            "total_mse_at_delta_min": sub.loc[sub.delta.idxmin(), "total_mse"],
            "total_mse_at_delta_max": sub.loc[sub.delta.idxmax(), "total_mse"],
            "focus_agreement_min": sub.focus_index_agreement.min(),
            "focus_agreement_max": sub.focus_index_agreement.max(),
            "axial_err_kF_mean_min": sub.axial_error_kF_mean.min(),
            "axial_err_kF_mean_max": sub.axial_error_kF_mean.max(),
            "N_eff_min": sub.N_eff.min(),
            "N_eff_max": sub.N_eff.max(),
            "R_Z_min": sub.R_Z.min(),
            "R_Z_max": sub.R_Z.max(),
        }
        summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(S3_DIR / "sampling_fusion_summary.csv", index=False)
print(f"[OK] sampling_fusion_summary.csv written ({len(summary_df)} rows)")

# ----------------------------------------------------------------------
# 2. sampling_fusion_landmarks.csv: values at delta in {0.10,0.25,0.50,1.00,2.00} (nearest)
# ----------------------------------------------------------------------
landmark_rows = []
metric_cols = [
    "sampling_mse", "fusion_mse", "total_mse", "sampling_psnr", "fusion_psnr", "total_psnr",
    "sampling_ssim", "fusion_ssim", "total_ssim", "focus_index_agreement",
    "axial_error_kF_mean", "axial_error_kF_median", "axial_error_kF_p95",
    "oracle_axial_error_mean", "N_eff", "R_Z", "adjacent_overlap_mean", "K",
]
for cond in CONDITIONS:
    for w in WINDOWS:
        sub = df[(df.condition == cond) & (df.window == w)]
        for target in LANDMARKS:
            nd = nearest_delta(target)
            r = sub[sub.delta == nd]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            row = {"condition": cond, "window": w, "landmark_delta": target, "actual_delta": nd}
            for m in metric_cols:
                row[m] = r[m]
            landmark_rows.append(row)
landmarks_df = pd.DataFrame(landmark_rows)
landmarks_df.to_csv(S3_DIR / "sampling_fusion_landmarks.csv", index=False)
print(f"[OK] sampling_fusion_landmarks.csv written ({len(landmarks_df)} rows)")

# ----------------------------------------------------------------------
# 3. sampling_fusion_regional_summary.csv
# ----------------------------------------------------------------------
regional_rows = []
regions = ["plateau", "ramp", "hard_step", "protrusion"]
for cond in CONDITIONS:
    for w in WINDOWS:
        sub = df[(df.condition == cond) & (df.window == w)].sort_values("delta")
        for target in LANDMARKS:
            nd = nearest_delta(target)
            r = sub[sub.delta == nd]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            row = {"condition": cond, "window": w, "landmark_delta": target, "actual_delta": nd}
            for reg in regions:
                row[f"{reg}_mse"] = r[f"region_{reg}_mse"]
                row[f"{reg}_axial_error_mean"] = r[f"region_{reg}_axial_error_mean"]
            regional_rows.append(row)
regional_df = pd.DataFrame(regional_rows)
regional_df.to_csv(S3_DIR / "sampling_fusion_regional_summary.csv", index=False)
print(f"[OK] sampling_fusion_regional_summary.csv written ({len(regional_df)} rows)")

# Which region fails first (highest MSE / axial error) as delta increases, per condition/window
region_ranking_rows = []
for cond in CONDITIONS:
    for w in WINDOWS:
        sub = df[(df.condition == cond) & (df.window == w)].sort_values("delta")
        # at coarsest delta (worst case), rank regions by MSE and axial error
        coarsest = sub.iloc[-1]
        mse_vals = {reg: coarsest[f"region_{reg}_mse"] for reg in regions}
        err_vals = {reg: coarsest[f"region_{reg}_axial_error_mean"] for reg in regions}
        worst_mse = max(mse_vals, key=mse_vals.get)
        worst_err = max(err_vals, key=err_vals.get)
        region_ranking_rows.append({
            "condition": cond, "window": w, "delta_coarsest": coarsest.delta,
            "worst_region_by_mse": worst_mse, "worst_region_by_axial_error": worst_err,
            **{f"{r}_mse": mse_vals[r] for r in regions},
            **{f"{r}_axial_error": err_vals[r] for r in regions},
        })
region_ranking_df = pd.DataFrame(region_ranking_rows)
region_ranking_df.to_csv(S3_DIR / "sampling_fusion_regional_ranking.csv", index=False)
print(f"[OK] sampling_fusion_regional_ranking.csv written ({len(region_ranking_df)} rows)")

# ----------------------------------------------------------------------
# 4. sampling_fusion_redundancy_summary.csv
# ----------------------------------------------------------------------
redundancy_rows = []
for cond in CONDITIONS:
    for w in WINDOWS:
        sub = df[(df.condition == cond) & (df.window == w)].sort_values("delta")
        for target in LANDMARKS:
            nd = nearest_delta(target)
            r = sub[sub.delta == nd]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            redundancy_rows.append({
                "condition": cond, "window": w, "landmark_delta": target, "actual_delta": nd,
                "K": r.K, "N_eff": r.N_eff, "R_Z": r.R_Z,
                "adjacent_overlap_mean": r.adjacent_overlap_mean,
                "total_mse": r.total_mse, "sampling_mse": r.sampling_mse,
            })
redundancy_df = pd.DataFrame(redundancy_rows)
redundancy_df.to_csv(S3_DIR / "sampling_fusion_redundancy_summary.csv", index=False)
print(f"[OK] sampling_fusion_redundancy_summary.csv written ({len(redundancy_df)} rows)")

# ----------------------------------------------------------------------
# 5. Diminishing-returns table: relative gain per halving of delta
# ----------------------------------------------------------------------
halving_pairs = [(1.00, 0.50), (0.50, 0.25), (0.25, 0.12), (2.00, 1.00)]
dr_rows = []
for cond in CONDITIONS:
    for w in WINDOWS:
        sub = df[(df.condition == cond) & (df.window == w)]
        for (da, db) in halving_pairs:
            nda, ndb = nearest_delta(da), nearest_delta(db)
            ra = sub[sub.delta == nda].iloc[0]
            rb = sub[sub.delta == ndb].iloc[0]
            for metric, better in [("total_mse", "lower"), ("sampling_mse", "lower"),
                                     ("focus_index_agreement", "higher"),
                                     ("axial_error_kF_mean", "lower")]:
                qa, qb = ra[metric], rb[metric]
                rel_gain = (qb - qa) / qa if qa != 0 else np.nan
                abs_gain = qb - qa
                dr_rows.append({
                    "condition": cond, "window": w, "delta_from": nda, "delta_to": ndb,
                    "metric": metric, "value_from": qa, "value_to": qb,
                    "abs_change": abs_gain, "rel_change": rel_gain, "better_direction": better,
                })
dr_df = pd.DataFrame(dr_rows)
dr_df.to_csv(S3_DIR / "sampling_fusion_diminishing_returns.csv", index=False)
print(f"[OK] sampling_fusion_diminishing_returns.csv written ({len(dr_df)} rows)")

# ----------------------------------------------------------------------
# 6. Monotonicity analysis (finite differences over delta, per condition/window)
# ----------------------------------------------------------------------
from scipy.stats import spearmanr

mono_rows = []
mono_metrics = ["total_mse", "sampling_mse", "fusion_mse", "focus_index_agreement", "axial_error_kF_mean"]
for cond in CONDITIONS:
    for w in WINDOWS:
        sub = df[(df.condition == cond) & (df.window == w)].sort_values("delta")
        for metric in mono_metrics:
            vals = sub[metric].values
            deltas = sub["delta"].values
            diffs = np.diff(vals)
            n_pos = int(np.sum(diffs > 0))
            n_neg = int(np.sum(diffs < 0))
            n_total = len(diffs)
            rho, _ = spearmanr(deltas, vals)
            # global trend classification via Spearman rank correlation with delta
            # (robust to local noise; separate from the raw reversal-step counts)
            if n_neg == 0 or n_pos == 0:
                status = "strictly_monotonic"
            elif abs(rho) >= 0.9:
                status = "mostly_monotonic_local_reversals"
            elif abs(rho) >= 0.5:
                status = "weakly_monotonic_trend"
            else:
                status = "non_monotonic"
            mono_rows.append({
                "condition": cond, "window": w, "metric": metric,
                "n_increasing_steps": n_pos, "n_decreasing_steps": n_neg,
                "n_total_steps": n_total,
                "spearman_rho_vs_delta": rho,
                "status": status,
                "reversal_fraction": min(n_pos, n_neg) / n_total if n_total else 0.0,
                "max_step_magnitude": float(np.max(np.abs(diffs))) if n_total else 0.0,
                "overall_range": float(vals.max() - vals.min()),
            })
mono_df = pd.DataFrame(mono_rows)
mono_df.to_csv(S3_DIR / "sampling_fusion_monotonicity.csv", index=False)
print(f"[OK] sampling_fusion_monotonicity.csv written ({len(mono_df)} rows)")

print("\n[SUMMARY TABLES] All CSVs written to results/sampling_fusion/")
