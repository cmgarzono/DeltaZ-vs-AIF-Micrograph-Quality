r"""
S4 analysis: summary/scaling/landmark tables, computed
strictly from results/multiplicity_noise/multiplicity_noise_raw.csv (no recomputation of images).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

RAW_PATH = ROOT / "results" / "multiplicity_noise" / "multiplicity_noise_raw.csv"
OUT_DIR = ROOT / "results" / "multiplicity_noise"
FIG_DIR = ROOT / "outputs" / "multiplicity_noise"
FIG_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["MSE", "PSNR", "laplacian_variance", "dct_hf_energy"]


def rule_window_label(row):
    if row["rule"] == "maxfocus":
        return f"maxfocus_w{int(row['window'])}"
    return row["rule"]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df["window"] = df["window"].replace("NA", np.nan)
    df["selection_entropy"] = pd.to_numeric(df["selection_entropy"], errors="coerce")
    df["selection_entropy_normalized"] = pd.to_numeric(df["selection_entropy_normalized"], errors="coerce")
    df["rule_label"] = df.apply(rule_window_label, axis=1)
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["sigma", "N", "rule", "window"]
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        sigma, N, rule, window = keys
        row = {"sigma": sigma, "N": N, "rule": rule, "window": window if pd.notna(window) else "NA"}
        for m in METRICS + ["selection_entropy", "selection_entropy_normalized", "clipping_fraction"]:
            vals = g[m].dropna()
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_std"] = np.nan
                row[f"{m}_median"] = np.nan
                row[f"{m}_P05"] = np.nan
                row[f"{m}_P95"] = np.nan
                continue
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"] = float(vals.std())
            row[f"{m}_median"] = float(vals.median())
            row[f"{m}_P05"] = float(np.percentile(vals, 5))
            row[f"{m}_P95"] = float(np.percentile(vals, 95))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["rule", "window", "sigma", "N"]).reset_index(drop=True)


def build_scaling_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Explicit dependence of MSE on N, per (rule, window, sigma):
    fitted log(MSE) vs log(N) slope, plus relative MSE change N=1 -> N=128."""
    rows = []
    for (rule, window, sigma), g in df.groupby(["rule", "window", "sigma"], dropna=False):
        g = g.groupby("N")["MSE"].mean().reset_index().sort_values("N")
        Ns = g["N"].values.astype(np.float64)
        mses = g["MSE"].values.astype(np.float64)
        valid = mses > 0
        if valid.sum() >= 2:
            slope = float(np.polyfit(np.log(Ns[valid]), np.log(mses[valid]), 1)[0])
        else:
            slope = np.nan
        mse_N1 = float(g.loc[g["N"] == 1, "MSE"].iloc[0]) if (g["N"] == 1).any() else np.nan
        mse_N128 = float(g.loc[g["N"] == 128, "MSE"].iloc[0]) if (g["N"] == 128).any() else np.nan
        rel_change = (mse_N128 - mse_N1) / mse_N1 if mse_N1 else np.nan
        rows.append({
            "rule": rule, "window": window if pd.notna(window) else "NA", "sigma": sigma,
            "log_log_slope_N_vs_MSE": slope,
            "mse_at_N1": mse_N1, "mse_at_N128": mse_N128,
            "relative_mse_change_N1_to_N128": rel_change,
        })
    return pd.DataFrame(rows).sort_values(["rule", "window", "sigma"]).reset_index(drop=True)


def build_landmarks(df: pd.DataFrame) -> pd.DataFrame:
    Ns = [1, 4, 16, 64, 128]
    sigmas = [0.01, 0.02, 0.04]
    sub = df[df["N"].isin(Ns) & df["sigma"].isin(sigmas)]
    rows = []
    for (sigma, N, rule, window), g in sub.groupby(["sigma", "N", "rule", "window"], dropna=False):
        row = {"sigma": sigma, "N": N, "rule": rule, "window": window if pd.notna(window) else "NA"}
        for m in ["MSE", "PSNR", "laplacian_variance", "dct_hf_energy", "selection_entropy_normalized"]:
            vals = g[m].dropna()
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_P05"] = np.nan
                row[f"{m}_P95"] = np.nan
                continue
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_P05"] = float(np.percentile(vals, 5))
            row[f"{m}_P95"] = float(np.percentile(vals, 95))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["sigma", "N", "rule", "window"]).reset_index(drop=True)


def main():
    df = load_raw()
    print(f"Loaded {len(df)} raw rows.")

    summary = build_summary(df)
    summary.to_csv(OUT_DIR / "multiplicity_noise_summary.csv", index=False)
    print(f"Wrote multiplicity_noise_summary.csv ({len(summary)} rows).")

    scaling = build_scaling_summary(df)
    scaling.to_csv(OUT_DIR / "multiplicity_noise_scaling_summary.csv", index=False)
    print(f"Wrote multiplicity_noise_scaling_summary.csv ({len(scaling)} rows).")

    landmarks = build_landmarks(df)
    landmarks.to_csv(OUT_DIR / "multiplicity_noise_landmarks.csv", index=False)
    print(f"Wrote multiplicity_noise_landmarks.csv ({len(landmarks)} rows).")

    return df, summary, scaling, landmarks


if __name__ == "__main__":
    main()
