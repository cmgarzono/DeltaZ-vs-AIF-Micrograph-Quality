r"""
S3 run: from the cached master frame stacks, compute Laplacian^2 evidence ONCE per
(condition, window) across all 801 master positions (never once per
delta), then for each of the 320 delta values select the appropriate
subset of master indices (via `master_lattice_indices`, identical to
`build_candidate_grid`) and produce the fusion / metrics / redundancy
row for the S3 experiment.

Loop order:

  for condition in 8:
      frame_stack = memmap(801, H, W)      # Phase A output, read-only
      for window in (3, 7, 15):
          E_stack = laplacian_energy(frame_stack[j], window) for j in 0..800
                    # computed ONCE per (condition, window)
          for delta in 320 values:
              idx = master_lattice_indices(delta)      # subset selection only
              k_F, I_F   from E_stack[idx], frame_stack[idx]
              k_or, I_or from height_map, z_master[idx], frame_stack[idx]
              metrics, redundancy, regional metrics -> one CSV row

k_or does not depend on photographic condition or window; it is computed
once per delta and cached across the condition/window loops.

Output: results/sampling_fusion/sampling_fusion_raw.csv (8 x 320 x 3 = 7680 rows), checkpointed
so a rerun does not duplicate rows or recompute finished (condition,
window) evidence.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from deltaz_aif.analysis.sampling_fusion import (  # noqa: E402
    build_master_lattice,
    master_lattice_indices,
    N_MASTER,
    nearest_plane_index,
    laplacian_energy,
    image_metrics,
    focus_selection_metrics,
    focus_index_agreement,
    axial_redundancy,
)

SPECIMEN_DIR = ROOT / "data" / "specimen"
CONDITIONS_DIR = ROOT / "data" / "specimen" / "conditions"
CACHE_DIR = ROOT / "results" / "sampling_fusion" / "cache"
OUT_DIR = ROOT / "results" / "sampling_fusion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["H_LF", "H_COARSE", "H_MIXED", "H_FINE", "P_LF", "P_COARSE", "P_MIXED", "P_FINE"]
WINDOWS = [3, 7, 15]
DELTAS = np.round(np.arange(1, 321) * 0.01, 2)

RAW_CSV = OUT_DIR / "sampling_fusion_raw.csv"
CKPT_JSON = OUT_DIR / "sampling_fusion_run_checkpoint.json"

FIELDNAMES = [
    "condition", "delta", "window", "K",
    "sampling_mse", "sampling_psnr", "sampling_ssim",
    "fusion_mse", "fusion_psnr", "fusion_ssim",
    "total_mse", "total_psnr", "total_ssim",
    "focus_index_agreement",
    "axial_error_kF_mean", "axial_error_kF_median", "axial_error_kF_p95",
    "oracle_axial_error_mean", "oracle_axial_error_median", "oracle_axial_error_p95",
    "N_eff", "R_Z", "adjacent_overlap_mean",
    "region_plateau_mse", "region_ramp_mse", "region_hard_step_mse", "region_protrusion_mse",
    "region_plateau_axial_error_mean", "region_ramp_axial_error_mean",
    "region_hard_step_axial_error_mean", "region_protrusion_axial_error_mean",
]


def _load_checkpoint():
    if CKPT_JSON.exists():
        return json.loads(CKPT_JSON.read_text())
    return {"completed_keys": []}


def _save_checkpoint(ckpt):
    CKPT_JSON.write_text(json.dumps(ckpt, indent=2))


def _row_key(condition, delta, window):
    return f"{condition}|{delta:.2f}|{window}"


def main(condition_subset=None, limit_deltas=None):
    h = np.load(SPECIMEN_DIR / "height_map.npy").astype(np.float64)
    shape = h.shape
    z_master = build_master_lattice(float(h.min()), float(h.max()))
    assert len(z_master) == N_MASTER == 801

    masks_npz = np.load(CACHE_DIR / "regional_masks.npz")
    region_masks = {
        "plateau": masks_npz["plateau"],
        "ramp": masks_npz["ramp"],
        "hard_step": masks_npz["hard_step"],
        "protrusion": masks_npz["protrusion"],
    }
    full_mask = np.ones(shape, dtype=bool)

    conditions = condition_subset if condition_subset else CONDITIONS
    deltas = DELTAS[:limit_deltas] if limit_deltas else DELTAS

    manifest = json.loads((CACHE_DIR / "master_render_manifest.json").read_text())
    completed_j = set(manifest["completed_j"])
    assert len(completed_j) == N_MASTER, (
        f"Phase A incomplete: {len(completed_j)}/{N_MASTER} master positions rendered. "
        "Run scripts/s3_master_render.py to completion first."
    )

    ckpt = _load_checkpoint()
    done_keys = set(ckpt["completed_keys"])

    # Pre-compute per-delta candidate index subsets and stack-oracle k_or
    # ONCE (condition/window independent) -- §9.
    delta_cache = {}
    for delta in deltas:
        idx = master_lattice_indices(float(delta))
        z_sub = z_master[idx]
        k_or = nearest_plane_index(h, z_sub)
        delta_cache[round(float(delta), 2)] = (idx, z_sub, k_or)

    write_header = not RAW_CSV.exists()
    with open(RAW_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for condition in conditions:
            oracle = np.load(CONDITIONS_DIR / f"oracle_{condition}.npy").astype(np.float64)
            frame_stack = np.memmap(
                CACHE_DIR / f"frames_{condition}.f64",
                dtype=np.float64, mode="r", shape=(N_MASTER,) + shape,
            )

            for window in WINDOWS:
                # Phase B: Laplacian^2 evidence, computed ONCE per
                # (condition, window) across all 801 master positions.
                remaining_deltas = [
                    d for d in deltas if _row_key(condition, d, window) not in done_keys
                ]
                if not remaining_deltas:
                    print(f"[skip] {condition} window={window}: all deltas already done")
                    continue

                t0 = time.time()
                E_stack = np.empty((N_MASTER,) + shape, dtype=np.float64)
                for j in range(N_MASTER):
                    E_stack[j] = laplacian_energy(np.asarray(frame_stack[j]), window)
                print(f"{condition} window={window}: evidence computed in {time.time()-t0:.1f}s "
                      f"({len(remaining_deltas)}/{len(deltas)} deltas remaining)", flush=True)

                for delta in remaining_deltas:
                    key = _row_key(condition, delta, window)
                    d_round = round(float(delta), 2)
                    idx, z_sub, k_or = delta_cache[d_round]
                    K = len(idx)

                    sub_frames = np.asarray(frame_stack[idx])   # (K,H,W)
                    sub_E = E_stack[idx]                        # (K,H,W)

                    k_F = np.argmax(sub_E, axis=0)
                    I_F = np.take_along_axis(sub_frames, k_F[None, :, :], axis=0)[0]
                    I_or = np.take_along_axis(sub_frames, k_or[None, :, :], axis=0)[0]

                    d_sampling = image_metrics(I_or, oracle, mask=None)
                    d_fusion = image_metrics(I_F, I_or, mask=None)
                    d_total = image_metrics(I_F, oracle, mask=None)

                    agr = focus_index_agreement(k_F, k_or, full_mask)
                    kF_err = focus_selection_metrics(k_F, h, z_sub, full_mask)
                    kor_err = focus_selection_metrics(k_or, h, z_sub, full_mask)
                    red = axial_redundancy(sub_E, full_mask)

                    region_row = {}
                    for rname, rmask in region_masks.items():
                        rm = image_metrics(I_F, oracle, mask=rmask)
                        rerr = focus_selection_metrics(k_F, h, z_sub, rmask)
                        region_row[f"region_{rname}_mse"] = rm["mse"]
                        region_row[f"region_{rname}_axial_error_mean"] = rerr["mean"]

                    row = {
                        "condition": condition, "delta": d_round, "window": window, "K": K,
                        "sampling_mse": d_sampling["mse"], "sampling_psnr": d_sampling["psnr"],
                        "sampling_ssim": d_sampling["ssim"],
                        "fusion_mse": d_fusion["mse"], "fusion_psnr": d_fusion["psnr"],
                        "fusion_ssim": d_fusion["ssim"],
                        "total_mse": d_total["mse"], "total_psnr": d_total["psnr"],
                        "total_ssim": d_total["ssim"],
                        "focus_index_agreement": agr,
                        "axial_error_kF_mean": kF_err["mean"], "axial_error_kF_median": kF_err["median"],
                        "axial_error_kF_p95": kF_err["p95"],
                        "oracle_axial_error_mean": kor_err["mean"], "oracle_axial_error_median": kor_err["median"],
                        "oracle_axial_error_p95": kor_err["p95"],
                        "N_eff": red["N_eff"], "R_Z": red["R_Z"],
                        "adjacent_overlap_mean": red["adjacent_overlap_mean"],
                        **region_row,
                    }
                    writer.writerow(row)
                    done_keys.add(key)

                f.flush()
                ckpt["completed_keys"] = sorted(done_keys)
                _save_checkpoint(ckpt)
                del E_stack

            del frame_stack

    print(f"Full run: {len(done_keys)} rows in {RAW_CSV}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--limit-deltas", type=int, default=None)
    args = ap.parse_args()
    main(condition_subset=args.conditions, limit_deltas=args.limit_deltas)
