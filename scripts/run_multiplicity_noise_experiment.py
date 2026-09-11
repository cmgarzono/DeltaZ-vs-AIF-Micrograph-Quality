r"""
S4 full run: 16,000 unique experimental rows.

For each of 5 sigma x 8 N x 100 replicates = 4,000 (sigma,N,replicate)
combinations, generate ONE noisy stack and apply ALL rules to it
(single, mean, max-focus w=3, max-focus w=7) -> 4 rows per combination
-> 4,000 x 4 = 16,000 rows total.

Writes results/multiplicity_noise/multiplicity_noise_raw.csv.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from deltaz_aif.analysis.multiplicity_noise import (  # noqa: E402
    N_LEVELS, N_REPLICATES, SIGMA_LEVELS, load_i_clean, run_condition,
)

ORACLE_PATH = ROOT / "data" / "specimen" / "conditions" / "oracle_H_MIXED.npy"
OUT_PATH = ROOT / "results" / "multiplicity_noise" / "multiplicity_noise_raw.csv"

FIELDNAMES = [
    "sigma", "N", "rule", "window", "replicate", "seed",
    "MSE", "PSNR", "laplacian_variance", "dct_hf_energy",
    "selection_entropy", "selection_entropy_normalized", "clipping_fraction",
]


def main():
    i_clean = load_i_clean(str(ORACLE_PATH))
    total_combos = len(SIGMA_LEVELS) * len(N_LEVELS) * N_REPLICATES
    t0 = time.time()
    n_done = 0

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for sigma in SIGMA_LEVELS:
            for N in N_LEVELS:
                for rep in range(N_REPLICATES):
                    rows = run_condition(i_clean, sigma, N, rep)
                    for row in rows:
                        writer.writerow(row.as_dict())
                    n_done += 1
                if n_done % 100 == 0 or n_done == total_combos:
                    elapsed = time.time() - t0
                    rate = n_done / elapsed if elapsed > 0 else 0.0
                    eta = (total_combos - n_done) / rate if rate > 0 else float("nan")
                    print(
                        f"[{n_done}/{total_combos}] sigma={sigma} N={N} "
                        f"elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta:.1f}s",
                        flush=True,
                    )

    elapsed = time.time() - t0
    print(f"DONE. {n_done} combinations, {n_done * 4} rows written to {OUT_PATH}. "
          f"Total time {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
