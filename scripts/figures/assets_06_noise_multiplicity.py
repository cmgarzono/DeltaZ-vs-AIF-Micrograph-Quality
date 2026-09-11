r"""Section I -- multiplicity and noise, image space. Reconstructed EXACTLY
via the frozen S4 pipeline functions (deltaz_aif.analysis.multiplicity_noise),
using the documented frozen clean source, deterministic seed policy, and
noise-generation routine -- no new noise realization, replicate=0 (the
first/canonical replicate) used throughout for visual export."""
from __future__ import annotations

import numpy as np
from assets_common import OUT, CONDITIONS_DIR, save_gray, save_npy, manifest_row
from deltaz_aif.analysis.multiplicity_noise import (
    load_i_clean, condition_seed, generate_noisy_stack,
    rule_single, rule_mean, rule_maxfocus,
)

SEC = OUT / "09_noise_multiplicity"
SIGMAS = [0.01, 0.02, 0.04]
NS = [1, 8, 32, 128]
REPLICATE = 0


def main():
    i_clean = load_i_clean(str(CONDITIONS_DIR / "oracle_H_MIXED.npy"))

    p = SEC / "clean_reference.png"
    save_gray(i_clean, p)
    save_npy(i_clean, SEC / "clean_reference.npy")
    manifest_row(filename=str(p.relative_to(OUT)), role="I_clean, frozen H_MIXED oracle crop (400x400)",
                 source_frozen_file="data/specimen/conditions/oracle_H_MIXED.npy",
                 direct_or_derived="deterministic_visualization", roi_name="s4_crop",
                 roi_coords="[170:570,1190:1590]", image_type="I_clean",
                 display_normalization="vmin=0,vmax=1,gray", source_script="scripts/assets_06_noise_multiplicity.py")

    for sigma in SIGMAS:
        for N in NS:
            seed = condition_seed(sigma, N, REPLICATE)
            stack = generate_noisy_stack(i_clean, sigma, N, seed)
            tag = f"sigma{sigma:.2f}_N{N:03d}_rep{REPLICATE}"

            outputs = {
                "single": (rule_single(stack), None),
                "mean": (rule_mean(stack), None),
            }
            for w in (3, 7):
                out, kmap = rule_maxfocus(stack, w)
                outputs[f"maxfocus_w{w}"] = (out, kmap)

            for rule_name, (out, kmap) in outputs.items():
                p = SEC / f"{rule_name}_{tag}.png"
                save_gray(out, p)
                save_npy(out, SEC / f"{rule_name}_{tag}.npy")
                manifest_row(filename=str(p.relative_to(OUT)), role=f"S4 {rule_name} output, sigma={sigma}, N={N}",
                             source_frozen_file="results/multiplicity_noise/multiplicity_noise_raw.csv (seed policy); data/specimen/conditions/oracle_H_MIXED.npy",
                             direct_or_derived="deterministic_visualization", sigma=sigma, N=N,
                             replicate_or_seed=seed, image_type=f"s4_{rule_name}",
                             display_normalization="vmin=0,vmax=1,gray", source_script="scripts/assets_06_noise_multiplicity.py")

                err = np.abs(out - i_clean)
                pe = SEC / f"err_{rule_name}_{tag}.png"
                save_gray(err, pe, vmin=0.0, vmax=0.2, cmap="inferno")
                save_npy(err, SEC / f"err_{rule_name}_{tag}.npy")
                manifest_row(filename=str(pe.relative_to(OUT)), role=f"S4 |{rule_name}-clean| error, sigma={sigma}, N={N}",
                             source_frozen_file="data/specimen/conditions/oracle_H_MIXED.npy",
                             direct_or_derived="deterministic_visualization", sigma=sigma, N=N,
                             replicate_or_seed=seed, image_type=f"s4_err_{rule_name}",
                             display_normalization="vmin=0,vmax=0.2,inferno", source_script="scripts/assets_06_noise_multiplicity.py")

                if kmap is not None:
                    pk = SEC / f"selmap_{rule_name}_{tag}.png"
                    save_gray(kmap.astype(np.float64), pk, vmin=0, vmax=max(1, N - 1), cmap="viridis")
                    manifest_row(filename=str(pk.relative_to(OUT)), role=f"S4 {rule_name} selection map, sigma={sigma}, N={N}",
                                 source_frozen_file="data/specimen/conditions/oracle_H_MIXED.npy",
                                 direct_or_derived="deterministic_visualization", sigma=sigma, N=N,
                                 replicate_or_seed=seed, image_type=f"s4_selmap_{rule_name}",
                                 display_normalization=f"vmin=0,vmax={max(1,N-1)}", source_script="scripts/assets_06_noise_multiplicity.py")
        print(f"  sigma={sigma} done.")
    print("Section I (noise/multiplicity) done.")


if __name__ == "__main__":
    main()
