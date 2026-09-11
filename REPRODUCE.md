# Reproducing the results

Exact, ordered commands to regenerate every number, table, and figure
reported in the paper from this repository alone. Run from the repository
root with the package on the path:

```bash
pip install -r requirements.txt
export PYTHONPATH=src   # PowerShell: $env:PYTHONPATH = "src"
```

Two arrays are bundled as frozen inputs under `data/specimen/` because an
outside user cannot regenerate them bit-identically without the original
source material. `Specimen()` loads them automatically — no environment
variable or external archive needed for anything in this file:
- `data/specimen/height_map.npy` — the physical specimen height map.
- `data/specimen/conditions/oracle_*.npy` — the 8 ideal all-in-focus
  photographic references.

Everything else — every CSV and figure in the computational pipeline — is
recomputed from these two arrays plus the code in `src/`. Regenerated
intermediate output not part of the tracked release (quick plots, the
manuscript-figure asset bank, the render cache) is written to
`outputs/` and `results/sampling_fusion/cache/`, both gitignored.

## 1. Specimen and regional masks (seconds)

```bash
python scripts/build_regional_masks.py
```
Rebuilds `results/sampling_fusion/cache/regional_masks.npz` from the
bundled specimen data.

## 2. Axial-coverage model (seconds)

```bash
python scripts/analytical_axial_coverage.py
```
Writes `results/axial_coverage/*.csv` and `outputs/axial_coverage/*.png`.

## 3. Spectral-evidence model (seconds)

```bash
python scripts/build_spectral_inputs.py       # freezes the 8 physical spectral inputs
python scripts/analytical_spectral_evidence.py  # evidence/sensitivity model + figures
```
Writes `results/spectral_evidence/*.csv` and the analytical figure data
used by Fig. 2.

## 4. PSF calibration (seconds)

```bash
python scripts/calibrate_psf.py
```
Writes `results/calibration/psf_calibration_table.csv`.

## 5. Photographic sampling/fusion experiment (LONG — the main simulation)

This is the expensive part of the study: 8 photographic conditions x 801
master axial positions x full 1080x1920 rendering, then 320 axial
spacings x 3 focus windows fused and scored (7680 outcomes). **Check
`--workers` before running on a laptop; expect several hours on a
multi-core workstation, and the render cache alone is on the order of
100 GB on disk** (gitignored — regenerated locally, disposable, not
needed once the raw results table exists).

```bash
# Render all 801 master axial positions per condition, cached to
# results/sampling_fusion/cache/*.f64 (LONG: hours; safe to resume --
# re-run the same command and it skips already-completed positions)
python scripts/render_master_stacks.py --workers 8

# Fuse + score all 320 deltas x 3 windows from the render cache
# (minutes once the render above is done)
python scripts/run_sampling_fusion_experiment.py

# Derived summary tables and Fig. 10
python scripts/summarize_sampling_fusion.py
python scripts/sampling_fusion_diagnostic_figures.py
```

Outputs land in `results/sampling_fusion/*.csv` (already
included in this repository as frozen reference results —
`sampling_fusion_raw.csv` should match bit-for-bit under the
deterministic renderer) and `outputs/sampling_fusion/*.png`.

## 6. Multiplicity/noise experiment (minutes)

```bash
python scripts/run_multiplicity_noise_experiment.py   # 16,000 rows
python scripts/summarize_multiplicity_noise.py
```
Outputs land in `results/multiplicity_noise/*.csv`.

## 7. Manuscript figures

Script names match the manuscript's own figure numbers. Fig. 4, Fig. 6,
Fig. 8, and Fig. 11 draw on an intermediate asset bank derived from the
frozen inputs and the results above, written to
`outputs/manuscript_assets/`:

```bash
python scripts/figures/assets_00_specimen.py
python scripts/figures/assets_01_conditions.py
python scripts/figures/assets_02_through_focus.py
python scripts/figures/assets_03_densification.py
python scripts/figures/assets_04_redundancy.py
python scripts/figures/assets_05_geometry_rois.py
python scripts/figures/assets_06_noise_multiplicity.py
```

Then the manuscript figures (matching `figures/`):

```bash
python scripts/figures/fig2.py
python scripts/figures/fig3.py
python scripts/figures/fig4.py
python scripts/figures/fig5.py
python scripts/figures/fig6.py
python scripts/figures/fig7.py
python scripts/figures/fig8.py
python scripts/figures/fig9.py
python scripts/figures/fig11.py
python scripts/figures/fig12.py
python scripts/figures/fig13.py
python scripts/figures/fig14.py
```

Fig. 10 is produced in step 5 above.

Fig. 1 (the documentary corpus / influence-dependence structure) is not
produced by this repository — it belongs to the documentary-analysis
pillar of the study, maintained separately from the computational
experiments released here, and is supplied as a finalized asset
(`figures/Figure1.png`). However, an interactive Corpus Dashboard
summarizing the documentary-analysis results (79 references, 14 variables,
91 relations) is included in this repository and can be accessed via
the [📊 Corpus Dashboard](Corpus%20Dashboard/README.md) link.
