# DeltaZ_AIF

**Axial Sampling Densification and All-in-Focus Micrograph Quality: The Gap Between Focal Coverage and Metric Response**

Carlos M. Garzón — Departamento de Física, Universidad Nacional de Colombia, Bogotá, Colombia
Contact: cmgarzono@unal.edu.co

This repository is the computational (in-silico) release for the study
*Axial Sampling Densification and All-in-Focus Micrograph Quality: The Gap
Between Focal Coverage and Metric Response*. It contains the analytical
axial-coverage/spectral-evidence model, the controlled photographic
sampling/fusion experiment, and the multiplicity/noise experiment, together
with the source code, frozen inputs, tabulated results, manuscript figures,
and scripts needed to reproduce the computational results reported in the
manuscript.

The documentary/corpus analysis (Fig. 1, Table 1) that motivated the
phenomenological framework is a separate pillar of the study. While the full
corpus analysis is a separate work, an interactive Corpus Dashboard summarizing
the key findings is available here:

**[📊 Open Corpus Dashboard](https://cmgarzono.github.io/DeltaZ-vs-AIF-Micrograph-Quality/Corpus%20Dashboard/Corpus_Dashboard.html)**

## Layout

```
DeltaZ_AIF/
├── src/deltaz_aif/       # Core library
│   ├── specimen/         # Physical specimen geometry
│   ├── photographic/     # Image-formation & photographic conditions
│   ├── focus/             # Point-spread-function model
│   ├── rendering/         # Through-focus depth-layer renderer
│   └── analysis/          # The four experiments (see table below)
├── scripts/                # Reproduction pipeline -- see REPRODUCE.md
│   └── figures/             # Manuscript figure generators
├── data/specimen/           # Frozen essential inputs (height map,
│                            #  8 photographic-condition references)
├── results/                 # Tabulated outputs, one directory per experiment
├── figures/                 # Manuscript-ready figures (Figure1.png .. Figure14.png)
└── REPRODUCE.md             # Exact, ordered commands to regenerate everything
```

## Quick start

Clone the repository and run from inside it — this is a reproducibility
package, not a standalone PyPI library: `src/` can be `pip install`-ed on
its own (e.g. to use `deltaz_aif` as a dependency elsewhere), but the
~230 MB of frozen scientific data under `data/` and the tabulated
`results/` ship only with the git repository, not with the wheel/sdist.

```bash
pip install -r requirements.txt
export PYTHONPATH=src   # PowerShell: $env:PYTHONPATH = "src"
```

To regenerate every result and figure from scratch, follow
**[REPRODUCE.md](REPRODUCE.md)**. It distinguishes the fast steps
(seconds-minutes) from the one genuinely expensive step — the photographic
through-focus render (hours, ~100 GB of disposable disk cache, not stored
in this repository).

## The four experiments

| Experiment | What it does | Code | Results |
|---|---|---|---|
| Axial coverage | Analytical axial-coverage / marginal-sensitivity model | `src/deltaz_aif/analysis/axial_coverage.py` | `results/axial_coverage/` |
| Spectral evidence | Spectral-content-weighted evidence model | `src/deltaz_aif/analysis/spectral_evidence.py` | `results/spectral_evidence/` |
| Sampling/fusion | Photographic sampling, oracle, and fusion experiment (7680 outcomes) | `src/deltaz_aif/analysis/sampling_fusion.py` | `results/sampling_fusion/` |
| Multiplicity/noise | Combination-rule experiment under repeated noisy observation (16000 outcomes) | `src/deltaz_aif/analysis/multiplicity_noise.py` | `results/multiplicity_noise/` |

## Frozen inputs

Two arrays are bundled directly under `data/specimen/` because they define
the fixed ground truth every experiment is evaluated against:

- `data/specimen/height_map.npy` — the physical specimen (1920x1080 px,
  -4 to +4 DOF, 27 pyramidal protrusions).
- `data/specimen/conditions/oracle_*.npy` — the 8 ideal all-in-focus
  photographic references (H/P texture families x LF/COARSE/MIXED/FINE
  spectral content).

See `REPRODUCE.md` for how every other file in this repository is
produced from these two arrays.

## Requirements

- Python 3.10+
- numpy, scipy, matplotlib, pandas, scikit-image, Pillow (see
  `requirements.txt`)

## Citing

If you use this code, please cite the manuscript (citation details to be
added on publication).

## License

MIT — see `LICENSE`.
