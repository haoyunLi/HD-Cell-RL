# Regional depth and spatially blocked calibration

This is a pseudo-generator experiment, not EM/PPO training. Existing weights,
shape priors, datasets, runs and untouched test regions remain unchanged.

## Frozen experiment plan

Use the existing four separated colorectal contexts: 10,827 unique 2-micrometer
bins and the same 18,132-gene panel. Retain the previous 2+2 calibration/audit
split as a regression view, then rotate four leave-one-patch-out folds. These
are development checks on one section, not independent tissue replicates.

The declared models are uniform mean, geometry-conditional mean,
geometry/type-conditional mean, plus a spatial residual field, and plus a
patch-level residual draw. Fit on complete 16-micrometer calibration blocks.
Standardization, Poisson regression, residual variance, spatial bandwidth and
source-library budgets must use calibration observations only. No held-out
real gene vector or depth map may be replayed or interpolated into a simulation.

Geometry features are modeled mask coverage, nuclear coverage and smoothed
nucleus density. Type density uses the existing simulation annotations. Those
labels may originate in expression-based annotation, so the composition model
is not independent morphology-only prediction. Masks themselves are synthetic
geometry, not real-cell GT. Types absent from a calibration fold are reported.

Observed counts do not identify technical capture efficiency. Conditional mean
and residual spatial variability are separated for diagnosis, without claiming
that either component has a proven biological or technical origin.

## Generation and verification

Reuse the existing nuclear/cytoplasm/membrane localization and separately traced
unmodeled biological sources. Estimate base source budgets from calibration
regions, then apply a coarse model-derived depth multiplier through an explicit
latent-count reservoir and binomial capture. The same bin's capture probability
applies to modeled and unmodeled RNA. Technical background remains zero for this
experiment. Captured source-bin counts must sum exactly to observed matrices.

Compare every held-out patch separately at 2/8/16 micrometers: counts and detected
genes, nonzero rate, spatial depth correlation, and binary/fractional cell-region
counts and variability. Cell-region masks can share mixed bins; they are not
true real-cell boundaries. Keep acceptance tolerances from the previous audit.
No tool ranking or GT segmentation score selects a model or changes a threshold.

The initial 2+2 split and the leave-one-out folds reuse observations and must not
be pooled as independent validation samples. Three random seeds measure
simulation variability, not three independent tissues. A smooth stochastic
field may improve a distribution without predicting the real map; evaluate and
label predictive means and generative distribution checks separately.

## Report plan

The final technical HTML report uses the build-report shared renderer, with a
technical summary, definitions, per-patch results, model/split specification,
robustness checks, limitations and next steps. The four-patch mean comparison
uses grouped bars; scale-dependent accuracy and failed checks use exact tables.
All figures/tables carry source metadata. A companion notebook exposes the
saved fitting and diagnostic tables. No report or dataset is published online.

## Completed development experiment: 2026-09-09

Outputs are in `runs/benchmarks/regional_depth_blocked_20260909/experiment/`:

- `report.html`: portable technical report; shared renderer validation and
  structural checks passed. Browser visual/interaction QA was unavailable
  because no installed Chromium headless shell was found. No browser downloaded.
- `per_patch_summary.csv`: per-split, per-patch, per-setting seed means.
- `multiscale_metrics.csv` and `cell_region_metrics.csv`: individual seed results.
- `acceptance_table.csv`: all 90 patch-setting-seed decisions, including rejected
  extrapolation attempts rather than silently omitting them.
- `analysis.ipynb`: validated and executed top-to-bottom; five code cells.
- `subdiagnostic/feature_support.csv`: calibration and audit feature ranges.
- `realizations/`: first seed only, retaining 19 successfully generated settings
  with component matrices, source-bin traces and multiplier fields.

There were 75 generation attempts: 57 completed and 18 rejected by the declared
64x computational-reservoir guard. Across the two split views, these correspond
to 63 generated patch evaluations and 27 rejected patch evaluations. None passed
all existing acceptance thresholds. These 90 entries are not independent samples.

The composition model predicted 15.60x and 18.41x actual mean depth for the
Stem-rich patch in the original and leave-one-out views. In the latter, 96% of
complete audit blocks exceeded the calibration range of Stem density. This is
evidence of log-link covariate extrapolation, not identified capture efficiency.

In leave-out patch_0078182, composition_mean had a seed-mean 2-um counts ratio of
1.1679 but a binary core-cell-region counts ratio of 1.5275. Spatial residuals
improved its 16-um neighbor correlation (0.8604 to 0.7853, real 0.7603) while its
cell-region counts ratio worsened to 1.5767. A local improvement is not a joint
realism pass or grounds to promote the generator.

CPU regression: 224 tests passed (58.494 seconds on the final recorded run),
including 11 regional-depth tests. Checks cover calibration isolation, source
budget/generation independence from audit counts, row-order independence,
spatial field separation, default capture backward compatibility, count
conservation and rejected extrapolation. Training/publication tests use local
temporary fixtures only; no formal training or remote publication occurred.

## Reproduction

Run the scientific Python environment, with bounded BLAS thread counts:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
Bin2Cell_Validation/bin/python -B scripts/run_regional_depth_calibration.py \
  --audit-dir runs/benchmarks/pseudo_realism_20260908T012500Z \
  --benchmark-dir runs/benchmarks/em_donor_disjoint_unseen_patches_20260903T192649Z \
  --validation-repo /projects/illinois/vetmed/cb/kwang222/Haoyun_Li/Bin2Cell_Validation \
  --output-dir runs/benchmarks/NEW_UNIQUE_RUN/experiment
```

The output directory must not exist. Fit/source snapshots are saved before
generation. Render the reviewed report input with the installed build-report
`deliver_portable_artifact.mjs`; the reporter itself does not invent an HTML
runtime. The final `delivery/` snapshot preserves later report and test code
separately from the source bytes used to execute the experiment.

Next: expand calibration support and test a shrinkage-controlled nonnegative
additive source-intensity model, then diagnose gene diversity conditional on
counts. Do not tune EM/RL weights or promote these failed generator settings.
