# Project corrections and generator acceptance

This work preserves historic datasets, training runs and evolution exports. No
PPO training or reward redesign is part of this experiment. Shape priors are
unchanged.

## Engineering contracts

- All patch export entry points use `PatchOwnershipMerger`. EM and distance use
  zero source scores and stable patch IDs; RL retains its source-patch-score
  rule. Conflicts are saved, not silently overwritten. Nuclear conflicts fail.
- The named EM section is `configs/em_assignment.frozen_baseline.yaml`. It keeps
  the retained beta=1.5, kappa=2, profile damping=0, positive-expression universe
  and 15-iteration cap. Current convergence checks still apply; lack of
  convergence must be reported, not hidden by the name “frozen”.
- Experiment manifests archive source bytes, including untracked project code,
  and record configuration, input fingerprints and the evaluation protocol.
  Large-input fingerprints explicitly use sampled bytes, not a full-file hash.
- Paired intervals resample whole patches. Previously inspected areas are
  development/audit evidence, not new held-out tests.
- Opt-in validation selects the post-update policy by merged core-cell IoU.
  Legacy development runs keep the old selection rule. `run.purpose: formal`
  requires validation and matching data-realism/objective admission evidence.

## Generator experiment, not a calibrated biological truth model

The observed matrix is the integer sum of three separately stored components:
modeled-cell RNA, unmodeled biological RNA, and technical background. Unmodeled
sources have stable source IDs but are not silently added as GT physical cells.
Their profiles come from generator-side scRNA, not inference-reference donors.
Their spatial kernels may reach occupied as well as empty bins.

Real counts are used to fit population-level budgets on two calibration
contexts. No per-bin real expression vector is copied into the synthetic data.
The other two already-inspected contexts are an audit split. A mask complement
is not a proven ambient compartment, and its signal does not identify a unique
biological source. The new source model is therefore a stated sensitivity model.

Compare the same gene panel and spatial regions at 2, 8 and 16 micrometers, plus
counts, detected genes and variability per cell-shaped region. Region sums are
not true real-cell profiles; mixed bins may contribute to multiple binary
regions. Fractional-weighted region summaries and traced synthetic source
counts are separate quantities. Do not judge success by making tool scores
lower. Do not run new held-out method comparisons or PPO until acceptance passes.

## Current execution

Results are in `runs/benchmarks/project_remediation_20260909/`:

- `em_context_regression_final/`: four fixed-parameter patch regressions, sixteen
  inset/halo comparisons, normalization-only controls and merged assignments.
  All original patches converge in 7–14 iterations. The 15-iteration cap is
  retained; some reduced-context controls do not converge and remain diagnostics.
- `cluster_statistics/`: reanalysis of saved five-method results with paired
  patch bootstrap. This does not rerun or tune any competitor.
- `source_separated_pseudo/`: seven generator controls × three seeds, compact
  source-separated matrices for the first seed, multiscale metrics and audit
  failures. No configuration passes the full predeclared acceptance checks.
- `delivery/`: final source snapshot plus the two external localization source
  files and input fingerprints. Earlier snapshots preserve the executed states;
  the delivery snapshot includes the final archival safeguard too.

The complete CPU test suite passed: **213 tests**. CPU policy updates occur only
in temporary unit-test fixtures; no formal model was trained, no existing run was
overwritten, and no GitHub publication was performed.

The frozen-code regression changed 17/43 cell exports; the sum of their barcode
set symmetric differences is 22 (a transfer may be counted in two cells).
Mean old/new per-cell assignment IoU is 0.98915. This is agreement between code
versions, not accuracy against GT. All 2,279 baseline output barcodes are unique.

Changing context altered the prior count even when a common cell's nuclear
counts stayed fixed. The normalization-only control changes up to 0.953% of
common-bin owners in an individual condition. The production prior is unchanged:
this quantifies a limitation rather than choosing a new formula from audit GT.

For one declared example (`components_sigma8`, seed 20260909), calibration mean
depth matches real HD (28.337 versus 28.353 counts/bin), but audit depth is low
(29.006 versus 41.035). On the 26 audit core-cell-shaped regions, median counts
are 577.5 versus 1497, and median detected genes 472.5 versus 1117.5. The original
high-depth pseudo has medians 6182 and 2573 on those same regions. Matching the
calibration mean is therefore not enough, and reducing excessive counts is not
equivalent to recovering the correct held-out distribution.

The support-only coverage diagnostic falls from roughly 0.999 to 0.510 in this
control. That is evidence the empty-mask shortcut changed, not evidence the
generator is realistic. Unknown-source mass, tissue support and coarse capture
heterogeneity remain uncertain. Do not replace production pseudo data, consume
fresh test regions or launch PPO from these controls.

Next data work should model between-region capture heterogeneity and assess it
with spatially blocked calibration checks. Keep counts, richness, spatial
correlation and cell-region variation as joint targets. Do not increase counts
everywhere or copy real per-bin expression merely to pass this audit.

## Running the checks

`scripts/audit_em_patch_sensitivity.py` consumes the named EM baseline directly.
Supply an existing benchmark directory and a new output directory. The runner
preserves the core, changes the available halo and inset grid, then repeats the
solve with the baseline prior count held fixed. That last control distinguishes
normalization effects from new context evidence; it is not a proposed production
prior formula. The old four-patch benchmark is regression evidence only.

`scripts/build_source_separated_pseudo.py` takes `--audit-dir`, `--benchmark-dir`,
`--validation-repo`, `--output-dir` and an optional `--config`. The default
`configs/pseudo_components.calibration.yaml` declares seven controls, three
seeds and acceptance tolerances before execution. Only the first seed's full
source-separated matrices are retained; all seeds retain numerical diagnostics.
The old mask-complement replay builder remains available as a legacy stress
test, not the new biological-source generator.

`scripts/summarize_donor_disjoint_em_benchmark.py` now creates a new timestamped
directory unless an explicit new directory is given. It does not overwrite an
old comparison report. Mean method scores are unchanged by re-estimating the CI.

## Before formal PPO

Merge `configs/ppo_validation.section.yaml` into a newly named patch config only
after data and objective validation. Its placeholder paths deliberately prevent
accidental use. It does not turn an old overfit run into independent validation.

The admission JSON must contain `training_evidence_identity` (from
`hd_cell_rl.patch_validation.training_evidence_identity` on the resolved config
and patch settings), `data_realism_passed: true`, the `data_realism_report` path
and its full SHA256 as `data_realism_sha256`, plus `objective_per_patch_csv` and
`objective_per_patch_sha256`. The objective CSV requires `patch_id`, `initial_iou`
and `final_iou`, exactly matching the declared validation patches. At least two
patches, positive mean/median change and a majority of positive changes are
required. These are minimum engineering gates, not proof of biological validity
or statistical significance. Do not fabricate admission evidence from a test
set or treat a sensitivity-grid report as independently accepted data.

Validation is greedy, deterministic and separate from policy updates. It runs
after an update, selects merged core-cell IoU, preserves random states/model mode
and saves the exact evaluated weights. When disabled, legacy development runs
keep their training-reward selection behavior. This opt-in distinction preserves
historical reproducibility; it is not permission to bypass gates for new formal
training.
