# Patch-level generalized EM assignment

The optional EM stage runs after confident nuclear seeds are loaded and before
the existing patch-level global-delta RL environment is reset. It works on one
physical 2 um barcode at a time. UMI counts are never expanded into synthetic
reads.

The candidate radius comes from the resolved episode-build config:

```text
<episodes run>/config/config_resolved.yaml
  environment.max_center_distance_um
```

`reward.r_max_um` is not used for candidate inclusion. It remains the reward
normalization scale used by the existing RL code.

EM-ready episode artifacts must set
`inputs.expression.filter_empty_nuclear_cells: false` and
`environment.radius_band_um: null`. The first setting keeps physical nuclei
whose selected nuclear expression is zero, so their initial `q` stays diffuse.
The second prevents the legacy radius-band rule from removing pairs that pass
MaxDis. `PatchDataset` raises an error for older incomplete artifacts instead of
silently dropping nuclei or candidate pairs.

## Patch context

Every EM-scored bin inside `outer_bounds` considers every loaded physical
nucleus within `environment.max_center_distance_um`. Core and margin cells use
the same candidate rule. Confident nuclear bins for margin cells stay in the EM
problem even when their coordinates fall outside `outer_bounds`, because they
anchor that cell's type posterior.

At runtime, `PatchDataset` queries the nuclei index over `outer_bounds` expanded
by MaxDis. This repairs older patch indexes whose static margin was too narrow.
The patch-index builder now writes the expanded context bounds and does not
truncate margin nuclei to a nearest or fixed-count list. If `--max-patch-cells`
is set and a complete patch is too large, the builder skips that patch instead
of producing an incomplete candidate set.

## Alternating update

The sparse arrays `pair_bin_index` and `candidate_cell_index` identify valid
bin-cell pairs. One E-step evaluates every pair with the same old cell-type
posterior:

```text
L_pair = sum_k q[pair_cell, k] * LL[pair_bin, k]
score  = spatial_weight * log(spatial_prior)
       + expression_weight * confidence[pair_bin] * L_pair
r      = ragged_softmax(score, grouped by physical bin)
```

After the complete E-step, one M-step recomputes every cell score from the new
responsibility matrix:

```text
M[cell, k] = log_prior[cell, k]
           + scatter_sum(r[pair] * confidence[bin] * LL[bin, k])
q = softmax(M, over cell type k)
```

No cell posterior is updated between individual bins, and an M-step starts from
the prior rather than accumulating the previous iteration's score.

An optional cell-specific expression mode also maintains one selected-gene
profile for each physical cell. It starts from the locked nuclear counts and a
scRNA prior given by `q @ reference_theta`. Its prior count is
`relative_prior_strength` times the patch median positive locked-nuclear UMI
total. The cell reliability is the nuclear UMI total divided by that total plus
the prior count. This makes the shrinkage relative to observed depth: scaling
all counts by the same factor does not change the normalized profile or its
reliability.

`reliability_mode: nuclear_heldout_predictive` is an experimental alternative.
For each physical cell, it holds out every positive-count locked nuclear bin
in turn, rebuilds both `q` and the nuclear profile without that bin, and selects
`lambda` on a configurable grid by held-out multinomial prediction. A cell with
fewer than two positive nuclear bins receives `lambda=0`. Non-nuclear bins do
not enter the fit. The result stores the selected weight, held-out gain per UMI,
and number of usable nuclear bins for each patch-cell instance.

The September 2026 current/capture-thinned validation did not support replacing
the default relative-depth rule with this estimator. The retained report is at
`runs/subdiagnostic/em_nuclear_heldout_reliability_summary_20260902T220000Z/`.

With this mode enabled, the E-step compatibility is:

```text
L_type = q[cell] @ LL[bin]
L_cell = normalized multinomial LL(bin counts | physical-cell profile)
L_pair = (1 - reliability[cell]) * L_type
       + reliability[cell] * L_cell
```

After the full responsibility update, the M-step first updates every `q`, then
updates every physical-cell profile from the same complete responsibility
matrix. The observed counts are shrunk toward the newly resolved `q @ theta`
prior. This is still a generalized EM-style alternating assignment algorithm,
not a claim of exact finite-mixture maximum likelihood.

Nuclear rows are reset to exact one-hot values after the raw E-step, after
damping, and before the M-step. They are also excluded from RL replacement.

## RL handoff

The full soft result remains attached to `PatchContext.em_assignment`. The
Torch patch environment hardens each row with top-1 ownership during `reset()`,
rebuilds the existing membership, owner, posterior, shape, and neighborhood
state, and then runs the unchanged global-delta objective.

Normalized entropy is the only ambiguity gate:

```text
H_norm = -sum(r * log(r)) / log(number of candidate cells)
is_ambiguous = H_norm > entropy_threshold
```

A one-candidate bin has entropy zero. Top-1 probability and the top-1/top-2
margin are stored for diagnostics but do not affect `is_ambiguous`. With the
default `refine_ambiguous_only: true`, only high-entropy, non-nuclear bins can
appear as REPLACE actions. EM probabilities are not added to the reward.

## Configuration

```yaml
em_assignment:
  enabled: true
  use_existing_rl_candidate_max_distance: true

  # Production-compatible default. `positive_expression` is currently a
  # diagnostic universe because real intracellular 2 um bins can have 0 UMI.
  bin_universe:
    non_nuclear_filter: all

  # Experimental v1 gate. Keep disabled unless assignment coverage is reported;
  # expression confidence changes with capture depth.
  background:
    enabled: false
    owned_logit_intercept: -6.0
    expression_confidence_weight: 12.0

  spatial_weight: 1.0
  expression_weight: 0.25
  damping: 0.5
  max_iterations: 15
  convergence:
    top_owner_change_fraction: 0.01
    mean_probability_change: 0.001
    mean_profile_total_variation: 0.001
  cell_specific_expression:
    # Disabled by default so existing runs and checkpoints keep their behavior.
    enabled: false
    # Dimensionless multiplier of patch median locked-nuclear UMI depth.
    relative_prior_strength: 1.0
    # Independent damping for the physical-cell profile M-step. Zero freezes
    # the nuclear-initialized profile; one applies the full batch M-step.
    profile_update_damping: 1.0
    # standard uses the complete previous profile. leave_one_out removes the
    # scored bin's own previous responsibility-weighted counts.
    # spatial_block_crossfit removes the complete non-nuclear spatial block.
    compatibility_mode: standard
    spatial_crossfit_block_size_um: 8.0
    # nuclear_depth preserves the original relative-depth reliability rule.
    # nuclear_heldout_predictive uses only locked nuclear bins and
    # leave-one-nuclear-bin-out predictive validation.
    reliability_mode: nuclear_depth
    heldout_reliability_grid_size: 101
  ambiguity:
    entropy_threshold: 0.50
  rl_integration:
    initialize_from_em: true
    lock_nuclear_bins: true
    refine_ambiguous_only: true
  artifacts:
    save: false
    debug_patch_ids: []
```

`expression_weight: 0` is the distance-only soft-assignment baseline. The
listed weights and entropy threshold are starting values, not calibrated
biological parameters.

`bin_universe.non_nuclear_filter: positive_expression` keeps every locked
nuclear seed but excludes non-nuclear bins whose selected-gene expression
confidence is zero. This option is useful for diagnosing forced expansion; it
is not yet a real-data background model. When `background.enabled: true`, the
solver keeps the full bin universe and reserves a fixed background mass from a
confidence-based logistic gate. Physical-cell responsibilities plus background
sum to one. This first gate is intentionally optional because capture thinning
changes its rejection rate.

The EM-only weight-sweep runner supports `--alternating-cell-profile` and a
`--cell-profile-relative-prior-strengths` grid for the physical-cell model. It
also supports `--profile-update-dampings`. This profile damping is independent
of responsibility damping: zero keeps the nuclear-initialized cell profile
fixed, while one reproduces the original full profile M-step. Positive values
mainly change the rate at which the solver approaches an updated-profile fixed
point, so comparisons must report convergence and the iteration limit.
`cell_specific_expression.compatibility_mode: leave_one_out` requires
`profile_update_damping: 1`. For candidate pair `(b, c)`, it computes
the cell-specific likelihood from the previous profile numerator after
subtracting `r[b, c] * x[b]`. The first E-step subtracts nothing from
non-nuclear bins because only locked nuclear counts initialized the profile.
Later iterations use the complete previous responsibility matrix. Computation
stays sparse over valid bin-cell pairs and CSR nonzero genes.
`compatibility_mode: spatial_block_crossfit` has the same damping requirement.
It maps non-nuclear bins to square blocks using absolute physical coordinates.
Each block is a held-out fold: scoring a bin removes every previous-iteration
`r[j, c] * x[j]` contribution from non-nuclear bins in that block. Nuclear bins
use fold `-1`, so their counts remain in every physical-cell profile. The
current implementation uses fixed square blocks; bins on opposite sides of a
block boundary can still influence one another.
It also retains two ceiling diagnostics: `--oracle-q-fixed` and
`--cell-specific-nuclear-profile`. The former uses GT
cell type and must never be used for inference. The latter builds one fixed
physical-cell profile from locked nuclear counts, shrunk toward the existing
scRNA reference theta. It tests whether cell-specific expression is the missing
signal, but it does not update that profile between iterations.

When artifact saving is enabled, each patch writes a compact NPZ containing the
ragged candidate rows, all responsibilities, top-owner diagnostics, nuclear and
ambiguity masks, and cell-type posteriors. Selected debug patches can also write
a long CSV with distance, spatial prior, expression compatibility, combined
score, and responsibility for each valid pair.

`evaluate_em_assignments` keeps pseudo ground truth outside inference and reports
top-1 accuracy, accuracy by entropy bin, per-cell IoU, optional boundary metrics,
and EM-to-RL correction/damage rates. Pass the rollout's actual REPLACE count as
`rl_replace_action_count`; final owner changes are reported separately because a
bin can be replaced more than once during a rollout.

Set `em_assignment.enabled: false` to keep the previous nuclear-only patch reset
and action behavior.
