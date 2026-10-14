# Reference-only frozen profile experiment

This development ablation removes **only the direct nuclear-count contribution**
from the frozen physical-cell expression profile. It does not disable nuclear
anchors, replace the independent scRNA reference, set lambda to zero, or freeze q.

## Matched definition

The current baseline initializes

```
q0[c] = existing posterior from confident nuclear bins
p_ref[c] = q0[c] @ theta
p_current[c] = normalize(nuclear_counts[c] + m * p_ref[c])
lambda[c] = nuclear_UMI[c] / (nuclear_UMI[c] + m)
```

Here `m = kappa * median(positive nuclear UMI in the patch)` and kappa stays 2.
Reference-only changes the third line to

```
p_reference_only[c] = normalize(m * p_ref[c])
```

Both paths retain identical epsilon regularization. The reference-only mean is
therefore q0 @ theta up to that numerical smoothing. Actual nuclear counts remain
in q0, the measured nuclear-depth metadata and outer lambda. They do not directly
enter the new physical-cell profile. This is not a no-nuclear-information control.

The E-step remains

```
L = (1-lambda) * (q_current @ LL_bin) + lambda * LL_bin_under_frozen_profile
score = alpha * log(spatial_prior) + beta * expression_confidence * L
```

The profile stays frozen throughout EM, including the q0 used to construct it.
The separate q_current still updates in every complete M-step. This differs from
type-only: scoring log(q0 @ theta) is not the same as q_current @ log(theta).

The opt-in `run_generalized_em(..., use_reference_only_cell_profile=True)` hook
requires standard cell-specific expression, profile_update_damping=0 and
nuclear-depth reliability. It rejects conflicting score overrides. Its default
is false; existing inference and RL configuration defaults are unchanged.

## Frozen experiment

- Same A/B modeled-only observed datasets, four development patches, 43 core cells.
- All candidate core/margin physical cells within episode `max_center_distance_um=20`.
- Alpha=1, beta=1.5, kappa=2, profile_update_damping=0, background disabled.
- Same responsibility damping and stopping criteria; common safety cap of 50 iterations.
- Three arms: distance-only, current frozen EM, reference-only frozen EM.
- Nuclear anchors remain locked; export uses the existing unique-owner merger.
- No GT-profile inference, no weight search, no PPO, no new test regions in this run.

The predeclared advancement evidence requires better mean IoU than both baselines
in A and B, at least three nondecreasing patches per comparison, more corrected
than damaged core-GT nonnuclear bins versus frozen EM, convergence, and old baseline
reproduction. Passing is evidence to propose future validation, not authorization
to replace the production baseline or a claim of independent generalization.

Inference and evaluation are separate commands. The inference runner denies known
generating-profile paths and has no profile-directory input. The evaluation phase
may read GT profiles for the existing expression-correlation metric after checking
hash-sealed assignments. The guards supplement the call graph; they are not an OS
security sandbox.

## Commands

Choose a new output directory; inference refuses overwrites.

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
Bin2Cell_Validation/bin/python scripts/run_em_reference_only.py \
  --source-dir runs/benchmarks/library_size_pseudo_hd_20260916T165847Z \
  --prior-dir runs/benchmarks/em_posterior_predictive_20260916T220700Z \
  --output-dir runs/benchmarks/YOUR_NEW_REFERENCE_ONLY_RUN

Bin2Cell_Validation/bin/python scripts/evaluate_em_posterior_predictive.py \
  --run-dir runs/benchmarks/YOUR_NEW_REFERENCE_ONLY_RUN \
  --source-dir runs/benchmarks/library_size_pseudo_hd_20260916T165847Z \
  --profile-dir runs/benchmarks/scdesign3_library_size_ab_20260916T163318Z

Bin2Cell_Validation/bin/python scripts/summarize_em_posterior_predictive.py \
  --run-dir runs/benchmarks/YOUR_NEW_REFERENCE_ONLY_RUN \
  --source-dir runs/benchmarks/library_size_pseudo_hd_20260916T165847Z \
  --prior-dir runs/benchmarks/em_posterior_predictive_20260916T220700Z
```

The existing evaluator and summary keep their historical filenames, but now read
the actual arm names from the completed inference record; Reference-only outputs
are not labelled posterior-predictive. Old runs and their results are not modified.
The final audit command is `scripts/audit_em_posterior_predictive.py`, with the same
run/source/profile arguments as evaluation and an EM-focused test log under
`subdiagnostic/tests.log`.

## Interpretation limits

Improvement would show that removing direct nuclear counts helps **under these
fixed settings**, not that all nuclear information is harmful. Failure would not
prove that the scRNA reference is poor: q0 errors, type-average loss of cell-specific
variation, the scoring blend and nuclear/non-nuclear expression differences remain
possible explanations. Existing A/B data share tissue geometry and are development
evidence, not independent donor replicates. This ablation alone cannot distinguish
nuclear sampling noise from compartment-specific bias.
