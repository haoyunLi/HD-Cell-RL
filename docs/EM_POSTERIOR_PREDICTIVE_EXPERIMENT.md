# Nuclear posterior-predictive compatibility experiment

This is an opt-in EM-only experiment, not a replacement for the current baseline.
The development experiment did **not** pass its advancement gate; do not promote
the scorer or tune it on new test patches on the strength of this result.

## Model and integration

`hd_cell_rl/em_posterior_predictive.py` obtains the same initial nuclear profile,
initial q, relative scRNA prior and reliability as the existing initializer.
The expression reference is the independent donor-disjoint scRNA reference.
It does not load generating profiles, GT owners, or any files.

Before existing epsilon regularization, define

```
a[c,g] = nuclear_counts[c,g] + m * (q_initial[c] @ theta)[g]
m = kappa * median(positive observed nuclear UMI in the patch)
A[c] = sum_g a[c,g]
```

The implementation retains the existing initializer/scorer's epsilon
regularization so the posterior mean matches the current plug-in scorer.
For a bin with N selected-gene UMIs, the alternative cell compatibility is

```
L_cell[b,c] = (
    sum_g [logGamma(a[c,g] + x[b,g]) - logGamma(a[c,g])]
    - [logGamma(A[c] + N[b]) - logGamma(A[c])]
) / N[b]
```

For N=0 the score is zero. For N=1 it equals the plug-in log posterior mean.
This integrates a whole count vector under a Dirichlet posterior; no synthetic
reads are created. Sparse CSR operations are grouped by candidate physical cell.
Small integer counts use stable log rising products, avoiding subtraction of
large nearly equal gamma values.

This is the Dirichlet-multinomial predictive log probability **without** the
bin-only multinomial coefficient, divided by N. Both current type and cell LL
omit that coefficient. Adding it only to the cell term would create a spurious
cell-dependent contribution after multiplication by varying lambda. Distribution
semantics and the independent numerical test use the
[SciPy Dirichlet-multinomial documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet_multinomial.html).

The existing solver gets a new optional keyword,
`fixed_cell_expression_compatibility`, which replaces only L_cell:

```
L_pair = (1 - lambda[cell]) * (q[cell] @ LL[bin]) + lambda[cell] * L_cell_pair
S_pair = alpha * log(spatial_prior) + beta * expression_confidence[bin] * L_pair
```

The E/M alternation for r and q, damping, convergence, nuclear locks, all-MaxDis
candidate set, discrete export and ownership merger are unchanged. The hook
requires frozen standard cell profiles; it rejects simultaneous cell-profile
updates and conflicting full-pair score overrides. Without the keyword,
existing behavior is unchanged. No RL reward or policy feature is changed.

This remains a generalized EM scoring framework. The auxiliary Dirichlet model
conditions on estimated q_initial and an empirical relative prior; it is not a
complete biological uncertainty model or exact finite-mixture maximum likelihood.
In particular, uncertainty around a nuclear profile does not model a systematic
nuclear-to-cytoplasmic expression shift.

## Reproduce the development experiment

Use the project's scientific Python environment, with bounded BLAS threads.
Choose a **new**, absent output directory; inference refuses overwrites.

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
Bin2Cell_Validation/bin/python scripts/run_em_posterior_predictive.py \
  --source-dir runs/benchmarks/library_size_pseudo_hd_20260916T165847Z \
  --prior-dir runs/benchmarks/em_profile_source_controls_20260916T211500Z \
  --output-dir runs/benchmarks/YOUR_NEW_PREDICTIVE_RUN

Bin2Cell_Validation/bin/python scripts/evaluate_em_posterior_predictive.py \
  --run-dir runs/benchmarks/YOUR_NEW_PREDICTIVE_RUN \
  --source-dir runs/benchmarks/library_size_pseudo_hd_20260916T165847Z \
  --profile-dir runs/benchmarks/scdesign3_library_size_ab_20260916T163318Z

Bin2Cell_Validation/bin/python scripts/summarize_em_posterior_predictive.py \
  --run-dir runs/benchmarks/YOUR_NEW_PREDICTIVE_RUN \
  --source-dir runs/benchmarks/library_size_pseudo_hd_20260916T165847Z \
  --prior-dir runs/benchmarks/em_profile_source_controls_20260916T211500Z
```

Inference has no profile-directory argument and rejects known generating-profile
paths via Python/HDF5 guards. Those are safeguards, not an OS security sandbox.
GT profiles are supplied only to the separate official evaluation command for
gene-correlation metrics, after inference assignments have been hash-sealed.

Predeclared advancement requires: improved mean IoU over both frozen EM and
distance-only in A and B; at least three nondecreasing patches per comparison;
more corrected than damaged bins versus frozen EM; convergence and baseline
reproduction. No weights or iteration are selected using GT.

The completed run is
`runs/benchmarks/em_posterior_predictive_20260916T220700Z`.
See its `report.md`, `method_comparison.csv`, `development_gate.json` and executed
notebook. Diagnostics and validation logs are under `subdiagnostic/`.
