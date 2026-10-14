# Real scRNA sources for pseudo Visium HD

This mode replaces scDesign3-generated expression profiles with observed raw
scRNA counts. It does not change nuclear/cell masks, cell-type annotations,
compartment kernels, EM, or RL. The existing synthetic mode remains available.

## Source assignment

`hd_cell_rl/spatial_profile_assignment.py` assigns one real source barcode to
each physical `cell_id`, matched by the existing cell-type label.

- Nucleus centres define a regular spatial grid (default 128 µm squares, origin
  `(0, 0)` in the recorded physical coordinate frame).
- A source barcode can occur at most once inside each patch.
- A source barcode also cannot repeat between cells whose nucleus centres are
  at most 64 µm apart, including cells on opposite sides of a patch boundary.
- Distant patches can reuse a source. Overlapping context appearances of the
  same physical cell retain the same mapping.
- Sorting physical IDs and source barcodes makes the result reproducible under
  input-row permutations. Pool exhaustion raises an error; no silent replacement,
  cell-type remapping, or borrowing from reference donors occurs.

Both distances are configurable generation settings, not optimized biological
parameters or EM candidate radii. The exclusion is based on nucleus-centre
distance, not a claim that every possible touching mask lies inside that radius.
Generation additionally rejects source reuse by two masks sharing a physical
bin, even if their centres lie beyond the configured radius. This removes
repeated source identities, not similarity between distinct cells.
The greedy sampler can fail even when a different global assignment might exist.

## Donor separation

Provide explicit `--generator-donors` and `--reference-donors`, or use
`--plan-donor-split`. The planner enumerates splits of up to 16 donors using only
type coverage and required within-patch source counts. It does not read method
performance or inferred ownership. Its ranking is documented in
`plan_donor_split()` and frozen in the source snapshot.

The current whole section has types that occur in only one scRNA donor. A
donor-disjoint split cannot represent those types in both generation and
reference. `donor_split.json` and `subdiagnostic/donor_split_type_coverage.csv`
report missing reference types explicitly. Local no-repeat capacity may force
additional donors onto the generation side and remove further reference types.

`reference_metadata.tsv` is the new eligible reference population. **Do not reuse
the old reference-count artifact after changing donor membership.** Rebuild it
from this population before comparisons. Missing types require a separately
authorized independent reference dataset or an explicit open-set evaluation
protocol; this runner does neither automatically.

## Assignment-only validation on the full section

From the HD-Cell-RL root:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
Bin2Cell_Validation/bin/python scripts/build_real_scrna_pseudo_hd.py \
  --ground-truth-cells ../Bin2Cell_Validation/outputs/ground_truth/colorectal_nucleus_based/bins_multi_owner_v2/represented_cell_metadata.csv.gz \
  --source-h5 ../Bin2Cell_Validation/data/single_cell/colorectal/CRC_GSE166555_raw_tumor.h5 \
  --source-metadata ../Bin2Cell_Validation/data/single_cell/colorectal/GSE166555_meta_data_tumor.tsv \
  --plan-donor-split --assignment-only \
  --output-dir runs/benchmarks/real_scrna_assignment_NEW_RUN
```

This writes the global mapping, split/coverage audits, input fingerprints and a
recoverable source snapshot. It does **not** generate HD expression.

## Generate HD counts

Use the same ground-truth and source inputs and the saved explicit donor lists.
Remove `--assignment-only`, choose a new output directory, and add:

```text
--bin-assignments ../Bin2Cell_Validation/outputs/ground_truth/colorectal_nucleus_based/bins_multi_owner_v2/ground_truth_bin_assignments.parquet
--original-hd-dir ../Bin2Cell_Validation/data/visium_hd/colorectal/output/binned_outputs/square_002um
--capture-probabilities-npy /path/to/previously_resolved_capture_field.npy
```

The capture array must follow the original H5 barcode order. Alternatively,
`--capture-probability P` applies a declared scalar in `[0,1]`. There is no default
depth calibration and no per-cell library-size normalization. `P=1` keeps every
on-panel molecule and should not be described as realistic HD capture.

The runner calls Bin2Cell_Validation's existing localization calibration,
nuclear kernel, cytoplasm/membrane weights, molecule-placement routine, sparse
aggregation, and 10x writers. Capture uses a separate random stream after
compartment placement. Output conserves source counts after panel filtering and
capture rejection; cells sharing a bin can both contribute molecules. Source
cells with zero nuclear counts are not removed.

Outputs keep original barcode and feature axes:

- `filtered_feature_bc_matrix/` and `spatial/` for existing tool entry points;
- `ground_truth_cell_assignments.csv`, using real `sc_cell_barcode` identities;
- `modeled_source_bin_counts.npz`, physical-cell rows in assignment CSV order,
  columns in original barcode order (records actual captured contributions);
- `pseudo_hd_summary.json` and `subdiagnostic/per_cell_capture.csv`;
- `experiment_manifest.json`, source snapshot, and archived external kernel code.

For gene-correlation evaluation use this assignment CSV and the **raw real scRNA
H5**, not the old synthetic H5. The physical geometry GT remains unchanged.

This entry point generates modeled-cell RNA only. It does not silently add or
recalibrate unmodeled biological expression or technical background. Existing
source-separated components need explicit alignment and budget checks before
combining them in a later experiment. New dataset generation is not evidence of
realism or improved method performance.

## Verification

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
Bin2Cell_Validation/bin/python -m unittest discover -s tests \
  -p 'test_spatial_profile_assignment.py' -v
```

Tests cover donor separation, unavoidable missing types, patch/boundary
exclusions, distant reuse, physical-identity consistency, order invariance,
raw-count validation, count conservation, capture fields, shared-bin
contributions, aggregation-block invariance, 10x export, and overwrite refusal.
Integration tests use the actual installed Bin2Cell_Validation kernels and skip
with an explicit reason if that adjacent repository is unavailable.

Do not treat patches reusing source cells as independent donor replicates.
Generator/reference donor separation also does not guarantee independence
between simulated training and test patches; future splits should separately
control source reuse across their evaluation partitions.
