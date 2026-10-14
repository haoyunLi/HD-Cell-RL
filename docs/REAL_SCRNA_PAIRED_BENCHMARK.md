# Paired real-scRNA data and algorithm comparison

Formal expression data belong in `Bin2Cell_Validation/outputs/pseudo_hd/`.
`HD-Cell-RL/runs/benchmarks/` contains evaluation results, logs and diagnostics;
its patch dataset entries are links to data stored in Bin2Cell_Validation.

## Paired conditions

`scripts/build_real_scrna_paired_dataset.py` builds a whole-section pair from
the previously validated, patch-exclusive real-scRNA source mapping:

- `no_background`: modeled-cell RNA only;
- `with_background`: the exact same modeled counts plus structured RNA outside
  all modeled masks, restricted to tissue.

Both preserve original barcode/gene axes and multi-cell contributions to bins.
They share source cells, compartment placement and capture draws. Background is
defined by zero geometric mask coverage, not by observed zero counts. Nuclear
and cytoplasmic bins inside masks do not receive added background.

The added component is a sensitivity model for unmodeled biological expression,
not white measurement noise or an identified ambient mechanism. Source types
follow nearby physical-cell types; expression comes from generator donors only.
Gaussian kernels use sigma 8 µm and truncate at 3 sigma and 128 µm tile edges.
Proposed density is one virtual source per 64 µm² of complement support. Poisson
source budgets match only the mean real complement depth on calibration patches.
Virtual sources are not extra GT cells. Modeled cells and virtual sources now
share a source-exclusion check: no real scRNA identity can repeat within a
128 µm generation tile or within 64 µm across tile boundaries. Distant reuse is
allowed. When a type-specific pool is exhausted, background sources are reduced;
modeled assignments are never replaced. The removed sources' counts are split
among retained sources of the same tile and type, conserving each budget exactly.
An infeasible positive-budget group fails rather than borrowing reference cells
or changing type. Real source gene proportions are used, but the imposed RNA
budget is not the raw single cell's original library size.
If no modeled cell lies within 72 µm, the nearest cell supplies the type.
The source audit records this fallback and its distance; a distant fallback
does not establish that the expression is biologically appropriate there.

Capture is a single probability fitted from expected source localization counts
and real counts inside masks on the first two sorted development patches. The
last two patches do not contribute to capture or background-budget fitting.
There is no per-cell library-size normalization and no method-score tuning.
This is not proof of realistic gene detection, cell variability or spatial
correlation; those are checked separately at 2/8/16 µm.

## Independent reference and evaluation scope

The reference is rebuilt from the new reference donors and raw counts, using
the existing `build_reference_counts_from_scrna.py` implementation. Generator
and reference donors and source barcodes are disjoint. It contains 28 types;
the other 8 are not silently relabeled or filled from generator cells.

All cells remain in generation and inference. Reference support is applied only
to post-hoc metric grouping:

- `method_comparison_supported.csv`: supported types, primary comparison;
- `method_comparison_missing_reference.csv`: unsupported types, diagnostic only;
- `evaluated_type_coverage.csv`: records actual evaluated cells, including zero
  coverage for types absent from the selected core cells.

On the existing four development patches, 42 of 43 core cells are supported.
The sole unsupported core cell is WNT5B+. This experiment does not evaluate all
28 supported types or all 8 missing types, and is not a new held-out test.
Margin cells continue to compete, irrespective of reference support.

## Algorithm protocol

`scripts/run_real_scrna_paired_benchmark.py` reuses the existing comparison
entry points: nucleus-only, distance-only, frozen cell-profile EM, Bin2Cell and
STCS. No PPO or Smurf runs. EM keeps alpha=1, beta=1.5, kappa=2 and profile update
damping=0; its iteration ceiling is 50 to check convergence, not to select a
GT-optimal iteration. Distance-only sets beta=0. No rewards or weights are fitted.

Each method uses the same confident nuclear seed mask. The EM bin filter remains
`positive_expression`, so adding complement RNA can introduce new eligible bins;
report that support change, not only IoU. Metrics use the existing core-cell
protocol and GT masks. Gene correlation uses the actual generating raw scRNA H5,
not the old synthetic H5. Paired intervals resample patches, not individual cells.

## Storage and safe activation

The versioned dataset directory contains both condition directories, the new
reference, source mapping, donor metadata, component matrices, provenance and
calibration settings. Patch test datasets also remain under that directory.

After validation, `scripts/activate_real_scrna_dataset.py` renames the old
`colorectal_nucleus_based_multi_owner_v2` directory to a timestamped backup and
creates an entry-point link to the explicitly selected condition. The user chose
`no_background` as the default. No files are deleted. Historical data can be
reproduced using the backup path; historical training/evolution outputs are not
modified. The separate old `colorectal_nucleus_based` directory is not changed.

The old reference must not be reused with the new donor split. See
`reference/reference_counts.npz` inside the versioned paired dataset.

## Previous four-patch pair

The 2026-09-22 pair is stored in
`Bin2Cell_Validation/outputs/pseudo_hd/colorectal_real_scrna_pair_20260922T194014Z/`.
This pair initially supplied the default `no_background` entry. The older
dataset remains in
`colorectal_nucleus_based_multi_owner_v2.backup_20260922_before_real_scrna`.

Both arms passed structural and paired-count checks, but neither passed all
predefined biological-realism thresholds. The user approved activation as an
experimental control despite that limitation. `EXPERIMENTAL_CONTROL.md` at the
active entry and `activation.json` in the paired root record that distinction.

Comparison outputs are under
`runs/benchmarks/real_scrna_noise_pair_20260922T194014Z/benchmark/`.

## Jointly exclusive background and sixteen-patch extension

`scripts/rebuild_unique_background_pair.py` versions an existing pair without
regenerating modeled expression, geometry, localization, donor split or reference.
It records all proposed/retained source identities, per-tile/type budgets and an
independent joint exclusion audit. Matrices shared with the previous dataset are
linked, so the previous version must be retained.

The new pair is
`Bin2Cell_Validation/outputs/pseudo_hd/colorectal_real_scrna_unique_background_20260922/`.
The canonical `colorectal_nucleus_based_multi_owner_v2` entry now points to this
pair's `no_background` arm. Its prior entry is preserved as
`colorectal_nucleus_based_multi_owner_v2.backup_20260922_before_unique_background`.
The background proposal count falls from 201,039 to 200,440. The 599 removals
affect 15 generation tiles; the total background budget remains 123,511,393 UMI.
Concentrating the same budget on fewer sources can change local RNA density.
That is a limitation to audit, not a claim that the new background is realistic.

`scripts/prepare_real_scrna_patch_comparison.py` freezes the original four patches
and twelve additional, spatially separated patches before tool evaluation. The
selection uses geometry, a recorded random seed and exclusion of recorded prior
test patches, not method scores. It contains 169 core cells: 146 have reference
support and 23 are missing-reference diagnostics. These are additional regions
of the same section, not new independent donors.

Generation tiles (128 µm) are distinct from evaluation patches (64 µm outer,
48 µm core). Context nuclei extend by the existing 20 µm candidate distance;
expression is extracted another 20 µm beyond them. The wider expression halo
means historical four-patch scores are not an isolated before/after test of
background exclusion. All five methods and both arms share the new contexts.

Pass `--evaluation-context <dataset>/evaluation16` to the paired benchmark runner.
Results include `method_comparison_by_cohort.csv` for original-four versus
additional-twelve results, alongside all-sixteen and per-patch tables. The
component links retain legacy `leave_out_*` names for loader compatibility;
there is no leave-out refitting in this extension. Calibration remains frozen.

`scripts/audit_real_scrna_background_update.py` independently checks donor
separation, raw source indices, unchanged modeled/reference inputs and exact
tile/type budgets. It also reports whether the evaluation contexts contain any
removed source centers. For this fixed sixteen-patch selection, they do not;
method scores therefore do not isolate the rare-type source-density reduction.
Results are stored under
`runs/benchmarks/real_scrna_unique_background_16patch_20260922/benchmark/`.
