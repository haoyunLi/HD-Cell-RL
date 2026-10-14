# HD Cell RL

Linear project: [HD Cell RL (P-KUN-2)](https://linear.app/kun-lab/project/hd-cell-rl-07fccae0c508)

Team: `Kun_Lab`

Project UUID: `dc3117ec-334b-45a2-9115-4ab6affcde8e`

**Latest results — September 22, 2026**

**Single-cell (scRNA) processing: Completed.** Real observed scRNA counts now replace simulated-cell profiles in the delivered whole-section pair. Source-cell assignment, donor-separated reference construction, and paired data generation are complete ([KUN-6](<https://linear.app/kun-lab/issue/KUN-6>)). The pair contains 167,509 modeled cells and 8,176,117 barcodes. Both arms share exactly the same modeled-cell counts, source mapping, compartment placement, and capture draws.

The latest benchmark evaluates 42 reference-supported core cells across the same four development patches; it is not a new held-out test. All 43 core cells participated in inference, including one unsupported WNT5B+ cell reported separately. The rebuilt reference supports 28 cell types overall, but the primary benchmark covers only six of them.

| Method | Mean IoU: no background | Mean IoU: with background | Gene Spearman: no background | Gene Spearman: with background |
| -- | -- | -- | -- | -- |
| Bin2Cell | 0.8467 | 0.6597 | 0.5264 | 0.4868 |
| Distance-only | 0.8185 | 0.6271 | 0.5259 | 0.4837 |
| Frozen EM | 0.7816 | 0.5806 | 0.5107 | 0.4660 |
| STCS | 0.6661 | 0.6637 | 0.5226 | 0.4819 |
| Nucleus-only | 0.3989 | 0.3989 | 0.4003 | 0.4003 |

Frozen EM remains below distance-only in both conditions (mean IoU 0.7816 vs 0.8185 without background; 0.5806 vs 0.6271 with background). Whether nuclear expression is an adequate starting estimate of whole-cell expression remains under investigation in [KUN-9](<https://linear.app/kun-lab/issue/KUN-9>). Gene Spearman is measured against the original generating raw scRNA profiles, not capture-matched ground truth. This run includes no PPO, SMURF, or method-weight tuning.

**Dataset locations — latest paired release** `20260922T194014Z`

* **Without background noise (**`no_background`**; benchmark label** `modeled_only`**):** `/projects/illinois/vetmed/cb/kwang222/Haoyun_Li/Bin2Cell_Validation/outputs/pseudo_hd/colorectal_real_scrna_pair_20260922T194014Z/no_background/`
* **With background noise (**`with_background`**; benchmark label** `source_separated`**):** `/projects/illinois/vetmed/cb/kwang222/Haoyun_Li/Bin2Cell_Validation/outputs/pseudo_hd/colorectal_real_scrna_pair_20260922T194014Z/with_background/`
* **Shared donor-separated reference:** `/projects/illinois/vetmed/cb/kwang222/Haoyun_Li/Bin2Cell_Validation/outputs/pseudo_hd/colorectal_real_scrna_pair_20260922T194014Z/reference/reference_counts.npz`
* **Latest benchmark results:** `/projects/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL/runs/benchmarks/real_scrna_noise_pair_20260922T194014Z/benchmark/`

Each condition contains `filtered_feature_bc_matrix/`, `spatial/`, `ground_truth_cell_assignments.csv`, and its summary. Formal expression data live in Bin2Cell_Validation; HD-Cell-RL stores the benchmark outputs. The active entry `/projects/illinois/vetmed/cb/kwang222/Haoyun_Li/Bin2Cell_Validation/outputs/pseudo_hd/colorectal_nucleus_based_multi_owner_v2` points to `no_background`. The previous dataset is preserved at `/projects/illinois/vetmed/cb/kwang222/Haoyun_Li/Bin2Cell_Validation/outputs/pseudo_hd/colorectal_nucleus_based_multi_owner_v2.backup_20260922_before_real_scrna`.

Here, “with background noise” means added structured RNA only in tissue outside all modeled masks. It models unmodeled biological expression; it is not white measurement noise or a validated technical ambient-RNA model. Modeled bins receive no added background.

Both datasets passed structural and paired-count integrity checks, but neither passed all predefined biological-realism thresholds. They are experimental controls. The current capture probability is approximately 0.25197, calibrated for mean depth; per-cell library-size normalization has not been performed, so [KUN-7](<https://linear.app/kun-lab/issue/KUN-7>) remains open. Bin2Cell's native `min_counts=1` filter omits four zero-count core nuclear bins in each arm, so these native results are not fully matched for seed retention.

Result sources: `/projects/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL/runs/benchmarks/real_scrna_noise_pair_20260922T194014Z/benchmark/report.md`, `method_comparison_supported.csv`, `data_quality_acceptance.json`, and `subdiagnostic/evaluation_checks.json`; dataset completion and activation are recorded in `/projects/illinois/vetmed/cb/kwang222/Haoyun_Li/Bin2Cell_Validation/outputs/pseudo_hd/colorectal_real_scrna_pair_20260922T194014Z/completion.json` and `activation.json`.

**Urgent next action:** [KUN-10: Code review](<https://linear.app/kun-lab/issue/KUN-10>), priority Urgent, status Todo. Review the current changes across both repositories and the reproducibility of this benchmark.

---

HD Cell RL develops and evaluates bin-to-cell assignment methods for high-resolution spatial transcriptomics. The project brings together pseudo Visium HD data generation, expectation-maximization (EM) initialization, patch-based reinforcement learning, and comparisons with distance-only, Bin2Cell, SMURF, and STCS baselines. The aim is to assess both cell-boundary recovery and the expression profiles reconstructed from assigned bins.

The work spans two neighboring repositories:

| Repository | Responsibility |
| -- | -- |
| `Bin2Cell_Validation` | Nuclear segmentation, cell-type annotation, cell-boundary construction, scDesign3 expression simulation, pseudo-HD generation, and ground-truth validation. |
| `HD-Cell-RL` | Input preparation, episode construction, EM assignment, patch RL training, checkpoint evaluation, method comparisons, and interactive result inspection. |

Both repositories are located under `/projects/illinois/vetmed/cb/kwang222/Haoyun_Li/`. The nested `HD-Cell-RL/Bin2Cell_Validation/` directory is a runtime environment; the data-generation repository is the neighboring directory.

The workflow starts with data and ground-truth construction in Bin2Cell_Validation. HD-Cell-RL then prepares nuclear seeds, expression inputs, and episodes for EM and patch RL. Method comparisons should use the same evaluation cells and compatible inputs, with saved configurations and run identifiers linking each result to its source data. The current focus is patch-based assignment; older single-cell PPO/GRPO workflows remain available for comparison.

Current work addresses how the source and depth of expression data affect assignment quality:

* [KUN-6: Replace simulated cells with real-cell expression](<https://linear.app/kun-lab/issue/KUN-6>). Completed: real-scRNA processing and paired experimental datasets have been delivered. The previous simulated-cell dataset is preserved. Biological-realism validation and urgent code review remain separate outstanding work.
* [KUN-7: Match cell library sizes to spatial data](<https://linear.app/kun-lab/issue/KUN-7>). Evaluate adjustments to per-cell count depth and compare the resulting distributions with the target spatial dataset at explicitly defined cell, bin, or regional scales.
* [KUN-8: Generate pseudobulk data from Xenium cells](<https://linear.app/kun-lab/issue/KUN-8>). Develop an additional data-generation route with documented aggregation or spatial mapping, source-cell provenance, and gene-panel coverage.
* [KUN-9: Investigate EM initialization underperforming distance-only](<https://linear.app/kun-lab/issue/KUN-9>). The reported concern is that nuclear expression may poorly represent the eventual whole-cell profile. Reproduce the gap and test this explanation alongside count sparsity and library-size effects; the cause is not yet established.

Evaluation should combine assignment metrics such as IoU, Dice, precision, and recall with reconstructed-expression comparisons where ground truth is available. Data checks should cover coordinate mapping, cell/bin identities, and count conservation. Record dataset versions, configurations, checkpoints, and evaluation cell sets so changes in data construction can be distinguished from changes in the assignment method.

Results can be inspected through the [patch debugger](<https://haoyunli.github.io/HD-Cell-RL/>) and the [project progress report](<https://haoyunli.github.io/HD-Cell-RL/progress-report/>).

Local copy synchronized on September 22, 2026. Linear is the source for current issue status.
