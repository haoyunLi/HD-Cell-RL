# Progress-report data sources — 2026-09-08

The presentation separates original full-slide construction, the latest completed method comparison, and the latest data-only realism audit. “Latest” means a saved completed result, not an unrun plan or a newer source-code version.

## Slide-to-source map

Paths are relative to the repository root unless noted otherwise.

| Slides | Scope | Source |
|---|---|---|
| 06–11, 13, 15 | Original full-slide geometry and compartment recipe | External validation repository: `Bin2Cell_Validation/outputs/ground_truth/colorectal_nucleus_based/`; original whole-slide summaries |
| 12, 14 | Latest local donor-disjoint dataset | `runs/benchmarks/em_donor_disjoint_unseen_patches_20260903T192649Z/inputs/summary.json`; `reference/reference_counts_donor_disjoint.summary.json` |
| 17, 21 | Frozen EM and split/implementation audit | Same benchmark: `comparison/leakage_audit.json`, `comparison/report.md` |
| 19–20 | Preserved August PPO configuration/training | `runs/colorectal_nucleus_based_patch_overfit4_w5_validated_20260825T114854Z/` |
| 22 | Latest completed five-method results, measured September 3 | Donor-disjoint benchmark: `comparison/method_comparison.csv` |
| 23 | Historical frozen-RL diagnosis, not a new RL run | `runs/benchmarks/colorectal_nucleus_based_multi_owner_v2_43cells_20260826T210246Z/evaluations/hd_cell_rl_em_plus_frozen_rl_v2_43cells_20260826T212325Z/diagnostics/reward_attribution/` |
| 24 | Combined 28-configuration / 78-realization audit | `runs/benchmarks/pseudo_realism_20260908T012500Z/comparison/experiment_manifest.json`, `audit_2um_summary.csv` |
| 25 | Library-size variation and marker-separation controls | Same realism experiment: `comparison/source_profile_depth.csv`, `profile_controls/profile_depth.csv`, `comparison/profile_separation_matched_depth.csv` |
| 26 | Corrected exact-gene-ID background audit | Same realism experiment: `tissue_background_panel_ids/tissue_background_counts.csv`; local unassigned fraction in `comparison/audit_2um_summary.csv` |
| 27 | Imposed nuclear shifts and localization controls | Same realism experiment: `comparison/nuclear_observation_shift.csv`, `comparison/audit_2um_summary.csv` |
| 28 | Completed versus still-needed work | Same realism experiment: `comparison/report.md` |

## Definitions that must stay with the numbers

- **Latest local dataset:** 43 core evaluation cells, 190 context nuclei, 233 contributing physical-cell profiles and 10,827 unique context barcodes. These populations differ.
- **Reference versus HD genes:** 15,143 selected inference-reference genes versus 18,132 aligned HD panel features.
- **Donor split:** 4,500 generator scRNA cells from five donors versus 3,596 reference cells from seven different donors. Zero donor/barcode overlap, but one parent cohort and one source tissue slide.
- **IoU:** slide 22 shows standard mean per-core-cell IoU. Fractional IoU is separate; neither is the support balanced accuracy on slide 24.
- **Support BA:** counts > 0 predicts inside/outside the pseudo masks, using all four contexts. The real-HD row is also tested against pseudo masks, not real cell-boundary GT. Depth KS uses the two audit patches; lower KS is a closer distribution, not proof of biological truth.
- **Latest audit count:** use the combined `comparison/experiment_manifest.json` (28/78), not root `summary.json` (the first batch, 24/66).
- **Library control:** displayed CV is the mean of three control seeds. Those controls also use membrane strength 0.25; they do not isolate library size relative to the original membrane=1 recipe.
- **Background:** 34.705% is local real expression outside modelled masks; 1.158% is whole-slide off-tissue panel counts. Different denominators, neither a measured in-tissue ambient rate.
- **Perturbations:** transport, background fractions and nuclear shifts are sensitivity settings, not measured biological error rates.
- **Method implementation:** Bin2Cell GEX StarDist returned zero objects in both latest conditions. Reported results are external nuclear labels plus fixed expansion. SMURF and RL were not rerun.
- **Code version:** September 3 method outputs predate September 8 correctness fixes. They have not been regenerated with those fixes.
- **Reuse:** the four patches were unseen at their September 3 selection; their later reuse for realism diagnostics is not another held-out test.

## Verification

All displayed values in the five new result/audit tables were checked against the source CSVs, including rounding and three-seed CV means. The combined manifest and dataset dimensions were checked separately. The 29 slide numbers match the bilingual speech script.

Local Chromium/Playwright checks covered desktop (1600×900, 1280×720), a 390×844 mobile viewport, missing images, console errors, navigation and updated-page bounds. Desktop projection checks passed. At narrow mobile widths, Reveal automatically switches to scroll view: the requested slide-22 link can report the adjacent slide as active. Use desktop presentation mode for the talk; exact mobile deep-link positioning remains a limitation of the existing viewer.

Only presentation files and the bilingual script changed during the data refresh. No data generation, method evaluation, formal training or source-run overwrite was performed. The subsequent Pages publication updates only the presentation and preserves the existing patch-debug snapshot.
