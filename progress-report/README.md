# HD-Cell-RL progress report

One-hour Reveal.js progress deck, refreshed on September 8, 2026. The 29 slides distinguish the original full-slide simulation, preserved August PPO overfit experiment, September 3 donor-disjoint five-method comparison, and the latest 28-configuration / 78-realization data realism audit.

See [data sources and metric definitions](DATA_SOURCES.md) and the bilingual speech script in `docs/PROGRESS_REPORT_SPEECH_CN_EN.md` at the repository root.

## Public link

The deployed presentation is available at:

**https://haoyunli.github.io/HD-Cell-RL/progress-report/**

It is published with the existing patch-debug GitHub Pages snapshot, so viewers only need a browser.

The September 8 refresh updates the presentation only; the existing patch-debug snapshot is preserved.

## View locally

From this directory:

```bash
python -m http.server 4173
```

Then open `http://127.0.0.1:4173`. Use the arrow keys to navigate. Add `?print-pdf` to the URL for Reveal.js PDF export mode.

Reveal.js 6.0.1 is vendored locally, so the presentation does not require internet access.

## Suggested pacing

- Motivation, cell reconstruction, and molecular ownership: 10 minutes
- Simulated-data construction, step by step: 18 minutes
- Generalized EM and patch RL architecture: 10 minutes
- Historical training and latest method comparison: 7 minutes
- Latest data realism audit and next gate: 10 minutes

Allow approximately 5 additional minutes for questions.

Many diagrams use Reveal fragments. Advance within a slide to reveal each step; this is intentional pacing for a mixed technical and non-technical audience.

## Sources used

- `runs/benchmarks/em_donor_disjoint_unseen_patches_20260903T192649Z`
- `runs/benchmarks/pseudo_realism_20260908T012500Z` (combined comparison manifest, not the first-batch root summary)
- `Bin2Cell_Validation/outputs/pseudo_hd/colorectal_nucleus_based_multi_owner_v2`
- `runs/benchmarks/colorectal_nucleus_based_multi_owner_v2_43cells_20260826T210246Z`
- `runs/colorectal_nucleus_based_patch_overfit4_w5_validated_20260825T114854Z`
- Ishaque et al., *The Challenge of Cell Segmentation in Spatially Resolved Transcriptomics*, arXiv:2606.09675 (2026)
- [10x Genomics Visium HD Spatial Gene Expression](https://www.10xgenomics.com/support/spatial-gene-expression-hd)
- [10x Genomics Space Ranger binning algorithms](https://www.10xgenomics.com/support/software/space-ranger/latest/algorithms-overview/gene-expression)
- [10x Genomics Xenium Prime 5K panel information](https://www.10xgenomics.com/support/software/xenium-panel-designer/latest/analysis/pre-designed-panels/pre-designed-xenium-prime-5k-parts)

The preserved PPO run is historical four-patch overfit. The latest method comparison is donor-disjoint but not independent-tissue validation. The newest realism audit reuses examined patches and does not establish a new method ranking. Source/denominator caveats remain visible beside the data.

The locally generated microscopy and workflow illustrations are presentation schematics. Actual project output is used for nuclear segmentation, whole-cell expansion, and patch-level diagnostics.

## Interactive explanation edition

This edition retains 29 slides and all existing experiment results. Use the public link above to share the presentation.

- 05 follows fixed schematic nuclei A–E through simulated masks, RNA placement, binning and the inference/evaluation boundary.
- 10 reveals the saved expansion overlay on the same H&E crop.
- 17 computes a deterministic educational EM example, with whole-patch E/M steps and fixed nuclear cell profiles.
- 19 separates a schematic transfer drawing from two measured historical REPLACE actions; GT is revealed last.
- 22 compares conditions with paired, identically scaled IoU dot plots; the exact table remains expandable.
- 24 compares pseudo coverage, nonzero pseudo counts and nonzero real counts across all four saved contexts.
- 25 plots actual Cycling T UMI empirical distributions; it does not reconstruct a distribution from aggregate CVs.
- 27 shifts saved nuclear bins across unchanged pseudo coverage and reports missing destinations separately.

Use the arrow keys or Next / Back / Replay. No animation autoplays. The current step and selected case/context are retained in the URL on the interactive pages. Mobile portrait uses a stacked reading layout; mobile landscape keeps the scene and explanation adjacent where space permits. Other legacy slides retain their fixed presentation layout. Reduced-motion mode removes transitions. PDF mode shows final states with static key-frame summaries for EM and both REPLACE outcomes.

### Design and evidence contract

The user approved the white desktop/mobile EM concept on September 8, 2026 (preview `exec-8ff03a04-3364-4ce0-acef-1b6fe7533001.png`). Locked elements are fixed spatial identities, scene-first reading order, directly labelled probabilities, a stepper, visible schematic/source caveats and editable data layers. Exact geometry and probability widths are computed in code, correcting the approximate concept image. The raster preview is not a scientific asset.

SVG owns the sparse schematic and quantitative chart labels; Canvas owns the dense, actual barcode maps. Reveal fragments own step navigation. These are local specialist passes, without delegated agents or new dependencies. The shared A–E geometry is a teaching device; the historical A/B transfer roles are not asserted to be those same physical cells. The production EM/RL implementation is untouched.

To regenerate the compact slide data from saved results, run from the repository root:

```bash
Bin2Cell_Validation/bin/python scripts/build_progress_report_story_data.py
node --test tests/progress_report_story.test.cjs
```

The builder reads existing matrices and annotations, reconciles profile summaries and reward components, and writes only `assets/story-data.js`. It does not regenerate pseudo data, run any method, train a policy or change source runs.
