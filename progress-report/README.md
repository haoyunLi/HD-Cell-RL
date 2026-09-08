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
