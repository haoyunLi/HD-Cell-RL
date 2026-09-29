# From 2 µm bins to physical cells

This is the UIUC-styled [web progress report](index.html). It contains a 32-slide, approximately one-hour research talk for an interdisciplinary audience. The older public deck remains available separately at `/progress-report/` on the project site.

The talk is designed for an approximately one-hour, interdisciplinary presentation. It first compares bulk, dissociated single-cell, and spatial RNA measurements, then introduces Visium HD. The opening goes on to explain why cell-level reconstruction matters, what Bin2Cell and STCS already do, the remaining ownership question, and the method's inputs and outputs. It uses one small results comparison, with explicit limits, rather than a long leaderboard.

## Present or share

- Open `index.html` in a browser from this repository. A viewer does not need Python; the deck includes its required Reveal.js files and source images.
- The GitHub Pages address is `https://haoyunli.github.io/HD-Cell-RL/progress-report-uiuc/` after the Pages snapshot is updated.
- Use arrow keys or click the navigation control to advance. Some diagrams reveal one explanatory step per advance: tissue → barcode → gene counts; nuclei → simulated cell boundaries; candidates → soft ownership; EM E-step → M-step; and RL before → after. Left arrow moves back through these steps. Press `S` for speaker notes, `F` for full screen, or `O` for slide overview.
- Motion is short and never loops. A reduced-motion browser preference removes transitions while preserving the step-by-step information.

## Update the report

Edit claims, titles, speaker notes, and sources in `slides.json`; edit diagrams and explanatory reveals in `deck.js` and `deck.css`. To sync the offline-friendly JavaScript data file:

```bash
python web/progress-report-uiuc/sync_slides.py
```

This writes only `slides-data.js` and requires no third-party Python packages. It does not modify the older deck or any training/evolution run.

## Evidence boundaries

The comparison on slide 29 (zero-based index; slide 30 in the viewer) comes from the saved [colorectal Xenium-derived report](../../runs/benchmarks/xenium_notebook_crc_4patch_final_v2_20260924/report.md) and [lung Prime 5K-derived report](../../runs/benchmarks/xenium_notebook_lung_prime_5k_4patch_20260925/report.md). Both use fixed settings and four patches. Their IoU values are diagnostic; neither is a final held-out model-selection result or measured Visium HD. The matched real HD/Xenium track is described separately, without treating its registration proxy as perfect cell truth. Illustrative diagrams and probabilities are labelled as such.

The [visual plan](CONCEPT.md) records the slide sequence and design rationale. The official [UIUC Vet Med template](https://vetmed.illinois.edu/designgroup/downloads/) and [Illinois colors](https://brand.illinois.edu/visual-identity/color) inform the visual system.
