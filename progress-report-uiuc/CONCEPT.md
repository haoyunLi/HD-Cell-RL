# Progress report redesign — visual concept

Status: implemented as a [32-slide web deck](index.html). The previously published deck at `/progress-report/` has not been changed.

## Brief

- Audience: interdisciplinary, including listeners without computational biology training.
- Length: approximately one hour; English slides, with a bilingual speaker script later.
- Deliverable: a shareable web presentation. Text and the main explanatory diagrams remain editable HTML/CSS/SVG; microscopy remains source imagery.
- Emphasis: explain the scientific problem and each method step visually. Show only a small, current, carefully qualified results section.

## Visual direction

**Thesis:** use the official Illinois Vet Med wide-blue content template as a quiet academic frame. The revised cover makes the actual H&E grid a full-width visual field instead of a floating tissue cutout. Content slides use one dominant source image or editable diagram, with a restrained navy/orange hierarchy.

**Figure test:** without reading the speaker notes, a viewer should be able to tell what was measured or changed, what the compared states are, and which takeaway the slide is making. Prefer a project image when the claim is about this project's data or workflow. Use a published image only when its visible content—not just its provenance—explains the point at presentation size. Mark illustrative counts, probabilities, and boundaries as such.

The visual frame follows the [University of Illinois wide-blue presentation template](https://vetmed.illinois.edu/designgroup/downloads/). Use [Illini Blue `#13294B` and Illini Orange `#FF5F05`](https://brand.illinois.edu/visual-identity/color), with orange reserved for emphasis. The [university typography guidance](https://brand.illinois.edu/visual-identity/typography) calls for Montserrat headings and Source Sans body text; both open-source fonts and their licenses are bundled with the web deck. Cyan/magenta can appear in scientific overlays where they encode distinct cell masks or nuclear contours; they are not additional brand accents.

The cover uses a real H&E grid; the simulation page uses aligned project images; the EM candidate page shows four eligible nuclei and one excluded nucleus. The EM probabilities are illustrative, not experimental results. All content-slide headings below are short takeaways rather than topic labels. None uses a colon or semicolon.

**Emphasis rule:** each slide may underline one, occasionally two, short terms that carry its point. The underline uses Illini Orange while the words remain navy, so the emphasis stays legible on a projector. The cover already uses orange for `2 µm`, so it needs no second treatment. Do not color whole sentences, mix underline and bold-color styles arbitrarily, or highlight every technical term.

## Narrative and approximate timing

The opening follows the research-talk skill's argument in order, rather than introducing an algorithm before its question. Each row in the slide map below names the one job of that slide. In the opening, slides 01–02 establish the measurement context; 03–06 explain scientific significance and the missing cell-level information; 07 states what earlier tools do and where inference is still required; 08 asks the research question in one sentence; 09 makes the method input and output explicit; 10 gives the proposed contribution without equations. Slide 11 bridges to the controlled answer key. For this one-hour interdisciplinary talk, the opening uses more than the guide's usual 6–8 slides because the audience also needs a visual explanation of 2, 8, and 16 µm measurement units and fractional boundary bins.

The scientific distinction is explicit: **significance** is obtaining credible physical-cell profiles and neighbors from spatial measurements; possible **impact** is better-grounded cell annotation and cell–cell communication hypotheses. Those downstream outcomes are prospective, not results claimed by this talk.

### Opening — 17 minutes

The opening follows the research-talk sequence: compare bulk, dissociated single-cell, and spatial RNA measurements → introduce Visium HD → show why cell-level reconstruction matters → identify the missing ownership information → review prior work and its limits → state the question, inputs, outputs, and proposed approach. The claim is deliberately modest: inferred cell ownership could enable better cell annotation and cell–cell communication analyses; the current outputs do not themselves validate those downstream findings.

| Slide | Takeaway title | One job / visual |
| --- | --- | --- |
| 00 | From 2 µm bins to physical cells | Title over an actual H&E region. |
| 01 | Spatial RNA keeps expression in tissue context | The measurement-level row of a published bulk/scRNA/spatial comparison, with three large takeaways. |
| 02 | Visium HD measures RNA in 2 µm squares | Project H&E → one highlighted barcode → an illustrative gene-count vector. |
| 03 | Cell-level questions need cell-level data | Visual conversion from spatial squares to physical-cell profiles to annotation and communication questions. |
| 04 | A 2 µm bin is not a cell | Show one cell crossing several bins and a boundary bin shared by neighbors. |
| 05 | Bigger bins still do not show cell boundaries | Matched 2/8/16 µm crops, one shared cell mask. |
| 06 | Boundary RNA can belong to either neighboring cell | Visible nuclei versus the unobserved RNA source for one orange barcode. |
| 07 | Existing tools still infer where boundary RNA belongs | Bin2Cell, STCS, and distance-only: evidence used, action taken, and specific limitation. |
| 08 | Which physical cell supplied this bin's RNA? | The research question alone, with a tissue image and one target barcode. |
| 09 | H&E and RNA counts become cell assignments | Explicit inputs and soft/final outputs; GT mask is visibly excluded from inference. |
| 10 | Nuclei anchor cells before we assign and refine bins | High-level project contribution, before technical equations or configuration. |
| 11 | A simulated tissue gives us an answer key | Reuse the earlier, clearer four-panel tissue illustration with new readable captions. |

Transition: after the audience knows the question and the tool's I/O, show why controlled data are needed to evaluate an otherwise unobserved RNA source.

### Controlled data and simulation — 16 minutes

| Slide | Takeaway title | One job / visual |
| --- | --- | --- |
| 12 | Simulation tells us where each bin's RNA came from | Molecule source → observed gene counts with source hidden → evaluation-only answer key. |
| 13 | We first find nuclei in the H&E image | Actual H&E crop with actual nuclear contours. |
| 14 | 8 µm RNA helps us label each nucleus | One parent 8 µm bin, a saved type-posterior example, and the selected nucleus label. |
| 15 | Cell type and neighbors guide cell size | Small/large-cell and crowded-region examples. |
| 16 | We simulate cell boundaries around each nucleus | Matched actual nucleus and generated-mask ROI; reveal mask over the same crop. |
| 17 | One bin can contain RNA from more than one cell | Fractional ground-truth ownership for a single 2 µm bin. |
| 18 | Real scRNA cells supply the RNA profiles | Current real-cell generator and independent generator/reference donor split; distinguish from the older scDesign3 iteration. |
| 19 | We place RNA differently across three cell regions | Three spatial patterns: nuclear center, cytoplasmic interior, and membrane edge. |
| 20 | We count the simulated molecules in 2 µm bins | Placed molecules → selected physical barcode → gene counts, with three RNA sources separate. |

Transition: controlled truth lets us ask whether an algorithm recovers owners from measurements it could actually observe.

### Inference and learning — 17 minutes

| Slide | Takeaway title | One job / visual |
| --- | --- | --- |
| 21 | We test the method on three kinds of data | Three visually distinct tracks: controlled pseudo HD, Xenium-derived bins, and paired measured HD/Xenium. |
| 22 | Every nearby nucleus can compete for a bin | All eligible cells within the existing candidate MaxDis, including patch-margin cells; nuclear seed bins locked. |
| 23 | EM weighs every candidate before updating a cell | A boundary-bin responsibility example in the batch E-step, then fractional evidence in the batch M-step. |
| 24 | Entropy shows which EM assignments are uncertain | Two bins with the same four candidate cells: one clear owner remains fixed, while a high-entropy bin becomes eligible for RL. |
| 25 | RL can change uncertain bin owners while nuclear seeds stay fixed | One `REPLACE A → B` storyboard; locked nuclei remain fixed. |
| 26 | RL scores each change using the whole patch | Explain expression, geometry, overlap/neighborhood, and shape as distinct terms; no formula wall. |
| 27 | PPO training has used only four patches | Clearly label the four-patch overfit scope and keep training separate from generalization evidence. |

Transition: the goal is not merely a higher number on one simulator, but a method that behaves consistently as the source of truth changes.

### Evidence and close — 10 minutes

| Slide | Takeaway title | One job / visual |
| --- | --- | --- |
| 28 | We compare methods under the same rules | One compact protocol slide: nucleus-only, Bin2Cell, STCS, distance-only EM, expression-aware EM. |
| 29 | Adding expression does not always help EM | One carefully sourced visual, no dense leaderboard. Separate observed finding from proposed explanation. |
| 30 | Each benchmark has its own blind spots | Pseudo-data realism, Xenium-derived truth, and cross-platform reference mismatch; no claim of perfect ground truth. |
| 31 | We need stronger validation before training PPO again | Three gates: realistic counts/background, independent regions, then an RL net-gain test. |

The implemented results slide uses the saved September 24 colorectal and September 25 lung Prime 5K Xenium-derived four-patch reports. It is marked as a fixed-setting diagnostic comparison, not final held-out model selection. This leaves most of the talk for the problem, simulator, EM, and RL explanations.

## Presentation behavior and build constraints

- Reveal the scale change from tissue to 2 µm bin; show a one-to-one matched H&E/mask overlay; reveal all MaxDis candidates before their soft weights; show one EM E-step → M-step cycle and one RL `REPLACE` before/after. These are presenter-controlled explanatory reveals, not evolving experimental results.
- Keep each slide to one claim, one primary visual, and at most a few short labels. Design for a projector, not a laptop: large body type and no tiny plot annotations.
- Respect reduced-motion preferences. Preserve all information when transitions are disabled and ensure the completed slide can be understood as a still image.
- Use editable HTML text and inline SVG for labels, arrows, cells, probability bars, and diagrams. Source microscopy stays raster.
- Keep the existing public deck and previous training/evolution artifacts untouched. The redesign lives alongside them until reviewed and approved.
