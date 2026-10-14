# HD Cell RL Runbook

This runbook lists the execution order for preprocessing, episode building, reward search, PPO/GRPO training, checkpoint evaluation, and method-baseline comparison.
Each step depends on outputs from earlier steps, so run in order unless you already have the required files.

Repository roles and placement rules are documented in [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md). Slurm entry points are grouped by purpose in [`jobs/`](jobs/README.md), and scheduler output is written to `logs/slurm/`.

## Live Patch Debugger

**[Open the latest published patch evaluation](https://haoyunli.github.io/HD-Cell-RL/)**

**[Open the project progress report](https://haoyunli.github.io/HD-Cell-RL/progress-report/)**

The public `gh-pages` snapshot keeps the latest patch debugger at the site root and the Reveal presentation under `/progress-report/`. Source code remains on `main`; each successful publication replaces the previous generated snapshot instead of accumulating run artifacts in Git history.

## Supported execution paths

The current patch workflow is episode construction → optional EM initialization → discrete patch RL → patch checkpoint evaluation. Single-cell PPO/GRPO remains available as a legacy comparison, not the main patch path.

- EM-only experiments: `scripts/run_em_weight_sweep.py` (no policy training).
- Patch training: `scripts/run_patch_training.py --config <patch-config.yaml>`.
- Patch evaluation: `scripts/evaluate_patch_checkpoint.py --help`.
- Legacy single-cell training/evaluation: the full-GRPO and single-cell PPO jobs below.

On an allocated HPC node these Python entry points can run directly; Slurm wrappers are optional. For CPU patch runs set `run.device: cpu`, `run.rollout_devices: [cpu]` and `patch_training.rollout_backend: torch_gpu` in a separate config. The last value is the existing backend name; it also accepts CPU devices. Do not change a previous run's saved config.

Use `requirements-core.txt` for the EM/patch runtime; the full `requirements.txt` also installs segmentation and annotation tools. See [engineering safeguards and verification](docs/ENGINEERING_SAFETY.md) for checkpoint compatibility, cache behavior and CPU test commands. No new training or public deployment is triggered by these safeguards. GitHub Pages publication now requires an explicit opt-in.

See [the September 9 project corrections](docs/PROJECT_REMEDIATION_20260909.md) for the named frozen EM baseline, deterministic cross-patch exports, patch-cluster confidence intervals, source-separated pseudo-data controls and opt-in validation checkpoint selection. New pseudo controls remain experimental until multiscale acceptance passes; do not replace old datasets or start formal PPO from a failed audit.

Legacy single-cell ADD reward expression mixture (not the patch global objective):
```text
ADD reward =
  w1 * zscore(new posterior-confidence gain over frontier)
+ w5 * zscore(old bin-posterior compatibility over frontier)
- w2 * distance_penalty
- w3 * overlap_penalty
+ w4 * neighbor_support
```
Default full-GRPO config uses `w1: 0.45` and `w5: 0.10`.

## 1) Crop Visium HD image
This step extracts the tissue region used for all downstream processing.
The output image is the input for segmentation.
```bash
sbatch jobs/preprocessing/run_crop_visium_hd.sbatch
```
Expected output:
- `workspace_outputs/colorectal_nucleus_based/intermediate/cropped_visium_hd_colorectal_nucleus_based.png`

## 2) Run Cellpose nuclear segmentation
This step runs nucleus segmentation on the cropped image and writes cell/nucleus masks and related artifacts.
These segmentation results are later mapped to official 2um bins.
```bash
sbatch jobs/preprocessing/run_cellpose_sam.sbatch
```
Expected output folder:
- `workspace_outputs/colorectal_nucleus_based/intermediate/cellpose_sam_colorectal_nucleus_based_output/`

## 3) Map pixel nuclear signal to official `square_002um` bins
This step converts pixel-level nuclear assignments into official Visium HD `square_002um` bin coordinates.
It also writes an overlay for quick visual QC.
```bash
sbatch jobs/preprocessing/run_pixel_to_square_002um_bins.sbatch
```
Expected outputs:
- `workspace_outputs/colorectal_nucleus_based/intermediate/nuclei/colorectal_nucleus_based_square_002um_nuclear_bins.csv.gz`
- `workspace_outputs/colorectal_nucleus_based/intermediate/nuclei/colorectal_nucleus_based_square_002um_nuclear_bins.summary.json`
- `workspace_outputs/colorectal_nucleus_based/intermediate/nuclei/colorectal_nucleus_based_square_002um_nuclear_bins_overlay.png`

### Export nucleus-based ground truth bins

The current pseudo-HD dataset stores exact cell and compartment assignments in a parquet table. Export it once into the CSV schema used by shape-reference construction and evaluation:

```bash
sbatch jobs/preprocessing/run_pseudo_pixel_to_square_002um_bins.sbatch
```

Expected outputs:
- `workspace_outputs/colorectal_nucleus_based/intermediate/ground_truth/colorectal_nucleus_based_square_002um_cell_bins.csv.gz`
- `workspace_outputs/colorectal_nucleus_based/intermediate/ground_truth/colorectal_nucleus_based_square_002um_nuclear_bins.csv.gz`
- `workspace_outputs/colorectal_nucleus_based/intermediate/ground_truth/colorectal_nucleus_based_square_002um_gt_bins.summary.json`

## 4) Merge filtered expression metadata + nuclear annotation
This step merges official HD filtered-bin metadata/expression indexing with nuclear-bin annotation into one RL-ready table.
The expression source can be a 10x `.h5` file or a `filtered_feature_bc_matrix/` directory.
The merged table is the main structural input for episode construction.
```bash
sbatch jobs/preprocessing/run_merge_square_002um_expression_with_nuclear.sbatch
```
Expected outputs:
- `workspace_outputs/colorectal_nucleus_based/intermediate/square_002um_filtered_nuclear/colorectal_nucleus_based_square_002um_filtered_rl.metadata.parquet`
- `workspace_outputs/colorectal_nucleus_based/intermediate/square_002um_filtered_nuclear/colorectal_nucleus_based_square_002um_filtered_rl.selected_features.tsv.gz`
- `workspace_outputs/colorectal_nucleus_based/intermediate/square_002um_filtered_nuclear/colorectal_nucleus_based_square_002um_filtered_rl.selected_feature_indices.npy`
- `workspace_outputs/colorectal_nucleus_based/intermediate/square_002um_filtered_nuclear/colorectal_nucleus_based_square_002um_filtered_rl.manifest.json`

## 5) Build nuclei table (one row per nucleus)

For `colorectal_nucleus_based`, build nucleus centers directly from the
fractional nuclear ground-truth bins. The center is weighted by
`nuclear_fraction`, and the radius is derived from covered nuclear area.

```bash
python preprocessing/build_nuclei_table_from_nuclear_bins.py \
  --nuclear-bins-path workspace_outputs/colorectal_nucleus_based/intermediate/ground_truth/colorectal_nucleus_based_square_002um_nuclear_bins.csv.gz \
  --output-path workspace_outputs/colorectal_nucleus_based/intermediate/nuclei/colorectal_nucleus_based_nuclei.parquet \
  --summary-path workspace_outputs/colorectal_nucleus_based/intermediate/nuclei/colorectal_nucleus_based_nuclei.summary.json
```
Expected outputs:
- `workspace_outputs/colorectal_nucleus_based/intermediate/nuclei/colorectal_nucleus_based_nuclei.parquet`
- `workspace_outputs/colorectal_nucleus_based/intermediate/nuclei/colorectal_nucleus_based_nuclei.summary.json`

## 6) Reuse the independent scRNA expression reference

Do not build the expression reference from the current pseudo-HD target. The
episode and PPO configs use the existing colorectal scRNA reference:

`workspace_outputs/pseudo_human_colorectal/intermediate/reference_sct/reference_counts_sct_tumor_aligned_hd_overlap_unique.npz`

It contains 38 cell types and 15,143 HD-overlap genes. This reference supplies
`theta[k,g]` and therefore `LL[b,k]` for both RL and generalized EM.

## 7) Reuse the existing shape prior

Do not fit the shape model from the current target assignments. The PPO configs
reuse:

`workspace_outputs/pseudo_human_colorectal/intermediate/reference_shape/tsne/human_colorectal_gt_shape_reference.shape_prior_model.npz`

Shape is a morphology reference rather than an scRNA count matrix. Keeping this
model fixed prevents the current target's whole-cell ground truth from entering
inference.

## 8) Build episodes
This step creates per-cell episode artifacts with candidate bins, geometry, and matrix references used by reward computation.
It is the main preprocessing stage before training or evaluation.

Run episode build only:
```bash
sbatch jobs/preprocessing/run_episode_build.sbatch
```

Optional debug subset:
```bash
EPISODE_LIMIT_NUCLEI=800 sbatch jobs/preprocessing/run_episode_build.sbatch
```

Manual run:
```bash
python scripts/run_episode_build.py --config configs/episode_build.template.yaml
```

## 9) Train PPO / GRPO policy
The current preferred training job is full-GRPO on CPU. It uses `configs/ppo_training.full_grpo.yaml`, the current policy feature schema, the `w1 + w5` expression reward mixture, and the optional `w6` morphology shape-prior reward from `shape_prior.weight`.

Run with the latest episode build:
```bash
sbatch jobs/training/run_ppo_training_full_grpo_cpu.sbatch
```

Run with a specific episode build:
```bash
EP_RUN="runs/colorectal_nucleus_based_episode_build_YYYYMMDDTHHMMSSZ"
sbatch --export=ALL,EPISODE_RUN_DIR="$EP_RUN" jobs/training/run_ppo_training_full_grpo_cpu.sbatch
```

Useful overrides:
```bash
sbatch --export=ALL,EPISODE_RUN_DIR="$EP_RUN",BATCH_CELLS=200,MAX_UPDATES=600,RUN_NAME=colorectal_nucleus_based_full_grpo jobs/training/run_ppo_training_full_grpo_cpu.sbatch
```

Alternative training configs:
- `configs/ppo_training.template.yaml`: PPO config.
- `configs/ppo_training.grpo.yaml`: PPO with optional same-cell group-relative auxiliary.
- `configs/ppo_training.full_grpo.yaml`: current full-GRPO training path.

Expected outputs:
- `runs/colorectal_nucleus_based_full_grpo_*/checkpoints/best_model.pt`
- `runs/colorectal_nucleus_based_full_grpo_*/checkpoints/final_model.pt`
- `runs/colorectal_nucleus_based_full_grpo_*/summary.json`
- `runs/colorectal_nucleus_based_full_grpo_*/logs/steps.jsonl`

## 10) Evaluate PPO / GRPO checkpoint
This evaluates a trained policy on the episode set, writes per-cell metrics, overlays, IoU distribution plots, and optional gene-correlation metrics against pseudo GT single-cell expression.

Run:
```bash
CHECKPOINT_PATH="runs/colorectal_nucleus_based_full_grpo_YYYYMMDDTHHMMSSZ/checkpoints/best_model.pt"
sbatch --export=ALL,CHECKPOINT_PATH="$CHECKPOINT_PATH" jobs/evaluation/run_evaluate_ppo_checkpoint.sbatch
```

Common overrides:
```bash
sbatch --export=ALL,\
CHECKPOINT_PATH="$CHECKPOINT_PATH",\
MAX_EPISODES=300,\
POLICY_MODE=greedy,\
RUN_DEVICE=cpu,\
EVAL_SEED=7,\
OVERLAY_MAX_CELLS=300,\
OVERLAY_SELECTION=top_reward \
jobs/evaluation/run_evaluate_ppo_checkpoint.sbatch
```

Expected outputs:
- `runs/colorectal_nucleus_based_ppo_eval_*/summary.json`
- `runs/colorectal_nucleus_based_ppo_eval_*/per_episode.csv`
- `runs/colorectal_nucleus_based_ppo_eval_*/overlays/`
- IoU distribution/CDF plots in the evaluation run directory.

Notes:
- Set `CHECKPOINT_PATH` manually; do not rely on implicit checkpoint discovery for final comparisons.
- Use the same `PPO_EVAL_RUN_DIR` from this step when comparing Bin2Cell, SMURF, and STCS so all methods use the same cell set.

## 11) Interactive patch / PPO debug app
The default React + TypeScript view shows one complete evaluated patch: predicted owner assignment, matched GT outlines, owner-correct/wrong bins, patch reward/score, patch-level overlap metrics, and a sortable per-cell table.

Run on a compute node:
```bash
PATCH_EVAL_RUN_DIR="runs/colorectal_nucleus_based_patch_overfit4_eval_YYYYMMDDTHHMMSSZ"
sbatch --export=ALL,PPO_EVAL_RUN_DIR="$PATCH_EVAL_RUN_DIR",DEBUG_UI=patch jobs/apps/run_ppo_debug_app.sbatch
```

If an older patch evaluation does not yet contain `patch_debug/manifest.json`, the job backfills the patch JSON and static overview plots from the existing assignment/GT files. It does not rerun GPU rollout.

The previous per-cell step replay remains available:
```bash
PPO_EVAL_RUN_DIR="runs/colorectal_nucleus_based_ppo_eval_YYYYMMDDTHHMMSSZ"
sbatch --export=ALL,PPO_EVAL_RUN_DIR="$PPO_EVAL_RUN_DIR",DEBUG_UI=streamlit,RUN_DEVICE=cpu jobs/apps/run_ppo_debug_app.sbatch
```

The job prints a LAN URL and an SSH tunnel command. If direct LAN access is blocked, use the printed SSH tunnel and open `http://localhost:<port>`.

### Publish the latest patch evaluation to GitHub Pages

`main` keeps only source code. Generated frontend files and `runs/` remain ignored. The publisher builds the React app, copies one completed evaluation's `patch_debug/`, adds the static progress report under `/progress-report/`, creates a fresh root commit in a temporary repository, and force-pushes that snapshot to `gh-pages`. The reachable `gh-pages` history therefore contains one commit and one evaluation.

The overfit patch evaluation job publishes its exact output after a successful run by default:

```bash
sbatch jobs/evaluation/run_evaluate_patch_overfit4.sbatch
```

Disable publication for a private or provisional run with `PUBLISH_GITHUB_PAGES=false`. To republish an existing evaluation manually:

```bash
python scripts/publish_patch_debug_pages.py \
  --eval-run-dir runs/colorectal_nucleus_based_patch_overfit4_eval_YYYYMMDDTHHMMSSZ \
  --push
```

Before the first publication, configure the repository's Pages source as `gh-pages` at `/ (root)` and make sure the environment can authenticate `git push` to `origin`. The patch debugger is available at `https://haoyunli.github.io/HD-Cell-RL/`; the presentation is available at `https://haoyunli.github.io/HD-Cell-RL/progress-report/`.

## 12) Run Bin2Cell / SMURF / STCS baselines with PPO-aligned nuclei
These method jobs are configured to use the same PPO-aligned nuclear bins instead of each tool's independent nuclear segmentation whenever possible. This keeps method comparison on the same nuclear seed level.

All three method jobs can evaluate against the same PPO checkpoint-evaluation cell set by setting `PPO_EVAL_RUN_DIR`.

### Bin2Cell
```bash
PPO_EVAL_RUN_DIR="runs/colorectal_nucleus_based_ppo_eval_YYYYMMDDTHHMMSSZ"
sbatch --export=ALL,PPO_EVAL_RUN_DIR="$PPO_EVAL_RUN_DIR" jobs/baselines/run_bin2cell.sbatch
```

Key behavior:
- Uses `EXTERNAL_NUCLEAR_BINS_PATH` to inject PPO-aligned `labels_he`.
- Skips Bin2Cell's own H&E nuclear segmentation when external nuclei are provided.
- Still runs Bin2Cell expansion/GEX/combine steps.

Main outputs:
- `workspace_outputs/colorectal_nucleus_based/bin2cell_results/colorectal_nucleus_based_bin2cell_assignments.csv`
- `runs/colorectal_nucleus_based_bin2cell_eval_*/summary.json`
- `runs/colorectal_nucleus_based_bin2cell_eval_*/overlays/`

### SMURF
```bash
PPO_EVAL_RUN_DIR="runs/colorectal_nucleus_based_ppo_eval_YYYYMMDDTHHMMSSZ"
sbatch --export=ALL,PPO_EVAL_RUN_DIR="$PPO_EVAL_RUN_DIR" jobs/baselines/run_smurf.sbatch
```

Key behavior:
- Uses `EXTERNAL_NUCLEAR_BINS_PATH` for PPO-aligned nuclear seeds.
- Writes PPO-format assignment outputs and evaluation summaries.

Main outputs:
- `workspace_outputs/colorectal_nucleus_based/smurf_results/colorectal_nucleus_based_smurf_assignments.csv`
- `runs/colorectal_nucleus_based_smurf_eval_*/summary.json`
- `runs/colorectal_nucleus_based_smurf_eval_*/overlays/`

### STCS
Set up the STCS environment once:
```bash
sbatch jobs/baselines/setup_stcs_env.sbatch
```

Then run STCS:
```bash
PPO_EVAL_RUN_DIR="runs/colorectal_nucleus_based_ppo_eval_YYYYMMDDTHHMMSSZ"
sbatch --export=ALL,PPO_EVAL_RUN_DIR="$PPO_EVAL_RUN_DIR" jobs/baselines/run_stcs.sbatch
```

Key behavior:
- Uses `EXTERNAL_NUCLEAR_BINS_PATH` for PPO-aligned nuclear seeds.
- By default uses `RESTRICT_TO_EVAL_CELLS=true` and `EVAL_CONTEXT_RADIUS_BINS=10` to reduce memory while preserving nearby competitor nuclei.

Main outputs:
- `workspace_outputs/colorectal_nucleus_based/stcs_results/colorectal_nucleus_based_stcs_assignments.csv`
- `runs/colorectal_nucleus_based_stcs_eval_*/summary.json`
- `runs/colorectal_nucleus_based_stcs_eval_*/overlays/`

## 13) Re-run method evaluation only
If Bin2Cell, SMURF, or STCS assignments already exist, use this job to re-run only PPO-format evaluation without re-running the full method pipeline.

Run all available methods:
```bash
PPO_EVAL_RUN_DIR="runs/colorectal_nucleus_based_ppo_eval_YYYYMMDDTHHMMSSZ"
sbatch --export=ALL,PPO_EVAL_RUN_DIR="$PPO_EVAL_RUN_DIR" jobs/evaluation/run_method_ppo_eval_only.sbatch
```

Run one method only:
```bash
sbatch --export=ALL,\
PPO_EVAL_RUN_DIR="$PPO_EVAL_RUN_DIR",\
RUN_BIN2CELL=true,\
RUN_SMURF=false,\
RUN_STCS=false \
jobs/evaluation/run_method_ppo_eval_only.sbatch
```

The shared evaluator writes consistent fields across PPO, Bin2Cell, SMURF, and STCS:
- matched-cell count
- IoU = intersection / union
- precision = intersection / predicted bins
- recall = intersection / GT bins
- Dice
- size ratio
- per-cell and summary gene-correlation metrics when GT single-cell expression is provided
- overlay plots with predicted bins and GT outline
