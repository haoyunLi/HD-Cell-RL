# HD Cell RL Runbook

This runbook lists the execution order for preprocessing, episode building, reward search, PPO/GRPO training, checkpoint evaluation, and method-baseline comparison.
Each step depends on outputs from earlier steps, so run in order unless you already have the required files.

Repository roles and placement rules are documented in [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md). Slurm entry points are grouped by purpose in [`jobs/`](jobs/README.md), and scheduler output is written to `logs/slurm/`.

Current main training path:
- Build episodes once.
- Train with `jobs/training/run_ppo_training_full_grpo_cpu.sbatch`.
- Evaluate checkpoints with `jobs/evaluation/run_evaluate_ppo_checkpoint.sbatch`.
- Compare Bin2Cell, SMURF, and STCS using the same PPO-evaluation cell set and the same PPO-aligned nuclear seeds.

Current ADD reward expression mixture:
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
- `workspace_outputs/human_colorectal/intermediate/cropped_visium_hd_human_Colorectal.png`

## 2) Run Cellpose nuclear segmentation
This step runs nucleus segmentation on the cropped image and writes cell/nucleus masks and related artifacts.
These segmentation results are later mapped to official 2um bins.
```bash
sbatch jobs/preprocessing/run_cellpose_sam.sbatch
```
Expected output folder:
- `workspace_outputs/human_colorectal/intermediate/cellpose_sam_human_colorectal_output/`

## 3) Map pixel nuclear signal to official `square_002um` bins
This step converts pixel-level nuclear assignments into official Visium HD `square_002um` bin coordinates.
It also writes an overlay for quick visual QC.
```bash
sbatch jobs/preprocessing/run_pixel_to_square_002um_bins.sbatch
```
Expected outputs:
- `workspace_outputs/human_colorectal/intermediate/cropped_visium_hd_human_Colorectal_square_002um_nuclear_bins.csv.gz`
- `workspace_outputs/human_colorectal/intermediate/cropped_visium_hd_human_Colorectal_square_002um_nuclear_bins.summary.json`
- `workspace_outputs/human_colorectal/intermediate/cropped_visium_hd_human_Colorectal_square_002um_nuclear_bins_overlay.png`

## 4) Merge filtered expression metadata + nuclear annotation
This step merges official HD filtered-bin metadata/expression indexing with nuclear-bin annotation into one RL-ready table.
The expression source can be a 10x `.h5` file or a `filtered_feature_bc_matrix/` directory.
The merged table is the main structural input for episode construction.
```bash
sbatch jobs/preprocessing/run_merge_square_002um_expression_with_nuclear.sbatch
```
Expected outputs:
- `workspace_outputs/human_colorectal/intermediate/square_002um_filtered_nuclear/human_colorectal_square_002um_filtered_rl.metadata.parquet`
- `workspace_outputs/human_colorectal/intermediate/square_002um_filtered_nuclear/human_colorectal_square_002um_filtered_rl.selected_features.tsv.gz`
- `workspace_outputs/human_colorectal/intermediate/square_002um_filtered_nuclear/human_colorectal_square_002um_filtered_rl.selected_feature_indices.npy`
- `workspace_outputs/human_colorectal/intermediate/square_002um_filtered_nuclear/human_colorectal_square_002um_filtered_rl.manifest.json`

## 5) Build nuclei table (one row per nucleus)
This step builds one nucleus-center record per cell (`center_x_um`, `center_y_um`) from segmentation outputs.
These centers are used by reward distance and overlap terms.
```bash
sbatch jobs/preprocessing/run_build_nuclei_table.sbatch
```
Expected outputs:
- `workspace_outputs/human_colorectal/intermediate/human_colorectal_nuclei.parquet`
- `workspace_outputs/human_colorectal/intermediate/human_colorectal_nuclei.summary.json`

## 6) Build aligned scRNA reference counts (`C[k,g]`)
This step builds reference counts by cell type on the HD-overlap gene set, which drives posterior-based expression reward.
It also writes the selected HD gene allowlist used for alignment.
```bash
sbatch jobs/preprocessing/run_build_reference_counts_sct.sbatch
```
Expected outputs:
- `workspace_outputs/human_colorectal/intermediate/reference_sct/hd_selected_feature_names_unique.txt`
- `workspace_outputs/human_colorectal/intermediate/reference_sct/reference_counts_sct_tumor_aligned_hd_overlap_unique.npz`
- `workspace_outputs/human_colorectal/intermediate/reference_sct/reference_counts_sct_tumor_aligned_hd_overlap_unique.summary.json`

## 7) Build GT shape reference features
This step builds per-cell GT shape features from pseudo whole-cell square_002um bins. It keeps `cell_type` labels from `ground_truth_cell_assignments.csv` and writes both raw and z-scored features.

Run:
```bash
sbatch jobs/preprocessing/run_build_reference_shape_features_gt.sbatch
```

Expected outputs:
- `workspace_outputs/pseudo_human_colorectal/intermediate/reference_shape/human_colorectal_gt_shape_reference.per_cell.csv.gz`
- `workspace_outputs/pseudo_human_colorectal/intermediate/reference_shape/human_colorectal_gt_shape_reference.cell_type_summary.csv`
- `workspace_outputs/pseudo_human_colorectal/intermediate/reference_shape/human_colorectal_gt_shape_reference.npz`
- `workspace_outputs/pseudo_human_colorectal/intermediate/reference_shape/human_colorectal_gt_shape_reference.summary.json`

Features:
- `log_area = log(area + 1)` where area is the number of unique assigned 2 um bins.
- `compactness = 4 * pi * area / perimeter^2`, using exposed 4-neighbor grid edges for perimeter.
- `solidity = area / convex_hull_area`, with small or failed hulls set to 1.0.
- `anisotropy = lambda1 / lambda2` from the coordinate covariance eigenvalues.

Build morphology clusters and the Gaussian shape-prior model:
```bash
sbatch jobs/preprocessing/run_plot_shape_reference_tsne.sbatch
sbatch jobs/preprocessing/run_build_shape_prior_model.sbatch
```

The cluster step writes `shape_cluster` labels from unsupervised shape features, not biological `cell_type`. The model step fits one Gaussian per shape cluster and writes:
- `workspace_outputs/pseudo_human_colorectal/intermediate/reference_shape/tsne/human_colorectal_gt_shape_reference.clustered_reference.csv.gz`
- `workspace_outputs/pseudo_human_colorectal/intermediate/reference_shape/tsne/human_colorectal_gt_shape_reference.shape_prior_model.npz`
- `workspace_outputs/pseudo_human_colorectal/intermediate/reference_shape/tsne/human_colorectal_gt_shape_reference.shape_prior_model.summary.json`

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
EP_RUN="runs/human_colorectal_episode_build_YYYYMMDDTHHMMSSZ"
sbatch --export=ALL,EPISODE_RUN_DIR="$EP_RUN" jobs/training/run_ppo_training_full_grpo_cpu.sbatch
```

Useful overrides:
```bash
sbatch --export=ALL,EPISODE_RUN_DIR="$EP_RUN",BATCH_CELLS=200,MAX_UPDATES=600,RUN_NAME=human_colorectal_full_grpo jobs/training/run_ppo_training_full_grpo_cpu.sbatch
```

Alternative training configs:
- `configs/ppo_training.template.yaml`: PPO config.
- `configs/ppo_training.grpo.yaml`: PPO with optional same-cell group-relative auxiliary.
- `configs/ppo_training.full_grpo.yaml`: current full-GRPO training path.

Expected outputs:
- `runs/human_colorectal_full_grpo_*/checkpoints/best_model.pt`
- `runs/human_colorectal_full_grpo_*/checkpoints/final_model.pt`
- `runs/human_colorectal_full_grpo_*/summary.json`
- `runs/human_colorectal_full_grpo_*/logs/steps.jsonl`

## 10) Evaluate PPO / GRPO checkpoint
This evaluates a trained policy on the episode set, writes per-cell metrics, overlays, IoU distribution plots, and optional gene-correlation metrics against pseudo GT single-cell expression.

Run:
```bash
CHECKPOINT_PATH="runs/human_colorectal_full_grpo_YYYYMMDDTHHMMSSZ/checkpoints/best_model.pt"
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
- `runs/human_colorectal_ppo_eval_*/summary.json`
- `runs/human_colorectal_ppo_eval_*/per_episode.csv`
- `runs/human_colorectal_ppo_eval_*/overlays/`
- IoU distribution/CDF plots in the evaluation run directory.

Notes:
- Set `CHECKPOINT_PATH` manually; do not rely on implicit checkpoint discovery for final comparisons.
- Use the same `PPO_EVAL_RUN_DIR` from this step when comparing Bin2Cell, SMURF, and STCS so all methods use the same cell set.

## 11) Interactive patch / PPO debug app
The default React + TypeScript view shows one complete evaluated patch: predicted owner assignment, matched GT outlines, owner-correct/wrong bins, patch reward/score, patch-level overlap metrics, and a sortable per-cell table.

Run on a compute node:
```bash
PATCH_EVAL_RUN_DIR="runs/human_colorectal_patch_overfit4_eval_YYYYMMDDTHHMMSSZ"
sbatch --export=ALL,PPO_EVAL_RUN_DIR="$PATCH_EVAL_RUN_DIR",DEBUG_UI=patch jobs/apps/run_ppo_debug_app.sbatch
```

If an older patch evaluation does not yet contain `patch_debug/manifest.json`, the job backfills the patch JSON and static overview plots from the existing assignment/GT files. It does not rerun GPU rollout.

The previous per-cell step replay remains available:
```bash
PPO_EVAL_RUN_DIR="runs/human_colorectal_ppo_eval_YYYYMMDDTHHMMSSZ"
sbatch --export=ALL,PPO_EVAL_RUN_DIR="$PPO_EVAL_RUN_DIR",DEBUG_UI=streamlit,RUN_DEVICE=cpu jobs/apps/run_ppo_debug_app.sbatch
```

The job prints a LAN URL and an SSH tunnel command. If direct LAN access is blocked, use the printed SSH tunnel and open `http://localhost:<port>`.

## 12) Run Bin2Cell / SMURF / STCS baselines with PPO-aligned nuclei
These method jobs are configured to use the same PPO-aligned nuclear bins instead of each tool's independent nuclear segmentation whenever possible. This keeps method comparison on the same nuclear seed level.

All three method jobs can evaluate against the same PPO checkpoint-evaluation cell set by setting `PPO_EVAL_RUN_DIR`.

### Bin2Cell
```bash
PPO_EVAL_RUN_DIR="runs/human_colorectal_ppo_eval_YYYYMMDDTHHMMSSZ"
sbatch --export=ALL,PPO_EVAL_RUN_DIR="$PPO_EVAL_RUN_DIR" jobs/baselines/run_bin2cell.sbatch
```

Key behavior:
- Uses `EXTERNAL_NUCLEAR_BINS_PATH` to inject PPO-aligned `labels_he`.
- Skips Bin2Cell's own H&E nuclear segmentation when external nuclei are provided.
- Still runs Bin2Cell expansion/GEX/combine steps.

Main outputs:
- `workspace_outputs/pseudo_human_colorectal/bin2cell_results_colorectal_0.25/human_colorectal_bin2cell_assignments.csv`
- `runs/human_colorectal_bin2cell_eval_*/summary.json`
- `runs/human_colorectal_bin2cell_eval_*/overlays/`

### SMURF
```bash
PPO_EVAL_RUN_DIR="runs/human_colorectal_ppo_eval_YYYYMMDDTHHMMSSZ"
sbatch --export=ALL,PPO_EVAL_RUN_DIR="$PPO_EVAL_RUN_DIR" jobs/baselines/run_smurf.sbatch
```

Key behavior:
- Uses `EXTERNAL_NUCLEAR_BINS_PATH` for PPO-aligned nuclear seeds.
- Writes PPO-format assignment outputs and evaluation summaries.

Main outputs:
- `workspace_outputs/pseudo_human_colorectal/smurf_results_colorectal_0.25/human_colorectal_smurf_assignments.csv`
- `runs/human_colorectal_smurf_eval_*/summary.json`
- `runs/human_colorectal_smurf_eval_*/overlays/`

### STCS
Set up the STCS environment once:
```bash
sbatch jobs/baselines/setup_stcs_env.sbatch
```

Then run STCS:
```bash
PPO_EVAL_RUN_DIR="runs/human_colorectal_ppo_eval_YYYYMMDDTHHMMSSZ"
sbatch --export=ALL,PPO_EVAL_RUN_DIR="$PPO_EVAL_RUN_DIR" jobs/baselines/run_stcs.sbatch
```

Key behavior:
- Uses `EXTERNAL_NUCLEAR_BINS_PATH` for PPO-aligned nuclear seeds.
- By default uses `RESTRICT_TO_EVAL_CELLS=true` and `EVAL_CONTEXT_RADIUS_BINS=10` to reduce memory while preserving nearby competitor nuclei.

Main outputs:
- `workspace_outputs/pseudo_human_colorectal/stcs_results_colorectal_0.25/human_colorectal_stcs_assignments.csv`
- `runs/human_colorectal_stcs_eval_*/summary.json`
- `runs/human_colorectal_stcs_eval_*/overlays/`

## 13) Re-run method evaluation only
If Bin2Cell, SMURF, or STCS assignments already exist, use this job to re-run only PPO-format evaluation without re-running the full method pipeline.

Run all available methods:
```bash
PPO_EVAL_RUN_DIR="runs/human_colorectal_ppo_eval_YYYYMMDDTHHMMSSZ"
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
