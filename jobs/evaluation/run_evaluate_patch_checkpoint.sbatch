#!/bin/bash
#SBATCH --job-name=EvalPatchRL
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --output=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL/logs/slurm/%x-%j.out
#SBATCH --error=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL/logs/slurm/%x-%j.err

set -euo pipefail

module load anaconda3

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL"
cd "$WORK_DIR"

source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$WORK_DIR/Bin2Cell_Validation"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCH_NUM_THREADS=32
export TORCH_NUM_INTEROP_THREADS=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pycache_${SLURM_JOB_ID:-$$}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_${SLURM_JOB_ID:-$$}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache_${SLURM_JOB_ID:-$$}}"
mkdir -p "$PYTHONPYCACHEPREFIX" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# Paste the patch-trained best checkpoint here before submitting.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-runs/human_colorectal_patch_full_grpo_20260520T203706Z/checkpoints/best_model.pt}"
PPO_EVAL_RUN_DIR="${PPO_EVAL_RUN_DIR:-runs/human_colorectal_ppo_eval_20260511T153658Z}"
PATCHES_INDEX_PATH="${PATCHES_INDEX_PATH:-runs/human_colorectal_patch_episode_build/human_colorectal_patches_index.csv}"
BASE_CONFIG="${BASE_CONFIG:-}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/human_colorectal_patch_eval_intermediate}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-runs}"
EVAL_RUN_NAME="${EVAL_RUN_NAME:-human_colorectal_patch_eval}"
POLICY_MODE="${POLICY_MODE:-greedy}"
RUN_DEVICE="${RUN_DEVICE:-cpu}"
MAX_TARGET_CELLS="${MAX_TARGET_CELLS:-0}"
EVAL_MODE="${EVAL_MODE:-random_patches}"
MAX_EVAL_PATCHES="${MAX_EVAL_PATCHES:-15}"
OVERLAY_MAX_CELLS="${OVERLAY_MAX_CELLS:-1000}"
OVERLAY_SELECTION="${OVERLAY_SELECTION:-first}"
EVAL_SEED="${EVAL_SEED:-7}"
GT_CELL_BINS_PATH="${GT_CELL_BINS_PATH:-workspace_outputs/pseudo_human_colorectal/intermediate/cropped_visium_hd_human_colorectal_pseudo_square_002um_cell_bins.csv.gz}"
GT_NUCLEAR_BINS_PATH="${GT_NUCLEAR_BINS_PATH:-workspace_outputs/pseudo_human_colorectal/intermediate/cropped_visium_hd_human_colorectal_pseudo_square_002um_nuclear_bins.csv.gz}"
GT_CELL_ASSIGNMENTS_CSV="${GT_CELL_ASSIGNMENTS_CSV:-colorectal_pseudo_visium_hd_output_full_0.25/ground_truth_cell_assignments.csv}"
GT_SC_EXPRESSION_H5="${GT_SC_EXPRESSION_H5:-colorectal_sc_data/CRC_GSE166555_raw_tumor.h5}"
EXTERNAL_NUCLEAR_BINS_PATH="${EXTERNAL_NUCLEAR_BINS_PATH:-workspace_outputs/pseudo_human_colorectal/intermediate/cropped_visium_hd_human_colorectal_pseudo_square_002um_nuclear_bins.csv.gz}"

if [[ -z "$CHECKPOINT_PATH" ]]; then
  echo "CHECKPOINT_PATH is required." >&2
  echo "Paste it into jobs/evaluation/run_evaluate_patch_checkpoint.sbatch or submit with:" >&2
  echo "  sbatch --export=ALL,CHECKPOINT_PATH=runs/<patch_train_run>/checkpoints/best_model.pt jobs/evaluation/run_evaluate_patch_checkpoint.sbatch" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint file not found: $CHECKPOINT_PATH" >&2
  exit 1
fi

echo "CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "PPO_EVAL_RUN_DIR: $PPO_EVAL_RUN_DIR"
echo "PATCHES_INDEX_PATH: $PATCHES_INDEX_PATH"
echo "BASE_CONFIG: ${BASE_CONFIG:-<from checkpoint>}"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EVAL_OUTPUT_ROOT: $EVAL_OUTPUT_ROOT"
echo "EVAL_RUN_NAME: $EVAL_RUN_NAME"
echo "POLICY_MODE: $POLICY_MODE"
echo "RUN_DEVICE: $RUN_DEVICE"
echo "MAX_TARGET_CELLS: $MAX_TARGET_CELLS"
echo "EVAL_MODE: $EVAL_MODE"
echo "MAX_EVAL_PATCHES: $MAX_EVAL_PATCHES"
echo "OVERLAY_MAX_CELLS: $OVERLAY_MAX_CELLS"
echo "OVERLAY_SELECTION: $OVERLAY_SELECTION"

CMD=(python scripts/evaluate_patch_checkpoint.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --patches-index-path "$PATCHES_INDEX_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --ppo_eval_run_dir "$PPO_EVAL_RUN_DIR" \
  --eval_run_name "$EVAL_RUN_NAME" \
  --eval_output_root "$EVAL_OUTPUT_ROOT" \
  --policy-mode "$POLICY_MODE" \
  --device "$RUN_DEVICE" \
  --max-target-cells "$MAX_TARGET_CELLS" \
  --eval-mode "$EVAL_MODE" \
  --max-eval-patches "$MAX_EVAL_PATCHES" \
  --overlay_max_cells "$OVERLAY_MAX_CELLS" \
  --overlay_selection "$OVERLAY_SELECTION" \
  --eval_seed "$EVAL_SEED" \
  --external_nuclear_bins_path "$EXTERNAL_NUCLEAR_BINS_PATH" \
  --gt_cell_bins_path "$GT_CELL_BINS_PATH" \
  --gt_nuclear_bins_path "$GT_NUCLEAR_BINS_PATH" \
  --gt_cell_assignments_csv "$GT_CELL_ASSIGNMENTS_CSV" \
  --gt_sc_expression_h5 "$GT_SC_EXPRESSION_H5")

if [ -n "$BASE_CONFIG" ]; then
  CMD+=(--base-config "$BASE_CONFIG")
fi

"${CMD[@]}"

conda deactivate
