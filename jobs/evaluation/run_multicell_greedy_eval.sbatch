#!/bin/bash
#SBATCH --job-name=MultiGreedy
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --output=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL/logs/slurm/%x-%j.out
#SBATCH --error=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL/logs/slurm/%x-%j.err

set -euo pipefail

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL"
cd "$WORK_DIR"

PYTHON_BIN="${PYTHON_BIN:-$WORK_DIR/Bin2Cell_Validation/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_${SLURM_JOB_ID:-$$}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache_${SLURM_JOB_ID:-$$}}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

PPO_EVAL_RUN_DIR="${PPO_EVAL_RUN_DIR:-runs/human_colorectal_ppo_eval_20260511T153658Z}"
OUTPUT_DIR="${OUTPUT_DIR:-workspace_outputs/pseudo_human_colorectal/multicell_greedy}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-runs}"
EVAL_RUN_NAME="${EVAL_RUN_NAME:-human_colorectal_multicell_greedy_eval}"
CONTEXT_RADIUS_UM="${CONTEXT_RADIUS_UM:-50}"
MAX_PATCH_CELLS="${MAX_PATCH_CELLS:-24}"
MAX_STEPS_PER_PATCH="${MAX_STEPS_PER_PATCH:-1000}"
MIN_ADD_SCORE="${MIN_ADD_SCORE:-0.0}"
MAX_TARGET_CELLS="${MAX_TARGET_CELLS:-0}"
OVERLAY_MAX_CELLS="${OVERLAY_MAX_CELLS:-300}"
OVERLAY_SELECTION="${OVERLAY_SELECTION:-first}"
EVAL_SEED="${EVAL_SEED:-7}"
PRED_MIN_NUCLEAR_OVERLAP_FRAC="${PRED_MIN_NUCLEAR_OVERLAP_FRAC:-0.3}"
PRED_MIN_NUCLEAR_OVERLAP_BINS="${PRED_MIN_NUCLEAR_OVERLAP_BINS:-2}"
GT_CELL_BINS_PATH="${GT_CELL_BINS_PATH:-workspace_outputs/pseudo_human_colorectal/intermediate/cropped_visium_hd_human_colorectal_pseudo_square_002um_cell_bins.csv.gz}"
GT_NUCLEAR_BINS_PATH="${GT_NUCLEAR_BINS_PATH:-workspace_outputs/pseudo_human_colorectal/intermediate/cropped_visium_hd_human_colorectal_pseudo_square_002um_nuclear_bins.csv.gz}"
GT_CELL_ASSIGNMENTS_CSV="${GT_CELL_ASSIGNMENTS_CSV:-colorectal_pseudo_visium_hd_output_full_0.25/ground_truth_cell_assignments.csv}"
GT_SC_EXPRESSION_H5="${GT_SC_EXPRESSION_H5:-colorectal_sc_data/CRC_GSE166555_raw_tumor.h5}"
GT_MIN_NUCLEAR_OVERLAP_FRAC="${GT_MIN_NUCLEAR_OVERLAP_FRAC:-0.3}"
GT_MIN_NUCLEAR_OVERLAP_BINS="${GT_MIN_NUCLEAR_OVERLAP_BINS:-2}"
EXTERNAL_NUCLEAR_BINS_PATH="${EXTERNAL_NUCLEAR_BINS_PATH:-workspace_outputs/pseudo_human_colorectal/intermediate/cropped_visium_hd_human_colorectal_pseudo_square_002um_nuclear_bins.csv.gz}"

echo "Working directory: $WORK_DIR"
echo "Python: $PYTHON_BIN"
echo "PPO_EVAL_RUN_DIR: $PPO_EVAL_RUN_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EVAL_OUTPUT_ROOT: $EVAL_OUTPUT_ROOT"
echo "EVAL_RUN_NAME: $EVAL_RUN_NAME"
echo "CONTEXT_RADIUS_UM: $CONTEXT_RADIUS_UM"
echo "MAX_PATCH_CELLS: $MAX_PATCH_CELLS"
echo "MAX_STEPS_PER_PATCH: $MAX_STEPS_PER_PATCH"
echo "MIN_ADD_SCORE: $MIN_ADD_SCORE"
echo "MAX_TARGET_CELLS: $MAX_TARGET_CELLS"
echo "OVERLAY_MAX_CELLS: $OVERLAY_MAX_CELLS"
echo "OVERLAY_SELECTION: $OVERLAY_SELECTION"
echo "GT_CELL_BINS_PATH: $GT_CELL_BINS_PATH"
echo "GT_NUCLEAR_BINS_PATH: $GT_NUCLEAR_BINS_PATH"
echo "GT_CELL_ASSIGNMENTS_CSV: $GT_CELL_ASSIGNMENTS_CSV"
echo "GT_SC_EXPRESSION_H5: $GT_SC_EXPRESSION_H5"

"$PYTHON_BIN" scripts/run_multicell_greedy_eval.py \
  --ppo_eval_run_dir "$PPO_EVAL_RUN_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --context_radius_um "$CONTEXT_RADIUS_UM" \
  --max_patch_cells "$MAX_PATCH_CELLS" \
  --max_steps_per_patch "$MAX_STEPS_PER_PATCH" \
  --min_add_score "$MIN_ADD_SCORE" \
  --max_target_cells "$MAX_TARGET_CELLS" \
  --external_nuclear_bins_path "$EXTERNAL_NUCLEAR_BINS_PATH" \
  --eval_run_name "$EVAL_RUN_NAME" \
  --eval_output_root "$EVAL_OUTPUT_ROOT" \
  --overlay_max_cells "$OVERLAY_MAX_CELLS" \
  --overlay_selection "$OVERLAY_SELECTION" \
  --eval_seed "$EVAL_SEED" \
  --pred_min_nuclear_overlap_frac "$PRED_MIN_NUCLEAR_OVERLAP_FRAC" \
  --pred_min_nuclear_overlap_bins "$PRED_MIN_NUCLEAR_OVERLAP_BINS" \
  --gt_cell_bins_path "$GT_CELL_BINS_PATH" \
  --gt_nuclear_bins_path "$GT_NUCLEAR_BINS_PATH" \
  --gt_cell_assignments_csv "$GT_CELL_ASSIGNMENTS_CSV" \
  --gt_sc_expression_h5 "$GT_SC_EXPRESSION_H5" \
  --gt_min_nuclear_overlap_frac "$GT_MIN_NUCLEAR_OVERLAP_FRAC" \
  --gt_min_nuclear_overlap_bins "$GT_MIN_NUCLEAR_OVERLAP_BINS"

echo "Multi-cell greedy evaluation complete at $(date)"
