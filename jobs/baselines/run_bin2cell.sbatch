#!/bin/bash
#SBATCH --job-name=bin2cell
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=800G
#SBATCH --output=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL/logs/slurm/%x-%j.out
#SBATCH --error=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL/logs/slurm/%x-%j.err

set -euo pipefail

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL"
cd "$WORK_DIR"

echo "Working directory: $WORK_DIR"
echo "Activating bin2cell virtual environment..."
source "$WORK_DIR/bin2cell_env/bin/activate"

THREADS="${SLURM_CPUS_PER_TASK:-70}"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"

PSEUDO_HD_DIR="${PSEUDO_HD_DIR:-colorectal_nucleus_based}"
IMAGE_PATH="${IMAGE_PATH:-workspace_outputs/colorectal_nucleus_based/intermediate/cropped_visium_hd_colorectal_nucleus_based.png}"
OUTPUT_DIR="${OUTPUT_DIR:-workspace_outputs/colorectal_nucleus_based/bin2cell_results}"
DATASET_NAME="${DATASET_NAME:-colorectal_nucleus_based}"
MPP="${MPP:-0.2737012522439323}"
PROB_THRESH_HE="${PROB_THRESH_HE:-0.4}"
PROB_THRESH_GEX="${PROB_THRESH_GEX:-0.05}"
NMS_THRESH_GEX="${NMS_THRESH_GEX:-0.5}"
MAX_BIN_DISTANCE="${MAX_BIN_DISTANCE:-5}"
GAUSSIAN_SIGMA="${GAUSSIAN_SIGMA:-5.0}"
EXTERNAL_NUCLEAR_BINS_PATH="${EXTERNAL_NUCLEAR_BINS_PATH:-workspace_outputs/colorectal_nucleus_based/intermediate/nuclei/colorectal_nucleus_based_square_002um_nuclear_bins.csv.gz}"

PPO_EVAL_RUN_DIR="${PPO_EVAL_RUN_DIR:-}"
EVAL_RUN_NAME="${EVAL_RUN_NAME:-colorectal_nucleus_based_bin2cell_eval}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-runs}"
OVERLAY_MAX_CELLS="${OVERLAY_MAX_CELLS:-300}"
OVERLAY_SELECTION="${OVERLAY_SELECTION:-first}"
EVAL_SEED="${EVAL_SEED:-7}"
PRED_MIN_NUCLEAR_OVERLAP_FRAC="${PRED_MIN_NUCLEAR_OVERLAP_FRAC:-0.3}"
PRED_MIN_NUCLEAR_OVERLAP_BINS="${PRED_MIN_NUCLEAR_OVERLAP_BINS:-2}"
GT_CELL_BINS_PATH="${GT_CELL_BINS_PATH:-workspace_outputs/colorectal_nucleus_based/intermediate/ground_truth/colorectal_nucleus_based_square_002um_cell_bins.csv.gz}"
GT_NUCLEAR_BINS_PATH="${GT_NUCLEAR_BINS_PATH:-workspace_outputs/colorectal_nucleus_based/intermediate/ground_truth/colorectal_nucleus_based_square_002um_nuclear_bins.csv.gz}"
GT_CELL_ASSIGNMENTS_CSV="${GT_CELL_ASSIGNMENTS_CSV:-colorectal_nucleus_based/ground_truth_cell_assignments.csv}"
GT_SC_EXPRESSION_H5="${GT_SC_EXPRESSION_H5:-../Bin2Cell_Validation/outputs/ground_truth/colorectal_nucleus_based/simulation/scdesign3/scdesign3_synthetic_counts.h5}"
GT_MIN_NUCLEAR_OVERLAP_FRAC="${GT_MIN_NUCLEAR_OVERLAP_FRAC:-0.3}"
GT_MIN_NUCLEAR_OVERLAP_BINS="${GT_MIN_NUCLEAR_OVERLAP_BINS:-2}"

if [[ -z "$PPO_EVAL_RUN_DIR" ]]; then
  PPO_EVAL_RUN_DIR=$(ls -dt runs/colorectal_nucleus_based_ppo_eval_* 2>/dev/null | head -n 1 || true)
fi
if [[ ! -d "$PPO_EVAL_RUN_DIR" ]]; then
  echo "Current nucleus-based PPO eval run not found. Set PPO_EVAL_RUN_DIR." >&2
  exit 1
fi

echo "PSEUDO_HD_DIR: $PSEUDO_HD_DIR"
echo "IMAGE_PATH: $IMAGE_PATH"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "DATASET_NAME: $DATASET_NAME"
echo "MPP: $MPP"
echo "PROB_THRESH_HE: $PROB_THRESH_HE"
echo "PROB_THRESH_GEX: $PROB_THRESH_GEX"
echo "NMS_THRESH_GEX: $NMS_THRESH_GEX"
echo "MAX_BIN_DISTANCE: $MAX_BIN_DISTANCE"
echo "GAUSSIAN_SIGMA: $GAUSSIAN_SIGMA"
echo "EXTERNAL_NUCLEAR_BINS_PATH: $EXTERNAL_NUCLEAR_BINS_PATH"
echo "PPO_EVAL_RUN_DIR: $PPO_EVAL_RUN_DIR"
echo "EVAL_RUN_NAME: $EVAL_RUN_NAME"
echo "EVAL_OUTPUT_ROOT: $EVAL_OUTPUT_ROOT"
echo "OVERLAY_MAX_CELLS: $OVERLAY_MAX_CELLS"
echo "OVERLAY_SELECTION: $OVERLAY_SELECTION"
echo "EVAL_SEED: $EVAL_SEED"
echo "PRED_MIN_NUCLEAR_OVERLAP_FRAC: $PRED_MIN_NUCLEAR_OVERLAP_FRAC"
echo "PRED_MIN_NUCLEAR_OVERLAP_BINS: $PRED_MIN_NUCLEAR_OVERLAP_BINS"
echo "GT_CELL_BINS_PATH: $GT_CELL_BINS_PATH"
echo "GT_NUCLEAR_BINS_PATH: $GT_NUCLEAR_BINS_PATH"
echo "GT_CELL_ASSIGNMENTS_CSV: $GT_CELL_ASSIGNMENTS_CSV"
echo "GT_SC_EXPRESSION_H5: $GT_SC_EXPRESSION_H5"
echo "GT_MIN_NUCLEAR_OVERLAP_FRAC: $GT_MIN_NUCLEAR_OVERLAP_FRAC"
echo "GT_MIN_NUCLEAR_OVERLAP_BINS: $GT_MIN_NUCLEAR_OVERLAP_BINS"

python preprocessing/run_bin2cell_analysis.py \
  --pseudo_hd_dir "$PSEUDO_HD_DIR" \
  --image_path "$IMAGE_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --dataset_name "$DATASET_NAME" \
  --mpp "$MPP" \
  --prob_thresh_he "$PROB_THRESH_HE" \
  --prob_thresh_gex "$PROB_THRESH_GEX" \
  --nms_thresh_gex "$NMS_THRESH_GEX" \
  --max_bin_distance "$MAX_BIN_DISTANCE" \
  --gaussian_sigma "$GAUSSIAN_SIGMA" \
  --external_nuclear_bins_path "$EXTERNAL_NUCLEAR_BINS_PATH" \
  --ppo_eval_run_dir "$PPO_EVAL_RUN_DIR" \
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

echo ""
echo "Bin2cell analysis completed at $(date)"
