#!/bin/bash
#SBATCH --job-name=BuildRefShapeGT
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --mem=800G
#SBATCH --output=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL/logs/slurm/%x-%j.out
#SBATCH --error=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL/logs/slurm/%x-%j.err

set -euo pipefail

module load anaconda3

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL"
cd "$WORK_DIR"

echo "Activating conda environment..."
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$WORK_DIR/Bin2Cell_Validation"

GT_CELL_BINS_PATH="${GT_CELL_BINS_PATH:-workspace_outputs/pseudo_human_colorectal/intermediate/cropped_visium_hd_human_colorectal_pseudo_square_002um_cell_bins.csv.gz}"
GT_CELL_ASSIGNMENTS_CSV="${GT_CELL_ASSIGNMENTS_CSV:-colorectal_pseudo_visium_hd_output_full_0.25/ground_truth_cell_assignments.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-workspace_outputs/pseudo_human_colorectal/intermediate/reference_shape}"
PREFIX="${PREFIX:-human_colorectal_gt_shape_reference}"
BIN_SIZE_UM="${BIN_SIZE_UM:-2.0}"
EPSILON="${EPSILON:-1e-8}"

mkdir -p "$OUTPUT_DIR"

python preprocessing/build_reference_shape_features_from_gt.py \
  --gt-cell-bins-path "$GT_CELL_BINS_PATH" \
  --gt-cell-assignments-csv "$GT_CELL_ASSIGNMENTS_CSV" \
  --per-cell-output-path "$OUTPUT_DIR/${PREFIX}.per_cell.csv.gz" \
  --cell-type-summary-output-path "$OUTPUT_DIR/${PREFIX}.cell_type_summary.csv" \
  --npz-output-path "$OUTPUT_DIR/${PREFIX}.npz" \
  --summary-path "$OUTPUT_DIR/${PREFIX}.summary.json" \
  --bin-size-um "$BIN_SIZE_UM" \
  --epsilon "$EPSILON"

conda deactivate
