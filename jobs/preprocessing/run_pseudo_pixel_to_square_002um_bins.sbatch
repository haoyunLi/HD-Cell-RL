#!/bin/bash
#SBATCH --job-name=ExportNucleusGTBins
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=120G
#SBATCH --output=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL/logs/slurm/%x-%j.out
#SBATCH --error=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL/logs/slurm/%x-%j.err

set -euo pipefail

module load anaconda3

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL"
cd "$WORK_DIR"

echo "Activating Bin2Cell_Validation conda environment..."
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$WORK_DIR/Bin2Cell_Validation"

PSEUDO_ROOT="${PSEUDO_ROOT:-colorectal_nucleus_based}"
GROUND_TRUTH_ROOT="${GROUND_TRUTH_ROOT:-../Bin2Cell_Validation/outputs/ground_truth/colorectal_nucleus_based}"
OUTPUT_DIR="${OUTPUT_DIR:-workspace_outputs/colorectal_nucleus_based/intermediate/ground_truth}"
BIN_ASSIGNMENTS_PATH="${BIN_ASSIGNMENTS_PATH:-$GROUND_TRUTH_ROOT/bins/ground_truth_bin_assignments.parquet}"
BARCODES_PATH="${BARCODES_PATH:-$PSEUDO_ROOT/filtered_feature_bc_matrix/barcodes.tsv.gz}"
CELL_ASSIGNMENTS_PATH="${CELL_ASSIGNMENTS_PATH:-$PSEUDO_ROOT/ground_truth_cell_assignments.csv}"
MICRONS_PER_PIXEL="${MICRONS_PER_PIXEL:-0.2737012522439323}"

mkdir -p "$OUTPUT_DIR"

CELL_OUTPUT_PATH="${CELL_OUTPUT_PATH:-$OUTPUT_DIR/colorectal_nucleus_based_square_002um_cell_bins.csv.gz}"
NUCLEAR_OUTPUT_PATH="${NUCLEAR_OUTPUT_PATH:-$OUTPUT_DIR/colorectal_nucleus_based_square_002um_nuclear_bins.csv.gz}"
SUMMARY_PATH="${SUMMARY_PATH:-$OUTPUT_DIR/colorectal_nucleus_based_square_002um_gt_bins.summary.json}"

echo "BIN_ASSIGNMENTS_PATH: $BIN_ASSIGNMENTS_PATH"
echo "BARCODES_PATH: $BARCODES_PATH"
echo "CELL_ASSIGNMENTS_PATH: $CELL_ASSIGNMENTS_PATH"
echo "CELL_OUTPUT_PATH: $CELL_OUTPUT_PATH"
echo "NUCLEAR_OUTPUT_PATH: $NUCLEAR_OUTPUT_PATH"
echo "SUMMARY_PATH: $SUMMARY_PATH"

python preprocessing/export_nucleus_based_gt_bins.py \
  --bin-assignments-path "$BIN_ASSIGNMENTS_PATH" \
  --barcodes-path "$BARCODES_PATH" \
  --cell-assignments-path "$CELL_ASSIGNMENTS_PATH" \
  --cell-output-path "$CELL_OUTPUT_PATH" \
  --nuclear-output-path "$NUCLEAR_OUTPUT_PATH" \
  --summary-path "$SUMMARY_PATH" \
  --microns-per-pixel "$MICRONS_PER_PIXEL"

conda deactivate
