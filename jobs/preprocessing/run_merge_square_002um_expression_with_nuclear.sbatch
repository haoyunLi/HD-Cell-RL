#!/bin/bash
#SBATCH --job-name=MergeSquare002um
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=160G
#SBATCH --output=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL/logs/slurm/%x-%j.out
#SBATCH --error=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL/logs/slurm/%x-%j.err

module load anaconda3

# Set up environment
WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL"
cd $WORK_DIR

# Activate conda environment
echo "Activating Bin2Cell_Validation conda environment..."
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate $WORK_DIR/Bin2Cell_Validation

# Build filtered-bin RL metadata aligned to official square_002um barcodes.
MATRIX_PATH="${MATRIX_PATH:-colorectal_nucleus_based/filtered_feature_bc_matrix}"
BARCODES_TSV_PATH="${BARCODES_TSV_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-workspace_outputs/colorectal_nucleus_based/intermediate/square_002um_filtered_nuclear}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-colorectal_nucleus_based_square_002um_filtered_rl}"

mkdir -p "$OUTPUT_DIR"
CMD=(
  python3 preprocessing/merge_square_002um_expression_with_nuclear.py
  --nuclear-annotation-path workspace_outputs/colorectal_nucleus_based/intermediate/ground_truth/colorectal_nucleus_based_square_002um_nuclear_bins.csv.gz
  --matrix-path "$MATRIX_PATH"
  --tissue-positions-parquet colorectal_nucleus_based/spatial/tissue_positions.parquet
  --output-dir "$OUTPUT_DIR"
  --prefix "$OUTPUT_PREFIX"
)

if [[ -n "$BARCODES_TSV_PATH" ]]; then
  CMD+=(--barcodes-tsv-path "$BARCODES_TSV_PATH")
fi

"${CMD[@]}"

# Deactivate conda environment
conda deactivate
