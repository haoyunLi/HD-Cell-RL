#!/bin/bash
#SBATCH --job-name=ShapeRefTSNE
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

# OpenBLAS in this cluster environment warns and can segfault above 64 threads.
# Keep the allocation large for memory, but cap BLAS/OpenMP worker threads before
# NumPy/sklearn are imported.
THREAD_CAP="${THREAD_CAP:-64}"
export OPENBLAS_NUM_THREADS="$THREAD_CAP"
export OMP_NUM_THREADS="$THREAD_CAP"
export MKL_NUM_THREADS="$THREAD_CAP"
export NUMEXPR_NUM_THREADS="$THREAD_CAP"
export VECLIB_MAXIMUM_THREADS="$THREAD_CAP"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
echo "Thread cap for BLAS/OpenMP libraries: $THREAD_CAP"

SHAPE_REFERENCE_NPZ="${SHAPE_REFERENCE_NPZ:-workspace_outputs/pseudo_human_colorectal/intermediate/reference_shape/human_colorectal_gt_shape_reference.npz}"
COORDINATES_CSV="${COORDINATES_CSV:-}"
OUTPUT_DIR="${OUTPUT_DIR:-workspace_outputs/pseudo_human_colorectal/intermediate/reference_shape/tsne}"
PREFIX="${PREFIX:-human_colorectal_gt_shape_reference}"
FEATURE_LAYER="${FEATURE_LAYER:-zscore}"
N_CLUSTERS="${N_CLUSTERS:-8}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-1000}"
PERPLEXITY="${PERPLEXITY:-30}"
MAX_CELLS="${MAX_CELLS:-0}"
RANDOM_SEED="${RANDOM_SEED:-7}"
N_ITER="${N_ITER:-1000}"
POINT_SIZE="${POINT_SIZE:-4}"
ALPHA="${ALPHA:-0.75}"
STRATIFY_BY_CELL_TYPE="${STRATIFY_BY_CELL_TYPE:-true}"

mkdir -p "$OUTPUT_DIR"

STRATIFY_FLAG=()
if [[ "$STRATIFY_BY_CELL_TYPE" == "false" || "$STRATIFY_BY_CELL_TYPE" == "0" ]]; then
  STRATIFY_FLAG+=(--no-stratify-by-cell-type)
fi

COORDINATES_FLAG=()
if [[ -n "$COORDINATES_CSV" ]]; then
  COORDINATES_FLAG+=(--coordinates-csv "$COORDINATES_CSV")
fi

python preprocessing/plot_shape_reference_tsne.py \
  --shape-reference-npz "$SHAPE_REFERENCE_NPZ" \
  "${COORDINATES_FLAG[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --prefix "$PREFIX" \
  --feature-layer "$FEATURE_LAYER" \
  --n-clusters "$N_CLUSTERS" \
  --min-cluster-size "$MIN_CLUSTER_SIZE" \
  --perplexity "$PERPLEXITY" \
  --max-cells "$MAX_CELLS" \
  --random-seed "$RANDOM_SEED" \
  --n-iter "$N_ITER" \
  --point-size "$POINT_SIZE" \
  --alpha "$ALPHA" \
  "${STRATIFY_FLAG[@]}"

conda deactivate
