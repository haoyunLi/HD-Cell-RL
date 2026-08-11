#!/bin/bash
#SBATCH --job-name=ShapePriorModel
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64G
#SBATCH --output=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL/logs/slurm/%x-%j.out
#SBATCH --error=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL/logs/slurm/%x-%j.err

set -euo pipefail

module load anaconda3

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL"
cd "$WORK_DIR"

source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$WORK_DIR/Bin2Cell_Validation"

REFERENCE_TABLE="${REFERENCE_TABLE:-workspace_outputs/pseudo_human_colorectal/intermediate/reference_shape/tsne/human_colorectal_gt_shape_reference.clustered_reference.csv.gz}"
NORMALIZATION_SUMMARY="${NORMALIZATION_SUMMARY:-workspace_outputs/pseudo_human_colorectal/intermediate/reference_shape/human_colorectal_gt_shape_reference.summary.json}"
OUTPUT_MODEL="${OUTPUT_MODEL:-workspace_outputs/pseudo_human_colorectal/intermediate/reference_shape/tsne/human_colorectal_gt_shape_reference.shape_prior_model.npz}"
OUTPUT_SUMMARY="${OUTPUT_SUMMARY:-workspace_outputs/pseudo_human_colorectal/intermediate/reference_shape/tsne/human_colorectal_gt_shape_reference.shape_prior_model.summary.json}"
CLUSTER_COL="${CLUSTER_COL:-shape_cluster}"
N_CLUSTERS="${N_CLUSTERS:-6}"
EPSILON="${EPSILON:-1.0e-4}"
RANDOM_SEED="${RANDOM_SEED:-7}"
RECLUSTER="${RECLUSTER:-false}"
SELECT_K="${SELECT_K:-false}"

EXTRA_FLAGS=()
if [[ "$RECLUSTER" == "true" || "$RECLUSTER" == "1" ]]; then
  EXTRA_FLAGS+=(--recluster)
fi
if [[ "$SELECT_K" == "true" || "$SELECT_K" == "1" ]]; then
  EXTRA_FLAGS+=(--select-k)
fi

python preprocessing/build_shape_prior_model.py \
  --reference-table "$REFERENCE_TABLE" \
  --normalization-summary "$NORMALIZATION_SUMMARY" \
  --output-model "$OUTPUT_MODEL" \
  --output-summary "$OUTPUT_SUMMARY" \
  --cluster-col "$CLUSTER_COL" \
  --n-clusters "$N_CLUSTERS" \
  --epsilon "$EPSILON" \
  --random-seed "$RANDOM_SEED" \
  "${EXTRA_FLAGS[@]}"

conda deactivate
