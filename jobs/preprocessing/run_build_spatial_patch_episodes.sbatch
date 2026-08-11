#!/bin/bash
#SBATCH --job-name=BuildPatchEpisodes
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL/logs/slurm/%x-%j.out
#SBATCH --error=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL/logs/slurm/%x-%j.err

set -euo pipefail

module load anaconda3

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/HD-Cell-RL"
cd "$WORK_DIR"

source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$WORK_DIR/Bin2Cell_Validation"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pycache_${SLURM_JOB_ID:-$$}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_${SLURM_JOB_ID:-$$}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache_${SLURM_JOB_ID:-$$}}"
mkdir -p "$PYTHONPYCACHEPREFIX" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

EPISODES_INDEX_PATH="${EPISODES_INDEX_PATH:-runs/human_colorectal_episode_build_20260529T212230Z/episodes_index.csv}"
NUCLEI_PATH="${NUCLEI_PATH:-workspace_outputs/pseudo_human_colorectal/intermediate/human_colorectal_nuclei.parquet}"
NUCLEI_FORMAT="${NUCLEI_FORMAT:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/human_colorectal_patch_episode_build}"
PREFIX="${PREFIX:-human_colorectal}"
PATCH_SIZE_UM="${PATCH_SIZE_UM:-64}"
STRIDE_UM="${STRIDE_UM:-32}"
CORE_SIZE_UM="${CORE_SIZE_UM:-48}"
N_JITTER_PASSES="${N_JITTER_PASSES:-2}"
MAX_PATCH_CELLS="${MAX_PATCH_CELLS:-64}"
MIN_CORE_CELLS="${MIN_CORE_CELLS:-1}"
MIN_PATCH_CELLS="${MIN_PATCH_CELLS:-2}"
SEED="${SEED:-7}"

python preprocessing/build_spatial_patch_episodes.py \
  --episodes-index-path "$EPISODES_INDEX_PATH" \
  --nuclei-path "$NUCLEI_PATH" \
  --nuclei-format "$NUCLEI_FORMAT" \
  --output-dir "$OUTPUT_DIR" \
  --prefix "$PREFIX" \
  --patch-size-um "$PATCH_SIZE_UM" \
  --stride-um "$STRIDE_UM" \
  --core-size-um "$CORE_SIZE_UM" \
  --n-jitter-passes "$N_JITTER_PASSES" \
  --max-patch-cells "$MAX_PATCH_CELLS" \
  --min-core-cells "$MIN_CORE_CELLS" \
  --min-patch-cells "$MIN_PATCH_CELLS" \
  --seed "$SEED"

conda deactivate
