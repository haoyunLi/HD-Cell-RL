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

EPISODES_INDEX_PATH="${EPISODES_INDEX_PATH:-}"
NUCLEI_PATH="${NUCLEI_PATH:-workspace_outputs/colorectal_nucleus_based/intermediate/nuclei/colorectal_nucleus_based_nuclei.parquet}"
NUCLEI_FORMAT="${NUCLEI_FORMAT:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/colorectal_nucleus_based_patch_episode_build}"
PREFIX="${PREFIX:-colorectal_nucleus_based}"
PATCH_SIZE_UM="${PATCH_SIZE_UM:-64}"
STRIDE_UM="${STRIDE_UM:-32}"
CORE_SIZE_UM="${CORE_SIZE_UM:-48}"
N_JITTER_PASSES="${N_JITTER_PASSES:-2}"
# 0 keeps every context nucleus within the episode candidate MaxDis.
MAX_PATCH_CELLS="${MAX_PATCH_CELLS:-0}"
MIN_CORE_CELLS="${MIN_CORE_CELLS:-1}"
MIN_PATCH_CELLS="${MIN_PATCH_CELLS:-2}"
SEED="${SEED:-7}"
OVERFIT_PATCH_COUNT="${OVERFIT_PATCH_COUNT:-4}"
OVERFIT_INDEX_PATH="${OVERFIT_INDEX_PATH:-$OUTPUT_DIR/${PREFIX}_patches_overfit4_64um_index.csv}"

if [[ -z "$EPISODES_INDEX_PATH" ]]; then
  LATEST_EPISODE_RUN=$(ls -dt runs/colorectal_nucleus_based_episode_build_* 2>/dev/null | head -n 1 || true)
  if [[ -n "$LATEST_EPISODE_RUN" ]]; then
    EPISODES_INDEX_PATH="$LATEST_EPISODE_RUN/episodes_index.csv"
  fi
fi
if [[ ! -f "$EPISODES_INDEX_PATH" ]]; then
  echo "Current nucleus-based episode index not found. Run jobs/preprocessing/run_episode_build.sbatch first or set EPISODES_INDEX_PATH." >&2
  exit 1
fi

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

python - "$OUTPUT_DIR/${PREFIX}_patches_index.csv" "$OVERFIT_INDEX_PATH" "$OVERFIT_PATCH_COUNT" "$SEED" <<'PY'
import sys
from pathlib import Path

import pandas as pd

source_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
count = int(sys.argv[3])
seed = int(sys.argv[4])
patches = pd.read_csv(source_path)
if count <= 0 or count > len(patches):
    raise ValueError(f"OVERFIT_PATCH_COUNT must be in [1, {len(patches)}], got {count}")
subset = patches.sample(n=count, random_state=seed).reset_index(drop=True)
output_path.parent.mkdir(parents=True, exist_ok=True)
subset.to_csv(output_path, index=False)
print(f"Wrote fixed overfit patch index: {output_path}")
PY

conda deactivate
