#!/bin/bash
#SBATCH --job-name=PatchGRPOCPU
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --mem=800G
#SBATCH --output=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL/logs/slurm/%x-%j.out
#SBATCH --error=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL/logs/slurm/%x-%j.err

set -euo pipefail

module load anaconda3

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL"
cd "$WORK_DIR"

source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$WORK_DIR/Bin2Cell_Validation"

export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export TORCH_NUM_THREADS=32
export TORCH_NUM_INTEROP_THREADS=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pycache_${SLURM_JOB_ID:-$$}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_${SLURM_JOB_ID:-$$}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache_${SLURM_JOB_ID:-$$}}"
mkdir -p "$PYTHONPYCACHEPREFIX" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

PATCH_CONFIG="${PATCH_CONFIG:-configs/ppo_training.patch_full_grpo.yaml}"
BASE_PPO_CONFIG="${BASE_PPO_CONFIG:-configs/ppo_training.full_grpo.yaml}"
PATCHES_INDEX_PATH="${PATCHES_INDEX_PATH:-runs/human_colorectal_patch_episode_build/human_colorectal_patches_index.csv}"
EPISODES_INDEX_PATH="${EPISODES_INDEX_PATH:-runs/human_colorectal_episode_build_20260529T212230Z/episodes_index.csv}"
RUN_NAME="${RUN_NAME:-human_colorectal_patch_full_grpo}"
RUN_DEVICE="${RUN_DEVICE:-cpu}"
BATCH_PATCHES="${BATCH_PATCHES:-8}"
MAX_UPDATES="${MAX_UPDATES:-}"
MAX_STEPS_PER_PATCH="${MAX_STEPS_PER_PATCH:-300}"
WARM_START_CHECKPOINT="${WARM_START_CHECKPOINT:-runs/human_colorectal_full_grpo_20260508T171415Z/checkpoints/best_model.pt}"

if [ ! -f "$PATCHES_INDEX_PATH" ]; then
  echo "Patch index not found: $PATCHES_INDEX_PATH"
  echo "Run jobs/preprocessing/run_build_spatial_patch_episodes.sbatch first, or set PATCHES_INDEX_PATH."
  exit 1
fi

TMP_BASE_CONFIG="$(mktemp --suffix=.yaml)"
TMP_PATCH_CONFIG="$(mktemp --suffix=.yaml)"
trap 'rm -f "$TMP_BASE_CONFIG" "$TMP_PATCH_CONFIG"' EXIT

python - "$BASE_PPO_CONFIG" "$TMP_BASE_CONFIG" "$EPISODES_INDEX_PATH" "$RUN_DEVICE" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, episodes_index_path, run_device = map(str, sys.argv[1:5])
with Path(src).open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
cfg.setdefault("run", {})["device"] = run_device
cfg.setdefault("run", {})["rollout_mode"] = "vectorized"
cfg.setdefault("run", {})["n_rollout_workers"] = 32
cfg.setdefault("inputs", {})["episodes_index_path"] = str(Path(episodes_index_path).resolve())
with Path(dst).open("w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

python - "$PATCH_CONFIG" "$TMP_PATCH_CONFIG" "$TMP_BASE_CONFIG" "$PATCHES_INDEX_PATH" "$RUN_NAME" "$RUN_DEVICE" "$BATCH_PATCHES" "$MAX_UPDATES" "$MAX_STEPS_PER_PATCH" "$WARM_START_CHECKPOINT" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, base_cfg, patches_index, run_name, run_device, batch_patches, max_updates, max_steps, warm_start = map(str, sys.argv[1:11])
with Path(src).open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
cfg["base_ppo_config"] = str(Path(base_cfg).resolve())
run = cfg.setdefault("run", {})
run["name"] = run_name
run["device"] = run_device
run["batch_patches"] = int(batch_patches)
if warm_start:
    run["warm_start_checkpoint"] = str(Path(warm_start).resolve())
if max_updates:
    run["max_updates"] = int(max_updates)
patch = cfg.setdefault("patch_training", {})
patch["patches_index_path"] = str(Path(patches_index).resolve())
patch["max_steps_per_patch"] = int(max_steps)
with Path(dst).open("w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

echo "PATCH_CONFIG: $TMP_PATCH_CONFIG"
echo "BASE_PPO_CONFIG: $TMP_BASE_CONFIG"
echo "PATCHES_INDEX_PATH: $PATCHES_INDEX_PATH"
echo "BATCH_PATCHES: $BATCH_PATCHES"
echo "MAX_STEPS_PER_PATCH: $MAX_STEPS_PER_PATCH"
echo "WARM_START_CHECKPOINT: ${WARM_START_CHECKPOINT:-<disabled>}"

python scripts/run_patch_training.py --config "$TMP_PATCH_CONFIG"

conda deactivate
