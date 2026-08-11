#!/bin/bash
#SBATCH --job-name=PatchOverfit4
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:4
#SBATCH --hint=nomultithread
#SBATCH --mem=240G
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
export TORCH_NUM_THREADS=16
export TORCH_NUM_INTEROP_THREADS=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pycache_${SLURM_JOB_ID:-$$}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_${SLURM_JOB_ID:-$$}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache_${SLURM_JOB_ID:-$$}}"
mkdir -p "$PYTHONPYCACHEPREFIX" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

PATCH_CONFIG="${PATCH_CONFIG:-configs/ppo_training.patch_overfit4.yaml}"
BASE_PPO_CONFIG="${BASE_PPO_CONFIG:-configs/ppo_training.full_grpo.yaml}"
PATCHES_INDEX_PATH="${PATCHES_INDEX_PATH:-runs/human_colorectal_patch_episode_build/human_colorectal_patches_overfit4_64um_index.csv}"
EPISODES_INDEX_PATH="${EPISODES_INDEX_PATH:-runs/human_colorectal_episode_build_20260529T212230Z/episodes_index.csv}"
RUN_NAME="${RUN_NAME:-human_colorectal_patch_overfit4}"
RUN_DEVICE="${RUN_DEVICE:-cuda}"
ROLLOUT_DEVICES="${ROLLOUT_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"
ROLLOUT_WORKER_MODE="${ROLLOUT_WORKER_MODE:-process}"
BATCH_PATCHES="${BATCH_PATCHES:-16}"
MAX_UPDATES="${MAX_UPDATES:-500}"
MAX_STEPS_PER_PATCH="${MAX_STEPS_PER_PATCH:-1024}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-torch_gpu}"
CACHE_PATCH_CONTEXTS="${CACHE_PATCH_CONTEXTS:-true}"
COMPETITION_MARGIN_ENABLED="${COMPETITION_MARGIN_ENABLED:-false}"
WARM_START_CHECKPOINT="${WARM_START_CHECKPOINT:-}"

if [ ! -f "$PATCHES_INDEX_PATH" ]; then
  echo "Fixed overfit patch index not found: $PATCHES_INDEX_PATH" >&2
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

python - "$PATCH_CONFIG" "$TMP_PATCH_CONFIG" "$TMP_BASE_CONFIG" "$PATCHES_INDEX_PATH" "$RUN_NAME" "$RUN_DEVICE" "$ROLLOUT_DEVICES" "$ROLLOUT_WORKER_MODE" "$BATCH_PATCHES" "$MAX_UPDATES" "$MAX_STEPS_PER_PATCH" "$ROLLOUT_BACKEND" "$CACHE_PATCH_CONTEXTS" "$COMPETITION_MARGIN_ENABLED" "$WARM_START_CHECKPOINT" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, base_cfg, patches_index, run_name, run_device, rollout_devices, rollout_worker_mode, batch_patches, max_updates, max_steps, rollout_backend, cache_patch_contexts, competition_margin_enabled, warm_start = map(str, sys.argv[1:16])
with Path(src).open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
cfg["base_ppo_config"] = str(Path(base_cfg).resolve())
run = cfg.setdefault("run", {})
run["name"] = run_name
run["device"] = run_device
run["rollout_devices"] = [item.strip() for item in rollout_devices.split(",") if item.strip()]
run["rollout_worker_mode"] = rollout_worker_mode
run["batch_patches"] = int(batch_patches)
run["warm_start_checkpoint"] = str(Path(warm_start).resolve()) if warm_start else ""
if max_updates:
    run["max_updates"] = int(max_updates)
patch = cfg.setdefault("patch_training", {})
patch["patches_index_path"] = str(Path(patches_index).resolve())
patch["max_steps_per_patch"] = int(max_steps)
patch["rollout_backend"] = rollout_backend
patch["cache_patch_contexts"] = cache_patch_contexts.strip().lower() not in {"0", "false", "no", "off"}
patch["competition_margin_enabled"] = competition_margin_enabled.strip().lower() not in {"0", "false", "no", "off"}
with Path(dst).open("w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

echo "PATCH_CONFIG: $TMP_PATCH_CONFIG"
echo "BASE_PPO_CONFIG: $TMP_BASE_CONFIG"
echo "PATCHES_INDEX_PATH: $PATCHES_INDEX_PATH"
echo "RUN_DEVICE: $RUN_DEVICE"
echo "ROLLOUT_DEVICES: $ROLLOUT_DEVICES"
echo "ROLLOUT_WORKER_MODE: $ROLLOUT_WORKER_MODE"
echo "BATCH_PATCHES: $BATCH_PATCHES"
echo "MAX_UPDATES: $MAX_UPDATES"
echo "MAX_STEPS_PER_PATCH: $MAX_STEPS_PER_PATCH"
echo "ROLLOUT_BACKEND: $ROLLOUT_BACKEND"
echo "CACHE_PATCH_CONTEXTS: $CACHE_PATCH_CONTEXTS"
echo "COMPETITION_MARGIN_ENABLED: $COMPETITION_MARGIN_ENABLED"
echo "WARM_START_CHECKPOINT: ${WARM_START_CHECKPOINT:-<disabled>}"

python scripts/run_patch_training.py --config "$TMP_PATCH_CONFIG"

conda deactivate
