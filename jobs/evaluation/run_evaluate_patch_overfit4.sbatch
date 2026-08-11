#!/bin/bash
#SBATCH --job-name=EvalPatchOverfit4
#SBATCH --account=kwang222-ic
#SBATCH --partition=bioshare
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --output=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL/logs/slurm/%x-%j.out
#SBATCH --error=/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL/logs/slurm/%x-%j.err

set -euo pipefail

module load anaconda3

WORK_DIR="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL"
cd "$WORK_DIR"

source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$WORK_DIR/Bin2Cell_Validation"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCH_NUM_THREADS=32
export TORCH_NUM_INTEROP_THREADS=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pycache_${SLURM_JOB_ID:-$$}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_${SLURM_JOB_ID:-$$}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache_${SLURM_JOB_ID:-$$}}"
mkdir -p "$PYTHONPYCACHEPREFIX" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-runs/human_colorectal_patch_overfit4_20260611T171558Z/checkpoints/best_model.pt}"
BASE_CONFIG="${BASE_CONFIG:-}"
EPISODES_INDEX_PATH="${EPISODES_INDEX_PATH:-runs/human_colorectal_episode_build_20260529T212230Z/episodes_index.csv}"
PPO_EVAL_RUN_DIR="${PPO_EVAL_RUN_DIR:-runs/human_colorectal_ppo_eval_20260511T153658Z}"
PATCHES_INDEX_PATH="${PATCHES_INDEX_PATH:-runs/human_colorectal_patch_episode_build/human_colorectal_patches_overfit4_64um_index.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/human_colorectal_patch_overfit4_eval_intermediate}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-runs}"
EVAL_RUN_NAME="${EVAL_RUN_NAME:-human_colorectal_patch_overfit4_eval}"
POLICY_MODE="${POLICY_MODE:-greedy}"
RUN_DEVICE="${RUN_DEVICE:-cuda}"
ROLLOUT_DEVICES="${ROLLOUT_DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-torch_gpu}"
OVERLAY_MAX_CELLS="${OVERLAY_MAX_CELLS:-1000}"
OVERLAY_SELECTION="${OVERLAY_SELECTION:-first}"
EVAL_SEED="${EVAL_SEED:-7}"
GT_CELL_BINS_PATH="${GT_CELL_BINS_PATH:-workspace_outputs/pseudo_human_colorectal/intermediate/cropped_visium_hd_human_colorectal_pseudo_square_002um_cell_bins.csv.gz}"
GT_NUCLEAR_BINS_PATH="${GT_NUCLEAR_BINS_PATH:-workspace_outputs/pseudo_human_colorectal/intermediate/cropped_visium_hd_human_colorectal_pseudo_square_002um_nuclear_bins.csv.gz}"
GT_CELL_ASSIGNMENTS_CSV="${GT_CELL_ASSIGNMENTS_CSV:-colorectal_pseudo_visium_hd_output_full_0.25/ground_truth_cell_assignments.csv}"
GT_SC_EXPRESSION_H5="${GT_SC_EXPRESSION_H5:-colorectal_sc_data/CRC_GSE166555_raw_tumor.h5}"
EXTERNAL_NUCLEAR_BINS_PATH="${EXTERNAL_NUCLEAR_BINS_PATH:-workspace_outputs/pseudo_human_colorectal/intermediate/cropped_visium_hd_human_colorectal_pseudo_square_002um_nuclear_bins.csv.gz}"
PUBLISH_GITHUB_PAGES="${PUBLISH_GITHUB_PAGES:-true}"
PAGES_REMOTE="${PAGES_REMOTE:-origin}"

if [[ -z "$CHECKPOINT_PATH" ]]; then
  echo "CHECKPOINT_PATH is required." >&2
  echo "Example:" >&2
  echo "  sbatch --export=ALL,CHECKPOINT_PATH=runs/human_colorectal_patch_overfit4_YYYYMMDDTHHMMSSZ/checkpoints/best_model.pt jobs/evaluation/run_evaluate_patch_overfit4.sbatch" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint file not found: $CHECKPOINT_PATH" >&2
  exit 1
fi
if [[ -z "$BASE_CONFIG" ]]; then
  BASE_CONFIG="$(dirname "$(dirname "$CHECKPOINT_PATH")")/base_config_used.yaml"
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "Base config file not found: $BASE_CONFIG" >&2
  exit 1
fi
if [[ ! -f "$EPISODES_INDEX_PATH" ]]; then
  echo "Episodes index file not found: $EPISODES_INDEX_PATH" >&2
  exit 1
fi
if [[ ! -f "$PATCHES_INDEX_PATH" ]]; then
  echo "Patch index file not found: $PATCHES_INDEX_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
EVAL_BASE_CONFIG="${EVAL_BASE_CONFIG:-$OUTPUT_DIR/base_config_eval_${SLURM_JOB_ID:-manual}.yaml}"
EVAL_RUN_PATH_FILE="${EVAL_RUN_PATH_FILE:-$OUTPUT_DIR/eval_run_${SLURM_JOB_ID:-manual}.txt}"

python - "$BASE_CONFIG" "$EVAL_BASE_CONFIG" "$EPISODES_INDEX_PATH" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, episodes_index_path = map(str, sys.argv[1:4])
with Path(src).open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
cfg.setdefault("inputs", {})["episodes_index_path"] = str(Path(episodes_index_path).resolve())
with Path(dst).open("w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

echo "CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "BASE_CONFIG: $BASE_CONFIG"
echo "EPISODES_INDEX_PATH: $EPISODES_INDEX_PATH"
echo "EVAL_BASE_CONFIG: $EVAL_BASE_CONFIG"
echo "PATCHES_INDEX_PATH: $PATCHES_INDEX_PATH"
echo "EVAL_RUN_NAME: $EVAL_RUN_NAME"
echo "POLICY_MODE: $POLICY_MODE"
echo "RUN_DEVICE: $RUN_DEVICE"
echo "ROLLOUT_DEVICES: $ROLLOUT_DEVICES"
echo "ROLLOUT_BACKEND: $ROLLOUT_BACKEND"
echo "PUBLISH_GITHUB_PAGES: $PUBLISH_GITHUB_PAGES"

python scripts/evaluate_patch_checkpoint.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --base-config "$EVAL_BASE_CONFIG" \
  --patches-index-path "$PATCHES_INDEX_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --ppo_eval_run_dir "$PPO_EVAL_RUN_DIR" \
  --eval_run_name "$EVAL_RUN_NAME" \
  --eval_output_root "$EVAL_OUTPUT_ROOT" \
  --policy-mode "$POLICY_MODE" \
  --device "$RUN_DEVICE" \
  --rollout-devices "$ROLLOUT_DEVICES" \
  --rollout-backend "$ROLLOUT_BACKEND" \
  --eval-run-path-file "$EVAL_RUN_PATH_FILE" \
  --eval-mode random_patches \
  --max-eval-patches 4 \
  --overlay_max_cells "$OVERLAY_MAX_CELLS" \
  --overlay_selection "$OVERLAY_SELECTION" \
  --eval_seed "$EVAL_SEED" \
  --external_nuclear_bins_path "$EXTERNAL_NUCLEAR_BINS_PATH" \
  --gt_cell_bins_path "$GT_CELL_BINS_PATH" \
  --gt_nuclear_bins_path "$GT_NUCLEAR_BINS_PATH" \
  --gt_cell_assignments_csv "$GT_CELL_ASSIGNMENTS_CSV" \
  --gt_sc_expression_h5 "$GT_SC_EXPRESSION_H5"

case "${PUBLISH_GITHUB_PAGES,,}" in
  1|true|yes)
    if [[ ! -s "$EVAL_RUN_PATH_FILE" ]]; then
      echo "Completed evaluation path was not written: $EVAL_RUN_PATH_FILE" >&2
      exit 1
    fi
    if ! command -v npm >/dev/null 2>&1; then
      if [[ ! -s "$HOME/.nvm/nvm.sh" ]]; then
        echo "npm is required to publish the patch debugger." >&2
        exit 1
      fi
      set +u
      source "$HOME/.nvm/nvm.sh"
      set -u
    fi
    python scripts/publish_patch_debug_pages.py \
      --eval-run-dir "$(<"$EVAL_RUN_PATH_FILE")" \
      --remote "$PAGES_REMOTE" \
      --push
    ;;
  0|false|no)
    echo "GitHub Pages publication disabled."
    ;;
  *)
    echo "PUBLISH_GITHUB_PAGES must be true or false: $PUBLISH_GITHUB_PAGES" >&2
    exit 1
    ;;
esac

conda deactivate
