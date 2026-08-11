"""Compatibility facade for patch training helpers.

The implementation is split across focused ``patch_*`` modules, but this
module keeps the original import path stable for scripts and tests.
"""

from __future__ import annotations

from .patch_assignment import _parse_square_barcode, patch_assignments_for_core_cells
from .patch_competition import (
    _COMPETITION_MARGIN_FEATURE_SCALE,
    _apply_competition_margin_np,
    _build_competition_candidates,
    _competition_enabled,
    _competition_feature_from_margin_np,
    _compute_competition_adjusted_rewards_np,
    _stop_delta_from_values_np,
    _summary_with_action_rewards,
)
from .patch_dataset import PatchDataset, _json_cell_list, _load_episode_artifact_map
from .patch_env_cpu import CachedMultiCellPatchEnv, MultiCellPatchEnv
from .patch_env_features import _aggregate_global_features, _stop_action_features_from_global
from .patch_env_torch import (
    TorchJointPatchEnv,
    TorchPatchEnv,
    TorchSingleAgentPatchEnv,
    _four_neighbor_index_from_grid_coords_np,
    _torch_anisotropy_from_sums,
    _torch_batched_hull_areas,
    _torch_candidate_hull_areas,
    _torch_grid_cell_corners,
    _torch_grid_cells_inside_hull,
    _torch_hull_area_and_boundary_points,
    _torch_hull_boundary_mask,
    _torch_hull_equations_from_boundary_points,
    _torch_shape_model_tensors,
    _torch_shape_raw_features_from_components,
    _torch_shape_reward_values,
)
from .patch_rollout import (
    _build_patch_env,
    _obs_to_numpy,
    _obs_uses_torch,
    collect_patch_contexts,
    collect_patch_trajectories,
    collect_patch_trajectories_batched,
    rollout_patch_episode,
)
from .patch_types import (
    PatchActionEvent,
    PatchBounds,
    PatchContext,
    PatchRolloutCache,
    PatchRolloutTransition,
    PatchStep,
    PatchTrainingSettings,
    PatchTrajectory,
    _PatchCellObservation,
    _TorchPatchCellObservation,
    _TorchShapeModelTensors,
    _TorchShapeState,
)
from .patch_update import _collate_patch_minibatch, _patch_group_advantages, build_patch_rollout_cache, patch_ppo_update
from .ppo_buffers import compute_gae_returns_and_advantages
from .ppo_config import PPOTrainingConfig
from .ppo_dataset import EpisodeDataset
from .ppo_feature_schema import (
    ACTION_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    A_CANDIDATE_CENTROID_DISTANCE,
    A_CANDIDATE_COMPACTNESS_GAIN,
    A_CANDIDATE_NEIGHBOR_SUPPORT,
    A_COMPETITION_MARGIN,
    A_FEATURE_1,
    A_FEATURE_2,
    A_FEATURE_3,
    A_FEATURE_4,
    A_FEATURE_5,
    A_FEATURE_6,
    A_FEATURE_7,
    A_FEATURE_8,
    A_FEATURE_9,
    A_FEATURE_10,
    A_IS_STOP_ACTION,
    G_ASSIGNED_FRAC,
    G_ASSIGNED_LL_MAX,
    G_ASSIGNED_LL_MEAN,
    G_CENTROID_DRIFT_SCALED,
    G_COMPACTNESS_PROXY,
    G_COMPACT_STREAK_SCALED,
    G_FRONTIER_ADD_REWARD_MAX,
    G_FRONTIER_ADD_REWARD_MEAN,
    G_FRONTIER_ADD_REWARD_TOPK_MEAN,
    G_GROW_RATIO_SCALED,
    G_N_BINS_SCALED,
    G_POSITIVE_FRONTIER_FRACTION,
    G_REMAINING_FRAC,
    G_SEED_SIZE_SCALED,
    G_STEP_FRAC,
)
from .ppo_state import (
    EpisodeContext,
    _build_static_action_template,
    _compute_state_feature_bundle,
    _fill_dynamic_action_features,
    _scale_grow_ratio_feature,
    _scale_seed_size_feature,
    _state_summary_from_bundle,
    _zscore_1d,
)
from .reward import compute_frontier_eligible_mask, compute_stop_delta

__all__ = [
    "PatchTrainingSettings",
    "PatchBounds",
    "PatchContext",
    "PatchActionEvent",
    "PatchStep",
    "PatchTrajectory",
    "PatchRolloutTransition",
    "PatchRolloutCache",
    "PatchDataset",
    "MultiCellPatchEnv",
    "CachedMultiCellPatchEnv",
    "TorchPatchEnv",
    "TorchSingleAgentPatchEnv",
    "TorchJointPatchEnv",
    "rollout_patch_episode",
    "collect_patch_contexts",
    "collect_patch_trajectories",
    "collect_patch_trajectories_batched",
    "build_patch_rollout_cache",
    "patch_ppo_update",
    "patch_assignments_for_core_cells",
]
