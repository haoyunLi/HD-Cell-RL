"""Shared global and STOP action feature builders for patch environments."""

from __future__ import annotations

import numpy as np

from .ppo_feature_schema import (
    ACTION_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
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
from .ppo_state import _scale_seed_size_feature


def _aggregate_global_features(
    *,
    summaries: list[dict[str, float]],
    total_bins: int,
    total_seed_bins: int,
    step_index: int,
    max_steps: int,
) -> np.ndarray:
    out = np.zeros((GLOBAL_FEATURE_DIM,), dtype=np.float32)
    if summaries:
        keys = summaries[0].keys()
        mean_summary = {key: float(np.mean([s[key] for s in summaries])) for key in keys}
    else:
        mean_summary = {}
    out[G_ASSIGNED_FRAC] = np.float32(mean_summary.get("assigned_frac", 0.0))
    out[G_STEP_FRAC] = np.float32(float(step_index) / max(1, int(max_steps)))
    out[G_N_BINS_SCALED] = np.float32(np.log1p(max(int(total_bins), 0)) / 8.0)
    out[G_ASSIGNED_LL_MEAN] = np.float32(mean_summary.get("assigned_ll_mean", 0.0))
    out[G_ASSIGNED_LL_MAX] = np.float32(mean_summary.get("assigned_ll_max", 0.0))
    out[G_REMAINING_FRAC] = np.float32(mean_summary.get("remaining_frac", 0.0))
    out[G_SEED_SIZE_SCALED] = np.float32(_scale_seed_size_feature(int(total_seed_bins)))
    out[G_GROW_RATIO_SCALED] = np.float32(mean_summary.get("grow_ratio_scaled", 0.0))
    out[G_POSITIVE_FRONTIER_FRACTION] = np.float32(mean_summary.get("positive_frontier_fraction", 0.0))
    out[G_CENTROID_DRIFT_SCALED] = np.float32(mean_summary.get("centroid_drift_scaled", 0.0))
    out[G_COMPACTNESS_PROXY] = np.float32(mean_summary.get("compactness_proxy", 0.0))
    out[G_FRONTIER_ADD_REWARD_TOPK_MEAN] = np.float32(mean_summary.get("frontier_add_reward_topk_mean", 0.0))
    out[G_FRONTIER_ADD_REWARD_MAX] = np.float32(mean_summary.get("frontier_add_reward_max", 0.0))
    out[G_FRONTIER_ADD_REWARD_MEAN] = np.float32(mean_summary.get("frontier_add_reward_mean", 0.0))
    out[G_COMPACT_STREAK_SCALED] = np.float32(0.0)
    return out


def _stop_action_features_from_global(global_features: np.ndarray) -> np.ndarray:
    out = np.zeros((ACTION_FEATURE_DIM,), dtype=np.float32)
    out[A_IS_STOP_ACTION] = np.float32(1.0)
    out[A_FEATURE_1] = np.float32(global_features[G_ASSIGNED_FRAC])
    out[A_FEATURE_2] = np.float32(global_features[G_STEP_FRAC])
    out[A_FEATURE_3] = np.float32(global_features[G_N_BINS_SCALED])
    out[A_FEATURE_4] = np.float32(global_features[G_ASSIGNED_LL_MEAN])
    out[A_FEATURE_5] = np.float32(global_features[G_REMAINING_FRAC])
    out[A_FEATURE_6] = np.float32(global_features[G_SEED_SIZE_SCALED])
    out[A_FEATURE_7] = np.float32(global_features[G_GROW_RATIO_SCALED])
    out[A_FEATURE_8] = np.float32(global_features[G_POSITIVE_FRONTIER_FRACTION])
    out[A_FEATURE_9] = np.float32(global_features[G_CENTROID_DRIFT_SCALED])
    out[A_FEATURE_10] = np.float32(global_features[G_COMPACTNESS_PROXY])
    return out
