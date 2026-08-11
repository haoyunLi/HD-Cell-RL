"""Dataclasses and shared types for patch-based training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .ppo_state import EpisodeContext


@dataclass(frozen=True)
class PatchTrainingSettings:
    """Runtime settings for patch-based training."""

    patches_index_path: Path
    batch_patches: int = 8
    max_steps_per_patch: int = 1000
    margin_cells_compete: bool = True
    use_core_cells_for_score: bool = True
    score_normalization: str = "mean_core_cells"
    rollout_backend: str = "legacy_cpu"
    reward_backend: str = "standard"
    stcs_reward_config: dict[str, Any] | None = None
    cache_patch_contexts: bool = False
    competition_margin_enabled: bool = True
    force_fill_expression_bins: bool = False
    fill_target: str = "reachable_expression_bins"
    stop_action_mode: str = "enabled"
    agent_mode: str = "multi_cell"
    after_fill_actions: str = "add_or_stop"
    global_delta_epsilon: float = 1.0e-6


@dataclass(frozen=True)
class PatchBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains_xy(self, xy: np.ndarray) -> np.ndarray:
        arr = np.asarray(xy, dtype=np.float64)
        return (
            (arr[:, 0] >= float(self.x_min))
            & (arr[:, 0] <= float(self.x_max))
            & (arr[:, 1] >= float(self.y_min))
            & (arr[:, 1] <= float(self.y_max))
        )


@dataclass(frozen=True)
class PatchContext:
    patch_id: str
    cells: tuple[EpisodeContext, ...]
    core_cell_ids: tuple[str, ...]
    margin_cell_ids: tuple[str, ...]
    outer_bounds: PatchBounds
    core_bounds: PatchBounds
    max_steps: int
    score_normalization: str = "mean_core_cells"
    reward_backend: str = "standard"
    competition_margin_enabled: bool = True
    force_fill_expression_bins: bool = False
    fill_target: str = "reachable_expression_bins"
    stop_action_mode: str = "enabled"
    force_fill_target_barcodes: tuple[str, ...] = ()
    agent_mode: str = "multi_cell"
    after_fill_actions: str = "add_or_stop"
    global_delta_epsilon: float = 1.0e-6

    @property
    def n_cells(self) -> int:
        return int(len(self.cells))

    @property
    def force_fill_target_count(self) -> int:
        return int(len(self.force_fill_target_barcodes))


@dataclass(frozen=True)
class PatchActionEvent:
    """Compact semantic record for one proposed patch owner change."""

    action_type: str
    cell_id: str
    barcode: str
    old_cell_id: str | None = None
    applied: bool = True


@dataclass(frozen=True)
class PatchStep:
    global_features: np.ndarray
    action_features: np.ndarray
    action_mask: np.ndarray
    action: int | np.ndarray
    reward: float
    done: bool
    old_log_prob: float
    old_value: float
    joint_group_id: int = -1
    joint_old_log_prob: float | None = None
    joint_old_value: float | None = None
    action_events: tuple[PatchActionEvent, ...] = ()
    n_local_actions: int = 0
    n_noop_actions: int = 0
    phase: str | None = None
    outcome: str | None = None
    patch_score_after: float | None = None
    raw_patch_score_after: float | None = None
    owned_target_count_after: int | None = None
    target_count: int | None = None


@dataclass(frozen=True)
class PatchTrajectory:
    patch_slot: int
    patch_id: str
    steps: tuple[PatchStep, ...]
    total_reward: float
    patch_score: float
    metrics: dict[str, float]
    final_masks: dict[str, np.ndarray]
    initial_masks: dict[str, np.ndarray] | None = None
    initial_patch_score: float | None = None
    initial_raw_patch_score: float | None = None
    initial_owned_target_count: int | None = None
    target_count: int | None = None


@dataclass(frozen=True)
class PatchRolloutTransition:
    patch_slot: int
    global_features: np.ndarray
    action_features: np.ndarray
    action_mask: np.ndarray
    action: int | np.ndarray
    reward: float
    done: bool
    old_log_prob: float
    old_value: float
    return_t: float
    advantage: float
    joint_group_id: int = -1


@dataclass(frozen=True)
class PatchRolloutCache:
    transitions: tuple[PatchRolloutTransition, ...]

    @property
    def n_transitions(self) -> int:
        return int(len(self.transitions))


@dataclass(frozen=True)
class _PatchCellObservation:
    summary: dict[str, float] | None
    add_rows: np.ndarray
    add_map: tuple[tuple[int, int, float], ...]
    competition_expr_raw: np.ndarray
    stop_term: float | None
    n_frontier: int
    n_legal: int
    n_blocked: int


@dataclass(frozen=True)
class _TorchPatchCellObservation:
    summary: dict[str, float] | None
    add_rows: Any
    add_cells: Any
    add_bins: Any
    add_rewards: Any
    competition_expr_raw: Any
    stop_term: float | None
    n_frontier: int
    n_legal: int
    n_blocked: int


@dataclass(frozen=True)
class _TorchShapeModelTensors:
    scaler_mean: Any
    scaler_std: Any
    means: Any
    inv_covariances: Any
    log_determinants: Any
    log_priors: Any
    n_features: int


@dataclass(frozen=True)
class _TorchShapeState:
    area: int
    perimeter: int
    sums: Any
    hull_area: Any
    hull_points: Any
    hull_equations: Any
    raw_features: Any
    current_reward: Any
