"""Torch-backed patch environment and shape-prior tensor helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .patch_competition import (
    _COMPETITION_MARGIN_FEATURE_SCALE,
    _build_competition_candidates,
    _competition_enabled,
    _empty_competition_candidates,
)
from .patch_env_features import _aggregate_global_features, _stop_action_features_from_global
from .patch_types import (
    PatchContext,
    _TorchPatchCellObservation,
    _TorchShapeModelTensors,
    _TorchShapeState,
)
from .ppo_feature_schema import (
    ACTION_FEATURE_DIM,
    A_CANDIDATE_CENTROID_DISTANCE,
    A_CANDIDATE_COMPACTNESS_GAIN,
    A_CANDIDATE_NEIGHBOR_SUPPORT,
    A_COMPETITION_MARGIN,
    A_FEATURE_5,
)
from .ppo_state import (
    EpisodeContext,
    _build_static_action_template,
    _scale_grow_ratio_feature,
    _scale_seed_size_feature,
)


class TorchPatchEnv:
    """Torch-backed patch environment for the full patch reward stack."""

    def __init__(self, context: PatchContext, *, device: Any) -> None:
        import torch

        self._torch = torch
        self._device = device
        self._ctx = context
        self._reward_backend = str(getattr(context, "reward_backend", "standard")).strip().lower()
        if self._reward_backend not in {"standard", "stcs"}:
            raise ValueError("PatchContext.reward_backend must be one of: standard, stcs")
        self._core_cell_ids = set(context.core_cell_ids)
        self._cell_ids = tuple(ctx.cell_id for ctx in context.cells)
        self._cell_index_by_id = {cell_id: idx for idx, cell_id in enumerate(self._cell_ids)}
        self._barcode_index_by_key: dict[str, int] = {}
        self._barcode_keys: list[str] = []
        self._barcode_indices: list[Any] = []
        barcode_to_cell_indices: dict[int, set[int]] = {}
        for cell_idx, ctx in enumerate(context.cells):
            indices: list[int] = []
            for raw_barcode in ctx.candidate_bin_ids:
                barcode = str(raw_barcode)
                barcode_index = self._barcode_index_by_key.get(barcode)
                if barcode_index is None:
                    barcode_index = len(self._barcode_keys)
                    self._barcode_index_by_key[barcode] = barcode_index
                    self._barcode_keys.append(barcode)
                indices.append(barcode_index)
                barcode_to_cell_indices.setdefault(int(barcode_index), set()).add(int(cell_idx))
            self._barcode_indices.append(torch.as_tensor(indices, device=device, dtype=torch.long))
        self._barcode_to_cell_indices = {
            barcode_index: tuple(sorted(cell_indices))
            for barcode_index, cell_indices in barcode_to_cell_indices.items()
        }
        self._force_fill_target_indices_set = {
            int(self._barcode_index_by_key[str(barcode)])
            for barcode in context.force_fill_target_barcodes
            if str(barcode) in self._barcode_index_by_key
        }
        self._force_fill_target_indices = torch.as_tensor(
            sorted(self._force_fill_target_indices_set),
            device=device,
            dtype=torch.long,
        )
        self._owned_force_fill_count = 0
        self._competition_enabled = bool(context.competition_margin_enabled) and _competition_enabled(context.cells)
        self._competition_affects_stop = any(
            bool(getattr(ctx, "competition_margin_affects_stop", True)) for ctx in context.cells
        )
        self._competition_candidates = (
            _build_competition_candidates(context)
            if self._competition_enabled
            else _empty_competition_candidates(context)
        )
        pair_offsets = np.zeros((len(context.cells),), dtype=np.int64)
        running_offset = 0
        for cell_idx, ctx in enumerate(context.cells):
            pair_offsets[int(cell_idx)] = int(running_offset)
            running_offset += int(ctx.n_bins)
        self._competition_pair_offsets = torch.as_tensor(pair_offsets, device=device, dtype=torch.long)
        self._competition_pair_boundaries = torch.as_tensor(
            pair_offsets[1:] if len(pair_offsets) > 1 else np.zeros((0,), dtype=np.int64),
            device=device,
            dtype=torch.long,
        )
        self._competition_other_cells: list[Any] = []
        self._competition_other_bins: list[Any] = []
        self._build_competition_candidate_tensors()

        self._outer_masks = tuple(
            torch.as_tensor(context.outer_bounds.contains_xy(ctx.candidate_bin_xy_um), device=device, dtype=torch.bool)
            for ctx in context.cells
        )
        self._templates = tuple(
            torch.as_tensor(
                _build_static_action_template(
                    ctx,
                    n_bins_scaled=float(np.log1p(ctx.n_bins) / 8.0),
                    seed_size_scaled=_scale_seed_size_feature(int(np.sum(ctx.initial_membership_mask))),
                ),
                device=device,
                dtype=torch.float32,
            )
            for ctx in context.cells
        )
        self._initial_masks = tuple(
            torch.as_tensor(np.asarray(ctx.initial_membership_mask, dtype=np.uint8) > 0, device=device)
            for ctx in context.cells
        )
        self._ll64 = tuple(
            torch.as_tensor(np.asarray(ctx.ll, dtype=np.float32), device=device, dtype=torch.float32)
            for ctx in context.cells
        )
        self._ll_mean_z64 = tuple(
            torch.as_tensor(np.asarray(ctx.ll_mean_z, dtype=np.float32), device=device, dtype=torch.float32)
            for ctx in context.cells
        )
        self._ll_max_z64 = tuple(
            torch.as_tensor(np.asarray(ctx.ll_max_z, dtype=np.float32), device=device, dtype=torch.float32)
            for ctx in context.cells
        )
        self._base_penalty32 = tuple(
            torch.as_tensor(np.asarray(ctx.base_penalty, dtype=np.float32), device=device, dtype=torch.float32)
            for ctx in context.cells
        )
        self._expression_conf64 = tuple(
            torch.as_tensor(np.asarray(ctx.expression_confidence, dtype=np.float32), device=device, dtype=torch.float32)
            for ctx in context.cells
        )
        self._stcs_reward32 = tuple(
            self._stcs_reward_tensor_for_context(ctx)
            for ctx in context.cells
        )
        self._xy64 = tuple(
            torch.as_tensor(np.asarray(ctx.candidate_bin_xy_um, dtype=np.float32), device=device, dtype=torch.float32)
            for ctx in context.cells
        )
        self._nucleus_xy64 = tuple(
            torch.as_tensor(np.asarray(ctx.nucleus_center_xy_um, dtype=np.float32), device=device, dtype=torch.float32)
            for ctx in context.cells
        )
        self._safe_neighbors = tuple(
            torch.as_tensor(
                np.where(np.asarray(ctx.neighbor_index, dtype=np.int64) >= 0, ctx.neighbor_index, ctx.n_bins),
                device=device,
                dtype=torch.long,
            )
            for ctx in context.cells
        )
        self._shape_grid_coords = tuple(
            torch.as_tensor(
                np.rint(
                    np.asarray(ctx.candidate_bin_xy_um, dtype=np.float64)
                    / max(float(ctx.shape_prior_bin_size_um), 1.0e-8)
                ).astype(np.int64),
                device=device,
                dtype=torch.long,
            )
            for ctx in context.cells
        )
        self._shape_four_neighbors = tuple(
            torch.as_tensor(
                _four_neighbor_index_from_grid_coords_np(coords.detach().cpu().numpy()),
                device=device,
                dtype=torch.long,
            )
            for coords in self._shape_grid_coords
        )
        self._shape_model_tensors = tuple(
            _torch_shape_model_tensors(
                getattr(ctx, "shape_prior_model", None),
                device=device,
                dtype=torch.float64,
            )
            if getattr(ctx, "shape_prior_model", None) is not None and float(getattr(ctx, "shape_prior_weight", 0.0)) > 0.0
            else None
            for ctx in context.cells
        )
        self._initial_seed_counts = tuple(int(np.sum(ctx.initial_membership_mask)) for ctx in context.cells)

        self._membership_masks: list[Any] = []
        self._owner_by_barcode = torch.empty((0,), device=device, dtype=torch.long)
        self._assigned_counts: list[int] = []
        self._score_sums: list[Any] = []
        self._sum_ll_mean_z: list[Any] = []
        self._sum_ll_max_z: list[Any] = []
        self._sum_xy: list[Any] = []
        self._shape_neighbor_counts: list[Any] = []
        self._shape_states: list[_TorchShapeState | None] = []
        self._shape_state_versions: list[int] = []
        self._competition_shape_delta_cache: dict[int, tuple[int, Any, Any]] = {}
        self._step_index = 0
        self._terminated = False
        self._truncated = False
        self._terminated_by_stop = False
        self._cell_rewards: dict[str, float] = {}
        self._stop_reward_value = 0.0
        self._cached_action_map: list[tuple[int, int, float]] = []
        self._cached_action_cells = torch.empty((0,), device=device, dtype=torch.long)
        self._cached_action_bins = torch.empty((0,), device=device, dtype=torch.long)
        self._cached_action_rewards = torch.empty((0,), device=device, dtype=torch.float32)
        self._cached_cell_observations: list[_TorchPatchCellObservation] = []
        self._last_obs: dict[str, Any] | None = None

    def _stcs_reward_tensor_for_context(self, ctx: EpisodeContext) -> Any:
        scores = getattr(ctx, "stcs_reward_scores", None)
        if self._reward_backend == "stcs":
            if scores is None:
                raise ValueError(
                    f"PatchContext.reward_backend='stcs' requires stcs_reward_scores for cell {ctx.cell_id!r}"
                )
            arr = np.asarray(scores, dtype=np.float32)
            if arr.shape != (int(ctx.n_bins),):
                raise ValueError(
                    f"stcs_reward_scores for cell {ctx.cell_id!r} must have shape ({ctx.n_bins},), got {arr.shape}"
                )
        else:
            arr = np.zeros((int(ctx.n_bins),), dtype=np.float32)
        return self._torch.as_tensor(arr, device=self._device, dtype=self._torch.float32)

    def _build_competition_candidate_tensors(self) -> None:
        torch = self._torch
        self._competition_other_cells = []
        self._competition_other_bins = []
        for cell_idx, cell_edges in enumerate(self._competition_candidates):
            n_bins = int(self._ctx.cells[int(cell_idx)].n_bins)
            max_competitors = max((len(edges) for edges in cell_edges), default=0)
            cells_arr = np.full((n_bins, max_competitors), -1, dtype=np.int64)
            bins_arr = np.zeros((n_bins, max_competitors), dtype=np.int64)
            for bin_idx, edges in enumerate(cell_edges):
                for edge_i, (other_cell_idx, other_bin_idx) in enumerate(edges):
                    cells_arr[int(bin_idx), int(edge_i)] = int(other_cell_idx)
                    bins_arr[int(bin_idx), int(edge_i)] = int(other_bin_idx)
            self._competition_other_cells.append(torch.as_tensor(cells_arr, device=self._device, dtype=torch.long))
            self._competition_other_bins.append(torch.as_tensor(bins_arr, device=self._device, dtype=torch.long))

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        torch = self._torch
        self._membership_masks = [mask.clone() for mask in self._initial_masks]
        self._owner_by_barcode = torch.full(
            (len(self._barcode_keys),),
            -1,
            device=self._device,
            dtype=torch.long,
        )
        self._step_index = 0
        self._terminated = False
        self._truncated = False
        self._terminated_by_stop = False
        self._cell_rewards = {cell_id: 0.0 for cell_id in self._cell_ids}
        self._stop_reward_value = 0.0
        self._assign_initial_seed_owners()
        self._owned_force_fill_count = self._count_owned_force_fill_barcodes()
        self._rebuild_incremental_state()
        obs = self._build_observation()
        self._last_obs = obs
        return obs, self._build_info()

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._terminated or self._truncated:
            raise RuntimeError("cannot step a finished patch environment")
        if self._last_obs is None:
            raise RuntimeError("reset must be called before step")
        action_i = int(action)
        if action_i < 0 or action_i >= int(self._last_obs["action_features"].shape[0]):
            raise ValueError(f"patch action out of range: {action_i}")

        affected_cells: tuple[int, ...] = ()
        if action_i == 0:
            if self._force_fill_enabled():
                if not self._force_fill_complete() and self._has_legal_add_actions():
                    raise ValueError("STOP action is disabled until the forced-fill expression target is complete")
                reward = 0.0
                self._truncated = not self._force_fill_complete()
                self._terminated = not self._truncated
                self._terminated_by_stop = False
            else:
                reward = float(self._stop_reward_value)
                self._terminated = True
                self._terminated_by_stop = True
        else:
            action_idx = action_i - 1
            cell_idx = int(self._cached_action_cells[action_idx].detach().cpu().item())
            bin_idx = int(self._cached_action_bins[action_idx].detach().cpu().item())
            reward = float(self._cached_action_rewards[action_idx].detach().cpu().item())
            barcode_index = int(self._barcode_indices[cell_idx][bin_idx].detach().cpu().item())
            if int(self._owner_by_barcode[barcode_index].detach().cpu().item()) >= 0:
                raise ValueError(f"invalid ADD action: barcode already owned: {self._barcode_keys[barcode_index]}")
            self._owner_by_barcode[barcode_index] = int(cell_idx)
            if int(barcode_index) in self._force_fill_target_indices_set:
                self._owned_force_fill_count += 1
            self._membership_masks[cell_idx][bin_idx] = True
            self._update_incremental_state(cell_idx=cell_idx, bin_idx=bin_idx)
            self._cell_rewards[str(self._cell_ids[cell_idx])] += float(reward)
            affected_cells = self._barcode_to_cell_indices.get(int(barcode_index), (int(cell_idx),))
            if self._force_fill_enabled() and self._force_fill_complete():
                self._terminated = True

        self._step_index += 1
        if self._step_index >= int(self._ctx.max_steps) and not self._terminated:
            self._truncated = True

        for cell_idx in affected_cells:
            self._cached_cell_observations[int(cell_idx)] = self._compute_cell_observation(int(cell_idx))
        obs = self._compose_observation()
        self._last_obs = obs
        return obs, float(reward), bool(self._terminated), bool(self._truncated), self._build_info()

    def final_masks(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for idx, ctx in enumerate(self._ctx.cells):
            out[str(ctx.cell_id)] = self._membership_masks[idx].detach().cpu().numpy().astype(np.uint8, copy=True)
        return out

    def patch_score(self) -> float:
        core = list(self._core_cell_ids)
        if not core:
            return 0.0
        total = sum(float(self._cell_rewards.get(cell_id, 0.0)) for cell_id in core)
        if self._terminated_by_stop:
            total += float(self._stop_reward_value)
        n_core = max(1, len(core))
        mode = str(self._ctx.score_normalization)
        if mode == "sum_core_cells":
            denom = 1.0
        elif mode == "sqrt_core_cells":
            denom = float(np.sqrt(n_core))
        elif mode == "mean_expression_bins":
            denom = float(max(1, self._force_fill_target_count()))
        elif mode == "sqrt_expression_bins":
            denom = float(np.sqrt(max(1, self._force_fill_target_count())))
        else:
            denom = float(n_core)
        return float(total / denom)

    def _force_fill_enabled(self) -> bool:
        return bool(getattr(self._ctx, "force_fill_expression_bins", False))

    def _force_fill_target_count(self) -> int:
        return int(len(self._force_fill_target_indices_set))

    def _force_fill_complete(self) -> bool:
        target_count = self._force_fill_target_count()
        return target_count <= 0 or int(self._owned_force_fill_count) >= target_count

    def _has_legal_add_actions(self) -> bool:
        return bool(int(self._cached_action_rewards.shape[0]) > 0)

    def _count_owned_force_fill_barcodes(self) -> int:
        if int(self._force_fill_target_indices.shape[0]) == 0:
            return 0
        owned = self._owner_by_barcode[self._force_fill_target_indices] >= 0
        return int(self._torch.sum(owned).detach().cpu().item())

    def _apply_force_fill_stop_mask(self, action_mask: Any, *, n_add_actions: int) -> None:
        if (
            self._force_fill_enabled()
            and str(getattr(self._ctx, "stop_action_mode", "enabled")) == "mask_until_filled"
            and not self._force_fill_complete()
            and int(n_add_actions) > 0
        ):
            action_mask[0] = False

    def _assign_initial_seed_owners(self) -> None:
        proposals: dict[int, list[tuple[float, str, int, int]]] = {}
        for cell_idx, ctx in enumerate(self._ctx.cells):
            seed_idx = np.flatnonzero(np.asarray(ctx.initial_membership_mask, dtype=np.uint8) > 0)
            for bin_idx in seed_idx.tolist():
                barcode_index = int(self._barcode_index_by_key[str(ctx.candidate_bin_ids[int(bin_idx)])])
                xy = np.asarray(ctx.candidate_bin_xy_um[int(bin_idx)], dtype=np.float64)
                center = np.asarray(ctx.nucleus_center_xy_um, dtype=np.float64)
                dist = float(np.sqrt(np.sum((xy - center) ** 2)))
                proposals.setdefault(barcode_index, []).append((dist, str(ctx.cell_id), cell_idx, int(bin_idx)))

        for barcode_index, candidates in proposals.items():
            candidates.sort(key=lambda item: (item[0], item[1]))
            _, _, owner_cell_idx, owner_bin_idx = candidates[0]
            self._owner_by_barcode[barcode_index] = int(owner_cell_idx)
            for _, _, cell_idx, bin_idx in candidates[1:]:
                self._membership_masks[cell_idx][bin_idx] = False
            self._membership_masks[owner_cell_idx][owner_bin_idx] = True

    def _rebuild_incremental_state(self) -> None:
        torch = self._torch
        self._assigned_counts = []
        self._score_sums = []
        self._sum_ll_mean_z = []
        self._sum_ll_max_z = []
        self._sum_xy = []
        self._shape_neighbor_counts = []
        self._shape_states = []
        self._shape_state_versions = []
        self._competition_shape_delta_cache = {}
        for cell_idx, ctx in enumerate(self._ctx.cells):
            mask = self._membership_masks[cell_idx]
            assigned_count = int(torch.sum(mask).detach().cpu().item())
            base_scores = torch.full(
                (ctx.n_cell_types,),
                float(ctx.log_prior),
                device=self._device,
                dtype=torch.float32,
            )
            if assigned_count > 0:
                score_sum = torch.sum(self._ll64[cell_idx][mask], dim=0) + base_scores
                sum_ll_mean_z = torch.sum(self._ll_mean_z64[cell_idx][mask])
                sum_ll_max_z = torch.sum(self._ll_max_z64[cell_idx][mask])
                sum_xy = torch.sum(self._xy64[cell_idx][mask], dim=0)
            else:
                score_sum = base_scores
                sum_ll_mean_z = torch.zeros((), device=self._device, dtype=torch.float32)
                sum_ll_max_z = torch.zeros((), device=self._device, dtype=torch.float32)
                sum_xy = torch.zeros((2,), device=self._device, dtype=torch.float32)
            self._assigned_counts.append(assigned_count)
            self._score_sums.append(score_sum)
            self._sum_ll_mean_z.append(sum_ll_mean_z)
            self._sum_ll_max_z.append(sum_ll_max_z)
            self._sum_xy.append(sum_xy)
            self._shape_neighbor_counts.append(self._build_shape_neighbor_counts(cell_idx=cell_idx))
            self._shape_states.append(self._build_shape_state(cell_idx=cell_idx))
            self._shape_state_versions.append(0)

    def _refresh_incremental_state_for_cell(self, *, cell_idx: int) -> None:
        torch = self._torch
        idx = int(cell_idx)
        ctx = self._ctx.cells[idx]
        mask = self._membership_masks[idx]
        assigned_count = int(torch.sum(mask).detach().cpu().item())
        base_scores = torch.full(
            (ctx.n_cell_types,),
            float(ctx.log_prior),
            device=self._device,
            dtype=torch.float32,
        )
        if assigned_count > 0:
            score_sum = torch.sum(self._ll64[idx][mask], dim=0) + base_scores
            sum_ll_mean_z = torch.sum(self._ll_mean_z64[idx][mask])
            sum_ll_max_z = torch.sum(self._ll_max_z64[idx][mask])
            sum_xy = torch.sum(self._xy64[idx][mask], dim=0)
        else:
            score_sum = base_scores
            sum_ll_mean_z = torch.zeros((), device=self._device, dtype=torch.float32)
            sum_ll_max_z = torch.zeros((), device=self._device, dtype=torch.float32)
            sum_xy = torch.zeros((2,), device=self._device, dtype=torch.float32)
        self._assigned_counts[idx] = assigned_count
        self._score_sums[idx] = score_sum
        self._sum_ll_mean_z[idx] = sum_ll_mean_z
        self._sum_ll_max_z[idx] = sum_ll_max_z
        self._sum_xy[idx] = sum_xy
        self._shape_neighbor_counts[idx] = self._build_shape_neighbor_counts(cell_idx=idx)
        self._shape_states[idx] = self._build_shape_state(cell_idx=idx)
        self._shape_state_versions[idx] += 1
        self._competition_shape_delta_cache.pop(idx, None)

    def _update_incremental_state(self, *, cell_idx: int, bin_idx: int) -> None:
        self._assigned_counts[cell_idx] += 1
        self._score_sums[cell_idx] = self._score_sums[cell_idx] + self._ll64[cell_idx][bin_idx]
        self._sum_ll_mean_z[cell_idx] = self._sum_ll_mean_z[cell_idx] + self._ll_mean_z64[cell_idx][bin_idx]
        self._sum_ll_max_z[cell_idx] = self._sum_ll_max_z[cell_idx] + self._ll_max_z64[cell_idx][bin_idx]
        self._sum_xy[cell_idx] = self._sum_xy[cell_idx] + self._xy64[cell_idx][bin_idx]
        self._shape_states[cell_idx] = self._updated_shape_state_after_add(cell_idx=cell_idx, bin_idx=bin_idx)
        self._update_shape_neighbor_counts_after_add(cell_idx=cell_idx, bin_idx=bin_idx)
        self._shape_state_versions[cell_idx] += 1
        self._competition_shape_delta_cache.pop(int(cell_idx), None)

    def _build_observation(self) -> dict[str, Any]:
        self._cached_cell_observations = [
            self._compute_cell_observation(cell_idx) for cell_idx in range(len(self._ctx.cells))
        ]
        return self._compose_observation()

    def _compute_cell_observation(self, cell_idx: int) -> _TorchPatchCellObservation:
        torch = self._torch
        ctx = self._ctx.cells[cell_idx]
        mask = self._membership_masks[cell_idx]
        assigned_count = int(self._assigned_counts[cell_idx])
        neighbor_support = self._compute_neighbor_support(cell_idx=cell_idx, mask=mask)
        if assigned_count > 0:
            frontier = (~mask) & (neighbor_support > 0.0)
        else:
            frontier = ~mask
        legal = frontier & self._outer_masks[cell_idx]
        legal = legal & (self._owner_by_barcode[self._barcode_indices[cell_idx]] < 0)

        n_frontier = int(torch.sum(frontier).detach().cpu().item())
        n_legal = int(torch.sum(legal).detach().cpu().item())
        is_core_cell = str(ctx.cell_id) in self._core_cell_ids
        add_rewards, summary, candidate_centroid_distance, candidate_compactness_gain, expr_new_raw = self._compute_cell_terms(
            cell_idx=cell_idx,
            ctx=ctx,
            mask=mask,
            legal=legal,
            n_legal=n_legal,
            neighbor_support=neighbor_support,
            need_summary=is_core_cell,
        )
        stop_term = None
        if is_core_cell:
            stop_delta = (
                self._torch_stop_delta(
                    add_rewards=add_rewards,
                    eligible_mask=legal,
                    stop_stat=str(ctx.stop_stat),
                    stop_top_k=int(ctx.stop_top_k),
                )
                if n_legal > 0
                else 0.0
            )
            stop_term = -float(ctx.stop_lambda) * stop_delta

        legal_idx = torch.nonzero(legal, as_tuple=False).flatten()
        if int(legal_idx.numel()) > 0:
            rows = self._templates[cell_idx][legal_idx + 1].clone()
            rows[:, A_FEATURE_5] = 0.0
            rows[:, A_CANDIDATE_CENTROID_DISTANCE] = candidate_centroid_distance[legal_idx]
            rows[:, A_CANDIDATE_COMPACTNESS_GAIN] = candidate_compactness_gain[legal_idx]
            rows[:, A_CANDIDATE_NEIGHBOR_SUPPORT] = neighbor_support[legal_idx].to(dtype=torch.float32)
            add_cells = torch.full_like(legal_idx, int(cell_idx), dtype=torch.long)
            add_bins = legal_idx.to(dtype=torch.long)
            add_reward_values = add_rewards[legal_idx].to(dtype=torch.float32)
        else:
            rows = torch.zeros((0, ACTION_FEATURE_DIM), device=self._device, dtype=torch.float32)
            add_cells = torch.empty((0,), device=self._device, dtype=torch.long)
            add_bins = torch.empty((0,), device=self._device, dtype=torch.long)
            add_reward_values = torch.empty((0,), device=self._device, dtype=torch.float32)

        return _TorchPatchCellObservation(
            summary=summary if is_core_cell else None,
            add_rows=rows,
            add_cells=add_cells,
            add_bins=add_bins,
            add_rewards=add_reward_values,
            competition_expr_raw=expr_new_raw.to(dtype=torch.float32),
            stop_term=stop_term,
            n_frontier=n_frontier,
            n_legal=n_legal,
            n_blocked=int(n_frontier - n_legal),
        )

    def _compose_observation(self) -> dict[str, Any]:
        torch = self._torch
        summaries = [item.summary for item in self._cached_cell_observations if item.summary is not None]
        stop_terms = [float(item.stop_term) for item in self._cached_cell_observations if item.stop_term is not None]
        add_rows = [item.add_rows for item in self._cached_cell_observations if int(item.add_rows.shape[0]) > 0]

        self._cached_action_map = []
        add_cell_chunks = [
            item.add_cells for item in self._cached_cell_observations if int(item.add_cells.shape[0]) > 0
        ]
        if add_cell_chunks:
            self._cached_action_cells = torch.cat(add_cell_chunks, dim=0)
            self._cached_action_bins = torch.cat(
                [item.add_bins for item in self._cached_cell_observations if int(item.add_bins.shape[0]) > 0],
                dim=0,
            )
            self._cached_action_rewards = torch.cat(
                [item.add_rewards for item in self._cached_cell_observations if int(item.add_rewards.shape[0]) > 0],
                dim=0,
            )
        else:
            self._cached_action_cells = torch.empty((0,), device=self._device, dtype=torch.long)
            self._cached_action_bins = torch.empty((0,), device=self._device, dtype=torch.long)
            self._cached_action_rewards = torch.empty((0,), device=self._device, dtype=torch.float32)

        competition_margin_features = None
        if self._competition_enabled and int(self._cached_action_rewards.shape[0]) > 0:
            adjusted_rewards, margins, adjusted_rewards_by_cell = self._compute_competition_adjusted_rewards_torch()
            self._cached_action_rewards = adjusted_rewards.to(dtype=torch.float32)
            competition_margin_features = torch.clamp(
                margins.to(dtype=torch.float32) / float(_COMPETITION_MARGIN_FEATURE_SCALE),
                min=-1.0,
                max=1.0,
            )
            if self._competition_affects_stop:
                summaries = []
                stop_terms = []
                for cell_idx, item in enumerate(self._cached_cell_observations):
                    if item.summary is None:
                        continue
                    ctx = self._ctx.cells[int(cell_idx)]
                    values = adjusted_rewards_by_cell.get(
                        int(cell_idx),
                        torch.zeros((0,), device=self._device, dtype=torch.float32),
                    )
                    summaries.append(
                        self._summary_with_action_rewards_torch(
                            item.summary,
                            values,
                            stop_top_k=int(ctx.stop_top_k),
                        )
                    )
                    stop_terms.append(
                        -float(ctx.stop_lambda)
                        * self._torch_stop_delta_from_values(
                            values,
                            stop_stat=str(ctx.stop_stat),
                            stop_top_k=int(ctx.stop_top_k),
                        )
                    )

        global_np = _aggregate_global_features(
            summaries=summaries,
            total_bins=sum(ctx.n_bins for ctx in self._ctx.cells),
            total_seed_bins=sum(int(np.sum(ctx.initial_membership_mask)) for ctx in self._ctx.cells),
            step_index=self._step_index,
            max_steps=self._ctx.max_steps,
        )
        global_features = torch.as_tensor(global_np, device=self._device, dtype=torch.float32)
        stop_row = torch.as_tensor(_stop_action_features_from_global(global_np), device=self._device, dtype=torch.float32)
        n_add_rows = sum(int(rows.shape[0]) for rows in add_rows)
        action_features = torch.zeros((1 + n_add_rows, ACTION_FEATURE_DIM), device=self._device, dtype=torch.float32)
        action_features[0] = stop_row
        if add_rows:
            action_features[1:] = torch.cat(add_rows, dim=0)
        if competition_margin_features is not None:
            action_features[1:, A_COMPETITION_MARGIN] = competition_margin_features
        action_mask = torch.ones((action_features.shape[0],), device=self._device, dtype=torch.bool)
        self._apply_force_fill_stop_mask(action_mask, n_add_actions=n_add_rows)
        self._stop_reward_value = float(np.mean(stop_terms)) if stop_terms else 0.0
        n_frontier = sum(item.n_frontier for item in self._cached_cell_observations)
        n_legal = sum(item.n_legal for item in self._cached_cell_observations)
        n_blocked = sum(item.n_blocked for item in self._cached_cell_observations)
        return {
            "global_features": global_features,
            "action_features": action_features,
            "action_mask": action_mask,
            "step_index": int(self._step_index),
            "patch_summary": {
                "patch_score": float(self.patch_score()),
                "n_patch_cells": float(len(self._ctx.cells)),
                "n_core_cells": float(len(self._core_cell_ids)),
                "n_margin_cells": float(max(0, len(self._ctx.cells) - len(self._core_cell_ids))),
                "n_frontier_actions": float(n_frontier),
                "n_legal_actions": float(n_legal),
                "n_blocked_frontier_actions": float(n_blocked),
                "n_force_fill_expression_bins": float(self._force_fill_target_count()),
                "n_force_fill_owned_expression_bins": float(self._owned_force_fill_count),
            },
        }

    def _compute_competition_adjusted_rewards_torch(self) -> tuple[Any, Any, dict[int, Any]]:
        torch = self._torch
        n_actions = int(self._cached_action_rewards.shape[0])
        if n_actions <= 0:
            empty = torch.zeros((0,), device=self._device, dtype=torch.float32)
            return empty, empty, {}

        pair_key_chunks: list[Any] = []
        for cell_idx, item in enumerate(self._cached_cell_observations):
            bins = item.add_bins.to(device=self._device, dtype=torch.long)
            if int(bins.shape[0]) == 0:
                continue
            pair_key_chunks.append(self._competition_pair_offsets[int(cell_idx)] + bins)
            other_cells = self._competition_other_cells[int(cell_idx)][bins]
            if int(other_cells.shape[1]) > 0:
                other_bins = self._competition_other_bins[int(cell_idx)][bins]
                valid = other_cells >= 0
                pair_key_chunks.append(self._competition_pair_offsets[other_cells[valid]] + other_bins[valid])

        if pair_key_chunks:
            pair_keys = torch.unique(torch.cat(pair_key_chunks, dim=0))
            score_pair_cells = torch.bucketize(
                pair_keys,
                self._competition_pair_boundaries,
                right=True,
            ).to(dtype=torch.long)
            score_pair_bins = pair_keys - self._competition_pair_offsets[score_pair_cells]
        else:
            score_pair_cells = torch.zeros((0,), device=self._device, dtype=torch.long)
            score_pair_bins = torch.zeros((0,), device=self._device, dtype=torch.long)

        competition_scores = self._competition_scores_by_cell_torch(
            pair_cells=score_pair_cells,
            pair_bins=score_pair_bins,
        )
        margin_chunks: list[Any] = []
        adjusted_chunks: list[Any] = []
        rewards_by_cell: dict[int, Any] = {}
        action_offset = 0
        for cell_idx, item in enumerate(self._cached_cell_observations):
            bins = item.add_bins.to(device=self._device, dtype=torch.long)
            n_cell_actions = int(bins.shape[0])
            if n_cell_actions == 0:
                continue
            ctx = self._ctx.cells[int(cell_idx)]
            weight = float(getattr(ctx, "competition_margin_weight", 0.0))
            clip = float(getattr(ctx, "competition_margin_clip", 5.0))
            base_rewards = self._cached_action_rewards[action_offset : action_offset + n_cell_actions].to(dtype=torch.float32)
            margin = torch.zeros((n_cell_actions,), device=self._device, dtype=torch.float32)
            if weight > 0.0:
                current_score = competition_scores[int(cell_idx)][bins].to(dtype=torch.float32)
                other_cells = self._competition_other_cells[int(cell_idx)][bins]
                if int(other_cells.shape[1]) > 0:
                    other_bins = self._competition_other_bins[int(cell_idx)][bins]
                    valid = other_cells >= 0
                    best_other = torch.full(
                        (n_cell_actions,),
                        -float("inf"),
                        device=self._device,
                        dtype=torch.float32,
                    )
                    for other_cell_idx, other_scores_by_bin in competition_scores.items():
                        safe_bins = torch.clamp(other_bins, min=0, max=max(0, int(other_scores_by_bin.shape[0]) - 1))
                        candidate_scores = other_scores_by_bin[safe_bins].to(dtype=torch.float32)
                        masked_scores = torch.where(
                            valid & (other_cells == int(other_cell_idx)),
                            candidate_scores,
                            torch.full_like(candidate_scores, -float("inf")),
                        )
                        best_other = torch.maximum(best_other, torch.max(masked_scores, dim=1).values)
                    has_other = torch.any(valid, dim=1)
                    raw_margin = current_score - best_other
                    margin = torch.where(
                        has_other,
                        torch.clamp(raw_margin, min=-clip, max=clip),
                        torch.zeros_like(raw_margin),
                    ).to(dtype=torch.float32)
            adjusted = base_rewards + float(weight) * margin
            margin_chunks.append(margin.to(dtype=torch.float32))
            adjusted_chunks.append(adjusted.to(dtype=torch.float32))
            rewards_by_cell[int(cell_idx)] = adjusted.to(dtype=torch.float32)
            action_offset += n_cell_actions

        if not adjusted_chunks:
            empty = torch.zeros((0,), device=self._device, dtype=torch.float32)
            return empty, empty, {}
        return (
            torch.cat(adjusted_chunks, dim=0).to(dtype=torch.float32),
            torch.cat(margin_chunks, dim=0).to(dtype=torch.float32),
            rewards_by_cell,
        )

    def _competition_shape_raw_delta_for_indices(self, *, cell_idx: int, candidate_idx: Any) -> Any:
        torch = self._torch
        cell_idx_i = int(cell_idx)
        version = int(self._shape_state_versions[cell_idx_i])
        candidate_idx = torch.unique(candidate_idx.to(device=self._device, dtype=torch.long))
        candidate_idx = candidate_idx[(candidate_idx >= 0) & (candidate_idx < int(self._ctx.cells[cell_idx_i].n_bins))]
        if int(candidate_idx.shape[0]) == 0:
            return torch.zeros((0,), device=self._device, dtype=torch.float32)

        cached = self._competition_shape_delta_cache.get(cell_idx_i)
        if cached is None or int(cached[0]) != version:
            values = torch.zeros((int(self._ctx.cells[cell_idx_i].n_bins),), device=self._device, dtype=torch.float32)
            cached_idx = torch.zeros((0,), device=self._device, dtype=torch.long)
        else:
            _, cached_idx, values = cached

        if int(cached_idx.shape[0]) == 0:
            missing = candidate_idx
        else:
            cached_mask = torch.zeros((int(self._ctx.cells[cell_idx_i].n_bins),), device=self._device, dtype=torch.bool)
            cached_mask[cached_idx] = True
            missing = candidate_idx[~cached_mask[candidate_idx]]
        if int(missing.shape[0]) > 0:
            values[missing] = self._compute_shape_raw_delta_for_indices(cell_idx=cell_idx_i, candidate_idx=missing).to(
                dtype=torch.float32
            )
            cached_idx = torch.unique(torch.cat((cached_idx, missing), dim=0))
            self._competition_shape_delta_cache[cell_idx_i] = (version, cached_idx, values)
        return values[candidate_idx].to(dtype=torch.float32)

    def _competition_scores_by_cell_torch(self, *, pair_cells: Any, pair_bins: Any) -> dict[int, Any]:
        torch = self._torch
        scores = {
            cell_idx: torch.zeros((int(ctx.n_bins),), device=self._device, dtype=torch.float32)
            for cell_idx, ctx in enumerate(self._ctx.cells)
        }
        entries: list[tuple[int, Any, Any, Any]] = []
        expr_chunks: list[Any] = []
        shape_chunks: list[Any] = []
        penalty_chunks: list[Any] = []
        w1_chunks: list[Any] = []
        shape_weight_chunks: list[Any] = []
        for cell_idx, ctx in enumerate(self._ctx.cells):
            cell_mask = pair_cells == int(cell_idx)
            idx = pair_bins[cell_mask]
            idx = idx[(idx >= 0) & (idx < int(ctx.n_bins))]
            if int(idx.shape[0]) == 0:
                continue
            expr = self._cached_cell_observations[int(cell_idx)].competition_expr_raw[idx].to(dtype=torch.float32)
            shape_raw = self._competition_shape_raw_delta_for_indices(cell_idx=int(cell_idx), candidate_idx=idx).to(
                dtype=torch.float32
            )
            penalty = self._base_penalty32[int(cell_idx)][idx].to(dtype=torch.float32)
            entries.append((int(cell_idx), idx, expr, shape_raw))
            expr_chunks.append(expr)
            shape_chunks.append(shape_raw)
            penalty_chunks.append(penalty)
            w1_chunks.append(torch.full_like(expr, float(ctx.w1), dtype=torch.float32))
            shape_weight_chunks.append(torch.full_like(expr, float(ctx.shape_prior_weight), dtype=torch.float32))

        if not entries:
            return scores

        expr_values = torch.cat(expr_chunks, dim=0)
        shape_values = torch.cat(shape_chunks, dim=0)
        penalties = torch.cat(penalty_chunks, dim=0)
        w1_values = torch.cat(w1_chunks, dim=0)
        shape_weight_values = torch.cat(shape_weight_chunks, dim=0)
        zscore_delta = max(float(getattr(ctx, "zscore_delta", 1.0e-8)) for ctx in self._ctx.cells)
        expr_z = self._zscore_values_torch(expr_values, zscore_delta=zscore_delta)
        shape_z = self._zscore_values_torch(shape_values, zscore_delta=zscore_delta)
        score_values = (w1_values * expr_z + shape_weight_values * shape_z - penalties).to(dtype=torch.float32)

        offset = 0
        for cell_idx, idx, expr, _shape_raw in entries:
            count = int(expr.shape[0])
            scores[int(cell_idx)][idx] = score_values[offset : offset + count]
            offset += count
        return scores

    def _compute_shape_raw_delta_for_indices(self, *, cell_idx: int, candidate_idx: Any) -> Any:
        torch = self._torch
        state = self._shape_states[int(cell_idx)]
        model = self._shape_model_tensors[int(cell_idx)]
        ctx = self._ctx.cells[int(cell_idx)]
        candidate_idx = candidate_idx.to(device=self._device, dtype=torch.long)
        out = torch.zeros((int(candidate_idx.shape[0]),), device=self._device, dtype=torch.float32)
        if (
            int(candidate_idx.shape[0]) == 0
            or state is None
            or model is None
            or float(getattr(ctx, "shape_prior_weight", 0.0)) <= 0.0
        ):
            return out

        valid = (candidate_idx >= 0) & (candidate_idx < int(ctx.n_bins)) & (~self._membership_masks[int(cell_idx)][candidate_idx])
        if not bool(torch.any(valid).detach().cpu().item()):
            return out
        valid_idx = candidate_idx[valid]
        candidate_coords = self._shape_grid_coords[int(cell_idx)][valid_idx]
        dtype = state.sums.dtype
        n_neighbors = self._shape_neighbor_counts[int(cell_idx)][valid_idx].to(dtype=dtype)
        area_after = torch.full((valid_idx.shape[0],), float(state.area + 1), device=self._device, dtype=dtype)
        perimeter_after = float(state.perimeter) + 4.0 - 2.0 * n_neighbors
        coords_f = candidate_coords.to(dtype=dtype)
        sums_after = state.sums.reshape(1, 5) + torch.stack(
            (
                coords_f[:, 0],
                coords_f[:, 1],
                coords_f[:, 0] * coords_f[:, 0],
                coords_f[:, 1] * coords_f[:, 1],
                coords_f[:, 0] * coords_f[:, 1],
            ),
            dim=1,
        )
        hull_area_after = _torch_candidate_hull_areas(
            base_points=state.hull_points,
            base_hull_area=state.hull_area,
            base_hull_equations=state.hull_equations,
            candidate_coords=candidate_coords,
            min_area=area_after,
            epsilon=1.0e-8,
        )
        raw_after = _torch_shape_raw_features_from_components(
            area=area_after,
            perimeter=perimeter_after,
            hull_area=hull_area_after,
            sums=sums_after,
            epsilon=1.0e-8,
        )
        after_reward = _torch_shape_reward_values(raw_after, model, mode=str(ctx.shape_prior_mode))
        out[valid] = (after_reward - state.current_reward).to(dtype=torch.float32)
        return out

    def _zscore_values_torch(self, values: Any, *, zscore_delta: float) -> Any:
        torch = self._torch
        values32 = values.to(dtype=torch.float32)
        if int(values32.numel()) == 0:
            return values32
        sigma = torch.std(values32, unbiased=False)
        if bool((sigma <= float(zscore_delta)).detach().cpu().item()):
            return torch.zeros_like(values32, dtype=torch.float32)
        return ((values32 - torch.mean(values32)) / (sigma + float(zscore_delta))).to(dtype=torch.float32)

    def _summary_with_action_rewards_torch(
        self,
        base_summary: dict[str, float],
        rewards: Any,
        *,
        stop_top_k: int,
    ) -> dict[str, float]:
        out = dict(base_summary)
        values = rewards.to(dtype=self._torch.float32)
        if int(values.numel()) == 0:
            out["positive_frontier_fraction"] = 0.0
            out["frontier_add_reward_mean"] = 0.0
            out["frontier_add_reward_std"] = 0.0
            out["frontier_add_reward_max"] = 0.0
            out["frontier_add_reward_topk_mean"] = 0.0
            return out
        out["positive_frontier_fraction"] = float(self._torch.mean((values > 0.0).to(self._torch.float32)).detach().cpu().item())
        out["frontier_add_reward_mean"] = float(self._torch.mean(values).detach().cpu().item())
        out["frontier_add_reward_std"] = float(self._torch.std(values, unbiased=False).detach().cpu().item())
        out["frontier_add_reward_max"] = float(self._torch.max(values).detach().cpu().item())
        out["frontier_add_reward_topk_mean"] = self._torch_stop_delta_from_values(
            values,
            stop_stat="topk_mean",
            stop_top_k=int(stop_top_k),
        )
        return out

    def _torch_stop_delta_from_values(self, values: Any, *, stop_stat: str, stop_top_k: int) -> float:
        values = values.to(dtype=self._torch.float32)
        if int(values.numel()) == 0:
            return 0.0
        if stop_stat == "max":
            return float(self._torch.max(values).detach().cpu().item())
        if stop_stat == "topk_mean":
            k = min(max(int(stop_top_k), 1), int(values.numel()))
            return float(self._torch.mean(self._torch.topk(values, k).values).detach().cpu().item())
        raise ValueError(f"unsupported stop_stat: {stop_stat!r}")

    def _compute_cell_terms(
        self,
        *,
        cell_idx: int,
        ctx: EpisodeContext,
        mask: Any,
        legal: Any,
        n_legal: int,
        neighbor_support: Any,
        need_summary: bool,
    ) -> tuple[Any, dict[str, float] | None, Any, Any, Any]:
        torch = self._torch
        n_bins = int(ctx.n_bins)
        assigned_count = int(self._assigned_counts[cell_idx])

        if assigned_count > 0:
            current_centroid_xy = self._sum_xy[cell_idx] / float(assigned_count)
            current_compactness_sum = torch.sum(neighbor_support[mask])
        else:
            current_centroid_xy = self._nucleus_xy64[cell_idx]
            current_compactness_sum = torch.zeros((), device=self._device, dtype=torch.float32)

        posterior = torch.softmax(self._score_sums[cell_idx], dim=0)
        current_confidence = torch.max(posterior) if int(posterior.numel()) > 0 else torch.zeros((), device=self._device)
        next_scores = self._score_sums[cell_idx].unsqueeze(0) + self._ll64[cell_idx]
        next_posterior = torch.softmax(next_scores, dim=1)
        next_confidence = torch.max(next_posterior, dim=1).values
        expr_new_raw = (next_confidence - current_confidence) * self._expression_conf64[cell_idx]
        expr_old_raw = torch.mv(self._ll64[cell_idx], posterior) * self._expression_conf64[cell_idx]
        if bool(ctx.normalize_expression_zscore):
            if int(n_legal) > 0:
                expr_new_term = self._zscore_over_mask(expr_new_raw, legal, float(ctx.zscore_delta))
                expr_old_term = self._zscore_over_mask(expr_old_raw, legal, float(ctx.zscore_delta))
            else:
                expr_new_term = torch.zeros_like(expr_new_raw, dtype=torch.float32)
                expr_old_term = torch.zeros_like(expr_old_raw, dtype=torch.float32)
        else:
            expr_new_term = expr_new_raw.to(dtype=torch.float32)
            expr_old_term = expr_old_raw.to(dtype=torch.float32)

        add_rewards = (
            float(ctx.w1) * expr_new_term
            + float(ctx.w5) * expr_old_term
            - self._base_penalty32[cell_idx]
            + float(ctx.w4) * neighbor_support.to(dtype=torch.float32)
        ).to(dtype=torch.float32)
        shape_term = self._compute_shape_term(cell_idx=cell_idx, ctx=ctx, legal=legal, n_legal=int(n_legal))
        if shape_term is not None:
            add_rewards = (add_rewards + float(ctx.shape_prior_weight) * shape_term).to(dtype=torch.float32)

        if need_summary and int(n_legal) > 0:
            frontier_rewards = add_rewards[legal]
            positive_frontier_fraction = float(torch.mean((frontier_rewards > 0.0).to(torch.float32)).detach().cpu().item())
            frontier_add_reward_mean = float(torch.mean(frontier_rewards).detach().cpu().item())
            frontier_add_reward_std = float(torch.std(frontier_rewards, unbiased=False).detach().cpu().item())
            frontier_add_reward_max = float(torch.max(frontier_rewards).detach().cpu().item())
            frontier_add_reward_topk_mean = self._torch_stop_delta(
                add_rewards=add_rewards,
                eligible_mask=legal,
                stop_stat="topk_mean",
                stop_top_k=int(ctx.stop_top_k),
            )
        else:
            positive_frontier_fraction = 0.0
            frontier_add_reward_mean = 0.0
            frontier_add_reward_std = 0.0
            frontier_add_reward_max = 0.0
            frontier_add_reward_topk_mean = 0.0

        candidate_centroid_distance = torch.zeros((n_bins,), device=self._device, dtype=torch.float32)
        candidate_compactness_gain = torch.zeros((n_bins,), device=self._device, dtype=torch.float32)
        if n_bins > 0 and int(n_legal) > 0:
            centroid_dist_um = torch.sqrt(torch.sum((self._xy64[cell_idx] - current_centroid_xy) ** 2, dim=1))
            scaled_dist = torch.clamp(centroid_dist_um / max(float(ctx.r_max_um), 1.0e-8), min=0.0, max=1.0)
            candidate_centroid_distance[legal] = scaled_dist[legal].to(dtype=torch.float32)
            if assigned_count > 0:
                current_compactness = current_compactness_sum / float(assigned_count)
                new_compactness = (current_compactness_sum + 2.0 * neighbor_support) / float(assigned_count + 1)
                candidate_compactness_gain[legal] = (new_compactness[legal] - current_compactness).to(dtype=torch.float32)

        summary = None
        if need_summary:
            if n_bins > 0:
                assigned_frac = float(assigned_count / n_bins)
                remaining_frac = float((n_bins - assigned_count) / n_bins)
            else:
                assigned_frac = 0.0
                remaining_frac = 0.0
            step_frac = float(self._step_index / max(1, int(ctx.max_steps)))
            if assigned_count > 0:
                assigned_ll_mean = float((self._sum_ll_mean_z[cell_idx] / float(assigned_count)).detach().cpu().item())
                assigned_ll_max = float((self._sum_ll_max_z[cell_idx] / float(assigned_count)).detach().cpu().item())
                compactness_proxy = float((current_compactness_sum / float(assigned_count)).detach().cpu().item())
                drift_vec = current_centroid_xy - self._nucleus_xy64[cell_idx]
                drift_um = torch.sqrt(torch.sum(drift_vec * drift_vec))
                centroid_drift_scaled = float(
                    torch.clamp(drift_um / max(float(ctx.r_max_um), 1.0e-8), min=0.0, max=1.0).detach().cpu().item()
                )
            else:
                assigned_ll_mean = 0.0
                assigned_ll_max = 0.0
                compactness_proxy = 0.0
                centroid_drift_scaled = 0.0
            summary = {
                "assigned_frac": assigned_frac,
                "step_frac": step_frac,
                "remaining_frac": remaining_frac,
                "grow_ratio_scaled": _scale_grow_ratio_feature(assigned_count, self._initial_seed_counts[cell_idx]),
                "positive_frontier_fraction": positive_frontier_fraction,
                "centroid_drift_scaled": centroid_drift_scaled,
                "compactness_proxy": compactness_proxy,
                "assigned_ll_mean": assigned_ll_mean,
                "assigned_ll_max": assigned_ll_max,
                "frontier_add_reward_topk_mean": frontier_add_reward_topk_mean,
                "frontier_add_reward_mean": frontier_add_reward_mean,
                "frontier_add_reward_std": frontier_add_reward_std,
                "frontier_add_reward_max": frontier_add_reward_max,
            }
        return add_rewards, summary, candidate_centroid_distance, candidate_compactness_gain, expr_new_raw.to(dtype=torch.float32)

    def _build_shape_neighbor_counts(self, *, cell_idx: int) -> Any:
        torch = self._torch
        mask = self._membership_masks[int(cell_idx)]
        padded = torch.zeros((mask.shape[0] + 1,), device=self._device, dtype=torch.float32)
        padded[: mask.shape[0]] = mask.to(dtype=torch.float32)
        return padded[self._shape_four_neighbors[int(cell_idx)]].sum(dim=1).to(dtype=torch.float32)

    def _update_shape_neighbor_counts_after_add(self, *, cell_idx: int, bin_idx: int) -> None:
        neighbors = self._shape_four_neighbors[int(cell_idx)][int(bin_idx)]
        valid = neighbors < int(self._membership_masks[int(cell_idx)].shape[0])
        if bool(self._torch.any(valid).detach().cpu().item()):
            self._shape_neighbor_counts[int(cell_idx)][neighbors[valid]] += 1.0

    def _build_shape_state(self, *, cell_idx: int) -> _TorchShapeState | None:
        torch = self._torch
        model = self._shape_model_tensors[cell_idx]
        ctx = self._ctx.cells[cell_idx]
        if model is None or float(getattr(ctx, "shape_prior_weight", 0.0)) <= 0.0:
            return None
        mask = self._membership_masks[cell_idx]
        coords = self._shape_grid_coords[cell_idx]
        assigned_count = int(torch.sum(mask).detach().cpu().item())
        coords_assigned = coords[mask]
        coords_f = coords_assigned.to(dtype=torch.float64)
        if assigned_count > 0:
            sums = torch.stack(
                (
                    torch.sum(coords_f[:, 0]),
                    torch.sum(coords_f[:, 1]),
                    torch.sum(coords_f[:, 0] * coords_f[:, 0]),
                    torch.sum(coords_f[:, 1] * coords_f[:, 1]),
                    torch.sum(coords_f[:, 0] * coords_f[:, 1]),
                )
            )
            support4 = self._shape_neighbor_counts[int(cell_idx)]
            perimeter = int(torch.sum((4.0 - support4)[mask]).detach().cpu().item())
            points = _torch_grid_cell_corners(coords_assigned, dtype=torch.float64)
            hull_area, hull_points = _torch_hull_area_and_boundary_points(
                points=points,
                min_area=torch.as_tensor(float(assigned_count), device=self._device, dtype=torch.float64),
                epsilon=1.0e-8,
            )
            hull_equations = _torch_hull_equations_from_boundary_points(hull_points, epsilon=1.0e-8)
        else:
            sums = torch.zeros((5,), device=self._device, dtype=torch.float64)
            perimeter = 0
            hull_area = torch.zeros((), device=self._device, dtype=torch.float64)
            hull_points = torch.zeros((0, 2), device=self._device, dtype=torch.float64)
            hull_equations = torch.zeros((0, 3), device=self._device, dtype=torch.float64)

        raw = _torch_shape_raw_features_from_components(
            area=torch.as_tensor([float(assigned_count)], device=self._device, dtype=torch.float64),
            perimeter=torch.as_tensor([float(perimeter)], device=self._device, dtype=torch.float64),
            hull_area=hull_area.reshape(1),
            sums=sums.reshape(1, 5),
            epsilon=1.0e-8,
        )[0]
        current_reward = _torch_shape_reward_values(raw.reshape(1, 4), model, mode=str(ctx.shape_prior_mode))[0]
        return _TorchShapeState(
            area=assigned_count,
            perimeter=perimeter,
            sums=sums,
            hull_area=hull_area,
            hull_points=hull_points,
            hull_equations=hull_equations,
            raw_features=raw,
            current_reward=current_reward,
        )

    def _updated_shape_state_after_add(self, *, cell_idx: int, bin_idx: int) -> _TorchShapeState | None:
        torch = self._torch
        old = self._shape_states[cell_idx]
        model = self._shape_model_tensors[cell_idx]
        ctx = self._ctx.cells[cell_idx]
        if old is None or model is None:
            return None
        dtype = old.sums.dtype
        coord = self._shape_grid_coords[cell_idx][int(bin_idx)].to(dtype=dtype)
        area = int(old.area + 1)
        n_neighbors = float(self._shape_neighbor_counts[int(cell_idx)][int(bin_idx)].detach().cpu().item())
        perimeter = int(old.perimeter + 4 - 2 * n_neighbors)
        sums = old.sums + torch.stack((coord[0], coord[1], coord[0] * coord[0], coord[1] * coord[1], coord[0] * coord[1]))
        min_area = torch.as_tensor(float(area), device=self._device, dtype=dtype)
        inside = _torch_grid_cells_inside_hull(
            coord.reshape(1, 2).to(dtype=torch.long),
            old.hull_equations,
            dtype=dtype,
            epsilon=1.0e-8,
        )
        if bool(inside[0].detach().cpu().item()):
            hull_area = torch.maximum(old.hull_area.to(dtype=dtype).reshape(()), min_area.reshape(()))
            hull_points = old.hull_points
            hull_equations = old.hull_equations
        else:
            points = torch.cat(
                (
                    old.hull_points,
                    _torch_grid_cell_corners(coord.reshape(1, 2).to(dtype=torch.long), dtype=dtype),
                ),
                dim=0,
            )
            hull_area, hull_points = _torch_hull_area_and_boundary_points(
                points=points,
                min_area=min_area,
                epsilon=1.0e-8,
            )
            hull_equations = _torch_hull_equations_from_boundary_points(hull_points, epsilon=1.0e-8)
        raw = _torch_shape_raw_features_from_components(
            area=torch.as_tensor([float(area)], device=self._device, dtype=dtype),
            perimeter=torch.as_tensor([float(perimeter)], device=self._device, dtype=dtype),
            hull_area=hull_area.reshape(1),
            sums=sums.reshape(1, 5),
            epsilon=1.0e-8,
        )[0]
        current_reward = _torch_shape_reward_values(raw.reshape(1, 4), model, mode=str(ctx.shape_prior_mode))[0]
        return _TorchShapeState(
            area=area,
            perimeter=perimeter,
            sums=sums,
            hull_area=hull_area,
            hull_points=hull_points,
            hull_equations=hull_equations,
            raw_features=raw,
            current_reward=current_reward,
        )

    def _compute_shape_term(self, *, cell_idx: int, ctx: EpisodeContext, legal: Any, n_legal: int) -> Any | None:
        torch = self._torch
        state = self._shape_states[cell_idx]
        model = self._shape_model_tensors[cell_idx]
        if state is None or model is None or float(ctx.shape_prior_weight) <= 0.0:
            return None
        out = torch.zeros((ctx.n_bins,), device=self._device, dtype=torch.float32)
        if int(n_legal) <= 0:
            return out
        legal_idx = torch.nonzero(legal, as_tuple=False).flatten()
        raw_delta = self._compute_shape_raw_delta_for_indices(cell_idx=int(cell_idx), candidate_idx=legal_idx).to(
            dtype=torch.float32
        )
        if bool(ctx.shape_prior_normalize_over_frontier):
            term_values = (raw_delta - torch.mean(raw_delta)) / (torch.std(raw_delta, unbiased=False) + float(ctx.zscore_delta))
        else:
            term_values = raw_delta
        if ctx.shape_prior_clip is not None:
            term_values = torch.clamp(term_values, -float(ctx.shape_prior_clip), float(ctx.shape_prior_clip))
        out[legal_idx] = term_values.to(dtype=torch.float32)
        return out

    def _compute_neighbor_support(self, *, cell_idx: int, mask: Any) -> Any:
        torch = self._torch
        padded = torch.zeros((mask.shape[0] + 1,), device=self._device, dtype=torch.float32)
        padded[: mask.shape[0]] = mask.to(dtype=torch.float32)
        touched = padded[self._safe_neighbors[cell_idx]].sum(dim=1)
        return touched / 8.0

    def _zscore_over_mask(self, values: Any, mask: Any, zscore_delta: float) -> Any:
        torch = self._torch
        values32 = values.to(dtype=torch.float32)
        masked = values32[mask]
        mu = torch.mean(masked)
        sigma = torch.std(masked, unbiased=False)
        return ((values32 - mu) / (sigma + float(zscore_delta))).to(dtype=torch.float32)

    def _torch_stop_delta(self, *, add_rewards: Any, eligible_mask: Any, stop_stat: str, stop_top_k: int) -> float:
        torch = self._torch
        eligible_rewards = add_rewards[eligible_mask]
        if stop_stat == "max":
            return float(torch.max(eligible_rewards).detach().cpu().item())
        if stop_stat == "topk_mean":
            if int(stop_top_k) <= 0:
                raise ValueError("stop_top_k must be > 0 when stop_stat='topk_mean'")
            k = min(int(stop_top_k), int(eligible_rewards.numel()))
            if k <= 0:
                return 0.0
            return float(torch.mean(torch.topk(eligible_rewards, k).values).detach().cpu().item())
        raise ValueError(f"unsupported stop_stat: {stop_stat!r}")

    def _build_info(self) -> dict[str, Any]:
        return {
            "patch_id": self._ctx.patch_id,
            "step_index": int(self._step_index),
            "max_steps": int(self._ctx.max_steps),
            "terminated": bool(self._terminated),
            "truncated": bool(self._truncated),
            "patch_score": float(self.patch_score()),
            "raw_total_core_reward": float(sum(self._cell_rewards.get(c, 0.0) for c in self._core_cell_ids)),
            "n_core_cells": int(len(self._core_cell_ids)),
            "n_margin_cells": int(max(0, len(self._ctx.cells) - len(self._core_cell_ids))),
            "n_patch_cells": int(len(self._ctx.cells)),
            "n_force_fill_expression_bins": int(self._force_fill_target_count()),
            "n_force_fill_owned_expression_bins": int(self._owned_force_fill_count),
        }


class TorchSingleAgentPatchEnv(TorchPatchEnv):
    """Single-active-cell patch environment with global patch-delta rewards."""

    def __init__(self, context: PatchContext, *, device: Any) -> None:
        super().__init__(context, device=device)
        self._active_cell_idx = 0
        self._single_active_cell = str(getattr(context, "agent_mode", "single_cell_global_delta")) == "single_cell_global_delta"
        self._cached_action_is_replace = self._torch.empty((0,), device=device, dtype=self._torch.bool)
        self._cached_action_old_cells = self._torch.empty((0,), device=device, dtype=self._torch.long)
        self._cached_action_old_bins = self._torch.empty((0,), device=device, dtype=self._torch.long)
        self._barcode_local_index_by_cell: list[dict[int, int]] = []
        for ctx in context.cells:
            self._barcode_local_index_by_cell.append(
                {
                    int(self._barcode_index_by_key[str(barcode)]): int(bin_idx)
                    for bin_idx, barcode in enumerate(ctx.candidate_bin_ids)
                }
            )

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        self._active_cell_idx = 0
        return super().reset()

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._terminated or self._truncated:
            raise RuntimeError("cannot step a finished patch environment")
        if self._last_obs is None:
            raise RuntimeError("reset must be called before step")
        action_i = int(action)
        if action_i <= 0 or action_i >= int(self._last_obs["action_features"].shape[0]):
            raise ValueError("single-agent patch mode only accepts exposed ADD/REPLACE actions")

        action_idx = action_i - 1
        before_score = float(self.patch_score())
        cell_idx = int(self._cached_action_cells[action_idx].detach().cpu().item())
        bin_idx = int(self._cached_action_bins[action_idx].detach().cpu().item())
        is_replace = bool(self._cached_action_is_replace[action_idx].detach().cpu().item())
        barcode_index = int(self._barcode_indices[cell_idx][bin_idx].detach().cpu().item())

        if is_replace:
            old_cell_idx = int(self._cached_action_old_cells[action_idx].detach().cpu().item())
            old_bin_idx = int(self._cached_action_old_bins[action_idx].detach().cpu().item())
            if old_cell_idx < 0 or old_bin_idx < 0:
                raise ValueError("REPLACE action is missing the previous owner")
            if old_cell_idx == cell_idx:
                raise ValueError("REPLACE action cannot replace the same cell owner")
            self._membership_masks[old_cell_idx][old_bin_idx] = False
            self._membership_masks[cell_idx][bin_idx] = True
            self._owner_by_barcode[barcode_index] = int(cell_idx)
            self._rebuild_incremental_state()
            self._owned_force_fill_count = self._count_owned_force_fill_barcodes()
        else:
            if int(self._owner_by_barcode[barcode_index].detach().cpu().item()) >= 0:
                raise ValueError(f"invalid ADD action: barcode already owned: {self._barcode_keys[barcode_index]}")
            self._owner_by_barcode[barcode_index] = int(cell_idx)
            self._membership_masks[cell_idx][bin_idx] = True
            self._update_incremental_state(cell_idx=cell_idx, bin_idx=bin_idx)
            if int(barcode_index) in self._force_fill_target_indices_set:
                self._owned_force_fill_count += 1

        after_score = float(self.patch_score())
        reward = float(after_score - before_score)
        self._cell_rewards[str(self._cell_ids[cell_idx])] += float(reward)
        if self._single_active_cell:
            self._active_cell_idx = (cell_idx + 1) % max(1, len(self._ctx.cells))
        self._step_index += 1
        if self._step_index >= int(self._ctx.max_steps) and not self._terminated:
            self._truncated = True

        obs = self._build_observation()
        self._last_obs = obs
        return obs, reward, bool(self._terminated), bool(self._truncated), self._build_info()

    def patch_score(self) -> float:
        return float(self._global_raw_objective().detach().cpu().item() / self._score_denominator())

    def _build_observation(self) -> dict[str, Any]:
        summaries = [
            self._compute_cell_observation(cell_idx).summary
            for cell_idx in range(len(self._ctx.cells))
        ]
        summaries = [item for item in summaries if item is not None]
        if not self._single_active_cell:
            return self._multi_cell_action_observation(summaries=summaries)

        n_cells = len(self._ctx.cells)
        for offset in range(max(1, n_cells)):
            cell_idx = (int(self._active_cell_idx) + offset) % max(1, n_cells)
            candidate = self._single_cell_action_observation(cell_idx)
            if int(candidate["action_features"].shape[0]) > 1:
                self._active_cell_idx = int(cell_idx)
                candidate["global_features"] = self._global_features_from_summaries(summaries)
                candidate["action_features"][0] = self._torch.as_tensor(
                    _stop_action_features_from_global(candidate["global_features"].detach().cpu().numpy()),
                    device=self._device,
                    dtype=self._torch.float32,
                )
                candidate["patch_summary"] = self._patch_summary_from_action_counts(
                    n_frontier=int(candidate.pop("_n_frontier")),
                    n_legal=int(candidate.pop("_n_legal")),
                    n_blocked=int(candidate.pop("_n_blocked")),
                )
                return candidate

        if self._force_fill_complete():
            self._terminated = True
        else:
            self._truncated = True
        return self._terminal_observation(summaries=summaries)

    def _multi_cell_action_observation(self, *, summaries: list[dict[str, float]]) -> dict[str, Any]:
        action_features_chunks = []
        cell_chunks = []
        bin_chunks = []
        reward_chunks = []
        replace_chunks = []
        old_cell_chunks = []
        old_bin_chunks = []
        n_frontier = 0
        n_legal = 0
        n_blocked = 0
        for cell_idx in range(len(self._ctx.cells)):
            candidate = self._single_cell_action_observation(cell_idx)
            n_frontier += int(candidate.pop("_n_frontier"))
            n_legal += int(candidate.pop("_n_legal"))
            n_blocked += int(candidate.pop("_n_blocked"))
            n_actions = int(candidate["action_features"].shape[0]) - 1
            if n_actions <= 0:
                continue
            action_features_chunks.append(candidate["action_features"][1:])
            cell_chunks.append(self._cached_action_cells)
            bin_chunks.append(self._cached_action_bins)
            reward_chunks.append(self._cached_action_rewards)
            replace_chunks.append(self._cached_action_is_replace)
            old_cell_chunks.append(self._cached_action_old_cells)
            old_bin_chunks.append(self._cached_action_old_bins)

        torch = self._torch
        global_features = self._global_features_from_summaries(summaries)
        stop_row = torch.as_tensor(
            _stop_action_features_from_global(global_features.detach().cpu().numpy()),
            device=self._device,
            dtype=torch.float32,
        )
        if action_features_chunks:
            add_rows = torch.cat(action_features_chunks, dim=0)
            action_features = torch.cat((stop_row.reshape(1, ACTION_FEATURE_DIM), add_rows), dim=0)
            action_mask = torch.ones((int(action_features.shape[0]),), device=self._device, dtype=torch.bool)
            action_mask[0] = False
            self._cached_action_cells = torch.cat(cell_chunks, dim=0)
            self._cached_action_bins = torch.cat(bin_chunks, dim=0)
            self._cached_action_rewards = torch.cat(reward_chunks, dim=0)
            self._cached_action_is_replace = torch.cat(replace_chunks, dim=0)
            self._cached_action_old_cells = torch.cat(old_cell_chunks, dim=0)
            self._cached_action_old_bins = torch.cat(old_bin_chunks, dim=0)
            return {
                "global_features": global_features,
                "action_features": action_features,
                "action_mask": action_mask,
                "step_index": int(self._step_index),
                "patch_summary": self._patch_summary_from_action_counts(
                    n_frontier=n_frontier,
                    n_legal=n_legal,
                    n_blocked=n_blocked,
                ),
            }

        if self._force_fill_complete():
            self._terminated = True
        else:
            self._truncated = True
        return self._terminal_observation(summaries=summaries)

    def _single_cell_action_observation(self, cell_idx: int) -> dict[str, Any]:
        torch = self._torch
        ctx = self._ctx.cells[int(cell_idx)]
        mask = self._membership_masks[int(cell_idx)]
        assigned_count = int(self._assigned_counts[int(cell_idx)])
        neighbor_support = self._compute_neighbor_support(cell_idx=int(cell_idx), mask=mask)
        frontier = ((~mask) & (neighbor_support > 0.0)) if assigned_count > 0 else (~mask)
        frontier = frontier & self._outer_masks[int(cell_idx)]
        barcode_indices = self._barcode_indices[int(cell_idx)]
        owners = self._owner_by_barcode[barcode_indices]
        filled = self._force_fill_complete()
        target_mask = torch.zeros((int(ctx.n_bins),), device=self._device, dtype=torch.bool)
        if int(self._force_fill_target_indices.shape[0]) > 0:
            target_mask = torch.isin(barcode_indices, self._force_fill_target_indices)
        add_legal = frontier & (owners < 0)
        replace_legal = frontier & (owners >= 0) & (owners != int(cell_idx))
        if not filled:
            add_legal = add_legal & target_mask
            replace_legal = replace_legal & target_mask
        elif str(getattr(self._ctx, "after_fill_actions", "add_or_stop")) == "replace_only":
            add_legal = torch.zeros_like(add_legal, dtype=torch.bool)

        n_frontier = int(torch.sum(frontier).detach().cpu().item())
        add_idx = torch.nonzero(add_legal, as_tuple=False).flatten().to(dtype=torch.long)
        replace_idx = torch.nonzero(replace_legal, as_tuple=False).flatten().to(dtype=torch.long)
        legal_idx_chunks = []
        old_cell_chunks = []
        old_bin_chunks = []
        reward_chunks = []
        replace_flag_chunks = []

        if int(add_idx.numel()) > 0:
            old_cells = torch.full((int(add_idx.numel()),), -1, device=self._device, dtype=torch.long)
            old_bins = torch.full((int(add_idx.numel()),), -1, device=self._device, dtype=torch.long)
            raw_delta = self._cell_add_global_delta(cell_idx=int(cell_idx), candidate_idx=add_idx)
            rewards = (raw_delta / self._score_denominator()).to(dtype=torch.float32)
            legal_idx_chunks.append(add_idx)
            old_cell_chunks.append(old_cells)
            old_bin_chunks.append(old_bins)
            reward_chunks.append(rewards)
            replace_flag_chunks.append(torch.zeros((int(add_idx.numel()),), device=self._device, dtype=torch.bool))

        if int(replace_idx.numel()) > 0:
            old_cells, old_bins = self._owner_cells_and_bins(cell_idx=int(cell_idx), bin_idx=replace_idx)
            raw_delta = self._cell_add_global_delta(cell_idx=int(cell_idx), candidate_idx=replace_idx)
            raw_delta = raw_delta + self._cell_remove_global_delta(cell_idx_tensor=old_cells, bin_idx_tensor=old_bins)
            rewards = (raw_delta / self._score_denominator()).to(dtype=torch.float32)
            keep = rewards > float(getattr(self._ctx, "global_delta_epsilon", 1.0e-6))
            if bool(torch.any(keep).detach().cpu().item()):
                replace_idx = replace_idx[keep]
                old_cells = old_cells[keep]
                old_bins = old_bins[keep]
                rewards = rewards[keep]
                legal_idx_chunks.append(replace_idx)
                old_cell_chunks.append(old_cells)
                old_bin_chunks.append(old_bins)
                reward_chunks.append(rewards)
                replace_flag_chunks.append(torch.ones((int(replace_idx.numel()),), device=self._device, dtype=torch.bool))

        if legal_idx_chunks:
            legal_idx = torch.cat(legal_idx_chunks, dim=0)
            old_cells = torch.cat(old_cell_chunks, dim=0)
            old_bins = torch.cat(old_bin_chunks, dim=0)
            rewards = torch.cat(reward_chunks, dim=0)
            is_replace = torch.cat(replace_flag_chunks, dim=0)
        else:
            legal_idx = torch.empty((0,), device=self._device, dtype=torch.long)
            old_cells = torch.empty((0,), device=self._device, dtype=torch.long)
            old_bins = torch.empty((0,), device=self._device, dtype=torch.long)
            rewards = torch.empty((0,), device=self._device, dtype=torch.float32)
            is_replace = torch.empty((0,), device=self._device, dtype=torch.bool)

        if int(legal_idx.numel()) == 0:
            return self._empty_active_observation(n_frontier=n_frontier, n_legal=0, n_blocked=n_frontier)

        legal = add_legal | replace_legal
        _, _summary, candidate_centroid_distance, candidate_compactness_gain, _expr_new_raw = self._compute_cell_terms(
            cell_idx=int(cell_idx),
            ctx=ctx,
            mask=mask,
            legal=legal,
            n_legal=int(torch.sum(legal).detach().cpu().item()),
            neighbor_support=neighbor_support,
            need_summary=False,
        )
        rows = self._templates[int(cell_idx)][legal_idx + 1].clone()
        rows[:, A_FEATURE_5] = is_replace.to(dtype=torch.float32)
        rows[:, A_CANDIDATE_CENTROID_DISTANCE] = candidate_centroid_distance[legal_idx]
        rows[:, A_CANDIDATE_COMPACTNESS_GAIN] = candidate_compactness_gain[legal_idx]
        rows[:, A_CANDIDATE_NEIGHBOR_SUPPORT] = neighbor_support[legal_idx].to(dtype=torch.float32)
        rows[:, A_COMPETITION_MARGIN] = torch.clamp(
            rewards / float(_COMPETITION_MARGIN_FEATURE_SCALE),
            min=-1.0,
            max=1.0,
        )

        stop_row = torch.zeros((1, ACTION_FEATURE_DIM), device=self._device, dtype=torch.float32)
        action_features = torch.cat((stop_row, rows), dim=0)
        action_mask = torch.ones((int(action_features.shape[0]),), device=self._device, dtype=torch.bool)
        action_mask[0] = False
        self._cached_action_cells = torch.full((int(legal_idx.numel()),), int(cell_idx), device=self._device, dtype=torch.long)
        self._cached_action_bins = legal_idx.to(dtype=torch.long)
        self._cached_action_rewards = rewards.to(dtype=torch.float32)
        self._cached_action_is_replace = is_replace
        self._cached_action_old_cells = old_cells
        self._cached_action_old_bins = old_bins
        return {
            "global_features": torch.zeros((0,), device=self._device, dtype=torch.float32),
            "action_features": action_features,
            "action_mask": action_mask,
            "step_index": int(self._step_index),
            "_n_frontier": n_frontier,
            "_n_legal": int(legal_idx.numel()),
            "_n_blocked": int(n_frontier - int(legal_idx.numel())),
        }

    def _global_features_from_summaries(self, summaries: list[dict[str, float]]) -> Any:
        global_np = _aggregate_global_features(
            summaries=summaries,
            total_bins=sum(ctx.n_bins for ctx in self._ctx.cells),
            total_seed_bins=sum(int(np.sum(ctx.initial_membership_mask)) for ctx in self._ctx.cells),
            step_index=self._step_index,
            max_steps=self._ctx.max_steps,
        )
        return self._torch.as_tensor(global_np, device=self._device, dtype=self._torch.float32)

    def _empty_active_observation(self, *, n_frontier: int, n_legal: int, n_blocked: int) -> dict[str, Any]:
        torch = self._torch
        self._cached_action_cells = torch.empty((0,), device=self._device, dtype=torch.long)
        self._cached_action_bins = torch.empty((0,), device=self._device, dtype=torch.long)
        self._cached_action_rewards = torch.empty((0,), device=self._device, dtype=torch.float32)
        self._cached_action_is_replace = torch.empty((0,), device=self._device, dtype=torch.bool)
        self._cached_action_old_cells = torch.empty((0,), device=self._device, dtype=torch.long)
        self._cached_action_old_bins = torch.empty((0,), device=self._device, dtype=torch.long)
        return {
            "global_features": torch.zeros((0,), device=self._device, dtype=torch.float32),
            "action_features": torch.zeros((1, ACTION_FEATURE_DIM), device=self._device, dtype=torch.float32),
            "action_mask": torch.zeros((1,), device=self._device, dtype=torch.bool),
            "step_index": int(self._step_index),
            "_n_frontier": int(n_frontier),
            "_n_legal": int(n_legal),
            "_n_blocked": int(n_blocked),
        }

    def _terminal_observation(self, *, summaries: list[dict[str, float]]) -> dict[str, Any]:
        torch = self._torch
        global_features = self._global_features_from_summaries(summaries)
        stop_row = torch.as_tensor(
            _stop_action_features_from_global(global_features.detach().cpu().numpy()),
            device=self._device,
            dtype=torch.float32,
        )
        self._cached_action_cells = torch.empty((0,), device=self._device, dtype=torch.long)
        self._cached_action_bins = torch.empty((0,), device=self._device, dtype=torch.long)
        self._cached_action_rewards = torch.empty((0,), device=self._device, dtype=torch.float32)
        self._cached_action_is_replace = torch.empty((0,), device=self._device, dtype=torch.bool)
        self._cached_action_old_cells = torch.empty((0,), device=self._device, dtype=torch.long)
        self._cached_action_old_bins = torch.empty((0,), device=self._device, dtype=torch.long)
        return {
            "global_features": global_features,
            "action_features": stop_row.reshape(1, ACTION_FEATURE_DIM),
            "action_mask": torch.ones((1,), device=self._device, dtype=torch.bool),
            "step_index": int(self._step_index),
            "patch_summary": self._patch_summary_from_action_counts(n_frontier=0, n_legal=0, n_blocked=0),
        }

    def _patch_summary_from_action_counts(self, *, n_frontier: int, n_legal: int, n_blocked: int) -> dict[str, float]:
        return {
            "patch_score": float(self.patch_score()),
            "n_patch_cells": float(len(self._ctx.cells)),
            "n_core_cells": float(len(self._core_cell_ids)),
            "n_margin_cells": float(max(0, len(self._ctx.cells) - len(self._core_cell_ids))),
            "n_frontier_actions": float(n_frontier),
            "n_legal_actions": float(n_legal),
            "n_blocked_frontier_actions": float(n_blocked),
            "n_force_fill_expression_bins": float(self._force_fill_target_count()),
            "n_force_fill_owned_expression_bins": float(self._owned_force_fill_count),
        }

    def _score_denominator(self) -> float:
        n_cells = max(1, len(self._ctx.cells))
        mode = str(self._ctx.score_normalization)
        if mode == "sum_core_cells":
            return 1.0
        if mode == "sqrt_core_cells":
            return float(np.sqrt(n_cells))
        if mode == "mean_expression_bins":
            return float(max(1, self._force_fill_target_count()))
        if mode == "sqrt_expression_bins":
            return float(np.sqrt(max(1, self._force_fill_target_count())))
        return float(n_cells)

    def _global_raw_objective(self) -> Any:
        torch = self._torch
        total = torch.zeros((), device=self._device, dtype=torch.float32)
        for cell_idx in range(len(self._ctx.cells)):
            total = total + self._cell_raw_objective(cell_idx=int(cell_idx))
        return total.to(dtype=torch.float32)

    def _cell_raw_objective(self, *, cell_idx: int) -> Any:
        torch = self._torch
        idx = int(cell_idx)
        ctx = self._ctx.cells[idx]
        mask = self._membership_masks[idx]
        if self._reward_backend == "stcs":
            if not bool(torch.any(mask).detach().cpu().item()):
                return torch.zeros((), device=self._device, dtype=torch.float32)
            return torch.sum(self._stcs_reward32[idx][mask]).to(dtype=torch.float32)
        if not bool(torch.any(mask).detach().cpu().item()):
            return torch.zeros((), device=self._device, dtype=torch.float32)
        posterior = torch.softmax(self._score_sums[idx], dim=0)
        confidence = torch.max(posterior) if int(posterior.numel()) > 0 else torch.zeros((), device=self._device)
        expr_conf_sum = torch.sum(self._expression_conf64[idx][mask])
        total = float(ctx.w1) * confidence.to(dtype=torch.float32) * expr_conf_sum.to(dtype=torch.float32)
        if float(ctx.w5) != 0.0:
            weighted_ll = self._ll64[idx] * self._expression_conf64[idx].reshape(-1, 1)
            total = total + float(ctx.w5) * torch.sum(torch.mv(weighted_ll[mask], posterior).to(dtype=torch.float32))
        total = total - torch.sum(self._base_penalty32[idx][mask]).to(dtype=torch.float32)
        support = self._compute_neighbor_support(cell_idx=idx, mask=mask)
        total = total + float(ctx.w4) * torch.sum(support[mask]).to(dtype=torch.float32)
        state = self._shape_states[idx]
        if state is not None and float(ctx.shape_prior_weight) > 0.0:
            total = total + float(ctx.shape_prior_weight) * state.current_reward.to(dtype=torch.float32)
        return total.to(dtype=torch.float32)

    def _cell_add_global_delta(self, *, cell_idx: int, candidate_idx: Any) -> Any:
        torch = self._torch
        ctx = self._ctx.cells[int(cell_idx)]
        idx = candidate_idx.to(device=self._device, dtype=torch.long)
        if int(idx.numel()) == 0:
            return torch.zeros((0,), device=self._device, dtype=torch.float32)
        if self._reward_backend == "stcs":
            return self._stcs_reward32[int(cell_idx)][idx].to(dtype=torch.float32)
        posterior = torch.softmax(self._score_sums[int(cell_idx)], dim=0)
        current_confidence = torch.max(posterior) if int(posterior.numel()) > 0 else torch.zeros((), device=self._device)
        current_expr_sum = torch.sum(self._expression_conf64[int(cell_idx)][self._membership_masks[int(cell_idx)]])
        next_scores = self._score_sums[int(cell_idx)].reshape(1, -1) + self._ll64[int(cell_idx)][idx]
        next_posterior = torch.softmax(next_scores, dim=1)
        next_confidence = torch.max(next_posterior, dim=1).values
        expr_conf = self._expression_conf64[int(cell_idx)][idx]
        delta = float(ctx.w1) * (
            next_confidence * (current_expr_sum + expr_conf) - current_confidence * current_expr_sum
        ).to(dtype=torch.float32)
        if float(ctx.w5) != 0.0:
            mask = self._membership_masks[int(cell_idx)]
            weighted_sum = torch.sum(
                self._ll64[int(cell_idx)][mask] * self._expression_conf64[int(cell_idx)][mask].reshape(-1, 1),
                dim=0,
            )
            current_old = torch.sum(posterior * weighted_sum)
            next_existing = torch.mv(next_posterior, weighted_sum)
            next_candidate = torch.sum(self._ll64[int(cell_idx)][idx] * next_posterior, dim=1) * expr_conf
            delta = delta + float(ctx.w5) * (next_existing + next_candidate - current_old).to(dtype=torch.float32)
        delta = delta - self._base_penalty32[int(cell_idx)][idx].to(dtype=torch.float32)
        support = self._compute_neighbor_support(cell_idx=int(cell_idx), mask=self._membership_masks[int(cell_idx)])
        delta = delta + float(ctx.w4) * (2.0 * support[idx]).to(dtype=torch.float32)
        if float(ctx.shape_prior_weight) > 0.0:
            delta = delta + float(ctx.shape_prior_weight) * self._compute_shape_raw_delta_for_indices(
                cell_idx=int(cell_idx),
                candidate_idx=idx,
            ).to(dtype=torch.float32)
        return delta.to(dtype=torch.float32)

    def _cell_remove_global_delta(self, *, cell_idx_tensor: Any, bin_idx_tensor: Any) -> Any:
        torch = self._torch
        out = torch.zeros((int(bin_idx_tensor.numel()),), device=self._device, dtype=torch.float32)
        for cell_idx in torch.unique(cell_idx_tensor).detach().cpu().numpy().astype(np.int64).tolist():
            if int(cell_idx) < 0:
                continue
            positions = torch.nonzero(cell_idx_tensor == int(cell_idx), as_tuple=False).flatten()
            bins = bin_idx_tensor[positions].to(dtype=torch.long)
            out[positions] = self._cell_remove_global_delta_for_bins(cell_idx=int(cell_idx), bin_idx=bins)
        return out.to(dtype=torch.float32)

    def _cell_remove_global_delta_for_bins(self, *, cell_idx: int, bin_idx: Any) -> Any:
        torch = self._torch
        ctx = self._ctx.cells[int(cell_idx)]
        idx = bin_idx.to(device=self._device, dtype=torch.long)
        if int(idx.numel()) == 0:
            return torch.zeros((0,), device=self._device, dtype=torch.float32)
        if self._reward_backend == "stcs":
            return (-self._stcs_reward32[int(cell_idx)][idx]).to(dtype=torch.float32)
        posterior = torch.softmax(self._score_sums[int(cell_idx)], dim=0)
        current_confidence = torch.max(posterior) if int(posterior.numel()) > 0 else torch.zeros((), device=self._device)
        current_expr_sum = torch.sum(self._expression_conf64[int(cell_idx)][self._membership_masks[int(cell_idx)]])
        prev_scores = self._score_sums[int(cell_idx)].reshape(1, -1) - self._ll64[int(cell_idx)][idx]
        prev_posterior = torch.softmax(prev_scores, dim=1)
        prev_confidence = torch.max(prev_posterior, dim=1).values
        expr_conf = self._expression_conf64[int(cell_idx)][idx]
        delta = float(ctx.w1) * (
            prev_confidence * torch.clamp(current_expr_sum - expr_conf, min=0.0) - current_confidence * current_expr_sum
        ).to(dtype=torch.float32)
        if float(ctx.w5) != 0.0:
            mask = self._membership_masks[int(cell_idx)]
            weighted_sum = torch.sum(
                self._ll64[int(cell_idx)][mask] * self._expression_conf64[int(cell_idx)][mask].reshape(-1, 1),
                dim=0,
            )
            current_old = torch.sum(posterior * weighted_sum)
            removed_weight = self._ll64[int(cell_idx)][idx] * expr_conf.reshape(-1, 1)
            prev_weighted = weighted_sum.reshape(1, -1) - removed_weight
            prev_old = torch.sum(prev_posterior * prev_weighted, dim=1)
            delta = delta + float(ctx.w5) * (prev_old - current_old).to(dtype=torch.float32)
        delta = delta + self._base_penalty32[int(cell_idx)][idx].to(dtype=torch.float32)
        support = self._compute_neighbor_support(cell_idx=int(cell_idx), mask=self._membership_masks[int(cell_idx)])
        delta = delta - float(ctx.w4) * (2.0 * support[idx]).to(dtype=torch.float32)
        if float(ctx.shape_prior_weight) > 0.0:
            delta = delta + float(ctx.shape_prior_weight) * self._compute_shape_raw_delta_after_remove_indices(
                cell_idx=int(cell_idx),
                candidate_idx=idx,
            ).to(dtype=torch.float32)
        return delta.to(dtype=torch.float32)

    def _compute_shape_raw_delta_after_remove_indices(self, *, cell_idx: int, candidate_idx: Any) -> Any:
        torch = self._torch
        state = self._shape_states[int(cell_idx)]
        model = self._shape_model_tensors[int(cell_idx)]
        ctx = self._ctx.cells[int(cell_idx)]
        idx = candidate_idx.to(device=self._device, dtype=torch.long)
        out = torch.zeros((int(idx.numel()),), device=self._device, dtype=torch.float32)
        if state is None or model is None or float(ctx.shape_prior_weight) <= 0.0 or int(idx.numel()) == 0:
            return out

        original_mask = self._membership_masks[int(cell_idx)]
        original_counts = self._shape_neighbor_counts[int(cell_idx)]
        try:
            for pos, raw_bin_idx in enumerate(idx.detach().cpu().numpy().astype(np.int64).tolist()):
                if raw_bin_idx < 0 or raw_bin_idx >= int(original_mask.shape[0]):
                    continue
                if not bool(original_mask[int(raw_bin_idx)].detach().cpu().item()):
                    continue
                new_mask = original_mask.clone()
                new_mask[int(raw_bin_idx)] = False
                self._membership_masks[int(cell_idx)] = new_mask
                self._shape_neighbor_counts[int(cell_idx)] = self._build_shape_neighbor_counts(cell_idx=int(cell_idx))
                new_state = self._build_shape_state(cell_idx=int(cell_idx))
                if new_state is not None:
                    out[int(pos)] = (new_state.current_reward - state.current_reward).to(dtype=torch.float32)
        finally:
            self._membership_masks[int(cell_idx)] = original_mask
            self._shape_neighbor_counts[int(cell_idx)] = original_counts
        return out.to(dtype=torch.float32)

    def _owner_cells_and_bins(self, *, cell_idx: int, bin_idx: Any) -> tuple[Any, Any]:
        torch = self._torch
        barcode_indices = self._barcode_indices[int(cell_idx)][bin_idx.to(dtype=torch.long)]
        owner_cells = self._owner_by_barcode[barcode_indices].to(dtype=torch.long)
        old_bins = torch.full_like(owner_cells, -1, dtype=torch.long)
        for pos, raw_owner in enumerate(owner_cells.detach().cpu().numpy().astype(np.int64).tolist()):
            if raw_owner < 0:
                continue
            barcode_index = int(barcode_indices[int(pos)].detach().cpu().item())
            old_bins[int(pos)] = int(self._barcode_local_index_by_cell[int(raw_owner)].get(barcode_index, -1))
        return owner_cells, old_bins

    def _build_info(self) -> dict[str, Any]:
        return {
            "patch_id": self._ctx.patch_id,
            "step_index": int(self._step_index),
            "max_steps": int(self._ctx.max_steps),
            "terminated": bool(self._terminated),
            "truncated": bool(self._truncated),
            "patch_score": float(self.patch_score()),
            "raw_total_core_reward": float(self._global_raw_objective().detach().cpu().item()),
            "n_core_cells": int(len(self._core_cell_ids)),
            "n_margin_cells": int(max(0, len(self._ctx.cells) - len(self._core_cell_ids))),
            "n_patch_cells": int(len(self._ctx.cells)),
            "n_force_fill_expression_bins": int(self._force_fill_target_count()),
            "n_force_fill_owned_expression_bins": int(self._owned_force_fill_count),
            "agent_mode": str(getattr(self._ctx, "agent_mode", "single_cell_global_delta")),
            "active_cell_index": int(self._active_cell_idx),
        }


class TorchJointPatchEnv(TorchSingleAgentPatchEnv):
    """Patch environment where each cell proposes at most one local action per macro-step."""

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, _info = super().reset()
        self._current_global_raw_objective_value = float(self._global_raw_objective().detach().cpu().item())
        self._last_step_applied = False
        self._last_step_phase = "postfill" if self._force_fill_complete() else "prefill"
        self._last_step_outcome = "reset"
        return obs, self._build_info()

    def patch_score(self) -> float:
        if not hasattr(self, "_current_global_raw_objective_value"):
            return super().patch_score()
        return float(self._current_global_raw_objective_value / self._score_denominator())

    def _build_info(self) -> dict[str, Any]:
        raw_total = (
            float(self._current_global_raw_objective_value)
            if hasattr(self, "_current_global_raw_objective_value")
            else float(self._global_raw_objective().detach().cpu().item())
        )
        return {
            "patch_id": self._ctx.patch_id,
            "step_index": int(self._step_index),
            "max_steps": int(self._ctx.max_steps),
            "terminated": bool(self._terminated),
            "truncated": bool(self._truncated),
            "patch_score": float(raw_total / self._score_denominator()),
            "raw_total_core_reward": float(raw_total),
            "n_core_cells": int(len(self._core_cell_ids)),
            "n_margin_cells": int(max(0, len(self._ctx.cells) - len(self._core_cell_ids))),
            "n_patch_cells": int(len(self._ctx.cells)),
            "n_force_fill_expression_bins": int(self._force_fill_target_count()),
            "n_force_fill_owned_expression_bins": int(self._owned_force_fill_count),
            "agent_mode": str(getattr(self._ctx, "agent_mode", "multi_cell_joint_global_delta")),
            "active_cell_index": int(self._active_cell_idx),
            "last_step_applied": bool(getattr(self, "_last_step_applied", False)),
            "last_step_phase": str(getattr(self, "_last_step_phase", "prefill")),
            "last_step_outcome": str(getattr(self, "_last_step_outcome", "reset")),
        }

    def _build_observation(self) -> dict[str, Any]:
        summaries = [self._fast_cell_summary(cell_idx) for cell_idx in range(len(self._ctx.cells))]
        return self._terminal_observation(summaries=summaries)

    def joint_observations(self) -> list[dict[str, Any]]:
        if self._terminated or self._truncated:
            return []
        summaries = [self._fast_cell_summary(cell_idx) for cell_idx in range(len(self._ctx.cells))]
        global_features = self._global_features_from_summaries(summaries)
        n_cells = len(self._ctx.cells)
        if n_cells <= 0:
            return []
        start = int(self._step_index) % n_cells
        return [
            self._joint_cell_observation(cell_idx=(start + offset) % n_cells, global_features=global_features)
            for offset in range(n_cells)
        ]

    def step_joint(self, selected_actions: list[dict[str, int | bool]]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._terminated or self._truncated:
            raise RuntimeError("cannot step a finished patch environment")

        prefill = not self._force_fill_complete()
        self._last_step_applied = False
        self._last_step_phase = "prefill" if prefill else "postfill"
        self._last_step_outcome = "no_proposal"
        proposals = [item for item in selected_actions if int(item.get("bin_idx", -1)) >= 0]
        proposal_barcodes = [int(item["barcode_index"]) for item in proposals]
        if len(proposal_barcodes) != len(set(proposal_barcodes)):
            raise ValueError("joint patch step cannot apply two actions to the same barcode")
        if not proposals:
            if prefill and self._joint_has_add_candidate():
                reward = -1.0 / self._score_denominator()
            else:
                reward = 0.0
                if prefill:
                    self._truncated = True
                else:
                    self._terminated = True
                self._last_step_outcome = "stop"
            self._step_index += 1
            if self._step_index >= int(self._ctx.max_steps) and not self._terminated:
                self._truncated = True
            obs = self._build_observation()
            self._last_obs = obs
            return obs, float(reward), bool(self._terminated), bool(self._truncated), self._build_info()

        masks_before = [mask.clone() for mask in self._membership_masks] if not prefill else None
        owners_before = self._owner_by_barcode.clone() if not prefill else None
        owned_force_fill_before = int(self._owned_force_fill_count)
        has_replace = any(bool(item.get("is_replace", False)) for item in proposals)
        affected_cells: set[int] = set()
        for item in proposals:
            cell_idx = int(item["cell_idx"])
            affected_cells.add(cell_idx)
            if bool(item.get("is_replace", False)):
                old_cell_idx = int(item["old_cell_idx"])
                if old_cell_idx >= 0:
                    affected_cells.add(old_cell_idx)
        raw_objective_before = float(self._current_global_raw_objective_value)
        raw_before_by_cell = {
            int(cell_idx): float(self._cell_raw_objective(cell_idx=int(cell_idx)).detach().cpu().item())
            for cell_idx in sorted(affected_cells)
        }

        for item in proposals:
            cell_idx = int(item["cell_idx"])
            bin_idx = int(item["bin_idx"])
            barcode_index = int(item["barcode_index"])
            is_replace = bool(item.get("is_replace", False))
            if is_replace:
                old_cell_idx = int(item["old_cell_idx"])
                old_bin_idx = int(item["old_bin_idx"])
                if old_cell_idx < 0 or old_bin_idx < 0:
                    raise ValueError("joint REPLACE action is missing the previous owner")
                if old_cell_idx == cell_idx:
                    raise ValueError("joint REPLACE action cannot replace the same owner")
                self._membership_masks[old_cell_idx][old_bin_idx] = False
                self._membership_masks[cell_idx][bin_idx] = True
                self._owner_by_barcode[barcode_index] = int(cell_idx)
            else:
                if int(self._owner_by_barcode[barcode_index].detach().cpu().item()) >= 0:
                    raise ValueError(f"invalid joint ADD action: barcode already owned: {self._barcode_keys[barcode_index]}")
                self._owner_by_barcode[barcode_index] = int(cell_idx)
                self._membership_masks[cell_idx][bin_idx] = True
                if not has_replace:
                    self._update_incremental_state(cell_idx=cell_idx, bin_idx=bin_idx)
                    if int(barcode_index) in self._force_fill_target_indices_set:
                        self._owned_force_fill_count += 1

        if has_replace:
            for cell_idx in sorted(affected_cells):
                self._refresh_incremental_state_for_cell(cell_idx=int(cell_idx))
            self._owned_force_fill_count = self._count_owned_force_fill_barcodes()
        raw_after = sum(
            float(self._cell_raw_objective(cell_idx=int(cell_idx)).detach().cpu().item())
            for cell_idx in sorted(affected_cells)
        )
        raw_before = sum(float(value) for value in raw_before_by_cell.values())
        raw_delta = float(raw_after - raw_before)
        self._current_global_raw_objective_value = float(raw_objective_before + raw_delta)
        reward = float(raw_delta / self._score_denominator())
        self._last_step_applied = True
        self._last_step_outcome = "applied"

        if not prefill and reward <= float(getattr(self._ctx, "global_delta_epsilon", 1.0e-6)):
            if masks_before is None or owners_before is None:
                raise RuntimeError("cannot roll back joint patch step without saved state")
            self._membership_masks = masks_before
            self._owner_by_barcode = owners_before
            self._owned_force_fill_count = owned_force_fill_before
            self._rebuild_incremental_state()
            self._current_global_raw_objective_value = float(raw_objective_before)
            self._terminated = True
            self._last_step_applied = False
            self._last_step_outcome = "rollback"
        elif prefill and not self._force_fill_complete() and not self._joint_has_add_candidate():
            self._truncated = True

        self._step_index += 1
        if self._step_index >= int(self._ctx.max_steps) and not self._terminated:
            self._truncated = True
        obs = self._build_observation()
        self._last_obs = obs
        return obs, float(reward), bool(self._terminated), bool(self._truncated), self._build_info()

    def _joint_cell_observation(self, *, cell_idx: int, global_features: Any) -> dict[str, Any]:
        torch = self._torch
        ctx = self._ctx.cells[int(cell_idx)]
        mask = self._membership_masks[int(cell_idx)]
        assigned_count = int(self._assigned_counts[int(cell_idx)])
        neighbor_support = self._compute_neighbor_support(cell_idx=int(cell_idx), mask=mask)
        frontier = ((~mask) & (neighbor_support > 0.0)) if assigned_count > 0 else (~mask)
        frontier = frontier & self._outer_masks[int(cell_idx)]
        barcode_indices = self._barcode_indices[int(cell_idx)]
        owners = self._owner_by_barcode[barcode_indices]
        filled = self._force_fill_complete()
        target_mask = torch.zeros((int(ctx.n_bins),), device=self._device, dtype=torch.bool)
        if int(self._force_fill_target_indices.shape[0]) > 0:
            target_mask = torch.isin(barcode_indices, self._force_fill_target_indices)

        add_legal = frontier & (owners < 0)
        replace_legal = frontier & (owners >= 0) & (owners != int(cell_idx))
        if not filled:
            add_legal = add_legal & target_mask
            replace_legal = replace_legal & target_mask
        elif str(getattr(self._ctx, "after_fill_actions", "add_or_stop")) == "replace_only":
            add_legal = torch.zeros_like(add_legal, dtype=torch.bool)

        add_idx = torch.nonzero(add_legal, as_tuple=False).flatten().to(dtype=torch.long)
        replace_idx = torch.nonzero(replace_legal, as_tuple=False).flatten().to(dtype=torch.long)
        legal_chunks = []
        replace_chunks = []
        old_cell_chunks = []
        old_bin_chunks = []
        if int(add_idx.numel()) > 0:
            legal_chunks.append(add_idx)
            replace_chunks.append(torch.zeros((int(add_idx.numel()),), device=self._device, dtype=torch.bool))
            old_cell_chunks.append(torch.full((int(add_idx.numel()),), -1, device=self._device, dtype=torch.long))
            old_bin_chunks.append(torch.full((int(add_idx.numel()),), -1, device=self._device, dtype=torch.long))
        if int(replace_idx.numel()) > 0:
            old_cells, old_bins = self._owner_cells_and_bins(cell_idx=int(cell_idx), bin_idx=replace_idx)
            legal_chunks.append(replace_idx)
            replace_chunks.append(torch.ones((int(replace_idx.numel()),), device=self._device, dtype=torch.bool))
            old_cell_chunks.append(old_cells)
            old_bin_chunks.append(old_bins)

        if legal_chunks:
            legal_idx = torch.cat(legal_chunks, dim=0)
            is_replace = torch.cat(replace_chunks, dim=0)
            old_cells = torch.cat(old_cell_chunks, dim=0)
            old_bins = torch.cat(old_bin_chunks, dim=0)
        else:
            legal_idx = torch.empty((0,), device=self._device, dtype=torch.long)
            is_replace = torch.empty((0,), device=self._device, dtype=torch.bool)
            old_cells = torch.empty((0,), device=self._device, dtype=torch.long)
            old_bins = torch.empty((0,), device=self._device, dtype=torch.long)

        legal = add_legal | replace_legal
        candidate_centroid_distance, candidate_compactness_gain = self._joint_candidate_geometry_features(
            cell_idx=int(cell_idx),
            ctx=ctx,
            mask=mask,
            legal=legal,
            neighbor_support=neighbor_support,
        )
        rows = torch.zeros((1 + int(legal_idx.numel()), ACTION_FEATURE_DIM), device=self._device, dtype=torch.float32)
        if int(legal_idx.numel()) > 0:
            rows[1:] = self._templates[int(cell_idx)][legal_idx + 1].clone()
            rows[1:, A_FEATURE_5] = is_replace.to(dtype=torch.float32)
            rows[1:, A_CANDIDATE_CENTROID_DISTANCE] = candidate_centroid_distance[legal_idx]
            rows[1:, A_CANDIDATE_COMPACTNESS_GAIN] = candidate_compactness_gain[legal_idx]
            rows[1:, A_CANDIDATE_NEIGHBOR_SUPPORT] = neighbor_support[legal_idx].to(dtype=torch.float32)

        action_mask = torch.ones((int(rows.shape[0]),), device=self._device, dtype=torch.bool)
        if not filled and int(add_idx.numel()) > 0:
            action_mask[0] = False
        action_barcodes = torch.full((int(rows.shape[0]),), -1, device=self._device, dtype=torch.long)
        action_cells = torch.full((int(rows.shape[0]),), int(cell_idx), device=self._device, dtype=torch.long)
        action_bins = torch.full((int(rows.shape[0]),), -1, device=self._device, dtype=torch.long)
        action_is_replace = torch.zeros((int(rows.shape[0]),), device=self._device, dtype=torch.bool)
        action_old_cells = torch.full((int(rows.shape[0]),), -1, device=self._device, dtype=torch.long)
        action_old_bins = torch.full((int(rows.shape[0]),), -1, device=self._device, dtype=torch.long)
        if int(legal_idx.numel()) > 0:
            action_barcodes[1:] = barcode_indices[legal_idx]
            action_bins[1:] = legal_idx
            action_is_replace[1:] = is_replace
            action_old_cells[1:] = old_cells
            action_old_bins[1:] = old_bins

        return {
            "global_features": global_features,
            "action_features": rows,
            "action_mask": action_mask,
            "step_index": int(self._step_index),
            "joint_action_barcodes": action_barcodes,
            "joint_action_cells": action_cells,
            "joint_action_bins": action_bins,
            "joint_action_is_replace": action_is_replace,
            "joint_action_old_cells": action_old_cells,
            "joint_action_old_bins": action_old_bins,
        }

    def _joint_has_add_candidate(self) -> bool:
        torch = self._torch
        filled = self._force_fill_complete()
        for cell_idx, ctx in enumerate(self._ctx.cells):
            mask = self._membership_masks[int(cell_idx)]
            assigned_count = int(self._assigned_counts[int(cell_idx)])
            neighbor_support = self._compute_neighbor_support(cell_idx=int(cell_idx), mask=mask)
            frontier = ((~mask) & (neighbor_support > 0.0)) if assigned_count > 0 else (~mask)
            frontier = frontier & self._outer_masks[int(cell_idx)]
            barcode_indices = self._barcode_indices[int(cell_idx)]
            owners = self._owner_by_barcode[barcode_indices]
            add_legal = frontier & (owners < 0)
            if not filled:
                if int(self._force_fill_target_indices.shape[0]) <= 0:
                    return bool(torch.any(add_legal).detach().cpu().item())
                target_mask = torch.isin(barcode_indices, self._force_fill_target_indices)
                add_legal = add_legal & target_mask
            elif str(getattr(self._ctx, "after_fill_actions", "add_or_stop")) == "replace_only":
                add_legal = torch.zeros_like(add_legal, dtype=torch.bool)
            if bool(torch.any(add_legal).detach().cpu().item()):
                return True
        return False

    def _fast_cell_summary(self, cell_idx: int) -> dict[str, float]:
        torch = self._torch
        ctx = self._ctx.cells[int(cell_idx)]
        assigned_count = int(self._assigned_counts[int(cell_idx)])
        n_bins = int(ctx.n_bins)
        if n_bins > 0:
            assigned_frac = float(assigned_count / n_bins)
            remaining_frac = float((n_bins - assigned_count) / n_bins)
        else:
            assigned_frac = 0.0
            remaining_frac = 0.0
        if assigned_count > 0:
            mask = self._membership_masks[int(cell_idx)]
            support = self._compute_neighbor_support(cell_idx=int(cell_idx), mask=mask)
            compactness_proxy = float((torch.sum(support[mask]) / float(assigned_count)).detach().cpu().item())
            assigned_ll_mean = float((self._sum_ll_mean_z[int(cell_idx)] / float(assigned_count)).detach().cpu().item())
            assigned_ll_max = float((self._sum_ll_max_z[int(cell_idx)] / float(assigned_count)).detach().cpu().item())
            centroid_xy = self._sum_xy[int(cell_idx)] / float(assigned_count)
            drift_vec = centroid_xy - self._nucleus_xy64[int(cell_idx)]
            drift_um = torch.sqrt(torch.sum(drift_vec * drift_vec))
            centroid_drift_scaled = float(
                torch.clamp(drift_um / max(float(ctx.r_max_um), 1.0e-8), min=0.0, max=1.0).detach().cpu().item()
            )
        else:
            compactness_proxy = 0.0
            assigned_ll_mean = 0.0
            assigned_ll_max = 0.0
            centroid_drift_scaled = 0.0
        return {
            "assigned_frac": assigned_frac,
            "step_frac": float(self._step_index / max(1, int(ctx.max_steps))),
            "remaining_frac": remaining_frac,
            "grow_ratio_scaled": _scale_grow_ratio_feature(assigned_count, self._initial_seed_counts[int(cell_idx)]),
            "positive_frontier_fraction": 0.0,
            "centroid_drift_scaled": centroid_drift_scaled,
            "compactness_proxy": compactness_proxy,
            "assigned_ll_mean": assigned_ll_mean,
            "assigned_ll_max": assigned_ll_max,
            "frontier_add_reward_topk_mean": 0.0,
            "frontier_add_reward_mean": 0.0,
            "frontier_add_reward_std": 0.0,
            "frontier_add_reward_max": 0.0,
        }

    def _joint_candidate_geometry_features(
        self,
        *,
        cell_idx: int,
        ctx: EpisodeContext,
        mask: Any,
        legal: Any,
        neighbor_support: Any,
    ) -> tuple[Any, Any]:
        torch = self._torch
        n_bins = int(ctx.n_bins)
        assigned_count = int(self._assigned_counts[int(cell_idx)])
        candidate_centroid_distance = torch.zeros((n_bins,), device=self._device, dtype=torch.float32)
        candidate_compactness_gain = torch.zeros((n_bins,), device=self._device, dtype=torch.float32)
        if n_bins <= 0 or not bool(torch.any(legal).detach().cpu().item()):
            return candidate_centroid_distance, candidate_compactness_gain
        if assigned_count > 0:
            current_centroid_xy = self._sum_xy[int(cell_idx)] / float(assigned_count)
            current_compactness_sum = torch.sum(neighbor_support[mask])
        else:
            current_centroid_xy = self._nucleus_xy64[int(cell_idx)]
            current_compactness_sum = torch.zeros((), device=self._device, dtype=torch.float32)
        centroid_dist_um = torch.sqrt(torch.sum((self._xy64[int(cell_idx)] - current_centroid_xy) ** 2, dim=1))
        scaled_dist = torch.clamp(centroid_dist_um / max(float(ctx.r_max_um), 1.0e-8), min=0.0, max=1.0)
        candidate_centroid_distance[legal] = scaled_dist[legal].to(dtype=torch.float32)
        if assigned_count > 0:
            current_compactness = current_compactness_sum / float(assigned_count)
            new_compactness = (current_compactness_sum + 2.0 * neighbor_support) / float(assigned_count + 1)
            candidate_compactness_gain[legal] = (new_compactness[legal] - current_compactness).to(dtype=torch.float32)
        return candidate_centroid_distance, candidate_compactness_gain


def _four_neighbor_index_from_grid_coords_np(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.int64)
    out = np.full((arr.shape[0], 4), arr.shape[0], dtype=np.int64)
    coord_to_index = {(int(x), int(y)): idx for idx, (x, y) in enumerate(arr.tolist())}
    offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for idx, (x, y) in enumerate(arr.tolist()):
        for pos, (dx, dy) in enumerate(offsets):
            out[idx, pos] = int(coord_to_index.get((int(x) + dx, int(y) + dy), arr.shape[0]))
    return out


def _torch_shape_model_tensors(shape_model: Any, *, device: Any, dtype: Any) -> _TorchShapeModelTensors:
    import torch

    priors = torch.as_tensor(np.asarray(shape_model.priors, dtype=np.float32), device=device, dtype=dtype)
    return _TorchShapeModelTensors(
        scaler_mean=torch.as_tensor(np.asarray(shape_model.scaler_mean, dtype=np.float32), device=device, dtype=dtype),
        scaler_std=torch.as_tensor(np.asarray(shape_model.scaler_std, dtype=np.float32), device=device, dtype=dtype),
        means=torch.as_tensor(np.asarray(shape_model.means, dtype=np.float32), device=device, dtype=dtype),
        inv_covariances=torch.as_tensor(
            np.asarray(shape_model.inv_covariances, dtype=np.float32),
            device=device,
            dtype=dtype,
        ),
        log_determinants=torch.as_tensor(
            np.asarray(shape_model.log_determinants, dtype=np.float32),
            device=device,
            dtype=dtype,
        ),
        log_priors=torch.log(torch.clamp(priors, min=1.0e-30)),
        n_features=int(shape_model.n_features),
    )


def _torch_grid_cell_corners(coords: Any, *, dtype: Any) -> Any:
    import torch

    if int(coords.shape[0]) == 0:
        return torch.zeros((0, 2), device=coords.device, dtype=dtype)
    offsets = torch.as_tensor(((0, 0), (1, 0), (0, 1), (1, 1)), device=coords.device, dtype=dtype)
    return (coords.to(dtype=dtype)[:, None, :] + offsets[None, :, :]).reshape(-1, 2)


def _torch_hull_area_and_boundary_points(*, points: Any, min_area: Any, epsilon: float) -> tuple[Any, Any]:
    import torch

    if int(points.shape[0]) == 0:
        return min_area.reshape(()), points.reshape(0, 2)
    pts = torch.unique(points.to(dtype=torch.long), dim=0).to(dtype=points.dtype)
    if int(pts.shape[0]) < 3:
        return torch.maximum(torch.zeros((), device=points.device, dtype=points.dtype), min_area.reshape(())), pts

    boundary = _torch_hull_boundary_mask(pts.reshape(1, pts.shape[0], 2), epsilon=epsilon)[0]
    boundary_pts = pts[boundary]
    if int(boundary_pts.shape[0]) < 3:
        return torch.maximum(torch.zeros((), device=points.device, dtype=points.dtype), min_area.reshape(())), boundary_pts

    centroid = torch.mean(boundary_pts, dim=0)
    angles = torch.atan2(boundary_pts[:, 1] - centroid[1], boundary_pts[:, 0] - centroid[0])
    order = torch.argsort(angles)
    sorted_pts = boundary_pts[order]
    shifted = torch.roll(sorted_pts, shifts=-1, dims=0)
    polygon_area = 0.5 * torch.abs(torch.sum(sorted_pts[:, 0] * shifted[:, 1] - sorted_pts[:, 1] * shifted[:, 0]))
    return torch.maximum(polygon_area, min_area.reshape(())), sorted_pts


def _torch_hull_equations_from_boundary_points(points: Any, *, epsilon: float) -> Any:
    import torch

    if int(points.shape[0]) < 3:
        return torch.zeros((0, 3), device=points.device, dtype=points.dtype)
    shifted = torch.roll(points, shifts=-1, dims=0)
    edge = shifted - points
    signed_twice_area = torch.sum(points[:, 0] * shifted[:, 1] - points[:, 1] * shifted[:, 0])
    if bool((torch.abs(signed_twice_area) <= float(epsilon)).detach().cpu().item()):
        return torch.zeros((0, 3), device=points.device, dtype=points.dtype)
    normals = torch.stack((edge[:, 1], -edge[:, 0]), dim=1)
    offsets = -torch.sum(normals * points, dim=1)
    if bool((signed_twice_area < 0.0).detach().cpu().item()):
        normals = -normals
        offsets = -offsets
    return torch.cat((normals, offsets.reshape(-1, 1)), dim=1)


def _torch_grid_cells_inside_hull(candidate_coords: Any, hull_equations: Any, *, dtype: Any, epsilon: float) -> Any:
    import torch

    n_candidates = int(candidate_coords.shape[0])
    if n_candidates == 0:
        return torch.zeros((0,), device=candidate_coords.device, dtype=torch.bool)
    if hull_equations is None or int(hull_equations.shape[0]) == 0:
        return torch.zeros((n_candidates,), device=candidate_coords.device, dtype=torch.bool)
    corners = _torch_grid_cell_corners(candidate_coords, dtype=dtype).reshape(n_candidates, 4, 2)
    equations = hull_equations.to(device=candidate_coords.device, dtype=dtype)
    signed = torch.einsum("ncd,hd->nch", corners, equations[:, :2]) + equations[:, 2].reshape(1, 1, -1)
    return torch.all(signed <= max(float(epsilon) * 10.0, 1.0e-9), dim=(1, 2))


def _torch_candidate_hull_areas(
    *,
    base_points: Any,
    base_hull_area: Any | None = None,
    base_hull_equations: Any | None = None,
    candidate_coords: Any,
    min_area: Any,
    epsilon: float,
) -> Any:
    import torch

    n_candidates = int(candidate_coords.shape[0])
    dtype = (
        base_points.dtype
        if torch.is_floating_point(base_points)
        else min_area.dtype
        if torch.is_floating_point(min_area)
        else torch.float32
    )
    if n_candidates == 0:
        return torch.zeros((0,), device=candidate_coords.device, dtype=dtype)

    out = torch.empty((n_candidates,), device=candidate_coords.device, dtype=dtype)
    inside = _torch_grid_cells_inside_hull(
        candidate_coords,
        base_hull_equations,
        dtype=dtype,
        epsilon=epsilon,
    )
    if bool(torch.any(inside).detach().cpu().item()):
        if base_hull_area is None:
            base_area = torch.zeros((), device=candidate_coords.device, dtype=dtype)
        else:
            base_area = base_hull_area.to(device=candidate_coords.device, dtype=dtype).reshape(())
        out[inside] = torch.maximum(base_area, min_area.to(dtype=dtype)[inside])

    needs_hull = ~inside
    needs_idx = torch.nonzero(needs_hull, as_tuple=False).flatten()
    if int(needs_idx.numel()) == 0:
        return out

    base = base_points.to(dtype=dtype)
    p_count = int(base.shape[0]) + 4
    max_cross_values = 8_000_000
    n_needs = int(needs_idx.numel())
    chunk_size = max(1, min(n_needs, max_cross_values // max(1, p_count * p_count * p_count)))
    hull_chunks: list[Any] = []
    for start in range(0, n_needs, chunk_size):
        stop = min(start + chunk_size, n_needs)
        idx = needs_idx[start:stop]
        coords = candidate_coords[idx]
        corners = _torch_grid_cell_corners(coords, dtype=dtype).reshape(stop - start, 4, 2)
        if int(base.shape[0]) > 0:
            points = torch.cat((base.reshape(1, base.shape[0], 2).expand(stop - start, -1, -1), corners), dim=1)
        else:
            points = corners
        hull_chunks.append(_torch_batched_hull_areas(points=points, min_area=min_area[idx], epsilon=epsilon))
    out[needs_idx] = torch.cat(hull_chunks, dim=0)
    return out


def _torch_batched_hull_areas(*, points: Any, min_area: Any, epsilon: float) -> Any:
    import torch

    if int(points.shape[1]) < 3:
        return min_area.to(dtype=points.dtype)
    boundary = _torch_hull_boundary_mask(points, epsilon=epsilon)
    counts = torch.sum(boundary.to(dtype=torch.long), dim=1)
    centroid = torch.sum(points * boundary[:, :, None].to(dtype=points.dtype), dim=1) / torch.clamp(
        counts.to(dtype=points.dtype)[:, None],
        min=1.0,
    )
    angles = torch.atan2(points[:, :, 1] - centroid[:, None, 1], points[:, :, 0] - centroid[:, None, 0])
    angles = torch.where(boundary, angles, torch.full_like(angles, 10.0))
    order = torch.argsort(angles, dim=1)
    gather_idx = order[:, :, None].expand(-1, -1, 2)
    sorted_pts = torch.gather(points, dim=1, index=gather_idx)
    ranks = torch.arange(points.shape[1], device=points.device).reshape(1, -1)
    next_pos = torch.where(ranks + 1 < counts[:, None], ranks + 1, torch.zeros_like(ranks))
    next_pts = torch.gather(sorted_pts, dim=1, index=next_pos[:, :, None].expand(-1, -1, 2))
    valid = ranks < counts[:, None]
    cross = sorted_pts[:, :, 0] * next_pts[:, :, 1] - sorted_pts[:, :, 1] * next_pts[:, :, 0]
    polygon_area = 0.5 * torch.abs(torch.sum(torch.where(valid, cross, torch.zeros_like(cross)), dim=1))
    polygon_area = torch.where(counts >= 3, polygon_area, torch.zeros_like(polygon_area))
    return torch.maximum(polygon_area, min_area.to(dtype=points.dtype))


def _torch_hull_boundary_mask(points: Any, *, epsilon: float) -> Any:
    import torch

    n_points = int(points.shape[1])
    src = points[:, :, None, None, :]
    dst = points[:, None, :, None, :]
    edge = dst - src
    rel = points[:, None, None, :, :] - src
    cross = edge[..., 0] * rel[..., 1] - edge[..., 1] * rel[..., 0]
    tol = max(float(epsilon) * 10.0, 1.0e-9)
    valid = torch.all(cross >= -tol, dim=3) | torch.all(cross <= tol, dim=3)
    edge_len = torch.sum((points[:, :, None, :] - points[:, None, :, :]) ** 2, dim=3)
    valid = valid & (edge_len > 0.0)
    eye = torch.eye(n_points, device=points.device, dtype=torch.bool).reshape(1, n_points, n_points)
    valid = valid & (~eye)
    return torch.any(valid, dim=2) | torch.any(valid, dim=1)


def _torch_shape_raw_features_from_components(
    *,
    area: Any,
    perimeter: Any,
    hull_area: Any,
    sums: Any,
    epsilon: float,
) -> Any:
    import torch

    dtype = sums.dtype if torch.is_floating_point(sums) else torch.float32
    area = area.to(dtype=dtype)
    perimeter = perimeter.to(dtype=dtype)
    hull_area = hull_area.to(dtype=dtype)
    sums = sums.to(dtype=dtype)
    compactness = (4.0 * float(np.pi) * area) / torch.clamp(perimeter * perimeter, min=float(epsilon))
    hull = torch.maximum(hull_area, area)
    solidity = torch.where(hull > float(epsilon), area / hull, torch.ones_like(hull))
    anisotropy = _torch_anisotropy_from_sums(area, sums, epsilon=epsilon)
    out = torch.stack((torch.log(area + 1.0), compactness, solidity, anisotropy), dim=1)
    return torch.where((area > 0.0).reshape(-1, 1), out, torch.zeros_like(out))


def _torch_anisotropy_from_sums(area: Any, sums: Any, *, epsilon: float) -> Any:
    import torch

    valid = area >= 3.0
    inv_n = 1.0 / torch.clamp(area, min=1.0)
    mean_x = sums[:, 0] * inv_n
    mean_y = sums[:, 1] * inv_n
    cov_xx = sums[:, 2] * inv_n - mean_x * mean_x
    cov_yy = sums[:, 3] * inv_n - mean_y * mean_y
    cov_xy = sums[:, 4] * inv_n - mean_x * mean_y
    half_trace = 0.5 * (cov_xx + cov_yy)
    disc = torch.sqrt(torch.clamp(0.25 * (cov_xx - cov_yy) * (cov_xx - cov_yy) + cov_xy * cov_xy, min=0.0))
    major = torch.clamp(half_trace + disc, min=0.0)
    minor = torch.clamp(half_trace - disc, min=0.0)
    values = torch.where(major <= float(epsilon), torch.ones_like(major), major / torch.clamp(minor, min=float(epsilon)))
    return torch.where(valid, values, torch.ones_like(values))


def _torch_shape_reward_values(raw_features: Any, model: _TorchShapeModelTensors, *, mode: str) -> Any:
    import torch

    z = (raw_features - model.scaler_mean.reshape(1, -1)) / torch.clamp(model.scaler_std.reshape(1, -1), min=1.0e-12)
    delta = z[:, None, :] - model.means[None, :, :]
    mahal = torch.einsum("nkf,kfg,nkg->nk", delta, model.inv_covariances, delta)
    ll = -0.5 * (
        float(model.n_features) * float(np.log(2.0 * np.pi))
        + model.log_determinants.reshape(1, -1)
        + mahal
    )
    value = str(mode).strip().lower()
    if value == "mixture":
        return torch.logsumexp(model.log_priors.reshape(1, -1) + ll, dim=1)
    if value in {"max", "best"}:
        return torch.max(ll, dim=1).values
    raise ValueError("shape_prior_mode must be 'mixture' or 'max'")
