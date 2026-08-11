"""CPU patch environments."""

from __future__ import annotations

from typing import Any

import numpy as np

from .patch_competition import (
    _apply_competition_margin_np,
    _build_competition_candidates,
    _competition_enabled,
    _empty_competition_candidates,
    _stop_delta_from_values_np,
    _summary_with_action_rewards,
)
from .patch_env_features import _aggregate_global_features, _stop_action_features_from_global
from .patch_types import PatchContext, _PatchCellObservation
from .ppo_feature_schema import ACTION_FEATURE_DIM
from .ppo_state import (
    _build_static_action_template,
    _compute_state_feature_bundle,
    _fill_dynamic_action_features,
    _scale_seed_size_feature,
    _state_summary_from_bundle,
)
from .reward import compute_frontier_eligible_mask, compute_stop_delta


class MultiCellPatchEnv:
    """Patch-level ADD/STOP environment with unique barcode ownership."""

    def __init__(self, context: PatchContext) -> None:
        self._ctx = context
        self._core_cell_ids = set(context.core_cell_ids)
        self._cell_ids = tuple(ctx.cell_id for ctx in context.cells)
        self._cell_index_by_id = {cell_id: idx for idx, cell_id in enumerate(self._cell_ids)}
        self._outer_masks = tuple(context.outer_bounds.contains_xy(ctx.candidate_bin_xy_um) for ctx in context.cells)
        self._templates = tuple(
            _build_static_action_template(
                ctx,
                n_bins_scaled=float(np.log1p(ctx.n_bins) / 8.0),
                seed_size_scaled=_scale_seed_size_feature(int(np.sum(ctx.initial_membership_mask))),
            )
            for ctx in context.cells
        )
        self._competition_enabled = bool(context.competition_margin_enabled) and _competition_enabled(context.cells)
        self._competition_affects_stop = any(
            bool(getattr(ctx, "competition_margin_affects_stop", True)) for ctx in context.cells
        )
        self._competition_candidates = (
            _build_competition_candidates(context)
            if self._competition_enabled
            else _empty_competition_candidates(context)
        )
        self._force_fill_target_barcodes = set(str(item) for item in context.force_fill_target_barcodes)
        self._owned_force_fill_count = 0
        self._membership_masks: list[np.ndarray] = []
        self._owner_by_barcode: dict[str, str] = {}
        self._step_index = 0
        self._terminated = False
        self._truncated = False
        self._terminated_by_stop = False
        self._cell_rewards: dict[str, float] = {}
        self._stop_reward_value = 0.0
        self._cached_action_map: list[tuple[int, int, float]] = []
        self._last_obs: dict[str, Any] | None = None

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        self._membership_masks = [
            np.asarray(ctx.initial_membership_mask, dtype=np.uint8).copy() for ctx in self._ctx.cells
        ]
        self._owner_by_barcode = {}
        self._step_index = 0
        self._terminated = False
        self._truncated = False
        self._terminated_by_stop = False
        self._cell_rewards = {cell_id: 0.0 for cell_id in self._cell_ids}
        self._stop_reward_value = 0.0
        self._assign_initial_seed_owners()
        self._owned_force_fill_count = self._count_owned_force_fill_barcodes()
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
            cell_idx, bin_idx, reward = self._cached_action_map[action_i - 1]
            ctx = self._ctx.cells[cell_idx]
            barcode = str(ctx.candidate_bin_ids[bin_idx])
            if barcode in self._owner_by_barcode:
                raise ValueError(f"invalid ADD action: barcode already owned: {barcode}")
            self._owner_by_barcode[barcode] = str(ctx.cell_id)
            if barcode in self._force_fill_target_barcodes:
                self._owned_force_fill_count += 1
            self._membership_masks[cell_idx][bin_idx] = np.uint8(1)
            self._cell_rewards[str(ctx.cell_id)] += float(reward)
            if self._force_fill_enabled() and self._force_fill_complete():
                self._terminated = True

        self._step_index += 1
        if self._step_index >= int(self._ctx.max_steps) and not self._terminated:
            self._truncated = True

        obs = self._build_observation()
        self._last_obs = obs
        return obs, float(reward), bool(self._terminated), bool(self._truncated), self._build_info()

    def final_masks(self) -> dict[str, np.ndarray]:
        return {
            str(ctx.cell_id): np.asarray(self._membership_masks[idx], dtype=np.uint8).copy()
            for idx, ctx in enumerate(self._ctx.cells)
        }

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
        return int(len(self._force_fill_target_barcodes))

    def _force_fill_complete(self) -> bool:
        target_count = self._force_fill_target_count()
        return target_count <= 0 or int(self._owned_force_fill_count) >= target_count

    def _has_legal_add_actions(self) -> bool:
        return bool(len(self._cached_action_map) > 0)

    def _count_owned_force_fill_barcodes(self) -> int:
        if not self._force_fill_target_barcodes:
            return 0
        return int(sum(1 for barcode in self._force_fill_target_barcodes if barcode in self._owner_by_barcode))

    def _apply_force_fill_stop_mask(self, action_mask: np.ndarray, *, n_add_actions: int) -> None:
        if (
            self._force_fill_enabled()
            and str(getattr(self._ctx, "stop_action_mode", "enabled")) == "mask_until_filled"
            and not self._force_fill_complete()
            and int(n_add_actions) > 0
        ):
            action_mask[0] = False

    def _assign_initial_seed_owners(self) -> None:
        proposals: dict[str, list[tuple[float, str, int, int]]] = {}
        for cell_idx, ctx in enumerate(self._ctx.cells):
            seed_idx = np.flatnonzero(np.asarray(ctx.initial_membership_mask, dtype=np.uint8) > 0)
            for bin_idx in seed_idx.tolist():
                barcode = str(ctx.candidate_bin_ids[int(bin_idx)])
                xy = np.asarray(ctx.candidate_bin_xy_um[int(bin_idx)], dtype=np.float64)
                center = np.asarray(ctx.nucleus_center_xy_um, dtype=np.float64)
                dist = float(np.sqrt(np.sum((xy - center) ** 2)))
                proposals.setdefault(barcode, []).append((dist, str(ctx.cell_id), cell_idx, int(bin_idx)))

        for barcode, candidates in proposals.items():
            candidates.sort(key=lambda item: (item[0], item[1]))
            _, owner, owner_cell_idx, owner_bin_idx = candidates[0]
            self._owner_by_barcode[barcode] = owner
            for _, _, cell_idx, bin_idx in candidates[1:]:
                self._membership_masks[cell_idx][bin_idx] = np.uint8(0)
            self._membership_masks[owner_cell_idx][owner_bin_idx] = np.uint8(1)

    def _build_observation(self) -> dict[str, Any]:
        summaries: list[dict[str, float]] = []
        cell_summaries: dict[int, dict[str, float]] = {}
        add_rows: list[np.ndarray] = []
        add_map: list[tuple[int, int, float]] = []
        competition_expr_by_cell: dict[int, np.ndarray] = {}
        membership_masks_by_cell: dict[int, np.ndarray] = {}
        stop_terms: list[float] = []
        n_blocked = 0
        n_frontier = 0
        n_legal = 0

        for cell_idx, ctx in enumerate(self._ctx.cells):
            mask = self._membership_masks[cell_idx]
            frontier = compute_frontier_eligible_mask(mask, ctx.neighbor_index)
            legal = np.asarray(frontier, dtype=bool).copy()
            legal &= self._outer_masks[cell_idx]
            frontier_idx = np.flatnonzero(legal)
            for bin_idx in frontier_idx.tolist():
                barcode = str(ctx.candidate_bin_ids[int(bin_idx)])
                if barcode in self._owner_by_barcode:
                    legal[int(bin_idx)] = False

            n_frontier += int(np.sum(frontier))
            n_legal += int(np.sum(legal))
            n_blocked += int(np.sum(frontier) - np.sum(legal))
            bundle = _compute_state_feature_bundle(
                ctx=ctx,
                membership_mask=mask,
                step_index=self._step_index,
                frontier_mask=legal,
            )
            competition_expr_by_cell[int(cell_idx)] = np.asarray(bundle.expr_raw, dtype=np.float32)
            membership_masks_by_cell[int(cell_idx)] = np.asarray(mask, dtype=np.uint8)
            if str(ctx.cell_id) in self._core_cell_ids:
                summary = _state_summary_from_bundle(bundle)
                summaries.append(summary)
                cell_summaries[int(cell_idx)] = summary
                stop_terms.append(
                    -float(ctx.stop_lambda)
                    * compute_stop_delta(
                        bundle.add_rewards,
                        legal,
                        stop_stat=str(ctx.stop_stat),
                        stop_top_k=int(ctx.stop_top_k),
                    )
                )

            action_features = self._templates[cell_idx].copy()
            _fill_dynamic_action_features(
                action_features=action_features,
                ctx=ctx,
                membership_mask=mask,
                bundle=bundle,
            )
            for bin_idx in np.flatnonzero(legal).tolist():
                add_rows.append(action_features[int(bin_idx) + 1].astype(np.float32, copy=True))
                add_map.append((cell_idx, int(bin_idx), float(bundle.add_rewards[int(bin_idx)])))

        if self._competition_enabled and add_map:
            add_map, add_rows, adjusted_rewards_by_cell = _apply_competition_margin_np(
                cells=self._ctx.cells,
                competition_candidates=self._competition_candidates,
                add_map=add_map,
                add_rows=add_rows,
                competition_expr_by_cell=competition_expr_by_cell,
                membership_masks_by_cell=membership_masks_by_cell,
            )
            if self._competition_affects_stop:
                summaries = []
                stop_terms = []
                for cell_idx, base_summary in cell_summaries.items():
                    ctx = self._ctx.cells[int(cell_idx)]
                    values = adjusted_rewards_by_cell.get(int(cell_idx), np.zeros((0,), dtype=np.float32))
                    summaries.append(
                        _summary_with_action_rewards(
                            base_summary,
                            values,
                            stop_top_k=int(ctx.stop_top_k),
                        )
                    )
                    stop_terms.append(
                        -float(ctx.stop_lambda)
                        * _stop_delta_from_values_np(
                            values,
                            stop_stat=str(ctx.stop_stat),
                            stop_top_k=int(ctx.stop_top_k),
                        )
                    )

        global_features = _aggregate_global_features(
            summaries=summaries,
            total_bins=sum(ctx.n_bins for ctx in self._ctx.cells),
            total_seed_bins=sum(int(np.sum(ctx.initial_membership_mask)) for ctx in self._ctx.cells),
            step_index=self._step_index,
            max_steps=self._ctx.max_steps,
        )
        stop_row = _stop_action_features_from_global(global_features)
        action_features = np.zeros((1 + len(add_rows), ACTION_FEATURE_DIM), dtype=np.float32)
        action_features[0] = stop_row
        if add_rows:
            action_features[1:] = np.vstack(add_rows).astype(np.float32, copy=False)
        action_mask = np.ones((action_features.shape[0],), dtype=bool)
        self._apply_force_fill_stop_mask(action_mask, n_add_actions=len(add_map))

        self._cached_action_map = add_map
        self._stop_reward_value = float(np.mean(stop_terms)) if stop_terms else 0.0
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


class CachedMultiCellPatchEnv(MultiCellPatchEnv):
    """CPU patch environment that caches unaffected per-cell observations."""

    def __init__(self, context: PatchContext) -> None:
        super().__init__(context)
        barcode_to_cells: dict[str, set[int]] = {}
        for cell_idx, ctx in enumerate(context.cells):
            for barcode in ctx.candidate_bin_ids:
                barcode_to_cells.setdefault(str(barcode), set()).add(int(cell_idx))
        self._barcode_to_cell_indices = {
            barcode: tuple(sorted(cell_indices)) for barcode, cell_indices in barcode_to_cells.items()
        }
        self._cached_cell_observations: list[_PatchCellObservation] = []

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        self._membership_masks = [
            np.asarray(ctx.initial_membership_mask, dtype=np.uint8).copy() for ctx in self._ctx.cells
        ]
        self._owner_by_barcode = {}
        self._step_index = 0
        self._terminated = False
        self._truncated = False
        self._terminated_by_stop = False
        self._cell_rewards = {cell_id: 0.0 for cell_id in self._cell_ids}
        self._stop_reward_value = 0.0
        self._assign_initial_seed_owners()
        self._owned_force_fill_count = self._count_owned_force_fill_barcodes()
        self._cached_cell_observations = [
            self._compute_cell_observation(cell_idx) for cell_idx in range(len(self._ctx.cells))
        ]
        obs = self._compose_observation()
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
            cell_idx, bin_idx, reward = self._cached_action_map[action_i - 1]
            ctx = self._ctx.cells[cell_idx]
            barcode = str(ctx.candidate_bin_ids[bin_idx])
            if barcode in self._owner_by_barcode:
                raise ValueError(f"invalid ADD action: barcode already owned: {barcode}")
            self._owner_by_barcode[barcode] = str(ctx.cell_id)
            if barcode in self._force_fill_target_barcodes:
                self._owned_force_fill_count += 1
            self._membership_masks[cell_idx][bin_idx] = np.uint8(1)
            self._cell_rewards[str(ctx.cell_id)] += float(reward)
            affected_cells = self._barcode_to_cell_indices.get(barcode, (cell_idx,))
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

    def _compute_cell_observation(self, cell_idx: int) -> _PatchCellObservation:
        ctx = self._ctx.cells[cell_idx]
        mask = self._membership_masks[cell_idx]
        frontier = compute_frontier_eligible_mask(mask, ctx.neighbor_index)
        legal = np.asarray(frontier, dtype=bool).copy()
        legal &= self._outer_masks[cell_idx]
        frontier_idx = np.flatnonzero(legal)
        for bin_idx in frontier_idx.tolist():
            barcode = str(ctx.candidate_bin_ids[int(bin_idx)])
            if barcode in self._owner_by_barcode:
                legal[int(bin_idx)] = False

        bundle = _compute_state_feature_bundle(
            ctx=ctx,
            membership_mask=mask,
            step_index=self._step_index,
            frontier_mask=legal,
        )
        summary = None
        stop_term = None
        if str(ctx.cell_id) in self._core_cell_ids:
            summary = _state_summary_from_bundle(bundle)
            stop_term = -float(ctx.stop_lambda) * compute_stop_delta(
                bundle.add_rewards,
                legal,
                stop_stat=str(ctx.stop_stat),
                stop_top_k=int(ctx.stop_top_k),
            )

        action_features = self._templates[cell_idx].copy()
        _fill_dynamic_action_features(
            action_features=action_features,
            ctx=ctx,
            membership_mask=mask,
            bundle=bundle,
        )
        add_rows: list[np.ndarray] = []
        add_map: list[tuple[int, int, float]] = []
        for bin_idx in np.flatnonzero(legal).tolist():
            add_rows.append(action_features[int(bin_idx) + 1].astype(np.float32, copy=True))
            add_map.append((cell_idx, int(bin_idx), float(bundle.add_rewards[int(bin_idx)])))

        if add_rows:
            rows = np.vstack(add_rows).astype(np.float32, copy=False)
        else:
            rows = np.zeros((0, ACTION_FEATURE_DIM), dtype=np.float32)
        n_frontier = int(np.sum(frontier))
        n_legal = int(np.sum(legal))
        return _PatchCellObservation(
            summary=summary,
            add_rows=rows,
            add_map=tuple(add_map),
            competition_expr_raw=np.asarray(bundle.expr_raw, dtype=np.float32),
            stop_term=stop_term,
            n_frontier=n_frontier,
            n_legal=n_legal,
            n_blocked=int(n_frontier - n_legal),
        )

    def _compose_observation(self) -> dict[str, Any]:
        summaries = [item.summary for item in self._cached_cell_observations if item.summary is not None]
        stop_terms = [float(item.stop_term) for item in self._cached_cell_observations if item.stop_term is not None]
        add_rows = [item.add_rows for item in self._cached_cell_observations if int(item.add_rows.shape[0]) > 0]
        add_map: list[tuple[int, int, float]] = []
        competition_expr_by_cell: dict[int, np.ndarray] = {}
        membership_masks_by_cell: dict[int, np.ndarray] = {}
        for item in self._cached_cell_observations:
            add_map.extend(item.add_map)
        for cell_idx, item in enumerate(self._cached_cell_observations):
            competition_expr_by_cell[int(cell_idx)] = np.asarray(item.competition_expr_raw, dtype=np.float32)
            membership_masks_by_cell[int(cell_idx)] = np.asarray(self._membership_masks[int(cell_idx)], dtype=np.uint8)

        if self._competition_enabled and add_map:
            flat_add_rows: list[np.ndarray] = []
            for item in self._cached_cell_observations:
                if int(item.add_rows.shape[0]) > 0:
                    flat_add_rows.extend([row for row in item.add_rows])
            add_map, adjusted_rows, adjusted_rewards_by_cell = _apply_competition_margin_np(
                cells=self._ctx.cells,
                competition_candidates=self._competition_candidates,
                add_map=add_map,
                add_rows=flat_add_rows,
                competition_expr_by_cell=competition_expr_by_cell,
                membership_masks_by_cell=membership_masks_by_cell,
            )
            if self._competition_affects_stop:
                summaries = []
                stop_terms = []
                for cell_idx, item in enumerate(self._cached_cell_observations):
                    if item.summary is None:
                        continue
                    ctx = self._ctx.cells[int(cell_idx)]
                    values = adjusted_rewards_by_cell.get(int(cell_idx), np.zeros((0,), dtype=np.float32))
                    summaries.append(
                        _summary_with_action_rewards(
                            item.summary,
                            values,
                            stop_top_k=int(ctx.stop_top_k),
                        )
                    )
                    stop_terms.append(
                        -float(ctx.stop_lambda)
                        * _stop_delta_from_values_np(
                            values,
                            stop_stat=str(ctx.stop_stat),
                            stop_top_k=int(ctx.stop_top_k),
                        )
                    )
            add_rows = [np.vstack(adjusted_rows).astype(np.float32, copy=False)] if adjusted_rows else []

        global_features = _aggregate_global_features(
            summaries=summaries,
            total_bins=sum(ctx.n_bins for ctx in self._ctx.cells),
            total_seed_bins=sum(int(np.sum(ctx.initial_membership_mask)) for ctx in self._ctx.cells),
            step_index=self._step_index,
            max_steps=self._ctx.max_steps,
        )
        stop_row = _stop_action_features_from_global(global_features)
        n_add_rows = sum(int(rows.shape[0]) for rows in add_rows)
        action_features = np.zeros((1 + n_add_rows, ACTION_FEATURE_DIM), dtype=np.float32)
        action_features[0] = stop_row
        if add_rows:
            action_features[1:] = np.vstack(add_rows).astype(np.float32, copy=False)
        action_mask = np.ones((action_features.shape[0],), dtype=bool)
        self._apply_force_fill_stop_mask(action_mask, n_add_actions=len(add_map))

        self._cached_action_map = add_map
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
