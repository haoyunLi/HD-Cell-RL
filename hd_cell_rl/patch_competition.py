"""Competition-margin reward helpers for patch environments."""

from __future__ import annotations

import numpy as np

from .patch_types import PatchContext
from .ppo_feature_schema import A_COMPETITION_MARGIN
from .ppo_state import EpisodeContext
from .shape_prior import compute_delta_shape_rewards_for_candidates


_COMPETITION_MARGIN_FEATURE_SCALE = 5.0


def _competition_enabled(cells: tuple[EpisodeContext, ...]) -> bool:
    return any(float(getattr(ctx, "competition_margin_weight", 0.0)) > 0.0 for ctx in cells)


def _empty_competition_candidates(
    context: PatchContext,
) -> tuple[tuple[tuple[tuple[int, int], ...], ...], ...]:
    return tuple(tuple(() for _ in range(int(ctx.n_bins))) for ctx in context.cells)


def _competition_feature_from_margin_np(margins: np.ndarray) -> np.ndarray:
    values = np.asarray(margins, dtype=np.float32) / np.float32(_COMPETITION_MARGIN_FEATURE_SCALE)
    return np.clip(values, -1.0, 1.0).astype(np.float32, copy=False)


def _stop_delta_from_values_np(values: np.ndarray, *, stop_stat: str, stop_top_k: int) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    mode = str(stop_stat)
    if mode == "max":
        return float(np.max(arr))
    if mode == "topk_mean":
        k = min(max(int(stop_top_k), 1), int(arr.size))
        return float(np.mean(np.partition(arr, arr.size - k)[arr.size - k :]))
    raise ValueError("reward.stop_stat must be 'max' or 'topk_mean'")


def _summary_with_action_rewards(
    base_summary: dict[str, float],
    rewards: np.ndarray,
    *,
    stop_top_k: int,
) -> dict[str, float]:
    out = dict(base_summary)
    arr = np.asarray(rewards, dtype=np.float64)
    if arr.size == 0:
        out["positive_frontier_fraction"] = 0.0
        out["frontier_add_reward_mean"] = 0.0
        out["frontier_add_reward_std"] = 0.0
        out["frontier_add_reward_max"] = 0.0
        out["frontier_add_reward_topk_mean"] = 0.0
        return out

    out["positive_frontier_fraction"] = float(np.mean(arr > 0.0))
    out["frontier_add_reward_mean"] = float(np.mean(arr))
    out["frontier_add_reward_std"] = float(np.std(arr, ddof=0))
    out["frontier_add_reward_max"] = float(np.max(arr))
    out["frontier_add_reward_topk_mean"] = _stop_delta_from_values_np(
        arr,
        stop_stat="topk_mean",
        stop_top_k=int(stop_top_k),
    )
    return out


def _build_competition_candidates(
    context: PatchContext,
) -> tuple[tuple[tuple[tuple[int, int], ...], ...], ...]:
    """Precompute same-bin competitors whose nuclei are within the configured radius."""
    barcode_to_bins: dict[str, list[tuple[int, int]]] = {}
    for cell_idx, ctx in enumerate(context.cells):
        for bin_idx, raw_barcode in enumerate(ctx.candidate_bin_ids):
            barcode_to_bins.setdefault(str(raw_barcode), []).append((int(cell_idx), int(bin_idx)))

    all_edges: list[tuple[tuple[tuple[int, int], ...], ...]] = []
    for cell_idx, ctx in enumerate(context.cells):
        cell_edges: list[tuple[tuple[int, int], ...]] = []
        weight = float(getattr(ctx, "competition_margin_weight", 0.0))
        radius = float(getattr(ctx, "competition_margin_radius_um", 20.0))
        radius2 = radius * radius
        for bin_idx, raw_barcode in enumerate(ctx.candidate_bin_ids):
            if weight <= 0.0:
                cell_edges.append(())
                continue
            xy = np.asarray(ctx.candidate_bin_xy_um[int(bin_idx)], dtype=np.float64)
            competitors: list[tuple[int, int]] = []
            for other_cell_idx, other_bin_idx in barcode_to_bins.get(str(raw_barcode), ()):
                if int(other_cell_idx) == int(cell_idx):
                    continue
                other_ctx = context.cells[int(other_cell_idx)]
                center = np.asarray(other_ctx.nucleus_center_xy_um, dtype=np.float64)
                dist2 = float(np.sum((xy - center) * (xy - center)))
                if dist2 <= radius2 + 1.0e-8:
                    competitors.append((int(other_cell_idx), int(other_bin_idx)))
            cell_edges.append(tuple(competitors))
        all_edges.append(tuple(cell_edges))
    return tuple(all_edges)


def _zscore_values_np(values: np.ndarray, *, zscore_delta: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr.astype(np.float32, copy=False)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=0))
    if sigma <= float(zscore_delta):
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mu) / (sigma + float(zscore_delta))).astype(np.float32, copy=False)


def _competition_pair_indices_by_cell(
    *,
    competition_candidates: tuple[tuple[tuple[tuple[int, int], ...], ...], ...],
    add_map: list[tuple[int, int, float]],
) -> dict[int, set[int]]:
    pairs: dict[int, set[int]] = {}
    for cell_idx, bin_idx, _ in add_map:
        cell_idx_i = int(cell_idx)
        bin_idx_i = int(bin_idx)
        pairs.setdefault(cell_idx_i, set()).add(bin_idx_i)
        for other_cell_idx, other_bin_idx in competition_candidates[cell_idx_i][bin_idx_i]:
            pairs.setdefault(int(other_cell_idx), set()).add(int(other_bin_idx))
    return pairs


def _competition_shape_raw_by_cell(
    *,
    cells: tuple[EpisodeContext, ...],
    pair_indices_by_cell: dict[int, set[int]],
    membership_masks_by_cell: dict[int, np.ndarray] | None,
) -> dict[int, np.ndarray]:
    masks = membership_masks_by_cell or {}
    out: dict[int, np.ndarray] = {}
    for cell_idx, indices in pair_indices_by_cell.items():
        ctx = cells[int(cell_idx)]
        n_bins = int(ctx.n_bins)
        raw = np.zeros((n_bins,), dtype=np.float32)
        if (
            indices
            and int(cell_idx) in masks
            and ctx.shape_prior_model is not None
            and float(getattr(ctx, "shape_prior_weight", 0.0)) > 0.0
        ):
            shape_raw, _, _ = compute_delta_shape_rewards_for_candidates(
                membership_mask=np.asarray(masks[int(cell_idx)], dtype=np.uint8),
                candidate_indices=np.asarray(sorted(indices), dtype=np.int64),
                candidate_bin_xy_um=ctx.candidate_bin_xy_um,
                shape_model=ctx.shape_prior_model,
                mode=str(ctx.shape_prior_mode),
                bin_size_um=float(ctx.shape_prior_bin_size_um),
                epsilon=1.0e-8,
            )
            raw = shape_raw.astype(np.float32, copy=False)
        out[int(cell_idx)] = raw
    return out


def _competition_scores_by_cell_np(
    *,
    cells: tuple[EpisodeContext, ...],
    competition_candidates: tuple[tuple[tuple[tuple[int, int], ...], ...], ...],
    add_map: list[tuple[int, int, float]],
    competition_expr_by_cell: dict[int, np.ndarray] | None,
    membership_masks_by_cell: dict[int, np.ndarray] | None,
) -> dict[int, np.ndarray]:
    pair_indices_by_cell = _competition_pair_indices_by_cell(
        competition_candidates=competition_candidates,
        add_map=add_map,
    )
    if not pair_indices_by_cell:
        return {}

    expr_inputs = {
        int(cell_idx): np.asarray(values, dtype=np.float32)
        for cell_idx, values in (competition_expr_by_cell or {}).items()
    }
    shape_raw_by_cell = _competition_shape_raw_by_cell(
        cells=cells,
        pair_indices_by_cell=pair_indices_by_cell,
        membership_masks_by_cell=membership_masks_by_cell,
    )

    entries: list[tuple[int, int]] = []
    expr_values: list[float] = []
    shape_values: list[float] = []
    for cell_idx in sorted(pair_indices_by_cell):
        ctx = cells[int(cell_idx)]
        expr = expr_inputs.get(int(cell_idx), np.zeros((int(ctx.n_bins),), dtype=np.float32))
        shape = shape_raw_by_cell.get(int(cell_idx), np.zeros((int(ctx.n_bins),), dtype=np.float32))
        for bin_idx in sorted(pair_indices_by_cell[int(cell_idx)]):
            bin_idx_i = int(bin_idx)
            if bin_idx_i < 0 or bin_idx_i >= int(ctx.n_bins):
                continue
            entries.append((int(cell_idx), bin_idx_i))
            expr_values.append(float(expr[bin_idx_i]) if bin_idx_i < int(expr.shape[0]) else 0.0)
            shape_values.append(float(shape[bin_idx_i]) if bin_idx_i < int(shape.shape[0]) else 0.0)

    if not entries:
        return {}

    zscore_delta = max(float(getattr(ctx, "zscore_delta", 1.0e-8)) for ctx in cells)
    expr_z = _zscore_values_np(np.asarray(expr_values, dtype=np.float32), zscore_delta=zscore_delta)
    shape_z = _zscore_values_np(np.asarray(shape_values, dtype=np.float32), zscore_delta=zscore_delta)

    scores = {cell_idx: np.zeros((int(ctx.n_bins),), dtype=np.float32) for cell_idx, ctx in enumerate(cells)}
    for entry_i, (cell_idx, bin_idx) in enumerate(entries):
        ctx = cells[int(cell_idx)]
        score = (
            float(ctx.w1) * float(expr_z[entry_i])
            + float(ctx.shape_prior_weight) * float(shape_z[entry_i])
            - float(ctx.base_penalty[int(bin_idx)])
        )
        scores[int(cell_idx)][int(bin_idx)] = np.float32(score)
    return scores


def _compute_competition_adjusted_rewards_np(
    *,
    cells: tuple[EpisodeContext, ...],
    competition_candidates: tuple[tuple[tuple[tuple[int, int], ...], ...], ...],
    add_map: list[tuple[int, int, float]],
    competition_expr_by_cell: dict[int, np.ndarray] | None = None,
    membership_masks_by_cell: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    if not add_map:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            {},
        )

    legal_scores = {(int(cell_idx), int(bin_idx)): float(score) for cell_idx, bin_idx, score in add_map}
    competition_scores = _competition_scores_by_cell_np(
        cells=cells,
        competition_candidates=competition_candidates,
        add_map=add_map,
        competition_expr_by_cell=competition_expr_by_cell,
        membership_masks_by_cell=membership_masks_by_cell,
    )
    adjusted_rewards = np.zeros((len(add_map),), dtype=np.float32)
    margins = np.zeros((len(add_map),), dtype=np.float32)
    rewards_by_cell: dict[int, list[float]] = {}

    for action_i, (cell_idx, bin_idx, base_score) in enumerate(add_map):
        cell_idx_i = int(cell_idx)
        bin_idx_i = int(bin_idx)
        base = float(base_score)
        margin = 0.0
        best_other: float | None = None
        current_score = base
        if cell_idx_i in competition_scores:
            scores = competition_scores[cell_idx_i]
            if 0 <= bin_idx_i < int(scores.shape[0]):
                current_score = float(scores[bin_idx_i])
        weight = float(getattr(cells[cell_idx_i], "competition_margin_weight", 0.0))
        if weight > 0.0:
            for other_cell_idx, other_bin_idx in competition_candidates[cell_idx_i][bin_idx_i]:
                other_cell_idx_i = int(other_cell_idx)
                other_bin_idx_i = int(other_bin_idx)
                if other_cell_idx_i in competition_scores:
                    scores = competition_scores[other_cell_idx_i]
                    if other_bin_idx_i < 0 or other_bin_idx_i >= int(scores.shape[0]):
                        continue
                    other_score = float(scores[other_bin_idx_i])
                else:
                    other_score = legal_scores.get((other_cell_idx_i, other_bin_idx_i))
                    if other_score is None:
                        continue
                if best_other is None or float(other_score) > best_other:
                    best_other = float(other_score)
            if best_other is not None:
                margin = current_score - best_other
                margin = float(
                    np.clip(
                        margin,
                        -float(getattr(cells[cell_idx_i], "competition_margin_clip", 5.0)),
                        float(getattr(cells[cell_idx_i], "competition_margin_clip", 5.0)),
                    )
                )

        adjusted = base + weight * margin
        adjusted_rewards[action_i] = np.float32(adjusted)
        margins[action_i] = np.float32(margin)
        rewards_by_cell.setdefault(cell_idx_i, []).append(float(adjusted))

    return (
        adjusted_rewards,
        margins,
        {cell_idx: np.asarray(values, dtype=np.float32) for cell_idx, values in rewards_by_cell.items()},
    )


def _apply_competition_margin_np(
    *,
    cells: tuple[EpisodeContext, ...],
    competition_candidates: tuple[tuple[tuple[tuple[int, int], ...], ...], ...],
    add_map: list[tuple[int, int, float]],
    add_rows: list[np.ndarray],
    competition_expr_by_cell: dict[int, np.ndarray] | None = None,
    membership_masks_by_cell: dict[int, np.ndarray] | None = None,
) -> tuple[list[tuple[int, int, float]], list[np.ndarray], dict[int, np.ndarray]]:
    """Apply counterfactual best-other-cell margin to legal ADD actions."""
    if not add_map:
        return [], [], {}

    adjusted_rewards, margins, rewards_by_cell = _compute_competition_adjusted_rewards_np(
        cells=cells,
        competition_candidates=competition_candidates,
        add_map=add_map,
        competition_expr_by_cell=competition_expr_by_cell,
        membership_masks_by_cell=membership_masks_by_cell,
    )
    adjusted_map: list[tuple[int, int, float]] = []
    adjusted_rows: list[np.ndarray] = []

    for action_i, (cell_idx, bin_idx, base_score) in enumerate(add_map):
        cell_idx_i = int(cell_idx)
        bin_idx_i = int(bin_idx)
        row = np.asarray(add_rows[action_i], dtype=np.float32).copy()
        row[A_COMPETITION_MARGIN] = _competition_feature_from_margin_np(margins[action_i : action_i + 1])[0]
        adjusted_rows.append(row)
        adjusted_map.append((cell_idx_i, bin_idx_i, float(adjusted_rewards[action_i])))

    return adjusted_map, adjusted_rows, rewards_by_cell
