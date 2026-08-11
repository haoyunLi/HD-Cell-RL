"""Local multi-cell greedy competition over existing per-cell rewards.

This module is intentionally evaluator-only. It does not train a policy; it
uses the same per-cell ADD reward terms as PPO, but lets neighboring cells
compete for shared candidate barcodes inside one local patch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from .ppo_state import EpisodeContext, _compute_state_feature_bundle
from .reward import compute_frontier_eligible_mask


_SQUARE_002UM_BARCODE_RE = re.compile(r"^s_\d+um_(\d+)_(\d+)(?:-\d+)?$")


@dataclass
class GreedyCellState:
    """Mutable state for one cell inside a local competition patch."""

    cell_id: str
    context: EpisodeContext
    membership_mask: np.ndarray
    barcode_to_index: dict[str, int]

    @classmethod
    def from_context(cls, ctx: EpisodeContext) -> "GreedyCellState":
        mask = np.asarray(ctx.initial_membership_mask, dtype=np.uint8).copy()
        if mask.shape != (ctx.n_bins,):
            raise ValueError(f"initial mask shape mismatch for cell {ctx.cell_id!r}")
        return cls(
            cell_id=str(ctx.cell_id),
            context=ctx,
            membership_mask=mask,
            barcode_to_index={str(barcode): idx for idx, barcode in enumerate(ctx.candidate_bin_ids)},
        )


@dataclass(frozen=True)
class GreedyActionRecord:
    """One selected ADD action in a multi-cell greedy patch."""

    step_index: int
    cell_id: str
    barcode: str
    candidate_index: int
    add_score: float


@dataclass(frozen=True)
class GreedyPatchResult:
    """Final ownership and diagnostics for one local competition patch."""

    target_cell_id: str
    steps: tuple[GreedyActionRecord, ...]
    owner_by_barcode: dict[str, str]
    owner_is_nuclear_by_barcode: dict[str, bool]
    n_patch_cells: int
    n_initial_owner_conflicts: int
    n_contested_candidate_barcodes: int
    n_blocked_frontier_actions: int
    stop_reason: str


ScoreFunction = Callable[[GreedyCellState, np.ndarray, int], np.ndarray]


def default_multicell_score_fn(
    state: GreedyCellState,
    legal_frontier: np.ndarray,
    step_index: int,
) -> np.ndarray:
    """Return current ADD rewards for one cell, masked to legal frontier bins."""
    bundle = _compute_state_feature_bundle(
        ctx=state.context,
        membership_mask=state.membership_mask,
        step_index=step_index,
        frontier_mask=np.asarray(legal_frontier, dtype=bool),
    )
    return np.asarray(bundle.add_rewards, dtype=np.float32)


def run_multicell_greedy_patch(
    *,
    target_cell_id: str,
    cell_states: list[GreedyCellState],
    max_steps: int,
    min_add_score: float = 0.0,
    score_fn: ScoreFunction = default_multicell_score_fn,
) -> GreedyPatchResult:
    """Run greedy ADD competition among cells in one local patch.

    At each step, every cell proposes its best legal frontier bin using the
    existing single-cell ADD reward. The patch accepts the single highest
    positive proposal across all cells. A barcode already owned by another
    cell cannot be stolen.
    """
    if max_steps < 0:
        raise ValueError("max_steps must be >= 0")
    if not cell_states:
        raise ValueError("cell_states must be non-empty")
    if str(target_cell_id) not in {state.cell_id for state in cell_states}:
        raise ValueError(f"target_cell_id {target_cell_id!r} is not present in cell_states")

    owner_by_barcode: dict[str, str] = {}
    owner_is_nuclear_by_barcode: dict[str, bool] = {}
    n_initial_owner_conflicts = 0
    barcode_candidate_counts: dict[str, int] = {}

    for state in cell_states:
        for barcode in state.context.candidate_bin_ids:
            barcode_candidate_counts[str(barcode)] = barcode_candidate_counts.get(str(barcode), 0) + 1

    for state in cell_states:
        seed_indices = np.flatnonzero(np.asarray(state.context.initial_membership_mask, dtype=np.uint8) > 0)
        for idx in seed_indices.tolist():
            barcode = str(state.context.candidate_bin_ids[int(idx)])
            current_owner = owner_by_barcode.get(barcode)
            if current_owner is None:
                owner_by_barcode[barcode] = state.cell_id
                owner_is_nuclear_by_barcode[barcode] = True
            elif current_owner != state.cell_id:
                state.membership_mask[int(idx)] = np.uint8(0)
                n_initial_owner_conflicts += 1

    steps: list[GreedyActionRecord] = []
    n_blocked_frontier_actions = 0
    stop_reason = "max_steps"

    for step_index in range(int(max_steps)):
        best_state: GreedyCellState | None = None
        best_index = -1
        best_score = -np.inf
        any_legal = False

        for state in cell_states:
            frontier = compute_frontier_eligible_mask(state.membership_mask, state.context.neighbor_index)
            legal = np.asarray(frontier, dtype=bool).copy()
            if not np.any(legal):
                continue

            frontier_indices = np.flatnonzero(frontier)
            for idx in frontier_indices.tolist():
                barcode = str(state.context.candidate_bin_ids[int(idx)])
                if barcode in owner_by_barcode:
                    legal[int(idx)] = False

            n_blocked_frontier_actions += int(np.sum(frontier) - np.sum(legal))
            if not np.any(legal):
                continue

            any_legal = True
            scores = np.asarray(score_fn(state, legal, step_index), dtype=np.float64)
            if scores.shape != (state.context.n_bins,):
                raise ValueError(
                    f"score_fn returned shape {scores.shape} for cell {state.cell_id!r}; "
                    f"expected {(state.context.n_bins,)}"
                )
            masked_scores = np.where(legal, scores, -np.inf)
            idx = int(np.argmax(masked_scores))
            score = float(masked_scores[idx])
            if score > best_score:
                best_state = state
                best_index = idx
                best_score = score

        if best_state is None or not any_legal:
            stop_reason = "no_legal_frontier"
            break
        if not np.isfinite(best_score) or best_score <= float(min_add_score):
            stop_reason = "no_positive_add_score"
            break

        barcode = str(best_state.context.candidate_bin_ids[best_index])
        best_state.membership_mask[best_index] = np.uint8(1)
        owner_by_barcode[barcode] = best_state.cell_id
        owner_is_nuclear_by_barcode[barcode] = False
        steps.append(
            GreedyActionRecord(
                step_index=step_index,
                cell_id=best_state.cell_id,
                barcode=barcode,
                candidate_index=best_index,
                add_score=best_score,
            )
        )

    n_contested_candidate_barcodes = int(sum(1 for count in barcode_candidate_counts.values() if count > 1))
    return GreedyPatchResult(
        target_cell_id=str(target_cell_id),
        steps=tuple(steps),
        owner_by_barcode=owner_by_barcode,
        owner_is_nuclear_by_barcode=owner_is_nuclear_by_barcode,
        n_patch_cells=int(len(cell_states)),
        n_initial_owner_conflicts=int(n_initial_owner_conflicts),
        n_contested_candidate_barcodes=n_contested_candidate_barcodes,
        n_blocked_frontier_actions=int(n_blocked_frontier_actions),
        stop_reason=stop_reason,
    )


def assignment_rows_for_target(
    *,
    result: GreedyPatchResult,
    target_state: GreedyCellState,
) -> list[dict[str, Any]]:
    """Convert one target cell's final mask into method assignment rows."""
    rows: list[dict[str, Any]] = []
    assigned = np.asarray(target_state.membership_mask, dtype=np.uint8) > 0
    nuclear = np.asarray(target_state.context.initial_membership_mask, dtype=np.uint8) > 0
    xy = np.asarray(target_state.context.candidate_bin_xy_um, dtype=np.float64)
    for idx in np.flatnonzero(assigned).tolist():
        barcode = str(target_state.context.candidate_bin_ids[int(idx)])
        row_col = parse_square_002um_barcode(barcode)
        array_row = row_col[0] if row_col is not None else int(round(float(xy[idx, 1]) / 2.0))
        array_col = row_col[1] if row_col is not None else int(round(float(xy[idx, 0]) / 2.0))
        rows.append(
            {
                "barcode": barcode,
                "cell_id": str(target_state.cell_id),
                "array_row": int(array_row),
                "array_col": int(array_col),
                "x_um": float(xy[idx, 0]),
                "y_um": float(xy[idx, 1]),
                "is_nuclear": bool(nuclear[idx]),
                "assignment_source": "multicell_greedy_competition",
                "patch_target_cell_id": str(result.target_cell_id),
            }
        )
    return rows


def parse_square_002um_barcode(barcode: str) -> tuple[int, int] | None:
    """Return ``(array_row, array_col)`` parsed from a 10x square-bin barcode."""
    match = _SQUARE_002UM_BARCODE_RE.match(str(barcode))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))
