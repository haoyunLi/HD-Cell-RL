"""Exact target-cell shape-add corrections for diagnostic greedy rollouts."""

from __future__ import annotations

from typing import Any


def exact_shape_add_corrections(env: Any) -> Any:
    """Return exact-minus-cached normalized shape-add deltas for all actions.

    The patch environment already calculates the old-owner removal by a full
    shape-state rebuild. Its target-owner addition uses a faster hull update
    that can differ from the objective evaluated after ``step``. This helper
    rebuilds only the target shape state for each exposed action and returns
    the correction needed to make greedy ranking match the exact objective.
    Environment ownership and membership are restored before returning.
    """

    torch = env._torch
    corrections = torch.zeros_like(
        env._cached_action_rewards,
        device=env._device,
        dtype=torch.float32,
    )
    if int(corrections.numel()) == 0:
        return corrections

    denominator = float(env._score_denominator())
    for raw_cell in (
        torch.unique(env._cached_action_cells)
        .detach()
        .cpu()
        .numpy()
        .astype("int64")
        .tolist()
    ):
        cell_idx = int(raw_cell)
        ctx = env._ctx.cells[cell_idx]
        original_state = env._shape_states[cell_idx]
        if original_state is None or float(ctx.shape_prior_weight) <= 0.0:
            continue
        positions = torch.nonzero(
            env._cached_action_cells == cell_idx,
            as_tuple=False,
        ).flatten()
        bins = env._cached_action_bins[positions].to(dtype=torch.long)
        cached_raw = env._compute_shape_raw_delta_for_indices(
            cell_idx=cell_idx,
            candidate_idx=bins,
        ).to(dtype=torch.float32)
        exact_raw = torch.zeros_like(cached_raw, dtype=torch.float32)
        original_mask = env._membership_masks[cell_idx]
        original_counts = env._shape_neighbor_counts[cell_idx]
        try:
            for local_position, raw_bin in enumerate(
                bins.detach().cpu().numpy().astype("int64").tolist()
            ):
                added_mask = original_mask.clone()
                added_mask[int(raw_bin)] = True
                env._membership_masks[cell_idx] = added_mask
                env._shape_neighbor_counts[cell_idx] = env._build_shape_neighbor_counts(
                    cell_idx=cell_idx
                )
                exact_state = env._build_shape_state(cell_idx=cell_idx)
                if exact_state is not None:
                    exact_raw[local_position] = (
                        exact_state.current_reward - original_state.current_reward
                    ).to(dtype=torch.float32)
        finally:
            env._membership_masks[cell_idx] = original_mask
            env._shape_neighbor_counts[cell_idx] = original_counts
        corrections[positions] = (
            float(ctx.shape_prior_weight)
            * (exact_raw - cached_raw)
            / denominator
        ).to(dtype=torch.float32)
    return corrections
