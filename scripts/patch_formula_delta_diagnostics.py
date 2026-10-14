"""Component-level checks for patch REPLACE delta diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def cached_replace_component_deltas(
    env: Any,
    cached_position: int,
) -> dict[str, float]:
    """Reconstruct the cached REPLACE score from its individual terms.

    This helper is evaluation-only. It reads the pre-action environment state
    and does not change ownership or membership.
    """

    torch = env._torch
    position = int(cached_position)
    target_cell = int(env._cached_action_cells[position].detach().cpu().item())
    target_bin = int(env._cached_action_bins[position].detach().cpu().item())
    old_cell = int(env._cached_action_old_cells[position].detach().cpu().item())
    old_bin = int(env._cached_action_old_bins[position].detach().cpu().item())
    if old_cell < 0 or old_bin < 0:
        raise ValueError("component diagnostic requires a REPLACE action")

    target_ctx = env._ctx.cells[target_cell]
    old_ctx = env._ctx.cells[old_cell]
    raw = {
        "expression_confidence": 0.0,
        "expression_compatibility": 0.0,
        "distance_penalty": 0.0,
        "overlap_penalty": 0.0,
        "neighbor_support": 0.0,
        "shape_prior": 0.0,
    }

    target_posterior = torch.softmax(env._score_sums[target_cell], dim=0)
    target_next_posterior = torch.softmax(
        env._score_sums[target_cell] + env._ll64[target_cell][target_bin],
        dim=0,
    )
    target_mask = env._membership_masks[target_cell]
    target_confidence_sum = torch.sum(env._expression_conf64[target_cell][target_mask])
    target_bin_confidence = env._expression_conf64[target_cell][target_bin]
    raw["expression_confidence"] += float(
        (
            float(target_ctx.w1)
            * (
                torch.max(target_next_posterior)
                * (target_confidence_sum + target_bin_confidence)
                - torch.max(target_posterior) * target_confidence_sum
            )
        )
        .detach()
        .cpu()
        .item()
    )
    if float(target_ctx.w5) != 0.0:
        weighted_sum = torch.sum(
            env._ll64[target_cell][target_mask]
            * env._expression_conf64[target_cell][target_mask].reshape(-1, 1),
            dim=0,
        )
        current_value = torch.sum(target_posterior * weighted_sum)
        next_value = torch.sum(target_next_posterior * weighted_sum)
        next_value = next_value + (
            torch.sum(env._ll64[target_cell][target_bin] * target_next_posterior)
            * target_bin_confidence
        )
        raw["expression_compatibility"] += float(
            (float(target_ctx.w5) * (next_value - current_value))
            .detach()
            .cpu()
            .item()
        )

    old_posterior = torch.softmax(env._score_sums[old_cell], dim=0)
    old_prev_posterior = torch.softmax(
        env._score_sums[old_cell] - env._ll64[old_cell][old_bin],
        dim=0,
    )
    old_mask = env._membership_masks[old_cell]
    old_confidence_sum = torch.sum(env._expression_conf64[old_cell][old_mask])
    old_bin_confidence = env._expression_conf64[old_cell][old_bin]
    raw["expression_confidence"] += float(
        (
            float(old_ctx.w1)
            * (
                torch.max(old_prev_posterior)
                * torch.clamp(old_confidence_sum - old_bin_confidence, min=0.0)
                - torch.max(old_posterior) * old_confidence_sum
            )
        )
        .detach()
        .cpu()
        .item()
    )
    if float(old_ctx.w5) != 0.0:
        weighted_sum = torch.sum(
            env._ll64[old_cell][old_mask]
            * env._expression_conf64[old_cell][old_mask].reshape(-1, 1),
            dim=0,
        )
        current_value = torch.sum(old_posterior * weighted_sum)
        removed_weight = (
            env._ll64[old_cell][old_bin]
            * old_bin_confidence.reshape(1)
        )
        previous_value = torch.sum(
            old_prev_posterior * (weighted_sum - removed_weight)
        )
        raw["expression_compatibility"] += float(
            (float(old_ctx.w5) * (previous_value - current_value))
            .detach()
            .cpu()
            .item()
        )

    raw["distance_penalty"] = float(
        -float(target_ctx.w2) * float(np.asarray(target_ctx.p_dis)[target_bin])
        + float(old_ctx.w2) * float(np.asarray(old_ctx.p_dis)[old_bin])
    )
    raw["overlap_penalty"] = float(
        -float(target_ctx.w3) * float(np.asarray(target_ctx.p_overlap)[target_bin])
        + float(old_ctx.w3) * float(np.asarray(old_ctx.p_overlap)[old_bin])
    )
    target_support = env._compute_neighbor_support(
        cell_idx=target_cell,
        mask=target_mask,
    )[target_bin]
    old_support = env._compute_neighbor_support(
        cell_idx=old_cell,
        mask=old_mask,
    )[old_bin]
    raw["neighbor_support"] = float(
        (
            2.0 * float(target_ctx.w4) * target_support
            - 2.0 * float(old_ctx.w4) * old_support
        )
        .detach()
        .cpu()
        .item()
    )

    target_shape_delta = env._compute_shape_raw_delta_for_indices(
        cell_idx=target_cell,
        candidate_idx=torch.as_tensor(
            [target_bin], device=env._device, dtype=torch.long
        ),
    )[0]
    old_shape_delta = env._compute_shape_raw_delta_after_remove_indices(
        cell_idx=old_cell,
        candidate_idx=torch.as_tensor(
            [old_bin], device=env._device, dtype=torch.long
        ),
    )[0]
    raw["shape_prior"] = float(
        (
            float(target_ctx.shape_prior_weight) * target_shape_delta
            + float(old_ctx.shape_prior_weight) * old_shape_delta
        )
        .detach()
        .cpu()
        .item()
    )

    denominator = float(env._score_denominator())
    result = {key: float(value / denominator) for key, value in raw.items()}
    original_mask = env._membership_masks[target_cell]
    original_counts = env._shape_neighbor_counts[target_cell]
    original_state = env._shape_states[target_cell]
    try:
        added_mask = original_mask.clone()
        added_mask[target_bin] = True
        env._membership_masks[target_cell] = added_mask
        env._shape_neighbor_counts[target_cell] = env._build_shape_neighbor_counts(
            cell_idx=target_cell
        )
        exact_added_state = env._build_shape_state(cell_idx=target_cell)
    finally:
        env._membership_masks[target_cell] = original_mask
        env._shape_neighbor_counts[target_cell] = original_counts
    exact_target_shape_delta = (
        0.0
        if original_state is None or exact_added_state is None
        else float(
            (exact_added_state.current_reward - original_state.current_reward)
            .detach()
            .cpu()
            .item()
        )
    )
    result["_shape_target_cached"] = float(
        float(target_ctx.shape_prior_weight)
        * float(target_shape_delta.detach().cpu().item())
        / denominator
    )
    result["_shape_target_exact"] = float(
        float(target_ctx.shape_prior_weight)
        * exact_target_shape_delta
        / denominator
    )
    result["_shape_old_exact"] = float(
        float(old_ctx.shape_prior_weight)
        * float(old_shape_delta.detach().cpu().item())
        / denominator
    )
    return result
