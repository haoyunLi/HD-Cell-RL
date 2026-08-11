"""PPO/GRPO update helpers for patch rollouts."""

from __future__ import annotations

from typing import Any

import numpy as np

from .patch_types import PatchRolloutCache, PatchRolloutTransition, PatchTrajectory
from .ppo_buffers import compute_gae_returns_and_advantages
from .ppo_feature_schema import ACTION_FEATURE_DIM, GLOBAL_FEATURE_DIM
from .ppo_state import _zscore_1d


def build_patch_rollout_cache(
    *,
    trajectories: list[PatchTrajectory],
    gamma: float,
    gae_lambda: float,
    normalize_advantages: bool,
    training_mode: str,
    group_size: int,
    norm_epsilon: float,
) -> PatchRolloutCache:
    transitions: list[PatchRolloutTransition] = []
    next_joint_group_id = 0
    mode = str(training_mode).strip().lower()
    group_bonus = None
    if mode == "full_grpo":
        group_bonus = _patch_group_advantages(
            trajectories=trajectories,
            group_size=int(group_size),
            norm_epsilon=float(norm_epsilon),
        )

    for traj_idx, traj in enumerate(trajectories):
        joint_steps = [step for step in traj.steps if int(getattr(step, "joint_group_id", -1)) >= 0]
        if joint_steps:
            grouped: dict[int, list] = {}
            group_order: list[int] = []
            for step in traj.steps:
                group_id = int(getattr(step, "joint_group_id", -1))
                if group_id < 0:
                    raise ValueError("cannot mix joint and non-joint patch steps in one trajectory")
                if group_id not in grouped:
                    grouped[group_id] = []
                    group_order.append(group_id)
                grouped[group_id].append(step)
            macro_rewards = np.asarray([grouped[group_id][0].reward for group_id in group_order], dtype=np.float64)
            macro_values = np.asarray(
                [
                    grouped[group_id][0].joint_old_value
                    if grouped[group_id][0].joint_old_value is not None
                    else np.mean([step.old_value for step in grouped[group_id]])
                    for group_id in group_order
                ],
                dtype=np.float64,
            )
            macro_dones = np.asarray([any(step.done for step in grouped[group_id]) for group_id in group_order], dtype=bool)
            if mode == "full_grpo":
                macro_returns = np.zeros_like(macro_rewards, dtype=np.float64)
                macro_advantages = np.full_like(macro_rewards, float(group_bonus[traj_idx]), dtype=np.float64)
            else:
                macro_returns, macro_advantages = compute_gae_returns_and_advantages(
                    macro_rewards,
                    macro_values,
                    gamma=float(gamma),
                    gae_lambda=float(gae_lambda),
                    dones=macro_dones,
                )
            for macro_i, source_group_id in enumerate(group_order):
                global_group_id = int(next_joint_group_id)
                next_joint_group_id += 1
                source_steps = grouped[source_group_id]
                global_features, action_features, action_mask, actions = _pack_joint_steps(source_steps)
                old_log_prob = (
                    float(source_steps[0].joint_old_log_prob)
                    if source_steps[0].joint_old_log_prob is not None
                    else float(sum(float(step.old_log_prob) for step in source_steps))
                )
                old_value = (
                    float(source_steps[0].joint_old_value)
                    if source_steps[0].joint_old_value is not None
                    else float(np.mean([step.old_value for step in source_steps]))
                )
                transitions.append(
                    PatchRolloutTransition(
                        patch_slot=int(traj.patch_slot),
                        global_features=global_features,
                        action_features=action_features,
                        action_mask=action_mask,
                        action=actions,
                        reward=float(macro_rewards[macro_i]),
                        done=bool(macro_dones[macro_i]),
                        old_log_prob=old_log_prob,
                        old_value=old_value,
                        return_t=float(macro_returns[macro_i]),
                        advantage=float(macro_advantages[macro_i]),
                        joint_group_id=global_group_id,
                    )
                )
            continue

        rewards = np.asarray([step.reward for step in traj.steps], dtype=np.float64)
        values = np.asarray([step.old_value for step in traj.steps], dtype=np.float64)
        dones = np.asarray([step.done for step in traj.steps], dtype=bool)
        if mode == "full_grpo":
            returns = np.zeros_like(rewards, dtype=np.float64)
            advantages = np.full_like(rewards, float(group_bonus[traj_idx]), dtype=np.float64)
        else:
            returns, advantages = compute_gae_returns_and_advantages(
                rewards,
                values,
                gamma=float(gamma),
                gae_lambda=float(gae_lambda),
                dones=dones,
            )
        for i, step in enumerate(traj.steps):
            transitions.append(
                PatchRolloutTransition(
                    patch_slot=int(traj.patch_slot),
                    global_features=step.global_features,
                    action_features=step.action_features,
                    action_mask=step.action_mask,
                    action=int(step.action),
                    reward=float(step.reward),
                    done=bool(step.done),
                    old_log_prob=float(step.old_log_prob),
                    old_value=float(step.old_value),
                    return_t=float(returns[i]),
                    advantage=float(advantages[i]),
                    joint_group_id=-1,
                )
            )

    if normalize_advantages and transitions:
        if any(int(t.joint_group_id) >= 0 for t in transitions):
            group_ids: list[int] = []
            group_advantages: list[float] = []
            seen: set[int] = set()
            for t in transitions:
                group_id = int(t.joint_group_id)
                key = group_id if group_id >= 0 else -len(seen) - 1
                if key in seen:
                    continue
                seen.add(key)
                group_ids.append(key)
                group_advantages.append(float(t.advantage))
            adv_by_group = {
                group_id: float(value)
                for group_id, value in zip(group_ids, _zscore_1d(np.asarray(group_advantages, dtype=np.float64)))
            }
            adv = np.asarray(
                [
                    adv_by_group[int(t.joint_group_id)]
                    if int(t.joint_group_id) >= 0
                    else float(t.advantage)
                    for t in transitions
                ],
                dtype=np.float64,
            )
        else:
            adv = _zscore_1d(np.asarray([t.advantage for t in transitions], dtype=np.float64))
        transitions = [
            PatchRolloutTransition(
                patch_slot=t.patch_slot,
                global_features=t.global_features,
                action_features=t.action_features,
                action_mask=t.action_mask,
                action=t.action,
                reward=t.reward,
                done=t.done,
                old_log_prob=t.old_log_prob,
                old_value=t.old_value,
                return_t=t.return_t,
                advantage=float(adv[i]),
                joint_group_id=t.joint_group_id,
            )
            for i, t in enumerate(transitions)
        ]
    return PatchRolloutCache(transitions=tuple(transitions))


def patch_ppo_update(
    *,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    cache: PatchRolloutCache,
    eps_clip: float,
    ppo_epochs: int,
    minibatch_size: int,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    target_kl: float | None,
    include_value_loss: bool,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    import torch

    if cache.n_transitions == 0:
        raise ValueError("patch rollout cache is empty")
    if any(int(t.joint_group_id) >= 0 for t in cache.transitions):
        return _patch_joint_ppo_update(
            model=model,
            optimizer=optimizer,
            cache=cache,
            eps_clip=eps_clip,
            ppo_epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            include_value_loss=include_value_loss,
            device=device,
            rng=rng,
        )
    indices = np.arange(cache.n_transitions, dtype=np.int64)
    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    kls: list[float] = []
    stop_early = False
    for _ in range(int(ppo_epochs)):
        if stop_early:
            break
        perm = rng.permutation(indices)
        for start in range(0, len(perm), int(minibatch_size)):
            mb = perm[start : start + int(minibatch_size)]
            batch = _collate_patch_minibatch(cache, mb, device=device)
            dist, values = model(batch["global_features"], batch["action_features"], batch["action_mask"])
            new_log_prob = dist.log_prob(batch["actions"])
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_log_prob - batch["old_log_probs"])
            surr1 = ratio * batch["advantages"]
            surr2 = torch.clamp(ratio, 1.0 - float(eps_clip), 1.0 + float(eps_clip)) * batch["advantages"]
            policy_loss = -torch.min(surr1, surr2).mean()
            if include_value_loss:
                value_loss = torch.mean((values - batch["returns"]) ** 2)
                loss = policy_loss + float(vf_coef) * value_loss - float(ent_coef) * entropy
            else:
                value_loss = values.new_tensor(0.0)
                loss = policy_loss - float(ent_coef) * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
            optimizer.step()

            with torch.no_grad():
                approx_kl = torch.mean(batch["old_log_probs"] - new_log_prob)
            losses.append(float(loss.item()))
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))
            kls.append(float(approx_kl.item()))
            if target_kl is not None and float(approx_kl.item()) > 1.5 * float(target_kl):
                stop_early = True
                break

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "total_loss": float(np.mean(losses)),
        "approx_kl": float(np.mean(kls)),
    }


def _patch_joint_ppo_update(
    *,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    cache: PatchRolloutCache,
    eps_clip: float,
    ppo_epochs: int,
    minibatch_size: int,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    target_kl: float | None,
    include_value_loss: bool,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    import torch

    group_ids = np.asarray(
        sorted({int(t.joint_group_id) for t in cache.transitions if int(t.joint_group_id) >= 0}),
        dtype=np.int64,
    )
    if int(group_ids.shape[0]) == 0:
        raise ValueError("joint patch rollout cache has no joint groups")

    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    kls: list[float] = []
    stop_early = False
    for _ in range(int(ppo_epochs)):
        if stop_early:
            break
        perm = rng.permutation(group_ids)
        for start in range(0, len(perm), int(minibatch_size)):
            mb_groups = perm[start : start + int(minibatch_size)]
            batch = _collate_patch_joint_minibatch(cache, mb_groups, device=device)
            dist, values = model(batch["global_features"], batch["action_features"], batch["action_mask"])
            local_log_prob = dist.log_prob(batch["actions"])
            group_count = int(batch["old_log_probs"].shape[0])
            group_index = batch["group_index"]
            new_log_prob = torch.zeros((group_count,), device=device, dtype=torch.float32)
            new_log_prob.index_add_(0, group_index, local_log_prob.to(dtype=torch.float32))
            value_sum = torch.zeros((group_count,), device=device, dtype=torch.float32)
            value_sum.index_add_(0, group_index, values.to(dtype=torch.float32))
            value_counts = torch.bincount(group_index, minlength=group_count).to(device=device, dtype=torch.float32)
            group_values = value_sum / torch.clamp(value_counts, min=1.0)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_prob - batch["old_log_probs"])
            surr1 = ratio * batch["advantages"]
            surr2 = torch.clamp(ratio, 1.0 - float(eps_clip), 1.0 + float(eps_clip)) * batch["advantages"]
            policy_loss = -torch.min(surr1, surr2).mean()
            if include_value_loss:
                value_loss = torch.mean((group_values - batch["returns"]) ** 2)
                loss = policy_loss + float(vf_coef) * value_loss - float(ent_coef) * entropy
            else:
                value_loss = group_values.new_tensor(0.0)
                loss = policy_loss - float(ent_coef) * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
            optimizer.step()

            with torch.no_grad():
                approx_kl = torch.mean(batch["old_log_probs"] - new_log_prob)
            losses.append(float(loss.item()))
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))
            kls.append(float(approx_kl.item()))
            if target_kl is not None and float(approx_kl.item()) > 1.5 * float(target_kl):
                stop_early = True
                break

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "total_loss": float(np.mean(losses)),
        "approx_kl": float(np.mean(kls)),
    }


def _pack_joint_steps(steps: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(steps) == 1 and np.asarray(steps[0].global_features).ndim == 2:
        return (
            np.asarray(steps[0].global_features, dtype=np.float32).copy(),
            np.asarray(steps[0].action_features, dtype=np.float32).copy(),
            np.asarray(steps[0].action_mask, dtype=bool).copy(),
            np.asarray(steps[0].action, dtype=np.int64).copy(),
        )
    n = int(len(steps))
    max_actions = max(int(np.asarray(step.action_features).shape[0]) for step in steps)
    global_batch = np.zeros((n, GLOBAL_FEATURE_DIM), dtype=np.float32)
    action_batch = np.zeros((n, max_actions, ACTION_FEATURE_DIM), dtype=np.float32)
    mask_batch = np.zeros((n, max_actions), dtype=bool)
    actions = np.zeros((n,), dtype=np.int64)
    for row, step in enumerate(steps):
        action_features = np.asarray(step.action_features, dtype=np.float32)
        action_mask = np.asarray(step.action_mask, dtype=bool)
        n_actions = int(action_features.shape[0])
        global_batch[row] = np.asarray(step.global_features, dtype=np.float32)
        action_batch[row, :n_actions, :] = action_features
        mask_batch[row, :n_actions] = action_mask
        actions[row] = int(step.action)
    return global_batch, action_batch, mask_batch, actions


def _joint_transition_arrays(t: PatchRolloutTransition) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global_features = np.asarray(t.global_features, dtype=np.float32)
    action_features = np.asarray(t.action_features, dtype=np.float32)
    action_mask = np.asarray(t.action_mask, dtype=bool)
    actions = np.asarray(t.action, dtype=np.int64)
    if global_features.ndim == 1:
        global_features = global_features.reshape(1, -1)
    if action_features.ndim == 2:
        action_features = action_features.reshape(1, action_features.shape[0], action_features.shape[1])
    if action_mask.ndim == 1:
        action_mask = action_mask.reshape(1, -1)
    if actions.ndim == 0:
        actions = actions.reshape(1)
    if int(global_features.shape[0]) != int(action_features.shape[0]) or int(actions.shape[0]) != int(global_features.shape[0]):
        raise ValueError("joint transition local action dimensions do not match")
    return global_features, action_features, action_mask, actions


def _patch_group_advantages(
    *,
    trajectories: list[PatchTrajectory],
    group_size: int,
    norm_epsilon: float,
) -> np.ndarray:
    if group_size <= 1:
        return np.asarray([traj.patch_score for traj in trajectories], dtype=np.float64)
    if len(trajectories) % int(group_size) != 0:
        raise ValueError("patch trajectory count must be divisible by group_size")
    out = np.zeros((len(trajectories),), dtype=np.float64)
    for start in range(0, len(trajectories), int(group_size)):
        stop = start + int(group_size)
        scores = np.asarray([traj.patch_score for traj in trajectories[start:stop]], dtype=np.float64)
        out[start:stop] = (scores - float(np.mean(scores))) / (float(np.std(scores, ddof=0)) + float(norm_epsilon))
    return out


def _collate_patch_minibatch(
    cache: PatchRolloutCache,
    indices: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    import torch

    idx = np.asarray(indices, dtype=np.int64)
    transitions = [cache.transitions[int(i)] for i in idx.tolist()]
    max_actions = max(int(t.action_features.shape[0]) for t in transitions)
    n = len(transitions)
    global_batch = np.zeros((n, GLOBAL_FEATURE_DIM), dtype=np.float32)
    action_batch = np.zeros((n, max_actions, ACTION_FEATURE_DIM), dtype=np.float32)
    mask_batch = np.zeros((n, max_actions), dtype=bool)
    actions = np.zeros((n,), dtype=np.int64)
    old_log_probs = np.zeros((n,), dtype=np.float32)
    returns = np.zeros((n,), dtype=np.float32)
    advantages = np.zeros((n,), dtype=np.float32)
    for row, t in enumerate(transitions):
        n_actions = int(t.action_features.shape[0])
        global_batch[row] = np.asarray(t.global_features, dtype=np.float32)
        action_batch[row, :n_actions, :] = np.asarray(t.action_features, dtype=np.float32)
        mask_batch[row, :n_actions] = np.asarray(t.action_mask, dtype=bool)
        actions[row] = int(t.action)
        old_log_probs[row] = np.float32(t.old_log_prob)
        returns[row] = np.float32(t.return_t)
        advantages[row] = np.float32(t.advantage)
    return {
        "global_features": torch.as_tensor(global_batch, device=device, dtype=torch.float32),
        "action_features": torch.as_tensor(action_batch, device=device, dtype=torch.float32),
        "action_mask": torch.as_tensor(mask_batch, device=device, dtype=torch.bool),
        "actions": torch.as_tensor(actions, device=device, dtype=torch.int64),
        "old_log_probs": torch.as_tensor(old_log_probs, device=device, dtype=torch.float32),
        "returns": torch.as_tensor(returns, device=device, dtype=torch.float32),
        "advantages": torch.as_tensor(advantages, device=device, dtype=torch.float32),
    }


def _collate_patch_joint_minibatch(
    cache: PatchRolloutCache,
    group_ids: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    import torch

    selected_groups = [int(group_id) for group_id in np.asarray(group_ids, dtype=np.int64).tolist()]
    group_to_row = {group_id: row for row, group_id in enumerate(selected_groups)}
    transitions = [
        t
        for t in cache.transitions
        if int(t.joint_group_id) in group_to_row
    ]
    if not transitions:
        raise ValueError("joint patch minibatch is empty")
    local_arrays = [_joint_transition_arrays(t) for t in transitions]
    max_actions = max(int(action_features.shape[1]) for _global, action_features, _mask, _actions in local_arrays)
    n = int(sum(int(global_features.shape[0]) for global_features, _action_features, _mask, _actions in local_arrays))
    n_groups = len(selected_groups)
    global_batch = np.zeros((n, GLOBAL_FEATURE_DIM), dtype=np.float32)
    action_batch = np.zeros((n, max_actions, ACTION_FEATURE_DIM), dtype=np.float32)
    mask_batch = np.zeros((n, max_actions), dtype=bool)
    actions = np.zeros((n,), dtype=np.int64)
    group_index = np.zeros((n,), dtype=np.int64)
    old_group_log_probs = np.zeros((n_groups,), dtype=np.float32)
    returns = np.zeros((n_groups,), dtype=np.float32)
    advantages = np.zeros((n_groups,), dtype=np.float32)
    group_seen: set[int] = set()
    row = 0
    for t, (global_features, action_features, action_mask, local_actions) in zip(transitions, local_arrays):
        group_id = int(t.joint_group_id)
        group_row = int(group_to_row[group_id])
        old_group_log_probs[group_row] = np.float32(t.old_log_prob)
        if group_id not in group_seen:
            group_seen.add(group_id)
            returns[group_row] = np.float32(t.return_t)
            advantages[group_row] = np.float32(t.advantage)
        for local_i in range(int(global_features.shape[0])):
            n_actions = int(action_features[local_i].shape[0])
            global_batch[row] = global_features[local_i]
            action_batch[row, :n_actions, :] = action_features[local_i]
            mask_batch[row, :n_actions] = action_mask[local_i]
            actions[row] = int(local_actions[local_i])
            group_index[row] = group_row
            row += 1

    return {
        "global_features": torch.as_tensor(global_batch, device=device, dtype=torch.float32),
        "action_features": torch.as_tensor(action_batch, device=device, dtype=torch.float32),
        "action_mask": torch.as_tensor(mask_batch, device=device, dtype=torch.bool),
        "actions": torch.as_tensor(actions, device=device, dtype=torch.int64),
        "group_index": torch.as_tensor(group_index, device=device, dtype=torch.int64),
        "old_log_probs": torch.as_tensor(old_group_log_probs, device=device, dtype=torch.float32),
        "returns": torch.as_tensor(returns, device=device, dtype=torch.float32),
        "advantages": torch.as_tensor(advantages, device=device, dtype=torch.float32),
    }
