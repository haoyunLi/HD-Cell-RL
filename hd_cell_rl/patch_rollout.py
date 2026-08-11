"""Patch rollout collection helpers."""

from __future__ import annotations

from typing import Any
import time

import numpy as np

from .patch_dataset import PatchDataset
from .patch_env_cpu import CachedMultiCellPatchEnv, MultiCellPatchEnv
from .patch_env_torch import TorchJointPatchEnv, TorchPatchEnv, TorchSingleAgentPatchEnv
from .patch_types import PatchActionEvent, PatchContext, PatchStep, PatchTrajectory
from .ppo_feature_schema import ACTION_FEATURE_DIM


def rollout_patch_episode(
    *,
    patch_slot: int,
    context: PatchContext,
    model: ActorCritic,
    device: torch.device,
    rng: np.random.Generator,
    policy_mode: str = "sample",
) -> PatchTrajectory:
    import torch

    env = MultiCellPatchEnv(context)
    obs, info = env.reset()
    steps: list[PatchStep] = []
    total_reward = 0.0
    mode = str(policy_mode).strip().lower()
    while True:
        global_t = torch.as_tensor(obs["global_features"], device=device, dtype=torch.float32).unsqueeze(0)
        action_t = torch.as_tensor(obs["action_features"], device=device, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(obs["action_mask"], device=device, dtype=torch.bool).unsqueeze(0)
        with torch.inference_mode():
            dist, value = model(global_t, action_t, mask_t)
            if mode == "greedy":
                action_tensor = torch.argmax(dist.probs, dim=1)
            else:
                action_tensor = dist.sample()
            old_log_prob = float(dist.log_prob(action_tensor).item())
            action = int(action_tensor.item())
            old_value = float(value.item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        total_reward += float(reward)
        steps.append(
            PatchStep(
                global_features=np.asarray(obs["global_features"], dtype=np.float32).copy(),
                action_features=np.asarray(obs["action_features"], dtype=np.float32).copy(),
                action_mask=np.asarray(obs["action_mask"], dtype=bool).copy(),
                action=action,
                reward=float(reward),
                done=done,
                old_log_prob=old_log_prob,
                old_value=old_value,
            )
        )
        obs = next_obs
        if done:
            break

    reward_per_step = float(total_reward / max(1, len(steps)))
    metrics = {
        "patch_score": float(info.get("patch_score", 0.0)),
        "raw_total_reward": float(total_reward),
        "reward_per_step": reward_per_step,
        "n_core_cells": float(info.get("n_core_cells", 0)),
        "n_margin_cells": float(info.get("n_margin_cells", 0)),
        "n_patch_cells": float(info.get("n_patch_cells", 0)),
        "n_force_fill_expression_bins": float(info.get("n_force_fill_expression_bins", 0)),
        "n_force_fill_owned_expression_bins": float(info.get("n_force_fill_owned_expression_bins", 0)),
        "n_patch_steps": float(len(steps)),
    }
    return PatchTrajectory(
        patch_slot=int(patch_slot),
        patch_id=str(context.patch_id),
        steps=tuple(steps),
        total_reward=float(total_reward),
        patch_score=float(info.get("patch_score", 0.0)),
        metrics=metrics,
        final_masks=env.final_masks(),
    )


def collect_patch_contexts(
    *,
    dataset: PatchDataset,
    batch_patches: int,
    group_size: int = 1,
    max_attempts_multiplier: int = 20,
) -> list[PatchContext]:
    base_count = int(batch_patches)
    if group_size > 1:
        base_count = max(1, int(batch_patches) // int(group_size))
    contexts: list[PatchContext] = []
    attempts = 0
    max_attempts = max(base_count * int(max_attempts_multiplier), base_count)
    while len(contexts) < base_count and attempts < max_attempts:
        rows = dataset.sample_rows(base_count - len(contexts))
        for row in rows.itertuples(index=False):
            attempts += 1
            ctx = dataset.load_patch_context(row)
            if ctx is None:
                continue
            contexts.append(ctx)
            if len(contexts) >= base_count:
                break
    if group_size > 1:
        expanded: list[PatchContext] = []
        for ctx in contexts:
            expanded.extend([ctx] * int(group_size))
        return expanded
    return contexts


def collect_patch_trajectories(
    *,
    contexts: list[PatchContext],
    model: ActorCritic,
    device: torch.device,
    rng: np.random.Generator,
    policy_mode: str = "sample",
) -> list[PatchTrajectory]:
    trajectories: list[PatchTrajectory] = []
    seeds = rng.integers(low=0, high=np.iinfo(np.uint32).max, size=len(contexts), dtype=np.uint64)
    for slot, ctx in enumerate(contexts):
        local_rng = np.random.default_rng(int(seeds[slot]))
        trajectories.append(
            rollout_patch_episode(
                patch_slot=slot,
                context=ctx,
                model=model,
                device=device,
                rng=local_rng,
                policy_mode=policy_mode,
            )
        )
    return trajectories


def _build_patch_env(*, context: PatchContext, device: Any, rollout_backend: str) -> Any:
    backend = str(rollout_backend).strip().lower()
    agent_mode = str(getattr(context, "agent_mode", "multi_cell")).strip().lower()
    if agent_mode == "multi_cell_joint_global_delta":
        if backend != "torch_gpu":
            raise ValueError(f"{agent_mode} patch agent mode requires rollout_backend='torch_gpu'")
        return TorchJointPatchEnv(context, device=device)
    if agent_mode in {"single_cell_global_delta", "multi_cell_global_delta"}:
        if backend != "torch_gpu":
            raise ValueError(f"{agent_mode} patch agent mode requires rollout_backend='torch_gpu'")
        return TorchSingleAgentPatchEnv(context, device=device)
    if backend == "legacy_cpu":
        return MultiCellPatchEnv(context)
    if backend == "cached_cpu":
        return CachedMultiCellPatchEnv(context)
    if backend == "torch_gpu":
        return TorchPatchEnv(context, device=device)
    raise ValueError("patch_training.rollout_backend must be one of: legacy_cpu, cached_cpu, torch_gpu")


def _obs_uses_torch(obs: dict[str, Any]) -> bool:
    value = obs.get("global_features")
    return hasattr(value, "detach") and hasattr(value, "to")


def _obs_to_numpy(value: Any, *, dtype: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy().astype(dtype, copy=True)
    return np.asarray(value, dtype=dtype).copy()


def _collect_patch_trajectories_joint(
    *,
    contexts: list[PatchContext],
    model: ActorCritic,
    device: torch.device,
    rng: np.random.Generator,
    policy_mode: str = "sample",
    rollout_backend: str = "legacy_cpu",
    capture_trace: bool = False,
) -> tuple[list[PatchTrajectory], dict[str, float]]:
    """Collect macro-step trajectories for multi-cell joint global-delta mode."""
    import torch

    if not contexts:
        return [], {
            "rollout_total_sec": 0.0,
            "rollout_tensor_sec": 0.0,
            "rollout_model_sec": 0.0,
            "rollout_env_sec": 0.0,
            "rollout_n_model_calls": 0.0,
        }
    if str(rollout_backend).strip().lower() != "torch_gpu":
        raise ValueError("multi_cell_joint_global_delta requires rollout_backend='torch_gpu'")

    seed = int(rng.integers(low=0, high=np.iinfo(np.uint32).max, dtype=np.uint64))
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)

    mode = str(policy_mode).strip().lower()
    envs = [TorchJointPatchEnv(ctx, device=device) for ctx in contexts]
    infos = []
    initial_masks: list[dict[str, np.ndarray] | None] = []
    for env in envs:
        _obs, info = env.reset()
        infos.append(info)
        initial_masks.append(env.final_masks() if capture_trace else None)

    trajectories: list[PatchTrajectory] = []
    tensor_sec = 0.0
    model_sec = 0.0
    env_sec = 0.0
    model_calls = 0
    t_total0 = time.perf_counter()

    for env_idx, env in enumerate(envs):
        steps: list[PatchStep] = []
        total_reward = 0.0
        macro_step_id = 0
        local_action_transitions = 0
        info = infos[env_idx]
        while True:
            local_observations = env.joint_observations()
            selected_actions: list[dict[str, int | bool]] = []
            local_records: list[tuple[dict[str, Any], Any, int, float, float]] = []
            proposed_barcodes: set[int] = set()

            if local_observations:
                t0 = time.perf_counter()
                max_actions = max(int(obs["action_features"].shape[0]) for obs in local_observations)
                global_t = torch.stack(
                    [obs["global_features"].to(device=device, dtype=torch.float32) for obs in local_observations],
                    dim=0,
                )
                action_t = torch.zeros(
                    (len(local_observations), max_actions, ACTION_FEATURE_DIM),
                    device=device,
                    dtype=torch.float32,
                )
                mask_t = torch.zeros((len(local_observations), max_actions), device=device, dtype=torch.bool)
                action_lengths: list[int] = []
                for row, obs in enumerate(local_observations):
                    action_features = obs["action_features"].to(device=device, dtype=torch.float32)
                    action_mask = obs["action_mask"].to(device=device, dtype=torch.bool)
                    n_actions = int(action_features.shape[0])
                    action_lengths.append(n_actions)
                    action_t[row, :n_actions] = action_features
                    mask_t[row, :n_actions] = action_mask
                tensor_sec += float(time.perf_counter() - t0)

                t0 = time.perf_counter()
                with torch.inference_mode():
                    dist, value = model(global_t, action_t, mask_t)
                    if mode == "greedy":
                        action_tensor = torch.argmax(dist.probs, dim=1)
                    else:
                        action_tensor = dist.sample()
                    old_log_prob_tensor = dist.log_prob(action_tensor)
                    batch_logits = dist.logits.detach()
                    batch_actions = action_tensor.detach()
                    batch_log_probs = old_log_prob_tensor.detach()
                    batch_values = value.detach()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                model_sec += float(time.perf_counter() - t0)
                model_calls += 1

                for row, obs in enumerate(local_observations):
                    n_actions = int(action_lengths[row])
                    action_mask = mask_t[row, :n_actions].clone()
                    forced_noop = False
                    if proposed_barcodes:
                        barcodes = obs["joint_action_barcodes"][:n_actions].to(device=device)
                        conflict = torch.zeros_like(action_mask, dtype=torch.bool)
                        for barcode_index in proposed_barcodes:
                            conflict = conflict | (barcodes == int(barcode_index))
                        conflict[0] = False
                        action_mask = action_mask & (~conflict)
                        if not bool(torch.any(action_mask).detach().cpu().item()):
                            action_mask[0] = True
                            forced_noop = True

                    if forced_noop:
                        action = 0
                        old_log_prob = 0.0
                    elif proposed_barcodes:
                        adjusted_logits = batch_logits[row, :n_actions].clone()
                        adjusted_logits[~action_mask] = torch.finfo(adjusted_logits.dtype).min
                        adjusted_dist = torch.distributions.Categorical(logits=adjusted_logits)
                        if mode == "greedy":
                            selected_action_tensor = torch.argmax(adjusted_dist.probs, dim=0)
                        else:
                            selected_action_tensor = adjusted_dist.sample()
                        old_log_prob = float(adjusted_dist.log_prob(selected_action_tensor).detach().cpu().item())
                        action = int(selected_action_tensor.detach().cpu().item())
                    else:
                        action = int(batch_actions[row].detach().cpu().item())
                        old_log_prob = float(batch_log_probs[row].detach().cpu().item())
                    old_value = float(batch_values[row].detach().cpu().item())

                    selected = _joint_action_from_observation(obs, action)
                    if int(selected.get("barcode_index", -1)) >= 0:
                        proposed_barcodes.add(int(selected["barcode_index"]))
                    selected_actions.append(selected)
                    local_records.append((obs, action_mask, action, old_log_prob, old_value))

            t0 = time.perf_counter()
            _next_obs, reward, terminated, truncated, info = env.step_joint(selected_actions)
            env_sec += float(time.perf_counter() - t0)
            done = bool(terminated or truncated)
            total_reward += float(reward)

            if local_records:
                action_events = (
                    _joint_action_events(
                        context=contexts[env_idx],
                        selected_actions=selected_actions,
                        applied=bool(info.get("last_step_applied", False)),
                    )
                    if capture_trace
                    else ()
                )
                joint_old_log_prob = float(sum(record[3] for record in local_records))
                joint_old_value = float(np.mean([record[4] for record in local_records]))
                local_action_transitions += int(len(local_records))
                max_record_actions = max(int(record[0]["action_features"].shape[0]) for record in local_records)
                n_records = int(len(local_records))
                global_batch = np.zeros((n_records, local_records[0][0]["global_features"].shape[0]), dtype=np.float32)
                action_batch = np.zeros((n_records, max_record_actions, ACTION_FEATURE_DIM), dtype=np.float32)
                mask_batch = np.zeros((n_records, max_record_actions), dtype=bool)
                actions = np.zeros((n_records,), dtype=np.int64)
                for row, (obs, action_mask, action, _old_log_prob, _old_value) in enumerate(local_records):
                    action_features = _obs_to_numpy(obs["action_features"], dtype=np.float32)
                    local_mask = _obs_to_numpy(action_mask, dtype=bool)
                    n_actions = int(action_features.shape[0])
                    global_batch[row] = _obs_to_numpy(obs["global_features"], dtype=np.float32)
                    action_batch[row, :n_actions, :] = action_features
                    mask_batch[row, :n_actions] = local_mask
                    actions[row] = int(action)
                steps.append(
                    PatchStep(
                        global_features=global_batch,
                        action_features=action_batch,
                        action_mask=mask_batch,
                        action=actions,
                        reward=float(reward),
                        done=done,
                        old_log_prob=joint_old_log_prob,
                        old_value=joint_old_value,
                        joint_group_id=int(macro_step_id),
                        joint_old_log_prob=joint_old_log_prob,
                        joint_old_value=joint_old_value,
                        action_events=action_events,
                        n_local_actions=int(len(selected_actions)) if capture_trace else 0,
                        n_noop_actions=(
                            int(sum(int(item.get("bin_idx", -1)) < 0 for item in selected_actions))
                            if capture_trace
                            else 0
                        ),
                        phase=str(info.get("last_step_phase")) if capture_trace else None,
                        outcome=str(info.get("last_step_outcome")) if capture_trace else None,
                        patch_score_after=float(info.get("patch_score", 0.0)) if capture_trace else None,
                        raw_patch_score_after=(
                            float(info.get("raw_total_core_reward", 0.0)) if capture_trace else None
                        ),
                        owned_target_count_after=(
                            int(info.get("n_force_fill_owned_expression_bins", 0)) if capture_trace else None
                        ),
                        target_count=(
                            int(info.get("n_force_fill_expression_bins", 0)) if capture_trace else None
                        ),
                    )
                )
            macro_step_id += 1
            if done:
                break

        reward_per_step = float(total_reward / max(1, macro_step_id))
        metrics = {
            "patch_score": float(info.get("patch_score", 0.0)),
            "raw_total_reward": float(total_reward),
            "reward_per_step": reward_per_step,
            "n_core_cells": float(info.get("n_core_cells", 0)),
            "n_margin_cells": float(info.get("n_margin_cells", 0)),
            "n_patch_cells": float(info.get("n_patch_cells", 0)),
            "n_force_fill_expression_bins": float(info.get("n_force_fill_expression_bins", 0)),
            "n_force_fill_owned_expression_bins": float(info.get("n_force_fill_owned_expression_bins", 0)),
            "n_patch_steps": float(macro_step_id),
            "n_local_action_transitions": float(local_action_transitions),
        }
        trajectories.append(
            PatchTrajectory(
                patch_slot=env_idx,
                patch_id=str(contexts[env_idx].patch_id),
                steps=tuple(steps),
                total_reward=float(total_reward),
                patch_score=float(info.get("patch_score", 0.0)),
                metrics=metrics,
                final_masks=env.final_masks(),
                initial_masks=initial_masks[env_idx],
                initial_patch_score=(
                    float(infos[env_idx].get("patch_score", 0.0)) if capture_trace else None
                ),
                initial_raw_patch_score=(
                    float(infos[env_idx].get("raw_total_core_reward", 0.0)) if capture_trace else None
                ),
                initial_owned_target_count=(
                    int(infos[env_idx].get("n_force_fill_owned_expression_bins", 0))
                    if capture_trace
                    else None
                ),
                target_count=(
                    int(infos[env_idx].get("n_force_fill_expression_bins", 0))
                    if capture_trace
                    else None
                ),
            )
        )

    timing = {
        "rollout_total_sec": float(time.perf_counter() - t_total0),
        "rollout_tensor_sec": float(tensor_sec),
        "rollout_model_sec": float(model_sec),
        "rollout_env_sec": float(env_sec),
        "rollout_n_model_calls": float(model_calls),
    }
    return trajectories, timing


def _joint_action_from_observation(obs: dict[str, Any], action: int) -> dict[str, int | bool]:
    action_i = int(action)
    if action_i <= 0:
        return {
            "cell_idx": -1,
            "bin_idx": -1,
            "barcode_index": -1,
            "is_replace": False,
            "old_cell_idx": -1,
            "old_bin_idx": -1,
        }
    return {
        "cell_idx": int(obs["joint_action_cells"][action_i].detach().cpu().item()),
        "bin_idx": int(obs["joint_action_bins"][action_i].detach().cpu().item()),
        "barcode_index": int(obs["joint_action_barcodes"][action_i].detach().cpu().item()),
        "is_replace": bool(obs["joint_action_is_replace"][action_i].detach().cpu().item()),
        "old_cell_idx": int(obs["joint_action_old_cells"][action_i].detach().cpu().item()),
        "old_bin_idx": int(obs["joint_action_old_bins"][action_i].detach().cpu().item()),
    }


def _joint_action_events(
    *,
    context: PatchContext,
    selected_actions: list[dict[str, int | bool]],
    applied: bool,
) -> tuple[PatchActionEvent, ...]:
    events: list[PatchActionEvent] = []
    for item in selected_actions:
        bin_idx = int(item.get("bin_idx", -1))
        cell_idx = int(item.get("cell_idx", -1))
        if bin_idx < 0 or cell_idx < 0:
            continue
        cell_ctx = context.cells[cell_idx]
        old_cell_idx = int(item.get("old_cell_idx", -1))
        events.append(
            PatchActionEvent(
                action_type="replace" if bool(item.get("is_replace", False)) else "add",
                cell_id=str(cell_ctx.cell_id),
                barcode=str(cell_ctx.candidate_bin_ids[bin_idx]),
                old_cell_id=(
                    str(context.cells[old_cell_idx].cell_id)
                    if 0 <= old_cell_idx < len(context.cells)
                    else None
                ),
                applied=bool(applied),
            )
        )
    return tuple(events)


def collect_patch_trajectories_batched(
    *,
    contexts: list[PatchContext],
    model: ActorCritic,
    device: torch.device,
    rng: np.random.Generator,
    policy_mode: str = "sample",
    rollout_backend: str = "legacy_cpu",
    capture_trace: bool = False,
) -> tuple[list[PatchTrajectory], dict[str, float]]:
    """Collect patch rollouts with one padded model forward per active timestep."""
    import torch

    if contexts and str(getattr(contexts[0], "agent_mode", "")).strip().lower() == "multi_cell_joint_global_delta":
        return _collect_patch_trajectories_joint(
            contexts=contexts,
            model=model,
            device=device,
            rng=rng,
            policy_mode=policy_mode,
            rollout_backend=rollout_backend,
            capture_trace=capture_trace,
        )

    if not contexts:
        return [], {
            "rollout_total_sec": 0.0,
            "rollout_tensor_sec": 0.0,
            "rollout_model_sec": 0.0,
            "rollout_env_sec": 0.0,
            "rollout_n_model_calls": 0.0,
        }

    seed = int(rng.integers(low=0, high=np.iinfo(np.uint32).max, dtype=np.uint64))
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)

    mode = str(policy_mode).strip().lower()
    envs = [_build_patch_env(context=ctx, device=device, rollout_backend=rollout_backend) for ctx in contexts]
    observations: list[dict[str, Any]] = []
    infos: list[dict[str, Any]] = []
    for env in envs:
        obs, info = env.reset()
        observations.append(obs)
        infos.append(info)

    steps_by_env: list[list[PatchStep]] = [[] for _ in envs]
    total_rewards = [0.0 for _ in envs]
    done = [False for _ in envs]
    tensor_sec = 0.0
    model_sec = 0.0
    env_sec = 0.0
    model_calls = 0
    t_total0 = time.perf_counter()

    while not all(done):
        active = [idx for idx, is_done in enumerate(done) if not is_done]
        t0 = time.perf_counter()
        max_actions = max(int(observations[idx]["action_features"].shape[0]) for idx in active)
        if _obs_uses_torch(observations[active[0]]):
            global_t = torch.stack([observations[idx]["global_features"].to(device=device) for idx in active], dim=0)
            action_t = torch.zeros(
                (len(active), max_actions, ACTION_FEATURE_DIM),
                device=device,
                dtype=torch.float32,
            )
            mask_t = torch.zeros((len(active), max_actions), device=device, dtype=torch.bool)
            for row, idx in enumerate(active):
                action_features = observations[idx]["action_features"].to(device=device)
                action_mask = observations[idx]["action_mask"].to(device=device)
                n_actions = int(action_features.shape[0])
                action_t[row, :n_actions] = action_features
                mask_t[row, :n_actions] = action_mask
        else:
            global_batch = np.stack(
                [np.asarray(observations[idx]["global_features"], dtype=np.float32) for idx in active],
                axis=0,
            )
            action_batch = np.zeros((len(active), max_actions, ACTION_FEATURE_DIM), dtype=np.float32)
            mask_batch = np.zeros((len(active), max_actions), dtype=bool)
            for row, idx in enumerate(active):
                action_features = np.asarray(observations[idx]["action_features"], dtype=np.float32)
                action_mask = np.asarray(observations[idx]["action_mask"], dtype=bool)
                n_actions = int(action_features.shape[0])
                action_batch[row, :n_actions] = action_features
                mask_batch[row, :n_actions] = action_mask
            global_t = torch.as_tensor(global_batch, device=device, dtype=torch.float32)
            action_t = torch.as_tensor(action_batch, device=device, dtype=torch.float32)
            mask_t = torch.as_tensor(mask_batch, device=device, dtype=torch.bool)
        tensor_sec += float(time.perf_counter() - t0)

        t0 = time.perf_counter()
        with torch.inference_mode():
            dist, values = model(global_t, action_t, mask_t)
            if mode == "greedy":
                action_tensor = torch.argmax(dist.probs, dim=1)
            else:
                action_tensor = dist.sample()
            old_log_prob_tensor = dist.log_prob(action_tensor)
            actions = action_tensor.detach().cpu().numpy().astype(np.int64, copy=False)
            old_log_probs = old_log_prob_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
            old_values = values.detach().cpu().numpy().astype(np.float64, copy=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        model_sec += float(time.perf_counter() - t0)
        model_calls += 1

        t0 = time.perf_counter()
        for row, idx in enumerate(active):
            obs = observations[idx]
            next_obs, reward, terminated, truncated, info = envs[idx].step(int(actions[row]))
            step_done = bool(terminated or truncated)
            total_rewards[idx] += float(reward)
            steps_by_env[idx].append(
                PatchStep(
                    global_features=_obs_to_numpy(obs["global_features"], dtype=np.float32),
                    action_features=_obs_to_numpy(obs["action_features"], dtype=np.float32),
                    action_mask=_obs_to_numpy(obs["action_mask"], dtype=bool),
                    action=int(actions[row]),
                    reward=float(reward),
                    done=step_done,
                    old_log_prob=float(old_log_probs[row]),
                    old_value=float(old_values[row]),
                )
            )
            observations[idx] = next_obs
            infos[idx] = info
            done[idx] = step_done
        env_sec += float(time.perf_counter() - t0)

    trajectories: list[PatchTrajectory] = []
    for idx, context in enumerate(contexts):
        steps = tuple(steps_by_env[idx])
        total_reward = float(total_rewards[idx])
        info = infos[idx]
        reward_per_step = float(total_reward / max(1, len(steps)))
        metrics = {
            "patch_score": float(info.get("patch_score", 0.0)),
            "raw_total_reward": float(total_reward),
            "reward_per_step": reward_per_step,
            "n_core_cells": float(info.get("n_core_cells", 0)),
            "n_margin_cells": float(info.get("n_margin_cells", 0)),
            "n_patch_cells": float(info.get("n_patch_cells", 0)),
            "n_force_fill_expression_bins": float(info.get("n_force_fill_expression_bins", 0)),
            "n_force_fill_owned_expression_bins": float(info.get("n_force_fill_owned_expression_bins", 0)),
            "n_patch_steps": float(len(steps)),
        }
        trajectories.append(
            PatchTrajectory(
                patch_slot=idx,
                patch_id=str(context.patch_id),
                steps=steps,
                total_reward=total_reward,
                patch_score=float(info.get("patch_score", 0.0)),
                metrics=metrics,
                final_masks=envs[idx].final_masks(),
            )
        )

    timing = {
        "rollout_total_sec": float(time.perf_counter() - t_total0),
        "rollout_tensor_sec": float(tensor_sec),
        "rollout_model_sec": float(model_sec),
        "rollout_env_sec": float(env_sec),
        "rollout_n_model_calls": float(model_calls),
    }
    return trajectories, timing
