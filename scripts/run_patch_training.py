#!/usr/bin/env python
"""Train the ActorCritic policy on overlapping spatial patch episodes."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
import datetime as dt
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hd_cell_rl.patch_training import (
    PatchDataset,
    PatchTrainingSettings,
    build_patch_rollout_cache,
    collect_patch_contexts,
    collect_patch_trajectories_batched,
    patch_ppo_update,
)
from hd_cell_rl.ppo_checkpoint import build_actor_critic_from_config, load_checkpoint_payload
from hd_cell_rl.ppo_config import load_ppo_training_config
from hd_cell_rl.ppo_run_io import _append_step_log, _build_metadata, _slugify, _write_json, _write_yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Patch training YAML config.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    patch_cfg = _load_yaml(Path(args.config))
    base_config_path = Path(str(patch_cfg["base_ppo_config"])).expanduser().resolve()
    base_config = load_ppo_training_config(base_config_path)
    base_config = _apply_base_config_overrides(base_config, dict(patch_cfg.get("base_config_overrides", {})))

    run_cfg = dict(patch_cfg.get("run", {}))
    patch_training = dict(patch_cfg.get("patch_training", {}))
    run_name = str(run_cfg.get("name", "human_colorectal_patch_full_grpo"))
    output_root = Path(str(run_cfg.get("output_root", base_config.output_root))).expanduser().resolve()
    seed = int(run_cfg.get("seed", base_config.seed if base_config.seed is not None else 7))
    device_name = str(run_cfg.get("device", base_config.device)).lower()
    rollout_device_names = _parse_rollout_devices(run_cfg.get("rollout_devices", []))
    max_updates = int(run_cfg.get("max_updates", base_config.max_updates))
    batch_patches = int(run_cfg.get("batch_patches", patch_training.get("batch_patches", 8)))
    policy_mode = str(run_cfg.get("policy_mode", "sample"))
    rollout_backend = _parse_rollout_backend(str(patch_training.get("rollout_backend", "legacy_cpu")))
    reward_backend = _parse_reward_backend(str(patch_training.get("reward_backend", "standard")))
    stcs_reward_raw = patch_training.get("stcs_reward", {})
    if stcs_reward_raw is None:
        stcs_reward_config: dict[str, Any] = {}
    elif isinstance(stcs_reward_raw, dict):
        stcs_reward_config = dict(stcs_reward_raw)
    else:
        raise ValueError("patch_training.stcs_reward must be a mapping")
    rollout_worker_mode = _parse_rollout_worker_mode(str(run_cfg.get("rollout_worker_mode", "thread")))
    fill_target = _parse_fill_target(str(patch_training.get("fill_target", "reachable_expression_bins")))
    stop_action_mode = _parse_stop_action_mode(str(patch_training.get("stop_action_mode", "enabled")))
    agent_mode = _parse_patch_agent_mode(str(patch_training.get("agent_mode", "multi_cell")))
    after_fill_actions = _parse_after_fill_actions(str(patch_training.get("after_fill_actions", "add_or_stop")))
    global_delta_epsilon = float(patch_training.get("global_delta_epsilon", 1.0e-6))
    if global_delta_epsilon < 0.0:
        raise ValueError("patch_training.global_delta_epsilon must be >= 0")
    if agent_mode in {"single_cell_global_delta", "multi_cell_global_delta", "multi_cell_joint_global_delta"} and rollout_backend != "torch_gpu":
        raise ValueError(f"patch_training.agent_mode={agent_mode!r} requires rollout_backend='torch_gpu'")
    warm_start_raw = run_cfg.get("warm_start_checkpoint", "")
    warm_start_checkpoint = None if warm_start_raw in (None, "") else Path(str(warm_start_raw)).expanduser().resolve()

    # Patch v1 uses the low-level ADD/STOP policy only. Keeping planner disabled
    # avoids mixing old single-cell COT assumptions into patch-level competition.
    config = replace(
        base_config,
        run_name=run_name,
        output_root=output_root,
        seed=seed,
        device=device_name,
        max_updates=max_updates,
        planner_enabled=False,
        vf_coef=float(run_cfg.get("vf_coef", base_config.vf_coef)),
    )
    settings = PatchTrainingSettings(
        patches_index_path=Path(str(patch_training["patches_index_path"])).expanduser().resolve(),
        batch_patches=batch_patches,
        max_steps_per_patch=int(patch_training.get("max_steps_per_patch", 1000)),
        margin_cells_compete=bool(patch_training.get("margin_cells_compete", True)),
        use_core_cells_for_score=bool(patch_training.get("use_core_cells_for_score", True)),
        score_normalization=_parse_score_normalization(str(patch_training.get("score_normalization", "mean_core_cells"))),
        rollout_backend=rollout_backend,
        reward_backend=reward_backend,
        stcs_reward_config=stcs_reward_config,
        cache_patch_contexts=bool(patch_training.get("cache_patch_contexts", False)),
        competition_margin_enabled=bool(patch_training.get("competition_margin_enabled", True)),
        force_fill_expression_bins=bool(patch_training.get("force_fill_expression_bins", False)),
        fill_target=fill_target,
        stop_action_mode=stop_action_mode,
        agent_mode=agent_mode,
        after_fill_actions=after_fill_actions,
        global_delta_epsilon=global_delta_epsilon,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"{_slugify(run_name)}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=False, exist_ok=False)
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    checkpoints_dir.mkdir()
    logs_dir.mkdir()
    base_config_used_path = run_dir / "base_config_used.yaml"
    patch_config_used = json.loads(json.dumps(patch_cfg))
    patch_config_used["base_ppo_config"] = str(base_config_used_path)
    patch_run_used = patch_config_used.setdefault("run", {})
    patch_run_used.update(
        {
            "name": run_name,
            "output_root": str(output_root),
            "seed": seed,
            "device": device_name,
            "rollout_devices": rollout_device_names,
            "max_updates": max_updates,
            "batch_patches": batch_patches,
            "policy_mode": policy_mode,
            "rollout_worker_mode": rollout_worker_mode,
            "warm_start_checkpoint": None if warm_start_checkpoint is None else str(warm_start_checkpoint),
        }
    )
    patch_training_used = patch_config_used.setdefault("patch_training", {})
    patch_training_used.update(
        {
            "patches_index_path": str(settings.patches_index_path),
            "max_steps_per_patch": settings.max_steps_per_patch,
            "margin_cells_compete": settings.margin_cells_compete,
            "use_core_cells_for_score": settings.use_core_cells_for_score,
            "score_normalization": settings.score_normalization,
            "rollout_backend": settings.rollout_backend,
            "reward_backend": settings.reward_backend,
            "stcs_reward": settings.stcs_reward_config or {},
            "cache_patch_contexts": settings.cache_patch_contexts,
            "competition_margin_enabled": settings.competition_margin_enabled,
            "force_fill_expression_bins": settings.force_fill_expression_bins,
            "fill_target": settings.fill_target,
            "stop_action_mode": settings.stop_action_mode,
            "agent_mode": settings.agent_mode,
            "after_fill_actions": settings.after_fill_actions,
            "global_delta_epsilon": settings.global_delta_epsilon,
        }
    )
    _write_yaml(base_config_used_path, config.to_serializable_dict())
    _write_yaml(run_dir / "patch_config_used.yaml", patch_config_used)
    _write_json(run_dir / "metadata.json", _build_metadata(run_dir, seed))

    device = _resolve_device(device_name)
    rollout_devices = _resolve_rollout_devices(rollout_device_names, fallback=device)
    _configure_threads(device, config)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    dataset = PatchDataset(base_config=config, settings=settings, rng=rng)
    model = build_actor_critic_from_config(config, device=device)
    warm_start_info = None
    if warm_start_checkpoint is not None:
        warm_start_info = _load_warm_start_weights(model=model, checkpoint_path=warm_start_checkpoint, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay))

    steps_log = logs_dir / "steps.jsonl"
    _append_step_log(
        steps_log,
        "run_start",
        {
            "run_name": run_name,
            "training_unit": "patch",
            "n_patches_total": dataset.n_patches,
            "batch_patches": batch_patches,
            "max_steps_per_patch": settings.max_steps_per_patch,
            "training_mode": config.training_mode,
            "group_size": config.group_relative_group_size,
            "device": str(device),
            "rollout_devices": [str(item) for item in rollout_devices],
            "rollout_backend": settings.rollout_backend,
            "reward_backend": settings.reward_backend,
            "stcs_reward": settings.stcs_reward_config or {},
            "rollout_worker_mode": rollout_worker_mode,
            "cache_patch_contexts": settings.cache_patch_contexts,
            "competition_margin_enabled": settings.competition_margin_enabled,
            "force_fill_expression_bins": settings.force_fill_expression_bins,
            "fill_target": settings.fill_target,
            "stop_action_mode": settings.stop_action_mode,
            "agent_mode": settings.agent_mode,
            "after_fill_actions": settings.after_fill_actions,
            "global_delta_epsilon": settings.global_delta_epsilon,
            "warm_start_checkpoint": None if warm_start_checkpoint is None else str(warm_start_checkpoint),
            "warm_start_info": warm_start_info,
        },
    )

    reward_history: list[float] = []
    best_moving_avg: float | None = None
    best_checkpoint: Path | None = None
    no_improve = 0
    logs: list[dict[str, Any]] = []
    process_pool: ProcessPoolExecutor | None = None
    if rollout_worker_mode == "process" and len(rollout_devices) > 1:
        process_pool = ProcessPoolExecutor(
            max_workers=len(rollout_devices),
            mp_context=mp.get_context("spawn"),
        )
    try:
        for update_idx in range(1, max_updates + 1):
            t0 = time.perf_counter()
            group_size = int(config.group_relative_group_size) if (
                bool(config.group_relative_enabled) or str(config.training_mode) == "full_grpo"
            ) else 1
            _append_step_log(
                steps_log,
                event="update_start",
                payload={
                    "update_index": int(update_idx),
                    "batch_patches": int(batch_patches),
                    "group_size": int(group_size),
                    "rollout_worker_mode": str(rollout_worker_mode),
                },
            )
            t_context0 = time.perf_counter()
            contexts = collect_patch_contexts(dataset=dataset, batch_patches=batch_patches, group_size=group_size)
            if not contexts:
                raise RuntimeError("failed to collect patch contexts")
            context_sec = float(time.perf_counter() - t_context0)
            _append_step_log(
                steps_log,
                event="contexts_collected",
                payload={
                    "update_index": int(update_idx),
                    "n_contexts": int(len(contexts)),
                    "n_unique_patch_ids": int(len({str(ctx.patch_id) for ctx in contexts})),
                    "time_context_sec": float(context_sec),
                },
            )

            model.eval()
            _append_step_log(
                steps_log,
                event="rollout_start",
                payload={
                    "update_index": int(update_idx),
                    "n_contexts": int(len(contexts)),
                    "rollout_devices": [str(item) for item in rollout_devices],
                    "rollout_backend": str(settings.rollout_backend),
                    "rollout_worker_mode": str(rollout_worker_mode),
                },
            )
            trajectories, rollout_timing = _collect_patch_trajectories_for_devices(
                contexts=contexts,
                model=model,
                config=config,
                device=device,
                rollout_devices=rollout_devices,
                rollout_backend=settings.rollout_backend,
                rollout_worker_mode=rollout_worker_mode,
                process_pool=process_pool,
                rng=rng,
                policy_mode=policy_mode,
                group_size=group_size,
            )
            _append_step_log(
                steps_log,
                event="rollout_complete",
                payload={
                    "update_index": int(update_idx),
                    "n_episodes": int(len(trajectories)),
                    "n_transitions": int(sum(len(traj.steps) for traj in trajectories)),
                    "time_rollout_total_sec": float(rollout_timing.get("rollout_total_sec", 0.0)),
                    "rollout_model_calls": float(rollout_timing.get("rollout_n_model_calls", 0.0)),
                },
            )
            model.train()
            t_cache0 = time.perf_counter()
            cache = build_patch_rollout_cache(
                trajectories=trajectories,
                gamma=float(config.gamma),
                gae_lambda=float(config.gae_lambda),
                normalize_advantages=bool(config.normalize_advantages),
                training_mode=str(config.training_mode),
                group_size=group_size,
                norm_epsilon=float(config.group_relative_norm_epsilon),
            )
            cache_sec = float(time.perf_counter() - t_cache0)
            t_ppo0 = time.perf_counter()
            _append_step_log(
                steps_log,
                event="ppo_update_start",
                payload={
                    "update_index": int(update_idx),
                    "n_transitions": int(cache.n_transitions),
                    "time_cache_sec": float(cache_sec),
                },
            )
            metrics = patch_ppo_update(
                model=model,
                optimizer=optimizer,
                cache=cache,
                eps_clip=float(config.eps_clip),
                ppo_epochs=int(config.ppo_epochs),
                minibatch_size=int(config.minibatch_size),
                vf_coef=float(config.vf_coef),
                ent_coef=float(config.ent_coef),
                max_grad_norm=float(config.max_grad_norm),
                target_kl=config.target_kl,
                include_value_loss=str(config.training_mode) != "full_grpo",
                device=device,
                rng=rng,
            )
            ppo_update_sec = float(time.perf_counter() - t_ppo0)
            _empty_cuda_cache(device)
            avg_patch_score = float(np.mean([traj.patch_score for traj in trajectories]))
            avg_raw_reward = float(np.mean([traj.total_reward for traj in trajectories]))
            avg_steps = float(np.mean([traj.metrics.get("n_patch_steps", float(len(traj.steps))) for traj in trajectories]))
            avg_core_cells = float(np.mean([traj.metrics.get("n_core_cells", 0.0) for traj in trajectories]))
            avg_force_fill_bins = float(
                np.mean([traj.metrics.get("n_force_fill_expression_bins", 0.0) for traj in trajectories])
            )
            avg_force_fill_owned_bins = float(
                np.mean([traj.metrics.get("n_force_fill_owned_expression_bins", 0.0) for traj in trajectories])
            )
            reward_history.append(avg_patch_score)
            moving_avg = None
            checkpoint_saved = False
            if len(reward_history) >= int(config.moving_avg_window):
                moving_avg = float(np.mean(reward_history[-int(config.moving_avg_window) :]))
                if best_moving_avg is None or moving_avg > best_moving_avg + float(config.min_improvement):
                    best_moving_avg = moving_avg
                    no_improve = 0
                    checkpoint_saved = True
                    best_checkpoint = checkpoints_dir / "best_model.pt"
                    _save_patch_checkpoint(
                        path=best_checkpoint,
                        model=model,
                        optimizer=optimizer,
                        update_index=update_idx,
                        best_moving_avg_reward=best_moving_avg,
                        base_config=config,
                        patch_config=patch_config_used,
                    )
                else:
                    no_improve += 1

            row = {
                "update_index": update_idx,
                "average_batch_reward": avg_patch_score,
                "average_raw_patch_reward": avg_raw_reward,
                "average_patch_steps": avg_steps,
                "average_core_cells": avg_core_cells,
                "average_force_fill_expression_bins": avg_force_fill_bins,
                "average_force_fill_owned_expression_bins": avg_force_fill_owned_bins,
                "moving_average_reward": moving_avg,
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
                "entropy": metrics["entropy"],
                "total_loss": metrics["total_loss"],
                "approx_kl": metrics["approx_kl"],
                "no_improve_count": no_improve,
                "n_episodes": len(trajectories),
                "n_transitions": cache.n_transitions,
                "time_context_sec": context_sec,
                "time_rollout_total_sec": rollout_timing["rollout_total_sec"],
                "time_rollout_tensor_sec": rollout_timing["rollout_tensor_sec"],
                "time_rollout_model_sec": rollout_timing["rollout_model_sec"],
                "time_rollout_env_sec": rollout_timing["rollout_env_sec"],
                "rollout_model_calls": rollout_timing["rollout_n_model_calls"],
                "rollout_worker_mode": rollout_worker_mode,
                "time_cache_sec": cache_sec,
                "time_ppo_update_sec": ppo_update_sec,
                "time_total_sec": float(time.perf_counter() - t0),
            }
            logs.append(row)
            _append_step_log(steps_log, "update_complete", {**row, "checkpoint_saved": checkpoint_saved})
            if len(reward_history) >= int(config.moving_avg_window) and no_improve >= int(config.patience):
                _append_step_log(steps_log, "early_stop", {"update_index": update_idx, "no_improve_count": no_improve})
                break
    finally:
        if process_pool is not None:
            process_pool.shutdown(wait=True, cancel_futures=True)
        dataset.close()

    if best_checkpoint is None:
        best_checkpoint = checkpoints_dir / "final_model.pt"
        _save_patch_checkpoint(
            path=best_checkpoint,
            model=model,
            optimizer=optimizer,
            update_index=len(logs),
            best_moving_avg_reward=best_moving_avg,
            base_config=config,
            patch_config=patch_config_used,
        )
    _write_json(
        run_dir / "summary.json",
        {
            "run_dir": str(run_dir),
            "best_checkpoint_path": str(best_checkpoint),
            "best_moving_average_reward": best_moving_avg,
            "updates_completed": len(logs),
            "training_unit": "patch",
            "batch_patches": batch_patches,
            "patches_index_path": str(settings.patches_index_path),
            "rollout_devices": [str(item) for item in rollout_devices],
            "rollout_backend": settings.rollout_backend,
            "reward_backend": settings.reward_backend,
            "stcs_reward": settings.stcs_reward_config or {},
            "rollout_worker_mode": rollout_worker_mode,
            "cache_patch_contexts": settings.cache_patch_contexts,
            "competition_margin_enabled": settings.competition_margin_enabled,
            "force_fill_expression_bins": settings.force_fill_expression_bins,
            "fill_target": settings.fill_target,
            "stop_action_mode": settings.stop_action_mode,
            "agent_mode": settings.agent_mode,
            "after_fill_actions": settings.after_fill_actions,
            "global_delta_epsilon": settings.global_delta_epsilon,
            "warm_start_checkpoint": None if warm_start_checkpoint is None else str(warm_start_checkpoint),
            "warm_start_info": warm_start_info,
        },
    )
    print(f"Patch training complete: {run_dir}")
    print(f"Best checkpoint: {best_checkpoint}")


def _save_patch_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    update_index: int,
    best_moving_avg_reward: float | None,
    base_config: Any,
    patch_config: dict[str, Any],
) -> None:
    torch.save(
        {
            "update_index": int(update_index),
            "best_moving_avg_reward": best_moving_avg_reward,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": base_config.to_serializable_dict(),
            "patch_config": patch_config,
            "training_unit": "patch",
        },
        path,
    )


def _load_warm_start_weights(*, model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    """Initialize matching model tensors from a previous single-cell checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"warm-start checkpoint not found: {checkpoint_path}")
    payload = load_checkpoint_payload(checkpoint_path)
    source_state = payload.get("model_state_dict")
    if not isinstance(source_state, dict):
        raise ValueError(f"warm-start checkpoint missing model_state_dict: {checkpoint_path}")
    target_state = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []
    skipped_missing: list[str] = []
    for key, tensor in source_state.items():
        if key not in target_state:
            skipped_missing.append(str(key))
            continue
        if tuple(target_state[key].shape) != tuple(tensor.shape):
            skipped_shape.append(str(key))
            continue
        matched[key] = tensor.detach().to(device=device, dtype=target_state[key].dtype)
    if not matched:
        raise RuntimeError(f"warm-start checkpoint has no tensors matching the patch model: {checkpoint_path}")
    target_state.update(matched)
    model.load_state_dict(target_state, strict=True)
    info = {
        "path": str(checkpoint_path),
        "loaded_tensors": int(len(matched)),
        "skipped_missing": int(len(skipped_missing)),
        "skipped_shape": int(len(skipped_shape)),
    }
    print(f"Warm-start loaded: {info}")
    return info


def _collect_patch_trajectories_for_devices(
    *,
    contexts: list[Any],
    model: torch.nn.Module,
    config: Any,
    device: torch.device,
    rollout_devices: list[torch.device],
    rollout_backend: str,
    rollout_worker_mode: str,
    process_pool: ProcessPoolExecutor | None,
    rng: np.random.Generator,
    policy_mode: str,
    group_size: int,
) -> tuple[list[Any], dict[str, float]]:
    if len(rollout_devices) <= 1:
        trajectories, timing = collect_patch_trajectories_batched(
            contexts=contexts,
            model=model,
            device=device,
            rng=rng,
            policy_mode=policy_mode,
            rollout_backend=rollout_backend,
        )
        return trajectories, timing

    worker_mode = _parse_rollout_worker_mode(rollout_worker_mode)
    state_dict_payload = {
        key: value.detach().cpu().numpy().copy()
        for key, value in model.state_dict().items()
    }
    seeds = rng.integers(low=0, high=np.iinfo(np.uint32).max, size=len(contexts), dtype=np.uint64)
    assignments = _assign_rollout_indices_to_devices(
        n_contexts=len(contexts),
        n_devices=len(rollout_devices),
        group_size=group_size,
    )

    pairs: list[tuple[int, Any]] = []
    timing_sum = {
        "rollout_total_sec": 0.0,
        "rollout_tensor_sec": 0.0,
        "rollout_model_sec": 0.0,
        "rollout_env_sec": 0.0,
        "rollout_n_model_calls": 0.0,
    }
    wall0 = time.perf_counter()
    payloads = [
        {
            "device": str(rollout_devices[device_index]),
            "indices": indices,
            "contexts": [contexts[idx] for idx in indices],
            "config": config,
            "state_dict": state_dict_payload,
            "seed": int(seeds[indices[0]]),
            "policy_mode": policy_mode,
            "rollout_backend": rollout_backend,
        }
        for device_index, indices in enumerate(assignments)
        if indices
    ]
    if worker_mode == "process":
        if process_pool is not None:
            futures = [process_pool.submit(_run_patch_rollout_process, payload) for payload in payloads]
            for future in futures:
                device_pairs, device_timing = future.result()
                pairs.extend(device_pairs)
                for key in timing_sum:
                    timing_sum[key] += float(device_timing.get(key, 0.0))
        else:
            with ProcessPoolExecutor(
                max_workers=len(rollout_devices),
                mp_context=mp.get_context("spawn"),
            ) as executor:
                futures = [executor.submit(_run_patch_rollout_process, payload) for payload in payloads]
                for future in futures:
                    device_pairs, device_timing = future.result()
                    pairs.extend(device_pairs)
                    for key in timing_sum:
                        timing_sum[key] += float(device_timing.get(key, 0.0))
    else:
        def run_on_device(payload: dict[str, Any]) -> tuple[list[tuple[int, Any]], dict[str, float]]:
            local_device = torch.device(str(payload["device"]))
            if local_device.type == "cuda":
                torch.cuda.set_device(local_device)
            local_model = build_actor_critic_from_config(config, device=local_device)
            local_model.load_state_dict(_state_dict_payload_to_tensors(payload["state_dict"]), strict=True)
            local_model.eval()
            local_rng = np.random.default_rng(int(payload["seed"]))
            trajectories, timing = collect_patch_trajectories_batched(
                contexts=list(payload["contexts"]),
                model=local_model,
                device=local_device,
                rng=local_rng,
                policy_mode=policy_mode,
                rollout_backend=rollout_backend,
            )
            return list(zip(list(payload["indices"]), trajectories)), timing

        with ThreadPoolExecutor(max_workers=len(rollout_devices)) as executor:
            futures = [executor.submit(run_on_device, payload) for payload in payloads]
            for future in futures:
                device_pairs, device_timing = future.result()
                pairs.extend(device_pairs)
                for key in timing_sum:
                    timing_sum[key] += float(device_timing.get(key, 0.0))
    timing_sum["rollout_total_sec"] = float(time.perf_counter() - wall0)
    return [traj for _, traj in sorted(pairs, key=lambda item: item[0])], timing_sum


def _run_patch_rollout_process(payload: dict[str, Any]) -> tuple[list[tuple[int, Any]], dict[str, float]]:
    local_device = torch.device(str(payload["device"]))
    if local_device.type == "cuda":
        torch.cuda.set_device(local_device)
    try:
        local_model = build_actor_critic_from_config(payload["config"], device=local_device)
        local_model.load_state_dict(_state_dict_payload_to_tensors(payload["state_dict"]), strict=True)
        local_model.eval()
        local_rng = np.random.default_rng(int(payload["seed"]))
        trajectories, timing = collect_patch_trajectories_batched(
            contexts=list(payload["contexts"]),
            model=local_model,
            device=local_device,
            rng=local_rng,
            policy_mode=str(payload["policy_mode"]),
            rollout_backend=str(payload["rollout_backend"]),
        )
        return list(zip(list(payload["indices"]), trajectories)), timing
    finally:
        if "local_model" in locals():
            del local_model
        _empty_cuda_cache(local_device)


def _empty_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _state_dict_payload_to_tensors(state_dict_payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        str(key): value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        for key, value in state_dict_payload.items()
    }


def _assign_rollout_indices_to_devices(*, n_contexts: int, n_devices: int, group_size: int) -> list[list[int]]:
    assignments = [[] for _ in range(max(1, int(n_devices)))]
    chunk_size = int(group_size) if int(group_size) > 1 and int(n_contexts) % int(group_size) == 0 else 1
    chunks = [list(range(start, min(start + chunk_size, int(n_contexts)))) for start in range(0, int(n_contexts), chunk_size)]
    for chunk_index, chunk in enumerate(chunks):
        assignments[chunk_index % len(assignments)].extend(chunk)
    return assignments


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def _apply_base_config_overrides(config: Any, overrides: dict[str, Any]) -> Any:
    """Apply patch-only overrides to the single-cell base PPO config."""
    if not overrides:
        return config
    reward = dict(overrides.get("reward", {}))
    shape_prior = dict(overrides.get("shape_prior", {}))
    stopping = dict(overrides.get("stopping", {}))
    fields: dict[str, Any] = {}
    allowed_reward = {
        "w1": "w1",
        "w2": "w2",
        "w3": "w3",
        "w4": "w4",
        "w5": "w5",
        "stop_lambda": "stop_lambda",
        "competition_margin_weight": "competition_margin_weight",
        "competition_margin_radius_um": "competition_margin_radius_um",
        "competition_margin_clip": "competition_margin_clip",
    }
    for raw_name, field_name in allowed_reward.items():
        if raw_name in reward:
            fields[field_name] = float(reward[raw_name])
    if "competition_margin_affects_stop" in reward:
        fields["competition_margin_affects_stop"] = bool(reward["competition_margin_affects_stop"])
    if "weight" in shape_prior:
        fields["shape_prior_weight"] = float(shape_prior["weight"])
    if "moving_avg_window" in stopping:
        moving_avg_window = int(stopping["moving_avg_window"])
        if moving_avg_window <= 0:
            raise ValueError("base_config_overrides.stopping.moving_avg_window must be > 0")
        fields["moving_avg_window"] = moving_avg_window
    if "min_improvement" in stopping:
        min_improvement = float(stopping["min_improvement"])
        if min_improvement < 0:
            raise ValueError("base_config_overrides.stopping.min_improvement must be >= 0")
        fields["min_improvement"] = min_improvement
    if "patience" in stopping:
        patience = int(stopping["patience"])
        if patience <= 0:
            raise ValueError("base_config_overrides.stopping.patience must be > 0")
        fields["patience"] = patience
    unknown_sections = sorted(set(overrides) - {"reward", "shape_prior", "stopping"})
    if unknown_sections:
        raise ValueError(f"unsupported base_config_overrides sections: {unknown_sections}")
    unknown_reward = sorted(set(reward) - set(allowed_reward) - {"competition_margin_affects_stop"})
    if unknown_reward:
        raise ValueError(f"unsupported base_config_overrides.reward keys: {unknown_reward}")
    unknown_shape = sorted(set(shape_prior) - {"weight"})
    if unknown_shape:
        raise ValueError(f"unsupported base_config_overrides.shape_prior keys: {unknown_shape}")
    unknown_stopping = sorted(set(stopping) - {"moving_avg_window", "min_improvement", "patience"})
    if unknown_stopping:
        raise ValueError(f"unsupported base_config_overrides.stopping keys: {unknown_stopping}")
    return replace(config, **fields)


def _parse_score_normalization(value: str) -> str:
    normalized = value.strip().lower()
    allowed = {"mean_core_cells", "sqrt_core_cells", "sum_core_cells", "mean_expression_bins", "sqrt_expression_bins"}
    if normalized not in allowed:
        raise ValueError(f"patch_training.score_normalization must be one of {sorted(allowed)}")
    return normalized


def _parse_fill_target(value: str) -> str:
    normalized = value.strip().lower()
    allowed = {"reachable_expression_bins"}
    if normalized not in allowed:
        raise ValueError(f"patch_training.fill_target must be one of {sorted(allowed)}")
    return normalized


def _parse_stop_action_mode(value: str) -> str:
    normalized = value.strip().lower()
    allowed = {"enabled", "mask_until_filled", "auto_no_improve_after_filled"}
    if normalized not in allowed:
        raise ValueError(f"patch_training.stop_action_mode must be one of {sorted(allowed)}")
    return normalized


def _parse_patch_agent_mode(value: str) -> str:
    normalized = value.strip().lower()
    allowed = {"multi_cell", "single_cell_global_delta", "multi_cell_global_delta", "multi_cell_joint_global_delta"}
    if normalized not in allowed:
        raise ValueError(f"patch_training.agent_mode must be one of {sorted(allowed)}")
    return normalized


def _parse_after_fill_actions(value: str) -> str:
    normalized = value.strip().lower()
    allowed = {"add_or_stop", "replace_only"}
    if normalized not in allowed:
        raise ValueError(f"patch_training.after_fill_actions must be one of {sorted(allowed)}")
    return normalized


def _parse_rollout_backend(value: str) -> str:
    normalized = value.strip().lower()
    allowed = {"legacy_cpu", "cached_cpu", "torch_gpu"}
    if normalized not in allowed:
        raise ValueError(f"patch_training.rollout_backend must be one of {sorted(allowed)}")
    return normalized


def _parse_reward_backend(value: str) -> str:
    normalized = value.strip().lower()
    allowed = {"standard", "stcs"}
    if normalized not in allowed:
        raise ValueError(f"patch_training.reward_backend must be one of {sorted(allowed)}")
    return normalized


def _parse_rollout_worker_mode(value: str) -> str:
    normalized = value.strip().lower()
    allowed = {"thread", "process"}
    if normalized not in allowed:
        raise ValueError(f"run.rollout_worker_mode must be one of {sorted(allowed)}")
    return normalized


def _parse_rollout_devices(raw: Any) -> list[str]:
    if raw in (None, ""):
        return []
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    raise ValueError("run.rollout_devices must be a comma-separated string or a list")


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested cuda but CUDA is not available")
    return torch.device(device_name)


def _resolve_rollout_devices(raw_devices: list[str], *, fallback: torch.device) -> list[torch.device]:
    if not raw_devices:
        return [fallback]
    devices = [torch.device(item) for item in raw_devices]
    for item in devices:
        if item.type != "cuda":
            continue
        if not torch.cuda.is_available():
            raise RuntimeError(f"requested rollout device {item}, but CUDA is not available")
        if item.index is not None and item.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"requested rollout device {item}, but only {torch.cuda.device_count()} CUDA device(s) are visible"
            )
    return devices


def _configure_threads(device: torch.device, config: Any) -> None:
    if device.type != "cpu":
        return
    torch.set_num_threads(max(1, int(config.n_rollout_workers)))
    torch.set_num_interop_threads(1)


if __name__ == "__main__":
    main()
