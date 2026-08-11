#!/usr/bin/env python
"""Evaluate a patch-trained checkpoint and merge overlapping cell predictions."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hd_cell_rl.patch_training import (
    PatchDataset,
    PatchContext,
    PatchTrainingSettings,
    PatchTrajectory,
    _parse_square_barcode,
    collect_patch_trajectories_batched,
    patch_assignments_for_core_cells,
)
from hd_cell_rl.ppo_checkpoint import build_actor_critic_from_config, load_checkpoint_payload
from hd_cell_rl.ppo_config import load_ppo_training_config
from preprocessing.ppo_format_assignment_eval import (
    add_ppo_format_assignment_eval_args,
    load_eval_cell_ids,
    run_ppo_format_assignment_evaluation,
    validate_ppo_format_assignment_eval_args,
)
from preprocessing.patch_debug_export import (
    export_patch_debug_bundle,
    update_evaluation_summary_with_patch_debug,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--patches-index-path", default=None)
    parser.add_argument("--output-dir", default="workspace_outputs/pseudo_human_colorectal/patch_eval")
    parser.add_argument("--max-target-cells", type=int, default=0)
    parser.add_argument("--eval-mode", choices=("ppo_cells", "random_patches"), default="ppo_cells")
    parser.add_argument("--max-eval-patches", type=int, default=0)
    parser.add_argument("--policy-mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--rollout-devices", default="")
    parser.add_argument("--rollout-backend", choices=("legacy_cpu", "cached_cpu", "torch_gpu"), default=None)
    parser.add_argument("--external_nuclear_bins_path", type=str, default=None)
    parser.add_argument(
        "--eval-run-path-file",
        type=str,
        default=None,
        help="Write the completed evaluation run directory to this file.",
    )
    add_ppo_format_assignment_eval_args(
        parser,
        default_eval_run_name="human_colorectal_patch_eval",
        method_label="patch RL",
    )
    args = parser.parse_args()
    if args.ppo_eval_run_dir is None:
        raise ValueError("--ppo_eval_run_dir is required")
    if int(args.max_target_cells) < 0:
        raise ValueError("--max-target-cells must be >= 0")
    if int(args.max_eval_patches) < 0:
        raise ValueError("--max-eval-patches must be >= 0")
    if args.eval_mode == "random_patches" and int(args.max_eval_patches) <= 0:
        raise ValueError("--max-eval-patches must be > 0 when --eval-mode=random_patches")
    validate_ppo_format_assignment_eval_args(args)
    return args


def main() -> None:
    args = _parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    payload = load_checkpoint_payload(checkpoint)
    patch_cfg = dict(payload.get("patch_config") or {})
    base_config_path = Path(str(args.base_config or patch_cfg.get("base_ppo_config"))).expanduser().resolve()
    base_config = load_ppo_training_config(base_config_path)
    config = replace(base_config, planner_enabled=False)
    patches_index_path = Path(
        str(args.patches_index_path or patch_cfg.get("patch_training", {}).get("patches_index_path"))
    ).expanduser().resolve()
    settings = PatchTrainingSettings(
        patches_index_path=patches_index_path,
        batch_patches=1,
        max_steps_per_patch=int(patch_cfg.get("patch_training", {}).get("max_steps_per_patch", 1000)),
        margin_cells_compete=bool(patch_cfg.get("patch_training", {}).get("margin_cells_compete", True)),
        use_core_cells_for_score=True,
        score_normalization=str(patch_cfg.get("patch_training", {}).get("score_normalization", "mean_core_cells")),
        rollout_backend=str(
            args.rollout_backend or patch_cfg.get("patch_training", {}).get("rollout_backend", "legacy_cpu")
        ),
        reward_backend=str(patch_cfg.get("patch_training", {}).get("reward_backend", "standard")),
        stcs_reward_config=dict(patch_cfg.get("patch_training", {}).get("stcs_reward", {}) or {}),
        cache_patch_contexts=bool(patch_cfg.get("patch_training", {}).get("cache_patch_contexts", False)),
        competition_margin_enabled=bool(
            patch_cfg.get("patch_training", {}).get("competition_margin_enabled", True)
        ),
        force_fill_expression_bins=bool(
            patch_cfg.get("patch_training", {}).get("force_fill_expression_bins", False)
        ),
        fill_target=str(patch_cfg.get("patch_training", {}).get("fill_target", "reachable_expression_bins")),
        stop_action_mode=str(patch_cfg.get("patch_training", {}).get("stop_action_mode", "enabled")),
        agent_mode=str(patch_cfg.get("patch_training", {}).get("agent_mode", "multi_cell")),
        after_fill_actions=str(patch_cfg.get("patch_training", {}).get("after_fill_actions", "add_or_stop")),
        global_delta_epsilon=float(patch_cfg.get("patch_training", {}).get("global_delta_epsilon", 1.0e-6)),
    )

    device = _resolve_device(str(args.device))
    rollout_devices = _resolve_rollout_devices(_parse_rollout_devices(args.rollout_devices), fallback=device)
    model = build_actor_critic_from_config(config, device=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    ppo_eval_dir = Path(str(args.ppo_eval_run_dir)).expanduser().resolve()
    patches_df = pd.read_csv(patches_index_path)
    rng = np.random.default_rng(int(args.eval_seed))
    if str(args.eval_mode) == "random_patches":
        selected_rows = _select_random_patch_rows(
            patches_df=patches_df,
            max_eval_patches=int(args.max_eval_patches),
            rng=rng,
        )
        target_ids = _target_ids_from_patch_rows(selected_rows)
        target_set = set(target_ids)
    else:
        target_ids = load_eval_cell_ids(ppo_eval_dir / "per_episode.csv")
        if int(args.max_target_cells) > 0:
            target_ids = target_ids[: int(args.max_target_cells)]
        target_set = set(target_ids)
        selected_rows = _select_patch_rows_for_target_cells(patches_df=patches_df, target_set=target_set)
    if not selected_rows:
        raise RuntimeError("no eval patches were selected")

    out_dir = Path(args.output_dir).expanduser().resolve() / f"patch_eval_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=False)
    assignments_csv = out_dir / "patch_rl_assignments.csv"
    patch_assignments_csv = out_dir / "patch_rl_patch_assignments.csv"
    merge_csv = out_dir / "patch_rl_merge_candidates.csv"
    if str(args.eval_mode) == "random_patches":
        args.ppo_eval_run_dir = str(
            _write_patch_sample_eval_source(
                source_eval_dir=ppo_eval_dir,
                output_dir=out_dir,
                target_cell_ids=target_ids,
                fallback_config_path=base_config_path,
            )
        )

    dataset = PatchDataset(base_config=config, settings=settings, rng=rng)
    contexts: list[PatchContext] = []
    best_rows_by_cell: dict[str, tuple[float, list[dict[str, Any]]]] = {}
    merge_records: list[dict[str, Any]] = []
    patch_assignment_rows: list[dict[str, Any]] = []
    try:
        for row in selected_rows:
            ctx = dataset.load_patch_context(row)
            if ctx is None:
                continue
            contexts.append(ctx)
    finally:
        dataset.close()
    if not contexts:
        raise RuntimeError("no eval patch contexts could be loaded")

    trajectories = _collect_eval_trajectories_for_devices(
        contexts=contexts,
        model=model,
        config=config,
        device=device,
        rollout_devices=rollout_devices,
        rollout_backend=settings.rollout_backend,
        rng=rng,
        policy_mode=str(args.policy_mode),
    )
    for idx, (ctx, traj) in enumerate(zip(contexts, trajectories), start=1):
        rows_by_cell = patch_assignments_for_core_cells(context=ctx, final_masks=traj.final_masks)
        for cell_id, rows in rows_by_cell.items():
            patch_assignment_rows.extend(rows)
            if cell_id not in target_set:
                continue
            score = float(traj.patch_score)
            merge_records.append(
                {
                    "cell_id": cell_id,
                    "patch_id": ctx.patch_id,
                    "merge_score": score,
                    "n_assigned_bins": len(rows),
                    "patch_score": traj.patch_score,
                }
            )
            current = best_rows_by_cell.get(cell_id)
            if current is None or score > current[0]:
                best_rows_by_cell[cell_id] = (score, rows)
        if idx % 25 == 0 or idx == len(contexts):
            print(f"Evaluated {idx}/{len(contexts)} patches")

    assignment_rows: list[dict[str, Any]] = []
    for _, rows in best_rows_by_cell.values():
        assignment_rows.extend(rows)
    if not assignment_rows:
        raise RuntimeError("patch evaluation produced no assignment rows")
    pd.DataFrame(assignment_rows).to_csv(assignments_csv, index=False)
    pd.DataFrame(patch_assignment_rows).to_csv(patch_assignments_csv, index=False)
    pd.DataFrame(merge_records).to_csv(merge_csv, index=False)

    pipeline = {
        "checkpoint": str(checkpoint),
        "base_config": str(base_config_path),
        "patches_index_path": str(patches_index_path),
        "assignments_csv": str(assignments_csv),
        "patch_assignments_csv": str(patch_assignments_csv),
        "merge_candidates_csv": str(merge_csv),
        "n_target_cells": int(len(target_ids)),
        "n_selected_patches": int(len(selected_rows)),
        "n_merged_cells": int(len(best_rows_by_cell)),
        "eval_mode": str(args.eval_mode),
        "max_eval_patches": int(args.max_eval_patches),
        "policy_mode": str(args.policy_mode),
        "rollout_backend": str(settings.rollout_backend),
        "reward_backend": str(settings.reward_backend),
        "competition_margin_enabled": bool(settings.competition_margin_enabled),
        "agent_mode": str(settings.agent_mode),
        "after_fill_actions": str(settings.after_fill_actions),
        "global_delta_epsilon": float(settings.global_delta_epsilon),
        "rollout_devices": [str(item) for item in rollout_devices],
    }
    with (out_dir / "patch_eval_pipeline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(pipeline, handle, indent=2)
        handle.write("\n")

    eval_run = run_ppo_format_assignment_evaluation(
        assignments_csv=assignments_csv,
        method_name="patch_rl",
        method_label="Patch RL",
        nuclear_source="ppo_episode_seed",
        external_nuclear_bins_path=(
            None
            if args.external_nuclear_bins_path is None
            else Path(str(args.external_nuclear_bins_path)).expanduser().resolve()
        ),
        args=args,
        pipeline_config=pipeline,
    )
    patch_debug_manifest = None
    if args.gt_cell_bins_path is not None:
        trajectory_metadata = {
            str(ctx.patch_id): _patch_debug_trajectory_metadata(context=ctx, trajectory=traj)
            for ctx, traj in zip(contexts, trajectories)
        }
        nucleus_centers = {
            str(ctx.patch_id): {
                str(cell.cell_id): np.asarray(cell.nucleus_center_xy_um, dtype=np.float64).tolist()
                for cell in ctx.cells
                if str(cell.cell_id) in set(ctx.core_cell_ids)
            }
            for ctx in contexts
        }
        patch_debug_manifest = export_patch_debug_bundle(
            patch_assignments_csv=patch_assignments_csv,
            patches_index_path=patches_index_path,
            per_episode_csv=eval_run / "per_episode.csv",
            gt_cell_bins_path=Path(str(args.gt_cell_bins_path)).expanduser().resolve(),
            output_dir=eval_run / "patch_debug",
            patch_ids=[ctx.patch_id for ctx in contexts],
            merge_candidates_csv=merge_csv,
            trajectory_metadata=trajectory_metadata,
            nucleus_centers=nucleus_centers,
            source_eval_run_dir=eval_run,
        )
        pipeline["patch_debug_manifest"] = str(patch_debug_manifest)
        update_evaluation_summary_with_patch_debug(
            summary_path=eval_run / "summary.json",
            manifest_path=patch_debug_manifest,
        )
        with (out_dir / "patch_eval_pipeline_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(pipeline, handle, indent=2)
            handle.write("\n")
    if args.eval_run_path_file is not None:
        eval_run_path_file = Path(str(args.eval_run_path_file)).expanduser().resolve()
        eval_run_path_file.parent.mkdir(parents=True, exist_ok=True)
        eval_run_path_file.write_text(f"{eval_run}\n", encoding="utf-8")
    print(f"Patch assignments: {assignments_csv}")
    print(f"Patch assignments by patch: {patch_assignments_csv}")
    print(f"Patch PPO-format evaluation: {eval_run}")
    if patch_debug_manifest is not None:
        print(f"Patch debug manifest: {patch_debug_manifest}")


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested cuda but CUDA is not available")
    return torch.device(device_name)


def _parse_rollout_devices(raw: str) -> list[str]:
    if raw in (None, ""):
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


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


def _collect_eval_trajectories_for_devices(
    *,
    contexts: list[PatchContext],
    model: torch.nn.Module,
    config: Any,
    device: torch.device,
    rollout_devices: list[torch.device],
    rollout_backend: str,
    rng: np.random.Generator,
    policy_mode: str,
) -> list[PatchTrajectory]:
    if len(rollout_devices) <= 1:
        trajectories, _ = collect_patch_trajectories_batched(
            contexts=contexts,
            model=model,
            device=device,
            rng=rng,
            policy_mode=policy_mode,
            rollout_backend=rollout_backend,
            capture_trace=True,
        )
        return trajectories

    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    seeds = rng.integers(low=0, high=np.iinfo(np.uint32).max, size=len(rollout_devices), dtype=np.uint64)
    assignments = [[] for _ in rollout_devices]
    for idx in range(len(contexts)):
        assignments[idx % len(rollout_devices)].append(idx)

    def run_on_device(device_index: int, indices: list[int]) -> list[tuple[int, PatchTrajectory]]:
        local_device = rollout_devices[device_index]
        if local_device.type == "cuda":
            torch.cuda.set_device(local_device)
        local_model = build_actor_critic_from_config(config, device=local_device)
        local_model.load_state_dict(state_dict, strict=True)
        local_model.eval()
        local_contexts = [contexts[idx] for idx in indices]
        local_rng = np.random.default_rng(int(seeds[device_index]))
        trajectories, _ = collect_patch_trajectories_batched(
            contexts=local_contexts,
            model=local_model,
            device=local_device,
            rng=local_rng,
            policy_mode=policy_mode,
            rollout_backend=rollout_backend,
            capture_trace=True,
        )
        return list(zip(indices, trajectories))

    pairs: list[tuple[int, PatchTrajectory]] = []
    with ThreadPoolExecutor(max_workers=len(rollout_devices)) as executor:
        futures = [
            executor.submit(run_on_device, device_index, indices)
            for device_index, indices in enumerate(assignments)
            if indices
        ]
        for future in futures:
            pairs.extend(future.result())
    return [traj for _, traj in sorted(pairs, key=lambda item: item[0])]


def _patch_debug_trajectory_metadata(
    *,
    context: PatchContext,
    trajectory: PatchTrajectory,
) -> dict[str, Any]:
    trace_available = trajectory.initial_masks is not None
    initial_owners = _owners_from_masks(context=context, masks=trajectory.initial_masks or {})
    final_owners = _owners_from_masks(context=context, masks=trajectory.final_masks)
    cumulative_reward = 0.0
    steps: list[dict[str, Any]] = []
    trace_barcodes = {row["barcode"] for row in initial_owners}
    trace_barcodes.update(row["barcode"] for row in final_owners)
    for step_index, step in enumerate(trajectory.steps, start=1):
        cumulative_reward += float(step.reward)
        actions = [
            {
                "type": str(event.action_type),
                "cell_id": str(event.cell_id),
                "old_cell_id": None if event.old_cell_id is None else str(event.old_cell_id),
                "barcode": str(event.barcode),
                "applied": bool(event.applied),
            }
            for event in step.action_events
        ]
        trace_barcodes.update(str(action["barcode"]) for action in actions)
        steps.append(
            {
                "step_index": int(step_index),
                "reward": float(step.reward),
                "cumulative_reward": float(cumulative_reward),
                "patch_score_after": step.patch_score_after,
                "raw_patch_score_after": step.raw_patch_score_after,
                "owned_target_count_after": step.owned_target_count_after,
                "target_count": step.target_count,
                "phase": step.phase,
                "outcome": step.outcome,
                "done": bool(step.done),
                "n_local_actions": int(step.n_local_actions),
                "n_noop_actions": int(step.n_noop_actions),
                "actions": actions,
            }
        )

    return {
        "patch_score": float(trajectory.patch_score),
        "total_reward": float(trajectory.total_reward),
        "n_steps": int(len(trajectory.steps)),
        "metrics": dict(trajectory.metrics),
        "trajectory": {
            "available": bool(trace_available),
            "initial_patch_score": trajectory.initial_patch_score,
            "initial_raw_patch_score": trajectory.initial_raw_patch_score,
            "initial_owned_target_count": trajectory.initial_owned_target_count,
            "target_count": trajectory.target_count,
            "initial_owners": initial_owners,
            "final_owners": final_owners,
            "steps": steps if trace_available else [],
            "bin_geometry": _trace_bin_geometry(context=context, barcodes=trace_barcodes),
        },
    }


def _owners_from_masks(
    *,
    context: PatchContext,
    masks: dict[str, np.ndarray],
) -> list[dict[str, str]]:
    owners: list[dict[str, str]] = []
    for cell in context.cells:
        mask = masks.get(str(cell.cell_id))
        if mask is None:
            continue
        for bin_idx in np.flatnonzero(np.asarray(mask, dtype=np.uint8) > 0).tolist():
            owners.append(
                {
                    "barcode": str(cell.candidate_bin_ids[int(bin_idx)]),
                    "cell_id": str(cell.cell_id),
                }
            )
    owners.sort(key=lambda row: (row["barcode"], row["cell_id"]))
    return owners


def _trace_bin_geometry(
    *,
    context: PatchContext,
    barcodes: set[str],
) -> dict[str, dict[str, float | int]]:
    geometry: dict[str, dict[str, float | int]] = {}
    for cell in context.cells:
        xy = np.asarray(cell.candidate_bin_xy_um, dtype=np.float64)
        for bin_idx, barcode_value in enumerate(cell.candidate_bin_ids):
            barcode = str(barcode_value)
            if barcode not in barcodes or barcode in geometry:
                continue
            row_col = _parse_square_barcode(barcode)
            geometry[barcode] = {
                "array_row": (
                    int(row_col[0]) if row_col is not None else int(round(float(xy[bin_idx, 1]) / 2.0))
                ),
                "array_col": (
                    int(row_col[1]) if row_col is not None else int(round(float(xy[bin_idx, 0]) / 2.0))
                ),
                "x_um": float(xy[bin_idx, 0]),
                "y_um": float(xy[bin_idx, 1]),
            }
    return geometry


def _select_patch_rows_for_target_cells(*, patches_df: pd.DataFrame, target_set: set[str]) -> list[Any]:
    selected_rows = []
    for row in patches_df.itertuples(index=False):
        core = set(str(x) for x in json.loads(getattr(row, "core_cell_ids")))
        if core & target_set:
            selected_rows.append(row)
    return selected_rows


def _select_random_patch_rows(*, patches_df: pd.DataFrame, max_eval_patches: int, rng: np.random.Generator) -> list[Any]:
    if patches_df.empty:
        raise RuntimeError("patches index is empty")
    n_rows = min(int(max_eval_patches), int(len(patches_df)))
    indices = rng.choice(len(patches_df), size=n_rows, replace=False)
    return list(patches_df.iloc[np.asarray(indices, dtype=np.int64)].itertuples(index=False))


def _target_ids_from_patch_rows(rows: list[Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for cell_id in json.loads(getattr(row, "core_cell_ids")):
            cell = str(cell_id)
            if cell not in seen:
                ordered.append(cell)
                seen.add(cell)
    if not ordered:
        raise RuntimeError("random patch sample has no core cells")
    return ordered


def _write_patch_sample_eval_source(
    *,
    source_eval_dir: Path,
    output_dir: Path,
    target_cell_ids: list[str],
    fallback_config_path: Path,
) -> Path:
    source_config = source_eval_dir / "config_used.yaml"
    if not source_config.exists():
        source_config = fallback_config_path
    if not source_config.exists():
        raise FileNotFoundError(f"PPO eval config_used.yaml not found and fallback base config is missing: {source_config}")
    sample_dir = output_dir / "ppo_eval_source_patch_sample"
    sample_dir.mkdir(parents=False, exist_ok=False)
    (sample_dir / "config_used.yaml").write_text(source_config.read_text(encoding="utf-8"), encoding="utf-8")
    pd.DataFrame({"cell_id": [str(cell_id) for cell_id in target_cell_ids]}).to_csv(
        sample_dir / "per_episode.csv",
        index=False,
    )
    return sample_dir


if __name__ == "__main__":
    main()
