#!/usr/bin/env python
"""Evaluate nucleus, distance-only, or generalized-EM patch initialization."""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hd_cell_rl.patch_training import (
    EMAssignmentConfig,
    PatchDataset,
    PatchTrainingSettings,
    _build_patch_env,
    patch_assignments_for_core_cells,
)
from hd_cell_rl.ppo_checkpoint import load_checkpoint_payload
from hd_cell_rl.patch_assignment import PatchOwnershipMerger
from hd_cell_rl.ppo_config import load_ppo_training_config
from preprocessing.ppo_format_assignment_eval import (
    add_ppo_format_assignment_eval_args,
    load_eval_cell_ids,
    run_ppo_format_assignment_evaluation,
    validate_ppo_format_assignment_eval_args,
)


_METHODS = {
    "nucleus_only": ("hd_cell_rl_nucleus_only", "HD-Cell-RL nucleus-only initialization"),
    "distance_only": ("hd_cell_rl_distance_only", "HD-Cell-RL distance-only soft assignment"),
    "em_only": ("hd_cell_rl_em_only", "HD-Cell-RL generalized EM-only"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint supplying the frozen patch/EM protocol.")
    parser.add_argument("--base-config", required=True, help="Resolved PPO config pointing at the evaluation data.")
    parser.add_argument(
        "--episodes-index-path-override",
        default=None,
        help="Optional episode-artifact index for a matched evaluation dataset.",
    )
    parser.add_argument(
        "--nuclei-path-override",
        default=None,
        help="Optional nuclei table paired with the episode index override.",
    )
    parser.add_argument(
        "--reference-npz-path-override",
        default=None,
        help="Optional donor-specific reference-count NPZ for the evaluation dataset.",
    )
    parser.add_argument("--patches-index-path", required=True)
    parser.add_argument("--mode", required=True, choices=tuple(_METHODS))
    parser.add_argument("--output-dir", required=True, help="Parent for versioned source and EM artifacts.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--em-spatial-weight", type=float, default=None)
    parser.add_argument("--em-expression-weight", type=float, default=None)
    parser.add_argument(
        "--em-non-nuclear-bin-filter",
        choices=("all", "positive_expression"),
        default=None,
    )
    parser.add_argument(
        "--em-background-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--em-background-owned-logit-intercept", type=float, default=None)
    parser.add_argument("--em-background-confidence-weight", type=float, default=None)
    parser.add_argument("--external_nuclear_bins_path", type=str, default=None)
    add_ppo_format_assignment_eval_args(
        parser,
        default_eval_run_name="colorectal_nucleus_based_patch_initialization",
        method_label="patch initialization",
    )
    args = parser.parse_args()
    if args.ppo_eval_run_dir is None:
        raise ValueError("--ppo_eval_run_dir is required")
    validate_ppo_format_assignment_eval_args(args)
    return args


def _settings_from_checkpoint(
    *,
    patch_cfg: dict[str, Any],
    patches_index_path: Path,
    mode: str,
    artifact_dir: Path,
    em_spatial_weight: float | None = None,
    em_expression_weight: float | None = None,
    em_non_nuclear_bin_filter: str | None = None,
    em_background_enabled: bool | None = None,
    em_background_owned_logit_intercept: float | None = None,
    em_background_confidence_weight: float | None = None,
) -> PatchTrainingSettings:
    patch_training = dict(patch_cfg.get("patch_training", {}) or {})
    em_config = EMAssignmentConfig.from_mapping(patch_cfg.get("em_assignment", {}))
    if mode == "nucleus_only":
        em_config = replace(
            em_config,
            enabled=False,
            save_artifacts=False,
            debug_patch_ids=(),
            artifact_dir=None,
        )
    else:
        if not em_config.enabled:
            raise ValueError("distance_only and em_only require an EM-enabled checkpoint protocol")
        em_config = replace(
            em_config,
            expression_weight=(0.0 if mode == "distance_only" else float(em_config.expression_weight)),
            save_artifacts=True,
            debug_patch_ids=(),
            artifact_dir=artifact_dir,
        )
        if em_spatial_weight is not None:
            em_config = replace(em_config, spatial_weight=float(em_spatial_weight))
        if em_expression_weight is not None:
            em_config = replace(em_config, expression_weight=float(em_expression_weight))
        if em_non_nuclear_bin_filter is not None:
            em_config = replace(
                em_config,
                non_nuclear_bin_filter=str(em_non_nuclear_bin_filter),
            )
        if em_background_enabled is not None:
            em_config = replace(
                em_config,
                background_enabled=bool(em_background_enabled),
            )
        if em_background_owned_logit_intercept is not None:
            em_config = replace(
                em_config,
                background_owned_logit_intercept=float(
                    em_background_owned_logit_intercept
                ),
            )
        if em_background_confidence_weight is not None:
            em_config = replace(
                em_config,
                background_confidence_weight=float(
                    em_background_confidence_weight
                ),
            )
        em_config.validate()
    return PatchTrainingSettings(
        patches_index_path=patches_index_path,
        batch_patches=1,
        max_steps_per_patch=int(patch_training.get("max_steps_per_patch", 1000)),
        margin_cells_compete=bool(patch_training.get("margin_cells_compete", True)),
        use_core_cells_for_score=bool(patch_training.get("use_core_cells_for_score", True)),
        score_normalization=str(patch_training.get("score_normalization", "mean_core_cells")),
        rollout_backend=str(patch_training.get("rollout_backend", "torch_gpu")),
        reward_backend=str(patch_training.get("reward_backend", "standard")),
        stcs_reward_config=dict(patch_training.get("stcs_reward", {}) or {}),
        cache_patch_contexts=False,
        competition_margin_enabled=bool(patch_training.get("competition_margin_enabled", True)),
        force_fill_expression_bins=bool(patch_training.get("force_fill_expression_bins", False)),
        fill_target=str(patch_training.get("fill_target", "reachable_expression_bins")),
        stop_action_mode=str(patch_training.get("stop_action_mode", "enabled")),
        agent_mode=str(patch_training.get("agent_mode", "multi_cell_global_delta")),
        after_fill_actions=str(patch_training.get("after_fill_actions", "replace_only")),
        global_delta_epsilon=float(patch_training.get("global_delta_epsilon", 1.0e-6)),
        em_assignment=em_config,
    )


def _select_patch_rows(patches_df: pd.DataFrame, target_ids: set[str]) -> list[Any]:
    selected: list[Any] = []
    for row in patches_df.itertuples(index=False):
        core_ids = {str(item) for item in json.loads(str(getattr(row, "core_cell_ids")))}
        if core_ids.intersection(target_ids):
            selected.append(row)
    return selected


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested cuda but CUDA is unavailable")
    return torch.device(name)


def _em_patch_summary(context: Any, *, initial_score: float, mode: str) -> dict[str, Any]:
    result = context.em_assignment
    row: dict[str, Any] = {
        "patch_id": str(context.patch_id),
        "mode": str(mode),
        "n_core_cells": int(len(context.core_cell_ids)),
        "n_margin_cells": int(len(context.margin_cell_ids)),
        "initial_global_objective": float(initial_score),
    }
    if result is None:
        return row
    counts = np.diff(np.asarray(result.candidate_row_splits, dtype=np.int64))
    row.update(
        {
            "candidate_max_distance_um": float(result.candidate_max_distance_um),
            "n_unique_bins": int(len(result.barcode_ids)),
            "n_candidate_pairs": int(len(result.candidate_cell_index)),
            "mean_candidates_per_bin": float(np.mean(counts)),
            "max_candidates_per_bin": int(np.max(counts)),
            "n_nuclear_locked": int(np.sum(result.is_nuclear_locked)),
            "n_unassigned": int(np.sum(result.is_unassigned)),
            "assignment_coverage": float(1.0 - np.mean(result.is_unassigned)),
            "mean_background_probability": float(
                np.mean(result.background_probability)
            ),
            "mean_normalized_entropy": float(np.mean(result.normalized_entropy)),
            "fraction_ambiguous": float(np.mean(result.is_ambiguous)),
            "iterations": int(len(result.iterations)),
            "converged": bool(result.converged),
            "n_disconnected_hard_islands": int(result.n_disconnected_hard_islands),
        }
    )
    return row


def main() -> None:
    args = _parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    base_config_path = Path(args.base_config).expanduser().resolve()
    patches_index_path = Path(args.patches_index_path).expanduser().resolve()
    source_eval_dir = Path(args.ppo_eval_run_dir).expanduser().resolve()
    payload = load_checkpoint_payload(checkpoint)
    patch_cfg = dict(payload.get("patch_config") or {})
    config = replace(load_ppo_training_config(base_config_path), planner_enabled=False)
    if args.episodes_index_path_override is not None:
        episodes_override = Path(
            args.episodes_index_path_override
        ).expanduser().resolve()
        if not episodes_override.is_file():
            raise FileNotFoundError(episodes_override)
        config = replace(config, episodes_index_path=episodes_override)
    if args.nuclei_path_override is not None:
        nuclei_override = Path(args.nuclei_path_override).expanduser().resolve()
        if not nuclei_override.is_file():
            raise FileNotFoundError(nuclei_override)
        config = replace(config, nuclei_path=nuclei_override)
    if args.reference_npz_path_override is not None:
        reference_override = Path(
            args.reference_npz_path_override
        ).expanduser().resolve()
        if not reference_override.is_file():
            raise FileNotFoundError(reference_override)
        config = replace(
            config,
            reference_path=reference_override,
            reference_format="npz",
        )
    target_ids = load_eval_cell_ids(source_eval_dir / "per_episode.csv")
    target_set = set(target_ids)
    patches_df = pd.read_csv(patches_index_path)
    selected_rows = _select_patch_rows(patches_df, target_set)
    if not selected_rows:
        raise RuntimeError("no patches contain the requested evaluation cells")

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_dir).expanduser().resolve() / f"{args.mode}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    source_dir = run_dir / "source"
    diagnostics_dir = run_dir / "diagnostics"
    em_dir = run_dir / "em_assignment"
    source_dir.mkdir()
    diagnostics_dir.mkdir()
    settings = _settings_from_checkpoint(
        patch_cfg=patch_cfg,
        patches_index_path=patches_index_path,
        mode=str(args.mode),
        artifact_dir=em_dir,
        em_spatial_weight=args.em_spatial_weight,
        em_expression_weight=args.em_expression_weight,
        em_non_nuclear_bin_filter=args.em_non_nuclear_bin_filter,
        em_background_enabled=args.em_background_enabled,
        em_background_owned_logit_intercept=(
            args.em_background_owned_logit_intercept
        ),
        em_background_confidence_weight=args.em_background_confidence_weight,
    )

    rng = np.random.default_rng(int(args.eval_seed))
    device = _resolve_device(str(args.device))
    dataset = PatchDataset(base_config=config, settings=settings, rng=rng)
    best_rows_by_cell: dict[str, tuple[float, list[dict[str, Any]]]] = {}
    ownership_merger = PatchOwnershipMerger()
    all_patch_rows: list[dict[str, Any]] = []
    patch_summaries: list[dict[str, Any]] = []
    try:
        for patch_index, patch_row in enumerate(selected_rows, start=1):
            context = dataset.load_patch_context(patch_row)
            if context is None:
                continue
            env = _build_patch_env(
                context=context,
                device=device,
                rollout_backend=str(settings.rollout_backend),
            )
            env.reset()
            final_masks = env.final_masks()
            initial_score = float(env.patch_score())
            rows_by_cell = patch_assignments_for_core_cells(
                context=context,
                final_masks=final_masks,
            )
            for cell_id, rows in rows_by_cell.items():
                if cell_id not in target_set:
                    continue
                for row in rows:
                    row["assignment_source"] = f"patch_{args.mode}"
                all_patch_rows.extend(rows)
            ownership_merger.add_patch(
                context=context, score=0.0,
                target_cell_ids=[c for c in rows_by_cell if c in target_set],
                rows=[r for cell, rr in rows_by_cell.items() if cell in target_set for r in rr],
            )
            patch_summaries.append(
                _em_patch_summary(context, initial_score=initial_score, mode=str(args.mode))
            )
            print(f"Initialized {patch_index}/{len(selected_rows)} patches ({context.patch_id})", flush=True)
            del env
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        dataset.close()

    assignment_rows, merge_conflicts = ownership_merger.finalize()
    best_rows_by_cell = ownership_merger.best_rows_by_cell
    merge_dir = run_dir / 'subdiagnostic'
    merge_dir.mkdir()
    pd.DataFrame(merge_conflicts, columns=['barcode', 'candidate_cell_id', 'source_patch_id',
        'source_patch_score', 'nuclear_owner', 'final_owner', 'rule']).to_csv(merge_dir / 'merge_conflicts.csv', index=False)
    pd.DataFrame(ownership_merger.patch_choices).to_csv(merge_dir / 'patch_choices.csv', index=False)
    if not assignment_rows:
        raise RuntimeError("patch initialization produced no assignments")
    assignments_csv = source_dir / "assignments.csv"
    patch_assignments_csv = source_dir / "patch_assignments.csv"
    pd.DataFrame(assignment_rows).to_csv(assignments_csv, index=False)
    pd.DataFrame(all_patch_rows).to_csv(patch_assignments_csv, index=False)
    pd.DataFrame(patch_summaries).to_csv(diagnostics_dir / "patch_summary.csv", index=False)

    method_name, method_label = _METHODS[str(args.mode)]
    pipeline = {
        "mode": str(args.mode),
        "method_name": method_name,
        "checkpoint_protocol": str(checkpoint),
        "base_config": str(base_config_path),
        "episodes_index_path_used": str(config.episodes_index_path),
        "nuclei_path_used": str(config.nuclei_path),
        "reference_path_used": str(config.reference_path),
        "patches_index_path": str(patches_index_path),
        "source_eval_dir": str(source_eval_dir),
        "assignments_csv": str(assignments_csv),
        "patch_assignments_csv": str(patch_assignments_csv),
        "n_target_cells": int(len(target_ids)),
        "n_selected_patches": int(len(selected_rows)),
        "n_evaluated_cells": int(len(best_rows_by_cell)),
        "device": str(device),
        "em": {
            "enabled": bool(settings.em_assignment.enabled),
            "spatial_weight": float(settings.em_assignment.spatial_weight),
            "expression_weight": float(settings.em_assignment.expression_weight),
            "damping": float(settings.em_assignment.damping),
            "max_iterations": int(settings.em_assignment.max_iterations),
            "mean_profile_total_variation": float(
                settings.em_assignment.mean_profile_total_variation
            ),
            "entropy_threshold": float(settings.em_assignment.entropy_threshold),
            "non_nuclear_bin_filter": str(
                settings.em_assignment.non_nuclear_bin_filter
            ),
            "background_enabled": bool(settings.em_assignment.background_enabled),
            "background_owned_logit_intercept": float(
                settings.em_assignment.background_owned_logit_intercept
            ),
            "background_confidence_weight": float(
                settings.em_assignment.background_confidence_weight
            ),
            "cell_specific_expression_enabled": bool(
                settings.em_assignment.cell_specific_expression_enabled
            ),
            "cell_profile_relative_prior_strength": float(
                settings.em_assignment.cell_profile_relative_prior_strength
            ),
            "cell_profile_update_damping": float(
                settings.em_assignment.cell_profile_update_damping
            ),
            "cell_profile_compatibility_mode": str(
                settings.em_assignment.cell_profile_compatibility_mode
            ),
            "spatial_crossfit_block_size_um": float(
                settings.em_assignment.spatial_crossfit_block_size_um
            ),
            "cell_profile_reliability_mode": str(
                settings.em_assignment.cell_profile_reliability_mode
            ),
            "heldout_reliability_grid_size": int(
                settings.em_assignment.heldout_reliability_grid_size
            ),
        },
    }
    with (run_dir / "pipeline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(pipeline, handle, indent=2)
        handle.write("\n")

    evaluation_context_dir = run_dir / "evaluation_context"
    evaluation_context_dir.mkdir()
    shutil.copy2(
        source_eval_dir / "per_episode.csv",
        evaluation_context_dir / "per_episode.csv",
    )
    with (evaluation_context_dir / "config_used.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(config.to_serializable_dict(), handle, sort_keys=False)
    args.ppo_eval_run_dir = str(evaluation_context_dir)

    eval_run = run_ppo_format_assignment_evaluation(
        assignments_csv=assignments_csv,
        method_name=method_name,
        method_label=method_label,
        nuclear_source="dominant_confident_nuclear_seed",
        external_nuclear_bins_path=(
            None
            if args.external_nuclear_bins_path is None
            else Path(args.external_nuclear_bins_path).expanduser().resolve()
        ),
        args=args,
        pipeline_config=pipeline,
    )
    print(f"Initialization source: {run_dir}")
    print(f"PPO-format evaluation: {eval_run}")


if __name__ == "__main__":
    main()
