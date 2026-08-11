#!/usr/bin/env python
"""Run local multi-cell greedy competition and PPO-format evaluation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hd_cell_rl.episode_artifacts import _build_nuclei_centers, _build_nuclei_spatial_index
from hd_cell_rl.multicell_greedy import (
    GreedyCellState,
    assignment_rows_for_target,
    run_multicell_greedy_patch,
)
from hd_cell_rl.ppo_config import load_ppo_training_config
from hd_cell_rl.ppo_dataset import EpisodeDataset, _load_table
from preprocessing.ppo_eval_metrics import normalize_cell_id
from preprocessing.ppo_format_assignment_eval import (
    add_ppo_format_assignment_eval_args,
    load_eval_cell_ids,
    run_ppo_format_assignment_evaluation,
    validate_ppo_format_assignment_eval_args,
)

logger = logging.getLogger(__name__)


def _now_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_episode_artifact_map(episodes_index_path: Path) -> dict[str, Path]:
    df = pd.read_csv(episodes_index_path, usecols=["cell_id", "artifact_path"])
    out: dict[str, Path] = {}
    for row in df.itertuples(index=False):
        cell_id = normalize_cell_id(getattr(row, "cell_id"))
        if cell_id is None or cell_id in out:
            continue
        out[cell_id] = Path(str(getattr(row, "artifact_path"))).expanduser().resolve()
    return out


def _write_eval_subset_dir(
    *,
    source_eval_dir: Path,
    subset_eval_dir: Path,
    target_cell_ids: list[str],
) -> Path:
    """Write a PPO-eval-like directory limited to the selected target cells."""
    subset_eval_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source_eval_dir / "config_used.yaml", subset_eval_dir / "config_used.yaml")
    per_episode = pd.read_csv(source_eval_dir / "per_episode.csv")
    keep = set(target_cell_ids)
    norm = per_episode["cell_id"].map(normalize_cell_id)
    filtered = per_episode.loc[norm.isin(keep)].copy()
    if filtered.empty:
        raise RuntimeError("PPO eval subset is empty after applying selected target cell IDs")
    filtered.to_csv(subset_eval_dir / "per_episode.csv", index=False)
    return subset_eval_dir


def _local_patch_cell_ids(
    *,
    target_cell_id: str,
    artifact_by_cell: dict[str, Path],
    spatial_index: Any,
    context_radius_um: float,
    max_patch_cells: int,
) -> list[str]:
    if target_cell_id not in spatial_index.cell_id_to_index:
        return [target_cell_id] if target_cell_id in artifact_by_cell else []

    own_idx = int(spatial_index.cell_id_to_index[target_cell_id])
    center = spatial_index.centers_xy_um[own_idx]
    nearby_idx = spatial_index.tree.query_ball_point(center, r=float(context_radius_um))
    idx_to_cell = {idx: cell_id for cell_id, idx in spatial_index.cell_id_to_index.items()}

    candidates: list[tuple[float, str]] = []
    for idx in nearby_idx:
        cell_id = idx_to_cell.get(int(idx))
        if cell_id is None or cell_id not in artifact_by_cell:
            continue
        delta = spatial_index.centers_xy_um[int(idx)] - center
        dist = float(np.sqrt(np.sum(delta * delta)))
        candidates.append((dist, cell_id))

    candidates.sort(key=lambda item: (item[0], item[1]))
    ordered = [target_cell_id]
    seen = {target_cell_id}
    for _, cell_id in candidates:
        if cell_id in seen:
            continue
        ordered.append(cell_id)
        seen.add(cell_id)
        if max_patch_cells > 0 and len(ordered) >= int(max_patch_cells):
            break
    return ordered


def _load_patch_states(
    *,
    dataset: EpisodeDataset,
    artifact_by_cell: dict[str, Path],
    patch_cell_ids: list[str],
    max_steps_per_episode: int | None,
) -> list[GreedyCellState]:
    states: list[GreedyCellState] = []
    for cell_id in patch_cell_ids:
        artifact_path = artifact_by_cell.get(cell_id)
        if artifact_path is None:
            continue
        ctx = dataset.load_episode_context(
            cell_id=cell_id,
            artifact_path=artifact_path,
            max_steps_per_episode=max_steps_per_episode,
            include_candidate_bin_ids=True,
        )
        if ctx is None or ctx.n_bins <= 0:
            continue
        if int(np.sum(ctx.initial_membership_mask)) <= 0:
            continue
        states.append(GreedyCellState.from_context(ctx))
    return states


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="workspace_outputs/pseudo_human_colorectal/multicell_greedy",
        help="Directory root for generated multi-cell greedy assignments.",
    )
    parser.add_argument(
        "--context_radius_um",
        type=float,
        default=50.0,
        help="Nucleus-center radius used to add local competitor cells around each target.",
    )
    parser.add_argument(
        "--max_patch_cells",
        type=int,
        default=24,
        help="Maximum cells in one local competition patch, including the target. Use 0 for no cap.",
    )
    parser.add_argument(
        "--max_steps_per_patch",
        type=int,
        default=1000,
        help="Maximum greedy ADD steps per local patch.",
    )
    parser.add_argument(
        "--min_add_score",
        type=float,
        default=0.0,
        help="Stop when the best legal ADD score is <= this value.",
    )
    parser.add_argument(
        "--max_target_cells",
        type=int,
        default=0,
        help="Optional cap on source PPO-eval target cells. Use 0 for all cells in per_episode.csv.",
    )
    parser.add_argument(
        "--external_nuclear_bins_path",
        type=str,
        default=None,
        help="Optional nuclear-bin source path recorded in the PPO-format summary.",
    )
    add_ppo_format_assignment_eval_args(
        parser,
        default_eval_run_name="human_colorectal_multicell_greedy_eval",
        method_label="multi-cell greedy",
    )
    args = parser.parse_args()
    if args.ppo_eval_run_dir is None:
        raise ValueError("--ppo_eval_run_dir is required because it defines the target cell set and config")
    if float(args.context_radius_um) <= 0:
        raise ValueError("--context_radius_um must be > 0")
    if int(args.max_patch_cells) < 0:
        raise ValueError("--max_patch_cells must be >= 0")
    if int(args.max_steps_per_patch) < 0:
        raise ValueError("--max_steps_per_patch must be >= 0")
    if int(args.max_target_cells) < 0:
        raise ValueError("--max_target_cells must be >= 0")
    validate_ppo_format_assignment_eval_args(args)
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = _parse_args()

    ppo_eval_run_dir = Path(str(args.ppo_eval_run_dir)).expanduser().resolve()
    per_episode_csv = ppo_eval_run_dir / "per_episode.csv"
    eval_config_path = ppo_eval_run_dir / "config_used.yaml"
    if not per_episode_csv.exists():
        raise FileNotFoundError(f"PPO eval per_episode.csv not found: {per_episode_csv}")
    if not eval_config_path.exists():
        raise FileNotFoundError(f"PPO eval config_used.yaml not found: {eval_config_path}")

    config = load_ppo_training_config(eval_config_path)
    target_cell_ids = load_eval_cell_ids(per_episode_csv)
    if int(args.max_target_cells) > 0:
        target_cell_ids = target_cell_ids[: int(args.max_target_cells)]
    if not target_cell_ids:
        raise RuntimeError(f"no target cell IDs found in {per_episode_csv}")

    out_dir = Path(str(args.output_dir)).expanduser().resolve() / f"multicell_greedy_{_now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)
    assignments_csv = out_dir / "multicell_greedy_assignments.csv"
    patch_summary_csv = out_dir / "multicell_greedy_patch_summary.csv"

    logger.info("Source PPO eval run: %s", ppo_eval_run_dir)
    logger.info("Target cells: %d", len(target_cell_ids))
    logger.info("Output directory: %s", out_dir)
    logger.info("Context radius: %.3f um", float(args.context_radius_um))
    logger.info("Max patch cells: %d", int(args.max_patch_cells))
    logger.info("Max steps per patch: %d", int(args.max_steps_per_patch))

    artifact_by_cell = _load_episode_artifact_map(config.episodes_index_path)
    nuclei_df = _load_table(config.nuclei_path, config.nuclei_format)
    centers = _build_nuclei_centers(df=nuclei_df, columns=config.nuclei_columns)
    spatial_index = _build_nuclei_spatial_index(centers)
    rng = np.random.default_rng(config.seed)
    dataset = EpisodeDataset(config=config, rng=rng)

    assignment_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    try:
        for target_idx, target_cell_id in enumerate(target_cell_ids, start=1):
            if target_cell_id not in artifact_by_cell:
                patch_rows.append(
                    {
                        "target_cell_id": target_cell_id,
                        "status": "missing_episode_artifact",
                        "n_patch_cells": 0,
                        "n_steps": 0,
                    }
                )
                continue

            patch_cell_ids = _local_patch_cell_ids(
                target_cell_id=target_cell_id,
                artifact_by_cell=artifact_by_cell,
                spatial_index=spatial_index,
                context_radius_um=float(args.context_radius_um),
                max_patch_cells=int(args.max_patch_cells),
            )
            states = _load_patch_states(
                dataset=dataset,
                artifact_by_cell=artifact_by_cell,
                patch_cell_ids=patch_cell_ids,
                max_steps_per_episode=config.max_steps_per_episode,
            )
            state_by_cell = {state.cell_id: state for state in states}
            target_state = state_by_cell.get(target_cell_id)
            if target_state is None:
                patch_rows.append(
                    {
                        "target_cell_id": target_cell_id,
                        "status": "missing_target_context",
                        "n_patch_cells": len(states),
                        "n_steps": 0,
                    }
                )
                continue

            result = run_multicell_greedy_patch(
                target_cell_id=target_cell_id,
                cell_states=states,
                max_steps=int(args.max_steps_per_patch),
                min_add_score=float(args.min_add_score),
            )
            assignment_rows.extend(assignment_rows_for_target(result=result, target_state=target_state))

            target_wins = sum(1 for step in result.steps if step.cell_id == target_cell_id)
            target_assigned = int(np.sum(target_state.membership_mask))
            target_nuclear = int(np.sum(target_state.context.initial_membership_mask))
            patch_rows.append(
                {
                    "target_cell_id": target_cell_id,
                    "status": "ok",
                    "n_patch_cells": int(result.n_patch_cells),
                    "n_competitor_cells": max(0, int(result.n_patch_cells) - 1),
                    "n_steps": int(len(result.steps)),
                    "n_target_greedy_steps": int(target_wins),
                    "n_non_target_greedy_steps": int(len(result.steps) - target_wins),
                    "n_target_assigned_bins": target_assigned,
                    "n_target_nuclear_bins": target_nuclear,
                    "n_target_added_bins": max(0, target_assigned - target_nuclear),
                    "n_contested_candidate_barcodes": int(result.n_contested_candidate_barcodes),
                    "n_blocked_frontier_actions": int(result.n_blocked_frontier_actions),
                    "n_initial_owner_conflicts": int(result.n_initial_owner_conflicts),
                    "stop_reason": result.stop_reason,
                }
            )
            if target_idx % 25 == 0 or target_idx == len(target_cell_ids):
                logger.info("Processed %d/%d target cells", target_idx, len(target_cell_ids))
    finally:
        dataset.close()

    assignments_df = pd.DataFrame(assignment_rows)
    if assignments_df.empty:
        raise RuntimeError("multi-cell greedy produced no assignment rows")
    assignments_df.to_csv(assignments_csv, index=False)
    patch_df = pd.DataFrame(patch_rows)
    patch_df.to_csv(patch_summary_csv, index=False)

    pipeline_summary = {
        "source_ppo_eval_run_dir": str(ppo_eval_run_dir),
        "source_ppo_eval_per_episode": str(per_episode_csv),
        "source_config": str(eval_config_path),
        "assignments_csv": str(assignments_csv),
        "patch_summary_csv": str(patch_summary_csv),
        "context_radius_um": float(args.context_radius_um),
        "max_patch_cells": int(args.max_patch_cells),
        "max_steps_per_patch": int(args.max_steps_per_patch),
        "min_add_score": float(args.min_add_score),
        "n_target_cells_requested": int(len(target_cell_ids)),
        "n_assignment_rows": int(len(assignments_df)),
        "n_ok_patches": int((patch_df.get("status", pd.Series(dtype=str)) == "ok").sum()),
    }
    if not patch_df.empty and "status" in patch_df.columns:
        ok_df = patch_df.loc[patch_df["status"] == "ok"].copy()
        pipeline_summary["status_counts"] = {
            str(k): int(v) for k, v in patch_df["status"].value_counts(dropna=False).items()
        }
        for col in (
            "n_patch_cells",
            "n_steps",
            "n_target_added_bins",
            "n_contested_candidate_barcodes",
            "n_blocked_frontier_actions",
        ):
            if col in ok_df.columns and len(ok_df) > 0:
                values = pd.to_numeric(ok_df[col], errors="coerce")
                pipeline_summary[f"mean_{col}"] = float(values.mean())
                pipeline_summary[f"median_{col}"] = float(values.median())

    eval_args = args
    if int(args.max_target_cells) > 0:
        subset_eval_dir = _write_eval_subset_dir(
            source_eval_dir=ppo_eval_run_dir,
            subset_eval_dir=out_dir / "source_ppo_eval_subset",
            target_cell_ids=target_cell_ids,
        )
        eval_args = argparse.Namespace(**vars(args))
        eval_args.ppo_eval_run_dir = str(subset_eval_dir)
        pipeline_summary["source_ppo_eval_subset_dir"] = str(subset_eval_dir)

    with (out_dir / "multicell_greedy_pipeline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(pipeline_summary, handle, indent=2)
        handle.write("\n")

    eval_run_dir = run_ppo_format_assignment_evaluation(
        assignments_csv=assignments_csv,
        method_name="multicell_greedy",
        method_label="Multi-cell greedy",
        nuclear_source="ppo_episode_seed",
        external_nuclear_bins_path=(
            None
            if args.external_nuclear_bins_path is None
            else Path(str(args.external_nuclear_bins_path)).expanduser().resolve()
        ),
        args=eval_args,
        pipeline_config=pipeline_summary,
    )
    logger.info("Multi-cell greedy assignments: %s", assignments_csv)
    logger.info("Multi-cell greedy patch summary: %s", patch_summary_csv)
    logger.info("PPO-format evaluation run: %s", eval_run_dir)


if __name__ == "__main__":
    main()
