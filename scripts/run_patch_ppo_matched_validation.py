#!/usr/bin/env python
"""Compare EM, exact-greedy w5, and a frozen PPO on validation patches.

This is a diagnostic runner, not a training entry point.  Patch selection and
the exact-greedy result are fixed inputs.  Ground truth is never exposed to the
policy; it is loaded only to score the already selected actions and owners.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hd_cell_rl.patch_types import PatchContext, PatchTrajectory
from hd_cell_rl.ppo_checkpoint import build_actor_critic_from_config, load_checkpoint_payload
from hd_cell_rl.ppo_config import load_ppo_training_config
from scripts.evaluate_patch_checkpoint import _collect_eval_trajectories_for_devices
from scripts.run_patch_objective_greedy_ablation import (
    PatchTruth,
    _cell_iou_rows,
    _mean_iou,
    _owner_accuracy,
    _settings_from_patch_config,
    _transition_outcome,
)
from scripts.run_patch_reward_formula_rollout import _load_context_worker, _load_truths


METHOD_ORDER = ("em_only", "w5_exact_greedy", "frozen_ppo_greedy")
METHOD_LABELS = {
    "em_only": "EM only",
    "w5_exact_greedy": "w5 exact greedy",
    "frozen_ppo_greedy": "Frozen PPO greedy",
}
COLORS = {
    "em_only": "#6B7280",
    "w5_exact_greedy": "#3B82F6",
    "frozen_ppo_greedy": "#D97706",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selected-patches-index", required=True)
    parser.add_argument("--exact-greedy-dir", required=True)
    parser.add_argument("--gt-cell-bins-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--expected-patches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if int(args.expected_patches) <= 0:
        raise ValueError("--expected-patches must be > 0")
    return args


def _owners_from_masks(
    *,
    context: PatchContext | Any,
    masks: dict[str, np.ndarray] | None,
) -> dict[str, str]:
    if masks is None:
        raise ValueError("trajectory did not capture initial ownership masks")
    owners: dict[str, str] = {}
    for cell in context.cells:
        cell_id = str(cell.cell_id)
        mask = masks.get(cell_id)
        if mask is None:
            continue
        indices = np.flatnonzero(np.asarray(mask, dtype=np.uint8) > 0)
        for index in indices.tolist():
            barcode = str(cell.candidate_bin_ids[int(index)])
            previous = owners.get(barcode)
            if previous is not None and previous != cell_id:
                raise RuntimeError(
                    f"physical barcode {barcode} has duplicate owners: {previous}, {cell_id}"
                )
            owners[barcode] = cell_id
    return owners


def _ppo_trajectory_tables(
    *,
    context: PatchContext | Any,
    truth: PatchTruth,
    trajectory: PatchTrajectory,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if str(trajectory.patch_id) != str(context.patch_id):
        raise RuntimeError(
            f"trajectory/context mismatch: {trajectory.patch_id} != {context.patch_id}"
        )
    initial_owners = _owners_from_masks(context=context, masks=trajectory.initial_masks)
    expected_final_owners = _owners_from_masks(context=context, masks=trajectory.final_masks)
    current_owners = dict(initial_owners)
    initial_iou = _mean_iou(initial_owners, truth)
    action_rows: list[dict[str, Any]] = []

    for step_index, step in enumerate(trajectory.steps, start=1):
        for event_index, event in enumerate(step.action_events, start=1):
            if not bool(event.applied):
                continue
            barcode = str(event.barcode)
            old_owner = current_owners.get(barcode)
            recorded_old = None if event.old_cell_id is None else str(event.old_cell_id)
            if str(event.action_type) == "replace" and recorded_old != old_owner:
                raise RuntimeError(
                    f"owner trace mismatch in {context.patch_id}/{barcode}: "
                    f"recorded={recorded_old}, tracked={old_owner}"
                )
            new_owner = str(event.cell_id)
            iou_before = _mean_iou(current_owners, truth)
            current_owners[barcode] = new_owner
            iou_after = _mean_iou(current_owners, truth)
            components = dict(step.objective_delta_components or {})
            action_rows.append(
                {
                    "split": "validation",
                    "method": "frozen_ppo_greedy",
                    "patch_id": str(context.patch_id),
                    "action_index": int(len(action_rows) + 1),
                    "step_index": int(step_index),
                    "event_index": int(event_index),
                    "action_type": str(event.action_type),
                    "barcode": barcode,
                    "old_cell_id": old_owner,
                    "new_cell_id": new_owner,
                    "gt_owner_cell_id": truth.gt_owner_by_barcode.get(barcode),
                    "outcome": _transition_outcome(
                        barcode=barcode,
                        old_owner=old_owner,
                        new_owner=new_owner,
                        truth=truth,
                    ),
                    "reward": float(step.reward),
                    "macro_iou_before": float(iou_before),
                    "macro_iou_after": float(iou_after),
                    "macro_iou_delta": float(iou_after - iou_before),
                    "available_action_count": step.available_action_count,
                    "chosen_action_reward": step.chosen_action_reward,
                    "best_available_action_reward": step.best_available_action_reward,
                    "chosen_action_reward_rank": step.chosen_action_reward_rank,
                    "objective_regret": (
                        np.nan
                        if step.chosen_action_reward is None
                        or step.best_available_action_reward is None
                        else float(step.best_available_action_reward - step.chosen_action_reward)
                    ),
                    "patch_score_after": step.patch_score_after,
                    **{
                        f"delta_{key}": float(value)
                        for key, value in components.items()
                    },
                }
            )

    if current_owners != expected_final_owners:
        raise RuntimeError(f"incremental owner tracking diverged in {context.patch_id}")

    changed = sorted(
        barcode
        for barcode in set(initial_owners) | set(expected_final_owners)
        if initial_owners.get(barcode) != expected_final_owners.get(barcode)
    )
    net_outcomes = [
        _transition_outcome(
            barcode=barcode,
            old_owner=initial_owners.get(barcode),
            new_owner=expected_final_owners.get(barcode),
            truth=truth,
        )
        for barcode in changed
    ]
    initial_correct, initial_evaluated, initial_accuracy = _owner_accuracy(initial_owners, truth)
    final_correct, final_evaluated, final_accuracy = _owner_accuracy(expected_final_owners, truth)
    final_iou = _mean_iou(expected_final_owners, truth)
    n_replace = sum(row["action_type"] == "replace" for row in action_rows)
    patch_row = {
        "split": "validation",
        "method": "frozen_ppo_greedy",
        "patch_id": str(context.patch_id),
        "initial_macro_cell_iou": float(initial_iou),
        "final_macro_cell_iou": float(final_iou),
        "macro_cell_iou_delta": float(final_iou - initial_iou),
        "initial_owner_correct": int(initial_correct),
        "initial_owner_evaluated": int(initial_evaluated),
        "initial_owner_accuracy": float(initial_accuracy),
        "final_owner_correct": int(final_correct),
        "final_owner_evaluated": int(final_evaluated),
        "final_owner_accuracy": float(final_accuracy),
        "n_steps": int(len(trajectory.steps)),
        "n_actions": int(len(action_rows)),
        "n_replace_actions": int(n_replace),
        "n_changed_bins": int(len(changed)),
        "n_net_corrected": int(net_outcomes.count("corrected")),
        "n_net_damaged": int(net_outcomes.count("damaged")),
        "n_net_no_gt": int(net_outcomes.count("no_gt")),
        "n_net_wrong_to_wrong": int(net_outcomes.count("wrong_to_wrong")),
        "n_net_unresolved": int(net_outcomes.count("unresolved_owner_mapping")),
        "total_reward": float(trajectory.total_reward),
        "initial_patch_score": trajectory.initial_patch_score,
        "final_patch_score": float(trajectory.patch_score),
        "termination_reason": (
            "max_steps"
            if len(trajectory.steps) >= int(context.max_steps)
            else "no_positive_action"
        ),
    }
    cell_rows = _cell_iou_rows(initial_owners, truth, stage="em_initial")
    cell_rows.extend(_cell_iou_rows(expected_final_owners, truth, stage="final"))
    for row in cell_rows:
        row.update({"split": "validation", "method": "frozen_ppo_greedy"})
    return patch_row, action_rows, cell_rows


def _assert_em_alignment(
    *,
    ppo_cells: pd.DataFrame,
    exact_cells: pd.DataFrame,
    atol: float = 1.0e-8,
) -> None:
    keys = ["patch_id", "cell_id"]
    measures = ["predicted_bins", "gt_bins", "intersection", "union", "iou"]
    left = ppo_cells.loc[ppo_cells.stage == "em_initial", keys + measures].copy()
    right = exact_cells.loc[exact_cells.stage == "em_initial", keys + measures].copy()
    for frame in (left, right):
        frame["patch_id"] = frame["patch_id"].astype(str)
        frame["cell_id"] = frame["cell_id"].astype(str)
    merged = left.merge(right, on=keys, how="outer", suffixes=("_ppo", "_exact"), indicator=True)
    if not bool(np.all(merged["_merge"] == "both")):
        raise RuntimeError("PPO and exact-greedy EM cell sets do not align")
    for measure in measures:
        a = pd.to_numeric(merged[f"{measure}_ppo"], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(merged[f"{measure}_exact"], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=0.0, atol=float(atol), equal_nan=True):
            maximum = float(np.nanmax(np.abs(a - b)))
            raise RuntimeError(
                f"PPO and exact-greedy EM initialization differ for {measure}; max diff={maximum}"
            )


def _make_comparison_tables(
    *,
    ppo_patch: pd.DataFrame,
    ppo_actions: pd.DataFrame,
    ppo_cells: pd.DataFrame,
    exact_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exact_patch_all = pd.read_csv(exact_dir / "per_patch.csv")
    exact_actions_all = pd.read_parquet(exact_dir / "actions.parquet")
    exact_cells_all = pd.read_csv(exact_dir / "per_cell.csv")
    exact_patch = exact_patch_all.loc[exact_patch_all.formula.astype(str) == "w5"].copy()
    exact_actions = exact_actions_all.loc[exact_actions_all.formula.astype(str) == "w5"].copy()
    exact_cells = exact_cells_all.loc[exact_cells_all.formula.astype(str) == "w5"].copy()
    expected_ids = set(ppo_patch.patch_id.astype(str))
    if set(exact_patch.patch_id.astype(str)) != expected_ids:
        raise RuntimeError("PPO and exact-greedy patch IDs do not match")
    _assert_em_alignment(ppo_cells=ppo_cells, exact_cells=exact_cells)

    exact_patch_out = exact_patch.copy()
    exact_patch_out["method"] = "w5_exact_greedy"
    exact_patch_out["n_replace_actions"] = exact_patch_out["n_actions"].astype(int)
    exact_patch_out["n_steps"] = exact_patch_out["n_actions"].astype(int)
    exact_patch_out["total_reward"] = exact_patch_out["formula_reward_sum"]

    exact_actions_out = exact_actions.rename(columns={"formula_reward": "reward"}).copy()
    exact_actions_out["method"] = "w5_exact_greedy"
    exact_actions_out["action_index"] = exact_actions_out["step_index"].astype(int)
    exact_actions_out["action_type"] = "replace"
    exact_actions_out["objective_regret"] = 0.0
    exact_actions_out["chosen_action_reward_rank"] = 1

    exact_cells_out = exact_cells.copy()
    exact_cells_out["method"] = "w5_exact_greedy"
    exact_cells_out["stage"] = exact_cells_out["stage"].replace(
        {"formula_greedy": "final"}
    )

    em_patch = exact_patch_out.copy()
    em_patch["method"] = "em_only"
    em_patch["final_macro_cell_iou"] = em_patch["initial_macro_cell_iou"]
    em_patch["macro_cell_iou_delta"] = 0.0
    em_patch["final_owner_correct"] = em_patch["initial_owner_correct"]
    em_patch["final_owner_evaluated"] = em_patch["initial_owner_evaluated"]
    em_patch["final_owner_accuracy"] = em_patch["initial_owner_accuracy"]
    for column in (
        "n_steps",
        "n_actions",
        "n_replace_actions",
        "n_changed_bins",
        "n_net_corrected",
        "n_net_damaged",
        "n_net_no_gt",
        "n_net_wrong_to_wrong",
        "n_net_unresolved",
        "total_reward",
    ):
        em_patch[column] = 0
    em_patch["termination_reason"] = "not_applicable"

    em_cells = exact_cells_out.loc[exact_cells_out.stage == "em_initial"].copy()
    em_cells["method"] = "em_only"
    em_final = em_cells.copy()
    em_final["stage"] = "final"
    em_cells = pd.concat([em_cells, em_final], ignore_index=True)

    patch_df = pd.concat([em_patch, exact_patch_out, ppo_patch], ignore_index=True, sort=False)
    action_df = pd.concat([exact_actions_out, ppo_actions], ignore_index=True, sort=False)
    cell_df = pd.concat([em_cells, exact_cells_out, ppo_cells], ignore_index=True, sort=False)
    order = {name: index for index, name in enumerate(METHOD_ORDER)}
    patch_df["method_order"] = patch_df.method.map(order)
    action_df["method_order"] = action_df.method.map(order)
    cell_df["method_order"] = cell_df.method.map(order)
    patch_df = patch_df.sort_values(["method_order", "patch_id"]).reset_index(drop=True)
    action_df = action_df.sort_values(["method_order", "patch_id", "action_index"]).reset_index(drop=True)
    cell_df = cell_df.sort_values(["method_order", "patch_id", "stage", "cell_id"]).reset_index(drop=True)
    return patch_df, action_df, cell_df


def _summary_table(
    *,
    patch_df: pd.DataFrame,
    action_df: pd.DataFrame,
    cell_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        patches = patch_df.loc[patch_df.method == method]
        actions = action_df.loc[action_df.method == method]
        cells = cell_df.loc[cell_df.method == method]
        initial = cells.loc[cells.stage == "em_initial", "iou"].astype(float)
        final = cells.loc[cells.stage == "final", "iou"].astype(float)
        deltas = patches.macro_cell_iou_delta.astype(float)
        outcomes = actions.outcome.value_counts().to_dict() if not actions.empty else {}
        n_replace = int(np.sum(actions.action_type.astype(str) == "replace")) if not actions.empty else 0
        corrected = int(outcomes.get("corrected", 0))
        damaged = int(outcomes.get("damaged", 0))
        ranks = pd.to_numeric(actions.get("chosen_action_reward_rank"), errors="coerce")
        regrets = pd.to_numeric(actions.get("objective_regret"), errors="coerce")
        row = {
            "split": "validation",
            "method": method,
            "n_patches": int(len(patches)),
            "n_cells": int(len(initial)),
            "pooled_em_initial_macro_cell_iou": float(initial.mean()),
            "pooled_final_macro_cell_iou": float(final.mean()),
            "pooled_macro_cell_iou_delta": float(final.mean() - initial.mean()),
            "mean_patch_initial_macro_cell_iou": float(patches.initial_macro_cell_iou.mean()),
            "mean_patch_final_macro_cell_iou": float(patches.final_macro_cell_iou.mean()),
            "mean_patch_macro_cell_iou_delta": float(deltas.mean()),
            "median_patch_macro_cell_iou_delta": float(deltas.median()),
            "fraction_patches_improved": float(np.mean(deltas > 1.0e-12)),
            "fraction_patches_damaged": float(np.mean(deltas < -1.0e-12)),
            "fraction_patches_unchanged": float(np.mean(np.abs(deltas) <= 1.0e-12)),
            "n_actions": int(len(actions)),
            "n_replace_actions": n_replace,
            "n_corrected_actions": corrected,
            "n_damaged_actions": damaged,
            "n_wrong_to_wrong_actions": int(outcomes.get("wrong_to_wrong", 0)),
            "n_no_gt_actions": int(outcomes.get("no_gt", 0)),
            "n_net_corrected": int(patches.n_net_corrected.sum()),
            "n_net_damaged": int(patches.n_net_damaged.sum()),
            "n_net_no_gt": int(patches.n_net_no_gt.sum()),
            "action_net_gain_per_100_replace": (
                np.nan if n_replace == 0 else float(100.0 * (corrected - damaged) / n_replace)
            ),
            "final_owner_net_gain_per_100_replace": (
                np.nan
                if n_replace == 0
                else float(
                    100.0
                    * (patches.n_net_corrected.sum() - patches.n_net_damaged.sum())
                    / n_replace
                )
            ),
            "mean_chosen_objective_rank": (
                np.nan if ranks is None or not bool(ranks.notna().any()) else float(ranks.mean())
            ),
            "fraction_objective_rank1": (
                np.nan
                if ranks is None or not bool(ranks.notna().any())
                else float(np.mean(ranks.dropna() == 1))
            ),
            "mean_objective_regret": (
                np.nan if regrets is None or not bool(regrets.notna().any()) else float(regrets.mean())
            ),
            "n_max_step_terminations": int(np.sum(patches.termination_reason == "max_steps")),
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    exact_final = float(
        summary.loc[summary.method == "w5_exact_greedy", "pooled_final_macro_cell_iou"].iloc[0]
    )
    summary["exact_greedy_pooled_iou_potential_gap"] = (
        exact_final - summary.pooled_final_macro_cell_iou
    )
    summary["passes_positive_pooled_iou"] = summary.pooled_macro_cell_iou_delta > 0.0
    summary["passes_positive_median_patch_iou"] = summary.median_patch_macro_cell_iou_delta > 0.0
    summary["passes_majority_patch_improvement"] = summary.fraction_patches_improved > 0.5
    summary["passes_net_correction"] = summary.n_net_corrected > summary.n_net_damaged
    summary["adds_validation_value"] = (
        summary.passes_positive_pooled_iou
        & summary.passes_positive_median_patch_iou
        & summary.passes_majority_patch_improvement
        & summary.passes_net_correction
    )
    return summary


def _action_outcome_table(action_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in ("w5_exact_greedy", "frozen_ppo_greedy"):
        actions = action_df.loc[action_df.method == method].copy()
        total = max(1, len(actions))
        correlation = float(actions[["reward", "macro_iou_delta"]].corr().iloc[0, 1])
        for outcome, group in actions.groupby("outcome", sort=False):
            ranks = pd.to_numeric(group.get("chosen_action_reward_rank"), errors="coerce")
            regrets = pd.to_numeric(group.get("objective_regret"), errors="coerce")
            rows.append(
                {
                    "split": "validation",
                    "method": method,
                    "outcome": str(outcome),
                    "n_actions": int(len(group)),
                    "fraction_actions": float(len(group) / total),
                    "macro_iou_delta_sum": float(group.macro_iou_delta.sum()),
                    "macro_iou_delta_mean": float(group.macro_iou_delta.mean()),
                    "macro_iou_delta_median": float(group.macro_iou_delta.median()),
                    "reward_mean": float(group.reward.mean()),
                    "reward_iou_pearson_all_actions": correlation,
                    "mean_chosen_objective_rank": (
                        np.nan if not bool(ranks.notna().any()) else float(ranks.mean())
                    ),
                    "mean_objective_regret": (
                        np.nan if not bool(regrets.notna().any()) else float(regrets.mean())
                    ),
                }
            )
    return pd.DataFrame(rows)


def _paired_patch_table(patch_df: pd.DataFrame) -> pd.DataFrame:
    initial = (
        patch_df.loc[patch_df.method == "em_only", ["patch_id", "initial_macro_cell_iou"]]
        .drop_duplicates("patch_id")
        .set_index("patch_id")
    )
    delta = patch_df.pivot(index="patch_id", columns="method", values="macro_cell_iou_delta")
    final = patch_df.pivot(index="patch_id", columns="method", values="final_macro_cell_iou")
    actions = patch_df.pivot(index="patch_id", columns="method", values="n_replace_actions")
    out = initial.copy()
    out["w5_exact_greedy_final_iou"] = final["w5_exact_greedy"]
    out["frozen_ppo_greedy_final_iou"] = final["frozen_ppo_greedy"]
    out["w5_exact_greedy_iou_delta"] = delta["w5_exact_greedy"]
    out["frozen_ppo_greedy_iou_delta"] = delta["frozen_ppo_greedy"]
    out["ppo_minus_exact_iou_delta"] = (
        delta["frozen_ppo_greedy"] - delta["w5_exact_greedy"]
    )
    out["ppo_better_than_exact"] = (
        delta["frozen_ppo_greedy"] > delta["w5_exact_greedy"]
    )
    out["w5_exact_greedy_replace"] = actions["w5_exact_greedy"].astype(int)
    out["frozen_ppo_greedy_replace"] = actions["frozen_ppo_greedy"].astype(int)
    return out.reset_index().sort_values("patch_id").reset_index(drop=True)


def _evolution_table(
    *,
    patch_df: pd.DataFrame,
    action_df: pd.DataFrame,
    n_points: int = 101,
) -> pd.DataFrame:
    grid = np.linspace(0.0, 1.0, int(n_points))
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        curves: list[np.ndarray] = []
        for patch in patch_df.loc[patch_df.method == method].itertuples(index=False):
            actions = action_df.loc[
                (action_df.method == method) & (action_df.patch_id == str(patch.patch_id))
            ].sort_values("action_index")
            values = [float(patch.initial_macro_cell_iou)]
            values.extend(actions.macro_iou_after.astype(float).tolist())
            if len(values) == 1:
                progress = np.asarray([0.0, 1.0])
                curve = np.asarray([values[0], values[0]], dtype=float)
            else:
                progress = np.linspace(0.0, 1.0, len(values))
                curve = np.asarray(values, dtype=float)
            curves.append(np.interp(grid, progress, curve))
        matrix = np.stack(curves, axis=0)
        sem = np.std(matrix, axis=0, ddof=1) / np.sqrt(matrix.shape[0])
        for index, progress in enumerate(grid):
            rows.append(
                {
                    "split": "validation",
                    "method": method,
                    "rollout_progress": float(progress),
                    "mean_patch_macro_cell_iou": float(np.mean(matrix[:, index])),
                    "sem_patch_macro_cell_iou": float(sem[index]),
                    "n_patches": int(matrix.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def _plot_comparison(
    *,
    patch_df: pd.DataFrame,
    action_df: pd.DataFrame,
    summary: pd.DataFrame,
    evolution: pd.DataFrame,
    output_dir: Path,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.5), constrained_layout=True)
    rng = np.random.default_rng(17)

    for method_index, method in enumerate(METHOD_ORDER):
        values = patch_df.loc[patch_df.method == method, "macro_cell_iou_delta"].to_numpy(float)
        jitter = rng.uniform(-0.075, 0.075, size=len(values))
        axes[0, 0].scatter(
            np.full(len(values), method_index) + jitter,
            values,
            s=16,
            alpha=0.8,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.35,
        )
        axes[0, 0].plot(
            [method_index - 0.16, method_index + 0.16],
            [np.mean(values), np.mean(values)],
            color="black",
            linewidth=1.3,
        )
    axes[0, 0].axhline(0.0, color="#9CA3AF", linewidth=0.8, linestyle="--")
    axes[0, 0].set_xticks(range(len(METHOD_ORDER)), [METHOD_LABELS[x] for x in METHOD_ORDER], rotation=14, ha="right")
    axes[0, 0].set_ylabel("Patch macro IoU change")
    axes[0, 0].set_title("a  Validation effect by patch", loc="left", fontweight="bold")

    for method in METHOD_ORDER:
        group = evolution.loc[evolution.method == method]
        x = group.rollout_progress.to_numpy(float)
        y = group.mean_patch_macro_cell_iou.to_numpy(float)
        sem = group.sem_patch_macro_cell_iou.to_numpy(float)
        axes[0, 1].plot(x, y, color=COLORS[method], linewidth=1.5, label=METHOD_LABELS[method])
        if method != "em_only":
            axes[0, 1].fill_between(x, y - sem, y + sem, color=COLORS[method], alpha=0.13, linewidth=0)
    axes[0, 1].set_xlabel("Normalized rollout progress")
    axes[0, 1].set_ylabel("Mean patch macro IoU")
    axes[0, 1].set_title("b  IoU evolution (mean ± SEM)", loc="left", fontweight="bold")
    axes[0, 1].legend(fontsize=6)

    active = summary.loc[summary.method != "em_only"].copy()
    x = np.arange(len(active))
    width = 0.34
    denom = active.n_replace_actions.replace(0, np.nan).astype(float)
    corrected_rate = 100.0 * active.n_corrected_actions / denom
    damaged_rate = 100.0 * active.n_damaged_actions / denom
    axes[1, 0].bar(x - width / 2, corrected_rate, width, color="#2A9D8F", label="Corrected")
    axes[1, 0].bar(x + width / 2, damaged_rate, width, color="#C85C5C", label="Damaged")
    axes[1, 0].set_xticks(x, [METHOD_LABELS[value] for value in active.method], rotation=14, ha="right")
    axes[1, 0].set_ylabel("Actions per 100 REPLACE")
    axes[1, 0].set_title("c  Action outcome efficiency", loc="left", fontweight="bold")
    axes[1, 0].legend(fontsize=6)
    for xpos, count in zip(x, active.n_replace_actions):
        axes[1, 0].text(xpos, 0.98, f"n={int(count)}", transform=axes[1, 0].get_xaxis_transform(), ha="center", va="top", fontsize=6)

    paired = patch_df.pivot(index="patch_id", columns="method", values="macro_cell_iou_delta")
    axes[1, 1].scatter(
        paired["w5_exact_greedy"],
        paired["frozen_ppo_greedy"],
        s=21,
        color=COLORS["frozen_ppo_greedy"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
    )
    limits = np.asarray(
        [
            np.nanmin(paired[["w5_exact_greedy", "frozen_ppo_greedy"]].to_numpy()),
            np.nanmax(paired[["w5_exact_greedy", "frozen_ppo_greedy"]].to_numpy()),
        ]
    )
    padding = max(0.002, float((limits[1] - limits[0]) * 0.08))
    limits = limits + np.asarray([-padding, padding])
    axes[1, 1].plot(limits, limits, color="#9CA3AF", linestyle="--", linewidth=0.8)
    axes[1, 1].axhline(0.0, color="#D1D5DB", linewidth=0.6)
    axes[1, 1].axvline(0.0, color="#D1D5DB", linewidth=0.6)
    axes[1, 1].set_xlim(limits)
    axes[1, 1].set_ylim(limits)
    axes[1, 1].set_xlabel("w5 exact-greedy IoU change")
    axes[1, 1].set_ylabel("Frozen-PPO IoU change")
    axes[1, 1].set_title("d  PPO vs objective potential", loc="left", fontweight="bold")

    stem = output_dir / "matched_validation_comparison"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _report_text(
    summary: pd.DataFrame,
    *,
    action_df: pd.DataFrame,
    checkpoint: Path,
    exact_dir: Path,
) -> str:
    ppo = summary.loc[summary.method == "frozen_ppo_greedy"].iloc[0]
    exact = summary.loc[summary.method == "w5_exact_greedy"].iloc[0]
    decision = bool(ppo.adds_validation_value)
    ppo_actions = action_df.loc[action_df.method == "frozen_ppo_greedy"]
    exact_actions = action_df.loc[action_df.method == "w5_exact_greedy"]
    ppo_outcomes = ppo_actions.outcome.value_counts().to_dict()
    exact_reward_iou_r = float(exact_actions[["reward", "macro_iou_delta"]].corr().iloc[0, 1])
    ppo_reward_iou_r = float(ppo_actions[["reward", "macro_iou_delta"]].corr().iloc[0, 1])
    ppo_no_gt_fraction = float(np.mean(ppo_actions.outcome == "no_gt"))
    ppo_corrected_mean = float(ppo_actions.loc[ppo_actions.outcome == "corrected", "macro_iou_delta"].mean())
    ppo_damaged_mean = float(ppo_actions.loc[ppo_actions.outcome == "damaged", "macro_iou_delta"].mean())
    lines = [
        "# Frozen PPO matched validation diagnostic",
        "",
        (
            "The frozen PPO adds reproducible value over EM on this validation set under the "
            "predeclared gates."
            if decision
            else "The frozen PPO does not clear the predeclared validation gates over EM."
        ),
        "",
        "All three methods use the same 20 validation patches and the same EM hard initialization. Ground truth is used only after each action has been selected. No held-out test patch was loaded, and no model was trained.",
        "",
        "## Main comparison",
        "",
        "| Method | Pooled cell IoU | Δ vs EM | Median patch Δ | Patches improved | REPLACE | Corrected | Damaged | Net / 100 REPLACE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        row = summary.loc[summary.method == method].iloc[0]
        net = row.action_net_gain_per_100_replace
        lines.append(
            f"| {METHOD_LABELS[method]} | {row.pooled_final_macro_cell_iou:.6f} | "
            f"{row.pooled_macro_cell_iou_delta:+.6f} | {row.median_patch_macro_cell_iou_delta:+.6f} | "
            f"{row.fraction_patches_improved:.1%} | {int(row.n_replace_actions)} | "
            f"{int(row.n_corrected_actions)} | {int(row.n_damaged_actions)} | "
            f"{'NA' if pd.isna(net) else f'{net:+.2f}'} |"
        )
    lines.extend(
        [
            "",
            "## PPO versus exact-greedy potential",
            "",
            f"The exact w5 rollout ends at pooled IoU {exact.pooled_final_macro_cell_iou:.6f}; the frozen PPO ends at {ppo.pooled_final_macro_cell_iou:.6f}. The remaining pooled-IoU gap to exact greedy is {ppo.exact_greedy_pooled_iou_potential_gap:+.6f}.",
            f"Across selected PPO actions, the mean objective rank is {ppo.mean_chosen_objective_rank:.3f}, rank 1 is selected {ppo.fraction_objective_rank1:.1%} of the time, and mean immediate-objective regret is {ppo.mean_objective_regret:.6g}.",
            "",
            "## What the experiment identifies",
            "",
            f"1. **The w5 objective is not sufficiently aligned with GT IoU.** Exact maximization of the same objective lowers pooled IoU by {exact.pooled_macro_cell_iou_delta:+.6f}; its immediate reward–IoU Pearson correlation is only {exact_reward_iou_r:.3f}.",
            f"2. **The PPO also does not closely reproduce w5 greedy selection.** Its mean action rank is {ppo.mean_chosen_objective_rank:.2f}, rank 1 frequency is {ppo.fraction_objective_rank1:.1%}, and it finishes {ppo.exact_greedy_pooled_iou_potential_gap:.6f} pooled IoU below exact greedy.",
            f"3. **Action counts alone conceal unequal damage.** PPO has {int(ppo_outcomes.get('corrected', 0))} corrected and {int(ppo_outcomes.get('damaged', 0))} damaged actions, but their mean IoU changes are {ppo_corrected_mean:+.6f} and {ppo_damaged_mean:+.6f}; reward–IoU correlation is {ppo_reward_iou_r:.3f}.",
            f"4. **Coverage and horizon remain secondary concerns.** {ppo_no_gt_fraction:.1%} of PPO REPLACE actions have no GT owner label and {int(ppo.n_max_step_terminations)}/{int(ppo.n_patches)} patches reach 16 steps. Extending the horizon alone is not justified because the complete exact-greedy rollout is also worse than EM.",
            "",
            "The next experiment should therefore improve or gate the objective before training another PPO: use GT only on validation to fit a damage-risk/action-acceptance diagnostic, especially for no-GT and high-magnitude damaging moves, while keeping inference GT-free.",
            "",
            "## Decision gates",
            "",
            f"- Positive pooled IoU change: {bool(ppo.passes_positive_pooled_iou)}",
            f"- Positive median per-patch IoU change: {bool(ppo.passes_positive_median_patch_iou)}",
            f"- More than half of patches improve: {bool(ppo.passes_majority_patch_improvement)}",
            f"- Net corrected owners exceed net damaged owners: {bool(ppo.passes_net_correction)}",
            f"- Overall validation decision: {'PASS' if decision else 'FAIL'}",
            "",
            "## Provenance and scope",
            "",
            f"- Frozen checkpoint: `{checkpoint}`",
            f"- Fixed exact-greedy source: `{exact_dir}`",
            "- The checkpoint came from one seed and four overfit training patches. This diagnostic measures transfer to validation, not test generalization.",
            "- `summary.csv` contains pooled and per-patch metrics; `action_outcome_diagnostics.csv` and `ppo_vs_exact_per_patch.csv` isolate the failure modes; `per_patch.csv`, `actions.parquet`, `per_cell.csv`, and `iou_evolution.csv` contain the supporting records.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    selected_index = Path(args.selected_patches_index).expanduser().resolve()
    exact_dir = Path(args.exact_greedy_dir).expanduser().resolve()
    gt_path = Path(args.gt_cell_bins_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    for path in (checkpoint, selected_index, exact_dir, gt_path):
        if not path.exists():
            raise FileNotFoundError(path)

    selected = pd.read_csv(selected_index)
    if "audit_split" not in selected.columns:
        raise ValueError("selected patch index must contain audit_split")
    selected = selected.loc[selected.audit_split.astype(str) == "validation"].copy()
    if len(selected) != int(args.expected_patches):
        raise RuntimeError(
            f"expected exactly {args.expected_patches} validation patches, found {len(selected)}"
        )
    if bool(selected.patch_id.astype(str).duplicated().any()):
        raise RuntimeError("validation patch IDs must be unique")
    exact_selected = pd.read_csv(exact_dir / "selected_patches.csv")
    exact_validation_ids = set(
        exact_selected.loc[
            exact_selected.audit_split.astype(str) == "validation", "patch_id"
        ].astype(str)
    )
    if exact_validation_ids != set(selected.patch_id.astype(str)):
        raise RuntimeError("selected validation patches differ from the exact-greedy source")

    payload = load_checkpoint_payload(checkpoint)
    patch_cfg = dict(payload.get("patch_config") or {})
    if not patch_cfg:
        raise ValueError("checkpoint is missing patch_config")
    base_config_path = Path(str(patch_cfg["base_ppo_config"])).expanduser().resolve()
    config = replace(load_ppo_training_config(base_config_path), planner_enabled=False)
    patch_section = dict(patch_cfg.get("patch_training", {}))
    max_steps = int(patch_section.get("max_steps_per_patch", 16))
    settings = _settings_from_patch_config(patch_cfg, max_steps=max_steps)
    settings = replace(
        settings,
        patches_index_path=selected_index,
        cache_patch_contexts=False,
        max_steps_per_patch=max_steps,
    )
    if not settings.em_assignment.enabled:
        raise ValueError("matched validation requires em_assignment.enabled=true")
    if settings.agent_mode != "multi_cell_global_delta":
        raise ValueError("matched validation requires multi_cell_global_delta")
    if settings.after_fill_actions != "replace_only":
        raise ValueError("matched validation requires after_fill_actions=replace_only")

    device_names = tuple(item.strip() for item in str(args.devices).split(",") if item.strip())
    if not device_names:
        raise ValueError("--devices must contain at least one device")
    devices = [torch.device(item) for item in device_names]
    if any(device.type == "cuda" for device in devices) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    primary_device = devices[0]
    model = build_actor_critic_from_config(config, device=primary_device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    records = selected.to_dict(orient="records")
    record_groups: list[list[dict[str, Any]]] = [[] for _ in devices]
    for index, record in enumerate(records):
        record_groups[index % len(devices)].append(record)
    contexts: list[PatchContext] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(
                _load_context_worker,
                rows=rows,
                base_config=config,
                settings=settings,
                seed=int(args.seed) + index,
            )
            for index, rows in enumerate(record_groups)
            if rows
        ]
        for future in futures:
            contexts.extend(future.result())
    if len(contexts) != len(records):
        raise RuntimeError(f"loaded {len(contexts)} of {len(records)} validation contexts")
    expected_order = [str(row["patch_id"]) for row in records]
    order = {patch_id: index for index, patch_id in enumerate(expected_order)}
    contexts.sort(key=lambda context: order[str(context.patch_id)])
    if [str(context.patch_id) for context in contexts] != expected_order:
        raise RuntimeError("loaded context IDs do not match selected patch IDs")
    print(f"Loaded {len(contexts)} validation contexts; loading evaluation-only GT.", flush=True)
    truths = _load_truths(gt_path=gt_path, contexts=contexts)

    print(
        f"Running frozen greedy PPO on {len(contexts)} patches with {', '.join(device_names)}.",
        flush=True,
    )
    trajectories = _collect_eval_trajectories_for_devices(
        contexts=contexts,
        model=model,
        config=config,
        device=primary_device,
        rollout_devices=devices,
        rollout_backend=settings.rollout_backend,
        rng=np.random.default_rng(int(args.seed)),
        policy_mode="greedy",
    )
    if len(trajectories) != len(contexts):
        raise RuntimeError("trajectory count does not match context count")

    patch_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for index, (context, trajectory) in enumerate(zip(contexts, trajectories), start=1):
        patch_row, actions, cells = _ppo_trajectory_tables(
            context=context,
            truth=truths[context.patch_id],
            trajectory=trajectory,
        )
        patch_rows.append(patch_row)
        action_rows.extend(actions)
        cell_rows.extend(cells)
        print(
            f"Scored PPO patch {index}/{len(contexts)}: {context.patch_id}; "
            f"REPLACE={patch_row['n_replace_actions']}, IoU Δ={patch_row['macro_cell_iou_delta']:+.6f}",
            flush=True,
        )
    ppo_patch = pd.DataFrame(patch_rows)
    ppo_actions = pd.DataFrame(action_rows)
    ppo_cells = pd.DataFrame(cell_rows)
    patch_df, action_df, cell_df = _make_comparison_tables(
        ppo_patch=ppo_patch,
        ppo_actions=ppo_actions,
        ppo_cells=ppo_cells,
        exact_dir=exact_dir,
    )
    summary = _summary_table(patch_df=patch_df, action_df=action_df, cell_df=cell_df)
    evolution = _evolution_table(patch_df=patch_df, action_df=action_df)
    action_outcomes = _action_outcome_table(action_df)
    paired_patches = _paired_patch_table(patch_df)

    run_dir = output_root / f"matched_validation_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    selected.to_csv(run_dir / "selected_patches.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    patch_df.drop(columns=["method_order"]).to_csv(run_dir / "per_patch.csv", index=False)
    action_df.drop(columns=["method_order"]).to_parquet(run_dir / "actions.parquet", index=False)
    cell_df.drop(columns=["method_order"]).to_csv(run_dir / "per_cell.csv", index=False)
    evolution.to_csv(run_dir / "iou_evolution.csv", index=False)
    action_outcomes.to_csv(run_dir / "action_outcome_diagnostics.csv", index=False)
    paired_patches.to_csv(run_dir / "ppo_vs_exact_per_patch.csv", index=False)
    config_payload = {
        "scope": "validation_only",
        "ground_truth_usage": "evaluation_only_after_action_selection",
        "training_performed": False,
        "held_out_test_loaded": False,
        "checkpoint": str(checkpoint),
        "checkpoint_update_index": payload.get("update_index"),
        "checkpoint_best_moving_average_reward": payload.get("best_moving_avg_reward"),
        "base_config": str(base_config_path),
        "selected_patches_index": str(selected_index),
        "exact_greedy_dir": str(exact_dir),
        "gt_cell_bins_path": str(gt_path),
        "devices": device_names,
        "seed": int(args.seed),
        "max_steps_per_patch": max_steps,
        "policy_mode": "greedy",
        "em_alignment_check": "exact per-patch/core-cell predicted_bins, gt_bins, intersection, union, and IoU",
        "methods": list(METHOD_ORDER),
        "decision_rule": (
            "PPO must have positive pooled IoU delta, positive median patch delta, "
            ">50% patches improved, and net corrected > net damaged"
        ),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2) + "\n", encoding="utf-8"
    )
    _plot_comparison(
        patch_df=patch_df,
        action_df=action_df,
        summary=summary,
        evolution=evolution,
        output_dir=run_dir,
    )
    (run_dir / "report.md").write_text(
        _report_text(
            summary,
            action_df=action_df,
            checkpoint=checkpoint,
            exact_dir=exact_dir,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Matched validation complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
