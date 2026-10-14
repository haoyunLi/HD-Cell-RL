#!/usr/bin/env python
"""Run fixed-formula sequential greedy REPLACE diagnostics on patch splits.

This runner is evaluation-only. Every trajectory starts from the same EM hard
ownership, enumerates all structurally legal REPLACE actions, and applies the
highest formula score while it exceeds the configured epsilon. Ground truth is
loaded separately and is used only after action selection for outcome and IoU
evaluation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import datetime as dt
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Iterable

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hd_cell_rl.patch_dataset import PatchDataset
from hd_cell_rl.patch_env_torch import TorchSingleAgentPatchEnv
from hd_cell_rl.patch_types import PatchContext
from hd_cell_rl.ppo_config import load_ppo_training_config
from preprocessing.ppo_eval_metrics import normalize_cell_id
from scripts.patch_formula_checkpoint import (
    load_formula_checkpoint,
    remove_formula_checkpoint,
    save_formula_checkpoint,
)
from scripts.patch_formula_delta_diagnostics import cached_replace_component_deltas
from scripts.patch_formula_exact_shape import exact_shape_add_corrections
from scripts.run_patch_objective_greedy_ablation import (
    OBJECTIVE_COMPONENTS,
    GreedyPatchResult,
    PatchTruth,
    _cell_iou_rows,
    _mean_iou,
    _owner_accuracy,
    _owners_from_env,
    _settings_from_patch_config,
    _transition_outcome,
)


_FORMULA_NAMES = ("baseline", "w5", "symmetric")
_COLORS = {
    "baseline": "#6B7280",
    "w5": "#3B82F6",
    "symmetric": "#D97706",
}


@dataclass(frozen=True)
class FormulaSpec:
    name: str
    expression_mode: str
    w1: float
    w2: float
    w3: float
    w4: float
    w5: float
    shape_weight: float
    symmetric_weight: float
    epsilon: float
    calibration_shape_lambda: float | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-config", required=True)
    parser.add_argument("--selected-patches-index", required=True)
    parser.add_argument("--gt-cell-bins-path", required=True)
    parser.add_argument("--reward-audit-dir", required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--formulas", default="baseline,w5,symmetric")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--patch-ids", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--expression-weight", type=float, default=0.15)
    parser.add_argument("--global-delta-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--max-steps", type=int, default=10000)
    args = parser.parse_args()
    if not np.isfinite(args.expression_weight) or float(args.expression_weight) <= 0.0:
        raise ValueError("--expression-weight must be finite and > 0")
    if not np.isfinite(args.global_delta_epsilon) or float(args.global_delta_epsilon) < 0.0:
        raise ValueError("--global-delta-epsilon must be finite and >= 0")
    if int(args.max_steps) <= 0:
        raise ValueError("--max-steps must be > 0")
    return args


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def _parse_formula_names(raw: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in str(raw).split(",") if item.strip())
    if not names:
        raise ValueError("--formulas must contain at least one formula")
    unknown = sorted(set(names) - set(_FORMULA_NAMES))
    if unknown:
        raise ValueError(f"unknown formulas: {unknown}; choose from {_FORMULA_NAMES}")
    if len(set(names)) != len(names):
        raise ValueError("--formulas contains duplicates")
    return names


def _calibrated_shape_weight(
    *,
    expression_weight: float,
    shape_lambda: float,
    expression_iqr: float,
    shape_iqr: float,
) -> float:
    values = (expression_weight, shape_lambda, expression_iqr, shape_iqr)
    if any(not np.isfinite(value) for value in values):
        raise ValueError("calibration values must be finite")
    if expression_weight <= 0.0 or shape_lambda < 0.0 or expression_iqr <= 0.0 or shape_iqr <= 0.0:
        raise ValueError("invalid calibration scales")
    return float(expression_weight * shape_lambda * expression_iqr / shape_iqr)


def _selected_combo_row(audit_dir: Path, expression_signal: str) -> pd.Series:
    path = audit_dir / "expression_shape_combinations.csv"
    frame = pd.read_csv(path)
    selected_flag = frame["selected_within_family_on_validation"].astype(str).str.lower().eq("true")
    selected = frame.loc[
        (frame["expression_signal"] == expression_signal)
        & (frame["audit_split"] == "validation")
        & selected_flag
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"expected one validation-selected row for {expression_signal!r} in {path}"
        )
    return selected.iloc[0]


def _build_formula_specs(
    *,
    base_config: Any,
    audit_dir: Path,
    expression_weight: float,
    epsilon: float,
) -> dict[str, FormulaSpec]:
    w5_row = _selected_combo_row(audit_dir, "existing_w5_raw")
    symmetric_row = _selected_combo_row(audit_dir, "symmetric_compatibility_raw")
    w5_shape = _calibrated_shape_weight(
        expression_weight=expression_weight,
        shape_lambda=float(w5_row["shape_lambda_standardized"]),
        expression_iqr=float(w5_row["expression_iqr"]),
        shape_iqr=float(w5_row["shape_iqr"]),
    )
    symmetric_shape = _calibrated_shape_weight(
        expression_weight=expression_weight,
        shape_lambda=float(symmetric_row["shape_lambda_standardized"]),
        expression_iqr=float(symmetric_row["expression_iqr"]),
        shape_iqr=float(symmetric_row["shape_iqr"]),
    )
    return {
        "baseline": FormulaSpec(
            name="baseline",
            expression_mode="global_w1",
            w1=float(base_config.w1),
            w2=float(base_config.w2),
            w3=float(base_config.w3),
            w4=float(base_config.w4),
            w5=float(base_config.w5),
            shape_weight=float(base_config.shape_prior_weight),
            symmetric_weight=0.0,
            epsilon=float(epsilon),
        ),
        "w5": FormulaSpec(
            name="w5",
            expression_mode="global_w5",
            w1=0.0,
            w2=float(base_config.w2),
            w3=0.0,
            w4=float(base_config.w4),
            w5=float(expression_weight),
            shape_weight=w5_shape,
            symmetric_weight=0.0,
            epsilon=float(epsilon),
            calibration_shape_lambda=float(w5_row["shape_lambda_standardized"]),
        ),
        "symmetric": FormulaSpec(
            name="symmetric",
            expression_mode="action_local_symmetric_compatibility",
            w1=0.0,
            w2=float(base_config.w2),
            w3=0.0,
            w4=float(base_config.w4),
            w5=0.0,
            shape_weight=symmetric_shape,
            symmetric_weight=float(expression_weight),
            epsilon=float(epsilon),
            calibration_shape_lambda=float(symmetric_row["shape_lambda_standardized"]),
        ),
    }


def _variant_context(
    context: PatchContext,
    *,
    formula: FormulaSpec,
    max_steps: int,
) -> PatchContext:
    cells = []
    for cell in context.cells:
        base_penalty = (
            float(formula.w2) * np.asarray(cell.p_dis, dtype=np.float32)
            + float(formula.w3) * np.asarray(cell.p_overlap, dtype=np.float32)
        ).astype(np.float32, copy=False)
        cells.append(
            replace(
                cell,
                base_penalty=base_penalty,
                w1=float(formula.w1),
                w2=float(formula.w2),
                w3=float(formula.w3),
                w4=float(formula.w4),
                w5=float(formula.w5),
                shape_prior_weight=float(formula.shape_weight),
            )
        )
    return replace(
        context,
        cells=tuple(cells),
        # Expose every structurally legal REPLACE. The formula-specific epsilon
        # is applied explicitly before each step.
        global_delta_epsilon=-1.0e30,
        max_steps=int(max_steps),
    )


def _symmetric_compatibility_raw(env: TorchSingleAgentPatchEnv) -> Any:
    torch_module = env._torch
    target_cells = env._cached_action_cells
    target_bins = env._cached_action_bins
    old_cells = env._cached_action_old_cells
    old_bins = env._cached_action_old_bins
    out = torch_module.zeros(
        (int(target_bins.numel()),),
        device=env._device,
        dtype=torch_module.float32,
    )
    target_compatibility = torch_module.zeros_like(out)
    old_compatibility = torch_module.zeros_like(out)
    confidence = torch_module.zeros_like(out)
    for raw_cell in torch_module.unique(target_cells).detach().cpu().numpy().astype(np.int64):
        positions = torch_module.nonzero(target_cells == int(raw_cell), as_tuple=False).flatten()
        posterior = torch_module.softmax(env._score_sums[int(raw_cell)], dim=0)
        bins = target_bins[positions]
        target_compatibility[positions] = torch_module.mv(
            env._ll64[int(raw_cell)][bins],
            posterior,
        )
        confidence[positions] = env._expression_conf64[int(raw_cell)][bins]
    for raw_cell in torch_module.unique(old_cells).detach().cpu().numpy().astype(np.int64):
        if int(raw_cell) < 0:
            continue
        positions = torch_module.nonzero(old_cells == int(raw_cell), as_tuple=False).flatten()
        posterior = torch_module.softmax(env._score_sums[int(raw_cell)], dim=0)
        old_compatibility[positions] = torch_module.mv(
            env._ll64[int(raw_cell)][old_bins[positions]],
            posterior,
        )
    return (confidence * (target_compatibility - old_compatibility)).to(
        dtype=torch_module.float32
    )


def _formula_action_scores(
    env: TorchSingleAgentPatchEnv,
    formula: FormulaSpec,
) -> tuple[Any, Any, Any]:
    shape_add_correction = exact_shape_add_corrections(env)
    base_rewards = (
        env._cached_action_rewards.to(dtype=env._torch.float32)
        + shape_add_correction
    )
    symmetric_raw = env._torch.zeros_like(base_rewards)
    if formula.expression_mode == "action_local_symmetric_compatibility":
        symmetric_raw = _symmetric_compatibility_raw(env)
    scores = base_rewards + (
        float(formula.symmetric_weight)
        * symmetric_raw
        / float(env._score_denominator())
    )
    return scores.to(dtype=env._torch.float32), symmetric_raw, shape_add_correction


def _outer_candidate_barcodes(context: PatchContext) -> frozenset[str]:
    bounds = context.outer_bounds
    out: set[str] = set()
    for cell in context.cells:
        xy = np.asarray(cell.candidate_bin_xy_um, dtype=np.float64)
        valid = (
            (xy[:, 0] >= float(bounds.x_min))
            & (xy[:, 0] <= float(bounds.x_max))
            & (xy[:, 1] >= float(bounds.y_min))
            & (xy[:, 1] <= float(bounds.y_max))
        )
        out.update(
            str(cell.candidate_bin_ids[index])
            for index in np.flatnonzero(valid).tolist()
        )
    return frozenset(out)


def _load_truths(
    *,
    gt_path: Path,
    contexts: Iterable[PatchContext],
) -> dict[str, PatchTruth]:
    context_list = list(contexts)
    owner_by_patch: dict[str, dict[str, str]] = {
        context.patch_id: {} for context in context_list
    }
    sets_by_patch: dict[str, dict[str, set[str]]] = {
        context.patch_id: {} for context in context_list
    }
    for chunk in pd.read_csv(
        gt_path,
        usecols=["barcode", "x_um", "y_um", "cell_id"],
        dtype={"barcode": "string"},
        chunksize=500_000,
    ):
        x = pd.to_numeric(chunk["x_um"], errors="coerce").to_numpy(dtype=np.float64)
        y = pd.to_numeric(chunk["y_um"], errors="coerce").to_numpy(dtype=np.float64)
        for context in context_list:
            bounds = context.outer_bounds
            mask = (
                (x >= float(bounds.x_min))
                & (x <= float(bounds.x_max))
                & (y >= float(bounds.y_min))
                & (y <= float(bounds.y_max))
            )
            if not bool(np.any(mask)):
                continue
            subset = chunk.loc[mask, ["barcode", "cell_id"]]
            owner_map = owner_by_patch[context.patch_id]
            gt_sets = sets_by_patch[context.patch_id]
            for row in subset.itertuples(index=False):
                owner = normalize_cell_id(getattr(row, "cell_id"))
                if owner is None:
                    continue
                barcode = str(getattr(row, "barcode"))
                owner_map[barcode] = owner
                gt_sets.setdefault(owner, set()).add(barcode)

    truths: dict[str, PatchTruth] = {}
    for context in context_list:
        owner_map = owner_by_patch[context.patch_id]
        if not owner_map:
            raise RuntimeError(f"no GT bins found inside patch {context.patch_id}")
        identity = {
            str(cell.cell_id): str(cell.cell_id)
            for cell in context.cells
        }
        truths[context.patch_id] = PatchTruth(
            patch_id=context.patch_id,
            core_cell_ids=tuple(str(value) for value in context.core_cell_ids),
            owner_to_gt=identity,
            gt_owner_by_barcode=owner_map,
            gt_barcodes_by_cell={
                owner: frozenset(barcodes)
                for owner, barcodes in sets_by_patch[context.patch_id].items()
            },
            outer_barcodes=_outer_candidate_barcodes(context),
        )
    return truths


def _run_formula_patch(
    *,
    context: PatchContext,
    truth: PatchTruth,
    formula: FormulaSpec,
    split: str,
    max_steps: int,
    device: torch.device,
) -> GreedyPatchResult:
    if device.type == "cuda":
        torch.cuda.set_device(device)
    variant = _variant_context(context, formula=formula, max_steps=max_steps)
    env = TorchSingleAgentPatchEnv(variant, device=device)
    env.reset()
    initial_owners = _owners_from_env(env)
    current_owners = dict(initial_owners)
    initial_components = env.objective_components()
    initial_macro_iou = _mean_iou(initial_owners, truth)
    action_rows: list[dict[str, Any]] = []
    cumulative_formula_reward = 0.0
    stop_reason = "no_structural_replace"

    while len(action_rows) < int(max_steps):
        if int(env._cached_action_rewards.numel()) == 0:
            break
        replace_positions = env._torch.nonzero(
            env._cached_action_is_replace,
            as_tuple=False,
        ).flatten()
        if int(replace_positions.numel()) == 0:
            break
        scores, symmetric_raw, shape_add_correction = _formula_action_scores(env, formula)
        replace_scores = scores[replace_positions]
        local_position = int(env._torch.argmax(replace_scores).detach().cpu().item())
        cached_position = int(replace_positions[local_position].detach().cpu().item())
        best_score = float(scores[cached_position].detach().cpu().item())
        if best_score <= float(formula.epsilon):
            stop_reason = "no_formula_delta_above_epsilon"
            break

        new_cell_idx = int(env._cached_action_cells[cached_position].detach().cpu().item())
        bin_idx = int(env._cached_action_bins[cached_position].detach().cpu().item())
        old_cell_idx = int(env._cached_action_old_cells[cached_position].detach().cpu().item())
        barcode_idx = int(env._barcode_indices[new_cell_idx][bin_idx].detach().cpu().item())
        barcode = str(env._barcode_keys[barcode_idx])
        new_owner = str(env._cell_ids[new_cell_idx])
        old_owner = str(env._cell_ids[old_cell_idx])
        before_components = env.objective_components()
        iou_before = _mean_iou(current_owners, truth)
        selected_symmetric_raw = float(symmetric_raw[cached_position].detach().cpu().item())
        selected_shape_add_correction = float(
            shape_add_correction[cached_position].detach().cpu().item()
        )
        selected_symmetric_reward = float(
            formula.symmetric_weight
            * selected_symmetric_raw
            / float(env._score_denominator())
        )
        cached_components = cached_replace_component_deltas(env, cached_position)

        _, base_reward, _, _, _ = env.step(cached_position + 1)
        current_owners[barcode] = new_owner
        after_components = env.objective_components()
        iou_after = _mean_iou(current_owners, truth)
        component_deltas = {
            key: float(after_components.get(key, 0.0) - before_components.get(key, 0.0))
            for key in OBJECTIVE_COMPONENTS
        }
        component_sum = float(sum(component_deltas.values()))
        if not np.isclose(float(base_reward), component_sum, rtol=0.0, atol=2.0e-5):
            raise RuntimeError(
                f"base reward reconstruction failed in {context.patch_id}: "
                f"{base_reward} vs {component_sum}"
            )
        reconstructed = component_sum + selected_symmetric_reward
        if not np.isclose(best_score, reconstructed, rtol=0.0, atol=2.0e-5):
            print(
                json.dumps(
                    {
                        "patch_id": context.patch_id,
                        "step_index": len(action_rows) + 1,
                        "cached_components": cached_components,
                        "exact_components": component_deltas,
                    },
                    indent=2,
                ),
                flush=True,
            )
            raise RuntimeError(
                f"formula reward reconstruction failed in {context.patch_id}: "
                f"{best_score} vs {reconstructed}"
            )
        cumulative_formula_reward += best_score
        action_rows.append(
            {
                "split": split,
                "formula": formula.name,
                "expression_mode": formula.expression_mode,
                "patch_id": context.patch_id,
                "step_index": len(action_rows) + 1,
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
                "formula_reward": best_score,
                "base_global_delta_reward": float(base_reward),
                "exact_shape_add_cache_correction": selected_shape_add_correction,
                "symmetric_compatibility_raw": selected_symmetric_raw,
                "symmetric_compatibility_reward": selected_symmetric_reward,
                "macro_iou_before": iou_before,
                "macro_iou_after": iou_after,
                "macro_iou_delta": float(iou_after - iou_before),
                "formula_reward_residual": float(best_score - reconstructed),
                **component_deltas,
            }
        )
    else:
        stop_reason = "max_steps"

    final_owners = _owners_from_env(env)
    if final_owners != current_owners:
        raise RuntimeError(f"owner tracking diverged in patch {context.patch_id}")
    final_components = env.objective_components()
    initial_correct, initial_evaluated, initial_accuracy = _owner_accuracy(initial_owners, truth)
    final_correct, final_evaluated, final_accuracy = _owner_accuracy(final_owners, truth)
    changed = sorted(
        barcode
        for barcode in set(initial_owners) | set(final_owners)
        if initial_owners.get(barcode) != final_owners.get(barcode)
    )
    net_outcomes = [
        _transition_outcome(
            barcode=barcode,
            old_owner=initial_owners.get(barcode),
            new_owner=final_owners.get(barcode),
            truth=truth,
        )
        for barcode in changed
    ]
    final_macro_iou = _mean_iou(final_owners, truth)
    patch_row = {
        "split": split,
        "formula": formula.name,
        "expression_mode": formula.expression_mode,
        "patch_id": context.patch_id,
        "w1": formula.w1,
        "w2": formula.w2,
        "w3": formula.w3,
        "w4": formula.w4,
        "w5": formula.w5,
        "shape_weight": formula.shape_weight,
        "symmetric_weight": formula.symmetric_weight,
        "global_delta_epsilon": formula.epsilon,
        "initial_macro_cell_iou": initial_macro_iou,
        "final_macro_cell_iou": final_macro_iou,
        "macro_cell_iou_delta": float(final_macro_iou - initial_macro_iou),
        "initial_owner_correct": initial_correct,
        "initial_owner_evaluated": initial_evaluated,
        "initial_owner_accuracy": initial_accuracy,
        "final_owner_correct": final_correct,
        "final_owner_evaluated": final_evaluated,
        "final_owner_accuracy": final_accuracy,
        "formula_reward_sum": cumulative_formula_reward,
        "n_actions": len(action_rows),
        "n_changed_bins": len(changed),
        "n_net_corrected": net_outcomes.count("corrected"),
        "n_net_damaged": net_outcomes.count("damaged"),
        "n_net_no_gt": net_outcomes.count("no_gt"),
        "n_net_wrong_to_wrong": net_outcomes.count("wrong_to_wrong"),
        "n_net_unresolved": net_outcomes.count("unresolved_owner_mapping"),
        "termination_reason": stop_reason,
    }
    for component in OBJECTIVE_COMPONENTS:
        patch_row[f"delta_{component}"] = float(
            final_components.get(component, 0.0)
            - initial_components.get(component, 0.0)
        )

    cell_rows = _cell_iou_rows(initial_owners, truth, stage="em_initial")
    cell_rows.extend(_cell_iou_rows(final_owners, truth, stage="formula_greedy"))
    for row in cell_rows:
        row.update(
            {
                "split": split,
                "formula": formula.name,
                "expression_mode": formula.expression_mode,
            }
        )
    return GreedyPatchResult(
        patch_row=patch_row,
        action_rows=tuple(action_rows),
        cell_rows=tuple(cell_rows),
    )


def _summarize(
    patch_df: pd.DataFrame,
    action_df: pd.DataFrame,
    cell_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for formula, patches in patch_df.groupby("formula", sort=False):
        actions = action_df.loc[action_df["formula"] == formula]
        cells = cell_df.loc[cell_df["formula"] == formula]
        initial_iou = float(cells.loc[cells.stage == "em_initial", "iou"].mean())
        final_iou = float(cells.loc[cells.stage == "formula_greedy", "iou"].mean())
        outcomes = actions.outcome.value_counts().to_dict()
        row = {
            "split": str(patches.iloc[0]["split"]),
            "formula": formula,
            "expression_mode": str(patches.iloc[0]["expression_mode"]),
            "n_patches": int(len(patches)),
            "n_cells": int(np.sum(cells.stage == "em_initial")),
            "em_initial_macro_cell_iou": initial_iou,
            "final_macro_cell_iou": final_iou,
            "macro_cell_iou_delta": float(final_iou - initial_iou),
            "initial_owner_accuracy": float(
                patches.initial_owner_correct.sum()
                / max(1, patches.initial_owner_evaluated.sum())
            ),
            "final_owner_accuracy": float(
                patches.final_owner_correct.sum()
                / max(1, patches.final_owner_evaluated.sum())
            ),
            "n_actions": int(len(actions)),
            "n_corrected_actions": int(outcomes.get("corrected", 0)),
            "n_damaged_actions": int(outcomes.get("damaged", 0)),
            "n_no_gt_actions": int(outcomes.get("no_gt", 0)),
            "n_wrong_to_wrong_actions": int(outcomes.get("wrong_to_wrong", 0)),
            "n_net_corrected": int(patches.n_net_corrected.sum()),
            "n_net_damaged": int(patches.n_net_damaged.sum()),
            "n_net_no_gt": int(patches.n_net_no_gt.sum()),
            "n_max_step_terminations": int(np.sum(patches.termination_reason == "max_steps")),
            "max_formula_reward_residual": (
                0.0
                if actions.empty
                else float(actions.formula_reward_residual.abs().max())
            ),
        }
        row["passes_iou_gate"] = bool(row["macro_cell_iou_delta"] >= 0.0)
        row["passes_damage_gate"] = bool(row["n_net_corrected"] > row["n_net_damaged"])
        row["passes_termination_gate"] = bool(row["n_max_step_terminations"] == 0)
        row["passes_all_gates"] = bool(
            row["passes_iou_gate"]
            and row["passes_damage_gate"]
            and row["passes_termination_gate"]
        )
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(
        ["final_macro_cell_iou", "n_net_damaged", "n_net_corrected"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    summary.insert(0, "validation_selection_rank", np.arange(1, len(summary) + 1))
    summary["selected_on_validation"] = False
    if not summary.empty and str(summary.iloc[0]["split"]) == "validation":
        summary.loc[0, "selected_on_validation"] = True
    return summary


def _evolution_table(
    patch_df: pd.DataFrame,
    action_df: pd.DataFrame,
    *,
    n_points: int = 101,
) -> pd.DataFrame:
    grid = np.linspace(0.0, 1.0, int(n_points))
    rows: list[dict[str, Any]] = []
    for formula, formula_patches in patch_df.groupby("formula", sort=False):
        curves = []
        for patch in formula_patches.itertuples(index=False):
            actions = action_df.loc[
                (action_df.formula == formula)
                & (action_df.patch_id == str(patch.patch_id))
            ].sort_values("step_index")
            values = [float(patch.initial_macro_cell_iou)]
            values.extend(actions.macro_iou_after.astype(float).tolist())
            if len(values) == 1:
                progress = np.asarray([0.0, 1.0])
                values_np = np.asarray([values[0], values[0]])
            else:
                progress = np.linspace(0.0, 1.0, len(values))
                values_np = np.asarray(values, dtype=np.float64)
            curves.append(np.interp(grid, progress, values_np))
        matrix = np.stack(curves, axis=0)
        mean = np.mean(matrix, axis=0)
        sem = np.std(matrix, axis=0, ddof=1) / np.sqrt(max(1, matrix.shape[0]))
        for index, progress in enumerate(grid.tolist()):
            rows.append(
                {
                    "split": str(formula_patches.iloc[0]["split"]),
                    "formula": formula,
                    "rollout_progress": progress,
                    "mean_macro_cell_iou": float(mean[index]),
                    "sem_macro_cell_iou": float(sem[index]),
                    "n_patches": int(matrix.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def _plot_results(
    *,
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
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    for formula, group in evolution.groupby("formula", sort=False):
        x = group.rollout_progress.to_numpy(dtype=float)
        y = group.mean_macro_cell_iou.to_numpy(dtype=float)
        sem = group.sem_macro_cell_iou.to_numpy(dtype=float)
        axes[0].plot(x, y, label=formula, color=_COLORS.get(formula, "#111827"), linewidth=1.5)
        axes[0].fill_between(x, y - sem, y + sem, color=_COLORS.get(formula, "#111827"), alpha=0.15, linewidth=0)
    axes[0].set_xlabel("Normalized rollout progress")
    axes[0].set_ylabel("Mean patch macro IoU")
    axes[0].set_title("a  IoU evolution", loc="left", fontweight="bold")
    axes[0].legend(fontsize=6)

    x = np.arange(len(summary))
    width = 0.34
    axes[1].bar(x - width / 2, summary.em_initial_macro_cell_iou, width, color="#D1D5DB", label="EM initial")
    axes[1].bar(x + width / 2, summary.final_macro_cell_iou, width, color=[_COLORS.get(name, "#111827") for name in summary.formula], label="Final")
    axes[1].set_xticks(x, summary.formula, rotation=20, ha="right")
    axes[1].set_ylabel("Macro cell IoU")
    axes[1].set_title("b  Initial vs final", loc="left", fontweight="bold")
    axes[1].legend(fontsize=6)

    axes[2].bar(x - width / 2, summary.n_net_corrected, width, color="#2A9D8F", label="Net corrected")
    axes[2].bar(x + width / 2, summary.n_net_damaged, width, color="#C85C5C", label="Net damaged")
    axes[2].set_xticks(x, summary.formula, rotation=20, ha="right")
    axes[2].set_ylabel("Changed bins")
    axes[2].set_title("c  Final ownership", loc="left", fontweight="bold")
    axes[2].legend(fontsize=6)

    stem = output_dir / "iou_reward_evolution"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_report(summary: pd.DataFrame, *, split: str) -> str:
    winner = summary.iloc[0]
    lines = [
        f"# {split} sequential reward-formula rollout",
        "",
        "Every patch starts from the same EM hard ownership. Actions are selected without ground truth. Ground truth is used only to label REPLACE outcomes and calculate IoU after each step.",
        "",
        "## Result",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- `{row.formula}`: IoU `{row.em_initial_macro_cell_iou:.6f}` → "
            f"`{row.final_macro_cell_iou:.6f}` (`{row.macro_cell_iou_delta:+.6f}`); "
            f"actions `{int(row.n_actions)}`; net corrected/damaged/no-GT "
            f"`{int(row.n_net_corrected)}/{int(row.n_net_damaged)}/{int(row.n_net_no_gt)}`."
        )
    if split == "validation":
        lines.extend(
            [
                "",
                "## Pre-registered selection",
                "",
                f"`{winner.formula}` ranks first by final macro IoU, then lower net damage, then higher net correction. Only this formula should be evaluated on the held-out test split.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Test decision",
                "",
                "A new PPO run is justified only if the held-out formula improves IoU, has more net corrected than net damaged bins, and reaches the formula stopping condition without the safety cap.",
            ]
        )
    lines.extend(
        [
            "",
            "The symmetric formula is an action-local score recomputed after every REPLACE; it is not described as the delta of a single scalar global objective. Baseline and `w5` use the existing exact Torch global-delta implementation.",
            "",
            "See `summary.csv`, `per_patch.csv`, `actions.parquet`, `per_cell.csv`, `iou_evolution.csv`, and `iou_reward_evolution.*`.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_context_worker(
    *,
    rows: list[dict[str, Any]],
    base_config: Any,
    settings: Any,
    seed: int,
) -> list[PatchContext]:
    dataset = PatchDataset(
        base_config=base_config,
        settings=settings,
        rng=np.random.default_rng(int(seed)),
    )
    contexts = []
    try:
        for raw in rows:
            context = dataset.load_patch_context(SimpleNamespace(**raw))
            if context is None:
                raise RuntimeError(f"failed to load patch context {raw['patch_id']}")
            contexts.append(context)
    finally:
        dataset.close()
    return contexts


def main() -> None:
    args = _parse_args()
    patch_config_path = Path(args.patch_config).expanduser().resolve()
    selected_index_path = Path(args.selected_patches_index).expanduser().resolve()
    gt_path = Path(args.gt_cell_bins_path).expanduser().resolve()
    audit_dir = Path(args.reward_audit_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    checkpoint_dir = (
        None if args.checkpoint_dir is None else Path(args.checkpoint_dir).expanduser().resolve()
    )
    formula_names = _parse_formula_names(args.formulas)
    devices = tuple(
        torch.device(item.strip())
        for item in str(args.devices).split(",")
        if item.strip()
    )
    if not devices:
        raise ValueError("--devices must contain at least one device")
    for device in devices:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA unavailable for requested device {device}")

    patch_cfg = _load_yaml(patch_config_path)
    base_config_path = Path(str(patch_cfg["base_ppo_config"])).expanduser().resolve()
    base_config = load_ppo_training_config(base_config_path)
    settings = _settings_from_patch_config(patch_cfg, max_steps=int(args.max_steps))
    settings = replace(
        settings,
        patches_index_path=selected_index_path,
        cache_patch_contexts=False,
        max_steps_per_patch=int(args.max_steps),
    )
    if not settings.em_assignment.enabled:
        raise ValueError("formula rollout requires em_assignment.enabled=true")
    if settings.agent_mode != "multi_cell_global_delta":
        raise ValueError("formula rollout requires multi_cell_global_delta")
    if settings.after_fill_actions != "replace_only":
        raise ValueError("formula rollout requires after_fill_actions=replace_only")

    selected = pd.read_csv(selected_index_path)
    if "audit_split" not in selected.columns:
        raise ValueError("selected patch index must contain audit_split")
    selected = selected.loc[selected.audit_split.astype(str) == str(args.split)].copy()
    requested = tuple(
        item.strip() for item in str(args.patch_ids or "").split(",") if item.strip()
    )
    if requested:
        requested_set = set(requested)
        selected = selected.loc[selected.patch_id.astype(str).isin(requested_set)].copy()
        missing = sorted(requested_set - set(selected.patch_id.astype(str)))
        if missing:
            raise RuntimeError(f"requested patch IDs not found in {args.split}: {missing}")
    expected_count = len(requested) if requested else 20
    if len(selected) != expected_count:
        raise RuntimeError(
            f"expected exactly {expected_count} {args.split} patches, found {len(selected)}"
        )
    row_records = selected.to_dict(orient="records")
    assignments: list[list[dict[str, Any]]] = [[] for _ in devices]
    for index, raw in enumerate(row_records):
        assignments[index % len(devices)].append(raw)
    contexts: list[PatchContext] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(
                _load_context_worker,
                rows=rows,
                base_config=base_config,
                settings=settings,
                seed=int(base_config.seed) + index,
            )
            for index, rows in enumerate(assignments)
            if rows
        ]
        for future in futures:
            contexts.extend(future.result())
    order = {str(raw["patch_id"]): index for index, raw in enumerate(row_records)}
    contexts.sort(key=lambda context: order[context.patch_id])
    print(f"Loaded {len(contexts)} {args.split} contexts; loading GT once.", flush=True)
    truths = _load_truths(gt_path=gt_path, contexts=contexts)

    all_specs = _build_formula_specs(
        base_config=base_config,
        audit_dir=audit_dir,
        expression_weight=float(args.expression_weight),
        epsilon=float(args.global_delta_epsilon),
    )
    specs = [all_specs[name] for name in formula_names]
    context_assignments: list[list[PatchContext]] = [[] for _ in devices]
    for index, context in enumerate(contexts):
        context_assignments[index % len(devices)].append(context)
    patch_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []

    for formula in specs:
        restored = None
        if checkpoint_dir is not None:
            restored = load_formula_checkpoint(
                directory=checkpoint_dir,
                split=str(args.split),
                formula=formula.__dict__,
                patch_ids=[context.patch_id for context in contexts],
            )
        if restored is not None:
            restored_patch, restored_action, restored_cell = restored
            patch_rows.extend(restored_patch.to_dict(orient="records"))
            action_rows.extend(restored_action.to_dict(orient="records"))
            cell_rows.extend(restored_cell.to_dict(orient="records"))
            print(
                f"Restored {formula.name}: actions={len(restored_action)}",
                flush=True,
            )
            continue

        def run_device(device_index: int) -> list[GreedyPatchResult]:
            results = []
            for index, context in enumerate(context_assignments[device_index], start=1):
                results.append(
                    _run_formula_patch(
                        context=context,
                        truth=truths[context.patch_id],
                        formula=formula,
                        split=str(args.split),
                        max_steps=int(args.max_steps),
                        device=devices[device_index],
                    )
                )
                print(
                    f"{devices[device_index]} {formula.name}: "
                    f"{index}/{len(context_assignments[device_index])} patches",
                    flush=True,
                )
            return results

        results: list[GreedyPatchResult] = []
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [
                executor.submit(run_device, index)
                for index, items in enumerate(context_assignments)
                if items
            ]
            for future in futures:
                results.extend(future.result())
        for result in results:
            patch_rows.append(result.patch_row)
            action_rows.extend(result.action_rows)
            cell_rows.extend(result.cell_rows)
        if checkpoint_dir is not None:
            save_formula_checkpoint(
                directory=checkpoint_dir,
                split=str(args.split),
                formula=formula.__dict__,
                patch_ids=[context.patch_id for context in contexts],
                patch_rows=(result.patch_row for result in results),
                action_rows=(row for result in results for row in result.action_rows),
                cell_rows=(row for result in results for row in result.cell_rows),
            )
        print(
            f"Completed {formula.name}: "
            f"actions={sum(result.patch_row['n_actions'] for result in results)}",
            flush=True,
        )

    patch_df = pd.DataFrame(patch_rows)
    action_df = pd.DataFrame(action_rows)
    cell_df = pd.DataFrame(cell_rows)
    if action_df.empty:
        action_df = pd.DataFrame(
            columns=[
                "split",
                "formula",
                "patch_id",
                "step_index",
                "outcome",
                "formula_reward_residual",
            ]
        )
    summary = _summarize(patch_df, action_df, cell_df)
    evolution = _evolution_table(patch_df, action_df)
    max_residual = (
        0.0
        if action_df.empty
        else float(action_df.formula_reward_residual.abs().max())
    )
    if max_residual > 2.0e-5:
        raise RuntimeError(f"formula reward residual exceeds tolerance: {max_residual}")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / f"formula_rollout_{args.split}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    selected.to_csv(output_dir / "selected_patches.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    patch_df.to_csv(output_dir / "per_patch.csv", index=False)
    action_df.to_parquet(output_dir / "actions.parquet", index=False)
    cell_df.to_csv(output_dir / "per_cell.csv", index=False)
    evolution.to_csv(output_dir / "iou_evolution.csv", index=False)
    _plot_results(summary=summary, evolution=evolution, output_dir=output_dir)
    config = {
        "split": str(args.split),
        "patch_config": str(patch_config_path),
        "base_config": str(base_config_path),
        "selected_patches_index": str(selected_index_path),
        "gt_cell_bins_path": str(gt_path),
        "reward_audit_dir": str(audit_dir),
        "formulas": [formula.__dict__ for formula in specs],
        "selection_rule": "highest final macro cell IoU, then lower net damage, then higher net correction",
        "devices": [str(device) for device in devices],
        "max_steps": int(args.max_steps),
        "temporary_checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
        "max_formula_reward_residual": max_residual,
        "ground_truth_usage": "evaluation_only_after_each_action_selection",
        "symmetric_scope": "action_local_score_not_scalar_global_objective_delta",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        _write_report(summary, split=str(args.split)),
        encoding="utf-8",
    )
    if checkpoint_dir is not None:
        remove_formula_checkpoint(checkpoint_dir)
    print(summary.to_string(index=False), flush=True)
    print(f"Formula rollout: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
