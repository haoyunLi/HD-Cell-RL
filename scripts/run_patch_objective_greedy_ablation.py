#!/usr/bin/env python
"""Ablate patch objective weights with exact greedy global-delta REPLACE.

This is a diagnostic runner, not a policy and not a training entry point.  It
uses the existing Torch patch environment, starts every trajectory from the
same EM ownership, and repeatedly applies the currently exposed REPLACE with
the largest exact global-objective delta.  Ground truth is loaded only after
the action is selected and is used solely for evaluation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hd_cell_rl.em_types import EMAssignmentConfig
from hd_cell_rl.patch_dataset import PatchDataset
from hd_cell_rl.patch_env_torch import TorchSingleAgentPatchEnv
from hd_cell_rl.patch_types import PatchContext, PatchTrainingSettings
from hd_cell_rl.ppo_config import load_ppo_training_config
from preprocessing.ppo_eval_metrics import normalize_cell_id


OBJECTIVE_COMPONENTS = (
    "expression_confidence",
    "expression_compatibility",
    "distance_penalty",
    "overlap_penalty",
    "neighbor_support",
    "shape_prior",
)


@dataclass(frozen=True)
class PatchTruth:
    patch_id: str
    core_cell_ids: tuple[str, ...]
    owner_to_gt: dict[str, str]
    gt_owner_by_barcode: dict[str, str]
    gt_barcodes_by_cell: dict[str, frozenset[str]]
    outer_barcodes: frozenset[str]


@dataclass(frozen=True)
class GreedyPatchResult:
    patch_row: dict[str, Any]
    action_rows: tuple[dict[str, Any], ...]
    cell_rows: tuple[dict[str, Any], ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-config", required=True)
    parser.add_argument("--patch-debug-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shape-scales", default="0,0.25,0.5,1")
    parser.add_argument("--w1-scales", default="1")
    parser.add_argument("--global-delta-epsilons", default="1e-6,1e-4,2.5e-4,5e-4,1e-3")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Safety cap only; normal termination is no remaining delta above epsilon.",
    )
    args = parser.parse_args()
    if int(args.max_steps) <= 0:
        raise ValueError("--max-steps must be > 0")
    return args


def _parse_float_list(raw: str, *, name: str, minimum: float = 0.0) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(raw).split(",") if item.strip())
    if not values:
        raise ValueError(f"{name} must contain at least one number")
    if any(not np.isfinite(value) or value < minimum for value in values):
        raise ValueError(f"{name} values must be finite and >= {minimum}")
    return values


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return payload


def _settings_from_patch_config(patch_cfg: dict[str, Any], *, max_steps: int) -> PatchTrainingSettings:
    patch = dict(patch_cfg.get("patch_training", {}))
    return PatchTrainingSettings(
        patches_index_path=Path(str(patch["patches_index_path"])).expanduser().resolve(),
        batch_patches=1,
        max_steps_per_patch=int(max_steps),
        margin_cells_compete=bool(patch.get("margin_cells_compete", True)),
        use_core_cells_for_score=bool(patch.get("use_core_cells_for_score", True)),
        score_normalization=str(patch.get("score_normalization", "mean_core_cells")),
        rollout_backend="torch_gpu",
        reward_backend=str(patch.get("reward_backend", "standard")),
        stcs_reward_config=dict(patch.get("stcs_reward", {}) or {}),
        cache_patch_contexts=False,
        competition_margin_enabled=bool(patch.get("competition_margin_enabled", True)),
        force_fill_expression_bins=bool(patch.get("force_fill_expression_bins", False)),
        fill_target=str(patch.get("fill_target", "reachable_expression_bins")),
        stop_action_mode=str(patch.get("stop_action_mode", "enabled")),
        agent_mode=str(patch.get("agent_mode", "multi_cell")),
        after_fill_actions=str(patch.get("after_fill_actions", "add_or_stop")),
        global_delta_epsilon=float(patch.get("global_delta_epsilon", 1.0e-6)),
        em_assignment=EMAssignmentConfig.from_mapping(patch_cfg.get("em_assignment", {})),
    )


def _load_patch_truth(debug_dir: Path) -> dict[str, PatchTruth]:
    manifest_path = debug_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    truths: dict[str, PatchTruth] = {}
    for item in manifest.get("patches", []):
        payload = json.loads((debug_dir / str(item["file"])).read_text(encoding="utf-8"))
        patch_id = str(payload["patch_id"])
        owner_to_gt: dict[str, str] = {}
        for cell in payload.get("cells", []):
            owner = normalize_cell_id(cell.get("cell_id"))
            gt_owner = normalize_cell_id(cell.get("matched_gt_cell_id"))
            if owner is not None and gt_owner is not None:
                owner_to_gt[owner] = gt_owner

        gt_owner_by_barcode: dict[str, str] = {}
        gt_sets: dict[str, set[str]] = {}
        outer = dict(payload.get("outer_bounds", {}))
        outer_barcodes: set[str] = set()
        for row in payload.get("bins", []):
            barcode = str(row["barcode"])
            x_um = float(row["x_um"])
            y_um = float(row["y_um"])
            if (
                float(outer["x_min"]) <= x_um <= float(outer["x_max"])
                and float(outer["y_min"]) <= y_um <= float(outer["y_max"])
            ):
                outer_barcodes.add(barcode)
            gt_owner = normalize_cell_id(row.get("gt_owner_cell_id"))
            if gt_owner is not None:
                gt_owner_by_barcode[barcode] = gt_owner
                gt_sets.setdefault(gt_owner, set()).add(barcode)

        truths[patch_id] = PatchTruth(
            patch_id=patch_id,
            core_cell_ids=tuple(str(value) for value in payload.get("core_cell_ids", [])),
            owner_to_gt=owner_to_gt,
            gt_owner_by_barcode=gt_owner_by_barcode,
            gt_barcodes_by_cell={key: frozenset(value) for key, value in gt_sets.items()},
            outer_barcodes=frozenset(outer_barcodes),
        )
    if not truths:
        raise RuntimeError(f"no patch truth payloads found in {debug_dir}")
    return truths


def _mapped_gt_owner(owner: str | None, truth: PatchTruth) -> str | None:
    normalized = normalize_cell_id(owner)
    if normalized is None:
        return None
    mapped = truth.owner_to_gt.get(normalized)
    if mapped is not None:
        return mapped
    gt_ids = truth.gt_barcodes_by_cell.keys()
    return normalized if normalized in gt_ids else None


def _owners_from_env(env: TorchSingleAgentPatchEnv) -> dict[str, str]:
    owner_indices = env._owner_by_barcode.detach().cpu().numpy().astype(np.int64, copy=False)
    return {
        str(env._barcode_keys[index]): str(env._cell_ids[int(owner_idx)])
        for index, owner_idx in enumerate(owner_indices.tolist())
        if int(owner_idx) >= 0
    }


def _cell_iou_rows(
    owners: dict[str, str],
    truth: PatchTruth,
    *,
    stage: str,
) -> list[dict[str, Any]]:
    pred_by_cell: dict[str, set[str]] = {cell_id: set() for cell_id in truth.core_cell_ids}
    core = set(truth.core_cell_ids)
    for barcode, owner in owners.items():
        if barcode in truth.outer_barcodes and owner in core:
            pred_by_cell[owner].add(barcode)

    rows: list[dict[str, Any]] = []
    for owner in truth.core_cell_ids:
        matched_gt = _mapped_gt_owner(owner, truth)
        pred = pred_by_cell[owner]
        gt = set() if matched_gt is None else set(truth.gt_barcodes_by_cell.get(matched_gt, ()))
        intersection = len(pred & gt)
        union = len(pred | gt)
        rows.append(
            {
                "patch_id": truth.patch_id,
                "stage": stage,
                "cell_id": owner,
                "matched_gt_cell_id": matched_gt,
                "predicted_bins": len(pred),
                "gt_bins": len(gt),
                "intersection": intersection,
                "union": union,
                "iou": np.nan if union == 0 else float(intersection / union),
            }
        )
    return rows


def _owner_accuracy(owners: dict[str, str], truth: PatchTruth) -> tuple[int, int, float]:
    correct = 0
    evaluated = 0
    for barcode, gt_owner in truth.gt_owner_by_barcode.items():
        mapped = _mapped_gt_owner(owners.get(barcode), truth)
        if mapped is None:
            continue
        evaluated += 1
        correct += int(mapped == gt_owner)
    accuracy = np.nan if evaluated == 0 else float(correct / evaluated)
    return correct, evaluated, accuracy


def _transition_outcome(
    *,
    barcode: str,
    old_owner: str | None,
    new_owner: str | None,
    truth: PatchTruth,
) -> str:
    gt_owner = truth.gt_owner_by_barcode.get(barcode)
    if gt_owner is None:
        return "no_gt"
    old_gt = _mapped_gt_owner(old_owner, truth)
    new_gt = _mapped_gt_owner(new_owner, truth)
    before = old_gt == gt_owner
    after = new_gt == gt_owner
    if before and not after:
        return "damaged"
    if not before and after:
        return "corrected"
    if before and after:
        return "unchanged_correct"
    if old_gt is None or new_gt is None:
        return "unresolved_owner_mapping"
    return "wrong_to_wrong"


def _mean_iou(owners: dict[str, str], truth: PatchTruth) -> float:
    values = [row["iou"] for row in _cell_iou_rows(owners, truth, stage="temporary")]
    finite = [float(value) for value in values if np.isfinite(value)]
    return np.nan if not finite else float(np.mean(finite))


def _variant_context(
    context: PatchContext,
    *,
    w1_weight: float,
    shape_weight: float,
    global_delta_epsilon: float,
    max_steps: int,
) -> PatchContext:
    return replace(
        context,
        cells=tuple(
            replace(
                cell,
                w1=float(w1_weight),
                shape_prior_weight=float(shape_weight),
            )
            for cell in context.cells
        ),
        global_delta_epsilon=float(global_delta_epsilon),
        max_steps=int(max_steps),
    )


def _run_greedy_patch(
    *,
    context: PatchContext,
    truth: PatchTruth,
    w1_scale: float,
    w1_weight: float,
    shape_scale: float,
    shape_weight: float,
    global_delta_epsilon: float,
    max_steps: int,
    device: torch.device,
) -> GreedyPatchResult:
    if device.type == "cuda":
        torch.cuda.set_device(device)
    variant = _variant_context(
        context,
        w1_weight=w1_weight,
        shape_weight=shape_weight,
        global_delta_epsilon=global_delta_epsilon,
        max_steps=max_steps,
    )
    env = TorchSingleAgentPatchEnv(variant, device=device)
    env.reset()
    initial_owners = _owners_from_env(env)
    current_owners = dict(initial_owners)
    initial_score = float(env.patch_score())
    initial_components = env.objective_components()
    initial_macro_iou = _mean_iou(initial_owners, truth)
    action_rows: list[dict[str, Any]] = []
    terminated = bool(env._terminated)
    truncated = bool(env._truncated)

    while not terminated and not truncated:
        rewards = env._cached_action_rewards
        if int(rewards.numel()) <= 0:
            raise RuntimeError(
                f"patch {context.patch_id} is active but exposes no greedy action"
            )
        action_idx = int(torch.argmax(rewards).detach().cpu().item())
        best_cached_reward = float(rewards[action_idx].detach().cpu().item())
        is_replace = bool(env._cached_action_is_replace[action_idx].detach().cpu().item())
        if not is_replace:
            raise RuntimeError("EM greedy ablation expected REPLACE-only actions after reset")
        new_cell_idx = int(env._cached_action_cells[action_idx].detach().cpu().item())
        bin_idx = int(env._cached_action_bins[action_idx].detach().cpu().item())
        old_cell_idx = int(env._cached_action_old_cells[action_idx].detach().cpu().item())
        barcode_index = int(env._barcode_indices[new_cell_idx][bin_idx].detach().cpu().item())
        barcode = str(env._barcode_keys[barcode_index])
        new_owner = str(env._cell_ids[new_cell_idx])
        old_owner = str(env._cell_ids[old_cell_idx])
        before_components = env.objective_components()
        iou_before = _mean_iou(current_owners, truth)

        _, reward, terminated, truncated, _ = env.step(action_idx + 1)
        current_owners[barcode] = new_owner
        after_components = env.objective_components()
        iou_after = _mean_iou(current_owners, truth)
        component_deltas = {
            key: float(after_components.get(key, 0.0) - before_components.get(key, 0.0))
            for key in OBJECTIVE_COMPONENTS
        }
        component_sum = float(sum(component_deltas.values()))
        if not np.isclose(reward, best_cached_reward, rtol=0.0, atol=2.0e-5):
            raise RuntimeError(
                f"cached/exact reward mismatch in {context.patch_id}: "
                f"{best_cached_reward} vs {reward}"
            )
        action_rows.append(
            {
                "patch_id": context.patch_id,
                "step_index": len(action_rows) + 1,
                "w1_scale": float(w1_scale),
                "w1_weight": float(w1_weight),
                "shape_scale": float(shape_scale),
                "shape_weight": float(shape_weight),
                "global_delta_epsilon": float(global_delta_epsilon),
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
                "reward": float(reward),
                "best_cached_reward": best_cached_reward,
                "macro_iou_before": iou_before,
                "macro_iou_after": iou_after,
                "macro_iou_delta": float(iou_after - iou_before),
                "component_sum": component_sum,
                "reward_residual": float(reward - component_sum),
                **component_deltas,
            }
        )

    final_owners = _owners_from_env(env)
    if final_owners != current_owners:
        raise RuntimeError(f"incremental owner tracking diverged in patch {context.patch_id}")
    final_score = float(env.patch_score())
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
    patch_row = {
        "patch_id": context.patch_id,
        "w1_scale": float(w1_scale),
        "w1_weight": float(w1_weight),
        "shape_scale": float(shape_scale),
        "shape_weight": float(shape_weight),
        "global_delta_epsilon": float(global_delta_epsilon),
        "initial_patch_score": initial_score,
        "final_patch_score": final_score,
        "objective_gain": float(final_score - initial_score),
        "initial_macro_cell_iou": initial_macro_iou,
        "final_macro_cell_iou": _mean_iou(final_owners, truth),
        "macro_cell_iou_delta": float(_mean_iou(final_owners, truth) - initial_macro_iou),
        "initial_owner_correct": initial_correct,
        "initial_owner_evaluated": initial_evaluated,
        "initial_owner_accuracy": initial_accuracy,
        "final_owner_correct": final_correct,
        "final_owner_evaluated": final_evaluated,
        "final_owner_accuracy": final_accuracy,
        "n_actions": len(action_rows),
        "n_changed_bins": len(changed),
        "n_net_corrected": net_outcomes.count("corrected"),
        "n_net_damaged": net_outcomes.count("damaged"),
        "n_net_no_gt": net_outcomes.count("no_gt"),
        "n_net_wrong_to_wrong": net_outcomes.count("wrong_to_wrong"),
        "n_net_unresolved": net_outcomes.count("unresolved_owner_mapping"),
        "hit_max_steps": bool(truncated and len(action_rows) >= int(max_steps)),
        "termination_reason": "max_steps" if truncated else "no_delta_above_epsilon",
    }
    for component in OBJECTIVE_COMPONENTS:
        patch_row[f"initial_{component}"] = float(initial_components.get(component, 0.0))
        patch_row[f"final_{component}"] = float(final_components.get(component, 0.0))
        patch_row[f"delta_{component}"] = float(
            final_components.get(component, 0.0) - initial_components.get(component, 0.0)
        )

    cell_rows = _cell_iou_rows(initial_owners, truth, stage="em_initial")
    cell_rows.extend(_cell_iou_rows(final_owners, truth, stage="objective_greedy"))
    for row in cell_rows:
        row.update(
            {
                "w1_scale": float(w1_scale),
                "w1_weight": float(w1_weight),
                "shape_scale": float(shape_scale),
                "shape_weight": float(shape_weight),
                "global_delta_epsilon": float(global_delta_epsilon),
            }
        )
    return GreedyPatchResult(
        patch_row=patch_row,
        action_rows=tuple(action_rows),
        cell_rows=tuple(cell_rows),
    )


def _safe_correlation(x: Iterable[float], y: Iterable[float]) -> float:
    a = np.asarray(tuple(x), dtype=np.float64)
    b = np.asarray(tuple(y), dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(finite)) < 2 or np.std(a[finite]) <= 0.0 or np.std(b[finite]) <= 0.0:
        return np.nan
    return float(np.corrcoef(a[finite], b[finite])[0, 1])


def _summarize_variants(
    patch_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    cells_df: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "w1_scale",
        "w1_weight",
        "shape_scale",
        "shape_weight",
        "global_delta_epsilon",
    ]
    rows: list[dict[str, Any]] = []
    for values, patch_group in patch_df.groupby(keys, sort=True, dropna=False):
        selector = np.ones(len(actions_df), dtype=bool)
        cell_selector = np.ones(len(cells_df), dtype=bool)
        for key, value in zip(keys, values, strict=True):
            if not actions_df.empty:
                selector &= np.isclose(actions_df[key].to_numpy(dtype=float), float(value))
            cell_selector &= np.isclose(cells_df[key].to_numpy(dtype=float), float(value))
        action_group = actions_df.loc[selector] if not actions_df.empty else actions_df
        cell_group = cells_df.loc[cell_selector]
        initial_iou = float(cell_group.loc[cell_group["stage"] == "em_initial", "iou"].mean())
        final_iou = float(cell_group.loc[cell_group["stage"] == "objective_greedy", "iou"].mean())
        n_actions = int(len(action_group))
        outcomes = action_group["outcome"].value_counts().to_dict() if n_actions else {}
        row = {
            **dict(zip(keys, values, strict=True)),
            "n_patches": int(len(patch_group)),
            "n_cells": int(np.sum(cell_group["stage"] == "em_initial")),
            "em_initial_macro_cell_iou": initial_iou,
            "final_macro_cell_iou": final_iou,
            "macro_cell_iou_delta": float(final_iou - initial_iou),
            "initial_owner_accuracy": float(
                patch_group["initial_owner_correct"].sum()
                / max(1, patch_group["initial_owner_evaluated"].sum())
            ),
            "final_owner_accuracy": float(
                patch_group["final_owner_correct"].sum()
                / max(1, patch_group["final_owner_evaluated"].sum())
            ),
            "total_objective_gain": float(patch_group["objective_gain"].sum()),
            "n_actions": n_actions,
            "n_corrected_actions": int(outcomes.get("corrected", 0)),
            "n_damaged_actions": int(outcomes.get("damaged", 0)),
            "n_no_gt_actions": int(outcomes.get("no_gt", 0)),
            "n_wrong_to_wrong_actions": int(outcomes.get("wrong_to_wrong", 0)),
            "n_unresolved_actions": int(outcomes.get("unresolved_owner_mapping", 0)),
            "fraction_no_gt_actions": float(outcomes.get("no_gt", 0) / max(1, n_actions)),
            "n_net_corrected": int(patch_group["n_net_corrected"].sum()),
            "n_net_damaged": int(patch_group["n_net_damaged"].sum()),
            "n_net_no_gt": int(patch_group["n_net_no_gt"].sum()),
            "n_patches_hit_max_steps": int(patch_group["hit_max_steps"].sum()),
            "reward_iou_delta_correlation": _safe_correlation(
                action_group["reward"] if n_actions else (),
                action_group["macro_iou_delta"] if n_actions else (),
            ),
        }
        row["passes_iou_gate"] = bool(row["final_macro_cell_iou"] >= row["em_initial_macro_cell_iou"])
        row["passes_damage_gate"] = bool(row["n_net_corrected"] > row["n_net_damaged"])
        row["passes_no_gt_gate"] = bool(row["fraction_no_gt_actions"] < 0.10)
        row["passes_termination_gate"] = bool(row["n_patches_hit_max_steps"] == 0)
        row["passes_all_gates"] = bool(
            row["passes_iou_gate"]
            and row["passes_damage_gate"]
            and row["passes_no_gt_gate"]
            and row["passes_termination_gate"]
        )
        for component in OBJECTIVE_COMPONENTS:
            row[f"total_delta_{component}"] = float(patch_group[f"delta_{component}"].sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["passes_all_gates", "macro_cell_iou_delta", "n_net_damaged", "n_actions"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def _write_report(
    summary: pd.DataFrame,
    *,
    base_w1_weight: float,
    base_shape_weight: float,
) -> str:
    best = summary.iloc[0].to_dict()
    current = summary.loc[
        np.isclose(summary["w1_scale"], 1.0)
        & np.isclose(summary["shape_scale"], 1.0)
        & np.isclose(summary["global_delta_epsilon"], 1.0e-6)
    ]
    current_row = None if current.empty else current.iloc[0].to_dict()
    lines = [
        "# Objective-greedy patch ablation",
        "",
        "Every trajectory starts from the same EM hard ownership. At each step, the runner applies the existing legal REPLACE with the largest exact global-objective delta. Ground truth is never used to select actions.",
        "",
        f"Base weights: `w1={base_w1_weight:.6g}`, `shape_prior={base_shape_weight:.6g}`.",
        "",
        "## Best observed setting",
        "",
        f"- w1: `{best['w1_weight']:.6g}` (scale `{best['w1_scale']:.3g}`)",
        f"- Shape weight: `{best['shape_weight']:.6g}` (scale `{best['shape_scale']:.3g}`)",
        f"- Global-delta epsilon: `{best['global_delta_epsilon']:.6g}`",
        f"- Macro cell IoU: `{best['em_initial_macro_cell_iou']:.6f}` → `{best['final_macro_cell_iou']:.6f}` (`{best['macro_cell_iou_delta']:+.6f}`)",
        f"- Actions: `{int(best['n_actions'])}`; net corrected/damaged/no-GT: `{int(best['n_net_corrected'])}/{int(best['n_net_damaged'])}/{int(best['n_net_no_gt'])}`",
        f"- Passed all diagnostic gates: `{bool(best['passes_all_gates'])}`",
    ]
    if current_row is not None:
        lines.extend(
            [
                "",
                "## Current objective setting under exact greedy selection",
                "",
                f"- Macro cell IoU: `{current_row['em_initial_macro_cell_iou']:.6f}` → `{current_row['final_macro_cell_iou']:.6f}` (`{current_row['macro_cell_iou_delta']:+.6f}`)",
                f"- Actions: `{int(current_row['n_actions'])}`; net corrected/damaged/no-GT: `{int(current_row['n_net_corrected'])}/{int(current_row['n_net_damaged'])}/{int(current_row['n_net_no_gt'])}`",
                f"- Reward/action-IoU correlation: `{current_row['reward_iou_delta_correlation']:.6f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "- If exact objective-greedy selection reduces IoU, the objective/threshold is misaligned with GT.",
            "- If exact objective-greedy selection improves IoU but PPO does not, policy training/action ranking is the main failure.",
            "- A zero-action setting preserves EM but does not demonstrate useful RL correction.",
            "",
            "See `summary.csv`, `per_patch.csv`, `actions.csv`, and `per_cell.csv` for the complete evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    patch_config_path = Path(args.patch_config).expanduser().resolve()
    patch_debug_dir = Path(args.patch_debug_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    patch_cfg = _load_yaml(patch_config_path)
    base_config_path = Path(str(patch_cfg["base_ppo_config"])).expanduser().resolve()
    base_config = load_ppo_training_config(base_config_path)
    settings = _settings_from_patch_config(patch_cfg, max_steps=int(args.max_steps))
    if not settings.em_assignment.enabled:
        raise ValueError("objective-greedy EM ablation requires em_assignment.enabled=true")
    if settings.agent_mode != "multi_cell_global_delta":
        raise ValueError("objective-greedy ablation currently requires agent_mode=multi_cell_global_delta")
    if settings.after_fill_actions != "replace_only":
        raise ValueError("objective-greedy ablation currently requires after_fill_actions=replace_only")

    shape_scales = _parse_float_list(args.shape_scales, name="shape scales")
    w1_scales = _parse_float_list(args.w1_scales, name="w1 scales")
    epsilons = _parse_float_list(args.global_delta_epsilons, name="global-delta epsilons")
    devices = tuple(torch.device(item.strip()) for item in str(args.devices).split(",") if item.strip())
    if not devices:
        raise ValueError("--devices must contain at least one device")
    for device in devices:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")

    truths = _load_patch_truth(patch_debug_dir)
    patches_df = pd.read_csv(settings.patches_index_path)
    selected = patches_df.loc[patches_df["patch_id"].astype(str).isin(truths)].copy()
    missing = sorted(set(truths) - set(selected["patch_id"].astype(str)))
    if missing:
        raise RuntimeError(f"patch debug IDs missing from patch index: {missing}")

    dataset = PatchDataset(base_config=base_config, settings=settings, rng=np.random.default_rng(base_config.seed))
    contexts: list[PatchContext] = []
    try:
        for row in selected.itertuples(index=False):
            context = dataset.load_patch_context(row)
            if context is None:
                raise RuntimeError(f"failed to load patch context {row.patch_id}")
            contexts.append(context)
    finally:
        dataset.close()
    if not contexts:
        raise RuntimeError("no patch contexts loaded")

    base_w1_weight = float(base_config.w1)
    base_shape_weight = float(base_config.shape_prior_weight)
    patch_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    assignments = [[] for _ in devices]
    for index, context in enumerate(contexts):
        assignments[index % len(devices)].append(context)

    print(
        f"Loaded {len(contexts)} patches; base w1={base_w1_weight:.6g}; "
        f"base shape weight={base_shape_weight:.6g}; running "
        f"{len(w1_scales) * len(shape_scales) * len(epsilons)} variants on "
        f"{len(devices)} device(s).",
        flush=True,
    )
    for w1_scale in w1_scales:
        w1_weight = float(base_w1_weight * w1_scale)
        for shape_scale in shape_scales:
            shape_weight = float(base_shape_weight * shape_scale)
            for epsilon in epsilons:
                def run_device(device_index: int) -> list[GreedyPatchResult]:
                    return [
                        _run_greedy_patch(
                            context=context,
                            truth=truths[context.patch_id],
                            w1_scale=w1_scale,
                            w1_weight=w1_weight,
                            shape_scale=shape_scale,
                            shape_weight=shape_weight,
                            global_delta_epsilon=epsilon,
                            max_steps=int(args.max_steps),
                            device=devices[device_index],
                        )
                        for context in assignments[device_index]
                    ]

                results: list[GreedyPatchResult] = []
                with ThreadPoolExecutor(max_workers=len(devices)) as executor:
                    futures = [
                        executor.submit(run_device, device_index)
                        for device_index, items in enumerate(assignments)
                        if items
                    ]
                    for future in futures:
                        results.extend(future.result())
                for result in results:
                    patch_rows.append(result.patch_row)
                    action_rows.extend(result.action_rows)
                    cell_rows.extend(result.cell_rows)
                print(
                    f"w1={w1_weight:.6g} shape={shape_weight:.6g} "
                    f"epsilon={epsilon:.6g}: "
                    f"actions={sum(result.patch_row['n_actions'] for result in results)}",
                    flush=True,
                )

    patch_df = pd.DataFrame(patch_rows)
    actions_df = pd.DataFrame(action_rows)
    cells_df = pd.DataFrame(cell_rows)
    summary_df = _summarize_variants(patch_df, actions_df, cells_df)
    max_residual = 0.0 if actions_df.empty else float(actions_df["reward_residual"].abs().max())
    if max_residual > 2.0e-5:
        raise RuntimeError(f"objective components do not reconstruct reward: max residual={max_residual}")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / f"objective_greedy_ablation_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    patch_df.to_csv(output_dir / "per_patch.csv", index=False)
    actions_df.to_csv(output_dir / "actions.csv", index=False)
    cells_df.to_csv(output_dir / "per_cell.csv", index=False)
    config = {
        "patch_config": str(patch_config_path),
        "base_config": str(base_config_path),
        "patch_debug_dir": str(patch_debug_dir),
        "patches_index_path": str(settings.patches_index_path),
        "patch_ids": [context.patch_id for context in contexts],
        "base_w1_weight": base_w1_weight,
        "base_shape_weight": base_shape_weight,
        "w1_scales": list(w1_scales),
        "shape_scales": list(shape_scales),
        "global_delta_epsilons": list(epsilons),
        "devices": [str(device) for device in devices],
        "max_steps_safety_cap": int(args.max_steps),
        "max_reward_reconstruction_residual": max_residual,
        "ground_truth_usage": "evaluation_only_after_action_selection",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        _write_report(
            summary_df,
            base_w1_weight=base_w1_weight,
            base_shape_weight=base_shape_weight,
        ),
        encoding="utf-8",
    )
    print(summary_df.to_string(index=False), flush=True)
    print(f"Objective-greedy ablation: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
