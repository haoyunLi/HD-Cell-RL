#!/usr/bin/env python
"""Audit raw REPLACE reward signals at EM-initial patch states.

The runner never uses ground truth to construct EM state or score an action.
It enumerates every legal entropy-eligible REPLACE at the fixed EM initial
state, records raw reward components, and joins pseudo ground truth afterward.
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

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hd_cell_rl.em_types import EMAssignmentConfig
from hd_cell_rl.patch_dataset import PatchDataset
from hd_cell_rl.patch_env_torch import TorchSingleAgentPatchEnv
from hd_cell_rl.patch_types import PatchContext, PatchTrainingSettings
from hd_cell_rl.ppo_config import load_ppo_training_config
from preprocessing.ppo_eval_metrics import normalize_cell_id


SIGNALS = (
    "current_w1_raw",
    "bounded_w1_v2_raw",
    "existing_w5_raw",
    "symmetric_compatibility_raw",
    "distance_raw",
    "overlap_raw",
    "neighbor_raw",
    "shape_raw",
    "current_weighted_raw",
)


@dataclass(frozen=True)
class _CellTransitionTerms:
    w1: Any
    w1_v2: Any
    w5: Any
    distance: Any
    overlap: Any
    neighbor: Any
    shape: Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-config", required=True)
    parser.add_argument("--full-patches-index", required=True)
    parser.add_argument("--gt-cell-bins-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--n-patches", type=int, default=100)
    parser.add_argument("--calibration-patches", type=int, default=60)
    parser.add_argument("--validation-patches", type=int, default=20)
    parser.add_argument("--min-core-cells", type=int, default=5)
    parser.add_argument("--min-center-distance-um", type=float, default=96.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    args = parser.parse_args()
    if int(args.n_patches) <= 0:
        raise ValueError("--n-patches must be > 0")
    if int(args.calibration_patches) < 0 or int(args.validation_patches) < 0:
        raise ValueError("split sizes must be >= 0")
    if int(args.calibration_patches) + int(args.validation_patches) > int(args.n_patches):
        raise ValueError("calibration + validation patches exceed --n-patches")
    if int(args.min_core_cells) <= 0:
        raise ValueError("--min-core-cells must be > 0")
    if float(args.min_center_distance_um) < 0.0:
        raise ValueError("--min-center-distance-um must be >= 0")
    return args


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _settings_from_config(
    patch_cfg: dict[str, Any],
    *,
    selected_index_path: Path,
) -> PatchTrainingSettings:
    patch = dict(patch_cfg.get("patch_training", {}))
    return PatchTrainingSettings(
        patches_index_path=selected_index_path,
        batch_patches=1,
        max_steps_per_patch=int(patch.get("max_steps_per_patch", 1000)),
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


def _select_spatially_separated_patches(
    patches: pd.DataFrame,
    *,
    n_patches: int,
    min_core_cells: int,
    min_center_distance_um: float,
    seed: int,
) -> pd.DataFrame:
    eligible = patches.loc[
        pd.to_numeric(patches["n_core_cells"], errors="coerce").fillna(0) >= int(min_core_cells)
    ].copy()
    if len(eligible) < int(n_patches):
        raise RuntimeError(
            f"only {len(eligible)} patches meet min_core_cells={min_core_cells}"
        )
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(eligible))
    centers = eligible[["center_x_um", "center_y_um"]].to_numpy(dtype=np.float64)
    selected: list[int] = []
    selected_centers: list[np.ndarray] = []
    min_sq = float(min_center_distance_um) ** 2
    for raw_index in order.tolist():
        center = centers[int(raw_index)]
        if selected_centers:
            existing = np.stack(selected_centers, axis=0)
            if bool(np.any(np.sum((existing - center) ** 2, axis=1) < min_sq)):
                continue
        selected.append(int(raw_index))
        selected_centers.append(center)
        if len(selected) >= int(n_patches):
            break
    if len(selected) < int(n_patches):
        raise RuntimeError(
            f"could select only {len(selected)}/{n_patches} patches with "
            f"min_center_distance_um={min_center_distance_um}"
        )
    return eligible.iloc[np.asarray(selected, dtype=np.int64)].reset_index(drop=True)


def _assign_splits(
    selected: pd.DataFrame,
    *,
    n_calibration: int,
    n_validation: int,
) -> pd.DataFrame:
    out = selected.copy()
    n = len(out)
    split = np.full((n,), "test", dtype=object)
    split[: int(n_calibration)] = "calibration"
    split[int(n_calibration) : int(n_calibration) + int(n_validation)] = "validation"
    out.insert(1, "audit_split", split)
    return out


def _reset_state_without_observation(env: TorchSingleAgentPatchEnv) -> None:
    """Reset EM ownership without building or filtering an action observation."""
    torch_module = env._torch
    env._active_cell_idx = 0
    env._membership_masks = [mask.clone() for mask in env._initial_masks]
    env._owner_by_barcode = torch_module.full(
        (len(env._barcode_keys),),
        -1,
        device=env._device,
        dtype=torch_module.long,
    )
    env._step_index = 0
    env._terminated = False
    env._truncated = False
    env._terminated_by_stop = False
    env._cell_rewards = {cell_id: 0.0 for cell_id in env._cell_ids}
    env._stop_reward_value = 0.0
    env._assign_initial_seed_owners()
    env._assign_em_initial_owners()
    env._owned_force_fill_count = env._count_owned_force_fill_barcodes()
    env._rebuild_incremental_state()
    env._last_obs = None


def _posterior_certainty(posterior: Any) -> Any:
    """Return 1 - normalized entropy for each posterior row."""
    if int(posterior.shape[-1]) <= 1:
        return torch.ones(posterior.shape[:-1], device=posterior.device, dtype=posterior.dtype)
    safe = torch.clamp(posterior, min=torch.finfo(posterior.dtype).tiny)
    entropy = -torch.sum(posterior * torch.log(safe), dim=-1)
    out = 1.0 - entropy / float(np.log(int(posterior.shape[-1])))
    return torch.clamp(out, min=0.0, max=1.0)


def _cell_transition_terms(
    env: TorchSingleAgentPatchEnv,
    *,
    cell_idx: int,
    bin_idx: Any,
    add: bool,
) -> _CellTransitionTerms:
    torch_module = env._torch
    idx = bin_idx.to(device=env._device, dtype=torch_module.long)
    ctx = env._ctx.cells[int(cell_idx)]
    if int(idx.numel()) == 0:
        empty = torch_module.empty((0,), device=env._device, dtype=torch_module.float32)
        return _CellTransitionTerms(*(empty for _ in range(7)))

    score = env._score_sums[int(cell_idx)]
    posterior = torch_module.softmax(score, dim=0)
    ll = env._ll64[int(cell_idx)][idx]
    expression_conf = env._expression_conf64[int(cell_idx)][idx]
    current_expression_sum = torch_module.sum(
        env._expression_conf64[int(cell_idx)][env._membership_masks[int(cell_idx)]]
    )
    next_score = score.reshape(1, -1) + ll if add else score.reshape(1, -1) - ll
    next_posterior = torch_module.softmax(next_score, dim=1)
    current_confidence = torch_module.max(posterior)
    next_confidence = torch_module.max(next_posterior, dim=1).values
    if add:
        w1 = next_confidence * (current_expression_sum + expression_conf) - current_confidence * current_expression_sum
    else:
        w1 = next_confidence * torch_module.clamp(
            current_expression_sum - expression_conf,
            min=0.0,
        ) - current_confidence * current_expression_sum

    certainty_delta = _posterior_certainty(next_posterior) - _posterior_certainty(
        posterior.reshape(1, -1)
    )[0]
    w1_v2 = expression_conf * certainty_delta

    mask = env._membership_masks[int(cell_idx)]
    weighted_sum = torch_module.sum(
        env._ll64[int(cell_idx)][mask]
        * env._expression_conf64[int(cell_idx)][mask].reshape(-1, 1),
        dim=0,
    )
    current_w5 = torch_module.sum(posterior * weighted_sum)
    if add:
        next_existing = torch_module.mv(next_posterior, weighted_sum)
        next_candidate = torch_module.sum(ll * next_posterior, dim=1) * expression_conf
        w5 = next_existing + next_candidate - current_w5
    else:
        removed = ll * expression_conf.reshape(-1, 1)
        previous_weighted = weighted_sum.reshape(1, -1) - removed
        previous_w5 = torch_module.sum(next_posterior * previous_weighted, dim=1)
        w5 = previous_w5 - current_w5

    p_dis = torch_module.as_tensor(
        np.asarray(ctx.p_dis, dtype=np.float32)[idx.detach().cpu().numpy()],
        device=env._device,
        dtype=torch_module.float32,
    )
    p_overlap = torch_module.as_tensor(
        np.asarray(ctx.p_overlap, dtype=np.float32)[idx.detach().cpu().numpy()],
        device=env._device,
        dtype=torch_module.float32,
    )
    support = env._compute_neighbor_support(
        cell_idx=int(cell_idx),
        mask=env._membership_masks[int(cell_idx)],
    )[idx].to(dtype=torch_module.float32)
    if add:
        distance = -p_dis
        overlap = -p_overlap
        neighbor = 2.0 * support
        shape = env._compute_shape_raw_delta_for_indices(
            cell_idx=int(cell_idx),
            candidate_idx=idx,
        )
    else:
        distance = p_dis
        overlap = p_overlap
        neighbor = -2.0 * support
        shape = env._compute_shape_raw_delta_after_remove_indices(
            cell_idx=int(cell_idx),
            candidate_idx=idx,
        )
    return _CellTransitionTerms(
        w1=w1.to(dtype=torch_module.float32),
        w1_v2=w1_v2.to(dtype=torch_module.float32),
        w5=w5.to(dtype=torch_module.float32),
        distance=distance.to(dtype=torch_module.float32),
        overlap=overlap.to(dtype=torch_module.float32),
        neighbor=neighbor.to(dtype=torch_module.float32),
        shape=shape.to(dtype=torch_module.float32),
    )


def _grouped_remove_terms(
    env: TorchSingleAgentPatchEnv,
    *,
    old_cells: Any,
    old_bins: Any,
) -> _CellTransitionTerms:
    torch_module = env._torch
    n = int(old_bins.numel())
    outputs = [torch_module.zeros((n,), device=env._device, dtype=torch_module.float32) for _ in range(7)]
    for raw_cell_idx in torch_module.unique(old_cells).detach().cpu().numpy().astype(np.int64).tolist():
        if int(raw_cell_idx) < 0:
            continue
        positions = torch_module.nonzero(old_cells == int(raw_cell_idx), as_tuple=False).flatten()
        terms = _cell_transition_terms(
            env,
            cell_idx=int(raw_cell_idx),
            bin_idx=old_bins[positions],
            add=False,
        )
        for output, values in zip(outputs, terms.__dict__.values(), strict=True):
            output[positions] = values
    return _CellTransitionTerms(*outputs)


def _audit_patch(
    *,
    context: PatchContext,
    split: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if device.type == "cuda":
        torch.cuda.set_device(device)
    env = TorchSingleAgentPatchEnv(context, device=device)
    _reset_state_without_observation(env)
    torch_module = env._torch
    denominator = float(env._score_denominator())
    em_result = context.em_assignment
    em_index = {} if em_result is None else em_result.barcode_to_index()
    rows: list[dict[str, Any]] = []
    max_residual = 0.0

    for target_cell_idx, ctx in enumerate(context.cells):
        mask = env._membership_masks[int(target_cell_idx)]
        assigned_count = int(env._assigned_counts[int(target_cell_idx)])
        support = env._compute_neighbor_support(cell_idx=int(target_cell_idx), mask=mask)
        frontier = ((~mask) & (support > 0.0)) if assigned_count > 0 else (~mask)
        frontier = frontier & env._outer_masks[int(target_cell_idx)]
        barcode_indices = env._barcode_indices[int(target_cell_idx)]
        owners = env._owner_by_barcode[barcode_indices]
        legal = frontier & (owners >= 0) & (owners != int(target_cell_idx))
        legal = legal & env._replace_allowed_by_barcode[barcode_indices]
        replace_idx = torch_module.nonzero(legal, as_tuple=False).flatten().to(dtype=torch_module.long)
        if int(replace_idx.numel()) == 0:
            continue
        old_cells, old_bins = env._owner_cells_and_bins(
            cell_idx=int(target_cell_idx),
            bin_idx=replace_idx,
        )
        valid = (old_cells >= 0) & (old_bins >= 0)
        replace_idx = replace_idx[valid]
        old_cells = old_cells[valid]
        old_bins = old_bins[valid]
        if int(replace_idx.numel()) == 0:
            continue

        add_terms = _cell_transition_terms(
            env,
            cell_idx=int(target_cell_idx),
            bin_idx=replace_idx,
            add=True,
        )
        remove_terms = _grouped_remove_terms(
            env,
            old_cells=old_cells,
            old_bins=old_bins,
        )
        combined = {
            "current_w1_raw": add_terms.w1 + remove_terms.w1,
            "bounded_w1_v2_raw": add_terms.w1_v2 + remove_terms.w1_v2,
            "existing_w5_raw": add_terms.w5 + remove_terms.w5,
            "distance_raw": add_terms.distance + remove_terms.distance,
            "overlap_raw": add_terms.overlap + remove_terms.overlap,
            "neighbor_raw": add_terms.neighbor + remove_terms.neighbor,
            "shape_raw": add_terms.shape + remove_terms.shape,
        }
        target_posterior = torch_module.softmax(env._score_sums[int(target_cell_idx)], dim=0)
        target_compatibility = torch_module.mv(
            env._ll64[int(target_cell_idx)][replace_idx],
            target_posterior,
        )
        old_compatibility = torch_module.zeros_like(target_compatibility)
        for raw_old_cell in torch_module.unique(old_cells).detach().cpu().numpy().astype(np.int64).tolist():
            positions = torch_module.nonzero(old_cells == int(raw_old_cell), as_tuple=False).flatten()
            old_posterior = torch_module.softmax(env._score_sums[int(raw_old_cell)], dim=0)
            old_compatibility[positions] = torch_module.mv(
                env._ll64[int(raw_old_cell)][old_bins[positions]],
                old_posterior,
            )
        expression_confidence = env._expression_conf64[int(target_cell_idx)][replace_idx]
        combined["symmetric_compatibility_raw"] = expression_confidence * (
            target_compatibility - old_compatibility
        )
        combined["current_weighted_raw"] = (
            float(ctx.w1) * combined["current_w1_raw"]
            + float(ctx.w5) * combined["existing_w5_raw"]
            + float(ctx.w2) * combined["distance_raw"]
            + float(ctx.w3) * combined["overlap_raw"]
            + float(ctx.w4) * combined["neighbor_raw"]
            + float(ctx.shape_prior_weight) * combined["shape_raw"]
        )
        exact_raw = env._cell_add_global_delta(
            cell_idx=int(target_cell_idx),
            candidate_idx=replace_idx,
        ) + env._cell_remove_global_delta(
            cell_idx_tensor=old_cells,
            bin_idx_tensor=old_bins,
        )
        residual = exact_raw - combined["current_weighted_raw"]
        if int(residual.numel()) > 0:
            max_residual = max(
                max_residual,
                float(torch_module.max(torch_module.abs(residual)).detach().cpu().item()),
            )

        arrays = {
            key: value.detach().cpu().numpy().astype(np.float64, copy=False)
            for key, value in combined.items()
        }
        replace_np = replace_idx.detach().cpu().numpy().astype(np.int64, copy=False)
        old_cells_np = old_cells.detach().cpu().numpy().astype(np.int64, copy=False)
        old_bins_np = old_bins.detach().cpu().numpy().astype(np.int64, copy=False)
        conf_np = expression_confidence.detach().cpu().numpy().astype(np.float64, copy=False)
        exact_np = exact_raw.detach().cpu().numpy().astype(np.float64, copy=False)
        for pos, target_bin_idx in enumerate(replace_np.tolist()):
            old_cell_idx = int(old_cells_np[pos])
            old_bin_idx = int(old_bins_np[pos])
            barcode = str(ctx.candidate_bin_ids[int(target_bin_idx)])
            em_bin_idx = em_index.get(barcode)
            row = {
                "patch_id": str(context.patch_id),
                "audit_split": str(split),
                "barcode": barcode,
                "old_cell_id": str(context.cells[old_cell_idx].cell_id),
                "new_cell_id": str(ctx.cell_id),
                "old_owner_is_core": str(context.cells[old_cell_idx].cell_id) in set(context.core_cell_ids),
                "new_owner_is_core": str(ctx.cell_id) in set(context.core_cell_ids),
                "expression_confidence": float(conf_np[pos]),
                "old_distance_um": float(
                    np.linalg.norm(
                        np.asarray(context.cells[old_cell_idx].candidate_bin_xy_um[old_bin_idx], dtype=np.float64)
                        - np.asarray(context.cells[old_cell_idx].nucleus_center_xy_um, dtype=np.float64)
                    )
                ),
                "new_distance_um": float(
                    np.linalg.norm(
                        np.asarray(ctx.candidate_bin_xy_um[int(target_bin_idx)], dtype=np.float64)
                        - np.asarray(ctx.nucleus_center_xy_um, dtype=np.float64)
                    )
                ),
                "current_exact_raw": float(exact_np[pos]),
                "current_reward_delta": float(exact_np[pos] / denominator),
                "reconstruction_residual": float(residual[pos].detach().cpu().item()),
            }
            for key in SIGNALS:
                row[key] = float(arrays[key][pos])
            if em_result is not None and em_bin_idx is not None:
                row.update(
                    {
                        "em_top1_probability": float(em_result.top1_probability[int(em_bin_idx)]),
                        "em_margin": float(em_result.margin[int(em_bin_idx)]),
                        "em_normalized_entropy": float(em_result.normalized_entropy[int(em_bin_idx)]),
                    }
                )
            rows.append(row)

    return rows, {
        "patch_id": str(context.patch_id),
        "audit_split": str(split),
        "n_cells": int(context.n_cells),
        "n_core_cells": int(len(context.core_cell_ids)),
        "n_margin_cells": int(len(context.margin_cell_ids)),
        "n_legal_replace_actions": int(len(rows)),
        "max_reconstruction_residual": float(max_residual),
    }


def _load_gt_for_barcodes(path: Path, barcodes: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for chunk in pd.read_csv(
        path,
        usecols=["barcode", "cell_id"],
        dtype={"barcode": "string"},
        chunksize=500_000,
    ):
        subset = chunk.loc[chunk["barcode"].astype(str).isin(barcodes)]
        for row in subset.itertuples(index=False):
            owner = normalize_cell_id(getattr(row, "cell_id"))
            if owner is not None:
                out[str(getattr(row, "barcode"))] = owner
    return out


def _classify_outcome(gt_owner: Any, old_owner: str, new_owner: str) -> str:
    normalized_gt = normalize_cell_id(gt_owner)
    if normalized_gt is None:
        return "no_gt"
    old_correct = normalize_cell_id(old_owner) == normalized_gt
    new_correct = normalize_cell_id(new_owner) == normalized_gt
    if old_correct and not new_correct:
        return "damaged"
    if not old_correct and new_correct:
        return "corrected"
    if old_correct and new_correct:
        return "unchanged_correct"
    return "wrong_to_wrong"


def _binary_auc(values: Iterable[float], labels: Iterable[int]) -> float:
    scores = np.asarray(tuple(values), dtype=np.float64)
    y = np.asarray(tuple(labels), dtype=np.int8)
    finite = np.isfinite(scores)
    scores = scores[finite]
    y = y[finite]
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(scores, method="average")
    rank_sum = float(np.sum(ranks[y == 1]))
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _signal_metrics(actions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ("calibration", "validation", "test", "all"):
        frame = actions if split == "all" else actions.loc[actions["audit_split"] == split]
        binary = frame.loc[frame["outcome"].isin(("corrected", "damaged"))].copy()
        labels = (binary["outcome"] == "corrected").astype(np.int8)
        for signal in SIGNALS:
            values = pd.to_numeric(frame[signal], errors="coerce")
            binary_values = pd.to_numeric(binary[signal], errors="coerce")
            corrected = binary_values.loc[binary["outcome"] == "corrected"]
            damaged = binary_values.loc[binary["outcome"] == "damaged"]
            rows.append(
                {
                    "audit_split": split,
                    "signal": signal,
                    "n_actions": int(len(frame)),
                    "n_corrected": int(len(corrected)),
                    "n_damaged": int(len(damaged)),
                    "n_no_gt": int(np.sum(frame["outcome"] == "no_gt")),
                    "auc_corrected_vs_damaged": _binary_auc(binary_values, labels),
                    "corrected_mean": float(corrected.mean()) if len(corrected) else np.nan,
                    "damaged_mean": float(damaged.mean()) if len(damaged) else np.nan,
                    "median": float(values.median()),
                    "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
                    "q05": float(values.quantile(0.05)),
                    "q95": float(values.quantile(0.95)),
                }
            )
    return pd.DataFrame(rows)


def _owner_recovery_metrics(actions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scored = actions.loc[actions["gt_owner_cell_id"].notna()].copy()
    for split in ("calibration", "validation", "test", "all"):
        frame = scored if split == "all" else scored.loc[scored["audit_split"] == split]
        groups = list(frame.groupby(["patch_id", "barcode"], sort=False))
        for signal in SIGNALS:
            recoverable = 0
            top1_recovered = 0
            initially_correct = 0
            positive_damage = 0
            for _, group in groups:
                old_owner = normalize_cell_id(group.iloc[0]["old_cell_id"])
                gt_owner = normalize_cell_id(group.iloc[0]["gt_owner_cell_id"])
                if old_owner == gt_owner:
                    initially_correct += 1
                    positive_damage += int(bool(np.any(group[signal].to_numpy(dtype=float) > 0.0)))
                    continue
                candidate_correct = group["new_cell_id"].map(normalize_cell_id) == gt_owner
                if not bool(candidate_correct.any()):
                    continue
                recoverable += 1
                best = group.sort_values([signal, "new_cell_id"], ascending=[False, True]).iloc[0]
                top1_recovered += int(normalize_cell_id(best["new_cell_id"]) == gt_owner)
            rows.append(
                {
                    "audit_split": split,
                    "signal": signal,
                    "n_recoverable_wrong_bins": int(recoverable),
                    "top1_recovered_bins": int(top1_recovered),
                    "top1_recovery_fraction": (
                        np.nan if recoverable == 0 else float(top1_recovered / recoverable)
                    ),
                    "n_initially_correct_bins": int(initially_correct),
                    "n_initially_correct_with_positive_damage": int(positive_damage),
                    "positive_damage_fraction": (
                        np.nan if initially_correct == 0 else float(positive_damage / initially_correct)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _outcome_summary(actions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, outcome), group in actions.groupby(["audit_split", "outcome"], sort=True):
        row = {
            "audit_split": split,
            "outcome": outcome,
            "n_actions": int(len(group)),
        }
        for signal in SIGNALS:
            row[f"{signal}_mean"] = float(group[signal].mean())
            row[f"{signal}_median"] = float(group[signal].median())
        rows.append(row)
    return pd.DataFrame(rows)


def _report_text(
    *,
    actions: pd.DataFrame,
    patch_summary: pd.DataFrame,
    signal_metrics: pd.DataFrame,
    recovery: pd.DataFrame,
) -> str:
    all_metrics = signal_metrics.loc[signal_metrics["audit_split"] == "all"].sort_values(
        "auc_corrected_vs_damaged",
        ascending=False,
    )
    test_metrics = signal_metrics.loc[signal_metrics["audit_split"] == "test"].sort_values(
        "auc_corrected_vs_damaged",
        ascending=False,
    )
    test_recovery = recovery.loc[recovery["audit_split"] == "test"].sort_values(
        "top1_recovery_fraction",
        ascending=False,
    )
    outcome_counts = actions["outcome"].value_counts().to_dict()
    best_all = all_metrics.iloc[0]
    best_test = test_metrics.iloc[0]
    best_recovery = test_recovery.iloc[0]
    return "\n".join(
        [
            "# EM-initial REPLACE reward audit",
            "",
            "This audit does not train PPO or reward weights. It scores every legal REPLACE at the fixed EM initial state. Ground truth is joined only after scoring.",
            "",
            "## Dataset",
            "",
            f"- Loaded patches: `{len(patch_summary)}`",
            f"- Legal REPLACE actions: `{len(actions)}`",
            f"- Corrected/damaged/wrong-to-wrong/no-GT: `{int(outcome_counts.get('corrected', 0))}/{int(outcome_counts.get('damaged', 0))}/{int(outcome_counts.get('wrong_to_wrong', 0))}/{int(outcome_counts.get('no_gt', 0))}`",
            f"- Maximum current-objective reconstruction residual: `{patch_summary['max_reconstruction_residual'].max():.3g}`",
            "",
            "## Signal comparison",
            "",
            f"Across all splits, the highest corrected-vs-damaged AUC is `{best_all['signal']}` at `{best_all['auc_corrected_vs_damaged']:.4f}`.",
            f"On the held-out test split, the highest AUC is `{best_test['signal']}` at `{best_test['auc_corrected_vs_damaged']:.4f}`.",
            f"For recoverable EM errors in the test split, the best top-owner signal is `{best_recovery['signal']}` with recovery fraction `{best_recovery['top1_recovery_fraction']:.4f}`.",
            "",
            "AUC measures whether a higher raw signal ranks corrected actions above damaged actions. It does not choose final reward weights. Use the calibration split to set scales, the validation split to choose a formula, and the test split once for the final check.",
            "",
            "See `signal_metrics.csv`, `owner_recovery_metrics.csv`, `outcome_summary.csv`, `patch_summary.csv`, and `actions.parquet` for complete results.",
            "",
        ]
    )


def _worker(
    *,
    rows: list[dict[str, Any]],
    device: torch.device,
    base_config: Any,
    settings: PatchTrainingSettings,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = PatchDataset(
        base_config=base_config,
        settings=settings,
        rng=np.random.default_rng(int(seed)),
    )
    actions: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    try:
        for index, raw in enumerate(rows, start=1):
            context = dataset.load_patch_context(SimpleNamespace(**raw))
            if context is None:
                summaries.append(
                    {
                        "patch_id": str(raw["patch_id"]),
                        "audit_split": str(raw["audit_split"]),
                        "status": "context_unavailable",
                        "n_legal_replace_actions": 0,
                        "max_reconstruction_residual": 0.0,
                    }
                )
                continue
            patch_actions, summary = _audit_patch(
                context=context,
                split=str(raw["audit_split"]),
                device=device,
            )
            summary["status"] = "ok"
            actions.extend(patch_actions)
            summaries.append(summary)
            if index % 5 == 0 or index == len(rows):
                print(
                    f"{device}: audited {index}/{len(rows)} patches, "
                    f"actions={len(actions)}",
                    flush=True,
                )
    finally:
        dataset.close()
    return actions, summaries


def main() -> None:
    args = _parse_args()
    patch_config_path = Path(args.patch_config).expanduser().resolve()
    full_index_path = Path(args.full_patches_index).expanduser().resolve()
    gt_path = Path(args.gt_cell_bins_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    patch_cfg = _load_yaml(patch_config_path)
    base_config_path = Path(str(patch_cfg["base_ppo_config"])).expanduser().resolve()
    base_config = load_ppo_training_config(base_config_path)
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

    patches = pd.read_csv(full_index_path)
    selected = _select_spatially_separated_patches(
        patches,
        n_patches=int(args.n_patches),
        min_core_cells=int(args.min_core_cells),
        min_center_distance_um=float(args.min_center_distance_um),
        seed=int(args.seed),
    )
    selected = _assign_splits(
        selected,
        n_calibration=int(args.calibration_patches),
        n_validation=int(args.validation_patches),
    )
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / f"replace_reward_audit_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    selected_index_path = output_dir / "selected_patches.csv"
    selected.to_csv(selected_index_path, index=False)
    settings = _settings_from_config(
        patch_cfg,
        selected_index_path=selected_index_path,
    )
    if not settings.em_assignment.enabled:
        raise ValueError("reward audit requires em_assignment.enabled=true")
    if settings.agent_mode != "multi_cell_global_delta":
        raise ValueError("reward audit requires agent_mode=multi_cell_global_delta")

    assignments: list[list[dict[str, Any]]] = [[] for _ in devices]
    for index, row in enumerate(selected.to_dict(orient="records")):
        assignments[index % len(devices)].append(row)
    all_actions: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(
                _worker,
                rows=rows,
                device=devices[index],
                base_config=base_config,
                settings=settings,
                seed=int(args.seed) + index,
            )
            for index, rows in enumerate(assignments)
            if rows
        ]
        for future in futures:
            actions, summaries = future.result()
            all_actions.extend(actions)
            all_summaries.extend(summaries)

    actions = pd.DataFrame(all_actions)
    patch_summary = pd.DataFrame(all_summaries)
    if actions.empty:
        raise RuntimeError("audit produced no legal REPLACE actions")
    max_residual = float(actions["reconstruction_residual"].abs().max())
    if max_residual > 2.0e-4:
        raise RuntimeError(
            f"raw components do not reconstruct current objective: {max_residual}"
        )

    gt_by_barcode = _load_gt_for_barcodes(gt_path, set(actions["barcode"].astype(str)))
    actions["gt_owner_cell_id"] = actions["barcode"].map(gt_by_barcode)
    actions["outcome"] = [
        _classify_outcome(gt_owner, old_owner, new_owner)
        for gt_owner, old_owner, new_owner in zip(
            actions["gt_owner_cell_id"],
            actions["old_cell_id"],
            actions["new_cell_id"],
            strict=True,
        )
    ]
    signal_metrics = _signal_metrics(actions)
    recovery = _owner_recovery_metrics(actions)
    outcome_summary = _outcome_summary(actions)

    actions.to_parquet(output_dir / "actions.parquet", index=False)
    patch_summary.to_csv(output_dir / "patch_summary.csv", index=False)
    signal_metrics.to_csv(output_dir / "signal_metrics.csv", index=False)
    recovery.to_csv(output_dir / "owner_recovery_metrics.csv", index=False)
    outcome_summary.to_csv(output_dir / "outcome_summary.csv", index=False)
    config = {
        "patch_config": str(patch_config_path),
        "base_config": str(base_config_path),
        "full_patches_index": str(full_index_path),
        "selected_patches_index": str(selected_index_path),
        "gt_cell_bins_path": str(gt_path),
        "ground_truth_usage": "joined_after_action_scoring",
        "n_requested_patches": int(args.n_patches),
        "split_counts": selected["audit_split"].value_counts().to_dict(),
        "min_core_cells": int(args.min_core_cells),
        "min_center_distance_um": float(args.min_center_distance_um),
        "seed": int(args.seed),
        "devices": [str(device) for device in devices],
        "signals": list(SIGNALS),
        "max_reconstruction_residual": max_residual,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        _report_text(
            actions=actions,
            patch_summary=patch_summary,
            signal_metrics=signal_metrics,
            recovery=recovery,
        ),
        encoding="utf-8",
    )
    print(signal_metrics.loc[signal_metrics["audit_split"].isin(("validation", "test"))].to_string(index=False))
    print(recovery.loc[recovery["audit_split"].isin(("validation", "test"))].to_string(index=False))
    print(f"REPLACE reward audit: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
