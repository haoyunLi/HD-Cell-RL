#!/usr/bin/env python
"""Run a cached, full generalized-EM weight sweep without any RL rollout."""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hd_cell_rl.em_assignment import (  # noqa: E402
    build_sparse_patch_em_input,
    initialize_q_from_nuclear_seeds,
    run_generalized_em,
    save_em_result_npz,
)
from hd_cell_rl.patch_dataset import PatchDataset  # noqa: E402
from hd_cell_rl.patch_assignment import PatchOwnershipMerger, validate_unique_ownership
from hd_cell_rl.ppo_checkpoint import load_checkpoint_payload  # noqa: E402
from hd_cell_rl.ppo_config import load_ppo_training_config  # noqa: E402
from preprocessing.ppo_eval_metrics import compute_spatial_overlap_metrics  # noqa: E402
from preprocessing.ppo_format_assignment_eval import load_eval_cell_ids  # noqa: E402
from scripts.evaluate_patch_initialization import (  # noqa: E402
    _select_patch_rows,
    _settings_from_checkpoint,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument(
        "--episodes-index-path-override",
        default=None,
        help="Optional episode-artifact index override for a matched stress dataset.",
    )
    parser.add_argument(
        "--nuclei-path-override",
        default=None,
        help="Optional nuclei table override paired with the episode index override.",
    )
    parser.add_argument(
        "--reference-npz-path-override",
        default=None,
        help="Optional donor-specific reference-count NPZ for the evaluation dataset.",
    )
    parser.add_argument("--patches-index-path", required=True)
    parser.add_argument("--source-eval-dir", required=True)
    parser.add_argument("--gt-cell-bins-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--spatial-weights", nargs="+", type=float, default=[1.0])
    parser.add_argument(
        "--expression-weights",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
    )
    parser.add_argument(
        "--non-nuclear-bin-filter",
        choices=("all", "positive_expression"),
        default="all",
    )
    parser.add_argument(
        "--background-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--background-owned-logit-intercept", type=float, default=-6.0)
    parser.add_argument("--background-confidence-weight", type=float, default=12.0)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Optional EM-only diagnostic override of the resolved maximum.",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=None,
        help="Optional EM-only diagnostic override of the resolved damping.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--oracle-q-fixed",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--gt-cell-assignments-csv", default=None)
    parser.add_argument("--reference-cell-types-npz", default=None)
    parser.add_argument(
        "--cell-specific-nuclear-profile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--nuclear-profile-prior-umis", type=float, default=500.0)
    parser.add_argument(
        "--alternating-cell-profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Alternate sparse ownership, cell-type posterior, and physical-cell "
            "expression-profile updates."
        ),
    )
    parser.add_argument(
        "--cell-profile-relative-prior-strengths",
        nargs="+",
        type=float,
        default=[1.0],
        help=(
            "Dimensionless scRNA prior strengths relative to the patch median "
            "positive locked-nuclear selected-gene UMI total."
        ),
    )
    parser.add_argument(
        "--profile-update-dampings",
        nargs="+",
        type=float,
        default=[1.0],
        help="Independent physical-cell profile M-step damping values in [0, 1].",
    )
    parser.add_argument(
        "--cell-profile-compatibility-modes",
        nargs="+",
        choices=("standard", "leave_one_out", "spatial_block_crossfit"),
        default=["standard"],
        help="Physical-cell expression compatibility used by alternating EM.",
    )
    parser.add_argument(
        "--cell-profile-reliability-modes",
        nargs="+",
        choices=("nuclear_depth", "nuclear_heldout_predictive"),
        default=["nuclear_depth"],
        help=(
            "Per-cell mixing weight estimator. nuclear_heldout_predictive uses "
            "only leave-one-nuclear-bin-out predictive evidence."
        ),
    )
    parser.add_argument(
        "--heldout-reliability-grid-size",
        type=int,
        default=101,
        help="Number of lambda values in [0, 1] for held-out predictive fitting.",
    )
    parser.add_argument(
        "--spatial-crossfit-block-sizes-um",
        nargs="+",
        type=float,
        default=[8.0],
        help=(
            "Absolute-coordinate spatial block sizes for spatial_block_crossfit. "
            "Each block is one held-out fold; locked nuclear bins are retained."
        ),
    )
    parser.add_argument(
        "--save-soft-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-patch soft-assignment NPZ files for every sweep variant.",
    )
    return parser.parse_args()


def _nuclear_profile_pair_compatibility(
    *,
    em_input: Any,
    bin_expression: sparse.csr_matrix,
    reference_theta: np.ndarray,
    prior_umis: float,
    epsilon: float,
) -> np.ndarray:
    if float(prior_umis) < 0.0:
        raise ValueError("nuclear_profile_prior_umis must be >= 0")
    expression = sparse.csr_matrix(bin_expression, dtype=np.float64)
    theta = np.asarray(reference_theta, dtype=np.float64)
    if expression.shape != (len(em_input.barcode_ids), theta.shape[1]):
        raise ValueError("patch expression and reference theta axes do not align")
    q_seed = initialize_q_from_nuclear_seeds(em_input)
    prior_profile = q_seed @ theta
    prior_profile /= np.maximum(
        np.sum(prior_profile, axis=1, keepdims=True),
        float(epsilon),
    )
    nuclear_counts = np.zeros_like(prior_profile, dtype=np.float64)
    locked_bins = np.flatnonzero(np.asarray(em_input.is_nuclear_locked, dtype=bool))
    locked_owner = np.asarray(em_input.locked_owner_cell_index, dtype=np.int64)
    for cell_idx in range(len(em_input.cell_ids)):
        selected = locked_bins[locked_owner[locked_bins] == int(cell_idx)]
        if selected.size:
            nuclear_counts[cell_idx] = np.asarray(
                expression[selected].sum(axis=0),
                dtype=np.float64,
            ).reshape(-1)
    profile = nuclear_counts + float(prior_umis) * prior_profile
    profile += float(epsilon)
    profile /= np.sum(profile, axis=1, keepdims=True)
    log_profile = np.log(profile)
    ll_by_bin_cell = np.asarray(expression @ log_profile.T, dtype=np.float64)
    totals = np.asarray(expression.sum(axis=1), dtype=np.float64).reshape(-1)
    positive = totals > 0.0
    ll_by_bin_cell[positive] /= totals[positive, None]
    ll_by_bin_cell[~positive] = 0.0
    pair_bin = np.asarray(em_input.pair_bin_index, dtype=np.int64)
    pair_cell = np.asarray(em_input.candidate_cell_index, dtype=np.int64)
    return ll_by_bin_cell[pair_bin, pair_cell]


def _load_oracle_type_context(
    *,
    enabled: bool,
    gt_cell_assignments_csv: str | None,
    reference_cell_types_npz: str | None,
) -> tuple[dict[str, str] | None, tuple[str, ...] | None]:
    if not enabled:
        return None, None
    if gt_cell_assignments_csv is None or reference_cell_types_npz is None:
        raise ValueError(
            "--oracle-q-fixed requires --gt-cell-assignments-csv and "
            "--reference-cell-types-npz"
        )
    assignments = pd.read_csv(Path(gt_cell_assignments_csv).expanduser().resolve())
    required = {"cell_id", "cell_type"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"oracle GT cell assignments missing columns: {sorted(missing)}")
    type_by_cell = {
        str(row.cell_id): str(row.cell_type)
        for row in assignments.itertuples(index=False)
    }
    with np.load(Path(reference_cell_types_npz).expanduser().resolve(), allow_pickle=False) as data:
        if "cell_types" not in data.files:
            raise ValueError("reference NPZ has no cell_types array")
        cell_types = tuple(str(value) for value in data["cell_types"].tolist())
    if len(set(cell_types)) != len(cell_types):
        raise ValueError("reference cell_types must be unique for oracle q")
    return type_by_cell, cell_types


def _oracle_q_for_input(
    *,
    cell_ids: tuple[str, ...],
    type_by_cell: dict[str, str],
    reference_cell_types: tuple[str, ...],
) -> np.ndarray:
    index_by_type = {cell_type: idx for idx, cell_type in enumerate(reference_cell_types)}
    q = np.zeros((len(cell_ids), len(reference_cell_types)), dtype=np.float64)
    for cell_idx, cell_id in enumerate(cell_ids):
        cell_type = type_by_cell.get(str(cell_id))
        if cell_type is None:
            raise ValueError(f"oracle q has no GT type for physical cell {cell_id!r}")
        type_idx = index_by_type.get(str(cell_type))
        if type_idx is None:
            raise ValueError(
                f"oracle GT type {cell_type!r} for cell {cell_id!r} is absent from reference"
            )
        q[cell_idx, type_idx] = 1.0
    return q


def _load_gt(path: Path, cell_ids: set[str]) -> pd.DataFrame:
    columns = ["cell_id", "barcode", "array_row", "array_col", "is_boundary"]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=1_000_000):
        chunk["cell_id"] = chunk["cell_id"].astype(str)
        keep = chunk["cell_id"].isin(cell_ids)
        if bool(keep.any()):
            pieces.append(chunk.loc[keep].copy())
    if not pieces:
        raise ValueError("none of the patch cells occur in dominant-owner GT")
    return pd.concat(pieces, ignore_index=True)


def _parse_barcode_grid(barcode: str, xy: np.ndarray) -> tuple[int, int]:
    parts = str(barcode).replace("-1", "").split("_")
    if len(parts) >= 4:
        try:
            return int(parts[-2]), int(parts[-1])
        except ValueError:
            pass
    return int(round(float(xy[1]) / 2.0)), int(round(float(xy[0]) / 2.0))


def _hard_assignment_rows(
    *,
    context: Any,
    result: Any,
    target_cell_ids: set[str],
) -> list[dict[str, Any]]:
    core = set(str(value) for value in context.core_cell_ids).intersection(target_cell_ids)
    rows: list[dict[str, Any]] = []
    for bin_idx, barcode in enumerate(result.barcode_ids):
        cell_idx = int(result.top1_cell_index[bin_idx])
        if cell_idx < 0:
            continue
        cell_id = str(result.cell_ids[cell_idx])
        if cell_id not in core:
            continue
        xy = np.asarray(result.barcode_xy_um[bin_idx], dtype=np.float64)
        array_row, array_col = _parse_barcode_grid(str(barcode), xy)
        rows.append(
            {
                "barcode": str(barcode),
                "cell_id": cell_id,
                "array_row": array_row,
                "array_col": array_col,
                "x_um": float(xy[0]),
                "y_um": float(xy[1]),
                "is_nuclear": bool(result.is_nuclear_locked[bin_idx]),
                "patch_id": str(context.patch_id),
                "em_top1_probability": float(result.top1_probability[bin_idx]),
                "em_normalized_entropy": float(result.normalized_entropy[bin_idx]),
                "em_background_probability": float(
                    result.background_probability[bin_idx]
                ),
            }
        )
    return rows


def _boundary_from_coords(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    return {
        coord
        for coord in coords
        if any((coord[0] + dr, coord[1] + dc) not in coords for dr, dc in offsets)
    }


def _evaluate_hard_assignments(
    *,
    assignment_rows: list[dict[str, Any]],
    gt: pd.DataFrame,
    target_cell_ids: tuple[str, ...],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    validate_unique_ownership(assignment_rows)
    assignments = pd.DataFrame(assignment_rows, columns=['barcode', 'cell_id', 'array_row', 'array_col'])
    pred_by_cell = {
        str(cell_id): set(group["barcode"].astype(str))
        for cell_id, group in assignments.groupby("cell_id", sort=False)
    }
    gt_by_cell = {
        str(cell_id): set(group["barcode"].astype(str))
        for cell_id, group in gt.groupby("cell_id", sort=False)
    }
    per_cell: list[dict[str, Any]] = []
    for cell_id in target_cell_ids:
        pred = pred_by_cell.get(str(cell_id), set())
        truth = gt_by_cell.get(str(cell_id), set())
        metrics = compute_spatial_overlap_metrics(pred, truth)
        pred_rows = assignments.loc[assignments["cell_id"].astype(str) == str(cell_id)]
        pred_coords = set(
            zip(
                pred_rows["array_row"].astype(int).tolist(),
                pred_rows["array_col"].astype(int).tolist(),
                strict=True,
            )
        )
        gt_rows = gt.loc[gt["cell_id"].astype(str) == str(cell_id)]
        gt_boundary = set(
            zip(
                gt_rows.loc[gt_rows["is_boundary"].astype(int) == 1, "array_row"].astype(int).tolist(),
                gt_rows.loc[gt_rows["is_boundary"].astype(int) == 1, "array_col"].astype(int).tolist(),
                strict=True,
            )
        )
        pred_boundary = _boundary_from_coords(pred_coords)
        boundary_overlap = len(pred_boundary & gt_boundary)
        boundary_precision = boundary_overlap / len(pred_boundary) if pred_boundary else 0.0
        boundary_recall = boundary_overlap / len(gt_boundary) if gt_boundary else 0.0
        boundary_f1 = (
            2.0 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
            if boundary_precision + boundary_recall > 0.0
            else 0.0
        )
        area_bias = (
            (int(metrics["pred_n_bins"]) - int(metrics["gt_n_bins"]))
            / int(metrics["gt_n_bins"])
            if int(metrics["gt_n_bins"]) > 0
            else np.nan
        )
        per_cell.append(
            {
                "cell_id": str(cell_id),
                **metrics,
                "boundary_f1": boundary_f1,
                "area_bias_fraction": area_bias,
            }
        )

    owner_by_barcode = {str(row["barcode"]): str(row["cell_id"]) for row in assignment_rows}
    gt_boundary_rows = gt.loc[
        gt["cell_id"].astype(str).isin(set(target_cell_ids))
        & (gt["is_boundary"].astype(int) == 1)
    ]
    boundary_correct = [
        owner_by_barcode.get(str(row.barcode)) == str(row.cell_id)
        for row in gt_boundary_rows.itertuples(index=False)
    ]
    per_cell_frame = pd.DataFrame(per_cell)
    summary = {
        "mean_core_cell_iou": float(per_cell_frame["iou"].mean()),
        "mean_core_cell_precision": float(per_cell_frame["precision"].mean()),
        "mean_core_cell_recall": float(per_cell_frame["recall"].mean()),
        "mean_boundary_f1": float(per_cell_frame["boundary_f1"].mean()),
        "gt_boundary_owner_accuracy": float(np.mean(boundary_correct)),
        "mean_area_bias_fraction": float(per_cell_frame["area_bias_fraction"].mean()),
        "mean_absolute_area_bias_fraction": float(
            per_cell_frame["area_bias_fraction"].abs().mean()
        ),
    }
    return summary, per_cell


def _variant_name(
    spatial_weight: float,
    expression_weight: float,
    relative_prior_strength: float | None = None,
    profile_update_damping: float | None = None,
    cell_profile_compatibility_mode: str | None = None,
    spatial_crossfit_block_size_um: float | None = None,
    cell_profile_reliability_mode: str | None = None,
    heldout_reliability_grid_size: int | None = None,
) -> str:
    spatial = str(float(spatial_weight)).replace(".", "p")
    expression = str(float(expression_weight)).replace(".", "p")
    name = f"alpha_{spatial}__beta_{expression}"
    if relative_prior_strength is not None:
        prior = str(float(relative_prior_strength)).replace(".", "p")
        name += f"__kappa_{prior}"
    if profile_update_damping is not None:
        update = str(float(profile_update_damping)).replace(".", "p")
        name += f"__profile_damping_{update}"
    if (
        cell_profile_compatibility_mode is not None
        and cell_profile_compatibility_mode != "standard"
    ):
        name += f"__profile_compat_{cell_profile_compatibility_mode}"
    if spatial_crossfit_block_size_um is not None:
        block_size = str(float(spatial_crossfit_block_size_um)).replace(".", "p")
        name += f"__block_um_{block_size}"
    if (
        cell_profile_reliability_mode is not None
        and cell_profile_reliability_mode != "nuclear_depth"
    ):
        name += f"__reliability_{cell_profile_reliability_mode}"
        if heldout_reliability_grid_size is not None:
            name += f"__lambda_grid_{int(heldout_reliability_grid_size)}"
    return name


def main() -> None:
    args = _parse_args()
    if bool(args.alternating_cell_profile) and bool(
        args.cell_specific_nuclear_profile
    ):
        raise ValueError(
            "--alternating-cell-profile and --cell-specific-nuclear-profile "
            "are mutually exclusive"
        )
    if any(
        not np.isfinite(value) or float(value) <= 0.0
        for value in args.cell_profile_relative_prior_strengths
    ):
        raise ValueError(
            "--cell-profile-relative-prior-strengths values must be finite and > 0"
        )
    if any(
        not np.isfinite(value) or not 0.0 <= float(value) <= 1.0
        for value in args.profile_update_dampings
    ):
        raise ValueError("--profile-update-dampings values must be finite and in [0, 1]")
    if (
        bool(args.alternating_cell_profile)
        and any(
            mode in {"leave_one_out", "spatial_block_crossfit"}
            for mode in args.cell_profile_compatibility_modes
        )
        and any(
            not np.isclose(float(value), 1.0)
            for value in args.profile_update_dampings
        )
    ):
        raise ValueError(
            "leave_one_out and spatial_block_crossfit compatibility require "
            "--profile-update-dampings 1"
        )
    if any(
        not np.isfinite(value) or float(value) <= 0.0
        for value in args.spatial_crossfit_block_sizes_um
    ):
        raise ValueError(
            "--spatial-crossfit-block-sizes-um values must be finite and > 0"
        )
    if args.max_iterations is not None and int(args.max_iterations) <= 0:
        raise ValueError("--max-iterations must be > 0")
    if int(args.heldout_reliability_grid_size) < 2:
        raise ValueError("--heldout-reliability-grid-size must be >= 2")
    if args.damping is not None and not (0.0 < float(args.damping) <= 1.0):
        raise ValueError("--damping must be in (0, 1]")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    base_config_path = Path(args.base_config).expanduser().resolve()
    patches_index_path = Path(args.patches_index_path).expanduser().resolve()
    source_eval_dir = Path(args.source_eval_dir).expanduser().resolve()
    gt_path = Path(args.gt_cell_bins_path).expanduser().resolve()
    oracle_type_by_cell, oracle_reference_types = _load_oracle_type_context(
        enabled=bool(args.oracle_q_fixed),
        gt_cell_assignments_csv=args.gt_cell_assignments_csv,
        reference_cell_types_npz=args.reference_cell_types_npz,
    )

    payload = load_checkpoint_payload(checkpoint)
    patch_cfg = dict(payload.get("patch_config") or {})
    settings = _settings_from_checkpoint(
        patch_cfg=patch_cfg,
        patches_index_path=patches_index_path,
        mode="em_only",
        artifact_dir=output_dir / "context_em_not_saved",
    )
    settings = replace(
        settings,
        cache_patch_contexts=True,
        em_assignment=replace(
            settings.em_assignment,
            save_artifacts=False,
            artifact_dir=None,
            debug_patch_ids=(),
            background_enabled=False,
            cell_specific_expression_enabled=False,
        ),
    )
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
    target_ids = tuple(load_eval_cell_ids(source_eval_dir / "per_episode.csv"))
    target_set = set(target_ids)
    patches_df = pd.read_csv(patches_index_path)
    selected_rows = _select_patch_rows(patches_df, target_set)
    if not selected_rows:
        raise RuntimeError("no patches contain target evaluation cells")

    dataset = PatchDataset(
        base_config=config,
        settings=settings,
        rng=np.random.default_rng(int(args.seed)),
    )
    contexts: list[Any] = []
    em_inputs: list[Any] = []
    bin_expressions: list[sparse.csr_matrix] = []
    reference_theta = (
        np.asarray(dataset.reference_theta, dtype=np.float64)
        if bool(args.cell_specific_nuclear_profile)
        or bool(args.alternating_cell_profile)
        else None
    )
    try:
        for patch_index, patch_row in enumerate(selected_rows, start=1):
            context = dataset.load_patch_context(patch_row)
            if context is None:
                continue
            contexts.append(context)
            em_input = build_sparse_patch_em_input(
                context=context,
                candidate_max_distance_um=float(context.candidate_max_distance_um),
                non_nuclear_bin_filter=str(args.non_nuclear_bin_filter),
            )
            em_inputs.append(em_input)
            if bool(args.cell_specific_nuclear_profile) or bool(
                args.alternating_cell_profile
            ):
                bin_expressions.append(
                    dataset.load_unique_patch_expression(
                        context=context,
                        barcode_ids=em_input.barcode_ids,
                    )
                )
            print(f"Loaded {patch_index}/{len(selected_rows)} patches ({context.patch_id})", flush=True)
    finally:
        dataset.close()
    if not contexts:
        raise RuntimeError("no patch contexts loaded")
    fixed_pair_compatibility: list[np.ndarray | None]
    if bool(args.cell_specific_nuclear_profile):
        if reference_theta is None or len(bin_expressions) != len(em_inputs):
            raise RuntimeError("cell-specific nuclear-profile inputs are incomplete")
        fixed_pair_compatibility = [
            _nuclear_profile_pair_compatibility(
                em_input=em_input,
                bin_expression=expression,
                reference_theta=reference_theta,
                prior_umis=float(args.nuclear_profile_prior_umis),
                epsilon=float(settings.em_assignment.epsilon),
            )
            for em_input, expression in zip(em_inputs, bin_expressions, strict=True)
        ]
    else:
        fixed_pair_compatibility = [None] * len(em_inputs)

    all_patch_cell_ids = {str(cell.cell_id) for context in contexts for cell in context.cells}
    gt = _load_gt(gt_path, all_patch_cell_ids)
    truth_owner = {
        str(row.barcode): str(row.cell_id)
        for row in gt.itertuples(index=False)
    }

    sweep_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    relative_prior_strengths: list[float | None] = (
        [float(value) for value in args.cell_profile_relative_prior_strengths]
        if bool(args.alternating_cell_profile)
        else [None]
    )
    profile_update_dampings: list[float | None] = (
        [float(value) for value in args.profile_update_dampings]
        if bool(args.alternating_cell_profile)
        else [None]
    )
    compatibility_settings: list[tuple[str | None, float | None]] = [(None, None)]
    if bool(args.alternating_cell_profile):
        compatibility_settings = []
        for value in args.cell_profile_compatibility_modes:
            mode = str(value)
            if mode == "spatial_block_crossfit":
                compatibility_settings.extend(
                    (mode, float(block_size))
                    for block_size in args.spatial_crossfit_block_sizes_um
                )
            else:
                compatibility_settings.append((mode, None))
    reliability_modes: list[str | None] = (
        [str(value) for value in args.cell_profile_reliability_modes]
        if bool(args.alternating_cell_profile)
        else [None]
    )
    for (
        relative_prior_strength,
        profile_update_damping,
        compatibility_setting,
        cell_profile_reliability_mode,
        spatial_weight,
        expression_weight,
    ) in product(
        relative_prior_strengths,
        profile_update_dampings,
        compatibility_settings,
        reliability_modes,
        args.spatial_weights,
        args.expression_weights,
    ):
                (
                    cell_profile_compatibility_mode,
                    spatial_crossfit_block_size_um,
                ) = compatibility_setting
                variant = _variant_name(
                    spatial_weight,
                    expression_weight,
                    relative_prior_strength,
                    profile_update_damping,
                    cell_profile_compatibility_mode,
                    spatial_crossfit_block_size_um,
                    cell_profile_reliability_mode,
                    (
                        int(args.heldout_reliability_grid_size)
                        if cell_profile_reliability_mode
                        == "nuclear_heldout_predictive"
                        else None
                    ),
                )
                variant_dir = output_dir / "variants" / variant
                artifact_dir = variant_dir / "em_assignment"
                variant_dir.mkdir(parents=True, exist_ok=False)
                if bool(args.save_soft_artifacts):
                    artifact_dir.mkdir()
                assignment_rows: list[dict[str, Any]] = []
                ownership_merger = PatchOwnershipMerger()
                n_bins = 0
                n_unassigned = 0
                n_locked = 0
                n_correct = 0
                analyzed_barcodes = set()
                n_truth_evaluable = 0
                entropy_values: list[np.ndarray] = []
                background_values: list[np.ndarray] = []
                iteration_counts: list[int] = []
                reliability_values: list[np.ndarray] = []
                heldout_gain_values: list[np.ndarray] = []
                heldout_bin_count_values: list[np.ndarray] = []
                relative_prior_counts: list[float] = []
                final_profile_tv_values: list[float] = []
                converged_all = True

                for patch_idx, (context, em_input, pair_compatibility) in enumerate(
                    zip(
                        contexts,
                        em_inputs,
                        fixed_pair_compatibility,
                        strict=True,
                    )
                ):
                    em_config = replace(
                        settings.em_assignment,
                        spatial_weight=float(spatial_weight),
                        expression_weight=float(expression_weight),
                        max_iterations=(
                            int(args.max_iterations)
                            if args.max_iterations is not None
                            else int(settings.em_assignment.max_iterations)
                        ),
                        damping=(
                            float(args.damping)
                            if args.damping is not None
                            else float(settings.em_assignment.damping)
                        ),
                        non_nuclear_bin_filter=str(args.non_nuclear_bin_filter),
                        background_enabled=bool(args.background_enabled),
                        background_owned_logit_intercept=float(
                            args.background_owned_logit_intercept
                        ),
                        background_confidence_weight=float(
                            args.background_confidence_weight
                        ),
                        cell_specific_expression_enabled=bool(
                            args.alternating_cell_profile
                        ),
                        cell_profile_relative_prior_strength=(
                            float(relative_prior_strength)
                            if relative_prior_strength is not None
                            else float(
                                settings.em_assignment.cell_profile_relative_prior_strength
                            )
                        ),
                        cell_profile_update_damping=(
                            float(profile_update_damping)
                            if profile_update_damping is not None
                            else float(settings.em_assignment.cell_profile_update_damping)
                        ),
                        cell_profile_compatibility_mode=(
                            str(cell_profile_compatibility_mode)
                            if cell_profile_compatibility_mode is not None
                            else str(
                                settings.em_assignment.cell_profile_compatibility_mode
                            )
                        ),
                        spatial_crossfit_block_size_um=(
                            float(spatial_crossfit_block_size_um)
                            if spatial_crossfit_block_size_um is not None
                            else float(
                                settings.em_assignment.spatial_crossfit_block_size_um
                            )
                        ),
                        cell_profile_reliability_mode=(
                            str(cell_profile_reliability_mode)
                            if cell_profile_reliability_mode is not None
                            else str(
                                settings.em_assignment.cell_profile_reliability_mode
                            )
                        ),
                        heldout_reliability_grid_size=int(
                            args.heldout_reliability_grid_size
                        ),
                        save_artifacts=False,
                        artifact_dir=None,
                    )
                    oracle_q = (
                        None
                        if oracle_type_by_cell is None
                        or oracle_reference_types is None
                        else _oracle_q_for_input(
                            cell_ids=em_input.cell_ids,
                            type_by_cell=oracle_type_by_cell,
                            reference_cell_types=oracle_reference_types,
                        )
                    )
                    if (
                        oracle_q is not None
                        and oracle_q.shape[1] != em_input.ll.shape[1]
                    ):
                        raise ValueError(
                            "oracle reference cell-type axis does not match EM LL axis"
                        )
                    result = run_generalized_em(
                        em_input,
                        em_config,
                        initial_cell_type_posterior=oracle_q,
                        freeze_cell_type_posterior=bool(args.oracle_q_fixed),
                        fixed_pair_expression_compatibility=pair_compatibility,
                        bin_gene_counts=(
                            bin_expressions[patch_idx]
                            if bool(args.alternating_cell_profile)
                            else None
                        ),
                        reference_theta=(
                            reference_theta
                            if bool(args.alternating_cell_profile)
                            else None
                        ),
                    )
                    if bool(args.save_soft_artifacts):
                        save_em_result_npz(
                            result,
                            artifact_dir / f"{result.patch_id}.em_assignment.npz",
                        )
                    ownership_merger.add_patch(
                        target_cell_ids=[c for c in context.core_cell_ids if c in target_set],
                        context=context, score=0.0, rows=_hard_assignment_rows(
                            context=context,
                            result=result,
                            target_cell_ids=target_set,
                        )
                    )
                    n_bins += int(result.n_bins)
                    analyzed_barcodes.update(map(str, result.barcode_ids))
                    n_unassigned += int(np.sum(result.is_unassigned))
                    n_locked += int(np.sum(result.is_nuclear_locked))
                    entropy_values.append(
                        np.asarray(result.normalized_entropy, dtype=np.float64)
                    )
                    background_values.append(
                        np.asarray(result.background_probability, dtype=np.float64)
                    )
                    iteration_counts.append(len(result.iterations))
                    reliability_values.append(
                        np.asarray(result.cell_profile_reliability, dtype=np.float64)
                    )
                    heldout_gain_values.append(
                        np.asarray(
                            result.cell_profile_reliability_heldout_gain,
                            dtype=np.float64,
                        )
                    )
                    heldout_bin_count_values.append(
                        np.asarray(
                            result.cell_profile_reliability_heldout_nuclear_bins,
                            dtype=np.int64,
                        )
                    )
                    core_cell_ids = set(context.core_cell_ids)
                    for cell_idx, cell_id in enumerate(result.cell_ids):
                        reliability_rows.append(
                            {
                                "variant": variant,
                                "patch_id": str(result.patch_id),
                                "cell_id": str(cell_id),
                                "is_core_cell": str(cell_id) in core_cell_ids,
                                "spatial_weight": float(spatial_weight),
                                "expression_weight": float(expression_weight),
                                "relative_prior_strength": relative_prior_strength,
                                "profile_update_damping": profile_update_damping,
                                "cell_profile_compatibility_mode": (
                                    cell_profile_compatibility_mode
                                ),
                                "cell_profile_reliability_mode": (
                                    cell_profile_reliability_mode
                                ),
                                "heldout_reliability_grid_size": int(
                                    args.heldout_reliability_grid_size
                                ),
                                "nuclear_selected_gene_umi": float(
                                    result.nuclear_expression_count_total[cell_idx]
                                ),
                                "positive_nuclear_bin_count": int(
                                    result.cell_profile_reliability_heldout_nuclear_bins[
                                        cell_idx
                                    ]
                                ),
                                "profile_reliability": float(
                                    result.cell_profile_reliability[cell_idx]
                                ),
                                "heldout_predictive_gain_per_umi": float(
                                    result.cell_profile_reliability_heldout_gain[cell_idx]
                                ),
                            }
                        )
                    relative_prior_counts.append(
                        float(result.relative_profile_prior_count)
                    )
                    final_profile_tv_values.append(
                        float(result.iterations[-1].mean_profile_total_variation)
                        if result.iterations
                        else 0.0
                    )
                    converged_all = converged_all and bool(result.converged)
                    patch_evaluable = 0
                    patch_correct = 0
                    for bin_idx, barcode in enumerate(result.barcode_ids):
                        truth = truth_owner.get(str(barcode))
                        if truth is None:
                            continue
                        patch_evaluable += 1
                        owner_idx = int(result.top1_cell_index[bin_idx])
                        predicted = (
                            str(result.cell_ids[owner_idx])
                            if owner_idx >= 0
                            else "__background__"
                        )
                        patch_correct += int(predicted == truth)
                    n_truth_evaluable += patch_evaluable
                    n_correct += patch_correct
                    patch_rows.append(
                        {
                            "variant": variant,
                            "patch_id": str(result.patch_id),
                            "spatial_weight": float(spatial_weight),
                            "expression_weight": float(expression_weight),
                            "relative_prior_strength": relative_prior_strength,
                            "profile_update_damping": profile_update_damping,
                            "cell_profile_compatibility_mode": (
                                cell_profile_compatibility_mode
                            ),
                            "spatial_crossfit_block_size_um": (
                                spatial_crossfit_block_size_um
                            ),
                            "cell_profile_reliability_mode": (
                                cell_profile_reliability_mode
                            ),
                            "heldout_reliability_grid_size": int(
                                args.heldout_reliability_grid_size
                            ),
                            "n_bins": int(result.n_bins),
                            "n_unassigned": int(np.sum(result.is_unassigned)),
                            "owner_accuracy": patch_correct / patch_evaluable,
                            "n_truth_evaluable": patch_evaluable,
                            "iterations": len(result.iterations),
                            "converged": bool(result.converged),
                            "relative_profile_prior_count": float(
                                result.relative_profile_prior_count
                            ),
                            "mean_cell_profile_reliability": float(
                                np.mean(result.cell_profile_reliability)
                            ),
                            "mean_heldout_predictive_gain_per_umi": float(
                                np.mean(
                                    result.cell_profile_reliability_heldout_gain
                                )
                            ),
                            "final_mean_profile_total_variation": (
                                float(
                                    result.iterations[
                                        -1
                                    ].mean_profile_total_variation
                                )
                                if result.iterations
                                else 0.0
                            ),
                        }
                    )

                assignment_rows, merge_conflicts = ownership_merger.finalize()
                merged_owner = {str(r['barcode']): str(r['cell_id']) for r in assignment_rows}
                core_truth = {b:c for b,c in truth_owner.items() if c in target_set and b in analyzed_barcodes}
                merged_accuracy = (sum(merged_owner.get(b) == c for b,c in core_truth.items()) / len(core_truth)
                                   if core_truth else float('nan'))
                merge_dir = variant_dir / 'subdiagnostic'
                merge_dir.mkdir()
                pd.DataFrame(merge_conflicts, columns=['barcode', 'candidate_cell_id', 'source_patch_id',
                    'source_patch_score', 'nuclear_owner', 'final_owner', 'rule']).to_csv(merge_dir / 'merge_conflicts.csv', index=False)
                pd.DataFrame(ownership_merger.patch_choices).to_csv(merge_dir / 'patch_choices.csv', index=False)
                assignments = pd.DataFrame(assignment_rows)
                assignments.to_csv(variant_dir / "assignments.csv", index=False)
                hard_metrics, per_cell = _evaluate_hard_assignments(
                    assignment_rows=assignment_rows,
                    gt=gt,
                    target_cell_ids=target_ids,
                )
                for row in per_cell:
                    cell_rows.append(
                        {
                            "variant": variant,
                            "spatial_weight": float(spatial_weight),
                            "expression_weight": float(expression_weight),
                            "relative_prior_strength": relative_prior_strength,
                            "profile_update_damping": profile_update_damping,
                            "cell_profile_compatibility_mode": (
                                cell_profile_compatibility_mode
                            ),
                            "spatial_crossfit_block_size_um": (
                                spatial_crossfit_block_size_um
                            ),
                            "cell_profile_reliability_mode": (
                                cell_profile_reliability_mode
                            ),
                            "heldout_reliability_grid_size": int(
                                args.heldout_reliability_grid_size
                            ),
                            **row,
                        }
                    )
                entropy = np.concatenate(entropy_values)
                background = np.concatenate(background_values)
                reliability = np.concatenate(reliability_values)
                heldout_gain = np.concatenate(heldout_gain_values)
                heldout_bin_count = np.concatenate(heldout_bin_count_values)
                heldout_cell_mask = heldout_bin_count >= 2
                sweep_rows.append(
                    {
                        "variant": variant,
                        "spatial_weight": float(spatial_weight),
                        "expression_weight": float(expression_weight),
                        "expression_mode": (
                            "alternating_cell_profile"
                            if bool(args.alternating_cell_profile)
                            else (
                                "fixed_nuclear_profile"
                                if bool(args.cell_specific_nuclear_profile)
                                else "type_posterior"
                            )
                        ),
                        "relative_prior_strength": relative_prior_strength,
                        "profile_update_damping": profile_update_damping,
                        "cell_profile_compatibility_mode": (
                            cell_profile_compatibility_mode
                        ),
                        "spatial_crossfit_block_size_um": (
                            spatial_crossfit_block_size_um
                        ),
                        "cell_profile_reliability_mode": (
                            cell_profile_reliability_mode
                        ),
                        "heldout_reliability_grid_size": int(
                            args.heldout_reliability_grid_size
                        ),
                        "n_bins": n_bins,
                        "n_nuclear_locked": n_locked,
                        "n_unassigned": n_unassigned,
                        "hard_assignment_coverage": 1.0 - n_unassigned / n_bins,
                        "owner_accuracy": merged_accuracy,
                        "n_truth_evaluable": len(core_truth),
                        "owner_accuracy_scope": "unique analyzed bins with core-cell GT, after global merge",
                        "patch_local_owner_accuracy": n_correct / n_truth_evaluable if n_truth_evaluable else float('nan'),
                        "patch_local_n_truth_evaluable": n_truth_evaluable,
                        "entropy_summary_scope": "patch-local bin observations; overlapping patches may repeat bins",
                        "mean_normalized_entropy": float(np.mean(entropy)),
                        "fraction_entropy_above_0p5": float(
                            np.mean(entropy > 0.5)
                        ),
                        "mean_background_probability": float(np.mean(background)),
                        "mean_iterations": float(np.mean(iteration_counts)),
                        "converged_all_patches": converged_all,
                        "mean_cell_profile_reliability": float(
                            np.mean(reliability)
                        ),
                        "median_cell_profile_reliability": float(
                            np.median(reliability)
                        ),
                        "fraction_cell_profile_reliability_zero": float(
                            np.mean(np.isclose(reliability, 0.0))
                        ),
                        "fraction_cell_profile_reliability_one": float(
                            np.mean(np.isclose(reliability, 1.0))
                        ),
                        "n_cells_with_predictive_holdouts": int(
                            np.sum(heldout_cell_mask)
                        ),
                        "mean_heldout_predictive_gain_per_umi": float(
                            np.mean(heldout_gain[heldout_cell_mask])
                            if np.any(heldout_cell_mask)
                            else 0.0
                        ),
                        "mean_relative_profile_prior_count": float(
                            np.mean(relative_prior_counts)
                        ),
                        "mean_final_profile_total_variation": float(
                            np.mean(final_profile_tv_values)
                        ),
                        **hard_metrics,
                    }
                )
                print(
                    f"Completed {variant}: "
                    f"IoU={hard_metrics['mean_core_cell_iou']:.6f}, "
                    f"owner_accuracy={merged_accuracy:.6f}",
                    flush=True,
                )

    sweep = pd.DataFrame(sweep_rows).sort_values(
        [
            "relative_prior_strength",
            "profile_update_damping",
            "cell_profile_compatibility_mode",
            "spatial_crossfit_block_size_um",
            "cell_profile_reliability_mode",
            "spatial_weight",
            "expression_weight",
        ],
        na_position="first",
    )
    sweep.to_csv(output_dir / "weight_sweep.csv", index=False)
    pd.DataFrame(patch_rows).to_csv(output_dir / "per_patch.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(output_dir / "per_cell.csv", index=False)
    pd.DataFrame(reliability_rows).to_csv(
        output_dir / "per_patch_cell_reliability.csv",
        index=False,
    )
    best = sweep.sort_values(
        ["mean_core_cell_iou", "owner_accuracy"], ascending=False
    ).iloc[0]
    summary = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint_protocol": str(checkpoint),
        "base_config": str(base_config_path),
        "episodes_index_path_used": str(config.episodes_index_path),
        "nuclei_path_used": str(config.nuclei_path),
        "reference_path_used": str(config.reference_path),
        "patches_index_path": str(patches_index_path),
        "source_eval_dir": str(source_eval_dir),
        "gt_cell_bins_path": str(gt_path),
        "n_patches": len(contexts),
        "n_target_core_cells": len(target_ids),
        "non_nuclear_bin_filter": str(args.non_nuclear_bin_filter),
        "background": {
            "enabled": bool(args.background_enabled),
            "owned_logit_intercept": float(args.background_owned_logit_intercept),
            "expression_confidence_weight": float(args.background_confidence_weight),
        },
        "full_batch_em_per_weight": True,
        "max_iterations": (
            int(args.max_iterations)
            if args.max_iterations is not None
            else int(settings.em_assignment.max_iterations)
        ),
        "damping": (
            float(args.damping)
            if args.damping is not None
            else float(settings.em_assignment.damping)
        ),
        "oracle_q_fixed": bool(args.oracle_q_fixed),
        "gt_cell_assignments_csv": args.gt_cell_assignments_csv,
        "reference_cell_types_npz": args.reference_cell_types_npz,
        "cell_specific_nuclear_profile": bool(args.cell_specific_nuclear_profile),
        "nuclear_profile_prior_umis": float(args.nuclear_profile_prior_umis),
        "alternating_cell_profile": bool(args.alternating_cell_profile),
        "cell_profile_relative_prior_strengths": [
            float(value) for value in args.cell_profile_relative_prior_strengths
        ],
        "profile_update_dampings": [
            float(value) for value in args.profile_update_dampings
        ],
        "cell_profile_compatibility_modes": [
            str(value) for value in args.cell_profile_compatibility_modes
        ],
        "spatial_crossfit_block_sizes_um": [
            float(value) for value in args.spatial_crossfit_block_sizes_um
        ],
        "cell_profile_reliability_modes": [
            str(value) for value in args.cell_profile_reliability_modes
        ],
        "heldout_reliability_grid_size": int(
            args.heldout_reliability_grid_size
        ),
        "save_soft_artifacts": bool(args.save_soft_artifacts),
        "best_by_validation_mean_core_cell_iou": best.to_dict(),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(sweep.to_string(index=False))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
