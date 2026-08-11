"""Export patch-level assignments, GT overlap, and debug metadata."""

from __future__ import annotations

from collections import defaultdict
import datetime as dt
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from preprocessing.patch_debug_plots import save_patch_overview_plot
from preprocessing.ppo_eval_metrics import normalize_cell_id


PATCH_DEBUG_SCHEMA_VERSION = "1.1"


def export_patch_debug_bundle(
    *,
    patch_assignments_csv: Path,
    patches_index_path: Path,
    per_episode_csv: Path,
    gt_cell_bins_path: Path,
    output_dir: Path,
    patch_ids: Iterable[str] | None = None,
    merge_candidates_csv: Path | None = None,
    trajectory_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    nucleus_centers: Mapping[str, Mapping[str, Iterable[float]]] | None = None,
    source_eval_run_dir: Path | None = None,
    bin_size_um: float = 2.0,
) -> Path:
    """Write one compact JSON file and one overview PNG per evaluated patch."""
    patch_assignments_csv = patch_assignments_csv.expanduser().resolve()
    patches_index_path = patches_index_path.expanduser().resolve()
    per_episode_csv = per_episode_csv.expanduser().resolve()
    gt_cell_bins_path = gt_cell_bins_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    for path in (patch_assignments_csv, patches_index_path, per_episode_csv, gt_cell_bins_path):
        if not path.exists():
            raise FileNotFoundError(path)

    assignments = _load_assignments(patch_assignments_csv)
    per_episode = _load_per_episode(per_episode_csv)
    patch_index = _load_patch_index(patches_index_path)
    selected_patch_ids = _resolve_patch_ids(
        patch_ids=patch_ids,
        assignments=assignments,
        patch_index=patch_index,
    )
    patch_index = patch_index.loc[patch_index["patch_id"].isin(selected_patch_ids)].copy()
    if patch_index.empty:
        raise RuntimeError("no selected patches were found in the patch index")

    merge_scores = _load_merge_scores(merge_candidates_csv)
    gt_rows_by_patch = _load_gt_rows_by_patch(
        gt_cell_bins_path=gt_cell_bins_path,
        patch_index=patch_index,
        per_episode=per_episode,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    patches_dir = output_dir / "patches"
    plots_dir = output_dir / "plots"
    patches_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []
    for patch_id in selected_patch_ids:
        row_match = patch_index.loc[patch_index["patch_id"] == patch_id]
        if row_match.empty:
            continue
        patch_row = row_match.iloc[0]
        payload = _build_patch_payload(
            patch_row=patch_row,
            assignments=assignments.loc[assignments["patch_id"] == patch_id].copy(),
            gt_rows=gt_rows_by_patch.get(patch_id, pd.DataFrame()),
            per_episode=per_episode,
            merge_score=merge_scores.get(patch_id),
            trajectory_metadata=None if trajectory_metadata is None else trajectory_metadata.get(patch_id),
            nucleus_centers=None if nucleus_centers is None else nucleus_centers.get(patch_id),
            bin_size_um=float(bin_size_um),
        )
        slug = _slug(patch_id)
        patch_json_path = patches_dir / f"{slug}.json"
        plot_path = plots_dir / f"{slug}.png"
        _write_json(patch_json_path, payload)
        saved_plot = save_patch_overview_plot(payload=payload, output_path=plot_path)
        manifest_entries.append(
            {
                "patch_id": patch_id,
                "file": f"patches/{patch_json_path.name}",
                "plot": None if saved_plot is None else f"plots/{plot_path.name}",
                "patch_score": payload.get("patch_score"),
                "total_reward": payload.get("total_reward"),
                "n_steps": payload.get("n_steps"),
                "n_core_cells": payload["counts"]["core_cells"],
                "n_predicted_bins": payload["counts"]["predicted_bins"],
                "n_gt_bins": payload["counts"]["gt_bins"],
                "trajectory_available": bool(payload["trajectory"]["available"]),
                "metrics": payload["metrics"],
            }
        )

    manifest = {
        "schema_version": PATCH_DEBUG_SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_eval_run_dir": None if source_eval_run_dir is None else str(source_eval_run_dir.resolve()),
        "source_patch_assignments_csv": str(patch_assignments_csv),
        "source_patch_index": str(patches_index_path),
        "source_per_episode_csv": str(per_episode_csv),
        "source_gt_cell_bins_path": str(gt_cell_bins_path),
        "bin_size_um": float(bin_size_um),
        "n_patches": int(len(manifest_entries)),
        "patches": manifest_entries,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def update_evaluation_summary_with_patch_debug(*, summary_path: Path, manifest_path: Path) -> None:
    """Attach patch-debug artifact counts to an existing evaluation summary."""
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    summary["patch_debug_manifest"] = str(manifest_path)
    summary["n_patch_debug_patches"] = int(manifest.get("n_patches", 0))
    summary["n_patch_debug_plots"] = int(
        sum(item.get("plot") is not None for item in manifest.get("patches", []))
    )
    _write_json(summary_path, summary)


def _load_assignments(path: Path) -> pd.DataFrame:
    required = {
        "patch_id",
        "cell_id",
        "barcode",
        "array_row",
        "array_col",
        "x_um",
        "y_um",
    }
    df = pd.read_csv(path)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"patch assignments missing columns: {sorted(missing)}")
    df = df.copy()
    df["patch_id"] = df["patch_id"].astype(str)
    df["cell_id"] = df["cell_id"].map(normalize_cell_id)
    df["barcode"] = df["barcode"].astype(str)
    df = df.dropna(subset=["cell_id", "barcode", "x_um", "y_um"])
    return df


def _load_per_episode(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "cell_id" not in df.columns:
        raise ValueError("per_episode.csv must contain cell_id")
    df = df.copy()
    for column in ("cell_id", "matched_pred_cell_id", "matched_gt_cell_id"):
        if column in df.columns:
            df[column] = df[column].map(normalize_cell_id)
    return df


def _load_patch_index(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "patch_id",
        "outer_x_min",
        "outer_x_max",
        "outer_y_min",
        "outer_y_max",
        "core_x_min",
        "core_x_max",
        "core_y_min",
        "core_y_max",
        "core_cell_ids",
        "margin_cell_ids",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"patch index missing columns: {sorted(missing)}")
    df = df.copy()
    df["patch_id"] = df["patch_id"].astype(str)
    return df


def _resolve_patch_ids(
    *,
    patch_ids: Iterable[str] | None,
    assignments: pd.DataFrame,
    patch_index: pd.DataFrame,
) -> list[str]:
    if patch_ids is not None:
        ordered = [str(value) for value in patch_ids]
    else:
        ordered = assignments["patch_id"].astype(str).drop_duplicates().tolist()
    available = set(patch_index["patch_id"].astype(str))
    return [patch_id for patch_id in ordered if patch_id in available]


def _load_merge_scores(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return {}
    df = pd.read_csv(resolved)
    if not {"patch_id", "patch_score"}.issubset(df.columns):
        return {}
    out: dict[str, float] = {}
    for patch_id, values in df.groupby(df["patch_id"].astype(str))["patch_score"]:
        numeric = pd.to_numeric(values, errors="coerce")
        numeric = numeric[np.isfinite(numeric.to_numpy(dtype=np.float64))]
        if len(numeric):
            out[str(patch_id)] = float(numeric.iloc[0])
    return out


def _load_gt_rows_by_patch(
    *,
    gt_cell_bins_path: Path,
    patch_index: pd.DataFrame,
    per_episode: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    matched_gt_ids = {
        str(value)
        for value in per_episode.get("matched_gt_cell_id", pd.Series(dtype="object")).dropna().tolist()
    }
    if not matched_gt_ids:
        return {}

    available_columns = pd.read_csv(gt_cell_bins_path, nrows=0).columns.tolist()
    preferred = [
        "barcode",
        "bin_id",
        "array_row",
        "array_col",
        "x_um",
        "y_um",
        "cell_id",
        "cell_type",
        "is_nuclear",
        "weight",
    ]
    usecols = [column for column in preferred if column in available_columns]
    required = {"array_row", "array_col", "x_um", "y_um", "cell_id"}
    if not required.issubset(usecols):
        raise ValueError(f"GT cell bins missing columns: {sorted(required.difference(usecols))}")
    if "barcode" not in usecols and "bin_id" not in usecols:
        raise ValueError("GT cell bins must contain barcode or bin_id")

    bounds = {
        str(row.patch_id): (
            float(row.outer_x_min),
            float(row.outer_x_max),
            float(row.outer_y_min),
            float(row.outer_y_max),
        )
        for row in patch_index.itertuples(index=False)
    }
    x_min = min(value[0] for value in bounds.values())
    x_max = max(value[1] for value in bounds.values())
    y_min = min(value[2] for value in bounds.values())
    y_max = max(value[3] for value in bounds.values())
    collected: dict[str, list[pd.DataFrame]] = defaultdict(list)

    for chunk in pd.read_csv(gt_cell_bins_path, usecols=usecols, chunksize=500_000):
        chunk = chunk.copy()
        chunk["cell_id"] = chunk["cell_id"].map(normalize_cell_id)
        chunk = chunk.loc[chunk["cell_id"].isin(matched_gt_ids)].copy()
        if chunk.empty:
            continue
        chunk = chunk.loc[
            chunk["x_um"].between(x_min, x_max, inclusive="both")
            & chunk["y_um"].between(y_min, y_max, inclusive="both")
        ].copy()
        if chunk.empty:
            continue
        if "barcode" not in chunk.columns:
            chunk["barcode"] = chunk["bin_id"].astype(str)
        else:
            chunk["barcode"] = chunk["barcode"].astype(str)
        for patch_id, (patch_x_min, patch_x_max, patch_y_min, patch_y_max) in bounds.items():
            subset = chunk.loc[
                chunk["x_um"].between(patch_x_min, patch_x_max, inclusive="both")
                & chunk["y_um"].between(patch_y_min, patch_y_max, inclusive="both")
            ].copy()
            if not subset.empty:
                collected[patch_id].append(subset)

    out: dict[str, pd.DataFrame] = {}
    for patch_id, frames in collected.items():
        combined = pd.concat(frames, ignore_index=True)
        if "weight" in combined.columns:
            combined["weight"] = pd.to_numeric(combined["weight"], errors="coerce").fillna(0.0)
            combined = combined.sort_values("weight", ascending=False)
        out[patch_id] = combined.drop_duplicates("barcode", keep="first").reset_index(drop=True)
    return out


def _build_patch_payload(
    *,
    patch_row: pd.Series,
    assignments: pd.DataFrame,
    gt_rows: pd.DataFrame,
    per_episode: pd.DataFrame,
    merge_score: float | None,
    trajectory_metadata: Mapping[str, Any] | None,
    nucleus_centers: Mapping[str, Iterable[float]] | None,
    bin_size_um: float,
) -> dict[str, Any]:
    patch_id = str(patch_row["patch_id"])
    core_cell_ids = _parse_cell_ids(patch_row["core_cell_ids"])
    margin_cell_ids = _parse_cell_ids(patch_row["margin_cell_ids"])
    eval_by_cell = {
        str(row["cell_id"]): row.to_dict()
        for _, row in per_episode.loc[per_episode["cell_id"].isin(core_cell_ids)].iterrows()
    }
    owner_to_gt = _build_owner_to_gt_map(per_episode)

    outer_bounds = {
        "x_min": float(patch_row["outer_x_min"]),
        "x_max": float(patch_row["outer_x_max"]),
        "y_min": float(patch_row["outer_y_min"]),
        "y_max": float(patch_row["outer_y_max"]),
    }
    core_bounds = {
        "x_min": float(patch_row["core_x_min"]),
        "x_max": float(patch_row["core_x_max"]),
        "y_min": float(patch_row["core_y_min"]),
        "y_max": float(patch_row["core_y_max"]),
    }
    assignments = assignments.loc[
        assignments["cell_id"].isin(core_cell_ids)
        & assignments["x_um"].between(outer_bounds["x_min"], outer_bounds["x_max"], inclusive="both")
        & assignments["y_um"].between(outer_bounds["y_min"], outer_bounds["y_max"], inclusive="both")
    ].copy()
    gt_rows = gt_rows.copy()
    metadata = dict(trajectory_metadata or {})
    raw_trajectory = dict(metadata.get("trajectory") or {})
    trajectory = _trajectory_record(raw_trajectory)
    trace_geometry = {
        str(barcode): dict(values)
        for barcode, values in raw_trajectory.get("bin_geometry", {}).items()
    }
    trajectory_final_owners = {
        str(row["barcode"]): str(row["cell_id"])
        for row in trajectory["final_owners"]
        if row.get("barcode") is not None and row.get("cell_id") is not None
    }

    pred_by_barcode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assignments.to_dict(orient="records"):
        pred_by_barcode[str(row["barcode"])].append(row)
    gt_by_barcode = {
        str(row["barcode"]): row
        for row in gt_rows.to_dict(orient="records")
    }

    bin_records: list[dict[str, Any]] = []
    for barcode in sorted(set(pred_by_barcode) | set(gt_by_barcode) | set(trace_geometry)):
        pred_rows = pred_by_barcode.get(barcode, [])
        gt_row = gt_by_barcode.get(barcode)
        trace_row = trace_geometry.get(barcode)
        owner_ids = sorted({str(row["cell_id"]) for row in pred_rows})
        predicted_owner = owner_ids[0] if owner_ids else None
        gt_owner = None if gt_row is None else normalize_cell_id(gt_row.get("cell_id"))
        matched_gt_owner = None if predicted_owner is None else owner_to_gt.get(predicted_owner)
        if pred_rows:
            source_row = pred_rows[0]
        elif gt_row is not None:
            source_row = gt_row
        else:
            source_row = trace_row
        if source_row is None:
            continue
        overlap_category = _overlap_category(
            predicted_owner=predicted_owner,
            predicted_matched_gt=matched_gt_owner,
            gt_owner=gt_owner,
        )
        bin_records.append(
            {
                "barcode": barcode,
                "array_row": int(source_row["array_row"]),
                "array_col": int(source_row["array_col"]),
                "x_um": float(source_row["x_um"]),
                "y_um": float(source_row["y_um"]),
                "predicted_owner_cell_id": predicted_owner,
                "predicted_owner_cell_ids": owner_ids,
                "predicted_matched_gt_cell_id": matched_gt_owner,
                "trajectory_final_owner_cell_id": trajectory_final_owners.get(barcode),
                "gt_owner_cell_id": gt_owner,
                "gt_cell_type": None if gt_row is None else _json_scalar(gt_row.get("cell_type")),
                "gt_is_nuclear": bool(gt_row.get("is_nuclear", False)) if gt_row is not None else False,
                "owner_conflict": bool(len(owner_ids) > 1),
                "overlap_category": overlap_category,
                "trace_only": bool(not pred_rows and gt_row is None),
                "inside_core": bool(
                    core_bounds["x_min"] <= float(source_row["x_um"]) <= core_bounds["x_max"]
                    and core_bounds["y_min"] <= float(source_row["y_um"]) <= core_bounds["y_max"]
                ),
            }
        )

    cells = _build_cell_records(
        core_cell_ids=core_cell_ids,
        assignments=assignments,
        gt_rows=gt_rows,
        eval_by_cell=eval_by_cell,
        owner_to_gt=owner_to_gt,
        nucleus_centers=nucleus_centers,
    )
    metrics = _compute_patch_metrics(bin_records=bin_records, cells=cells)
    patch_score = _finite_or_none(metadata.get("patch_score", merge_score))
    total_reward = _finite_or_none(metadata.get("total_reward"))
    n_steps = _int_or_none(metadata.get("n_steps", metadata.get("n_patch_steps")))
    rollout_metrics = {
        str(key): _json_scalar(value)
        for key, value in dict(metadata.get("metrics", {})).items()
    }

    return {
        "schema_version": PATCH_DEBUG_SCHEMA_VERSION,
        "patch_id": patch_id,
        "bin_size_um": float(bin_size_um),
        "outer_bounds": outer_bounds,
        "core_bounds": core_bounds,
        "patch_score": patch_score,
        "total_reward": total_reward,
        "n_steps": n_steps,
        "metrics": metrics,
        "counts": {
            "core_cells": int(len(core_cell_ids)),
            "margin_cells": int(len(margin_cell_ids)),
            "predicted_bins": int(sum(row["predicted_owner_cell_id"] is not None for row in bin_records)),
            "gt_bins": int(sum(row["gt_owner_cell_id"] is not None for row in bin_records)),
            "display_bins": int(len(bin_records)),
            "trace_only_bins": int(sum(bool(row["trace_only"]) for row in bin_records)),
            "owner_conflicts": int(sum(bool(row["owner_conflict"]) for row in bin_records)),
        },
        "core_cell_ids": core_cell_ids,
        "margin_cell_ids": margin_cell_ids,
        "rollout_metrics": rollout_metrics,
        "trajectory": trajectory,
        "cells": cells,
        "bins": bin_records,
    }


def _build_owner_to_gt_map(per_episode: pd.DataFrame) -> dict[str, str]:
    if "matched_gt_cell_id" not in per_episode.columns:
        return {}
    candidates: dict[str, tuple[float, str]] = {}
    for row in per_episode.to_dict(orient="records"):
        target_cell = normalize_cell_id(row.get("cell_id"))
        pred_owner = normalize_cell_id(row.get("matched_pred_cell_id"))
        gt_owner = normalize_cell_id(row.get("matched_gt_cell_id"))
        if gt_owner is None:
            continue
        score = _finite_or_none(row.get("pred_nuclear_overlap_bins")) or 0.0
        for owner in (pred_owner, target_cell if pred_owner == target_cell else None):
            if owner is None:
                continue
            current = candidates.get(owner)
            if current is None or score > current[0]:
                candidates[owner] = (score, gt_owner)
    return {owner: value[1] for owner, value in candidates.items()}


def _build_cell_records(
    *,
    core_cell_ids: list[str],
    assignments: pd.DataFrame,
    gt_rows: pd.DataFrame,
    eval_by_cell: dict[str, dict[str, Any]],
    owner_to_gt: dict[str, str],
    nucleus_centers: Mapping[str, Iterable[float]] | None,
) -> list[dict[str, Any]]:
    pred_sets = {
        str(cell_id): set(group["barcode"].astype(str))
        for cell_id, group in assignments.groupby("cell_id", sort=False)
    }
    gt_sets = {
        str(cell_id): set(group["barcode"].astype(str))
        for cell_id, group in gt_rows.groupby("cell_id", sort=False)
    }
    records: list[dict[str, Any]] = []
    for cell_id in core_cell_ids:
        eval_row = eval_by_cell.get(cell_id, {})
        matched_gt = owner_to_gt.get(cell_id) or normalize_cell_id(eval_row.get("matched_gt_cell_id"))
        pred = pred_sets.get(cell_id, set())
        gt = gt_sets.get(str(matched_gt), set()) if matched_gt is not None else set()
        intersection = len(pred & gt)
        union = len(pred | gt)
        center = None
        if nucleus_centers is not None and cell_id in nucleus_centers:
            values = list(nucleus_centers[cell_id])
            if len(values) == 2:
                center = [float(values[0]), float(values[1])]
        records.append(
            {
                "cell_id": cell_id,
                "matched_gt_cell_id": matched_gt,
                "gt_cell_type": _json_scalar(eval_row.get("gt_cell_type")),
                "nucleus_center_xy_um": center,
                "predicted_bins": int(len(pred)),
                "gt_bins": int(len(gt)),
                "intersection": int(intersection),
                "union": int(union),
                "patch_iou": _ratio(intersection, union),
                "patch_dice": _ratio(2 * intersection, len(pred) + len(gt)),
                "patch_precision": _ratio(intersection, len(pred)),
                "patch_recall": _ratio(intersection, len(gt)),
                "eval_iou": _finite_or_none(eval_row.get("pred_iou")),
                "eval_dice": _finite_or_none(eval_row.get("pred_dice")),
                "eval_precision": _finite_or_none(eval_row.get("pred_precision")),
                "eval_recall": _finite_or_none(eval_row.get("pred_recall")),
                "gene_spearman_r": _finite_or_none(eval_row.get("gene_spearman_r")),
            }
        )
    records.sort(
        key=lambda row: (
            row["patch_iou"] is None,
            float("inf") if row["patch_iou"] is None else float(row["patch_iou"]),
        )
    )
    return records


def _compute_patch_metrics(*, bin_records: list[dict[str, Any]], cells: list[dict[str, Any]]) -> dict[str, Any]:
    pred_count = sum(row["predicted_owner_cell_id"] is not None for row in bin_records)
    gt_count = sum(row["gt_owner_cell_id"] is not None for row in bin_records)
    foreground_intersection = sum(
        row["predicted_owner_cell_id"] is not None and row["gt_owner_cell_id"] is not None
        for row in bin_records
    )
    foreground_union = pred_count + gt_count - foreground_intersection
    correct = sum(row["overlap_category"] == "correct_owner" for row in bin_records)
    wrong = sum(row["overlap_category"] == "wrong_owner" for row in bin_records)
    unmatched = sum(row["overlap_category"] == "unmatched_owner" for row in bin_records)
    cell_ious = [
        float(row["patch_iou"])
        for row in cells
        if row.get("patch_iou") is not None
    ]
    return {
        "foreground_iou": _ratio(foreground_intersection, foreground_union),
        "foreground_precision": _ratio(foreground_intersection, pred_count),
        "foreground_recall": _ratio(foreground_intersection, gt_count),
        "owner_accuracy": _ratio(correct, foreground_intersection),
        "owner_micro_iou": _ratio(correct, pred_count + gt_count - correct),
        "macro_cell_iou": None if not cell_ious else float(np.mean(cell_ious)),
        "correct_owner_bins": int(correct),
        "wrong_owner_bins": int(wrong),
        "unmatched_owner_bins": int(unmatched),
        "pred_only_bins": int(sum(row["overlap_category"] == "pred_only" for row in bin_records)),
        "gt_only_bins": int(sum(row["overlap_category"] == "gt_only" for row in bin_records)),
        "foreground_intersection_bins": int(foreground_intersection),
        "foreground_union_bins": int(foreground_union),
    }


def _overlap_category(
    *,
    predicted_owner: str | None,
    predicted_matched_gt: str | None,
    gt_owner: str | None,
) -> str:
    if predicted_owner is None and gt_owner is None:
        return "unscored"
    if predicted_owner is None:
        return "gt_only"
    if gt_owner is None:
        return "pred_only"
    if predicted_matched_gt is None:
        return "unmatched_owner"
    return "correct_owner" if predicted_matched_gt == gt_owner else "wrong_owner"


def _trajectory_record(value: Any) -> dict[str, Any]:
    raw = dict(value or {})
    available = bool(raw.get("available", False))
    return {
        "available": available,
        "capture_status": "exact" if available else "final_only",
        "initial_patch_score": _finite_or_none(raw.get("initial_patch_score")),
        "initial_raw_patch_score": _finite_or_none(raw.get("initial_raw_patch_score")),
        "initial_owned_target_count": _int_or_none(raw.get("initial_owned_target_count")),
        "target_count": _int_or_none(raw.get("target_count")),
        "initial_owners": list(raw.get("initial_owners", [])) if available else [],
        "final_owners": list(raw.get("final_owners", [])) if available else [],
        "steps": list(raw.get("steps", [])) if available else [],
    }


def _parse_cell_ids(value: Any) -> list[str]:
    raw = json.loads(value) if isinstance(value, str) else value
    out: list[str] = []
    for item in raw:
        normalized = normalize_cell_id(item)
        if normalized is not None:
            out.append(normalized)
    return out


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if float(denominator) <= 0.0:
        return None
    return float(numerator) / float(denominator)


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _int_or_none(value: Any) -> int | None:
    number = _finite_or_none(value)
    return None if number is None else int(number)


def _json_scalar(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _finite_or_none(value)
    return str(value)


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text[:100] if text else "patch"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, allow_nan=False)
        handle.write("\n")
