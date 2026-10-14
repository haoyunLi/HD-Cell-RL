#!/usr/bin/env python3
"""Quantify radial-growth and GT-derived shape-prior bias in pseudo-HD evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hd_cell_rl.shape_prior import (
    compute_shape_reward,
    load_shape_prior_model,
    shape_features_from_grid_coords,
)


METHOD_TOKENS = {
    "hd_cell_rl_nucleus_only": "nucleus_only",
    "hd_cell_rl_distance_only": "distance_only",
    "hd_cell_rl_em_only": "em_only",
    "hd_cell_rl_em_plus_frozen_rl": "em_plus_frozen_rl",
    "bin2cell": "bin2cell",
    "stcs": "stcs",
    "smurf": "smurf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluated-cells-csv", required=True, type=Path)
    parser.add_argument("--gt-cell-bins", required=True, type=Path)
    parser.add_argument("--nuclei", required=True, type=Path)
    parser.add_argument("--selected-context-bins", required=True, type=Path)
    parser.add_argument("--cell-metadata", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--shape-prior", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--csv-chunk-size", type=int, default=500_000)
    return parser.parse_args()


def selected_gt(path: Path, cell_ids: set[int], chunk_size: int) -> pd.DataFrame:
    columns = ["cell_id", "cell_type", "barcode", "array_row", "array_col"]
    parts = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunk_size):
        current = chunk.loc[chunk["cell_id"].isin(cell_ids)]
        if not current.empty:
            parts.append(current.copy())
    if not parts:
        raise ValueError("No selected evaluated cells were found in GT bins")
    return pd.concat(parts, ignore_index=True)


def parse_barcode_grid(barcodes: pd.Series) -> np.ndarray:
    parts = barcodes.astype(str).str.extract(r"s_002um_(\d+)_(\d+)(?:-\d+)?$")
    if parts.isna().any(axis=None):
        raise ValueError("At least one selected-context barcode is not a square-2um ID")
    row = parts[0].astype(np.int64).to_numpy()
    col = parts[1].astype(np.int64).to_numpy()
    return np.column_stack((col, row))


def iou(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else np.nan


def feature_and_score(coords: np.ndarray, model) -> tuple[np.ndarray, float]:
    raw = shape_features_from_grid_coords(coords)
    score = compute_shape_reward(model.transform_raw_features(raw), model, mode="mixture")
    return raw, float(score)


def summarize(values) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def discover_method_evaluations(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(root.glob("*/per_episode.csv")):
        name = path.parent.name
        for token, method in METHOD_TOKENS.items():
            if token in name:
                if method in found:
                    raise ValueError(f"Duplicate formal evaluation for {method}")
                found[method] = path
                break
    return found


def main() -> None:
    args = parse_args()
    evaluated = pd.read_csv(args.evaluated_cells_csv)
    id_column = "matched_gt_cell_id" if "matched_gt_cell_id" in evaluated else "cell_id"
    cell_ids = (
        pd.to_numeric(evaluated[id_column], errors="coerce")
        .dropna()
        .astype(np.int64)
        .drop_duplicates()
        .tolist()
    )
    cell_id_set = set(cell_ids)
    gt = selected_gt(args.gt_cell_bins, cell_id_set, args.csv_chunk_size)
    gt_by_cell = {int(key): value.copy() for key, value in gt.groupby("cell_id")}

    nuclei = pd.read_parquet(args.nuclei, columns=["cell_id", "center_x_um", "center_y_um"])
    nuclei = nuclei.loc[nuclei["cell_id"].isin(cell_id_set)].set_index("cell_id").loc[cell_ids]
    context = pd.read_parquet(args.selected_context_bins)
    context_grid = parse_barcode_grid(context["barcode"])
    context_xy = context[["x_um", "y_um"]].to_numpy(dtype=np.float64)
    context_barcodes = context["barcode"].astype(str).to_numpy()

    metadata_columns = [
        "cell_id",
        "target_reached",
        "target_cell_area_um2",
        "final_cell_area_um2",
        "available_growth_pixels",
        "final_nuclear_percent",
    ]
    metadata = pd.read_csv(args.cell_metadata, usecols=metadata_columns)
    metadata = metadata.loc[metadata["cell_id"].isin(cell_id_set)].set_index("cell_id").loc[cell_ids]
    model = load_shape_prior_model(args.shape_prior)

    method_paths = discover_method_evaluations(args.evaluation_root)
    method_iou: dict[str, dict[int, float]] = {}
    for method, path in method_paths.items():
        frame = pd.read_csv(path, usecols=["matched_gt_cell_id", "pred_iou"])
        frame = frame.dropna(subset=["matched_gt_cell_id"])
        frame["matched_gt_cell_id"] = frame["matched_gt_cell_id"].astype(np.int64)
        method_iou[method] = dict(
            zip(frame["matched_gt_cell_id"], frame["pred_iou"], strict=False)
        )

    rows = []
    for cell_id in cell_ids:
        current = gt_by_cell[cell_id]
        gt_barcodes = set(current["barcode"].astype(str))
        n_bins = len(gt_barcodes)
        center = nuclei.loc[cell_id, ["center_x_um", "center_y_um"]].to_numpy(
            dtype=np.float64
        )
        distance = np.linalg.norm(context_xy - center[None, :], axis=1)
        nearest = np.argsort(distance, kind="stable")[:n_bins]
        radial_barcodes = set(context_barcodes[nearest].tolist())
        radial_coords = context_grid[nearest]
        gt_coords = current[["array_col", "array_row"]].to_numpy(dtype=np.int64)

        side = int(np.ceil(np.sqrt(max(n_bins, 1))))
        grid_x, grid_y = np.meshgrid(np.arange(side), np.arange(side), indexing="xy")
        square_coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))[:n_bins]
        fragmented_coords = square_coords * 2
        line_coords = np.column_stack((np.arange(n_bins), np.zeros(n_bins, dtype=np.int64)))

        gt_raw, gt_score = feature_and_score(gt_coords, model)
        radial_raw, radial_score = feature_and_score(radial_coords, model)
        fragmented_raw, fragmented_score = feature_and_score(fragmented_coords, model)
        line_raw, line_score = feature_and_score(line_coords, model)
        meta = metadata.loc[cell_id]
        row = {
            "cell_id": cell_id,
            "cell_type": str(current["cell_type"].iloc[0]),
            "n_gt_bins": n_bins,
            "target_reached": bool(meta["target_reached"]),
            "target_cell_area_um2": float(meta["target_cell_area_um2"]),
            "final_cell_area_um2": float(meta["final_cell_area_um2"]),
            "available_growth_pixels": int(meta["available_growth_pixels"]),
            "final_nuclear_percent": float(meta["final_nuclear_percent"]),
            "radial_same_area_iou": iou(gt_barcodes, radial_barcodes),
            "gt_shape_log_likelihood": gt_score,
            "radial_shape_log_likelihood": radial_score,
            "fragmented_shape_log_likelihood": fragmented_score,
            "line_shape_log_likelihood": line_score,
        }
        for prefix, raw in (
            ("gt", gt_raw),
            ("radial", radial_raw),
            ("fragmented", fragmented_raw),
            ("line", line_raw),
        ):
            for feature, value in zip(model.feature_names, raw, strict=False):
                row[f"{prefix}_{feature}"] = float(value)
        for method, values in method_iou.items():
            row[f"iou_{method}"] = values.get(cell_id, np.nan)
        rows.append(row)

    per_cell = pd.DataFrame(rows)
    method_rows = []
    for method in sorted(method_iou):
        column = f"iou_{method}"
        reached = per_cell.loc[per_cell["target_reached"], column]
        blocked = per_cell.loc[~per_cell["target_reached"], column]
        method_rows.append(
            {
                "method": method,
                "n_cells": int(per_cell[column].notna().sum()),
                "mean_iou": float(per_cell[column].mean()),
                "mean_iou_target_reached": float(reached.mean()),
                "mean_iou_target_blocked": float(blocked.mean()),
                "delta_reached_minus_blocked": float(reached.mean() - blocked.mean()),
                "correlation_iou_with_radial_same_area_iou": float(
                    per_cell[[column, "radial_same_area_iou"]].corr().iloc[0, 1]
                ),
            }
        )
    method_summary = pd.DataFrame(method_rows)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_cell_path = output_dir / "geometry_bias_per_cell.csv"
    method_path = output_dir / "method_geometry_strata.csv"
    summary_path = output_dir / "summary.json"
    per_cell.to_csv(per_cell_path, index=False)
    method_summary.to_csv(method_path, index=False)
    summary = {
        "n_evaluated_cells": len(cell_ids),
        "n_target_reached": int(per_cell["target_reached"].sum()),
        "n_target_blocked": int((~per_cell["target_reached"]).sum()),
        "ground_truth_generator": (
            "nearest-nucleus Euclidean expansion until sampled target area/capacity"
        ),
        "radial_same_area_control_uses_gt_area": True,
        "radial_same_area_iou": summarize(per_cell["radial_same_area_iou"]),
        "shape_log_likelihood": {
            "ground_truth": summarize(per_cell["gt_shape_log_likelihood"]),
            "radial_same_area": summarize(per_cell["radial_shape_log_likelihood"]),
            "fragmented_same_area": summarize(
                per_cell["fragmented_shape_log_likelihood"]
            ),
            "line_same_area": summarize(per_cell["line_shape_log_likelihood"]),
        },
        "shape_reference_origin": "older pseudo ground-truth masks",
        "methods_included": sorted(method_iou),
        "per_cell_csv": str(per_cell_path),
        "method_geometry_strata_csv": str(method_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(method_summary.to_string(index=False))


if __name__ == "__main__":
    main()
