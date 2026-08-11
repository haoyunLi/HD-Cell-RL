#!/usr/bin/env python
"""Build GT-derived shape reference features from per-cell bin assignments."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.spatial import ConvexHull, QhullError
except Exception:  # pragma: no cover - exercised only when scipy is unavailable.
    ConvexHull = None  # type: ignore[assignment]
    QhullError = Exception  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)
DEFAULT_EPSILON = 1e-8
SHAPE_FEATURE_NAMES = ("log_area", "compactness", "solidity", "anisotropy")


def configure_logging(verbose: bool = False) -> None:
    """Configure process logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def build_shape_reference_features(
    gt_cell_bins_path: str | Path,
    *,
    gt_cell_assignments_csv: str | Path | None = None,
    cell_id_column: str = "cell_id",
    cell_type_column: str = "cell_type",
    x_column: str | None = None,
    y_column: str | None = None,
    bin_size_um: float = 2.0,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute per-cell GT shape features and global z-score normalization.

    Area and perimeter are measured in grid units: area is the number of unique
    assigned bins/pixels and perimeter is the number of exposed 4-neighbor edges.
    """
    if bin_size_um <= 0:
        raise ValueError("bin_size_um must be > 0")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")

    bins_df, coord_cols = _load_gt_cell_bins(
        gt_cell_bins_path,
        cell_id_column=cell_id_column,
        x_column=x_column,
        y_column=y_column,
    )
    x_col, y_col = coord_cols
    LOGGER.info("Loaded %d GT cell-bin rows using coordinates (%s, %s)", len(bins_df), x_col, y_col)

    if bins_df.empty:
        raise ValueError("GT cell bins table is empty after loading required columns")

    features = _compute_features_for_cells(
        bins_df,
        cell_id_column="cell_id",
        x_column=x_col,
        y_column=y_col,
        bin_size_um=float(bin_size_um),
        epsilon=float(epsilon),
    )
    features = _attach_cell_annotations(
        features,
        bins_df=bins_df,
        gt_cell_assignments_csv=gt_cell_assignments_csv,
        original_cell_id_column=cell_id_column,
        cell_type_column=cell_type_column,
    )
    features, normalization = _add_zscore_features(features, feature_names=SHAPE_FEATURE_NAMES, epsilon=float(epsilon))
    cell_type_summary = _build_cell_type_summary(features, feature_names=SHAPE_FEATURE_NAMES)

    summary: dict[str, Any] = {
        "gt_cell_bins_path": str(Path(gt_cell_bins_path).expanduser().resolve()),
        "gt_cell_assignments_csv": None
        if gt_cell_assignments_csv is None
        else str(Path(gt_cell_assignments_csv).expanduser().resolve()),
        "cell_id_column": cell_id_column,
        "cell_type_column": cell_type_column,
        "x_column": x_col,
        "y_column": y_col,
        "bin_size_um": float(bin_size_um),
        "epsilon": float(epsilon),
        "n_input_rows": int(len(bins_df)),
        "n_cells": int(len(features)),
        "n_cells_with_cell_type": int(features["cell_type"].notna().sum()) if "cell_type" in features.columns else 0,
        "feature_names": list(SHAPE_FEATURE_NAMES),
        "normalization": normalization,
    }
    return features, cell_type_summary, summary


def write_shape_reference_outputs(
    *,
    per_cell_df: pd.DataFrame,
    cell_type_summary_df: pd.DataFrame,
    summary: dict[str, Any],
    per_cell_output_path: str | Path,
    cell_type_summary_output_path: str | Path,
    npz_output_path: str | Path,
    summary_output_path: str | Path,
) -> dict[str, Path]:
    """Write shape-reference CSV/NPZ/JSON outputs."""
    per_cell_path = Path(per_cell_output_path)
    cell_type_path = Path(cell_type_summary_output_path)
    npz_path = Path(npz_output_path)
    summary_path = Path(summary_output_path)
    for path in (per_cell_path, cell_type_path, npz_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    _write_table(per_cell_df, per_cell_path)
    _write_table(cell_type_summary_df, cell_type_path)

    raw_feature_cols = list(SHAPE_FEATURE_NAMES)
    z_feature_cols = [f"{name}_z" for name in SHAPE_FEATURE_NAMES]
    cell_types = (
        per_cell_df["cell_type"].fillna("").astype(str).to_numpy(dtype="U")
        if "cell_type" in per_cell_df.columns
        else np.asarray([""] * len(per_cell_df), dtype="U")
    )
    if cell_type_summary_df.empty:
        cell_type_labels = np.asarray([], dtype="U")
        cell_type_n_cells = np.asarray([], dtype=np.int64)
        cell_type_feature_means = np.zeros((0, len(raw_feature_cols)), dtype=np.float64)
        cell_type_feature_stds = np.zeros((0, len(raw_feature_cols)), dtype=np.float64)
        cell_type_feature_medians = np.zeros((0, len(raw_feature_cols)), dtype=np.float64)
    else:
        cell_type_labels = cell_type_summary_df["cell_type"].astype(str).to_numpy(dtype="U")
        cell_type_n_cells = cell_type_summary_df["n_cells"].to_numpy(dtype=np.int64)
        cell_type_feature_means = cell_type_summary_df.loc[
            :, [f"{name}_mean" for name in raw_feature_cols]
        ].to_numpy(dtype=np.float64)
        cell_type_feature_stds = cell_type_summary_df.loc[
            :, [f"{name}_std" for name in raw_feature_cols]
        ].to_numpy(dtype=np.float64)
        cell_type_feature_medians = cell_type_summary_df.loc[
            :, [f"{name}_median" for name in raw_feature_cols]
        ].to_numpy(dtype=np.float64)
    np.savez_compressed(
        npz_path,
        cell_ids=per_cell_df["cell_id"].astype(str).to_numpy(dtype="U"),
        cell_types=cell_types,
        feature_names=np.asarray(raw_feature_cols, dtype="U"),
        shape_features=per_cell_df.loc[:, raw_feature_cols].to_numpy(dtype=np.float64),
        shape_features_zscore=per_cell_df.loc[:, z_feature_cols].to_numpy(dtype=np.float64),
        cell_type_labels=cell_type_labels,
        cell_type_n_cells=cell_type_n_cells,
        cell_type_feature_means=cell_type_feature_means,
        cell_type_feature_stds=cell_type_feature_stds,
        cell_type_feature_medians=cell_type_feature_medians,
        area=per_cell_df["area"].to_numpy(dtype=np.float64),
        perimeter=per_cell_df["perimeter"].to_numpy(dtype=np.float64),
        hull_area=per_cell_df["convex_hull_area"].to_numpy(dtype=np.float64),
    )

    payload = dict(summary)
    if not cell_type_summary_df.empty:
        payload["cell_type_counts"] = {
            str(row.cell_type): int(row.n_cells) for row in cell_type_summary_df.itertuples(index=False)
        }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    LOGGER.info("Wrote per-cell shape features: %s", per_cell_path)
    LOGGER.info("Wrote cell-type shape summary: %s", cell_type_path)
    LOGGER.info("Wrote shape-reference NPZ: %s", npz_path)
    LOGGER.info("Wrote summary JSON: %s", summary_path)
    return {
        "per_cell": per_cell_path,
        "cell_type_summary": cell_type_path,
        "npz": npz_path,
        "summary": summary_path,
    }


def _load_gt_cell_bins(
    path: str | Path,
    *,
    cell_id_column: str,
    x_column: str | None,
    y_column: str | None,
) -> tuple[pd.DataFrame, tuple[str, str]]:
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"GT cell bins file not found: {table_path}")

    available_columns = _read_columns(table_path)
    if cell_id_column not in available_columns:
        raise ValueError(f"GT cell bins table is missing cell_id column {cell_id_column!r}")
    x_col, y_col = _resolve_coordinate_columns(available_columns, x_column=x_column, y_column=y_column)

    usecols = [cell_id_column, x_col, y_col]
    if "cell_type" in available_columns:
        usecols.append("cell_type")
    if "barcode" in available_columns:
        usecols.append("barcode")

    df = _read_table(table_path, columns=usecols)
    df = df.dropna(subset=[cell_id_column, x_col, y_col]).copy()
    df["cell_id"] = df[cell_id_column].map(_normalize_cell_id)
    df = df.loc[df["cell_id"].notna()].copy()
    if df.empty:
        return df, (x_col, y_col)

    df[x_col] = pd.to_numeric(df[x_col], errors="raise").round().astype(np.int64)
    df[y_col] = pd.to_numeric(df[y_col], errors="raise").round().astype(np.int64)
    keep_cols = ["cell_id", x_col, y_col]
    if "cell_type" in df.columns:
        keep_cols.append("cell_type")
    if "barcode" in df.columns:
        keep_cols.append("barcode")
    df = df.loc[:, keep_cols].drop_duplicates(subset=["cell_id", x_col, y_col]).reset_index(drop=True)
    return df, (x_col, y_col)


def _resolve_coordinate_columns(
    columns: set[str],
    *,
    x_column: str | None,
    y_column: str | None,
) -> tuple[str, str]:
    if x_column is not None or y_column is not None:
        if x_column is None or y_column is None:
            raise ValueError("x_column and y_column must be provided together")
        missing = [col for col in (x_column, y_column) if col not in columns]
        if missing:
            raise ValueError(f"coordinate columns missing from GT cell bins table: {missing}")
        return str(x_column), str(y_column)

    candidates = (
        ("array_col", "array_row"),
        ("bin_x_index", "bin_y_index"),
        ("x", "y"),
        ("col", "row"),
    )
    for x_col, y_col in candidates:
        if x_col in columns and y_col in columns:
            return x_col, y_col
    raise ValueError(
        "could not infer integer grid coordinate columns; provide --x-column and --y-column "
        "or include array_col/array_row, bin_x_index/bin_y_index, or x/y"
    )


def _compute_features_for_cells(
    df: pd.DataFrame,
    *,
    cell_id_column: str,
    x_column: str,
    y_column: str,
    bin_size_um: float,
    epsilon: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cell_id, group in df.groupby(cell_id_column, sort=False):
        coords = group.loc[:, [x_column, y_column]].to_numpy(dtype=np.int64, copy=True)
        if coords.shape[0] == 0:
            continue
        area = int(coords.shape[0])
        perimeter = int(_count_exposed_edges(coords))
        compactness = float((4.0 * math.pi * float(area)) / max(float(perimeter * perimeter), epsilon))
        convex_hull_area = float(_convex_hull_area_for_grid_cells(coords, epsilon=epsilon))
        solidity = float(area / convex_hull_area) if convex_hull_area > epsilon else 1.0
        anisotropy = float(_anisotropy(coords, epsilon=epsilon))

        rows.append(
            {
                "cell_id": str(cell_id),
                "area": area,
                "log_area": float(math.log(float(area) + 1.0)),
                "perimeter": perimeter,
                "perimeter_um": float(perimeter * bin_size_um),
                "compactness": compactness,
                "convex_hull_area": convex_hull_area,
                "solidity": solidity,
                "anisotropy": anisotropy,
                "centroid_x": float(np.mean(coords[:, 0])),
                "centroid_y": float(np.mean(coords[:, 1])),
            }
        )
    if not rows:
        raise ValueError("no per-cell shape features could be computed")
    return pd.DataFrame(rows)


def _count_exposed_edges(coords: np.ndarray) -> int:
    occupied = {(int(x), int(y)) for x, y in coords.tolist()}
    exposed = 0
    for x, y in occupied:
        if (x + 1, y) not in occupied:
            exposed += 1
        if (x - 1, y) not in occupied:
            exposed += 1
        if (x, y + 1) not in occupied:
            exposed += 1
        if (x, y - 1) not in occupied:
            exposed += 1
    return int(exposed)


def _convex_hull_area_for_grid_cells(coords: np.ndarray, *, epsilon: float) -> float:
    area = float(coords.shape[0])
    if coords.shape[0] <= 2 or ConvexHull is None:
        return area

    corners = np.vstack(
        (
            coords,
            coords + np.asarray([1, 0], dtype=np.int64),
            coords + np.asarray([0, 1], dtype=np.int64),
            coords + np.asarray([1, 1], dtype=np.int64),
        )
    ).astype(np.float64)
    corners = np.unique(corners, axis=0)
    if corners.shape[0] < 3:
        return area

    try:
        hull = ConvexHull(corners)
        hull_area = float(hull.volume)
    except (QhullError, ValueError):
        return area
    if not np.isfinite(hull_area) or hull_area <= epsilon:
        return area
    return float(max(hull_area, area))


def _anisotropy(coords: np.ndarray, *, epsilon: float) -> float:
    if coords.shape[0] < 3:
        return 1.0
    values = coords.astype(np.float64, copy=False)
    cov = np.cov(values, rowvar=False, bias=True)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return 1.0
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.asarray(eigvals, dtype=np.float64))[::-1]
    lambda1 = float(max(eigvals[0], 0.0))
    lambda2 = float(max(eigvals[1], 0.0))
    if lambda1 <= epsilon:
        return 1.0
    return float(lambda1 / max(lambda2, epsilon))


def _attach_cell_annotations(
    features: pd.DataFrame,
    *,
    bins_df: pd.DataFrame,
    gt_cell_assignments_csv: str | Path | None,
    original_cell_id_column: str,
    cell_type_column: str,
) -> pd.DataFrame:
    annotated = features.copy()
    annotations: pd.DataFrame | None = None
    if gt_cell_assignments_csv is not None:
        assignments_path = Path(gt_cell_assignments_csv)
        if not assignments_path.exists():
            raise FileNotFoundError(f"GT cell assignments CSV not found: {assignments_path}")
        assignments = pd.read_csv(assignments_path)
        if original_cell_id_column not in assignments.columns:
            raise ValueError(f"GT assignments table is missing cell ID column {original_cell_id_column!r}")
        annotations = pd.DataFrame(
            {
                "cell_id": assignments[original_cell_id_column].map(_normalize_cell_id),
            }
        )
        for col in (cell_type_column, "sc_cell_barcode"):
            if col in assignments.columns:
                annotations[col] = assignments[col]
        annotations = annotations.loc[annotations["cell_id"].notna()].drop_duplicates(subset=["cell_id"], keep="first")

    if annotations is None and "cell_type" in bins_df.columns:
        annotations = (
            bins_df.loc[:, ["cell_id", "cell_type"]]
            .dropna(subset=["cell_type"])
            .drop_duplicates(subset=["cell_id"], keep="first")
        )

    if annotations is not None and not annotations.empty:
        annotated = annotated.merge(annotations, on="cell_id", how="left", validate="one_to_one")
    if "cell_type" not in annotated.columns:
        annotated["cell_type"] = pd.NA
    return annotated


def _add_zscore_features(
    df: pd.DataFrame,
    *,
    feature_names: tuple[str, ...],
    epsilon: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    out = df.copy()
    normalization: dict[str, dict[str, float]] = {}
    for name in feature_names:
        values = out[name].to_numpy(dtype=np.float64)
        mean = float(np.mean(values))
        std_raw = float(np.std(values, ddof=0))
        std_used = float(std_raw if std_raw > epsilon else epsilon)
        out[f"{name}_z"] = (values - mean) / std_used
        normalization[name] = {"mean": mean, "std": std_raw, "std_used": std_used}
    return out, normalization


def _build_cell_type_summary(df: pd.DataFrame, *, feature_names: tuple[str, ...]) -> pd.DataFrame:
    if "cell_type" not in df.columns:
        return pd.DataFrame()
    typed = df.dropna(subset=["cell_type"]).copy()
    typed = typed.loc[typed["cell_type"].astype(str).str.len() > 0]
    if typed.empty:
        return pd.DataFrame(columns=["cell_type", "n_cells"])

    rows: list[dict[str, Any]] = []
    for cell_type, group in typed.groupby("cell_type", sort=True):
        row: dict[str, Any] = {"cell_type": str(cell_type), "n_cells": int(len(group))}
        for name in feature_names:
            values = group[name].to_numpy(dtype=np.float64)
            row[f"{name}_mean"] = float(np.mean(values))
            row[f"{name}_std"] = float(np.std(values, ddof=0))
            row[f"{name}_median"] = float(np.median(values))
        rows.append(row)
    return pd.DataFrame(rows)


def _read_columns(path: Path) -> set[str]:
    if path.suffix == ".parquet":
        return set(pd.read_parquet(path, engine="auto").columns)
    return set(pd.read_csv(path, compression="infer", nrows=0).columns)


def _read_table(path: Path, *, columns: list[str]) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns, compression="infer", low_memory=False)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    suffixes = path.suffixes
    if suffixes[-2:] == [".csv", ".gz"]:
        df.to_csv(path, index=False, compression="gzip")
    elif suffixes[-1:] == [".csv"]:
        df.to_csv(path, index=False)
    elif suffixes[-1:] == [".parquet"]:
        df.to_parquet(path, index=False)
    else:
        raise ValueError("table output path must end with .csv, .csv.gz, or .parquet")


def _normalize_cell_id(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        prefix = text[:-2]
        if prefix.lstrip("+-").isdigit():
            return prefix
    return text


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build GT shape reference features from per-cell assigned bins")
    parser.add_argument("--gt-cell-bins-path", required=True, help="GT whole-cell bin table, usually pseudo square_002um cell bins")
    parser.add_argument("--gt-cell-assignments-csv", default=None, help="Optional ground_truth_cell_assignments.csv with cell_type labels")
    parser.add_argument("--per-cell-output-path", required=True, help="Output per-cell shape feature table (.csv/.csv.gz/.parquet)")
    parser.add_argument("--cell-type-summary-output-path", required=True, help="Output per-cell-type shape summary table")
    parser.add_argument("--npz-output-path", required=True, help="Output compressed NPZ with raw/z-scored shape feature matrices")
    parser.add_argument("--summary-path", required=True, help="Output JSON summary")
    parser.add_argument("--cell-id-column", default="cell_id")
    parser.add_argument("--cell-type-column", default="cell_type")
    parser.add_argument("--x-column", default=None, help="Optional integer grid x column; default infers array_col/bin_x_index/x")
    parser.add_argument("--y-column", default=None, help="Optional integer grid y column; default infers array_row/bin_y_index/y")
    parser.add_argument("--bin-size-um", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    configure_logging(verbose=bool(args.verbose))

    per_cell_df, cell_type_summary_df, summary = build_shape_reference_features(
        gt_cell_bins_path=args.gt_cell_bins_path,
        gt_cell_assignments_csv=args.gt_cell_assignments_csv,
        cell_id_column=str(args.cell_id_column),
        cell_type_column=str(args.cell_type_column),
        x_column=args.x_column,
        y_column=args.y_column,
        bin_size_um=float(args.bin_size_um),
        epsilon=float(args.epsilon),
    )
    write_shape_reference_outputs(
        per_cell_df=per_cell_df,
        cell_type_summary_df=cell_type_summary_df,
        summary=summary,
        per_cell_output_path=args.per_cell_output_path,
        cell_type_summary_output_path=args.cell_type_summary_output_path,
        npz_output_path=args.npz_output_path,
        summary_output_path=args.summary_path,
    )
    LOGGER.info("Done: shape reference for %d cells", len(per_cell_df))


if __name__ == "__main__":
    main()
