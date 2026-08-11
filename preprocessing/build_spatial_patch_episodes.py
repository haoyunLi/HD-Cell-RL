#!/usr/bin/env python
"""Build overlapping spatial patch index for multi-cell patch training."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-index-path", required=True)
    parser.add_argument("--nuclei-path", required=True)
    parser.add_argument("--nuclei-format", default="auto", choices=("auto", "csv", "tsv", "parquet"))
    parser.add_argument("--cell-id-column", default="cell_id")
    parser.add_argument("--center-x-column", default="center_x_um")
    parser.add_argument("--center-y-column", default="center_y_um")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="human_colorectal_patch")
    parser.add_argument("--patch-size-um", type=float, default=64.0)
    parser.add_argument("--stride-um", type=float, default=32.0)
    parser.add_argument("--core-size-um", type=float, default=48.0)
    parser.add_argument("--n-jitter-passes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-core-cells", type=int, default=1)
    parser.add_argument("--min-patch-cells", type=int, default=2)
    parser.add_argument("--max-patch-cells", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    episodes_index_path = Path(args.episodes_index_path).expanduser().resolve()
    nuclei_path = Path(args.nuclei_path).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_df = pd.read_csv(episodes_index_path, usecols=["cell_id"])
    episode_cells = set(episode_df["cell_id"].astype(str))
    nuclei_format = _normalize_format(str(args.nuclei_format), nuclei_path)
    nuclei_df = _load_table(nuclei_path, nuclei_format)
    centers = _build_nuclei_centers(
        nuclei_df,
        {
            "cell_id": str(args.cell_id_column),
            "center_x_um": str(args.center_x_column),
            "center_y_um": str(args.center_y_column),
        },
    )
    centers = {cell_id: xy for cell_id, xy in centers.items() if cell_id in episode_cells}
    if not centers:
        raise RuntimeError("no nuclei centers overlap with episodes index cells")

    cell_ids = np.asarray(list(centers.keys()), dtype=object)
    xy = np.vstack([centers[str(cell_id)] for cell_id in cell_ids]).astype(np.float64)
    patch_size = float(args.patch_size_um)
    stride = float(args.stride_um)
    core_size = float(args.core_size_um)
    if not (0.0 < core_size <= patch_size):
        raise ValueError("--core-size-um must be >0 and <= --patch-size-um")
    if stride <= 0:
        raise ValueError("--stride-um must be >0")

    rng = np.random.default_rng(int(args.seed))
    spatial_index = cKDTree(xy)
    outer_query_radius = float(np.sqrt(2.0) * patch_size / 2.0)
    centers_to_try: list[tuple[float, float, str]] = []
    x_min, y_min = np.min(xy, axis=0)
    x_max, y_max = np.max(xy, axis=0)
    pad = patch_size / 2.0
    base_x = np.arange(x_min - pad, x_max + pad + stride, stride)
    base_y = np.arange(y_min - pad, y_max + pad + stride, stride)
    for pass_idx in range(int(args.n_jitter_passes) + 1):
        if pass_idx == 0:
            dx = 0.0
            dy = 0.0
        else:
            dx = float(rng.uniform(-0.5 * stride, 0.5 * stride))
            dy = float(rng.uniform(-0.5 * stride, 0.5 * stride))
        for cx in base_x + dx:
            for cy in base_y + dy:
                centers_to_try.append((float(cx), float(cy), f"grid{pass_idx}"))

    rows: list[dict[str, Any]] = []
    covered_core: set[str] = set()
    for cx, cy, source in centers_to_try:
        candidate_indices = spatial_index.query_ball_point([cx, cy], outer_query_radius)
        if not candidate_indices:
            continue
        row = _make_patch_row(
            cx=cx,
            cy=cy,
            source=source,
            cell_ids=cell_ids,
            xy=xy,
            candidate_indices=candidate_indices,
            patch_size=patch_size,
            core_size=core_size,
            min_core_cells=int(args.min_core_cells),
            min_patch_cells=int(args.min_patch_cells),
            max_patch_cells=int(args.max_patch_cells),
            patch_index=len(rows),
        )
        if row is None:
            continue
        rows.append(row)
        covered_core.update(json.loads(row["core_cell_ids"]))

    missing = sorted(set(cell_ids.astype(str)) - covered_core)
    for cell_id in missing:
        center = centers[str(cell_id)]
        candidate_indices = spatial_index.query_ball_point(center, outer_query_radius)
        row = _make_patch_row(
            cx=float(center[0]),
            cy=float(center[1]),
            source="rescue",
            cell_ids=cell_ids,
            xy=xy,
            candidate_indices=candidate_indices,
            patch_size=patch_size,
            core_size=core_size,
            min_core_cells=1,
            min_patch_cells=1,
            max_patch_cells=int(args.max_patch_cells),
            patch_index=len(rows),
        )
        if row is not None:
            rows.append(row)
            covered_core.update(json.loads(row["core_cell_ids"]))

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise RuntimeError("no valid spatial patches were generated")
    patches_index = out_dir / f"{args.prefix}_patches_index.csv"
    out_df.to_csv(patches_index, index=False)

    summary = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "episodes_index_path": str(episodes_index_path),
        "nuclei_path": str(nuclei_path),
        "patches_index_path": str(patches_index),
        "n_input_cells": int(len(cell_ids)),
        "n_patches": int(len(out_df)),
        "n_cells_covered_as_core": int(len(covered_core)),
        "n_cells_missing_core": int(len(set(cell_ids.astype(str)) - covered_core)),
        "patch_size_um": patch_size,
        "stride_um": stride,
        "core_size_um": core_size,
        "n_jitter_passes": int(args.n_jitter_passes),
        "min_core_cells": int(args.min_core_cells),
        "min_patch_cells": int(args.min_patch_cells),
        "max_patch_cells": int(args.max_patch_cells),
        "mean_patch_cells": float(out_df["n_patch_cells"].mean()),
        "median_patch_cells": float(out_df["n_patch_cells"].median()),
        "mean_core_cells": float(out_df["n_core_cells"].mean()),
        "median_core_cells": float(out_df["n_core_cells"].median()),
    }
    with (out_dir / f"{args.prefix}_patches_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Wrote patch index: {patches_index}")
    print(f"Summary: {summary}")


def _make_patch_row(
    *,
    cx: float,
    cy: float,
    source: str,
    cell_ids: np.ndarray,
    xy: np.ndarray,
    candidate_indices: list[int] | np.ndarray,
    patch_size: float,
    core_size: float,
    min_core_cells: int,
    min_patch_cells: int,
    max_patch_cells: int,
    patch_index: int,
) -> dict[str, Any] | None:
    half = patch_size / 2.0
    core_half = core_size / 2.0
    outer = (cx - half, cx + half, cy - half, cy + half)
    core = (cx - core_half, cx + core_half, cy - core_half, cy + core_half)
    candidate_idx = np.asarray(candidate_indices, dtype=np.int64)
    candidate_xy = xy[candidate_idx]
    in_outer = (
        (candidate_xy[:, 0] >= outer[0])
        & (candidate_xy[:, 0] <= outer[1])
        & (candidate_xy[:, 1] >= outer[2])
        & (candidate_xy[:, 1] <= outer[3])
    )
    outer_idx = candidate_idx[in_outer]
    outer_xy = xy[outer_idx]
    in_core = (
        (outer_xy[:, 0] >= core[0])
        & (outer_xy[:, 0] <= core[1])
        & (outer_xy[:, 1] >= core[2])
        & (outer_xy[:, 1] <= core[3])
    )
    core_idx = outer_idx[in_core]
    patch_cells = cell_ids[outer_idx].astype(str).tolist()
    core_cells = cell_ids[core_idx].astype(str).tolist()
    if len(core_cells) < min_core_cells or len(patch_cells) < min_patch_cells:
        return None
    if max_patch_cells > 0 and len(patch_cells) > max_patch_cells:
        if len(core_cells) > max_patch_cells:
            core_dist = np.sum((xy[core_idx] - np.asarray([cx, cy], dtype=np.float64)) ** 2, axis=1)
            keep_core = np.argsort(core_dist)[:max_patch_cells]
            core_idx = core_idx[keep_core]
            core_cells = cell_ids[core_idx].astype(str).tolist()
            patch_cells = list(core_cells)
        else:
            core_set = set(core_cells)
            margin_idx = np.asarray(
                [idx for idx in outer_idx.tolist() if str(cell_ids[int(idx)]) not in core_set],
                dtype=np.int64,
            )
            if margin_idx.size:
                margin_dist = np.sum((xy[margin_idx] - np.asarray([cx, cy], dtype=np.float64)) ** 2, axis=1)
                keep_margin = margin_idx[np.argsort(margin_dist)[: max_patch_cells - len(core_cells)]]
                patch_cells = core_cells + cell_ids[keep_margin].astype(str).tolist()
            else:
                patch_cells = list(core_cells)
    margin_cells = [cell for cell in patch_cells if cell not in set(core_cells)]
    return {
        "patch_id": f"patch_{patch_index:07d}",
        "source": source,
        "center_x_um": float(cx),
        "center_y_um": float(cy),
        "outer_x_min": float(outer[0]),
        "outer_x_max": float(outer[1]),
        "outer_y_min": float(outer[2]),
        "outer_y_max": float(outer[3]),
        "core_x_min": float(core[0]),
        "core_x_max": float(core[1]),
        "core_y_min": float(core[2]),
        "core_y_max": float(core[3]),
        "n_patch_cells": int(len(patch_cells)),
        "n_core_cells": int(len(core_cells)),
        "n_margin_cells": int(len(margin_cells)),
        "patch_cell_ids": json.dumps(patch_cells),
        "core_cell_ids": json.dumps(core_cells),
        "margin_cell_ids": json.dumps(margin_cells),
    }


def _normalize_format(fmt: str, path: Path) -> str:
    if fmt != "auto":
        return fmt
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".tsv":
        return "tsv"
    return "csv"


def _load_table(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "tsv":
        return pd.read_csv(path, sep="\t")
    if fmt == "csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported table format: {fmt}")


def _build_nuclei_centers(df: pd.DataFrame, columns: dict[str, str]) -> dict[str, np.ndarray]:
    missing = [col for col in columns.values() if col not in df.columns]
    if missing:
        raise ValueError(f"nuclei table is missing columns: {missing}")
    out: dict[str, np.ndarray] = {}
    selected = df[[columns["cell_id"], columns["center_x_um"], columns["center_y_um"]]]
    for cell_id_raw, x_raw, y_raw in selected.itertuples(index=False, name=None):
        cell_id = str(cell_id_raw)
        x = float(x_raw)
        y = float(y_raw)
        out[cell_id] = np.asarray([x, y], dtype=np.float32)
    return out


if __name__ == "__main__":
    main()
