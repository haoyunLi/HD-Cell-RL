#!/usr/bin/env python3
"""Build paired patch-local pseudo-HD datasets with controlled capture thinning.

The source pseudo-HD matrix is restricted to the union of the fixed patch
context bounds.  Two otherwise identical 10x-style datasets are written:

* ``high_depth_local`` keeps the selected source counts exactly;
* ``capture_thinned`` applies one seeded binomial draw to every gene/bin count.

All barcodes and feature axes remain stable, and only ``in_tissue`` is masked
outside the selected patch contexts.  This makes downstream method outputs
directly alignable to the original physical barcode and cell IDs.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io, sparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pseudo-hd-dir", required=True)
    parser.add_argument("--bin-metadata", required=True)
    parser.add_argument("--patch-index", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--capture-probability", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")


def selected_context_bins(metadata: pd.DataFrame, patches: pd.DataFrame) -> np.ndarray:
    selected = np.zeros(len(metadata), dtype=bool)
    x = metadata["x_um"].to_numpy(dtype=np.float64)
    y = metadata["y_um"].to_numpy(dtype=np.float64)
    for patch in patches.itertuples(index=False):
        selected |= (
            (x >= float(patch.context_x_min))
            & (x <= float(patch.context_x_max))
            & (y >= float(patch.context_y_min))
            & (y <= float(patch.context_y_max))
        )
    return selected


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def write_matrix_market_gzip(matrix: sparse.spmatrix, output: Path) -> None:
    temporary = output.with_suffix("")
    io.mmwrite(temporary, matrix, field="integer", symmetry="general")
    with temporary.open("rb") as source, gzip.open(output, "wb", compresslevel=6) as target:
        shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
    temporary.unlink()


def write_dataset(
    *,
    output_dir: Path,
    source_dir: Path,
    positions: pd.DataFrame,
    selected_barcodes: set[str],
    matrix: sparse.coo_matrix,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output_dir}")
    matrix_dir = output_dir / "filtered_feature_bc_matrix"
    spatial_dir = output_dir / "spatial"
    matrix_dir.mkdir(parents=True)
    spatial_dir.mkdir(parents=True)

    write_matrix_market_gzip(matrix, matrix_dir / "matrix.mtx.gz")
    for name in ("features.tsv.gz", "barcodes.tsv.gz"):
        link_or_copy(source_dir / "filtered_feature_bc_matrix" / name, matrix_dir / name)

    source_spatial = source_dir / "spatial"
    for source in source_spatial.iterdir():
        if source.name == "tissue_positions.parquet":
            continue
        link_or_copy(source, spatial_dir / source.name)

    masked_positions = positions.copy()
    masked_positions["in_tissue"] = (
        masked_positions["barcode"].astype(str).isin(selected_barcodes).astype(np.int8)
    )
    masked_positions.to_parquet(spatial_dir / "tissue_positions.parquet", index=False)


def run(args: argparse.Namespace) -> None:
    probability = float(args.capture_probability)
    if not 0.0 < probability <= 1.0:
        raise ValueError("--capture-probability must be in (0, 1]")

    source_dir = Path(args.input_pseudo_hd_dir).resolve()
    metadata_path = Path(args.bin_metadata).resolve()
    patch_index_path = Path(args.patch_index).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_parquet(
        metadata_path,
        columns=["matrix_col_index", "barcode", "x_um", "y_um"],
    )
    patches = pd.read_csv(patch_index_path)
    require_columns(
        metadata,
        {"matrix_col_index", "barcode", "x_um", "y_um"},
        metadata_path,
    )
    require_columns(
        patches,
        {"context_x_min", "context_x_max", "context_y_min", "context_y_max"},
        patch_index_path,
    )
    if metadata["matrix_col_index"].duplicated().any():
        raise ValueError("Bin metadata has duplicate matrix_col_index values")

    selected_rows = selected_context_bins(metadata, patches)
    selected_metadata = metadata.loc[selected_rows].copy()
    if selected_metadata.empty:
        raise ValueError("Patch contexts selected no physical bins")
    selected_columns = selected_metadata["matrix_col_index"].to_numpy(dtype=np.int64)
    selected_barcodes = set(selected_metadata["barcode"].astype(str))

    matrix_path = source_dir / "filtered_feature_bc_matrix" / "matrix.mtx.gz"
    source_matrix = io.mmread(matrix_path).tocoo(copy=False)
    if source_matrix.shape[1] != len(metadata):
        raise ValueError(
            "Matrix barcode axis and bin metadata differ: "
            f"{source_matrix.shape[1]} != {len(metadata)}"
        )
    column_mask = np.zeros(source_matrix.shape[1], dtype=bool)
    column_mask[selected_columns] = True
    keep = column_mask[source_matrix.col]
    rows = source_matrix.row[keep].astype(np.int32, copy=True)
    columns = source_matrix.col[keep].astype(np.int32, copy=True)
    counts = np.rint(source_matrix.data[keep]).astype(np.int64, copy=True)
    if (counts < 0).any() or not np.allclose(counts, source_matrix.data[keep]):
        raise ValueError("Source Matrix Market counts must be nonnegative integers")
    shape = source_matrix.shape
    del source_matrix, keep, column_mask

    high_matrix = sparse.coo_matrix(
        (counts.astype(np.int32), (rows, columns)), shape=shape, dtype=np.int32
    )
    high_matrix.sum_duplicates()
    rng = np.random.default_rng(int(args.seed))
    thinned_counts = rng.binomial(counts, probability)
    retained = thinned_counts > 0
    thinned_matrix = sparse.coo_matrix(
        (
            thinned_counts[retained].astype(np.int32),
            (rows[retained], columns[retained]),
        ),
        shape=shape,
        dtype=np.int32,
    )
    thinned_matrix.sum_duplicates()

    positions_path = source_dir / "spatial" / "tissue_positions.parquet"
    positions = pd.read_parquet(positions_path)
    require_columns(positions, {"barcode", "in_tissue"}, positions_path)
    position_barcodes = set(positions["barcode"].astype(str))
    missing_selected_barcodes = selected_barcodes.difference(position_barcodes)
    if missing_selected_barcodes:
        raise ValueError(
            "Selected matrix barcodes are missing from tissue positions: "
            f"{len(missing_selected_barcodes)}"
        )

    high_dir = output_root / "high_depth_local"
    thinned_dir = output_root / "capture_thinned"
    write_dataset(
        output_dir=high_dir,
        source_dir=source_dir,
        positions=positions,
        selected_barcodes=selected_barcodes,
        matrix=high_matrix,
    )
    write_dataset(
        output_dir=thinned_dir,
        source_dir=source_dir,
        positions=positions,
        selected_barcodes=selected_barcodes,
        matrix=thinned_matrix,
    )

    source_total = int(counts.sum())
    thinned_total = int(thinned_counts.sum())
    summary = {
        "input_pseudo_hd_dir": os.fspath(source_dir),
        "bin_metadata": os.fspath(metadata_path),
        "patch_index": os.fspath(patch_index_path),
        "matrix_shape": [int(value) for value in shape],
        "n_patches": int(len(patches)),
        "n_selected_context_bins": int(len(selected_metadata)),
        "n_source_nonzero_entries": int(len(counts)),
        "n_thinned_nonzero_entries": int(retained.sum()),
        "source_total_umis": source_total,
        "thinned_total_umis": thinned_total,
        "configured_capture_probability": probability,
        "realized_molecule_retention": (
            float(thinned_total / source_total) if source_total else 0.0
        ),
        "seed": int(args.seed),
        "outside_context_counts": 0,
        "barcode_and_feature_axes_preserved": True,
        "high_depth_local_dir": os.fspath(high_dir),
        "capture_thinned_dir": os.fspath(thinned_dir),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    selected_metadata.to_parquet(output_root / "selected_context_bins.parquet", index=False)
    print(json.dumps(summary, indent=2))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
