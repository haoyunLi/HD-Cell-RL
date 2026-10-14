#!/usr/bin/env python3
"""Build patch-local pseudo-HD mixtures with real spatial background.

The pseudo cell-derived counts and the real Visium HD counts are aligned by
their shared feature and physical-barcode axes.  For each requested real-count
fraction, the expected total molecule count is held at the real selected-patch
total while the contribution is split between pseudo signal and real spatial
expression.  These mixtures are robustness tests, not generative biological
models: the real contribution contains structured tissue expression as well as
ambient/background RNA.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import io, sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_patch_capture_stress_dataset import (  # noqa: E402
    require_columns,
    selected_context_bins,
    write_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo-hd-dir", required=True)
    parser.add_argument("--real-10x-h5", required=True)
    parser.add_argument("--bin-metadata", required=True)
    parser.add_argument("--patch-index", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--real-fractions",
        type=float,
        nargs="+",
        default=(0.2, 0.5),
        help="Expected fractions of final molecules contributed by real data.",
    )
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def _decode(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def _read_pseudo_feature_names(pseudo_hd_dir: Path) -> list[str]:
    path = pseudo_hd_dir / "filtered_feature_bc_matrix" / "features.tsv.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [line.rstrip("\n").split("\t")[1] for line in handle]


def read_selected_real_matrix(
    *,
    real_h5_path: Path,
    selected_columns: np.ndarray,
    expected_shape: tuple[int, int],
    expected_barcodes: np.ndarray,
    expected_feature_names: list[str],
) -> sparse.coo_matrix:
    """Read only selected CSC columns from a 10x H5 matrix."""

    selected_columns = np.asarray(selected_columns, dtype=np.int64)
    if selected_columns.ndim != 1 or len(np.unique(selected_columns)) != len(selected_columns):
        raise ValueError("selected_columns must be a one-dimensional unique array")
    order = np.argsort(selected_columns, kind="stable")
    selected_columns = selected_columns[order]
    expected_barcodes = np.asarray(expected_barcodes, dtype=str)[order]

    rows_parts: list[np.ndarray] = []
    columns_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    with h5py.File(real_h5_path, "r") as handle:
        matrix = handle["matrix"]
        shape = tuple(int(value) for value in np.asarray(matrix["shape"]))
        if shape != expected_shape:
            raise ValueError(f"Real and pseudo matrix shapes differ: {shape} != {expected_shape}")

        real_features = _decode(np.asarray(matrix["features"]["name"]))
        if real_features != expected_feature_names:
            raise ValueError("Real and pseudo feature-name axes are not identical")
        real_barcodes = np.asarray(matrix["barcodes"])[selected_columns]
        if _decode(real_barcodes) != expected_barcodes.tolist():
            raise ValueError("Real and pseudo selected barcode axes are not identical")

        indptr = np.asarray(matrix["indptr"], dtype=np.int64)
        for column in selected_columns:
            start = int(indptr[column])
            end = int(indptr[column + 1])
            if end <= start:
                continue
            rows_parts.append(np.asarray(matrix["indices"][start:end], dtype=np.int32))
            columns_parts.append(np.full(end - start, int(column), dtype=np.int32))
            data_parts.append(np.asarray(matrix["data"][start:end], dtype=np.int64))

    if not data_parts:
        return sparse.coo_matrix(expected_shape, dtype=np.int32)
    rows = np.concatenate(rows_parts)
    columns = np.concatenate(columns_parts)
    data = np.concatenate(data_parts)
    return sparse.coo_matrix((data, (rows, columns)), shape=expected_shape)


def calibrated_cell_retention(
    *, pseudo_total: int, real_total: int, real_fraction: float
) -> float:
    if pseudo_total <= 0 or real_total <= 0:
        raise ValueError("pseudo_total and real_total must both be positive")
    if not 0.0 <= real_fraction <= 1.0:
        raise ValueError("real_fraction must be in [0, 1]")
    probability = (1.0 - real_fraction) * float(real_total) / float(pseudo_total)
    if probability > 1.0 + 1.0e-12:
        raise ValueError(
            "The requested mixture needs more pseudo molecules than the source contains"
        )
    return min(max(probability, 0.0), 1.0)


def sample_sparse_mixture(
    *,
    pseudo_matrix: sparse.coo_matrix,
    real_matrix: sparse.coo_matrix,
    pseudo_probability: float,
    real_probability: float,
    rng: np.random.Generator,
) -> tuple[sparse.coo_matrix, int, int]:
    if pseudo_matrix.shape != real_matrix.shape:
        raise ValueError("pseudo_matrix and real_matrix must have the same shape")
    if not 0.0 <= pseudo_probability <= 1.0:
        raise ValueError("pseudo_probability must be in [0, 1]")
    if not 0.0 <= real_probability <= 1.0:
        raise ValueError("real_probability must be in [0, 1]")

    pseudo_counts = rng.binomial(
        np.rint(pseudo_matrix.data).astype(np.int64), pseudo_probability
    )
    real_counts = rng.binomial(
        np.rint(real_matrix.data).astype(np.int64), real_probability
    )
    pseudo_keep = pseudo_counts > 0
    real_keep = real_counts > 0
    pseudo_sample = sparse.coo_matrix(
        (
            pseudo_counts[pseudo_keep].astype(np.int32),
            (pseudo_matrix.row[pseudo_keep], pseudo_matrix.col[pseudo_keep]),
        ),
        shape=pseudo_matrix.shape,
        dtype=np.int32,
    )
    real_sample = sparse.coo_matrix(
        (
            real_counts[real_keep].astype(np.int32),
            (real_matrix.row[real_keep], real_matrix.col[real_keep]),
        ),
        shape=real_matrix.shape,
        dtype=np.int32,
    )
    mixture = (pseudo_sample + real_sample).tocoo()
    mixture.sum_duplicates()
    mixture.eliminate_zeros()
    return mixture, int(pseudo_counts.sum()), int(real_counts.sum())


def _fraction_label(fraction: float) -> str:
    return f"real_background_{int(round(fraction * 100)):02d}pct"


def run(args: argparse.Namespace) -> None:
    fractions = [float(value) for value in args.real_fractions]
    if not fractions or len(set(fractions)) != len(fractions):
        raise ValueError("--real-fractions must contain unique values")
    if any(value < 0.0 or value > 1.0 for value in fractions):
        raise ValueError("--real-fractions values must be in [0, 1]")

    pseudo_hd_dir = Path(args.pseudo_hd_dir).resolve()
    real_h5_path = Path(args.real_10x_h5).resolve()
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
        metadata, {"matrix_col_index", "barcode", "x_um", "y_um"}, metadata_path
    )
    require_columns(
        patches,
        {"context_x_min", "context_x_max", "context_y_min", "context_y_max"},
        patch_index_path,
    )
    selected = metadata.loc[selected_context_bins(metadata, patches)].copy()
    if selected.empty:
        raise ValueError("Patch contexts selected no physical bins")
    selected_columns = selected["matrix_col_index"].to_numpy(dtype=np.int64)
    selected_barcodes = selected["barcode"].astype(str).to_numpy()

    pseudo_matrix = io.mmread(
        pseudo_hd_dir / "filtered_feature_bc_matrix" / "matrix.mtx.gz"
    ).tocoo(copy=False)
    pseudo_matrix.data = np.rint(pseudo_matrix.data).astype(np.int64)
    expected_feature_names = _read_pseudo_feature_names(pseudo_hd_dir)
    real_matrix = read_selected_real_matrix(
        real_h5_path=real_h5_path,
        selected_columns=selected_columns,
        expected_shape=pseudo_matrix.shape,
        expected_barcodes=selected_barcodes,
        expected_feature_names=expected_feature_names,
    )

    pseudo_total = int(pseudo_matrix.data.sum())
    real_total = int(real_matrix.data.sum())
    positions_path = pseudo_hd_dir / "spatial" / "tissue_positions.parquet"
    positions = pd.read_parquet(positions_path)
    selected_barcode_set = set(selected_barcodes)
    summaries = []
    seed_sequence = np.random.SeedSequence(int(args.seed))
    for fraction, child_seed in zip(fractions, seed_sequence.spawn(len(fractions))):
        pseudo_probability = calibrated_cell_retention(
            pseudo_total=pseudo_total,
            real_total=real_total,
            real_fraction=fraction,
        )
        mixture, retained_pseudo, retained_real = sample_sparse_mixture(
            pseudo_matrix=pseudo_matrix,
            real_matrix=real_matrix,
            pseudo_probability=pseudo_probability,
            real_probability=fraction,
            rng=np.random.default_rng(child_seed),
        )
        label = _fraction_label(fraction)
        output_dir = output_root / label
        write_dataset(
            output_dir=output_dir,
            source_dir=pseudo_hd_dir,
            positions=positions,
            selected_barcodes=selected_barcode_set,
            matrix=mixture,
        )
        realized_total = retained_pseudo + retained_real
        summaries.append(
            {
                "label": label,
                "output_dir": str(output_dir),
                "configured_real_fraction": fraction,
                "calibrated_pseudo_retention_probability": pseudo_probability,
                "real_retention_probability": fraction,
                "retained_pseudo_umis": retained_pseudo,
                "retained_real_umis": retained_real,
                "realized_total_umis": realized_total,
                "realized_real_fraction": (
                    float(retained_real / realized_total) if realized_total else 0.0
                ),
                "n_nonzero_entries": int(mixture.nnz),
            }
        )

    summary = {
        "pseudo_hd_dir": str(pseudo_hd_dir),
        "real_10x_h5": str(real_h5_path),
        "bin_metadata": str(metadata_path),
        "patch_index": str(patch_index_path),
        "n_selected_context_bins": int(len(selected)),
        "matrix_shape": [int(value) for value in pseudo_matrix.shape],
        "source_pseudo_umis": pseudo_total,
        "selected_real_umis": real_total,
        "target_expected_umis_per_condition": real_total,
        "seed": int(args.seed),
        "interpretation": (
            "Controlled robustness mixtures; real counts include structured tissue "
            "expression and are not a pure ambient-RNA estimate."
        ),
        "conditions": summaries,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    selected.to_parquet(output_root / "selected_context_bins.parquet", index=False)
    print(json.dumps(summary, indent=2))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
