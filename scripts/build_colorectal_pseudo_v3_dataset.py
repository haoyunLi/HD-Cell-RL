#!/usr/bin/env python3
"""Build the formal patch-local colorectal pseudo-v3 count matrix.

Cell-derived pseudo counts are thinned with a spatially varying capture field
estimated from smoothed real 2 um total-count depth.  The probabilities are
calibrated so the retained cell-derived total plus structured ambient equals
the real total across the fixed patch contexts in expectation.

Structured ambient uses observed molecules from real bins without a positive
pseudo ground-truth owner.  Their genes are preserved, but their destinations
are sampled within a local Gaussian neighborhood so ambient RNA can enter both
background and cell-covered bins.  Ground truth is used only to construct the
simulation and is never exposed to inference methods.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io, sparse
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_patch_capture_stress_dataset import write_dataset  # noqa: E402
from scripts.build_patch_real_background_stress_dataset import (  # noqa: E402
    read_selected_real_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pseudo-hd-dir", required=True, type=Path)
    parser.add_argument("--real-10x-h5", required=True, type=Path)
    parser.add_argument("--selected-context-bins", required=True, type=Path)
    parser.add_argument("--fractional-gt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gene-allowlist", type=Path, default=None)
    parser.add_argument("--capture-field-sigma-um", type=float, default=16.0)
    parser.add_argument("--capture-field-neighbors", type=int, default=128)
    parser.add_argument("--ambient-sigma-um", type=float, default=12.0)
    parser.add_argument("--ambient-radius-um", type=float, default=32.0)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def read_feature_names(matrix_dir: Path) -> list[str]:
    path = matrix_dir / "features.tsv.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [line.rstrip("\n").split("\t")[1] for line in handle]


def smoothed_capture_field(
    coordinates: np.ndarray,
    real_totals: np.ndarray,
    *,
    sigma_um: float,
    n_neighbors: int,
) -> np.ndarray:
    if sigma_um <= 0:
        raise ValueError("capture-field sigma must be positive")
    if n_neighbors <= 0:
        raise ValueError("capture-field neighbor count must be positive")
    coordinates = np.asarray(coordinates, dtype=np.float64)
    real_totals = np.asarray(real_totals, dtype=np.float64)
    if coordinates.shape != (len(real_totals), 2):
        raise ValueError("capture coordinates and totals do not align")
    tree = cKDTree(coordinates)
    k = min(int(n_neighbors), len(coordinates))
    distances, neighbors = tree.query(
        coordinates,
        k=k,
        distance_upper_bound=3.0 * float(sigma_um),
        workers=-1,
    )
    if k == 1:
        distances = distances[:, None]
        neighbors = neighbors[:, None]
    valid = neighbors < len(coordinates)
    safe_neighbors = np.where(valid, neighbors, 0)
    weights = np.exp(-0.5 * (distances / float(sigma_um)) ** 2)
    weights[~valid] = 0.0
    denominators = weights.sum(axis=1)
    if np.any(denominators <= 0):
        raise RuntimeError("capture smoothing produced an empty neighborhood")
    field = (weights * real_totals[safe_neighbors]).sum(axis=1) / denominators
    positive = field[field > 0]
    fallback = float(np.median(positive)) if len(positive) else 1.0
    field = np.where(field > 0, field, fallback)
    return field


def calibrate_capture_probabilities(
    source_totals: np.ndarray,
    capture_field: np.ndarray,
    target_total: int,
) -> np.ndarray:
    source_totals = np.asarray(source_totals, dtype=np.float64)
    field = np.asarray(capture_field, dtype=np.float64)
    if source_totals.shape != field.shape:
        raise ValueError("source totals and capture field must have equal shape")
    source_total = float(source_totals.sum())
    if target_total < 0:
        raise ValueError("target cell-derived total cannot be negative")
    if target_total > source_total + 1.0e-8:
        raise ValueError(
            f"target cell-derived total {target_total} exceeds source {source_total:.0f}"
        )
    if target_total == 0:
        return np.zeros_like(source_totals, dtype=np.float64)
    positive_field = field[field > 0]
    if not len(positive_field):
        raise ValueError("capture field contains no positive values")
    normalized = field / float(np.mean(positive_field))

    def expected(scale: float) -> float:
        return float(np.dot(source_totals, np.clip(scale * normalized, 0.0, 1.0)))

    lower = 0.0
    upper = 1.0
    while expected(upper) < target_total and upper < 1.0e12:
        upper *= 2.0
    if expected(upper) + 1.0e-6 < target_total:
        raise RuntimeError("could not calibrate capture probabilities")
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if expected(midpoint) < target_total:
            lower = midpoint
        else:
            upper = midpoint
    probabilities = np.clip(upper * normalized, 0.0, 1.0)
    if abs(float(np.dot(source_totals, probabilities)) - target_total) > 1.0e-4:
        raise RuntimeError("capture calibration missed its expected-count target")
    return probabilities


def redistribute_background_ambient(
    *,
    real_matrix: sparse.coo_matrix,
    selected_columns: np.ndarray,
    background_columns: set[int],
    coordinates: np.ndarray,
    sigma_um: float,
    radius_um: float,
    rng: np.random.Generator,
) -> sparse.coo_matrix:
    if sigma_um <= 0 or radius_um <= 0:
        raise ValueError("ambient sigma and radius must be positive")
    selected_columns = np.asarray(selected_columns, dtype=np.int64)
    if len(selected_columns) != len(coordinates):
        raise ValueError("selected columns and coordinates do not align")
    local_by_global = {
        int(global_column): int(local_index)
        for local_index, global_column in enumerate(selected_columns)
    }
    missing_background = background_columns.difference(local_by_global)
    if missing_background:
        raise ValueError("background columns must be selected context columns")

    csc = real_matrix.tocsc(copy=False)
    tree = cKDTree(np.asarray(coordinates, dtype=np.float64))
    output_rows: list[np.ndarray] = []
    output_columns: list[np.ndarray] = []
    output_counts: list[np.ndarray] = []
    for source_global in sorted(background_columns):
        local_index = local_by_global[int(source_global)]
        start = int(csc.indptr[source_global])
        end = int(csc.indptr[source_global + 1])
        if end <= start:
            continue
        genes = np.asarray(csc.indices[start:end], dtype=np.int32)
        counts = np.rint(csc.data[start:end]).astype(np.int64)
        if (counts < 0).any() or not np.allclose(counts, csc.data[start:end]):
            raise ValueError("real ambient source counts must be nonnegative integers")
        molecule_genes = np.repeat(genes, counts)
        if not len(molecule_genes):
            continue
        candidate_local = np.asarray(
            tree.query_ball_point(coordinates[local_index], r=float(radius_um)),
            dtype=np.int64,
        )
        if not len(candidate_local):
            candidate_local = np.asarray([local_index], dtype=np.int64)
        deltas = coordinates[candidate_local] - coordinates[local_index]
        distances_sq = np.einsum("ij,ij->i", deltas, deltas)
        weights = np.exp(-0.5 * distances_sq / float(sigma_um) ** 2)
        weights /= weights.sum()
        molecule_dest_local = rng.choice(
            candidate_local,
            size=len(molecule_genes),
            replace=True,
            p=weights,
        )
        molecule_dest_global = selected_columns[molecule_dest_local]
        keys = molecule_genes.astype(np.int64) * real_matrix.shape[1] + molecule_dest_global
        unique_keys, unique_counts = np.unique(keys, return_counts=True)
        output_rows.append((unique_keys // real_matrix.shape[1]).astype(np.int32))
        output_columns.append((unique_keys % real_matrix.shape[1]).astype(np.int32))
        output_counts.append(unique_counts.astype(np.int32))

    if not output_counts:
        return sparse.coo_matrix(real_matrix.shape, dtype=np.int32)
    ambient = sparse.coo_matrix(
        (
            np.concatenate(output_counts),
            (np.concatenate(output_rows), np.concatenate(output_columns)),
        ),
        shape=real_matrix.shape,
        dtype=np.int32,
    )
    ambient.sum_duplicates()
    return ambient


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    points = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    return {
        str(point): float(value)
        for point, value in zip(points, np.quantile(values, points))
    }


def selected_gene_totals(
    matrix: sparse.spmatrix,
    feature_names: list[str],
    allowlist_path: Path | None,
    selected_columns: np.ndarray,
) -> tuple[np.ndarray | None, int]:
    if allowlist_path is None:
        return None, 0
    allowlist = {
        line.strip()
        for line in allowlist_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    row_mask = np.asarray([name in allowlist for name in feature_names], dtype=bool)
    totals = np.asarray(matrix.tocsr()[row_mask, :][:, selected_columns].sum(axis=0)).ravel()
    return totals.astype(np.int64), int(row_mask.sum())


def run(args: argparse.Namespace) -> dict[str, object]:
    for path in (
        args.source_pseudo_hd_dir,
        args.real_10x_h5,
        args.selected_context_bins,
        args.fractional_gt,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.gene_allowlist is not None and not args.gene_allowlist.exists():
        raise FileNotFoundError(args.gene_allowlist)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {args.output_dir}")

    selected = pd.read_parquet(args.selected_context_bins).sort_values(
        "matrix_col_index"
    )
    required_selected = {"matrix_col_index", "barcode", "x_um", "y_um"}
    missing_selected = required_selected.difference(selected.columns)
    if missing_selected:
        raise ValueError(
            f"selected-context table is missing columns: {sorted(missing_selected)}"
        )
    selected_columns = selected["matrix_col_index"].to_numpy(dtype=np.int64)
    if len(np.unique(selected_columns)) != len(selected_columns):
        raise ValueError("selected context columns are not unique")
    coordinates = selected[["x_um", "y_um"]].to_numpy(dtype=np.float64)
    selected_barcodes = selected["barcode"].astype(str).to_numpy()

    matrix_dir = args.source_pseudo_hd_dir / "filtered_feature_bc_matrix"
    pseudo = io.mmread(matrix_dir / "matrix.mtx.gz").tocoo(copy=False)
    pseudo.data = np.rint(pseudo.data).astype(np.int64)
    if (pseudo.data < 0).any():
        raise ValueError("source pseudo matrix contains negative counts")
    feature_names = read_feature_names(matrix_dir)
    real = read_selected_real_matrix(
        real_h5_path=args.real_10x_h5,
        selected_columns=selected_columns,
        expected_shape=pseudo.shape,
        expected_barcodes=selected_barcodes,
        expected_feature_names=feature_names,
    ).tocoo(copy=False)

    selected_lookup = np.zeros(pseudo.shape[1], dtype=bool)
    selected_lookup[selected_columns] = True
    keep_pseudo = selected_lookup[pseudo.col]
    pseudo_selected = sparse.coo_matrix(
        (
            pseudo.data[keep_pseudo],
            (pseudo.row[keep_pseudo], pseudo.col[keep_pseudo]),
        ),
        shape=pseudo.shape,
    )
    source_pseudo_totals = np.bincount(
        pseudo_selected.col,
        weights=pseudo_selected.data,
        minlength=pseudo.shape[1],
    )[selected_columns].astype(np.int64)
    real_totals = np.bincount(
        real.col,
        weights=real.data,
        minlength=real.shape[1],
    )[selected_columns].astype(np.int64)

    gt = pd.read_parquet(
        args.fractional_gt,
        columns=["barcode_index", "cell_id", "cell_coverage_fraction"],
    )
    selected_set = set(selected_columns.tolist())
    covered_columns = set(
        gt.loc[
            gt["barcode_index"].astype(np.int64).isin(selected_set)
            & (pd.to_numeric(gt["cell_id"], errors="raise") > 0)
            & (pd.to_numeric(gt["cell_coverage_fraction"], errors="raise") > 0),
            "barcode_index",
        ].astype(np.int64)
    )
    background_columns = selected_set.difference(covered_columns)
    if not background_columns:
        raise ValueError("selected contexts contain no GT-background bins")
    real_background_mask = np.isin(real.col, np.asarray(sorted(background_columns)))
    ambient_source_total = int(real.data[real_background_mask].sum())
    real_total = int(real.data.sum())
    target_cell_total = real_total - ambient_source_total
    if target_cell_total <= 0:
        raise ValueError("ambient source consumes the entire real count target")

    capture_field = smoothed_capture_field(
        coordinates,
        real_totals,
        sigma_um=float(args.capture_field_sigma_um),
        n_neighbors=int(args.capture_field_neighbors),
    )
    capture_probabilities = calibrate_capture_probabilities(
        source_pseudo_totals,
        capture_field,
        target_cell_total,
    )
    probability_by_global = np.zeros(pseudo.shape[1], dtype=np.float64)
    probability_by_global[selected_columns] = capture_probabilities
    rng = np.random.default_rng(int(args.seed))
    retained_counts = rng.binomial(
        pseudo_selected.data.astype(np.int64),
        probability_by_global[pseudo_selected.col],
    )
    retained_mask = retained_counts > 0
    captured = sparse.coo_matrix(
        (
            retained_counts[retained_mask].astype(np.int32),
            (
                pseudo_selected.row[retained_mask],
                pseudo_selected.col[retained_mask],
            ),
        ),
        shape=pseudo.shape,
        dtype=np.int32,
    )
    captured.sum_duplicates()

    ambient = redistribute_background_ambient(
        real_matrix=real,
        selected_columns=selected_columns,
        background_columns=background_columns,
        coordinates=coordinates,
        sigma_um=float(args.ambient_sigma_um),
        radius_um=float(args.ambient_radius_um),
        rng=rng,
    )
    if int(ambient.data.sum()) != ambient_source_total:
        raise AssertionError("ambient redistribution did not conserve molecules")
    final = (captured + ambient).tocoo()
    final.sum_duplicates()
    final.eliminate_zeros()
    if np.any(~selected_lookup[final.col]):
        raise AssertionError("final matrix contains counts outside selected contexts")

    positions = pd.read_parquet(
        args.source_pseudo_hd_dir / "spatial" / "tissue_positions.parquet"
    )
    write_dataset(
        output_dir=args.output_dir,
        source_dir=args.source_pseudo_hd_dir,
        positions=positions,
        selected_barcodes=set(selected_barcodes),
        matrix=final,
    )
    diagnostics_dir = args.output_dir / "diagnostics"
    diagnostics_dir.mkdir(exist_ok=True)
    final_totals = np.bincount(
        final.col, weights=final.data, minlength=final.shape[1]
    )[selected_columns].astype(np.int64)
    ambient_totals = np.bincount(
        ambient.col, weights=ambient.data, minlength=ambient.shape[1]
    )[selected_columns].astype(np.int64)
    captured_totals = np.bincount(
        captured.col, weights=captured.data, minlength=captured.shape[1]
    )[selected_columns].astype(np.int64)
    capture_table = selected.copy()
    capture_table["source_pseudo_total"] = source_pseudo_totals
    capture_table["real_total"] = real_totals
    capture_table["smoothed_real_depth"] = capture_field
    capture_table["capture_probability"] = capture_probabilities
    capture_table["retained_cell_total"] = captured_totals
    capture_table["ambient_total"] = ambient_totals
    capture_table["final_total"] = final_totals
    capture_table["is_gt_background"] = capture_table["matrix_col_index"].isin(
        background_columns
    )
    capture_table.to_parquet(diagnostics_dir / "capture_ambient_by_bin.parquet", index=False)

    selected_gene_summary: dict[str, object] | None = None
    if args.gene_allowlist is not None:
        pseudo_gene_totals, n_selected_genes = selected_gene_totals(
            pseudo_selected,
            feature_names,
            args.gene_allowlist,
            selected_columns,
        )
        real_gene_totals, _ = selected_gene_totals(
            real, feature_names, args.gene_allowlist, selected_columns
        )
        final_gene_totals, _ = selected_gene_totals(
            final, feature_names, args.gene_allowlist, selected_columns
        )
        selected_gene_summary = {
            "n_selected_genes": n_selected_genes,
            "source_pseudo_mean_per_bin": float(pseudo_gene_totals.mean()),
            "real_mean_per_bin": float(real_gene_totals.mean()),
            "final_mean_per_bin": float(final_gene_totals.mean()),
            "source_pseudo_median_per_bin": float(np.median(pseudo_gene_totals)),
            "real_median_per_bin": float(np.median(real_gene_totals)),
            "final_median_per_bin": float(np.median(final_gene_totals)),
            "real_quantiles": quantiles(real_gene_totals),
            "final_quantiles": quantiles(final_gene_totals),
        }

    source_assignments = args.source_pseudo_hd_dir / "ground_truth_cell_assignments.csv"
    if source_assignments.exists():
        shutil.copy2(source_assignments, args.output_dir / source_assignments.name)
    summary: dict[str, object] = {
        "version": "colorectal_pseudo_v3_patch_local",
        "source_pseudo_hd_dir": str(args.source_pseudo_hd_dir.resolve()),
        "real_10x_h5": str(args.real_10x_h5.resolve()),
        "selected_context_bins": str(args.selected_context_bins.resolve()),
        "fractional_gt": str(args.fractional_gt.resolve()),
        "matrix_shape": [int(value) for value in final.shape],
        "n_selected_context_bins": int(len(selected_columns)),
        "n_gt_covered_bins": int(len(covered_columns)),
        "n_gt_background_bins": int(len(background_columns)),
        "source_pseudo_total_umis": int(pseudo_selected.data.sum()),
        "real_total_umis": real_total,
        "real_background_ambient_source_umis": ambient_source_total,
        "ambient_target_fraction": float(ambient_source_total / real_total),
        "target_cell_derived_umis": target_cell_total,
        "expected_retained_cell_umis": float(
            np.dot(source_pseudo_totals, capture_probabilities)
        ),
        "realized_retained_cell_umis": int(captured.data.sum()),
        "realized_ambient_umis": int(ambient.data.sum()),
        "realized_final_umis": int(final.data.sum()),
        "realized_final_to_real_total_ratio": float(final.data.sum() / real_total),
        "capture_field": {
            "source": "Gaussian-smoothed real per-bin total UMI depth",
            "sigma_um": float(args.capture_field_sigma_um),
            "neighbors": int(args.capture_field_neighbors),
            "probability_min": float(capture_probabilities.min()),
            "probability_mean": float(capture_probabilities.mean()),
            "probability_max": float(capture_probabilities.max()),
            "calibration": "indexed per-bin probabilities with exact expected total",
        },
        "ambient": {
            "source": "real molecules from selected bins with no positive pseudo-GT owner",
            "sigma_um": float(args.ambient_sigma_um),
            "radius_um": float(args.ambient_radius_um),
            "gene_identity_preserved": True,
            "molecule_count_conserved": True,
        },
        "all_gene_depth": {
            "real_mean_per_bin": float(real_totals.mean()),
            "final_mean_per_bin": float(final_totals.mean()),
            "real_median_per_bin": float(np.median(real_totals)),
            "final_median_per_bin": float(np.median(final_totals)),
            "real_quantiles": quantiles(real_totals),
            "final_quantiles": quantiles(final_totals),
        },
        "selected_gene_depth": selected_gene_summary,
        "seed": int(args.seed),
        "shape_prior_built": False,
    }
    (args.output_dir / "pseudo_hd_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
