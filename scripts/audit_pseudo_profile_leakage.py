#!/usr/bin/env python3
"""Audit pseudo-HD source-profile, compartment, and reference-cohort leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hd_cell_rl.reward import compute_reference_distribution
from preprocessing.ppo_format_assignment_eval import _TenXColumnExpressionReader


COMPARTMENTS = ("whole_gt", "nuclear_overlap", "cytoplasm_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial-matrix", required=True, type=Path)
    parser.add_argument("--source-sc-h5", required=True, type=Path)
    parser.add_argument("--gt-cell-assignments", required=True, type=Path)
    parser.add_argument("--gt-cell-bins", required=True, type=Path)
    parser.add_argument("--gt-nuclear-bins", required=True, type=Path)
    parser.add_argument("--evaluated-cells-csv", required=True, type=Path)
    parser.add_argument("--reference-npz", required=True, type=Path)
    parser.add_argument("--reference-summary", required=True, type=Path)
    parser.add_argument("--pseudo-summary", required=True, type=Path)
    parser.add_argument("--shape-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epsilon", type=float, default=1.0e-8)
    parser.add_argument("--csv-chunk-size", type=int, default=500_000)
    return parser.parse_args()


def read_selected_bins(
    path: Path,
    cell_ids: set[int],
    usecols: list[str],
    chunk_size: int,
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size):
        current = chunk.loc[chunk["cell_id"].isin(cell_ids)]
        if not current.empty:
            selected.append(current.copy())
    if not selected:
        return pd.DataFrame(columns=usecols)
    return pd.concat(selected, ignore_index=True)


def standardized_ranks(values: np.ndarray) -> np.ndarray:
    ranks = np.apply_along_axis(rankdata, 1, np.asarray(values, dtype=np.float64))
    ranks -= ranks.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(ranks, axis=1, keepdims=True)
    norm[norm <= 0] = 1.0
    return ranks / norm


def safe_spearman(left: np.ndarray, right: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        left = left[mask]
        right = right[mask]
    if left.size < 2 or left.sum() <= 0 or right.sum() <= 0:
        return np.nan
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else np.nan


def summarize(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    valid = array[np.isfinite(array)]
    if not len(valid):
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": int(len(valid)),
        "mean": float(valid.mean()),
        "median": float(np.median(valid)),
        "min": float(valid.min()),
        "max": float(valid.max()),
    }


def main() -> None:
    args = parse_args()
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive")
    if args.csv_chunk_size <= 0:
        raise ValueError("--csv-chunk-size must be positive")

    evaluated = pd.read_csv(args.evaluated_cells_csv)
    id_column = (
        "matched_gt_cell_id"
        if "matched_gt_cell_id" in evaluated.columns
        else "cell_id"
    )
    cell_ids = (
        pd.to_numeric(evaluated[id_column], errors="coerce")
        .dropna()
        .astype(np.int64)
        .drop_duplicates()
        .tolist()
    )
    if not cell_ids:
        raise ValueError("No evaluated physical cells were found")
    cell_id_set = set(cell_ids)

    assignments = pd.read_csv(args.gt_cell_assignments)
    required_assignment = {"cell_id", "cell_type", "sc_cell_barcode"}
    if missing := required_assignment.difference(assignments.columns):
        raise ValueError(f"GT cell assignments missing columns: {sorted(missing)}")
    assignments["cell_id"] = pd.to_numeric(assignments["cell_id"], errors="raise").astype(
        np.int64
    )
    assignments = assignments.loc[assignments["cell_id"].isin(cell_id_set)].copy()
    assignments = assignments.set_index("cell_id").loc[cell_ids]

    whole = read_selected_bins(
        args.gt_cell_bins,
        cell_id_set,
        ["cell_id", "barcode", "nuclear_fraction", "cytoplasm_fraction"],
        args.csv_chunk_size,
    )
    nuclear = read_selected_bins(
        args.gt_nuclear_bins,
        cell_id_set,
        ["cell_id", "barcode"],
        args.csv_chunk_size,
    )
    bins_by_compartment: dict[str, dict[int, list[str]]] = {
        "whole_gt": {
            int(cell_id): group["barcode"].astype(str).tolist()
            for cell_id, group in whole.groupby("cell_id")
        },
        "nuclear_overlap": {
            int(cell_id): group["barcode"].astype(str).tolist()
            for cell_id, group in nuclear.groupby("cell_id")
        },
        "cytoplasm_only": {
            int(cell_id): group.loc[
                group["nuclear_fraction"].to_numpy(dtype=np.float64) <= 0,
                "barcode",
            ].astype(str).tolist()
            for cell_id, group in whole.groupby("cell_id")
        },
    }

    reference = np.load(args.reference_npz, allow_pickle=True)
    reference_counts = np.asarray(reference["reference_counts"], dtype=np.float64)
    reference_types = np.asarray(reference["cell_types"]).astype(str)
    genes = np.asarray(reference["genes"]).astype(str).tolist()
    if reference_counts.shape != (len(reference_types), len(genes)):
        raise ValueError("Reference counts, cell types, and genes have inconsistent axes")

    spatial_reader = _TenXColumnExpressionReader(args.spatial_matrix, genes)
    source_reader = _TenXColumnExpressionReader(args.source_sc_h5, genes)
    try:
        source_expression = np.stack(
            [
                source_reader.sum_barcodes(
                    [str(assignments.loc[cell_id, "sc_cell_barcode"])]
                )
                for cell_id in cell_ids
            ]
        )
        expression = {
            compartment: np.stack(
                [
                    spatial_reader.sum_barcodes(
                        bins_by_compartment[compartment].get(cell_id, [])
                    )
                    for cell_id in cell_ids
                ]
            )
            for compartment in COMPARTMENTS
        }
    finally:
        spatial_reader.close()
        source_reader.close()

    source_ranks = standardized_ranks(source_expression)
    true_types = assignments["cell_type"].astype(str).to_numpy()
    theta = compute_reference_distribution(reference_counts, epsilon=args.epsilon)
    log_theta = np.log(theta)
    type_to_index = {value: index for index, value in enumerate(reference_types)}

    per_cell_rows: list[dict] = []
    pair_rows: list[dict] = []
    summary_by_compartment: dict[str, dict] = {}
    for compartment in COMPARTMENTS:
        current = expression[compartment]
        current_ranks = standardized_ranks(current)
        correlations = current_ranks @ source_ranks.T
        rank_matrix = np.argsort(np.argsort(-correlations, axis=1), axis=1) + 1
        exact_rank = rank_matrix[np.arange(len(cell_ids)), np.arange(len(cell_ids))]

        totals = current.sum(axis=1)
        weighted = np.einsum("bg,kg->bk", current, log_theta, optimize=False)
        ll = np.zeros_like(weighted)
        positive = totals > 0
        ll[positive] = weighted[positive] / totals[positive, None]
        predicted_type_index = np.argmax(ll, axis=1)
        predicted_types = reference_types[predicted_type_index]
        type_correct = predicted_types == true_types
        sorted_ll = np.sort(ll, axis=1)
        type_margin = sorted_ll[:, -1] - sorted_ll[:, -2]

        exact_values = []
        union_values = []
        same_type_values = []
        different_type_values = []
        for row_index, cell_id in enumerate(cell_ids):
            exact = float(correlations[row_index, row_index])
            union_mask = (current[row_index] > 0) | (source_expression[row_index] > 0)
            union = safe_spearman(
                current[row_index], source_expression[row_index], union_mask
            )
            other = np.arange(len(cell_ids)) != row_index
            same = other & (true_types == true_types[row_index])
            different = true_types != true_types[row_index]
            same_values = correlations[row_index, same]
            different_values = correlations[row_index, different]
            exact_values.append(exact)
            union_values.append(union)
            same_type_values.extend(same_values.tolist())
            different_type_values.extend(different_values.tolist())
            per_cell_rows.append(
                {
                    "compartment": compartment,
                    "cell_id": cell_id,
                    "cell_type": true_types[row_index],
                    "source_sc_barcode": str(
                        assignments.loc[cell_id, "sc_cell_barcode"]
                    ),
                    "n_spatial_bins": len(
                        bins_by_compartment[compartment].get(cell_id, [])
                    ),
                    "spatial_total_counts": float(totals[row_index]),
                    "source_total_counts": float(source_expression[row_index].sum()),
                    "spatial_to_source_total_ratio": (
                        float(totals[row_index] / source_expression[row_index].sum())
                        if source_expression[row_index].sum() > 0
                        else np.nan
                    ),
                    "exact_source_spearman_all_genes": exact,
                    "exact_source_spearman_nonzero_union": union,
                    "exact_source_rank_among_evaluated_cells": int(
                        exact_rank[row_index]
                    ),
                    "exact_source_is_top1": bool(exact_rank[row_index] == 1),
                    "best_source_cell_id": int(
                        cell_ids[int(np.argmax(correlations[row_index]))]
                    ),
                    "mean_other_same_type_source_spearman": (
                        float(same_values.mean()) if len(same_values) else np.nan
                    ),
                    "mean_different_type_source_spearman": (
                        float(different_values.mean())
                        if len(different_values)
                        else np.nan
                    ),
                    "reference_predicted_cell_type": predicted_types[row_index],
                    "reference_true_type_top1": bool(type_correct[row_index]),
                    "reference_type_ll_margin": float(type_margin[row_index]),
                }
            )
            for source_index, source_cell_id in enumerate(cell_ids):
                pair_rows.append(
                    {
                        "compartment": compartment,
                        "spatial_cell_id": cell_id,
                        "spatial_cell_type": true_types[row_index],
                        "source_cell_id": source_cell_id,
                        "source_cell_type": true_types[source_index],
                        "is_exact_source": row_index == source_index,
                        "spearman_all_genes": float(
                            correlations[row_index, source_index]
                        ),
                        "source_rank": int(rank_matrix[row_index, source_index]),
                    }
                )

        summary_by_compartment[compartment] = {
            "exact_source_spearman_all_genes": summarize(exact_values),
            "exact_source_spearman_nonzero_union": summarize(union_values),
            "other_same_type_source_spearman": summarize(same_type_values),
            "different_type_source_spearman": summarize(different_type_values),
            "fraction_exact_source_top1_among_evaluated_cells": float(
                np.mean(exact_rank == 1)
            ),
            "fraction_exact_source_top3_among_evaluated_cells": float(
                np.mean(exact_rank <= 3)
            ),
            "median_exact_source_rank": float(np.median(exact_rank)),
            "fraction_true_cell_type_top1_against_em_reference": float(
                np.mean(type_correct)
            ),
            "spatial_to_source_total_ratio": summarize(
                np.divide(
                    totals,
                    source_expression.sum(axis=1),
                    out=np.full_like(totals, np.nan),
                    where=source_expression.sum(axis=1) > 0,
                )
            ),
        }

    within_compartment = [
        safe_spearman(expression["nuclear_overlap"][i], expression["cytoplasm_only"][i])
        for i in range(len(cell_ids))
    ]
    source_corr = source_ranks @ source_ranks.T
    source_same = []
    source_different = []
    for i in range(len(cell_ids)):
        for j in range(i + 1, len(cell_ids)):
            target = source_same if true_types[i] == true_types[j] else source_different
            target.append(float(source_corr[i, j]))

    pseudo_summary = json.loads(args.pseudo_summary.read_text())
    reference_summary = json.loads(args.reference_summary.read_text())
    shape_summary = json.loads(args.shape_summary.read_text())
    pseudo_parent = "CRC_GSE166555_raw_tumor.h5"
    reference_parent = Path(str(reference_summary["matrix_h5_path"])).name
    same_parent_cohort = reference_parent == pseudo_parent
    provenance = {
        "pseudo_source_sc_h5": str(args.source_sc_h5.resolve()),
        "pseudo_scdesign3_parent_cohort": pseudo_parent,
        "em_reference_npz": str(args.reference_npz.resolve()),
        "em_reference_matrix_h5": reference_summary["matrix_h5_path"],
        "em_reference_parent_cohort": reference_parent,
        "pseudo_and_em_reference_share_parent_cohort": same_parent_cohort,
        "pseudo_expression_origin": "one unique scDesign3 synthetic profile per physical cell",
        "pseudo_count_conservation_passed": pseudo_summary.get(
            "count_conservation_passed"
        ),
        "pseudo_real_reference_cells_assigned": pseudo_summary.get(
            "real_reference_cells_assigned"
        ),
        "pseudo_total_assigned_molecules": pseudo_summary.get(
            "total_assigned_molecules"
        ),
        "pseudo_has_global_spatial_capture_thinning": False,
        "pseudo_has_explicit_ambient_background_component": False,
        "pseudo_compartment_model": (
            "gene-specific nuclear binomial split, membrane-gene routing, and "
            "cytoplasm routing; all retained on-panel molecules are placed"
        ),
        "shape_reference_is_scRNA_derived": False,
        "shape_reference_is_pseudo_gt_mask_derived": True,
        "shape_reference_gt_cell_bins_path": shape_summary.get(
            "gt_cell_bins_path"
        ),
        "shape_reference_n_cells": shape_summary.get("n_cells"),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_cell_path = output_dir / "profile_compartment_per_cell.csv"
    pair_path = output_dir / "source_identification_pairs.csv"
    provenance_path = output_dir / "provenance.json"
    summary_path = output_dir / "summary.json"
    pd.DataFrame(per_cell_rows).to_csv(per_cell_path, index=False)
    pd.DataFrame(pair_rows).to_csv(pair_path, index=False)
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    summary = {
        "n_evaluated_cells": len(cell_ids),
        "n_cell_types": int(len(np.unique(true_types))),
        "n_common_genes": len(genes),
        "compartments": summary_by_compartment,
        "within_cell_nuclear_vs_cytoplasm_spearman_all_genes": summarize(
            within_compartment
        ),
        "source_profile_same_type_pair_spearman_all_genes": summarize(source_same),
        "source_profile_different_type_pair_spearman_all_genes": summarize(
            source_different
        ),
        "provenance": provenance,
        "per_cell_csv": str(per_cell_path),
        "source_identification_pairs_csv": str(pair_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
