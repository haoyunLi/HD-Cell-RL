#!/usr/bin/env python3
"""Prepare donor-disjoint, patch-complete inputs for colorectal pseudo v3.

The output contains every positive ground-truth cell owner whose mask
contributes to a selected patch context.  Expression labels may be explicitly
remapped when a fine subtype cannot be represented on both sides of a strict
donor split.  Geometry and fractional ownership are otherwise unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-index", required=True, type=Path)
    parser.add_argument("--bin-metadata", required=True, type=Path)
    parser.add_argument("--fractional-gt", required=True, type=Path)
    parser.add_argument("--represented-cell-metadata", required=True, type=Path)
    parser.add_argument("--scrna-metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generator-donors", nargs="+", required=True)
    parser.add_argument("--reference-donors", nargs="+", required=True)
    parser.add_argument("--donor-column", default="case_id")
    parser.add_argument("--cell-id-column", default="cell")
    parser.add_argument("--cell-type-column", default="sct_cell_type")
    parser.add_argument("--min-cells-per-type", type=int, default=50)
    parser.add_argument(
        "--cell-type-remap",
        action="append",
        default=[],
        metavar="SOURCE=TARGET",
        help="Explicit expression-label remap; may be supplied more than once.",
    )
    return parser.parse_args()


def parse_remaps(values: list[str]) -> dict[str, str]:
    remaps: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"cell-type remap must be SOURCE=TARGET: {value!r}")
        source, target = (part.strip() for part in value.split("=", 1))
        if not source or not target:
            raise ValueError(f"cell-type remap must have nonempty labels: {value!r}")
        if source in remaps and remaps[source] != target:
            raise ValueError(f"conflicting remaps for {source!r}")
        remaps[source] = target
    return remaps


def select_context_bins(metadata: pd.DataFrame, patches: pd.DataFrame) -> np.ndarray:
    required = {
        "context_x_min",
        "context_x_max",
        "context_y_min",
        "context_y_max",
    }
    missing = required.difference(patches.columns)
    if missing:
        raise ValueError(f"patch index is missing columns: {sorted(missing)}")
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


def parse_cell_id_lists(values: pd.Series) -> set[int]:
    result: set[int] = set()
    for value in values.astype(str):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("patch cell ID fields must contain JSON lists")
        result.update(int(cell_id) for cell_id in parsed)
    return result


def prepare_inputs(args: argparse.Namespace) -> dict[str, object]:
    if args.min_cells_per_type <= 0:
        raise ValueError("--min-cells-per-type must be positive")
    remaps = parse_remaps(list(args.cell_type_remap))
    generator_donors = tuple(str(value) for value in args.generator_donors)
    reference_donors = tuple(str(value) for value in args.reference_donors)
    overlap = set(generator_donors).intersection(reference_donors)
    if overlap:
        raise ValueError(f"generator/reference donors overlap: {sorted(overlap)}")

    paths = (
        args.patch_index,
        args.bin_metadata,
        args.fractional_gt,
        args.represented_cell_metadata,
        args.scrna_metadata,
    )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    patches = pd.read_csv(args.patch_index)
    bin_metadata = pd.read_parquet(
        args.bin_metadata,
        columns=["matrix_col_index", "barcode", "x_um", "y_um"],
    )
    if bin_metadata["matrix_col_index"].duplicated().any():
        raise ValueError("bin metadata contains duplicate matrix_col_index values")
    selected_mask = select_context_bins(bin_metadata, patches)
    selected_bins = bin_metadata.loc[selected_mask].copy()
    if selected_bins.empty:
        raise ValueError("selected patch contexts contain no bins")
    selected_indices = set(selected_bins["matrix_col_index"].astype(np.int64))

    gt = pd.read_parquet(args.fractional_gt)
    required_gt = {"barcode_index", "cell_id", "cell_coverage_fraction"}
    missing_gt = required_gt.difference(gt.columns)
    if missing_gt:
        raise ValueError(f"fractional GT is missing columns: {sorted(missing_gt)}")
    context_gt = gt.loc[
        gt["barcode_index"].astype(np.int64).isin(selected_indices)
    ]
    positive_context_gt = context_gt.loc[
        (pd.to_numeric(context_gt["cell_id"], errors="raise") > 0)
        & (pd.to_numeric(context_gt["cell_coverage_fraction"], errors="raise") > 0)
    ]
    contributing_cell_ids = set(
        positive_context_gt["cell_id"].astype(np.int64).tolist()
    )
    if not contributing_cell_ids:
        raise ValueError("selected contexts have no positive GT cell owners")

    subset_gt = gt.loc[
        gt["cell_id"].astype(np.int64).isin(contributing_cell_ids)
        & (pd.to_numeric(gt["cell_coverage_fraction"], errors="raise") > 0)
    ].copy()
    represented = pd.read_csv(args.represented_cell_metadata)
    required_cells = {"cell_id", "cell_type"}
    missing_cells = required_cells.difference(represented.columns)
    if missing_cells:
        raise ValueError(
            f"represented-cell metadata is missing columns: {sorted(missing_cells)}"
        )
    if represented["cell_id"].duplicated().any():
        raise ValueError("represented-cell metadata has duplicate cell IDs")
    cells = represented.loc[
        represented["cell_id"].astype(np.int64).isin(contributing_cell_ids)
    ].copy()
    found_ids = set(cells["cell_id"].astype(np.int64))
    if found_ids != contributing_cell_ids:
        missing_ids = sorted(contributing_cell_ids.difference(found_ids))
        raise ValueError(f"positive GT owners lack represented metadata: {missing_ids[:10]}")
    cells["original_cell_type"] = cells["cell_type"].astype(str)
    cells["cell_type"] = cells["original_cell_type"].replace(remaps)
    cells = cells.sort_values("cell_id").reset_index(drop=True)

    target_counts = (
        cells.groupby("cell_type", sort=True)
        .size()
        .rename("n")
        .reset_index()
    )
    target_types = target_counts["cell_type"].astype(str).tolist()

    metadata_header = pd.read_csv(args.scrna_metadata, sep="\t", nrows=0)
    requested_columns = [
        args.cell_id_column,
        args.cell_type_column,
        args.donor_column,
        "sample_id",
        "source_id",
    ]
    metadata_columns = list(dict.fromkeys(requested_columns))
    missing_metadata = set(metadata_columns).difference(metadata_header.columns)
    if missing_metadata:
        raise ValueError(
            f"scRNA metadata is missing columns: {sorted(missing_metadata)}"
        )
    scrna = pd.read_csv(
        args.scrna_metadata,
        sep="\t",
        low_memory=False,
        usecols=metadata_columns,
    )
    for column in metadata_columns:
        scrna[column] = scrna[column].astype(str)
    if scrna[args.cell_id_column].duplicated().any():
        raise ValueError("scRNA metadata has duplicate cell IDs")
    available_donors = set(scrna[args.donor_column])
    missing_donors = set(generator_donors + reference_donors).difference(
        available_donors
    )
    if missing_donors:
        raise ValueError(f"requested donors are absent: {sorted(missing_donors)}")

    generator_meta = scrna.loc[
        scrna[args.donor_column].isin(generator_donors)
        & scrna[args.cell_type_column].isin(target_types)
    ].copy()
    reference_meta = scrna.loc[
        scrna[args.donor_column].isin(reference_donors)
        & scrna[args.cell_type_column].isin(target_types)
    ].copy()
    generator_ids = set(generator_meta[args.cell_id_column])
    reference_ids = set(reference_meta[args.cell_id_column])
    if generator_ids.intersection(reference_ids):
        raise AssertionError("generator/reference scRNA cell barcodes overlap")

    donor_counts = pd.DataFrame({"cell_type": target_types})
    for label, frame in (
        ("generator", generator_meta),
        ("reference", reference_meta),
    ):
        counts = frame[args.cell_type_column].value_counts()
        donor_counts[f"{label}_n"] = donor_counts["cell_type"].map(counts).fillna(0).astype(int)
    donor_counts["minimum_required"] = int(args.min_cells_per_type)
    donor_counts["passes"] = (
        (donor_counts["generator_n"] >= args.min_cells_per_type)
        & (donor_counts["reference_n"] >= args.min_cells_per_type)
    )
    if not donor_counts["passes"].all():
        failed = donor_counts.loc[~donor_counts["passes"]]
        raise ValueError(
            "strict donor split lacks target-type coverage:\n"
            + failed.to_string(index=False)
        )

    patch_cell_ids = parse_cell_id_lists(patches["patch_cell_ids"])
    core_cell_ids = parse_cell_id_lists(patches["core_cell_ids"])
    contribution_only_ids = contributing_cell_ids.difference(patch_cell_ids)

    selected_bins.to_parquet(args.output_dir / "selected_context_bins.parquet", index=False)
    subset_gt.to_parquet(args.output_dir / "fractional_gt_contributing_cells.parquet", index=False)
    cells.to_csv(args.output_dir / "ground_truth_cells.csv.gz", index=False)
    target_counts.to_csv(args.output_dir / "target_cell_counts.csv", index=False)
    generator_meta.to_csv(args.output_dir / "generator_metadata.tsv", sep="\t", index=False)
    reference_meta.to_csv(args.output_dir / "reference_metadata.tsv", sep="\t", index=False)
    donor_counts.to_csv(args.output_dir / "donor_split_cell_type_counts.csv", index=False)
    pd.DataFrame({"cell_id": sorted(core_cell_ids)}).to_csv(
        args.output_dir / "core_cell_ids.csv", index=False
    )
    pd.DataFrame({"cell_id": sorted(patch_cell_ids)}).to_csv(
        args.output_dir / "patch_context_cell_ids.csv", index=False
    )
    pd.DataFrame({"cell_id": sorted(contributing_cell_ids)}).to_csv(
        args.output_dir / "expression_contributing_cell_ids.csv", index=False
    )

    summary: dict[str, object] = {
        "version": "colorectal_pseudo_v3_patch_local",
        "patch_index": str(args.patch_index.resolve()),
        "bin_metadata": str(args.bin_metadata.resolve()),
        "fractional_gt": str(args.fractional_gt.resolve()),
        "represented_cell_metadata": str(args.represented_cell_metadata.resolve()),
        "scrna_metadata": str(args.scrna_metadata.resolve()),
        "n_patches": int(len(patches)),
        "n_selected_context_bins": int(len(selected_bins)),
        "n_core_cells": int(len(core_cell_ids)),
        "n_patch_context_cells": int(len(patch_cell_ids)),
        "n_expression_contributing_cells": int(len(contributing_cell_ids)),
        "n_contribution_only_cells": int(len(contribution_only_ids)),
        "n_subset_fractional_gt_pairs": int(len(subset_gt)),
        "generator_donors": list(generator_donors),
        "reference_donors": list(reference_donors),
        "donor_column": str(args.donor_column),
        "generator_reference_donor_overlap": [],
        "generator_reference_cell_barcode_overlap": 0,
        "n_generator_scrna_cells": int(len(generator_meta)),
        "n_reference_scrna_cells": int(len(reference_meta)),
        "min_cells_per_type": int(args.min_cells_per_type),
        "target_cell_types": target_types,
        "cell_type_remap": remaps,
        "geometry_changed": False,
        "shape_prior_built": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    summary = prepare_inputs(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
