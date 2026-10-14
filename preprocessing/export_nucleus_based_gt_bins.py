#!/usr/bin/env python
"""Export fractional nucleus-based ground truth into the RL bin-table schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_BIN_COLUMNS = (
    "barcode_index",
    "array_row",
    "array_col",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
    "cell_id",
    "cell_coverage_fraction",
    "nuclear_fraction",
    "cytoplasm_fraction",
    "membrane_fraction",
)


def export_gt_bins(
    *,
    bin_assignments_path: Path,
    barcodes_path: Path,
    cell_assignments_path: Path,
    cell_output_path: Path,
    nuclear_output_path: Path,
    summary_path: Path,
    microns_per_pixel: float,
) -> dict[str, object]:
    if microns_per_pixel <= 0:
        raise ValueError("microns_per_pixel must be > 0")
    for path in (bin_assignments_path, barcodes_path, cell_assignments_path):
        if not path.exists():
            raise FileNotFoundError(path)

    bins = pd.read_parquet(bin_assignments_path, columns=list(REQUIRED_BIN_COLUMNS))
    bins = bins.loc[
        (pd.to_numeric(bins["cell_id"], errors="coerce").fillna(0) > 0)
        & (pd.to_numeric(bins["cell_coverage_fraction"], errors="coerce").fillna(0) > 0)
    ].copy()
    if bins.empty:
        raise ValueError("bin assignment table contains no assigned cells")

    barcodes = pd.read_csv(
        barcodes_path,
        sep="\t",
        header=None,
        usecols=[0],
        names=["barcode"],
        compression="infer",
        dtype="string",
    )["barcode"].astype(str).to_numpy()
    barcode_indices = pd.to_numeric(bins["barcode_index"], errors="raise").to_numpy(dtype=np.int64)
    if barcode_indices.min() < 0 or barcode_indices.max() >= len(barcodes):
        raise ValueError("barcode_index is outside the supplied barcode table")
    if bins.duplicated(["barcode_index", "cell_id"]).any():
        raise ValueError(
            "bin assignment table contains duplicate (barcode_index, cell_id) pairs"
        )
    compartment_sum = bins[
        ["nuclear_fraction", "cytoplasm_fraction", "membrane_fraction"]
    ].sum(axis=1)
    if not np.allclose(
        compartment_sum.to_numpy(dtype=np.float64),
        bins["cell_coverage_fraction"].to_numpy(dtype=np.float64),
        atol=1e-7,
    ):
        raise ValueError("compartment fractions do not conserve cell coverage")
    total_coverage = bins.groupby("barcode_index", sort=False)[
        "cell_coverage_fraction"
    ].sum()
    if (total_coverage > 1.0 + 1e-7).any():
        raise ValueError("summed cell coverage exceeds one for at least one barcode")
    bins["barcode"] = barcodes[barcode_indices]
    bins["bin_id"] = bins["barcode"]

    assignments = pd.read_csv(cell_assignments_path, usecols=["cell_id", "cell_type"])
    assignments = assignments.drop_duplicates("cell_id", keep="first")
    bins = bins.merge(assignments, on="cell_id", how="left", validate="many_to_one")

    bins["in_tissue"] = np.int8(1)
    bins["x_um"] = bins["pxl_col_in_fullres"].astype(np.float64) * float(microns_per_pixel)
    bins["y_um"] = bins["pxl_row_in_fullres"].astype(np.float64) * float(microns_per_pixel)
    bins["is_nuclear"] = (bins["nuclear_fraction"] > 0).astype(np.int8)
    bins["is_cytoplasm"] = (bins["cytoplasm_fraction"] > 0).astype(np.int8)
    bins["is_boundary"] = (bins["membrane_fraction"] > 0).astype(np.int8)
    bins["is_interior"] = ((bins["nuclear_fraction"] + bins["cytoplasm_fraction"]) > 0).astype(np.int8)
    bins["weight"] = bins["cell_coverage_fraction"].astype(np.float64)

    output_columns = [
        "bin_id",
        "barcode",
        "array_row",
        "array_col",
        "in_tissue",
        "x_um",
        "y_um",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
        "cell_id",
        "cell_type",
        "is_nuclear",
        "is_cytoplasm",
        "is_boundary",
        "is_interior",
        "weight",
        "cell_coverage_fraction",
        "nuclear_fraction",
        "cytoplasm_fraction",
        "membrane_fraction",
    ]
    cell_bins = bins.loc[:, output_columns].reset_index(drop=True)
    nuclear_bins = cell_bins.loc[cell_bins["is_nuclear"] == 1].copy()

    for path, frame in ((cell_output_path, cell_bins), (nuclear_output_path, nuclear_bins)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, compression="infer")

    owners_per_barcode = cell_bins.groupby("barcode", sort=False)["cell_id"].nunique()
    summary: dict[str, object] = {
        "bin_assignments_path": str(bin_assignments_path.resolve()),
        "barcodes_path": str(barcodes_path.resolve()),
        "cell_assignments_path": str(cell_assignments_path.resolve()),
        "cell_output_path": str(cell_output_path.resolve()),
        "nuclear_output_path": str(nuclear_output_path.resolve()),
        "microns_per_pixel": float(microns_per_pixel),
        "n_cell_bin_rows": int(len(cell_bins)),
        "n_unique_cell_barcodes": int(cell_bins["barcode"].nunique()),
        "n_multi_cell_barcodes": int((owners_per_barcode > 1).sum()),
        "fraction_multi_cell_barcodes": float((owners_per_barcode > 1).mean()),
        "max_cells_per_barcode": int(owners_per_barcode.max()),
        "n_nuclear_bin_rows": int(len(nuclear_bins)),
        "n_cells": int(cell_bins["cell_id"].nunique()),
        "n_missing_cell_types": int(cell_bins["cell_type"].isna().sum()),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-assignments-path", required=True, type=Path)
    parser.add_argument("--barcodes-path", required=True, type=Path)
    parser.add_argument("--cell-assignments-path", required=True, type=Path)
    parser.add_argument("--cell-output-path", required=True, type=Path)
    parser.add_argument("--nuclear-output-path", required=True, type=Path)
    parser.add_argument("--summary-path", required=True, type=Path)
    parser.add_argument("--microns-per-pixel", type=float, default=0.2737012522439323)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_gt_bins(
        bin_assignments_path=args.bin_assignments_path,
        barcodes_path=args.barcodes_path,
        cell_assignments_path=args.cell_assignments_path,
        cell_output_path=args.cell_output_path,
        nuclear_output_path=args.nuclear_output_path,
        summary_path=args.summary_path,
        microns_per_pixel=args.microns_per_pixel,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
