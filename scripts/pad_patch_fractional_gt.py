#!/usr/bin/env python3
"""Pad a physical-cell subset of fractional GT to the full barcode axis.

The legacy pseudo-HD molecule-localization script validates that its bin table
contains every original barcode, even when expression is generated for only a
patch-local subset of physical cells.  This utility preserves every positive
pair for the requested cells and adds a single zero-coverage background row for
each otherwise uncovered barcode.  No expression or positive ownership is
invented by the padding rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FRACTION_COLUMNS = (
    "cell_coverage_fraction",
    "nuclear_fraction",
    "cytoplasm_fraction",
    "membrane_fraction",
)


def build_padded_fractional_gt(
    full_gt: pd.DataFrame,
    contributing_cell_ids: set[int],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return positive requested-cell pairs plus zero rows on missing barcodes."""

    required = {
        "barcode_index",
        "array_row",
        "array_col",
        "cell_id",
        *FRACTION_COLUMNS,
    }
    missing = required.difference(full_gt.columns)
    if missing:
        raise ValueError(f"fractional GT is missing columns: {sorted(missing)}")
    if not contributing_cell_ids:
        raise ValueError("contributing-cell set is empty")

    frame = full_gt.copy()
    frame["barcode_index"] = pd.to_numeric(
        frame["barcode_index"], errors="raise"
    ).astype(np.int64)
    frame["cell_id"] = pd.to_numeric(frame["cell_id"], errors="raise").astype(
        np.int64
    )
    if frame.duplicated(["barcode_index", "cell_id"]).any():
        raise ValueError("fractional GT contains duplicate barcode/cell pairs")
    barcode_indices = np.sort(frame["barcode_index"].unique())
    expected = np.arange(len(barcode_indices), dtype=np.int64)
    if not np.array_equal(barcode_indices, expected):
        raise ValueError("fractional GT does not span a contiguous zero-based barcode axis")

    fractions = frame.loc[:, FRACTION_COLUMNS].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(fractions).all() or (fractions < -1.0e-12).any():
        raise ValueError("fractional GT contains invalid fractions")

    positive = frame.loc[
        frame["cell_id"].isin(contributing_cell_ids)
        & (frame["cell_coverage_fraction"] > 0)
    ].copy()
    found = set(positive["cell_id"].astype(np.int64))
    if found != contributing_cell_ids:
        absent = sorted(contributing_cell_ids.difference(found))
        raise ValueError(f"requested cells have no positive GT pairs: {absent[:10]}")

    coordinate_rows = (
        frame.sort_values(["barcode_index", "cell_id"])
        .drop_duplicates("barcode_index", keep="first")
        .loc[:, ["barcode_index", "array_row", "array_col"]]
    )
    covered = set(positive["barcode_index"].astype(np.int64))
    padding = coordinate_rows.loc[
        ~coordinate_rows["barcode_index"].isin(covered)
    ].copy()
    padding["cell_id"] = np.int64(0)
    for column in FRACTION_COLUMNS:
        padding[column] = 0.0

    # Keep the full schema used by the source table.  Fields unrelated to the
    # required geometry/fractions are explicitly missing on zero-coverage rows.
    for column in frame.columns:
        if column not in padding.columns:
            padding[column] = pd.NA
    padding = padding.loc[:, frame.columns]
    result = pd.concat([positive, padding], ignore_index=True)
    result = result.sort_values(["barcode_index", "cell_id"]).reset_index(drop=True)

    if not np.array_equal(
        np.sort(result["barcode_index"].unique()), expected
    ):
        raise AssertionError("padded table does not cover the full barcode axis")
    if result.duplicated(["barcode_index", "cell_id"]).any():
        raise AssertionError("padded table introduced duplicate pairs")
    if (result.loc[result["cell_id"].eq(0), list(FRACTION_COLUMNS)] != 0).any().any():
        raise AssertionError("padding rows must remain exactly zero coverage")

    owners_per_bin = positive.groupby("barcode_index")["cell_id"].nunique()
    summary = {
        "n_barcodes": int(len(expected)),
        "n_requested_cells": int(len(contributing_cell_ids)),
        "n_positive_pairs": int(len(positive)),
        "n_positive_barcodes": int(positive["barcode_index"].nunique()),
        "n_zero_padding_rows": int(len(padding)),
        "n_multi_owner_positive_barcodes": int((owners_per_bin > 1).sum()),
        "max_positive_owners_per_barcode": int(owners_per_bin.max()),
    }
    return result, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-fractional-gt", required=True, type=Path)
    parser.add_argument("--contributing-cell-ids", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.full_fractional_gt, args.contributing_cell_ids):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (args.output, args.summary):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite output: {path}")

    ids = pd.read_csv(args.contributing_cell_ids)
    if "cell_id" not in ids:
        raise ValueError("contributing-cell CSV is missing cell_id")
    cell_ids = set(pd.to_numeric(ids["cell_id"], errors="raise").astype(np.int64))
    padded, summary = build_padded_fractional_gt(
        pd.read_parquet(args.full_fractional_gt), cell_ids
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    padded.to_parquet(args.output, index=False)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
