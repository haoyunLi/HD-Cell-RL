#!/usr/bin/env python3
"""Create noisy observed nuclear bins while preserving clean GT for evaluation.

Version 1 models mild segmentation error with boundary-biased false negatives,
rare interior false negatives, adjacent false-positive bins, and nucleus-center
jitter.  The output is an observation artifact only; fractional cell GT and
clean nuclear GT are not modified.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.build_nuclei_table_from_nuclear_bins import (  # noqa: E402
    build_nuclei_table,
)


BARCODE_PATTERN = re.compile(r"^s_002um_(\d+)_(\d+)-1$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-nuclear-bins", required=True, type=Path)
    parser.add_argument("--selected-context-bins", required=True, type=Path)
    parser.add_argument("--cell-ids", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--boundary-drop-probability", type=float, default=0.20)
    parser.add_argument("--interior-drop-probability", type=float, default=0.02)
    parser.add_argument("--false-positive-fraction", type=float, default=0.10)
    parser.add_argument("--center-jitter-sd-um", type=float, default=1.0)
    parser.add_argument("--minimum-retained-bins", type=int, default=2)
    parser.add_argument("--bin-size-um", type=float, default=2.0)
    parser.add_argument("--microns-per-pixel", type=float, default=0.2737012522439323)
    parser.add_argument("--seed", type=int, default=44)
    return parser.parse_args()


def validate_probability(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def add_grid_coordinates(context: pd.DataFrame) -> pd.DataFrame:
    context = context.copy()
    if {"array_row", "array_col"}.issubset(context.columns):
        context["array_row"] = context["array_row"].astype(np.int64)
        context["array_col"] = context["array_col"].astype(np.int64)
        return context
    parsed = context["barcode"].astype(str).str.extract(BARCODE_PATTERN)
    if parsed.isna().any().any():
        bad = context.loc[parsed.isna().any(axis=1), "barcode"].head().tolist()
        raise ValueError(f"cannot parse 2 um array coordinates from barcodes: {bad}")
    context["array_row"] = parsed[0].astype(np.int64)
    context["array_col"] = parsed[1].astype(np.int64)
    return context


def boundary_mask_for_group(group: pd.DataFrame) -> np.ndarray:
    coordinates = set(
        zip(
            group["array_row"].astype(int),
            group["array_col"].astype(int),
        )
    )
    result = []
    for row, column in zip(group["array_row"], group["array_col"]):
        neighbors = (
            (int(row) - 1, int(column)),
            (int(row) + 1, int(column)),
            (int(row), int(column) - 1),
            (int(row), int(column) + 1),
        )
        result.append(any(neighbor not in coordinates for neighbor in neighbors))
    return np.asarray(result, dtype=bool)


def build_noisy_observations(args: argparse.Namespace) -> dict[str, object]:
    boundary_drop = validate_probability(
        args.boundary_drop_probability, "boundary-drop probability"
    )
    interior_drop = validate_probability(
        args.interior_drop_probability, "interior-drop probability"
    )
    false_positive_fraction = validate_probability(
        args.false_positive_fraction, "false-positive fraction"
    )
    if args.center_jitter_sd_um < 0:
        raise ValueError("center-jitter SD must be nonnegative")
    if args.minimum_retained_bins <= 0:
        raise ValueError("minimum retained bins must be positive")
    for path in (args.clean_nuclear_bins, args.selected_context_bins, args.cell_ids):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cell_ids_frame = pd.read_csv(args.cell_ids)
    if "cell_id" not in cell_ids_frame.columns:
        raise ValueError("cell-ID table must contain cell_id")
    selected_cell_ids = set(cell_ids_frame["cell_id"].astype(np.int64))
    if not selected_cell_ids:
        raise ValueError("cell-ID table is empty")
    clean = pd.read_csv(args.clean_nuclear_bins, compression="infer")
    required_clean = {
        "barcode",
        "array_row",
        "array_col",
        "cell_id",
        "cell_type",
        "x_um",
        "y_um",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
        "nuclear_fraction",
    }
    missing_clean = required_clean.difference(clean.columns)
    if missing_clean:
        raise ValueError(f"clean nuclear bins are missing columns: {sorted(missing_clean)}")
    clean = clean.loc[clean["cell_id"].astype(np.int64).isin(selected_cell_ids)].copy()

    context = add_grid_coordinates(pd.read_parquet(args.selected_context_bins))
    required_context = {"barcode", "x_um", "y_um", "array_row", "array_col"}
    missing_context = required_context.difference(context.columns)
    if missing_context:
        raise ValueError(
            f"selected context bins are missing columns: {sorted(missing_context)}"
        )
    context = context.drop_duplicates("barcode", keep="first").copy()
    context_barcodes = set(context["barcode"].astype(str))
    clean = clean.loc[clean["barcode"].astype(str).isin(context_barcodes)].copy()
    found_cell_ids = set(clean["cell_id"].astype(np.int64))
    if found_cell_ids != selected_cell_ids:
        missing_ids = sorted(selected_cell_ids.difference(found_cell_ids))
        raise ValueError(f"selected cells lack in-context clean nuclear bins: {missing_ids[:10]}")
    if clean["barcode"].duplicated().any():
        raise ValueError("clean dominant nuclear bins must have unique physical barcodes")
    context_by_grid = {
        (int(row.array_row), int(row.array_col)): row
        for row in context.itertuples(index=False)
    }
    clean_barcodes = set(clean["barcode"].astype(str))

    rng = np.random.default_rng(int(args.seed))
    retained_groups: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    cell_type_by_id = (
        clean[["cell_id", "cell_type"]]
        .drop_duplicates("cell_id")
        .set_index("cell_id")["cell_type"]
        .to_dict()
    )
    for cell_id, group in clean.groupby("cell_id", sort=True):
        group = group.copy().reset_index(drop=True)
        boundary = boundary_mask_for_group(group)
        drop_probabilities = np.where(boundary, boundary_drop, interior_drop)
        keep = rng.random(len(group)) >= drop_probabilities
        minimum = min(int(args.minimum_retained_bins), len(group))
        if int(keep.sum()) < minimum:
            priority = np.argsort(drop_probabilities, kind="stable")[:minimum]
            keep[priority] = True
        retained = group.loc[keep].copy()
        retained["noise_origin"] = "retained_clean_nuclear"
        retained_groups.append(retained)
        diagnostics.append(
            {
                "cell_id": int(cell_id),
                "cell_type": str(cell_type_by_id[int(cell_id)]),
                "clean_bins": int(len(group)),
                "clean_boundary_bins": int(boundary.sum()),
                "retained_clean_bins": int(len(retained)),
                "dropped_boundary_bins": int((~keep & boundary).sum()),
                "dropped_interior_bins": int((~keep & ~boundary).sum()),
            }
        )

    observed = pd.concat(retained_groups, ignore_index=True)
    claimed_barcodes = set(observed["barcode"].astype(str))
    false_positive_rows: list[dict[str, object]] = []
    diagnostic_by_cell = {int(row["cell_id"]): row for row in diagnostics}
    for cell_id, group in observed.groupby("cell_id", sort=True):
        clean_count = int((clean["cell_id"].astype(np.int64) == int(cell_id)).sum())
        requested = int(np.rint(false_positive_fraction * clean_count))
        candidate_records: dict[str, object] = {}
        for row, column in zip(group["array_row"], group["array_col"]):
            for coordinate in (
                (int(row) - 1, int(column)),
                (int(row) + 1, int(column)),
                (int(row), int(column) - 1),
                (int(row), int(column) + 1),
            ):
                candidate = context_by_grid.get(coordinate)
                if candidate is None:
                    continue
                barcode = str(candidate.barcode)
                if barcode in clean_barcodes or barcode in claimed_barcodes:
                    continue
                candidate_records[barcode] = candidate
        available = list(candidate_records.values())
        if requested > 0 and available:
            chosen = rng.choice(
                len(available), size=min(requested, len(available)), replace=False
            )
            for index in np.atleast_1d(chosen):
                candidate = available[int(index)]
                barcode = str(candidate.barcode)
                claimed_barcodes.add(barcode)
                x_um = float(candidate.x_um)
                y_um = float(candidate.y_um)
                false_positive_rows.append(
                    {
                        "bin_id": barcode,
                        "barcode": barcode,
                        "array_row": int(candidate.array_row),
                        "array_col": int(candidate.array_col),
                        "in_tissue": 1,
                        "x_um": x_um,
                        "y_um": y_um,
                        "pxl_row_in_fullres": y_um / float(args.microns_per_pixel),
                        "pxl_col_in_fullres": x_um / float(args.microns_per_pixel),
                        "cell_id": int(cell_id),
                        "cell_type": str(cell_type_by_id[int(cell_id)]),
                        "is_nuclear": 1,
                        "is_cytoplasm": 0,
                        "is_boundary": 1,
                        "is_interior": 1,
                        "weight": 1.0,
                        "cell_coverage_fraction": 1.0,
                        "nuclear_fraction": 1.0,
                        "cytoplasm_fraction": 0.0,
                        "membrane_fraction": 0.0,
                        "noise_origin": "adjacent_false_positive",
                    }
                )
        diagnostic_by_cell[int(cell_id)]["requested_false_positive_bins"] = requested
        diagnostic_by_cell[int(cell_id)]["added_false_positive_bins"] = min(
            requested, len(available)
        )

    if false_positive_rows:
        observed = pd.concat(
            [observed, pd.DataFrame(false_positive_rows)],
            ignore_index=True,
            sort=False,
        )
    if observed["barcode"].duplicated().any():
        raise AssertionError("noisy observed nuclear bins lost unique barcode ownership")
    if set(observed["cell_id"].astype(np.int64)) != selected_cell_ids:
        raise AssertionError("nuclear noise removed an entire selected nucleus")
    observed = observed.sort_values(["cell_id", "array_row", "array_col"]).reset_index(drop=True)

    nuclei = build_nuclei_table(
        observed,
        bin_size_um=float(args.bin_size_um),
        microns_per_pixel=float(args.microns_per_pixel),
    )
    jitter_x = rng.normal(0.0, float(args.center_jitter_sd_um), size=len(nuclei))
    jitter_y = rng.normal(0.0, float(args.center_jitter_sd_um), size=len(nuclei))
    nuclei["pre_jitter_center_x_um"] = nuclei["center_x_um"]
    nuclei["pre_jitter_center_y_um"] = nuclei["center_y_um"]
    nuclei["center_jitter_x_um"] = jitter_x
    nuclei["center_jitter_y_um"] = jitter_y
    nuclei["center_x_um"] += jitter_x
    nuclei["center_y_um"] += jitter_y
    nuclei["center_x_fullres_px"] = (
        nuclei["center_x_um"] / float(args.microns_per_pixel)
    )
    nuclei["center_y_fullres_px"] = (
        nuclei["center_y_um"] / float(args.microns_per_pixel)
    )

    diagnostics_frame = pd.DataFrame(diagnostics).sort_values("cell_id")
    diagnostics_dir = args.output_dir / "diagnostics"
    diagnostics_dir.mkdir(exist_ok=True)
    observed_path = args.output_dir / "observed_nuclear_bins.csv.gz"
    nuclei_path = args.output_dir / "observed_nuclei.parquet"
    observed.to_csv(observed_path, index=False, compression="gzip")
    nuclei.to_parquet(nuclei_path, index=False)
    diagnostics_frame.to_csv(diagnostics_dir / "nuclear_noise_per_cell.csv", index=False)

    n_false_positive = int(
        observed["noise_origin"].eq("adjacent_false_positive").sum()
    )
    summary: dict[str, object] = {
        "clean_nuclear_bins": str(args.clean_nuclear_bins.resolve()),
        "selected_context_bins": str(args.selected_context_bins.resolve()),
        "cell_ids": str(args.cell_ids.resolve()),
        "n_cells": int(len(selected_cell_ids)),
        "n_clean_nuclear_bins": int(len(clean)),
        "n_retained_clean_nuclear_bins": int(
            observed["noise_origin"].eq("retained_clean_nuclear").sum()
        ),
        "n_dropped_clean_nuclear_bins": int(
            len(clean) - observed["noise_origin"].eq("retained_clean_nuclear").sum()
        ),
        "n_added_false_positive_bins": n_false_positive,
        "n_observed_nuclear_bins": int(len(observed)),
        "unique_observed_barcode_ownership": True,
        "all_nuclei_retained": True,
        "boundary_drop_probability": boundary_drop,
        "interior_drop_probability": interior_drop,
        "false_positive_fraction": false_positive_fraction,
        "center_jitter_sd_um": float(args.center_jitter_sd_um),
        "realized_center_jitter_rms_um": float(
            np.sqrt(np.mean(jitter_x**2 + jitter_y**2))
        ),
        "minimum_retained_bins": int(args.minimum_retained_bins),
        "seed": int(args.seed),
        "clean_gt_modified": False,
        "shape_prior_built": False,
        "observed_nuclear_bins_path": str(observed_path.resolve()),
        "observed_nuclei_path": str(nuclei_path.resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    print(json.dumps(build_noisy_observations(parse_args()), indent=2))


if __name__ == "__main__":
    main()
