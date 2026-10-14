#!/usr/bin/env python
"""Build one physical-nucleus row per cell from weighted 2 um nuclear bins."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "cell_id",
    "x_um",
    "y_um",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
    "nuclear_fraction",
)


def build_nuclei_table(
    nuclear_bins: pd.DataFrame,
    *,
    bin_size_um: float,
    microns_per_pixel: float,
) -> pd.DataFrame:
    """Aggregate fractional nuclear-bin coverage into centers and equivalent radii."""
    if bin_size_um <= 0.0:
        raise ValueError("bin_size_um must be > 0")
    if microns_per_pixel <= 0.0:
        raise ValueError("microns_per_pixel must be > 0")
    missing = [column for column in REQUIRED_COLUMNS if column not in nuclear_bins.columns]
    if missing:
        raise ValueError(f"nuclear bin table is missing required columns: {missing}")

    bins = nuclear_bins.loc[:, REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        bins[column] = pd.to_numeric(bins[column], errors="raise")
    bins = bins.loc[(bins["cell_id"] > 0) & (bins["nuclear_fraction"] > 0.0)].copy()
    if bins.empty:
        raise ValueError("nuclear bin table contains no positive nuclear assignments")
    if not np.isfinite(bins.to_numpy(dtype=np.float64)).all():
        raise ValueError("nuclear bin table contains NaN or Inf")
    if (bins["nuclear_fraction"] > 1.0 + 1.0e-8).any():
        raise ValueError("nuclear_fraction must lie in (0, 1]")

    bins["weighted_x_um"] = bins["nuclear_fraction"] * bins["x_um"]
    bins["weighted_y_um"] = bins["nuclear_fraction"] * bins["y_um"]
    grouped = (
        bins.groupby("cell_id", sort=True)
        .agg(
            nuclear_fraction_sum=("nuclear_fraction", "sum"),
            nuclear_bin_count=("nuclear_fraction", "size"),
            weighted_x_um=("weighted_x_um", "sum"),
            weighted_y_um=("weighted_y_um", "sum"),
            min_x_fullres_px=("pxl_col_in_fullres", "min"),
            max_x_fullres_px=("pxl_col_in_fullres", "max"),
            min_y_fullres_px=("pxl_row_in_fullres", "min"),
            max_y_fullres_px=("pxl_row_in_fullres", "max"),
        )
        .reset_index()
    )
    grouped["center_x_um"] = grouped["weighted_x_um"] / grouped["nuclear_fraction_sum"]
    grouped["center_y_um"] = grouped["weighted_y_um"] / grouped["nuclear_fraction_sum"]
    grouped["nuclear_area_um2"] = (
        grouped["nuclear_fraction_sum"] * float(bin_size_um) ** 2
    )
    grouped["radius_um"] = np.sqrt(grouped["nuclear_area_um2"] / math.pi)
    grouped["nuclear_pixel_count"] = np.rint(
        grouped["nuclear_area_um2"] / float(microns_per_pixel) ** 2
    ).astype(np.int64)
    grouped["center_x_fullres_px"] = grouped["center_x_um"] / float(microns_per_pixel)
    grouped["center_y_fullres_px"] = grouped["center_y_um"] / float(microns_per_pixel)
    half_bin_px = 0.5 * float(bin_size_um) / float(microns_per_pixel)
    grouped["bbox_min_x_fullres_px"] = np.floor(
        grouped["min_x_fullres_px"] - half_bin_px
    ).astype(np.int64)
    grouped["bbox_max_x_fullres_px"] = np.ceil(
        grouped["max_x_fullres_px"] + half_bin_px
    ).astype(np.int64)
    grouped["bbox_min_y_fullres_px"] = np.floor(
        grouped["min_y_fullres_px"] - half_bin_px
    ).astype(np.int64)
    grouped["bbox_max_y_fullres_px"] = np.ceil(
        grouped["max_y_fullres_px"] + half_bin_px
    ).astype(np.int64)

    columns = (
        "cell_id",
        "center_x_um",
        "center_y_um",
        "radius_um",
        "nuclear_pixel_count",
        "center_x_fullres_px",
        "center_y_fullres_px",
        "bbox_min_x_fullres_px",
        "bbox_max_x_fullres_px",
        "bbox_min_y_fullres_px",
        "bbox_max_y_fullres_px",
        "nuclear_bin_count",
        "nuclear_area_um2",
    )
    return grouped.loc[:, columns].sort_values("cell_id").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuclear-bins-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--summary-path", required=True, type=Path)
    parser.add_argument("--bin-size-um", type=float, default=2.0)
    parser.add_argument("--microns-per-pixel", type=float, default=0.2737012522439323)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.nuclear_bins_path.exists():
        raise FileNotFoundError(args.nuclear_bins_path)
    bins = pd.read_csv(
        args.nuclear_bins_path,
        usecols=list(REQUIRED_COLUMNS),
        compression="infer",
    )
    nuclei = build_nuclei_table(
        bins,
        bin_size_um=float(args.bin_size_um),
        microns_per_pixel=float(args.microns_per_pixel),
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    nuclei.to_parquet(args.output_path, index=False)
    summary = {
        "nuclear_bins_path": str(args.nuclear_bins_path.resolve()),
        "output_path": str(args.output_path.resolve()),
        "n_nuclear_bin_rows": int(len(bins)),
        "n_nuclei": int(len(nuclei)),
        "bin_size_um": float(args.bin_size_um),
        "microns_per_pixel": float(args.microns_per_pixel),
        "mean_radius_um": float(nuclei["radius_um"].mean()),
        "median_radius_um": float(nuclei["radius_um"].median()),
        "max_radius_um": float(nuclei["radius_um"].max()),
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
