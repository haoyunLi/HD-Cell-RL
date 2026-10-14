#!/usr/bin/env python3
"""Audit official Visium HD bin and Xenium cell coordinates before benchmarking.

The two assays share the imported H&E raster, but not a pixel coordinate system:
Visium bins use H&E full-resolution pixels, while Xenium geometry is in microns
in the DAPI frame. 10x's ``he_imagealignment.csv`` maps H&E pixels to DAPI
pixels; this script inverts it for plotting Xenium cells on H&E.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import cv2


def geometry_from_official_files(root: Path) -> dict[str, object]:
    hd = root / "visium_hd/binned_outputs/square_002um/spatial"
    xenium = root / "xenium"
    scale = json.loads((hd / "scalefactors_json.json").read_text())
    experiment = json.loads((xenium / "extracted/experiment.xenium").read_text())
    h = np.loadtxt(
        xenium / "supplemental/Xenium_Prime_Human_Lung_Cancer_FFPE_he_imagealignment.csv",
        delimiter=",",
    )
    if h.shape != (3, 3) or not np.allclose(h[2], [0.0, 0.0, 1.0]):
        raise ValueError("Xenium H&E alignment is not a 3x3 affine transform")
    return {
        "hd_microns_per_pixel": float(scale["microns_per_pixel"]),
        "hd_bin_size_um": float(scale["bin_size_um"]),
        "hd_spot_diameter_fullres_px": float(scale["spot_diameter_fullres"]),
        "xenium_microns_per_pixel": float(experiment["pixel_size"]),
        "he_to_xenium_pixels": h,
    }


def transform_xenium_um_to_he_px(xy_um: np.ndarray, geometry: dict[str, object]) -> np.ndarray:
    """Map Xenium micron (x,y) coordinates into full-resolution H&E pixels."""
    xy = np.asarray(xy_um, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("Xenium coordinates must have shape (N, 2)")
    homogeneous = np.column_stack((xy / float(geometry["xenium_microns_per_pixel"]), np.ones(len(xy))))
    return (homogeneous @ np.linalg.inv(geometry["he_to_xenium_pixels"]).T)[:, :2]


def fit_hd_grid_to_he_px(positions: pd.DataFrame) -> np.ndarray:
    """Fit the projective HD grid transform encoded by official bin centers."""
    sample = positions.iloc[np.linspace(0, len(positions) - 1, min(len(positions), 10000), dtype=np.int64)]
    grid = sample[["array_col", "array_row"]].to_numpy(dtype=np.float64)
    pixels = sample[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=np.float64)
    h, _ = cv2.findHomography(grid, pixels, method=0)
    if h is None:
        raise ValueError("Could not estimate HD projective transform")
    predicted = project_points(grid, h)
    if float(np.max(np.abs(predicted - pixels))) > 1e-3:
        raise ValueError("HD tissue_positions cannot be represented by one projective transform")
    return h


def project_points(xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    mapped = np.column_stack((xy, np.ones(len(xy)))) @ matrix.T
    return mapped[:, :2] / mapped[:, 2, None]


def run_audit(root: Path, row_start: int, col_start: int, size_bins: int) -> dict[str, object]:
    geometry = geometry_from_official_files(root)
    pos = pd.read_parquet(root / "visium_hd/binned_outputs/square_002um/spatial/tissue_positions.parquet")
    if pos.duplicated(["array_row", "array_col"]).any():
        raise ValueError("HD tissue positions contain duplicate grid coordinates")
    a = fit_hd_grid_to_he_px(pos)
    hd_mpp = float(geometry["hd_microns_per_pixel"])
    center = np.array([[col_start + size_bins / 2, row_start + size_bins / 2]], dtype=np.float64)
    center_px = project_points(center, a)
    observed_col_step = float(np.linalg.norm(project_points(center + [1, 0], a) - center_px) * hd_mpp)
    observed_row_step = float(np.linalg.norm(project_points(center + [0, 1], a) - center_px) * hd_mpp)
    nominal = float(geometry["hd_bin_size_um"])
    if not (np.isclose(observed_col_step, nominal, atol=0.01) and np.isclose(observed_row_step, nominal, atol=0.01)):
        raise ValueError("HD array step does not match official 2-um bin size")
    spot_um = float(geometry["hd_spot_diameter_fullres_px"]) * hd_mpp
    if not np.isclose(spot_um, nominal, atol=0.01):
        raise ValueError("HD spot diameter and microns_per_pixel disagree")

    cells = pd.read_parquet(root / "xenium/extracted/cells.parquet", columns=["cell_id", "x_centroid", "y_centroid"])
    he_xy = transform_xenium_um_to_he_px(cells[["x_centroid", "y_centroid"]].to_numpy(), geometry)
    grid = project_points(he_xy, np.linalg.inv(a))
    row = np.rint(grid[:, 1]).astype(np.int64)
    col = np.rint(grid[:, 0]).astype(np.int64)
    max_row = int(pos.array_row.max()) + 1
    max_col = int(pos.array_col.max()) + 1
    valid = (row >= 0) & (row < max_row) & (col >= 0) & (col < max_col)
    tissue_grid = np.zeros((max_row, max_col), dtype=bool)
    tissue_grid[pos.array_row.to_numpy(dtype=np.int64), pos.array_col.to_numpy(dtype=np.int64)] = pos.in_tissue.to_numpy(dtype=bool)
    on_tissue = np.zeros(len(cells), dtype=bool)
    on_tissue[valid] = tissue_grid[row[valid], col[valid]]
    in_patch = (
        (grid[:, 1] >= row_start) & (grid[:, 1] < row_start + size_bins)
        & (grid[:, 0] >= col_start) & (grid[:, 0] < col_start + size_bins)
    )
    patch = pos.loc[
        pos.array_row.between(row_start, row_start + size_bins - 1)
        & pos.array_col.between(col_start, col_start + size_bins - 1)
    ]
    if len(patch) != size_bins * size_bins:
        raise ValueError("Requested patch extends outside the HD array grid")
    if int(in_patch.sum()) < 20 or float(patch.in_tissue.mean()) < 0.95:
        raise ValueError("Pilot patch lacks enough Xenium cells or HD tissue support")
    # Projective and unit round-trips are independent arithmetic checks; physical
    # landmark overlays remain necessary before trusting the biological match.
    sample = np.linspace(0, len(cells) - 1, 100, dtype=np.int64)
    dapi = np.column_stack((he_xy[sample], np.ones(len(sample)))) @ geometry["he_to_xenium_pixels"].T
    roundtrip_um = dapi[:, :2] * float(geometry["xenium_microns_per_pixel"])
    max_roundtrip_um = float(np.max(np.abs(roundtrip_um - cells.loc[sample, ["x_centroid", "y_centroid"]].to_numpy())))
    if max_roundtrip_um > 1e-5:
        raise ValueError("Xenium/H&E transform fails round-trip validation")
    return {
        "hd_microns_per_pixel": hd_mpp,
        "hd_bin_size_um": nominal,
        "hd_spot_diameter_um": spot_um,
        "hd_row_step_um": observed_row_step,
        "hd_col_step_um": observed_col_step,
        "xenium_microns_per_pixel": float(geometry["xenium_microns_per_pixel"]),
        "he_to_xenium_pixels": geometry["he_to_xenium_pixels"].tolist(),
        "hd_grid_to_he_pixels_projective": a.tolist(),
        "max_xenium_roundtrip_error_um": max_roundtrip_um,
        "xenium_cells_total": int(len(cells)),
        "xenium_centroids_in_hd_grid": int(valid.sum()),
        "xenium_centroids_on_hd_tissue": int(on_tissue.sum()),
        "pilot_array_row_start": row_start,
        "pilot_array_col_start": col_start,
        "pilot_size_bins": size_bins,
        "pilot_tissue_fraction": float(patch.in_tissue.mean()),
        "pilot_xenium_centroids": int(in_patch.sum()),
        "pilot_xenium_centroids_on_hd_tissue": int((in_patch & on_tissue).sum()),
        "note": "Xenium is evaluation-only; arithmetic QC does not replace image-overlay QC.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/paired_hd_xenium/lung_cancer_experiment2"))
    parser.add_argument("--row-start", type=int, default=1840)
    parser.add_argument("--col-start", type=int, default=1240)
    parser.add_argument("--size-bins", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_audit(args.data_root, args.row_start, args.col_start, args.size_bins)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
