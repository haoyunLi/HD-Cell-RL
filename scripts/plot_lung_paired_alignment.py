#!/usr/bin/env python3
"""Visual QC: overlay official HD bin centers and Xenium cell boundaries on H&E."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from qc_lung_paired_alignment import geometry_from_official_files, transform_xenium_um_to_he_px


def read_uncompressed_tiled_crop(path: Path, box: tuple[int, int, int, int]) -> np.ndarray:
    """Read only intersecting TIFF tiles; the 10x full-resolution BTF is 3.3 GB."""
    x0, y0, x1, y1 = box
    with tifffile.TiffFile(path) as tiff:
        page = tiff.pages[0]
        if int(page.compression) != 1 or int(page.predictor) != 1 or page.dtype != np.uint8:
            raise ValueError("Expected uncompressed uint8 10x tissue image; use a general TIFF reader otherwise")
        height, width, channels = page.shape
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height and channels == 3):
            raise ValueError("Requested crop outside full-resolution H&E image")
        tile_h, tile_w = int(page.tilelength), int(page.tilewidth)
        tiles_per_row = (width + tile_w - 1) // tile_w
        out = np.empty((y1 - y0, x1 - x0, channels), dtype=np.uint8)
        with path.open("rb") as handle:
            for tile_row in range(y0 // tile_h, (y1 - 1) // tile_h + 1):
                for tile_col in range(x0 // tile_w, (x1 - 1) // tile_w + 1):
                    tile_index = tile_row * tiles_per_row + tile_col
                    handle.seek(int(page.dataoffsets[tile_index]))
                    raw = handle.read(int(page.databytecounts[tile_index]))
                    tile = np.frombuffer(raw, dtype=np.uint8).reshape(tile_h, tile_w, channels)
                    sx0, sy0 = tile_col * tile_w, tile_row * tile_h
                    ox0, oy0 = max(x0, sx0), max(y0, sy0)
                    ox1, oy1 = min(x1, sx0 + tile_w), min(y1, sy0 + tile_h)
                    out[oy0-y0:oy1-y0, ox0-x0:ox1-x0] = tile[oy0-sy0:oy1-sy0, ox0-sx0:ox1-sx0]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/paired_hd_xenium/lung_cancer_experiment2"))
    parser.add_argument("--alignment-qc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root
    qc = json.loads(args.alignment_qc.read_text())
    row0, col0, n = (int(qc[k]) for k in ("pilot_array_row_start", "pilot_array_col_start", "pilot_size_bins"))
    positions = pd.read_parquet(root / "visium_hd/binned_outputs/square_002um/spatial/tissue_positions.parquet")
    patch = positions.loc[
        positions.array_row.between(row0, row0 + n - 1)
        & positions.array_col.between(col0, col0 + n - 1)
    ]
    xy = patch[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()
    x0, y0 = np.floor(xy.min(axis=0) - 40).astype(int)
    x1, y1 = np.ceil(xy.max(axis=0) + 40).astype(int)
    box = (x0, y0, x1, y1)
    image_path = root / "visium_hd/input/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5K_Experiment2_tissue_image.btf"
    image = read_uncompressed_tiled_crop(image_path, box)
    geometry = geometry_from_official_files(root)

    cells = pd.read_parquet(root / "xenium/extracted/cells.parquet", columns=["cell_id", "x_centroid", "y_centroid"])
    centroids_px = transform_xenium_um_to_he_px(cells[["x_centroid", "y_centroid"]].to_numpy(), geometry)
    inside = (
        (centroids_px[:, 0] >= x0) & (centroids_px[:, 0] < x1)
        & (centroids_px[:, 1] >= y0) & (centroids_px[:, 1] < y1)
    )
    selected = cells.loc[inside, "cell_id"]
    boundaries = pd.read_parquet(
        root / "xenium/extracted/cell_boundaries.parquet",
        columns=["cell_id", "vertex_x", "vertex_y"],
    )
    boundaries = boundaries.loc[boundaries.cell_id.isin(selected)]
    boundary_xy = transform_xenium_um_to_he_px(boundaries[["vertex_x", "vertex_y"]].to_numpy(), geometry)
    boundaries = boundaries.assign(he_x=boundary_xy[:, 0], he_y=boundary_xy[:, 1])

    fig, ax = plt.subplots(figsize=(12, 12), dpi=140)
    ax.imshow(image, extent=(x0, x1, y1, y0))
    for _, group in boundaries.groupby("cell_id", sort=False):
        ax.plot(group.he_x, group.he_y, color="lime", lw=0.45, alpha=0.8)
    subset = patch.iloc[::20]
    ax.scatter(subset.pxl_col_in_fullres, subset.pxl_row_in_fullres, s=1.0, c="deepskyblue", alpha=0.7)
    ax.scatter(centroids_px[inside, 0], centroids_px[inside, 1], s=3.0, c="yellow", alpha=0.7)
    ax.set(xlim=(x0, x1), ylim=(y1, y0), xlabel="H&E full-resolution x pixel", ylabel="H&E full-resolution y pixel")
    ax.set_title("Pilot patch: Xenium cell boundaries (green), centroids (yellow), HD bin centers (blue, sampled)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"output": str(args.output), "image_box_he_pixels": [int(v) for v in box], "xenium_cells_in_box": int(inside.sum()), "boundary_vertices": len(boundaries)}))


if __name__ == "__main__":
    main()
