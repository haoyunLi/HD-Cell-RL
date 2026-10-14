#!/usr/bin/env python3
"""Run the upstream STCS StarDist H&E seed step once for every comparison method."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from csbdeep.utils import normalize
from PIL import Image
from stardist.models import StarDist2D
import tifffile


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--patch-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prob-thresh", type=float, default=0.1)
    p.add_argument("--model-basedir", type=Path,
                   default=Path.home() / ".keras/models/StarDist2D/2D_versatile_he",
                   help="Local cache of the official pretrained StarDist model, avoiding a network re-download")
    a = p.parse_args()
    patch = a.patch_dir
    a.output_dir.mkdir(parents=True, exist_ok=True)
    image = np.asarray(Image.open(patch / "he_patch_fullres.png").convert("RGB"))
    # Exactly the model, normalization, threshold, and tile count in the
    # checked-in upstream STCS run_stardist_pipeline/config.py; the only
    # adaptation is that input is already a common cropped H&E raster.
    model_folder = a.model_basedir / "2D_versatile_he"
    if model_folder.exists() and (model_folder / "weights_best.h5").exists():
        model = StarDist2D(config=None, name="2D_versatile_he", basedir=a.model_basedir)
    else:
        model = StarDist2D.from_pretrained("2D_versatile_he")
    labels, _ = model.predict_instances(normalize(image), prob_thresh=a.prob_thresh, n_tiles=(4, 4, 1))
    tifffile.imwrite(a.output_dir / "stardist_labels.tif", labels.astype(np.int32))

    positions = pd.read_parquet(patch / "spatial/tissue_positions.parquet")
    x = np.floor(positions.pxl_col_in_fullres.to_numpy()).astype(np.int64)
    y = np.floor(positions.pxl_row_in_fullres.to_numpy()).astype(np.int64)
    inside = (x >= 0) & (x < labels.shape[1]) & (y >= 0) & (y < labels.shape[0])
    bin_label = np.zeros(len(positions), dtype=np.int64)
    bin_label[inside] = labels[y[inside], x[inside]]
    seeds = positions.loc[bin_label > 0].copy()
    seeds["cell_id"] = bin_label[bin_label > 0]
    seeds["is_nuclear"] = True
    if seeds.barcode.duplicated().any():
        raise ValueError("One physical bin received multiple nuclear owners")

    with h5py.File(patch / "filtered_feature_bc_matrix.h5", "r") as h5:
        m = h5["matrix"]
        matrix_barcodes = pd.Index(np.asarray(m["barcodes"][:]).astype(str))
        matrix_col = matrix_barcodes.get_indexer(seeds.barcode.astype(str))
        if np.any(matrix_col < 0):
            raise ValueError("Nuclear seed barcode not found in patch matrix")
        ptr = np.asarray(m["indptr"][:], dtype=np.int64)
        seeds["has_expression"] = ptr[matrix_col + 1] > ptr[matrix_col]
    scales = json.loads((patch / "spatial/scalefactors_json.json").read_text())
    mpp = float(scales["microns_per_pixel"])
    seeds["x_um"] = seeds.pxl_col_in_fullres * mpp
    seeds["y_um"] = seeds.pxl_row_in_fullres * mpp
    csv = a.output_dir / "nuclear_bins.csv.gz"
    seeds[["barcode", "cell_id", "is_nuclear", "has_expression", "array_row", "array_col",
           "pxl_col_in_fullres", "pxl_row_in_fullres", "x_um", "y_um"]].to_csv(csv, index=False, compression="gzip")

    nuclei = seeds.groupby("cell_id", sort=True).agg(
        center_x_um=("x_um", "mean"), center_y_um=("y_um", "mean"),
        n_nuclear_bins=("barcode", "size"), n_expressed_nuclear_bins=("has_expression", "sum"),
    ).reset_index()
    nuclei["radius_um"] = np.sqrt(nuclei.n_nuclear_bins * 4.0 / math.pi)
    nuclei["cell_id"] = nuclei.cell_id.astype(str)
    nuclei.to_parquet(a.output_dir / "nuclei.parquet", index=False)
    summary = {
        "upstream_model": "STCS run_stardist_pipeline: StarDist2D 2D_versatile_he",
        "prob_thresh": a.prob_thresh, "n_tiles": [4, 4, 1],
        "image_native_mpp": mpp,
        "n_detected_image_labels": int(np.max(labels)),
        "n_nuclei_with_hd_bins": int(len(nuclei)),
        "n_nuclei_with_expressed_hd_seeds": int((nuclei.n_expressed_nuclear_bins > 0).sum()),
        "n_nuclear_bins": int(len(seeds)),
        "n_expressed_nuclear_bins": int(seeds.has_expression.sum()),
        "n_zero_expression_nuclear_bins": int((~seeds.has_expression).sum()),
        "note": "All methods receive this same seed table; native filtering differences are audited, not silently filled.",
    }
    (a.output_dir / "seed_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
