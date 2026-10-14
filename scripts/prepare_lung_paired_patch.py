#!/usr/bin/env python3
"""Extract one aligned real Visium HD pilot patch without touching method logic."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse
from scipy.io import mmwrite
import yaml

from plot_lung_paired_alignment import read_uncompressed_tiled_crop


def _copy_h5_features(source: h5py.Group, target: h5py.Group) -> None:
    for key in source.keys():
        source.copy(key, target)
    for key, value in source.attrs.items():
        target.attrs[key] = value


def extract_patch(
    *, root: Path, output: Path, core_row: int, core_col: int, core_size: int, halo_bins: int,
) -> dict[str, object]:
    source_hd = root / "visium_hd/binned_outputs/square_002um"
    scale = json.loads((source_hd / "spatial/scalefactors_json.json").read_text())
    bin_size_um = float(scale["bin_size_um"])
    repo = Path(__file__).resolve().parents[1]
    episode_config = yaml.safe_load((repo / "configs/episode_build.template.yaml").read_text())
    candidate_max_distance_um = float(
        episode_config["environment"]["max_center_distance_um"]
    )
    # EM evaluates bins up to MaxDis beyond the core. Their complete nucleus
    # candidate set requires one more MaxDis of image/context on every side.
    required_halo_bins = int(np.ceil(2.0 * candidate_max_distance_um / bin_size_um))
    if core_size <= 0 or halo_bins < required_halo_bins:
        raise ValueError(
            f"core_size must be positive and halo_bins must be >= {required_halo_bins} "
            "to cover scored-bin and candidate-cell MaxDis"
        )
    context_row = core_row - halo_bins
    context_col = core_col - halo_bins
    context_size = core_size + 2 * halo_bins
    if context_row < 0 or context_col < 0:
        raise ValueError("context extends beyond HD grid")
    pos = pd.read_parquet(source_hd / "spatial/tissue_positions.parquet")
    tissue = pos.in_tissue.to_numpy(dtype=bool)
    keep = (
        tissue
        & pos.array_row.between(context_row, context_row + context_size - 1).to_numpy()
        & pos.array_col.between(context_col, context_col + context_size - 1).to_numpy()
    )
    chosen = pos.loc[keep].copy()
    if chosen.empty or chosen.barcode.duplicated().any():
        raise ValueError("patch has no unique in-tissue bins")
    # The H5 barcode order is *not* the tissue_positions row-major order in
    # this 10x release. Join by the stable physical barcode, never by row.
    source_h5 = source_hd / "filtered_feature_bc_matrix.h5"
    with h5py.File(source_h5, "r") as source:
        m = source["matrix"]
        if int(tissue.sum()) != m["barcodes"].shape[0]:
            raise ValueError("HD positions and matrix have different in-tissue bin counts")
        source_barcodes = np.asarray(m["barcodes"][:])
        if len(pd.Index(source_barcodes).unique()) != len(source_barcodes):
            raise ValueError("HD matrix contains duplicate physical barcodes")
        matrix_col = pd.Index(source_barcodes).get_indexer(chosen.barcode.to_numpy(dtype="S"))
        if np.any(matrix_col < 0):
            raise ValueError("Some selected tissue barcodes are missing from the HD H5")
        order = np.argsort(matrix_col, kind="stable")
        chosen = chosen.iloc[order].reset_index(drop=True)
        matrix_col = matrix_col[order]
        sample = np.linspace(0, len(chosen) - 1, min(100, len(chosen)), dtype=np.int64)
        actual = np.asarray(m["barcodes"][matrix_col[sample]], dtype="S")
        expected = chosen.barcode.iloc[sample].to_numpy(dtype="S")
        if not np.array_equal(actual, expected):
            raise ValueError("HD matrix columns do not match tissue_positions barcode order")
        ptr = np.asarray(m["indptr"][:], dtype=np.int64)
        lengths = ptr[matrix_col + 1] - ptr[matrix_col]
        out_ptr = np.concatenate(([0], np.cumsum(lengths)))
        out_indices = np.empty(int(out_ptr[-1]), dtype=m["indices"].dtype)
        out_data = np.empty(int(out_ptr[-1]), dtype=m["data"].dtype)
        # Adjacent row-major columns are read as runs, avoiding ~20,000 tiny HDF5 reads.
        run_starts = np.r_[0, np.flatnonzero(np.diff(matrix_col) != 1) + 1]
        run_ends = np.r_[run_starts[1:], len(matrix_col)]
        for start, end in zip(run_starts, run_ends):
            in_start, in_end = int(ptr[matrix_col[start]]), int(ptr[matrix_col[end - 1] + 1])
            out_start, out_end = int(out_ptr[start]), int(out_ptr[end])
            out_indices[out_start:out_end] = m["indices"][in_start:in_end]
            out_data[out_start:out_end] = m["data"][in_start:in_end]
        genes = int(m["shape"][0])
        barcodes = source_barcodes[matrix_col]
        h5_path = output / "filtered_feature_bc_matrix.h5"
        output.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path, "w") as target:
            for key, value in source.attrs.items():
                target.attrs[key] = value
            dst = target.create_group("matrix")
            for key, value in m.attrs.items():
                dst.attrs[key] = value
            _copy_h5_features(m["features"], dst.create_group("features"))
            dst.create_dataset("barcodes", data=barcodes, compression="gzip")
            dst.create_dataset("data", data=out_data, compression="gzip")
            dst.create_dataset("indices", data=out_indices, compression="gzip")
            dst.create_dataset("indptr", data=out_ptr, compression="gzip")
            dst.create_dataset("shape", data=np.array([genes, len(chosen)], dtype=np.int64))
        ids = np.asarray(m["features/id"][:])
        names = np.asarray(m["features/name"][:])
        types = np.asarray(m["features/feature_type"][:])

    matrix = sparse.csc_matrix((out_data, out_indices, out_ptr), shape=(genes, len(chosen)))
    mtx_dir = output / "filtered_feature_bc_matrix"
    mtx_dir.mkdir(exist_ok=True)
    with gzip.open(mtx_dir / "matrix.mtx.gz", "wb") as handle:
        mmwrite(handle, matrix)
    with gzip.open(mtx_dir / "barcodes.tsv.gz", "wt") as handle:
        handle.writelines(b.decode() + "\n" for b in barcodes)
    with gzip.open(mtx_dir / "features.tsv.gz", "wt") as handle:
        handle.writelines(f"{a.decode()}\t{b.decode()}\t{c.decode()}\n" for a, b, c in zip(ids, names, types))

    xy = chosen[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()
    x0, y0 = np.floor(xy.min(axis=0) - 60).astype(int)
    x1, y1 = np.ceil(xy.max(axis=0) + 60).astype(int)
    source_image = root / "visium_hd/input/Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5K_Experiment2_tissue_image.btf"
    crop = read_uncompressed_tiled_crop(source_image, (int(x0), int(y0), int(x1), int(y1)))
    image_path = output / "he_patch_fullres.png"
    crop_image = Image.fromarray(crop)
    crop_image.save(image_path)
    spatial = output / "spatial"
    spatial.mkdir(exist_ok=True)
    crop_image.save(spatial / "tissue_hires_image.png")
    lowres_scale = 0.1
    lowres = crop_image.resize(
        (max(1, round(crop_image.width * lowres_scale)), max(1, round(crop_image.height * lowres_scale))),
        Image.Resampling.LANCZOS,
    )
    lowres.save(spatial / "tissue_lowres_image.png")
    local_pos = chosen.copy()
    local_pos["pxl_col_in_fullres"] -= x0
    local_pos["pxl_row_in_fullres"] -= y0
    local_pos.to_parquet(spatial / "tissue_positions.parquet", index=False)
    scale["tissue_hires_scalef"] = 1.0
    scale["tissue_lowres_scalef"] = lowres_scale
    (spatial / "scalefactors_json.json").write_text(json.dumps(scale, indent=2) + "\n")
    core = chosen.loc[
        chosen.array_row.between(core_row, core_row + core_size - 1)
        & chosen.array_col.between(core_col, core_col + core_size - 1)
    ]
    manifest = {
        "dataset": "10x Visium HD Human Lung Cancer post Xenium Prime 5K Experiment 2",
        "source_hd_h5": str(source_h5.resolve()),
        "source_hd_tissue_image": str(source_image.resolve()),
        "image_path": str(image_path.resolve()),
        "core_array_bounds": [core_row, core_row + core_size, core_col, core_col + core_size],
        "context_array_bounds": [context_row, context_row + context_size, context_col, context_col + context_size],
        "context_halo_bins": halo_bins,
        "candidate_max_distance_um": candidate_max_distance_um,
        "required_context_halo_bins": required_halo_bins,
        "image_crop_origin_fullres_he_px": [int(x0), int(y0)],
        "n_context_bins": int(len(chosen)),
        "n_core_bins": int(len(core)),
        "n_genes": genes,
        "n_nonzero_counts": int(matrix.nnz),
        "sum_counts": int(matrix.sum()),
        "matrix_barcode_order_verified": True,
        "evaluation": "core only; halo is inference context",
    }
    (output / "patch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/paired_hd_xenium/lung_cancer_experiment2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core-row", type=int, default=1840)
    parser.add_argument("--core-col", type=int, default=1240)
    parser.add_argument("--core-size", type=int, default=100)
    parser.add_argument("--halo-bins", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(extract_patch(root=args.data_root, output=args.output, core_row=args.core_row,
                                   core_col=args.core_col, core_size=args.core_size, halo_bins=args.halo_bins), indent=2))


if __name__ == "__main__":
    main()
