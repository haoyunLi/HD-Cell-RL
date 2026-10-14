#!/usr/bin/env python3
"""Independent source-to-matrix and coordinate audit for Xenium notebook patches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from scipy.io import mmread
from shapely import contains_xy

from build_xenium_notebook_patch_benchmark import _barcode, _notebook_barcode, _polygons


def audit(data: Path) -> dict:
    manifest = json.loads((data / "dataset_manifest.json").read_text())
    source = Path(manifest["source"])
    source_cells = pd.read_parquet(source / "cells.parquet", columns=["cell_id", "x_centroid", "y_centroid"])
    reference = data / "reference/reference_counts.npz"
    with np.load(reference) as ref:
        ref_genes = set(ref["genes"].astype(str))
        n_types = len(ref["cell_types"])
    raw_selection: list[list[pd.DataFrame]] = [[] for _ in manifest["patches"]]
    for batch in pq.ParquetFile(source / "transcripts.parquet").iter_batches(
        columns=["cell_id", "feature_name", "x_location", "y_location", "qv"], batch_size=500_000
    ):
        frame = batch.to_pandas()
        frame = frame.loc[frame.qv >= 20.0]
        row = np.ceil(frame.y_location.to_numpy() / 2.0).astype(np.int64)
        col = np.ceil(frame.x_location.to_numpy() / 2.0).astype(np.int64)
        frame = frame.assign(row=row, col=col)
        for i, entry in enumerate(manifest["patches"]):
            r0, r1, c0, c1 = entry["context_array_bounds"]
            selected = frame.loc[(row >= r0) & (row < r1) & (col >= c0) & (col < c1),
                                 ["cell_id", "feature_name", "row", "col"]]
            if len(selected):
                raw_selection[i].append(selected)
    checks = []
    for entry, chunks in zip(manifest["patches"], raw_selection):
        patch = data / entry["patch_id"]
        positions = pd.read_parquet(patch / "spatial/tissue_positions.parquet")
        assert list(positions.columns) == ["barcode", "in_tissue", "array_row", "array_col",
                                           "pxl_col_in_fullres", "pxl_row_in_fullres"]
        labels = pd.read_csv(patch / "cell_id_map.csv", dtype={"cell_id": str, "xenium_cell_id": str})
        id_map = dict(zip(labels.xenium_cell_id, labels.cell_id))
        barcode_map = pd.read_parquet(patch / "barcode_map.parquet")
        assert positions.barcode.is_unique and barcode_map.barcode.is_unique
        assert np.array_equal(positions.barcode.to_numpy(), _barcode(positions.array_row.to_numpy(),
                                                                  positions.array_col.to_numpy()))
        assert np.array_equal(barcode_map.notebook_barcode.to_numpy(), _notebook_barcode(
            positions.array_row.to_numpy(), positions.array_col.to_numpy()))
        ox, oy = entry["coordinate_origin_um"]
        assert np.allclose(positions.pxl_col_in_fullres + ox, 2 * positions.array_col - 1)
        assert np.allclose(positions.pxl_row_in_fullres + oy, 2 * positions.array_row - 1)
        assert entry["context_halo_bins"] * 2 >= 2 * entry["candidate_max_distance_um"]
        with h5py.File(patch / "filtered_feature_bc_matrix.h5") as handle:
            matrix_group = handle["matrix"]
            barcodes = matrix_group["barcodes"][:].astype(str)
            genes = matrix_group["features/name"][:].astype(str)
            h5_matrix = sparse.csc_matrix((matrix_group["data"][:], matrix_group["indices"][:],
                                           matrix_group["indptr"][:]),
                                          shape=tuple(matrix_group["shape"][:]))
        mtx_matrix = mmread(patch / "filtered_feature_bc_matrix/matrix.mtx.gz").tocsc()
        assert (h5_matrix != mtx_matrix).nnz == 0
        assert np.array_equal(barcodes, positions.barcode.astype(str).to_numpy())
        assert ref_genes.issubset(set(genes))
        raw = pd.concat(chunks, ignore_index=True)
        assert len(raw) == entry["n_transcripts"] == int(h5_matrix.sum())
        raw["barcode"] = _barcode(raw.row.to_numpy(), raw.col.to_numpy())
        observed = raw.groupby(["feature_name", "barcode"], sort=False).size().rename("raw_n").reset_index()
        observed["gene_index"] = pd.Index(genes).get_indexer(observed.feature_name.astype(str))
        observed["bin_index"] = pd.Index(barcodes).get_indexer(observed.barcode.astype(str))
        assert np.all(observed.gene_index >= 0) and np.all(observed.bin_index >= 0)
        assert np.array_equal(h5_matrix[observed.gene_index.to_numpy(), observed.bin_index.to_numpy()].A1,
                              observed.raw_n.to_numpy())
        assert len(observed) == h5_matrix.nnz
        gt = raw.loc[raw.cell_id.ne("UNASSIGNED")].groupby(["barcode", "cell_id"], sort=False).size()
        saved_gt = pd.read_parquet(patch / "fractional_bin_truth.parquet")
        recovered = saved_gt.assign(xenium_cell_id=saved_gt.cell_id.astype(str).map(
            {v: k for k, v in id_map.items()})).set_index(["barcode", "xenium_cell_id"]).n_transcripts
        pd.testing.assert_series_equal(gt.sort_index(), recovered.sort_index(), check_names=False)
        seeds = pd.read_csv(patch / "nuclei/nuclear_bins.csv.gz", dtype={"barcode": str, "cell_id": str})
        nuclei = pd.read_parquet(patch / "nuclei/nuclei.parquet")
        assert seeds.barcode.is_unique and set(seeds.barcode).issubset(set(barcodes))
        joined = seeds.merge(nuclei, on="cell_id", validate="many_to_one")
        max_seed_distance = float(np.hypot(joined.x_um - joined.center_x_um,
                                           joined.y_um - joined.center_y_um).max())
        assert max_seed_distance <= entry["candidate_max_distance_um"] + 1e-5
        original_nucleus_ids = set(seeds.cell_id.map({v: k for k, v in id_map.items()}))
        polygons = _polygons(source / "nucleus_boundaries.parquet", original_nucleus_ids)
        n_bad_seed = 0
        for cell_id, group in seeds.groupby("cell_id"):
            xenium_id = {v: k for k, v in id_map.items()}[cell_id]
            n_bad_seed += int((~contains_xy(polygons[xenium_id], group.x_um.to_numpy() + ox,
                                            group.y_um.to_numpy() + oy)).sum())
        assert n_bad_seed == 0
        context_r0, context_r1, context_c0, context_c1 = entry["context_array_bounds"]
        source_context_cells = source_cells.loc[
            source_cells.x_centroid.between(2 * context_c0 - 1, 2 * context_c1 - 1)
            & source_cells.y_centroid.between(2 * context_r0 - 1, 2 * context_r1 - 1)
        ]
        seeded_xenium_ids = {k for k, v in id_map.items() if v in set(seeds.cell_id)}
        n_unseeded_context = int((~source_context_cells.cell_id.isin(seeded_xenium_ids)).sum())
        core_r0, core_r1, core_c0, core_c1 = entry["core_array_bounds"]
        source_core_cells = source_cells.loc[
            source_cells.x_centroid.between(2 * core_c0 - 1, 2 * core_c1 - 1)
            & source_cells.y_centroid.between(2 * core_r0 - 1, 2 * core_r1 - 1)
        ]
        n_unseeded_core = int((~source_core_cells.cell_id.isin(seeded_xenium_ids)).sum())
        checks.append(dict(patch=entry["patch_id"], pass_all=True, n_transcripts=len(raw),
                           n_genes=len(genes), n_barcodes=len(barcodes), n_nuclear_bins=len(seeds),
                           n_multi_owner_bins=int((saved_gt.groupby("barcode").size() > 1).sum()),
                           max_seed_distance_um=max_seed_distance,
                           n_unseeded_centroid_cells_context=n_unseeded_context,
                           n_unseeded_centroid_cells_core=n_unseeded_core,
                           n_reference_genes=len(ref_genes), n_reference_types=n_types))
    return {"source": str(source), "reference": str(reference), "checks": checks,
            "pass_all": True,
            "pass_all_scope": "Observed transcript aggregation, barcode/coordinate axes, matrix identity, and retained seed validity; not complete coverage of every source Xenium nucleus or cell polygon."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
