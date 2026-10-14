#!/usr/bin/env python3
"""Adapt the published Xenium-to-2um notebook to common, traceable patch inputs.

Only the data interface is new: qv filtering, whole-cell transcript selection,
ceil(x/2) binning, and count aggregation follow generate_synthetic_data_Xenium.ipynb.
Cell-boundary polygons are written as held-out evaluation truth, never as input
to the assignment methods. Nucleus polygons provide the same seeds to all methods.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from scipy import sparse
from scipy.io import mmwrite
from shapely import Polygon, contains_xy
from shapely.ops import unary_union


DEFAULT_CENTERS = ((6300, 1000), (5400, 1400), (3400, 1200), (9600, 2400))


def _barcode(row: np.ndarray, col: np.ndarray) -> np.ndarray:
    # STCS's unchanged Visium-HD reader parses exactly s_002um_ROW_COL-1.
    return np.char.add(np.char.add("s_002um_", np.char.zfill(row.astype(str), 5)),
                       np.char.add("_", np.char.add(np.char.zfill(col.astype(str), 5), "-1")))


def _notebook_barcode(row: np.ndarray, col: np.ndarray) -> np.ndarray:
    return np.char.add(np.char.add("2um_", np.char.zfill(row.astype(str), 5)),
                       np.char.add("_", np.char.zfill(col.astype(str), 5)))


def _polygons(path: Path, cell_ids: set[str]) -> dict[str, Polygon]:
    frame = pd.read_parquet(path)
    frame = frame.loc[frame.cell_id.isin(cell_ids)]
    result: dict[str, Polygon] = {}
    for cell_id, group in frame.groupby("cell_id", sort=False):
        # Prime 5K can contain several nuclear rings per physical cell.
        # Single-ring v1 cells retain precisely their previous geometry.
        rings = [Polygon(part[["vertex_x", "vertex_y"]].to_numpy())
                 for _, part in group.groupby("label_id", sort=False)]
        rings = [ring for ring in rings if ring.is_valid and not ring.is_empty]
        if rings:
            result[str(cell_id)] = rings[0] if len(rings) == 1 else unary_union(rings)
    return result


def _raster_owners(polygons: dict[str, Polygon], x: np.ndarray, y: np.ndarray,
                   cell_labels: dict[str, int]) -> tuple[np.ndarray, int]:
    owner = np.zeros(len(x), dtype=np.int64)
    collision_mask = np.zeros(len(x), dtype=bool)
    for cell_id, polygon in polygons.items():
        minx, miny, maxx, maxy = polygon.bounds
        nearby = np.flatnonzero((x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy))
        if not len(nearby):
            continue
        contained = nearby[contains_xy(polygon, x[nearby], y[nearby])]
        collision_mask[contained[owner[contained] != 0]] = True
        owner[contained[owner[contained] == 0]] = cell_labels[cell_id]
    owner[collision_mask] = 0
    return owner, int(collision_mask.sum())


def _write_matrix(patch: Path, matrix: sparse.csc_matrix, genes: list[str], barcodes: list[str],
                  library_id: str = "Xenium_notebook_crc") -> None:
    with h5py.File(patch / "filtered_feature_bc_matrix.h5", "w") as handle:
        handle.attrs["library_ids"] = np.asarray([library_id.encode()])
        handle.attrs["chemistry_description"] = "Notebook-derived 2um synthetic HD"
        handle.attrs["software_version"] = "HD-Cell-RL data adapter"
        handle.attrs["filetype"] = "matrix"
        handle.attrs["version"] = 2
        handle.attrs["original_gem_groups"] = np.asarray([1])
        group = handle.create_group("matrix")
        group.create_dataset("data", data=matrix.data.astype(np.int32), compression="gzip")
        group.create_dataset("indices", data=matrix.indices.astype(np.int32), compression="gzip")
        group.create_dataset("indptr", data=matrix.indptr.astype(np.int64), compression="gzip")
        group.create_dataset("shape", data=np.asarray(matrix.shape, dtype=np.int64))
        group.create_dataset("barcodes", data=np.asarray(barcodes, dtype="S"))
        features = group.create_group("features")
        for key, value in (("id", genes), ("name", genes),
                           ("feature_type", ["Gene Expression"] * len(genes)),
                           ("genome", ["GRCh38"] * len(genes))):
            features.create_dataset(key, data=np.asarray(value, dtype="S"))
        features.create_dataset("_all_tag_keys", data=np.asarray([b"genome"]))
    mtx = patch / "filtered_feature_bc_matrix"
    mtx.mkdir()
    with gzip.open(mtx / "matrix.mtx.gz", "wb") as handle:
        mmwrite(handle, matrix, field="integer")
    with gzip.open(mtx / "barcodes.tsv.gz", "wt") as handle:
        handle.write("\n".join(barcodes) + "\n")
    with gzip.open(mtx / "features.tsv.gz", "wt") as handle:
        handle.writelines(f"{gene}\t{gene}\tGene Expression\n" for gene in genes)


def build(source: Path, output: Path, centers: tuple[tuple[int, int], ...],
          core_size: int, halo: int, *,
          dataset_label: str = "Xenium V1 Human Colorectal Cancer Addon FFPE, notebook-style 2um bins",
          library_id: str = "Xenium_notebook_crc") -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite benchmark data: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cells = pd.read_parquet(source / "cells.parquet", columns=["cell_id", "x_centroid", "y_centroid"])
    # Numeric labels are required by the unchanged Bin2Cell external-seed adapter.
    all_labels = {cell_id: i + 1 for i, cell_id in enumerate(sorted(cells.cell_id.astype(str)))}
    specs = []
    for center_x, center_y in centers:
        core_col = round(center_x / 2) - core_size // 2
        core_row = round(center_y / 2) - core_size // 2
        specs.append(dict(name=f"core_x{center_x}_y{center_y}", core_col=core_col,
                          core_row=core_row, context_col=core_col - halo,
                          context_row=core_row - halo, context_size=core_size + 2 * halo))
    captured: list[list[pd.DataFrame]] = [[] for _ in specs]
    columns = ["cell_id", "feature_name", "x_location", "y_location", "qv"]
    n_qv20 = 0
    for batch in pq.ParquetFile(source / "transcripts.parquet").iter_batches(columns=columns, batch_size=500_000):
        frame = batch.to_pandas()
        frame = frame.loc[frame.qv >= 20.0].copy()
        n_qv20 += len(frame)
        row = np.ceil(frame.y_location.to_numpy() / 2.0).astype(np.int32)
        col = np.ceil(frame.x_location.to_numpy() / 2.0).astype(np.int32)
        frame["array_row"] = row
        frame["array_col"] = col
        for i, spec in enumerate(specs):
            selected = frame.loc[
                (row >= spec["context_row"]) & (row < spec["context_row"] + spec["context_size"])
                & (col >= spec["context_col"]) & (col < spec["context_col"] + spec["context_size"]),
                ["cell_id", "feature_name", "array_row", "array_col"],
            ]
            if len(selected):
                captured[i].append(selected)
    manifests = []
    shared_genes: set[str] | None = None
    for spec, chunks in zip(specs, captured):
        patch = output / spec["name"]
        patch.mkdir()
        transcript = pd.concat(chunks, ignore_index=True)
        transcript["barcode"] = _barcode(transcript.array_row.to_numpy(), transcript.array_col.to_numpy())
        positions = transcript[["barcode", "array_row", "array_col"]].drop_duplicates().sort_values(
            ["array_row", "array_col"], kind="stable").reset_index(drop=True)
        positions["in_tissue"] = 1
        x_origin = 2 * spec["context_col"] - 1
        y_origin = 2 * spec["context_row"] - 1
        positions["pxl_col_in_fullres"] = 2 * positions.array_col - 1 - x_origin
        positions["pxl_row_in_fullres"] = 2 * positions.array_row - 1 - y_origin
        # Bin2Cell's 10x reader renames position columns by *order*, so keep
        # the official tissue_positions schema, not only the same column names.
        positions = positions[["barcode", "in_tissue", "array_row", "array_col",
                               "pxl_col_in_fullres", "pxl_row_in_fullres"]]
        pd.DataFrame({"barcode": positions.barcode.astype(str),
                      "notebook_barcode": _notebook_barcode(positions.array_row.to_numpy(),
                                                            positions.array_col.to_numpy())}).to_parquet(
            patch / "barcode_map.parquet", index=False)
        barcodes = positions.barcode.astype(str).tolist()
        genes = sorted(transcript.feature_name.astype(str).unique())
        shared_genes = set(genes) if shared_genes is None else shared_genes.intersection(genes)
        gene_index = pd.Index(genes).get_indexer(transcript.feature_name.astype(str))
        bin_index = pd.Index(barcodes).get_indexer(transcript.barcode.astype(str))
        matrix = sparse.coo_matrix((np.ones(len(transcript), dtype=np.int32), (gene_index, bin_index)),
                                   shape=(len(genes), len(barcodes))).tocsc()
        matrix.sum_duplicates()
        _write_matrix(patch, matrix, genes, barcodes, library_id=library_id)
        spatial = patch / "spatial"
        spatial.mkdir()
        positions.to_parquet(spatial / "tissue_positions.parquet", index=False)
        scale = {"bin_size_um": 2.0, "microns_per_pixel": 1.0, "tissue_hires_scalef": 1.0,
                 "tissue_lowres_scalef": 1.0, "spot_diameter_fullres": 2.0}
        (spatial / "scalefactors_json.json").write_text(json.dumps(scale, indent=2) + "\n")
        # The notebook has no H&E. This neutral raster only satisfies the existing
        # image loader; external Xenium nuclei, never image segmentation, supply seeds.
        image = Image.new("RGB", (2 * spec["context_size"], 2 * spec["context_size"]), "white")
        image.save(patch / "he_patch_fullres.png")
        image.save(spatial / "tissue_hires_image.png")
        image.save(spatial / "tissue_lowres_image.png")
        cells_here = cells.loc[cells.x_centroid.between(x_origin - 40, x_origin + 2 * spec["context_size"] + 40)
                               & cells.y_centroid.between(y_origin - 40, y_origin + 2 * spec["context_size"] + 40)]
        cell_ids = set(cells_here.cell_id.astype(str))
        nuclei_poly = _polygons(source / "nucleus_boundaries.parquet", cell_ids)
        cell_poly = _polygons(source / "cell_boundaries.parquet", cell_ids)
        x = (2 * positions.array_col - 1).to_numpy(dtype=float)
        y = (2 * positions.array_row - 1).to_numpy(dtype=float)
        nuclear_owner, nuclear_collisions = _raster_owners(nuclei_poly, x, y, all_labels)
        seed_rows = positions.loc[nuclear_owner > 0].copy()
        seed_rows["cell_id"] = nuclear_owner[nuclear_owner > 0].astype(str)
        seed_rows["is_nuclear"] = True
        seed_rows["has_expression"] = True
        # Matrix positions and nuclei must share one patch-local physical frame.
        seed_rows["x_um"] = seed_rows.pxl_col_in_fullres
        seed_rows["y_um"] = seed_rows.pxl_row_in_fullres
        nuclei_dir = patch / "nuclei"
        nuclei_dir.mkdir()
        seed_rows[["barcode", "cell_id", "is_nuclear", "has_expression", "array_row", "array_col",
                   "pxl_col_in_fullres", "pxl_row_in_fullres", "x_um", "y_um"]].to_csv(
            nuclei_dir / "nuclear_bins.csv.gz", index=False, compression="gzip")
        nuclei = seed_rows.groupby("cell_id", sort=True).agg(center_x_um=("x_um", "mean"),
                     center_y_um=("y_um", "mean"), n_nuclear_bins=("barcode", "size"),
                     n_expressed_nuclear_bins=("has_expression", "sum")).reset_index()
        nuclei["radius_um"] = np.sqrt(nuclei.n_nuclear_bins * 4.0 / np.pi)
        nuclei.to_parquet(nuclei_dir / "nuclei.parquet", index=False)
        # Geometric mask truth is an independent output, never read by a method.
        core = positions.array_row.between(spec["core_row"], spec["core_row"] + core_size - 1) & positions.array_col.between(
            spec["core_col"], spec["core_col"] + core_size - 1)
        core_pos = positions.loc[core].copy()
        core_x = (2 * core_pos.array_col - 1).to_numpy(dtype=float)
        core_y = (2 * core_pos.array_row - 1).to_numpy(dtype=float)
        mask_owner, mask_collisions = _raster_owners(cell_poly, core_x, core_y, all_labels)
        core_pos["gt_cell_id"] = mask_owner.astype(str)
        core_pos.to_parquet(patch / "core_cell_mask_truth.parquet", index=False)
        # Keep multi-owner transcript truth; do not collapse to a fabricated hard owner.
        gt = transcript.loc[transcript.cell_id.ne("UNASSIGNED")].groupby(
            ["barcode", "cell_id"], sort=False).size().rename("n_transcripts").reset_index()
        gt["cell_id"] = gt.cell_id.map(all_labels).astype("int64").astype(str)
        gt.to_parquet(patch / "fractional_bin_truth.parquet", index=False)
        pd.DataFrame({"cell_id": [all_labels[c] for c in sorted(cell_ids)],
                      "xenium_cell_id": sorted(cell_ids)}).to_csv(patch / "cell_id_map.csv", index=False)
        manifest = dict(dataset=dataset_label,
                        patch_id=spec["name"], core_array_bounds=[spec["core_row"], spec["core_row"] + core_size,
                        spec["core_col"], spec["core_col"] + core_size],
                        context_array_bounds=[spec["context_row"], spec["context_row"] + spec["context_size"],
                        spec["context_col"], spec["context_col"] + spec["context_size"]],
                        context_halo_bins=halo, candidate_max_distance_um=20.0,
                        image_path=str((patch / "he_patch_fullres.png").resolve()),
                        coordinate_origin_um=[x_origin, y_origin],
                        image_role="neutral loader placeholder; not used for nuclear inference",
                        transcript_rule="whole-cell Xenium transcripts, qv>=20, ceil(x/2), ceil(y/2); only barcode names adapted to 10x s_002um_ROW_COL-1",
                        n_context_bins=len(positions), n_core_bins=len(core_pos), n_genes=len(genes),
                        n_transcripts=len(transcript), n_nuclei_with_seed=len(nuclei),
                        n_nuclear_bins=len(seed_rows), n_nucleus_overlap_bins=nuclear_collisions,
                        n_cell_mask_overlap_bins=mask_collisions, n_gt_multi_owner_bins=int(
                        (gt.groupby("barcode").size() > 1).sum()),
                        evaluation="core only; cell boundaries held out from inference")
        (patch / "patch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        manifests.append(manifest)
    summary = {"source": str(source.resolve()), "notebook": "generate_synthetic_data_Xenium.ipynb",
               "n_source_qv20_transcripts": n_qv20, "patches": manifests,
               "source_xenium_microns_per_pixel": float(json.loads(
                   (source / "experiment.xenium").read_text())["pixel_size"])
                   if (source / "experiment.xenium").is_file() else None,
               "synthetic_raster_microns_per_pixel": 1.0}
    (output / "reference_gene_allowlist.txt").write_text("\n".join(sorted(shared_genes or set())) + "\n")
    (output / "dataset_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core-size", type=int, default=60)
    parser.add_argument("--halo-bins", type=int, default=20)
    parser.add_argument("--center", type=int, nargs=2, action="append", metavar=("X_UM", "Y_UM"))
    parser.add_argument("--dataset-label", default="Xenium V1 Human Colorectal Cancer Addon FFPE, notebook-style 2um bins")
    parser.add_argument("--library-id", default="Xenium_notebook_crc")
    args = parser.parse_args()
    if args.halo_bins < 20:
        raise ValueError("A 20-bin context halo is required for the 20um RL candidate MaxDis")
    centers = tuple(map(tuple, args.center)) if args.center else DEFAULT_CENTERS
    result = build(args.source, args.output, centers, args.core_size, args.halo_bins,
                   dataset_label=args.dataset_label, library_id=args.library_id)
    print(json.dumps({"n_patches": len(result["patches"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
