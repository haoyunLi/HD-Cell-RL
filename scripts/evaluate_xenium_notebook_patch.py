#!/usr/bin/env python3
"""Evaluate common-method assignments against held-out Xenium cell boundaries.

The primary mask IoU uses every 2um grid point in the core, including bins
without a transcript. A second IoU conditions on observed expression bins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

from build_xenium_notebook_patch_benchmark import _polygons, _raster_owners, _barcode


def evaluate(patch: Path, assignments: Path, method: str, source: Path) -> tuple[dict, pd.DataFrame]:
    manifest = json.loads((patch / "patch_manifest.json").read_text())
    r0, r1, c0, c1 = manifest["core_array_bounds"]
    rows, cols = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    barcodes = _barcode(rows, cols)
    mapping = pd.read_csv(patch / "cell_id_map.csv", dtype={"cell_id": str, "xenium_cell_id": str})
    cell_labels = dict(zip(mapping.xenium_cell_id, mapping.cell_id.astype(np.int64)))
    truth_polygons = _polygons(source / "cell_boundaries.parquet", set(cell_labels))
    gt, collisions = _raster_owners(truth_polygons, 2 * cols - 1.0, 2 * rows - 1.0, cell_labels)
    predicted = pd.read_csv(assignments, dtype={"barcode": str, "cell_id": str})
    if predicted.barcode.duplicated().any():
        raise ValueError(f"Duplicate physical barcode ownership in {method}: {assignments}")
    predicted.cell_id = predicted.cell_id.fillna("0")
    lookup = predicted.set_index("barcode").cell_id
    owners = pd.Series(barcodes).map(lookup).fillna("0").astype(np.int64).to_numpy()
    observed = pd.read_parquet(patch / "spatial/tissue_positions.parquet", columns=["barcode"])
    is_observed = np.isin(barcodes, observed.barcode.astype(str).to_numpy())
    nuclei = pd.read_parquet(patch / "nuclei/nuclei.parquet")
    origin_x, origin_y = manifest["coordinate_origin_um"]
    nucleus_x = nuclei.center_x_um + origin_x
    nucleus_y = nuclei.center_y_um + origin_y
    core_cells = nuclei.loc[nucleus_x.between(2 * c0 - 1, 2 * c1 - 1, inclusive="left")
                             & nucleus_y.between(2 * r0 - 1, 2 * r1 - 1, inclusive="left"), "cell_id"].astype(str)
    if not len(core_cells):
        raise ValueError("No core nuclei to score")
    cell_rows = []
    for cell_id in core_cells:
        label = int(cell_id)
        true_mask = gt.reshape((r1 - r0, c1 - c0)) == label
        pred_mask = owners.reshape((r1 - r0, c1 - c0)) == label
        if not true_mask.any():
            continue
        intersection = int(np.logical_and(true_mask, pred_mask).sum())
        union = int(np.logical_or(true_mask, pred_mask).sum())
        observed_mask = is_observed.reshape(true_mask.shape)
        intersection_observed = int((true_mask & pred_mask & observed_mask).sum())
        union_observed = int(((true_mask | pred_mask) & observed_mask).sum())
        ring = binary_dilation(true_mask, iterations=1) & ~true_mask
        cell_rows.append(dict(patch=manifest["patch_id"], method=method, cell_id=cell_id,
                              gt_area_bins=int(true_mask.sum()), pred_area_bins=int(pred_mask.sum()),
                              intersection_bins=intersection, union_bins=union,
                              iou=intersection / union if union else np.nan,
                              observed_iou=intersection_observed / union_observed if union_observed else np.nan,
                              area_ratio=float(pred_mask.sum() / true_mask.sum()),
                              outer_ring_false_positive_bins=int((pred_mask & ring).sum())))
    cells = pd.DataFrame(cell_rows)
    gt_fractional = pd.read_parquet(patch / "fractional_bin_truth.parquet")
    gt_fractional = gt_fractional.loc[gt_fractional.barcode.isin(barcodes)]
    owner_by_barcode = pd.Series(owners.astype(str), index=barcodes)
    gt_fractional["pred_cell_id"] = gt_fractional.barcode.map(owner_by_barcode)
    weighted_correct = int(gt_fractional.loc[
        gt_fractional.cell_id.astype(str).eq(gt_fractional.pred_cell_id.astype(str)), "n_transcripts"].sum())
    weighted_total = int(gt_fractional.n_transcripts.sum())
    summary = dict(patch=manifest["patch_id"], method=method, n_core_cells=len(cells),
                   n_ambiguous_gt_mask_bins=collisions,
                   n_core_grid_bins=len(barcodes), n_observed_core_bins=int(is_observed.sum()),
                   n_assigned_core_bins=int((owners != 0).sum()),
                   mean_cell_iou=float(cells.iou.mean()), median_cell_iou=float(cells.iou.median()),
                   mean_observed_cell_iou=float(cells.observed_iou.mean()),
                   median_area_ratio=float(cells.area_ratio.median()),
                   outer_ring_false_positive_bins=int(cells.outer_ring_false_positive_bins.sum()),
                   transcript_mass_correct=weighted_correct,
                   transcript_mass_total=weighted_total,
                   transcript_mass_accuracy=weighted_correct / weighted_total if weighted_total else np.nan)
    return summary, cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary, cells = evaluate(args.patch_dir, args.assignments, args.method, args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    cells.to_csv(args.output_dir / "per_cell.csv", index=False)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
