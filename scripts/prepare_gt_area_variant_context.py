#!/usr/bin/env python3
"""Reuse frozen 16-patch axes with a new fractional GT and pseudo expression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocessing.export_nucleus_based_gt_bins import export_gt_bins


def link(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.symlink_to(source.resolve(strict=True), target_is_directory=source.is_dir())


def build_context(base: Path, dataset: Path, fractional_gt: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for name in ("selection", "episode_context", "barcodes.parquet", "genes.csv"):
        link(base / name, output / name)
    (output / "matrices").mkdir()
    link(base / "matrices/real.npz", output / "matrices/real.npz")
    bins = pd.read_parquet(base / "barcodes.parquet")
    indices = bins.matrix_col_index.to_numpy(np.int64)
    if bins.barcode.duplicated().any() or len(np.unique(indices)) != len(indices):
        raise ValueError("frozen physical barcode axes are not unique")
    components = {}
    for name in ("modeled_cell", "unmodeled_biological"):
        x = sparse.load_npz(dataset / "components" / (name + ".npz"))
        if indices.max() >= x.shape[0]:
            raise ValueError("variant expression lacks a selected barcode")
        components[name] = x[indices].tocsr()
    components["observed"] = (components["modeled_cell"] + components["unmodeled_biological"]).tocsr()
    components["technical_background"] = sparse.csr_matrix(components["observed"].shape, dtype=np.int64)
    shared = output / "patch_components/shared"
    shared.mkdir(parents=True)
    for name, matrix in components.items():
        sparse.save_npz(shared / (name + ".npz"), matrix)
    patches = pd.read_csv(base / "selection/test_patches_index.csv")
    for patch in patches.patch_id:
        directory = output / "patch_components/realizations" / ("leave_out_" + patch) / "uniform_budget"
        directory.mkdir(parents=True)
        for name in components:
            link(shared / (name + ".npz"), directory / (name + ".npz"))

    gt = pd.read_parquet(fractional_gt)
    if gt.duplicated(["barcode_index", "cell_id"]).any():
        raise ValueError("duplicate physical barcode/cell GT pair")
    selection = set(indices.tolist())
    contributing = set(gt.loc[gt.barcode_index.isin(selection) & gt.cell_id.gt(0), "cell_id"].astype(int))
    nuclei = pd.read_parquet(base / "episode_context/nuclei_patch_context.parquet", columns=["cell_id"])
    contributing |= set(pd.to_numeric(nuclei.cell_id).astype(int))
    selected_gt = gt.loc[gt.cell_id.isin(contributing)].copy()
    core_ids = set(pd.read_csv(base / "selection/source_eval/per_episode.csv").cell_id.astype(int))
    core_gt = selected_gt.loc[selected_gt.cell_id.isin(core_ids)]
    if not core_gt.barcode_index.isin(selection).all():
        raise ValueError("a scored core cell extends beyond the frozen expression context")
    inputs = output / "inputs"
    inputs.mkdir()
    selected_gt.to_parquet(inputs / "fractional_gt_contributing_cells.parquet", index=False)
    master = pd.read_csv(dataset / "ground_truth_cell_assignments.csv")
    master.loc[master.cell_id.isin(contributing)].to_csv(
        inputs / "ground_truth_cells.csv.gz", index=False)

    ground_truth = output / "ground_truth"
    cell_bins = ground_truth / "fractional_cell_bins.csv.gz"
    nuclear_bins = ground_truth / "nuclear_bins.csv.gz"
    summary = export_gt_bins(
        bin_assignments_path=inputs / "fractional_gt_contributing_cells.parquet",
        barcodes_path=dataset / "no_background/filtered_feature_bc_matrix/barcodes.tsv.gz",
        cell_assignments_path=dataset / "ground_truth_cell_assignments.csv",
        cell_output_path=cell_bins,
        nuclear_output_path=nuclear_bins,
        summary_path=ground_truth / "export_summary.json",
        microns_per_pixel=0.2737012522439323,
    )
    dominant = selected_gt.loc[selected_gt.is_dominant_owner & selected_gt.cell_id.gt(0),
                               ["barcode_index", "cell_id"]]
    if dominant.barcode_index.duplicated().any():
        raise ValueError("dominant GT owner is not unique by barcode")
    barcodes = pd.read_csv(dataset / "no_background/filtered_feature_bc_matrix/barcodes.tsv.gz",
                           sep="\t", header=None, usecols=[0], dtype=str)[0].to_numpy()
    dominant_map = dict(zip(barcodes[dominant.barcode_index.to_numpy(np.int64)],
                            dominant.cell_id.astype(str), strict=True))
    frame = pd.read_csv(cell_bins, dtype={"cell_id":str})
    frame = frame.loc[frame.cell_id.eq(frame.barcode.map(dominant_map))]
    if frame.barcode.duplicated().any():
        raise ValueError("dominant cell-bin export is not unique")
    dominant_path = ground_truth / "dominant_cell_bins.csv.gz"
    frame.to_csv(dominant_path, index=False)
    frame.loc[frame.is_nuclear.astype(bool)].to_csv(
        ground_truth / "dominant_nuclear_bins.csv.gz", index=False)
    result = dict(base_context=str(base), dataset=str(dataset), fractional_gt=str(fractional_gt),
                  n_patches=len(patches), n_barcodes=len(bins), n_contributing_cells=len(contributing),
                  n_fractional_rows=len(selected_gt), n_dominant_bins=len(frame),
                  selected_barcode_axis_unchanged=True, ground_truth_export=summary)
    (output / "variant_context_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-context", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--fractional-gt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_context(args.base_context.resolve(strict=True), args.dataset_dir.resolve(strict=True),
                           args.fractional_gt.resolve(strict=True), args.output_dir.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
