#!/usr/bin/env python3
"""Summarize paired GT-size stress tests without selecting method parameters."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hd_cell_rl.experiment_protocol import paired_cluster_mean_ci


NEIGHBORS = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)


def boundary(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {point for point in coords if any((point[0] + dr, point[1] + dc) not in coords
                                           for dr, dc in NEIGHBORS)}


def outer_ring(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(row + dr, col + dc) for row, col in coords for dr, dc in NEIGHBORS} - coords


def f1(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    return 2 * overlap / (len(left) + len(right))


def paired_source_identity(dataset: Path) -> tuple[pd.DataFrame, str]:
    """Identity and nucleus-center contract for the paired geometry arms."""
    columns = ["cell_id", "cell_type", "sc_cell_barcode", "source_donor", "x_um", "y_um"]
    frame = pd.read_csv(dataset / "ground_truth_cell_assignments.csv", usecols=columns,
                        dtype={name: str for name in columns[:4]})
    if frame.cell_id.duplicated().any():
        raise ValueError("source cell IDs are not unique")
    frame = frame.sort_values("cell_id").reset_index(drop=True)
    reference = dataset / "reference/reference_counts.npz"
    digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    return frame, digest


def check_paired_sources(sources: dict[float, tuple[pd.DataFrame, str]]) -> None:
    base, reference = sources[1.0]
    identity = ["cell_id", "cell_type", "sc_cell_barcode", "source_donor"]
    for scale, (frame, digest) in sources.items():
        if digest != reference or not frame[identity].equals(base[identity]):
            raise ValueError(f"scale {scale} changed a source cell or the reference")
        if not np.allclose(frame[["x_um", "y_um"]], base[["x_um", "y_um"]], rtol=0, atol=1e-10):
            raise ValueError(f"scale {scale} changed nuclear centers")


def source_summary(dataset: Path, context: Path, benchmark: Path, geometry_metadata: Path, scale: float) -> dict:
    bins = pd.read_parquet(context / "barcodes.parquet", columns=["matrix_col_index"])
    gt = pd.read_parquet(context / "inputs/fractional_gt_contributing_cells.parquet",
                         columns=["barcode_index", "cell_id", "cell_coverage_fraction"])
    covered_ids = set(gt.loc[gt.cell_id.gt(0) & gt.cell_coverage_fraction.gt(0), "barcode_index"].astype(int))
    covered = bins.matrix_col_index.isin(covered_ids).to_numpy()
    local_owner_counts = gt.loc[gt.barcode_index.isin(set(bins.matrix_col_index)) & gt.cell_id.gt(0)].groupby(
        "barcode_index").cell_id.nunique()
    modeled = sparse.load_npz(context / "patch_components/shared/modeled_cell.npz")
    background = sparse.load_npz(context / "patch_components/shared/unmodeled_biological.npz")
    if modeled.shape != background.shape or len(covered) != modeled.shape[0]:
        raise ValueError("source component axes differ")
    if modeled[~covered].nnz or background[covered].nnz:
        raise ValueError("source components violate GT coverage")
    umi = np.asarray(modeled.sum(axis=1)).ravel()
    genes = np.diff(modeled.indptr)
    bg = np.asarray(background.sum(axis=1)).ravel()
    meta = pd.read_csv(geometry_metadata)
    cores = pd.read_csv(context / "selection/source_eval/per_episode.csv", dtype={"cell_id":str})
    core_meta = meta.loc[meta.cell_id.astype(str).isin(set(cores.cell_id))]
    settings = yaml.safe_load((context / "episode_context/episode_build.capture_thinned.yaml").read_text())
    max_distance = float(settings["environment"]["max_center_distance_um"])
    nuclei = pd.read_parquet(context / "episode_context/nuclei_patch_context.parquet",
                             columns=["cell_id", "center_x_um", "center_y_um"])
    nuclei["cell_id"] = nuclei.cell_id.astype(str)
    nuclei = nuclei.set_index("cell_id")
    cell_bins = pd.read_csv(context / "ground_truth/fractional_cell_bins.csv.gz",
                            usecols=["cell_id", "array_row", "array_col", "x_um", "y_um"], dtype={"cell_id":str})
    cell_bins = cell_bins.loc[cell_bins.cell_id.isin(set(cores.cell_id))].join(
        nuclei, on="cell_id", validate="many_to_one")
    if cell_bins[["center_x_um", "center_y_um"]].isna().any().any():
        raise ValueError("core GT cell has no fixed nucleus center")
    distance = np.hypot(cell_bins.x_um-cell_bins.center_x_um,
                        cell_bins.y_um-cell_bins.center_y_um)
    in_em_radius = distance <= max_distance + 1e-9
    seeds = pd.read_csv(benchmark / "common_confident_nuclear_bins.csv.gz",
                        usecols=["barcode", "cell_id"], dtype={"barcode":str, "cell_id":str})
    seeds = seeds.loc[seeds.cell_id.isin(set(cores.cell_id))].copy()
    grid = seeds.barcode.str.extract(r"s_002um_(\d+)_(\d+)(?:-\d+)?$")
    if grid.isna().any().any():
        raise ValueError("confident nuclear seed has an invalid 2 µm barcode")
    seeds["array_row"] = grid[0].astype(int)
    seeds["array_col"] = grid[1].astype(int)
    by_seed = {cell: group[["array_row", "array_col"]].to_numpy(float)
               for cell, group in seeds.groupby("cell_id")}
    bin_to_seed_distances = []
    no_seed = []
    for cell, group in cell_bins.groupby("cell_id"):
        points = by_seed.get(cell)
        if points is None:
            no_seed.append(cell)
            continue
        bin_to_seed_distances.extend(cKDTree(points).query(
            group[["array_row", "array_col"]].to_numpy(float))[0])
    seed_distances = np.asarray(bin_to_seed_distances)
    return dict(scale=scale, n_context_bins=len(bins), n_covered_context_bins=int(covered.sum()),
                n_multiowner_context_bins=int((local_owner_counts > 1).sum()),
                multiowner_fraction_among_covered_bins=float((local_owner_counts > 1).mean()),
                n_core_cells=len(core_meta), mean_core_area_um2=float(core_meta.final_cell_area_um2.mean()),
                median_core_area_um2=float(core_meta.final_cell_area_um2.median()),
                core_target_reached_fraction=float(core_meta.target_reached.mean()),
                em_candidate_max_distance_um=max_distance,
                fraction_core_gt_bin_pairs_within_em_distance=float(in_em_radius.mean()),
                n_core_cells_without_confident_seed=len(no_seed),
                fraction_core_gt_bin_pairs_with_confident_seed=float(len(seed_distances) / len(cell_bins)),
                fraction_seeded_core_gt_bin_pairs_within_bin2cell_5=float(np.mean(seed_distances <= 5)),
                p95_gt_bin_distance_from_confident_seed=float(np.quantile(seed_distances, .95)),
                modeled_counts=int(modeled.sum()), background_counts=int(background.sum()),
                mean_modeled_umi_per_covered_bin=float(np.mean(umi[covered])),
                median_modeled_umi_per_covered_bin=float(np.median(umi[covered])),
                mean_detected_genes_per_covered_bin=float(np.mean(genes[covered])),
                covered_nonzero_fraction=float(np.mean(umi[covered] > 0)),
                mean_background_umi_per_uncovered_bin=float(np.mean(bg[~covered])),
                uncovered_nonzero_fraction=float(np.mean(bg[~covered] > 0)))


def method_rows(benchmark: Path, context: Path, dataset: Path, scale: float) -> list[dict]:
    gt = pd.read_csv(context / "ground_truth/fractional_cell_bins.csv.gz",
                     dtype={"cell_id":str, "barcode":str})
    truth = {cell: group for cell, group in gt.groupby("cell_id", sort=False)}
    support = pd.read_csv(dataset / "cell_reference_support.csv", dtype={"cell_id":str})
    supported = set(support.loc[support.reference_supported, "cell_id"])
    rows = []
    for condition in ("modeled_only", "source_separated"):
        paths = sorted((benchmark / condition / "evaluations").glob("*/summary.json"))
        if len(paths) != 5:
            raise ValueError(f"{scale}: {condition} has {len(paths)} rather than five methods")
        for path in paths:
            summary = json.loads(path.read_text())
            method = summary["method"].removeprefix("hd_cell_rl_")
            assignment_path = next(Path(value) for key, value in summary.items()
                                   if key.endswith("_assignments_path"))
            assignments = pd.read_csv(assignment_path, dtype={"cell_id":str, "barcode":str})
            if assignments.barcode.duplicated().any():
                raise ValueError(f"{method} exported duplicate barcode ownership")
            predicted = {cell: group for cell, group in assignments.groupby("cell_id", sort=False)}
            cells = pd.read_csv(path.parent / "per_episode.csv",
                                dtype={"cell_id":str, "matched_gt_cell_id":str,
                                       "matched_pred_cell_id":str})
            for cell in cells.itertuples(index=False):
                if cell.cell_id not in supported or pd.isna(cell.matched_gt_cell_id):
                    continue
                true = truth.get(str(cell.matched_gt_cell_id))
                if true is None:
                    raise ValueError("evaluated GT cell absent from fractional export")
                pred = predicted.get(str(cell.matched_pred_cell_id)) if pd.notna(cell.matched_pred_cell_id) else None
                tbar = set(true.barcode)
                pbar = set(pred.barcode) if pred is not None else set()
                tcoords = set(zip(true.array_row.astype(int), true.array_col.astype(int), strict=True))
                pcoords = set(zip(pred.array_row.astype(int), pred.array_col.astype(int), strict=True)) if pred is not None else set()
                outside = pcoords - tcoords
                ring = outer_ring(tcoords)
                pboundary = boundary(pcoords)
                tboundary = set(zip(true.loc[true.is_boundary.astype(bool), "array_row"].astype(int),
                                    true.loc[true.is_boundary.astype(bool), "array_col"].astype(int), strict=True))
                value = dict(scale=scale, condition=condition, method=method, cell_id=cell.cell_id,
                             patch_id=None, gt_n_bins=len(tbar), pred_n_bins=len(pbar),
                             evaluator_iou=float(cell.pred_iou), full_export_iou=len(tbar & pbar) / len(tbar | pbar),
                             area_bias_fraction=(len(pbar) - len(tbar)) / len(tbar),
                             boundary_f1=f1(pboundary, tboundary),
                             false_positive_bins=len(outside), false_positive_ring_bins=len(outside & ring),
                             false_positive_far_bins=len(outside - ring))
                if method == "bin2cell" and pred is not None and "segmentation_source" in pred:
                    coordinates = [tuple(x) for x in pred[["array_row", "array_col"]].to_numpy()]
                    false = pred.loc[[point in outside for point in coordinates]]
                    ring_false = pred.loc[[point in outside and point in ring for point in coordinates]]
                    for source in ("H&E", "H&E_expanded", "GEX"):
                        key = source.replace("&", "and").replace(" ", "_")
                        value["false_positive_" + key + "_bins"] = int(false.segmentation_source.eq(source).sum())
                        value["false_positive_ring_" + key + "_bins"] = int(ring_false.segmentation_source.eq(source).sum())
                rows.append(value)
    mapping = pd.read_csv(context / "selection/source_eval/per_episode.csv", dtype={"cell_id":str})
    result = pd.DataFrame(rows).merge(mapping, on="cell_id", how="left", validate="many_to_one", suffixes=("", "_frozen"))
    if result.patch_id_frozen.isna().any():
        raise ValueError("an evaluated cell was not in the frozen patch selection")
    result["patch_id"] = result.patch_id_frozen
    return result.drop(columns="patch_id_frozen").to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", nargs=5, metavar=("SCALE", "DATASET", "CONTEXT", "BENCHMARK", "GEOMETRY_META"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    depth, cells, captures, sources = [], [], {}, {}
    for scale_text, dataset_text, context_text, benchmark_text, metadata_text in args.arm:
        scale = float(scale_text)
        dataset, context, benchmark, metadata = [Path(item).resolve(strict=True) for item in
                                                 (dataset_text, context_text, benchmark_text, metadata_text)]
        captured = pd.read_csv(dataset / "subdiagnostic/per_cell_capture.csv.gz").sort_values("cell_id").reset_index(drop=True)
        captures[scale] = captured
        sources[scale] = paired_source_identity(dataset)
        depth.append(source_summary(dataset, context, benchmark, metadata, scale))
        cells.extend(method_rows(benchmark, context, dataset, scale))
    if 1.0 not in captures or any(not frame.equals(captures[1.0]) for frame in captures.values()):
        raise ValueError("paired per-cell raw/captured count summaries differ across geometry arms")
    check_paired_sources(sources)
    depth_frame = pd.DataFrame(depth).sort_values("scale")
    cell_frame = pd.DataFrame(cells)
    expected = set(cell_frame.loc[cell_frame.scale.eq(1.0), "cell_id"])
    for scale, group in cell_frame.groupby("scale"):
        if set(group.cell_id) != expected:
            raise ValueError(f"scale {scale} changed the evaluated core-cell set")
    metrics = ("evaluator_iou", "full_export_iou", "area_bias_fraction", "boundary_f1",
               "false_positive_bins", "false_positive_ring_bins", "false_positive_far_bins")
    aggregate = cell_frame.groupby(["scale", "condition", "method"], as_index=False).agg(
        n_cells=("cell_id", "size"), n_patches=("patch_id", "nunique"),
        **{"mean_" + name: (name, "mean") for name in metrics})
    depth_frame.to_csv(out / "data_quality_by_scale.csv", index=False)
    cell_frame.to_csv(out / "per_cell_geometry_metrics.csv", index=False)
    aggregate.to_csv(out / "method_geometry_comparison.csv", index=False)
    distance = aggregate.loc[aggregate.method.eq("distance_only"),
                             ["scale", "condition", "mean_evaluator_iou"]].rename(
        columns={"mean_evaluator_iou": "distance_only_iou"})
    relative = aggregate.merge(distance, on=["scale", "condition"], validate="many_to_one")
    relative["iou_delta_vs_distance_only"] = relative.mean_evaluator_iou - relative.distance_only_iou
    relative.to_csv(out / "method_vs_distance_by_scale.csv", index=False)
    paired_methods = []
    distance_cells = cell_frame.loc[cell_frame.method.eq("distance_only"),
                                    ["scale", "condition", "cell_id", "patch_id", "evaluator_iou"]].rename(
        columns={"evaluator_iou": "distance_iou"})
    compared = cell_frame.merge(distance_cells, on=["scale", "condition", "cell_id", "patch_id"],
                                validate="many_to_one")
    for (scale, condition, method), group in compared.groupby(["scale", "condition", "method"]):
        delta = (group.evaluator_iou - group.distance_iou).to_numpy(float)
        low, high = paired_cluster_mean_ci(delta, group.patch_id.to_numpy(),
                                           rng=np.random.default_rng(20260922))
        paired_methods.append(dict(scale=scale, condition=condition, method=method,
                                   mean_iou_delta_vs_distance=float(delta.mean()),
                                   patch_bootstrap_low=low, patch_bootstrap_high=high,
                                   n_cells=len(group), n_patches=group.patch_id.nunique()))
    pd.DataFrame(paired_methods).to_csv(out / "paired_method_vs_distance.csv", index=False)
    paired = []
    keys = ["condition", "method", "cell_id", "patch_id"]
    original = cell_frame.loc[cell_frame.scale.eq(1.0), keys + list(metrics)]
    if original.duplicated(keys).any():
        raise ValueError("baseline per-cell method rows are not unique")
    for scale, current in cell_frame.loc[~cell_frame.scale.eq(1.0)].groupby("scale"):
        joined = current.merge(original, on=keys, how="outer", validate="one_to_one",
                               suffixes=("_variant", "_baseline"), indicator=True)
        if not joined._merge.eq("both").all():
            raise ValueError(f"scale {scale} is not paired to the same baseline cells")
        for (condition, method), group in joined.groupby(["condition", "method"]):
            for metric in ("evaluator_iou", "boundary_f1", "area_bias_fraction", "false_positive_ring_bins"):
                delta = (group[metric + "_variant"] - group[metric + "_baseline"]).to_numpy(float)
                low, high = paired_cluster_mean_ci(delta, group.patch_id.to_numpy(),
                                                   rng=np.random.default_rng(20260922))
                paired.append(dict(scale=scale, condition=condition, method=method, metric=metric,
                                   mean_delta=float(delta.mean()), patch_bootstrap_low=low,
                                   patch_bootstrap_high=high, n_cells=len(group),
                                   n_patches=group.patch_id.nunique()))
    pd.DataFrame(paired).to_csv(out / "paired_delta_vs_original.csv", index=False)
    native_columns = [name for name in cell_frame if name.startswith("false_positive_") and name.endswith("_bins")
                      and name not in metrics]
    if native_columns:
        cell_frame.loc[cell_frame.method.eq("bin2cell")].groupby(["scale", "condition"], as_index=False)[
            native_columns].sum().to_csv(out / "bin2cell_false_positive_by_source.csv", index=False)
    print(aggregate.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
