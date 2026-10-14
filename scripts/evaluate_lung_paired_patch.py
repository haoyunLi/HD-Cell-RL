#!/usr/bin/env python3
"""Compare HD bin-to-cell methods with registered Xenium proxy cells.

The primary cell matching follows the STCS paper's three-stage idea:
StarDist nuclear pixels spatially join Xenium polygons, predicted/known cells
are compared as polygons, and maximal-intersection matches are reciprocal.
The native Xenium cell boundary is used as the reference polygon. The STCS
repository does not release a separate matched-Xenium evaluator, so this is
an explicitly documented reimplementation of that Methods protocol, not a
claim to reproduce its whole-slide numeric results exactly.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import Delaunay
from scipy.stats import pearsonr, spearmanr
from shapely import points as shapely_points
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
import tifffile

from qc_lung_paired_alignment import (
    fit_hd_grid_to_he_px, geometry_from_official_files, project_points,
    transform_xenium_um_to_he_px,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def alpha_shape_um(xy: np.ndarray, alpha: float = 0.05) -> Polygon:
    """Alpha-shape cell footprint using physical microns; 1/alpha=20 um."""
    coordinates = np.unique(np.asarray(xy, dtype=np.float64), axis=0)
    if len(coordinates) < 4:
        return MultiPoint(coordinates).convex_hull.buffer(1.0)
    try:
        triangulation = Delaunay(coordinates)
    except Exception:
        return MultiPoint(coordinates).convex_hull.buffer(1.0)
    triangles = coordinates[triangulation.simplices]
    ab = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1)
    bc = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1)
    ca = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1)
    side1 = triangles[:, 1] - triangles[:, 0]
    side2 = triangles[:, 2] - triangles[:, 0]
    double_area = np.abs(side1[:, 0] * side2[:, 1] - side1[:, 1] * side2[:, 0])
    radius = ab * bc * ca / np.maximum(2.0 * double_area, 1e-12)
    retained = triangles[radius <= 1.0 / alpha]
    if len(retained) == 0:
        return MultiPoint(coordinates).convex_hull.buffer(1.0)
    geometry = unary_union([Polygon(triangle) for triangle in retained]).buffer(1.0)
    return geometry if geometry.is_valid else geometry.buffer(0)


def _load_xenium_polygons(root: Path, patch: Path) -> tuple[list[str], list[object], pd.DataFrame]:
    manifest = json.loads((patch / "patch_manifest.json").read_text())
    origin = np.asarray(manifest["image_crop_origin_fullres_he_px"], dtype=np.float64)
    geometry = geometry_from_official_files(root)
    mpp = float(geometry["hd_microns_per_pixel"])
    positions = pd.read_parquet(patch / "spatial/tissue_positions.parquet")
    local_xy = positions[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()
    low, high = local_xy.min(axis=0) - 100, local_xy.max(axis=0) + 100
    cells = pd.read_parquet(root / "xenium/extracted/cells.parquet", columns=["cell_id", "x_centroid", "y_centroid"])
    he = transform_xenium_um_to_he_px(cells[["x_centroid", "y_centroid"]].to_numpy(), geometry)
    local = he - origin
    selected = np.all((local >= low) & (local <= high), axis=1)
    cells = cells.loc[selected].copy()
    cells["he_x_um"] = he[selected, 0] * mpp
    cells["he_y_um"] = he[selected, 1] * mpp
    boundaries = pd.read_parquet(root / "xenium/extracted/cell_boundaries.parquet",
                                 columns=["cell_id", "vertex_x", "vertex_y", "label_id"])
    boundaries = boundaries.loc[boundaries.cell_id.isin(cells.cell_id)]
    xy = transform_xenium_um_to_he_px(boundaries[["vertex_x", "vertex_y"]].to_numpy(), geometry) * mpp
    boundaries = boundaries.assign(x_um=xy[:, 0], y_um=xy[:, 1])
    geom_by_id: dict[str, object] = {}
    for cell_id, group in boundaries.groupby("cell_id", sort=False):
        parts = []
        for _, one in group.groupby("label_id", sort=False):
            polygon = Polygon(one[["x_um", "y_um"]].to_numpy())
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if not polygon.is_empty and polygon.area > 0:
                parts.append(polygon)
        if parts:
            geom_by_id[str(cell_id)] = unary_union(parts)
    cells = cells.loc[cells.cell_id.isin(geom_by_id)].reset_index(drop=True)
    ids = cells.cell_id.astype(str).tolist()
    polygons = [geom_by_id[cell_id] for cell_id in ids]
    return ids, polygons, cells


def _nuclear_majority_votes(
    label_path: Path, xenium_ids: list[str], xenium_polygons: list[object],
    origin_px: tuple[int, int], hd_mpp: float,
) -> tuple[dict[str, str], dict[str, int]]:
    label_image = tifffile.imread(label_path)
    yy, xx = np.nonzero(label_image > 0)
    labels = label_image[yy, xx].astype(np.int64)
    physical = np.column_stack((xx + origin_px[0], yy + origin_px[1])) * hd_mpp
    tree = STRtree(xenium_polygons)
    links = tree.query(shapely_points(physical[:, 0], physical[:, 1]), predicate="within")
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for point_idx, xe_idx in links.T:
        votes[str(labels[int(point_idx)])][xenium_ids[int(xe_idx)]] += 1
    winner = {cell: sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
              for cell, counts in votes.items()}
    diagnostics = {"n_stardist_nuclear_pixels": int(len(labels)),
                   "n_nuclear_pixels_in_xenium_polygon": int(len(links[0])),
                   "n_nuclei_with_xenium_vote": int(len(winner))}
    return winner, diagnostics


def _load_h5(path: Path) -> tuple[sparse.csc_matrix, list[str], list[str], np.ndarray]:
    with h5py.File(path, "r") as file:
        m = file["matrix"]
        matrix = sparse.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]),
                                   shape=tuple(m["shape"][:]))
        barcodes = [x.decode() for x in m["barcodes"][:]]
        genes = [x.decode() for x in m["features/name"][:]]
        types = np.asarray([x.decode() for x in m["features/feature_type"][:]])
    return matrix, barcodes, genes, types


def _gene_context(root: Path, patch: Path, xe_core_ids: set[str]) -> dict[str, object]:
    # Gene comparison is evaluation-only; Xenium never enters EM inference.
    hd, hd_barcodes, hd_genes, _ = _load_h5(patch / "filtered_feature_bc_matrix.h5")
    xe, xe_barcodes, xe_genes, xe_types = _load_h5(root / "xenium/extracted/cell_feature_matrix.h5")
    hd_gene_index = {}
    for i, gene in enumerate(hd_genes):
        hd_gene_index.setdefault(gene, i)
    xe_gene_index = {gene: i for i, (gene, kind) in enumerate(zip(xe_genes, xe_types))
                     if kind == "Gene Expression"}
    core_indices = [i for i, cell in enumerate(xe_barcodes) if cell in xe_core_ids]
    # The paper selects HVGs only after restricting to genes shared by HD and
    # Xenium. Its Seurat-v3 HVG implementation is not distributed with the
    # released evaluator; variance on log-normalized counts is an explicit
    # pilot-only substitute, held fixed across all methods.
    shared_gene_names = sorted(set(xe_gene_index).intersection(hd_gene_index))
    gene_rows = np.array([xe_gene_index[name] for name in shared_gene_names], dtype=np.int64)
    core = np.asarray(xe[gene_rows, :][:, core_indices].T.toarray(), dtype=np.float64)
    total = core.sum(axis=1, keepdims=True)
    lognorm = np.log1p(core / np.maximum(total, 1.0) * 1e4)
    # Fixed from Xenium core cells before looking at method assignments.
    variable = np.var(lognorm, axis=0)
    selected = np.argsort(variable)[-min(2000, len(variable)):]
    shared = [shared_gene_names[i] for i in selected]
    return {"hd": hd, "hd_barcode_to_idx": {b: i for i, b in enumerate(hd_barcodes)},
            "xe": xe, "xe_cell_to_idx": {c: i for i, c in enumerate(xe_barcodes)},
            "hd_gene_rows": np.asarray([hd_gene_index[g] for g in shared], dtype=np.int64),
            "xe_gene_rows": np.asarray([xe_gene_index[g] for g in shared], dtype=np.int64),
            "n_shared_hvg": len(shared),
            "hvg_rule": "shared HD/Xenium genes, then top 2000 variance of log1p(Xenium core CPM); exploratory substitute for paper Seurat-v3 HVG"}


def _gene_similarity(context: dict[str, object], predicted_barcodes: list[str], xe_id: str) -> dict[str, float]:
    hd_indices = [context["hd_barcode_to_idx"][b] for b in predicted_barcodes
                  if b in context["hd_barcode_to_idx"]]
    if not hd_indices or xe_id not in context["xe_cell_to_idx"]:
        return {"gene_spearman": np.nan, "gene_pearson": np.nan, "gene_rmse": np.nan, "n_genes_both_nonzero": 0}
    a = np.asarray(context["hd"][context["hd_gene_rows"], :][:, hd_indices].sum(axis=1)).ravel()
    b = np.asarray(context["xe"][context["xe_gene_rows"], context["xe_cell_to_idx"][xe_id]].toarray()).ravel()
    both = (a > 0) & (b > 0)
    if int(both.sum()) < 2:
        return {"gene_spearman": np.nan, "gene_pearson": np.nan, "gene_rmse": np.nan, "n_genes_both_nonzero": int(both.sum())}
    a = np.log1p(a[both] / max(a.sum(), 1.0) * 1e4)
    b = np.log1p(b[both] / max(b.sum(), 1.0) * 1e4)
    rmse = float(np.sqrt(np.mean((a-b)**2)))
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return {"gene_spearman": np.nan, "gene_pearson": np.nan,
                "gene_rmse": rmse, "n_genes_both_nonzero": int(both.sum())}
    return {"gene_spearman": float(spearmanr(a, b).statistic),
            "gene_pearson": float(pearsonr(a, b).statistic),
            "gene_rmse": rmse,
            "n_genes_both_nonzero": int(both.sum())}


def select_reciprocal_matches(
    candidate_rows: list[tuple], nuclear_votes: dict[str, str],
    xe_core_ids: set[str],
) -> list[tuple]:
    """Keep mutual maximum-intersection pairs agreeing with nucleus vote."""
    best_by_xe: dict[str, tuple] = {}
    best_by_pred: dict[str, tuple] = {}
    for row in candidate_rows:
        xe_id, cell_id = row[1], row[0]
        if xe_id not in best_by_xe or (row[2], cell_id) > (best_by_xe[xe_id][2], best_by_xe[xe_id][0]):
            best_by_xe[xe_id] = row
        if cell_id not in best_by_pred or (row[2], xe_id) > (best_by_pred[cell_id][2], best_by_pred[cell_id][1]):
            best_by_pred[cell_id] = row
    matched = [row for row in candidate_rows
               if best_by_xe[row[1]] is row and best_by_pred[row[0]] is row
               and nuclear_votes.get(row[0]) == row[1]
               and row[1] in xe_core_ids]
    if len({row[0] for row in matched}) != len(matched) or len({row[1] for row in matched}) != len(matched):
        raise ValueError("Reciprocal matching did not produce unique cell pairs")
    return matched


def evaluate_method(
    *, name: str, assignment_path: Path, patch: Path, core_cells: set[str],
    xe_core_ids: set[str], xe_ids: list[str], xe_polygons: list[object],
    nuclear_votes: dict[str, str], gene_context: dict[str, object],
    hd_mpp: float, seed_owner: dict[str, str] | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    assignments = pd.read_csv(assignment_path, dtype={"barcode": str, "cell_id": str})
    if assignments.barcode.duplicated().any():
        raise ValueError(f"{name}: duplicate ownership of a physical HD barcode")
    seed_owner = seed_owner or {}
    assigned_seed = assignments.loc[assignments.barcode.isin(seed_owner)]
    seed_conflicts = assigned_seed.loc[
        assigned_seed.cell_id != assigned_seed.barcode.map(seed_owner)
    ]
    if len(seed_conflicts):
        raise ValueError(f"{name}: {len(seed_conflicts)} nuclear seeds changed owner")
    positions = pd.read_parquet(patch / "spatial/tissue_positions.parquet")
    position_lookup = positions.set_index("barcode")
    for field in ("array_row", "array_col"):
        if field in assignments:
            expected = position_lookup[field].reindex(assignments.barcode).to_numpy()
            observed = assignments[field].to_numpy()
            if np.any(pd.notna(expected) & (expected != observed)):
                raise ValueError(f"{name}: {field} disagrees with 10x barcode positions")
    # All coordinate fields come from one canonical 10x barcode lookup. Method
    # exports may carry their own row/col fields, but those must not alter the
    # evaluation join or create ambiguous Pandas suffixes.
    assignments = assignments.drop(columns=[
        "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"
    ], errors="ignore")
    n_input_assignments = len(assignments)
    assignments = assignments.merge(positions[["barcode", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"]],
                                    on="barcode", how="inner", validate="one_to_one")
    if len(assignments) != n_input_assignments:
        raise ValueError(f"{name}: some assignment barcodes are absent from canonical patch positions")
    core_r0, core_r1, core_c0, core_c1 = json.loads(
        (patch / "patch_manifest.json").read_text()
    )["core_array_bounds"]
    n_assigned_core_bins = int((
        assignments.array_row.between(core_r0, core_r1 - 1)
        & assignments.array_col.between(core_c0, core_c1 - 1)
    ).sum())
    by_cell = assignments.groupby("cell_id", sort=False)
    xe_polygon_by_id = dict(zip(xe_ids, xe_polygons))
    candidate_rows = []
    predicted_polygons: dict[str, object] = {}
    predicted_barcodes: dict[str, list[str]] = {}
    n_core_with_assignment = 0
    n_core_with_vote = 0
    for cell_id in sorted(core_cells):
        if cell_id not in by_cell.indices:
            continue
        n_core_with_assignment += 1
        group = by_cell.get_group(cell_id)
        xy = group[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=np.float64)
        # Patch coordinates must be restored to the same full H&E origin as XE.
        origin = json.loads((patch / "patch_manifest.json").read_text())["image_crop_origin_fullres_he_px"]
        xy += np.asarray(origin)
        polygon = alpha_shape_um(xy * hd_mpp)
        predicted_polygons[cell_id] = polygon
        predicted_barcodes[cell_id] = group.barcode.tolist()
        if nuclear_votes.get(cell_id) in xe_polygon_by_id:
            n_core_with_vote += 1
    # The paper computes maximum-intersection choices in BOTH directions, then
    # keeps reciprocal pairs. This cannot be replaced with a last-row-wins
    # dictionary or a reference-only maximum. The nucleus vote is a separate
    # first-stage candidate constraint on the reciprocal polygon match.
    xe_tree = STRtree(xe_polygons)
    for cell_id, polygon in predicted_polygons.items():
        for xe_index in xe_tree.query(polygon, predicate="intersects"):
            xe_id = xe_ids[int(xe_index)]
            reference = xe_polygon_by_id[xe_id]
            intersection = float(polygon.intersection(reference).area)
            if intersection <= 0:
                continue
            union = float(polygon.union(reference).area)
            candidate_rows.append((cell_id, xe_id, intersection,
                                   intersection / union, polygon.area,
                                   reference.area, predicted_barcodes[cell_id]))
    matched = select_reciprocal_matches(candidate_rows, nuclear_votes, xe_core_ids)
    per_cell = []
    for cell_id, xe_id, area, iou, pred_area, xe_area, barcodes in matched:
        per_cell.append({"method": name, "pred_cell_id": cell_id, "xenium_cell_id": xe_id,
                         "intersection_um2": area, "iou": iou,
                         "pred_area_um2": pred_area, "xenium_area_um2": xe_area,
                         "n_predicted_bins": len(barcodes),
                         **_gene_similarity(gene_context, barcodes, xe_id)})
    frame = pd.DataFrame(per_cell)
    def mean(col: str) -> float:
        return float(frame[col].mean()) if not frame.empty else float("nan")
    summary = {"method": name, "n_core_seed_cells": len(core_cells),
               "n_nuclear_seed_bins_retained": len(assigned_seed),
               "n_nuclear_seed_bins_missing": len(seed_owner) - len(assigned_seed),
               "n_core_cells_with_assignment": n_core_with_assignment,
               "n_core_cells_with_nuclear_xenium_vote": n_core_with_vote,
               "n_candidate_polygon_intersections": len(candidate_rows),
               "n_matched_reciprocal": len(frame),
               "n_xenium_core_cells": len(xe_core_ids),
               "matched_pred_fraction": len(frame) / max(1, len(core_cells)),
               "matched_xenium_fraction": len(frame) / max(1, len(xe_core_ids)),
               "mean_iou": mean("iou"), "median_iou": float(frame.iou.median()) if not frame.empty else float("nan"),
               "mean_gene_spearman": mean("gene_spearman"),
               "mean_gene_pearson": mean("gene_pearson"), "mean_gene_rmse": mean("gene_rmse"),
               "n_valid_gene_correlations": int(frame.gene_spearman.notna().sum()) if not frame.empty else 0,
               "n_shared_xenium_hvg": gene_context["n_shared_hvg"],
               "n_assigned_context_bins": len(assignments),
               "n_assigned_core_bins": n_assigned_core_bins}
    return summary, frame


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data/paired_hd_xenium/lung_cancer_experiment2"))
    p.add_argument("--patch-dir", type=Path, required=True)
    p.add_argument("--nuclei-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--method", nargs=2, action="append", metavar=("NAME", "ASSIGNMENTS_CSV"), required=True)
    a = p.parse_args()
    manifest = json.loads((a.patch_dir / "patch_manifest.json").read_text())
    core_r0, core_r1, core_c0, core_c1 = manifest["core_array_bounds"]
    origin = tuple(manifest["image_crop_origin_fullres_he_px"])
    geometry = geometry_from_official_files(a.data_root)
    hd_mpp = float(geometry["hd_microns_per_pixel"])
    seeds = pd.read_csv(a.nuclei_dir / "nuclear_bins.csv.gz", dtype={"cell_id": str})
    seed_owner = dict(zip(seeds.barcode.astype(str), seeds.cell_id.astype(str)))
    centers = seeds.groupby("cell_id")[["array_row", "array_col"]].mean()
    core_cells = set(centers.loc[
        centers.array_row.between(core_r0, core_r1-1)
        & centers.array_col.between(core_c0, core_c1-1)
    ].index.astype(str))
    xe_ids, xe_polygons, xe_cells = _load_xenium_polygons(a.data_root, a.patch_dir)
    global_px = xe_cells[["he_x_um", "he_y_um"]].to_numpy() / hd_mpp
    full_positions = pd.read_parquet(
        a.data_root / "visium_hd/binned_outputs/square_002um/spatial/tissue_positions.parquet"
    )
    grid_to_he = fit_hd_grid_to_he_px(full_positions)
    grid = project_points(global_px, np.linalg.inv(grid_to_he))
    xe_core = set(xe_cells.loc[(grid[:, 1] >= core_r0) & (grid[:, 1] < core_r1)
                               & (grid[:, 0] >= core_c0) & (grid[:, 0] < core_c1), "cell_id"])
    votes, vote_diag = _nuclear_majority_votes(a.nuclei_dir / "stardist_labels.tif", xe_ids, xe_polygons, origin, hd_mpp)
    gene_context = _gene_context(a.data_root, a.patch_dir, xe_core)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rows, details = [], []
    for name, path in a.method:
        row, per_cell = evaluate_method(name=name, assignment_path=Path(path), patch=a.patch_dir,
                                        core_cells=core_cells, xe_core_ids=xe_core,
                                        xe_ids=xe_ids, xe_polygons=xe_polygons,
                                        nuclear_votes=votes, gene_context=gene_context,
                                        hd_mpp=hd_mpp, seed_owner=seed_owner)
        rows.append(row)
        details.append(per_cell)
        print(json.dumps(row), flush=True)
    # Conditional mean IoU can improve simply by dropping difficult matches.
    # Report both zero-filled core-cell IoU and the strictly paired intersection
    # of cell IDs that all methods matched to the same Xenium cell.
    matched_pairs = [set(zip(frame.pred_cell_id.astype(str), frame.xenium_cell_id.astype(str)))
                     for frame in details]
    common_pairs = set.intersection(*matched_pairs) if matched_pairs else set()
    valid_gene_pairs = [set(zip(
        frame.loc[frame.gene_spearman.notna(), "pred_cell_id"].astype(str),
        frame.loc[frame.gene_spearman.notna(), "xenium_cell_id"].astype(str),
    )) for frame in details]
    common_gene_pairs = set.intersection(*valid_gene_pairs) if valid_gene_pairs else set()
    for row, frame in zip(rows, details):
        paired = frame.loc[[tuple(pair) in common_pairs for pair in
                            zip(frame.pred_cell_id.astype(str), frame.xenium_cell_id.astype(str))]]
        row["mean_iou_all_core_seeds_zero_unmatched"] = (
            float(frame.iou.sum()) / max(1, len(core_cells)) if not frame.empty else 0.0
        )
        row["n_common_matched_pairs"] = len(common_pairs)
        row["mean_iou_common_pairs"] = float(paired.iou.mean()) if not paired.empty else float("nan")
        row["mean_gene_spearman_common_pairs"] = (
            float(paired.gene_spearman.mean()) if not paired.empty else float("nan")
        )
        row["n_valid_gene_correlations_common_pairs"] = (
            int(paired.gene_spearman.notna().sum()) if not paired.empty else 0
        )
        same_gene_cells = frame.loc[[tuple(pair) in common_gene_pairs for pair in
                                     zip(frame.pred_cell_id.astype(str), frame.xenium_cell_id.astype(str))]]
        row["n_common_gene_evaluable_pairs"] = len(common_gene_pairs)
        row["mean_gene_spearman_common_evaluable"] = (
            float(same_gene_cells.gene_spearman.mean()) if not same_gene_cells.empty else float("nan")
        )
    pd.DataFrame(rows).to_csv(a.output_dir / "method_comparison.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(a.output_dir / "per_cell_matched.csv", index=False)
    protocol = {"paper": "STCS bioRxiv 2026.02.26.708370v1, Cell-level Alignment Methods",
                "upstream_stcs_commit": "7cdd74dfc3750d03a73490b193ed0fb0f792d768",
                "matching": "StarDist nucleus pixel majority vote inside registered Xenium native polygon; alpha=0.05 predicted polygon in um; both-direction maximum intersection, reciprocal and nuclear-vote agreement",
                "difference_from_paper": "Native Xenium cell-boundary polygons replace convex-hull/alpha-shape polygons reconstructed from Xenium spatial units; predicted bin alpha-shape gets a 1-um bin-footprint buffer; pilot patch rather than whole slide; HVG variance substitute for Seurat-v3; no CellTypist-derived cell type metrics.",
                "gene_rule": gene_context["hvg_rule"], "nuclear_vote_diagnostics": vote_diag,
                "common_matched_pairs_across_methods": len(common_pairs),
                "common_gene_evaluable_pairs_across_methods": len(common_gene_pairs),
                "n_core_seed_cells": len(core_cells), "n_xenium_core_cells": len(xe_core),
                "xenium_use": "evaluation only, never assignment inference"}
    repo = Path(__file__).resolve().parents[1]
    tracked_implementation_inputs = [
        "hd_cell_rl/em_assignment.py", "hd_cell_rl/em_types.py",
        "preprocessing/build_reference_counts_from_scrna.py",
        "preprocessing/run_bin2cell_analysis.py", "preprocessing/run_stcs_analysis.py",
        "scripts/convert_lung_gse131907_rds_to_10x.R",
        "scripts/prepare_lung_paired_patch.py", "scripts/build_lung_pilot_nuclei.py",
        "scripts/run_lung_pilot_em.py", "scripts/qc_lung_paired_alignment.py",
        "scripts/evaluate_lung_paired_patch.py",
    ]
    protocol["source_sha256"] = {
        item: sha256_file(repo / item) for item in tracked_implementation_inputs
    }
    protocol["patch_h5_sha256"] = sha256_file(a.patch_dir / "filtered_feature_bc_matrix.h5")
    protocol["common_nuclear_bins_sha256"] = sha256_file(a.nuclei_dir / "nuclear_bins.csv.gz")
    reference = a.data_root / "reference/gse131907/lung_primary_author_type_reference_counts.npz"
    protocol["independent_reference_sha256"] = sha256_file(reference)
    (a.output_dir / "evaluation_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    main()
