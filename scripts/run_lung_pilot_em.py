#!/usr/bin/env python3
"""Run frozen EM and distance-only controls on one real HD patch, without RL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hd_cell_rl.em_assignment import run_generalized_em, save_em_result_npz
from hd_cell_rl.em_types import EMAssignmentConfig, SparseEMInput
from hd_cell_rl.episode_artifacts import _MatrixOnDemandExpressionLoader
from hd_cell_rl.reward import compute_expression_confidence, compute_reference_distribution


def _make_input(patch: Path, nuclei_dir: Path, reference: Path) -> tuple[SparseEMInput, object, dict[str, object]]:
    manifest = json.loads((patch / "patch_manifest.json").read_text())
    core_r0, core_r1, core_c0, core_c1 = manifest["core_array_bounds"]
    halo = int(manifest["context_halo_bins"])
    episode_cfg = yaml.safe_load((REPO_ROOT / "configs/episode_build.template.yaml").read_text())
    max_distance = float(episode_cfg["environment"]["max_center_distance_um"])
    scales = json.loads((patch / "spatial/scalefactors_json.json").read_text())
    bin_size_um = float(scales["bin_size_um"])
    if halo * bin_size_um < 2.0 * max_distance:
        raise ValueError("Patch halo cannot guarantee all candidate nuclei for scored core bins")
    if "candidate_max_distance_um" in manifest and not np.isclose(
        float(manifest["candidate_max_distance_um"]), max_distance
    ):
        raise ValueError("Patch manifest and resolved RL candidate MaxDis disagree")
    # Every bin scored within core+MaxDis has its full MaxDis nucleus context
    # inside core+2*MaxDis, the extracted 20-bin halo.
    score_halo_bins = int(np.ceil(max_distance / bin_size_um))
    positions = pd.read_parquet(patch / "spatial/tissue_positions.parquet")
    seeds = pd.read_csv(nuclei_dir / "nuclear_bins.csv.gz", dtype={"barcode": str, "cell_id": str})
    nuclei = pd.read_parquet(nuclei_dir / "nuclei.parquet")
    nuclei["cell_id"] = nuclei.cell_id.astype(str)
    if seeds.barcode.duplicated().any() or nuclei.cell_id.duplicated().any():
        raise ValueError("Duplicate seed barcode or physical nucleus ID")
    candidate_cells = tuple(nuclei.cell_id.tolist())
    centers = nuclei[["center_x_um", "center_y_um"]].to_numpy(dtype=np.float64)
    tree = cKDTree(centers)
    locked_by_barcode = dict(zip(seeds.barcode, seeds.cell_id))
    nonnuclear_score = (
        positions.array_row.between(core_r0 - score_halo_bins, core_r1 + score_halo_bins - 1)
        & positions.array_col.between(core_c0 - score_halo_bins, core_c1 + score_halo_bins - 1)
    )
    selected = positions.loc[nonnuclear_score | positions.barcode.isin(locked_by_barcode)].copy()
    with np.load(reference, allow_pickle=False) as data:
        reference_counts = np.asarray(data["reference_counts"], dtype=np.float64)
    theta = compute_reference_distribution(reference_counts, epsilon=1e-12)
    with h5py.File(patch / "filtered_feature_bc_matrix.h5", "r") as h5:
        barcode_index = pd.Index(np.asarray(h5["matrix/barcodes"][:]).astype(str))
    matrix_col = barcode_index.get_indexer(selected.barcode)
    if np.any(matrix_col < 0):
        raise ValueError("Selected EM barcode is missing in patch matrix")
    loader = _MatrixOnDemandExpressionLoader(
        matrix_path=patch / "filtered_feature_bc_matrix.h5",
        reference_npz_path=reference,
        reference_genes_key="genes",
        cache_size=0,
    )
    try:
        ll, totals = loader.compute_ll_and_bin_counts_for_columns(matrix_col, np.log(theta))
        expression = loader.load_sparse_columns(matrix_col)
    finally:
        loader.close()
    # Match the frozen pseudo benchmark's positive-expression non-nuclear
    # filter. Nuclear anchors stay even when they contain zero selected UMIs.
    keep = (totals > 0) | selected.barcode.isin(locked_by_barcode).to_numpy()
    n_filtered_zero = int((~keep).sum())
    selected = selected.loc[keep].copy()
    ll = ll[keep]
    totals = totals[keep]
    expression = expression[keep, :]
    mpp = float(scales["microns_per_pixel"])
    # The actual 10x bin centers, not array_row*2 or a global affine approximation.
    xy = selected[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=np.float64) * mpp
    candidates = tree.query_ball_point(xy, r=max_distance)
    has_candidate = np.fromiter((len(c) > 0 for c in candidates), count=len(candidates), dtype=bool)
    omitted = selected.loc[~has_candidate]
    if any(barcode in locked_by_barcode for barcode in omitted.barcode):
        raise ValueError("Confident nuclear seed outside MaxDis of all nuclei")
    selected = selected.loc[has_candidate].reset_index(drop=True)
    ll = ll[has_candidate]
    totals = totals[has_candidate]
    expression = expression[has_candidate, :]
    xy = xy[has_candidate]
    candidates = [c for c, ok in zip(candidates, has_candidate) if ok]
    rows = np.array([len(c) for c in candidates], dtype=np.int64)
    row_splits = np.r_[0, np.cumsum(rows)]
    pair_cell = np.concatenate([np.asarray(sorted(c), dtype=np.int64) for c in candidates])
    pair_bin = np.repeat(np.arange(len(selected), dtype=np.int64), rows)
    distances = np.linalg.norm(xy[pair_bin] - centers[pair_cell], axis=1)
    if np.max(distances) > max_distance + 1e-5:
        raise ValueError("Candidate graph exceeded RL MaxDis")
    cell_to_idx = {cell: i for i, cell in enumerate(candidate_cells)}
    locked = selected.barcode.isin(locked_by_barcode).to_numpy()
    locked_owner = np.full(len(selected), -1, dtype=np.int64)
    locked_owner[locked] = [cell_to_idx[locked_by_barcode[b]] for b in selected.loc[locked, "barcode"]]
    for i in np.flatnonzero(locked):
        if locked_owner[i] not in pair_cell[row_splits[i]:row_splits[i+1]]:
            raise ValueError(f"Seed owner outside MaxDis for barcode {selected.barcode.iloc[i]}")

    confidence = compute_expression_confidence(totals, pseudocount=5.0)
    em_input = SparseEMInput(
        patch_id=str(manifest.get("patch_id", "lung_pilot_core_1840_1240")),
        candidate_max_distance_um=max_distance,
        barcode_ids=tuple(selected.barcode.astype(str)),
        barcode_xy_um=xy.astype(np.float32),
        ll=ll.astype(np.float32),
        expression_confidence=confidence.astype(np.float32),
        candidate_row_splits=row_splits.astype(np.int64),
        candidate_cell_index=pair_cell.astype(np.int64),
        pair_bin_index=pair_bin.astype(np.int64),
        pair_distance_um=distances.astype(np.float32),
        cell_ids=candidate_cells,
        is_nuclear_locked=locked,
        locked_owner_cell_index=locked_owner,
        log_prior=np.full((len(candidate_cells), theta.shape[0]), -np.log(theta.shape[0]), dtype=np.float64),
    )
    checks = {
        "candidate_max_distance_config_path": "configs/episode_build.template.yaml:environment.max_center_distance_um",
        "candidate_max_distance_um": max_distance,
        "n_scored_bins": len(selected), "n_scored_core_bins": int(selected.loc[
            selected.array_row.between(core_r0, core_r1 - 1)
            & selected.array_col.between(core_c0, core_c1 - 1)
        ].shape[0]),
        "n_dropped_non_nuclear_bins_without_candidate": len(omitted),
        "n_filtered_zero_selected_gene_non_nuclear_bins": n_filtered_zero,
        "n_cells_core_and_context": len(candidate_cells), "n_candidate_pairs": len(pair_cell),
        "max_candidates_per_bin": int(rows.max()), "mean_candidates_per_bin": float(rows.mean()),
        "n_locked_nuclear_bins": int(locked.sum()),
        "max_pair_distance_um": float(distances.max()),
        "reference_n_types": int(theta.shape[0]), "reference_n_genes": int(theta.shape[1]),
        "bin_expression_counts_sum": float(totals.sum()),
        "core_only_evaluation": True,
    }
    return em_input, (expression, theta, selected), checks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--patch-dir", type=Path, required=True)
    p.add_argument("--nuclei-dir", type=Path, required=True)
    p.add_argument("--reference-npz", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    em_input, (expression, theta, selected), checks = _make_input(a.patch_dir, a.nuclei_dir, a.reference_npz)
    (a.output_dir / "input_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    for method, beta in (("distance_only", 0.0), ("em", 1.5)):
        config = EMAssignmentConfig(
            enabled=True, spatial_weight=1.0, expression_weight=beta,
            damping=0.5, max_iterations=50,
            top_owner_change_fraction=0.01, mean_probability_change=0.001,
            non_nuclear_bin_filter="positive_expression", background_enabled=False,
            cell_specific_expression_enabled=True,
            cell_profile_relative_prior_strength=2.0,
            cell_profile_update_damping=0.0,
            cell_profile_compatibility_mode="standard",
            cell_profile_reliability_mode="nuclear_depth",
        )
        result = run_generalized_em(em_input, config, bin_gene_counts=expression, reference_theta=theta)
        result.validate()
        folder = a.output_dir / method
        folder.mkdir(exist_ok=True)
        save_em_result_npz(result, folder / "soft_assignment.npz")
        owners = np.asarray(result.cell_ids, dtype=str)[result.top1_cell_index]
        frame = pd.DataFrame({"barcode": result.barcode_ids, "cell_id": owners,
                              "is_nuclear": result.is_nuclear_locked,
                              "normalized_entropy": result.normalized_entropy})
        frame = frame.loc[~result.is_unassigned].copy()
        frame = frame.merge(selected[["barcode", "array_row", "array_col"]], on="barcode", validate="one_to_one")
        if frame.barcode.duplicated().any():
            raise ValueError("EM produced duplicate physical barcode ownership")
        frame.to_csv(folder / "assignments.csv", index=False)
        summary = {"method": method, "spatial_weight": 1.0, "expression_weight": beta,
                   "profile_relative_prior_strength": 2.0, "profile_update_damping": 0.0,
                   "background_enabled": False, "n_iterations": len(result.iterations),
                   "converged": bool(result.converged), "n_assignments": int(len(frame)),
                   "n_ambiguous": int(result.is_ambiguous.sum()),
                   "mean_normalized_entropy": float(np.mean(result.normalized_entropy)),
                   "n_disconnected_hard_islands": int(result.n_disconnected_hard_islands)}
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
