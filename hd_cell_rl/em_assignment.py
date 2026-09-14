"""Patch-level bin-to-cell generalized EM soft assignment.

The latent observation is one physical 2 um bin/barcode.  Candidate ownership
is represented as ragged ``(bin, physical cell)`` pairs, so repeated barcodes in
per-cell episode artifacts remain one observation and no dense bin-by-cell
matrix is required.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import csv
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from .em_types import (
    EMAssignmentConfig,
    EMAssignmentResult,
    EMIterationDiagnostics,
    SparseEMInput,
)
from .patch_types import PatchContext
from .ppo_state import EpisodeContext, _zscore_1d
from .reward import build_eight_neighbor_index


LOGGER = logging.getLogger(__name__)


def build_sparse_patch_em_input(
    *,
    context: PatchContext,
    candidate_max_distance_um: float,
    non_nuclear_bin_filter: str = "all",
) -> SparseEMInput:
    """Build one unique-barcode sparse EM problem from a loaded patch context.

    All non-nuclear bins inside ``outer_bounds`` are scored.  Confident nuclear
    bins for both core and margin cells are also included, even when the margin
    nucleus lies outside the outer bounds, so those anchors continue to
    contribute in every M-step.
    """
    d_max = float(candidate_max_distance_um)
    bin_filter = str(non_nuclear_bin_filter).strip().lower()
    if bin_filter not in {"all", "positive_expression"}:
        raise ValueError(
            "non_nuclear_bin_filter must be one of: all, positive_expression"
        )
    if d_max <= 0.0:
        raise ValueError("candidate_max_distance_um must be > 0")
    if not context.cells:
        raise ValueError("cannot build generalized EM input for an empty patch")

    n_types = int(context.cells[0].n_cell_types)
    if n_types <= 0:
        raise ValueError("generalized EM requires at least one reference cell type")
    cell_ids = tuple(str(ctx.cell_id) for ctx in context.cells)
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("PatchContext cell IDs must be unique")

    canonical: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    locked_owner_by_barcode: dict[str, int] = {}
    for cell_idx, ctx in enumerate(context.cells):
        if int(ctx.n_cell_types) != n_types:
            raise ValueError("all patch cells must use the same reference cell-type dimension")
        if len(ctx.candidate_bin_ids) != int(ctx.n_bins):
            raise ValueError(f"candidate barcode count mismatch for cell {ctx.cell_id!r}")
        seed = np.asarray(ctx.initial_membership_mask, dtype=np.uint8) > 0
        for bin_idx, raw_barcode in enumerate(ctx.candidate_bin_ids):
            barcode = str(raw_barcode)
            xy = np.asarray(ctx.candidate_bin_xy_um[bin_idx], dtype=np.float64)
            ll_row = np.asarray(ctx.ll[bin_idx], dtype=np.float64)
            conf = float(ctx.expression_confidence[bin_idx])
            existing = canonical.get(barcode)
            if existing is None:
                canonical[barcode] = (xy.copy(), ll_row.copy(), conf)
            else:
                if not np.allclose(existing[0], xy, rtol=0.0, atol=1.0e-4):
                    raise ValueError(f"physical barcode {barcode!r} has inconsistent coordinates")
                if not np.allclose(existing[1], ll_row, rtol=1.0e-5, atol=1.0e-6):
                    raise ValueError(f"physical barcode {barcode!r} has inconsistent LL rows")
                if not np.isclose(existing[2], conf, rtol=1.0e-5, atol=1.0e-6):
                    raise ValueError(
                        f"physical barcode {barcode!r} has inconsistent expression confidence"
                    )
            if bool(seed[bin_idx]):
                previous = locked_owner_by_barcode.get(barcode)
                if previous is not None and int(previous) != int(cell_idx):
                    raise ValueError(
                        f"confident nuclear barcode {barcode!r} has multiple physical owners: "
                        f"{cell_ids[int(previous)]!r}, {ctx.cell_id!r}"
                    )
                locked_owner_by_barcode[barcode] = int(cell_idx)

    selected_barcodes = []
    for barcode, (xy, _ll, conf) in canonical.items():
        inside = bool(context.outer_bounds.contains_xy(xy.reshape(1, 2))[0])
        locked = barcode in locked_owner_by_barcode
        include_non_nuclear = bin_filter == "all" or float(conf) > 0.0
        if locked or (inside and include_non_nuclear):
            selected_barcodes.append(barcode)
    selected_barcodes.sort()
    if not selected_barcodes:
        raise ValueError(f"patch {context.patch_id!r} has no EM-scored or nuclear bins")

    barcode_ids = tuple(selected_barcodes)
    barcode_xy = np.vstack([canonical[barcode][0] for barcode in barcode_ids]).astype(
        np.float64,
        copy=False,
    )
    ll = np.vstack([canonical[barcode][1] for barcode in barcode_ids]).astype(
        np.float64,
        copy=False,
    )
    expression_confidence = np.asarray(
        [canonical[barcode][2] for barcode in barcode_ids],
        dtype=np.float64,
    )
    if not np.isfinite(barcode_xy).all() or not np.isfinite(ll).all():
        raise ValueError("generalized EM input contains NaN or Inf")
    if np.any(expression_confidence < 0.0) or np.any(expression_confidence > 1.0):
        raise ValueError("expression confidence must lie in [0, 1]")

    cell_centers = np.vstack(
        [np.asarray(ctx.nucleus_center_xy_um, dtype=np.float64) for ctx in context.cells]
    )
    cell_tree = cKDTree(cell_centers)
    candidates_by_bin = cell_tree.query_ball_point(barcode_xy, r=d_max)
    row_splits = np.zeros((len(barcode_ids) + 1,), dtype=np.int64)
    candidate_cells: list[int] = []
    pair_distances: list[float] = []
    locked_owner = np.full((len(barcode_ids),), -1, dtype=np.int64)
    is_locked = np.zeros((len(barcode_ids),), dtype=bool)
    for bin_idx, raw_candidates in enumerate(candidates_by_bin):
        candidates = sorted(int(item) for item in raw_candidates)
        barcode = barcode_ids[bin_idx]
        locked = locked_owner_by_barcode.get(barcode)
        if locked is not None:
            if int(locked) not in candidates:
                owner_id = cell_ids[int(locked)]
                raise ValueError(
                    f"nuclear barcode {barcode!r} lies outside candidate MaxDis of owner {owner_id!r}"
                )
            is_locked[bin_idx] = True
            locked_owner[bin_idx] = int(locked)
        if not candidates:
            raise ValueError(
                f"physical barcode {barcode!r} has no nucleus within candidate MaxDis={d_max:g} um"
            )
        xy = barcode_xy[bin_idx]
        for cell_idx in candidates:
            distance = float(np.sqrt(np.sum((xy - cell_centers[cell_idx]) ** 2)))
            if distance > d_max + 1.0e-7:
                continue
            candidate_cells.append(int(cell_idx))
            pair_distances.append(distance)
        row_splits[bin_idx + 1] = len(candidate_cells)

    candidate_cell_index = np.asarray(candidate_cells, dtype=np.int64)
    counts = np.diff(row_splits)
    pair_bin_index = np.repeat(np.arange(len(barcode_ids), dtype=np.int64), counts)
    pair_distance_um = np.asarray(pair_distances, dtype=np.float64)
    if candidate_cell_index.size == 0 or np.any(counts <= 0):
        raise ValueError("every EM-scored bin must have at least one candidate cell")

    log_prior_values = []
    for ctx in context.cells:
        # EpisodeContext currently stores one scalar uniform log prior.  Keeping
        # this expansion here preserves that existing behavior without creating
        # a second prior implementation.
        log_prior_values.append(
            np.full((n_types,), float(ctx.log_prior), dtype=np.float64)
        )
    log_prior = np.vstack(log_prior_values)
    return SparseEMInput(
        patch_id=str(context.patch_id),
        candidate_max_distance_um=d_max,
        barcode_ids=barcode_ids,
        barcode_xy_um=barcode_xy.astype(np.float32),
        ll=ll.astype(np.float32),
        expression_confidence=expression_confidence.astype(np.float32),
        candidate_row_splits=row_splits,
        candidate_cell_index=candidate_cell_index,
        pair_bin_index=pair_bin_index,
        pair_distance_um=pair_distance_um.astype(np.float32),
        cell_ids=cell_ids,
        is_nuclear_locked=is_locked,
        locked_owner_cell_index=locked_owner,
        log_prior=log_prior,
    )


def run_generalized_em(
    em_input: SparseEMInput,
    config: EMAssignmentConfig,
    *,
    initial_cell_type_posterior: np.ndarray | None = None,
    freeze_cell_type_posterior: bool = False,
    fixed_pair_expression_compatibility: np.ndarray | None = None,
    bin_gene_counts: sparse.spmatrix | np.ndarray | None = None,
    reference_theta: np.ndarray | None = None,
) -> EMAssignmentResult:
    """Run batch E/M alternation over a complete sparse patch ownership state."""
    config.validate()
    row_splits = np.asarray(em_input.candidate_row_splits, dtype=np.int64)
    pair_bin = np.asarray(em_input.pair_bin_index, dtype=np.int64)
    pair_cell = np.asarray(em_input.candidate_cell_index, dtype=np.int64)
    ll = np.asarray(em_input.ll, dtype=np.float64)
    conf = np.asarray(em_input.expression_confidence, dtype=np.float64)
    locked = np.asarray(em_input.is_nuclear_locked, dtype=bool)
    locked_owner = np.asarray(em_input.locked_owner_cell_index, dtype=np.int64)
    n_bins = int(ll.shape[0])
    n_cells = int(len(em_input.cell_ids))
    n_types = int(ll.shape[1])
    counts = np.diff(row_splits)
    if row_splits.shape != (n_bins + 1,) or np.any(counts <= 0):
        raise ValueError("candidate_row_splits are invalid")
    if pair_bin.shape != pair_cell.shape:
        raise ValueError("pair_bin_index and candidate_cell_index must have the same shape")
    if pair_cell.size != int(row_splits[-1]):
        raise ValueError("candidate pair count does not match candidate_row_splits")
    if np.any(pair_cell < 0) or np.any(pair_cell >= n_cells):
        raise ValueError("candidate_cell_index contains an invalid cell index")
    if np.any(locked_owner[locked] < 0):
        raise ValueError("locked nuclear rows require a physical owner")
    fixed_compatibility = (
        None
        if fixed_pair_expression_compatibility is None
        else np.asarray(fixed_pair_expression_compatibility, dtype=np.float64)
    )
    if fixed_compatibility is not None:
        if fixed_compatibility.shape != pair_cell.shape:
            raise ValueError(
                "fixed_pair_expression_compatibility must have shape (N_pairs,)"
            )
        if not np.isfinite(fixed_compatibility).all():
            raise ValueError("fixed_pair_expression_compatibility contains NaN or Inf")
    cell_specific_enabled = bool(config.cell_specific_expression_enabled)
    leave_one_out_enabled = (
        cell_specific_enabled
        and config.cell_profile_compatibility_mode == "leave_one_out"
    )
    spatial_block_crossfit_enabled = (
        cell_specific_enabled
        and config.cell_profile_compatibility_mode == "spatial_block_crossfit"
    )
    excluded_profile_enabled = (
        leave_one_out_enabled or spatial_block_crossfit_enabled
    )
    if cell_specific_enabled and fixed_compatibility is not None:
        raise ValueError(
            "alternating cell-specific expression and fixed pair compatibility "
            "are mutually exclusive"
        )

    spatial_prior = compute_spatial_prior(
        pair_distance_um=em_input.pair_distance_um,
        candidate_row_splits=row_splits,
        candidate_max_distance_um=float(em_input.candidate_max_distance_um),
    )
    background_probability = compute_background_probability(
        expression_confidence=conf,
        is_nuclear_locked=locked,
        enabled=bool(config.background_enabled),
        owned_logit_intercept=float(config.background_owned_logit_intercept),
        expression_confidence_weight=float(config.background_confidence_weight),
    )
    background_for_statistics = (
        background_probability if bool(config.background_enabled) else None
    )
    owned_probability = 1.0 - background_probability
    r = (
        spatial_prior.astype(np.float64, copy=True)
        * owned_probability[pair_bin]
    )
    _lock_nuclear_rows(
        responsibilities=r,
        candidate_row_splits=row_splits,
        candidate_cell_index=pair_cell,
        is_nuclear_locked=locked,
        locked_owner_cell_index=locked_owner,
    )
    if initial_cell_type_posterior is None:
        if freeze_cell_type_posterior:
            raise ValueError(
                "freeze_cell_type_posterior requires initial_cell_type_posterior"
            )
        q = initialize_q_from_nuclear_seeds(em_input)
    else:
        q = _validate_cell_type_posterior(
            initial_cell_type_posterior,
            n_cells=n_cells,
            n_types=n_types,
        )
    if cell_specific_enabled:
        if bin_gene_counts is None or reference_theta is None:
            raise ValueError(
                "cell-specific expression requires bin_gene_counts and reference_theta"
            )
        expression = _validate_bin_gene_counts(
            bin_gene_counts,
            n_bins=n_bins,
        )
        theta = _normalize_reference_theta(
            reference_theta,
            n_types=n_types,
            n_genes=int(expression.shape[1]),
            epsilon=float(config.epsilon),
        )
        (
            cell_profile,
            cell_profile_reliability,
            nuclear_expression_total,
            relative_profile_prior_count,
        ) = initialize_cell_expression_profiles(
            em_input=em_input,
            bin_gene_counts=expression,
            reference_theta=theta,
            cell_type_posterior=q,
            relative_prior_strength=float(
                config.cell_profile_relative_prior_strength
            ),
            epsilon=float(config.epsilon),
        )
        bin_expression_total = np.asarray(
            expression.sum(axis=1),
            dtype=np.float64,
        ).reshape(-1)
        positive_locked = locked & (bin_expression_total > 0.0)
        heldout_reliability_nuclear_bins = np.bincount(
            locked_owner[positive_locked],
            minlength=n_cells,
        ).astype(np.int64, copy=False)
        heldout_reliability_gain = np.zeros((n_cells,), dtype=np.float64)
        if config.cell_profile_reliability_mode == "nuclear_heldout_predictive":
            (
                cell_profile_reliability,
                heldout_reliability_gain,
                heldout_reliability_nuclear_bins,
            ) = estimate_nuclear_heldout_profile_reliability(
                em_input=em_input,
                bin_gene_counts=expression,
                reference_theta=theta,
                reference_prior_count=relative_profile_prior_count,
                lambda_grid_size=int(config.heldout_reliability_grid_size),
                epsilon=float(config.epsilon),
            )
        if excluded_profile_enabled:
            cell_profile_pseudocounts = cell_profile * (
                nuclear_expression_total + relative_profile_prior_count
                + float(config.epsilon) * int(expression.shape[1])
            )[:, None]
            profile_pair_responsibility = np.zeros_like(r, dtype=np.float64)
            _lock_nuclear_rows(
                responsibilities=profile_pair_responsibility,
                candidate_row_splits=row_splits,
                candidate_cell_index=pair_cell,
                is_nuclear_locked=locked,
                locked_owner_cell_index=locked_owner,
            )
            if spatial_block_crossfit_enabled:
                spatial_crossfit_fold_index = assign_spatial_crossfit_folds(
                    barcode_xy_um=em_input.barcode_xy_um,
                    is_nuclear_locked=locked,
                    block_size_um=float(config.spatial_crossfit_block_size_um),
                )
            else:
                spatial_crossfit_fold_index = np.empty((0,), dtype=np.int64)
        else:
            cell_profile_pseudocounts = np.empty((n_cells, 0), dtype=np.float64)
            profile_pair_responsibility = np.empty((0,), dtype=np.float64)
            spatial_crossfit_fold_index = np.empty((0,), dtype=np.int64)
    else:
        expression = None
        theta = None
        bin_expression_total = np.empty((0,), dtype=np.float64)
        cell_profile = np.empty((n_cells, 0), dtype=np.float64)
        cell_profile_pseudocounts = np.empty((n_cells, 0), dtype=np.float64)
        profile_pair_responsibility = np.empty((0,), dtype=np.float64)
        spatial_crossfit_fold_index = np.empty((0,), dtype=np.int64)
        cell_profile_reliability = np.zeros((n_cells,), dtype=np.float64)
        heldout_reliability_gain = np.zeros((n_cells,), dtype=np.float64)
        heldout_reliability_nuclear_bins = np.zeros((n_cells,), dtype=np.int64)
        nuclear_expression_total = np.zeros((n_cells,), dtype=np.float64)
        relative_profile_prior_count = 0.0
    diagnostics: list[EMIterationDiagnostics] = []
    converged = False
    final_compatibility = np.zeros_like(r, dtype=np.float64)
    final_type_compatibility = np.zeros_like(r, dtype=np.float64)
    final_cell_compatibility = np.zeros_like(r, dtype=np.float64)
    final_score = np.zeros_like(r, dtype=np.float64)

    for iteration in range(1, int(config.max_iterations) + 1):
        old_r = r.copy()
        old_q = q.copy()
        old_cell_profile = cell_profile.copy()

        type_compatibility = compute_type_expression_compatibility(
            q=old_q,
            ll=ll,
            pair_bin_index=pair_bin,
            candidate_cell_index=pair_cell,
        )
        if cell_specific_enabled:
            if expression is None:
                raise RuntimeError("cell-specific expression matrix was not initialized")
            if leave_one_out_enabled:
                cell_compatibility = (
                    compute_leave_one_out_cell_expression_compatibility(
                        bin_gene_counts=expression,
                        cell_profile_pseudocounts=cell_profile_pseudocounts,
                        profile_pair_responsibility=profile_pair_responsibility,
                        pair_bin_index=pair_bin,
                        candidate_cell_index=pair_cell,
                        epsilon=float(config.epsilon),
                    )
                )
            elif spatial_block_crossfit_enabled:
                cell_compatibility = (
                    compute_spatial_block_crossfit_cell_expression_compatibility(
                        bin_gene_counts=expression,
                        cell_profile_pseudocounts=cell_profile_pseudocounts,
                        profile_pair_responsibility=profile_pair_responsibility,
                        pair_bin_index=pair_bin,
                        candidate_cell_index=pair_cell,
                        spatial_fold_index=spatial_crossfit_fold_index,
                        epsilon=float(config.epsilon),
                    )
                )
            else:
                cell_compatibility = compute_cell_expression_compatibility(
                    bin_gene_counts=expression,
                    cell_expression_profile=old_cell_profile,
                    pair_bin_index=pair_bin,
                    candidate_cell_index=pair_cell,
                    epsilon=float(config.epsilon),
                )
            reliability_by_pair = cell_profile_reliability[pair_cell]
            e_step_compatibility = (
                (1.0 - reliability_by_pair) * type_compatibility
                + reliability_by_pair * cell_compatibility
            )
        elif fixed_compatibility is not None:
            cell_compatibility = fixed_compatibility
            e_step_compatibility = fixed_compatibility
        else:
            cell_compatibility = np.zeros_like(type_compatibility)
            e_step_compatibility = type_compatibility

        raw_r, compatibility, combined_score = compute_all_responsibilities(
            q=old_q,
            ll=ll,
            expression_confidence=conf,
            spatial_prior=spatial_prior,
            candidate_row_splits=row_splits,
            pair_bin_index=pair_bin,
            candidate_cell_index=pair_cell,
            spatial_weight=float(config.spatial_weight),
            expression_weight=float(config.expression_weight),
            epsilon=float(config.epsilon),
            owned_probability_by_bin=owned_probability,
            fixed_pair_expression_compatibility=e_step_compatibility,
        )
        _lock_nuclear_rows(
            responsibilities=raw_r,
            candidate_row_splits=row_splits,
            candidate_cell_index=pair_cell,
            is_nuclear_locked=locked,
            locked_owner_cell_index=locked_owner,
        )
        if float(config.damping) < 1.0:
            r = (
                (1.0 - float(config.damping)) * old_r
                + float(config.damping) * raw_r
            )
        else:
            r = raw_r
        _renormalize_rows(r, row_splits, target_mass=owned_probability)
        _lock_nuclear_rows(
            responsibilities=r,
            candidate_row_splits=row_splits,
            candidate_cell_index=pair_cell,
            is_nuclear_locked=locked,
            locked_owner_cell_index=locked_owner,
        )

        if freeze_cell_type_posterior:
            q = old_q
        else:
            q = update_all_cell_type_posteriors(
                responsibilities=r,
                ll=ll,
                expression_confidence=conf,
                pair_bin_index=pair_bin,
                candidate_cell_index=pair_cell,
                log_prior=em_input.log_prior,
            )
        if cell_specific_enabled:
            if expression is None or theta is None:
                raise RuntimeError("cell-specific expression inputs were not initialized")
            raw_cell_profile = update_all_cell_expression_profiles(
                responsibilities=r,
                bin_gene_counts=expression,
                pair_bin_index=pair_bin,
                candidate_cell_index=pair_cell,
                cell_type_posterior=q,
                reference_theta=theta,
                reference_prior_count=relative_profile_prior_count,
                epsilon=float(config.epsilon),
            )
            cell_profile = apply_cell_profile_update_damping(
                old_profile=old_cell_profile,
                raw_profile=raw_cell_profile,
                damping=float(config.cell_profile_update_damping),
            )
            if excluded_profile_enabled:
                expected_expression_total = np.bincount(
                    pair_cell,
                    weights=r * bin_expression_total[pair_bin],
                    minlength=n_cells,
                )
                cell_profile_pseudocounts = cell_profile * (
                    expected_expression_total + relative_profile_prior_count
                    + float(config.epsilon) * int(expression.shape[1])
                )[:, None]
                profile_pair_responsibility = r.copy()
        old_top = _top_owner_indices(
            old_r,
            row_splits,
            pair_cell,
            background_probability=background_for_statistics,
        )
        new_top = _top_owner_indices(
            r,
            row_splits,
            pair_cell,
            background_probability=background_for_statistics,
        )
        evaluated = ~locked
        fraction_changed = (
            float(np.mean(old_top[evaluated] != new_top[evaluated]))
            if np.any(evaluated)
            else 0.0
        )
        mean_probability_change = float(np.mean(np.abs(r - old_r)))
        entropy = normalized_responsibility_entropy(
            r,
            row_splits,
            background_probability=background_for_statistics,
        )
        ambiguous = (~locked) & (entropy > float(config.entropy_threshold))
        q_change = np.abs(q - old_q)
        if cell_specific_enabled:
            profile_total_variation = 0.5 * np.sum(
                np.abs(cell_profile - old_cell_profile),
                axis=1,
            )
        else:
            profile_total_variation = np.zeros((n_cells,), dtype=np.float64)
        item = EMIterationDiagnostics(
            iteration=int(iteration),
            candidate_max_distance_used=float(em_input.candidate_max_distance_um),
            number_of_unique_bins=n_bins,
            number_of_cells=n_cells,
            number_of_candidate_pairs=int(pair_cell.size),
            mean_candidates_per_bin=float(np.mean(counts)),
            max_candidates_per_bin=int(np.max(counts)),
            mean_responsibility_change=mean_probability_change,
            fraction_top_owner_changed=fraction_changed,
            mean_normalized_entropy=float(np.mean(entropy)),
            median_normalized_entropy=float(np.median(entropy)),
            fraction_ambiguous=float(np.mean(ambiguous)),
            number_nuclear_locked=int(np.sum(locked)),
            mean_q_change=float(np.mean(q_change)),
            max_q_change=float(np.max(q_change)),
            mean_profile_total_variation=float(np.mean(profile_total_variation)),
            max_profile_total_variation=float(np.max(profile_total_variation)),
            mean_cell_profile_reliability=float(
                np.mean(cell_profile_reliability)
            ),
        )
        diagnostics.append(item)
        LOGGER.info("patch generalized EM iteration: %s", asdict(item))
        final_compatibility = compatibility
        final_type_compatibility = type_compatibility
        final_cell_compatibility = cell_compatibility
        final_score = combined_score
        if (
            fraction_changed <= float(config.top_owner_change_fraction)
            and mean_probability_change <= float(config.mean_probability_change)
            and float(np.max(q_change)) <= float(config.max_q_change)
            and (
                not cell_specific_enabled
                or float(np.mean(profile_total_variation))
                <= float(config.mean_profile_total_variation)
            )
        ):
            converged = True
            break

    top1, top1_prob, top2_prob, margin = responsibility_top_statistics(
        responsibilities=r,
        candidate_row_splits=row_splits,
        candidate_cell_index=pair_cell,
        background_probability=background_for_statistics,
    )
    entropy = normalized_responsibility_entropy(
        r,
        row_splits,
        background_probability=background_for_statistics,
    )
    is_ambiguous = (~locked) & (entropy > float(config.entropy_threshold))
    islands = count_disconnected_hard_islands(
        barcode_ids=em_input.barcode_ids,
        barcode_xy_um=em_input.barcode_xy_um,
        top1_cell_index=top1,
        is_nuclear_locked=locked,
        locked_owner_cell_index=locked_owner,
        n_cells=n_cells,
    )
    result = EMAssignmentResult(
        patch_id=str(em_input.patch_id),
        candidate_max_distance_um=float(em_input.candidate_max_distance_um),
        barcode_ids=tuple(em_input.barcode_ids),
        barcode_index=np.arange(n_bins, dtype=np.int64),
        barcode_xy_um=np.asarray(em_input.barcode_xy_um, dtype=np.float32),
        candidate_row_splits=row_splits.astype(np.int64, copy=True),
        candidate_cell_index=pair_cell.astype(np.int64, copy=True),
        pair_bin_index=pair_bin.astype(np.int64, copy=True),
        pair_distance_um=np.asarray(em_input.pair_distance_um, dtype=np.float32),
        spatial_prior=spatial_prior.astype(np.float32),
        expression_confidence=conf.astype(np.float32),
        responsibility=r.astype(np.float32),
        background_probability=background_probability.astype(np.float32),
        top1_cell_index=top1.astype(np.int64),
        top1_probability=top1_prob.astype(np.float32),
        top2_probability=top2_prob.astype(np.float32),
        margin=margin.astype(np.float32),
        normalized_entropy=entropy.astype(np.float32),
        is_nuclear_locked=locked.copy(),
        locked_owner_cell_index=locked_owner.astype(np.int64, copy=True),
        is_ambiguous=is_ambiguous.astype(bool),
        is_unassigned=(top1 < 0),
        cell_ids=tuple(em_input.cell_ids),
        cell_type_posterior=q.astype(np.float32),
        cell_expression_profile=cell_profile.astype(np.float32),
        cell_profile_reliability=cell_profile_reliability.astype(np.float32),
        cell_profile_reliability_heldout_gain=heldout_reliability_gain.astype(
            np.float32
        ),
        cell_profile_reliability_heldout_nuclear_bins=(
            heldout_reliability_nuclear_bins.astype(np.int64, copy=True)
        ),
        nuclear_expression_count_total=nuclear_expression_total.astype(np.float32),
        relative_profile_prior_count=float(relative_profile_prior_count),
        final_type_expression_compatibility=final_type_compatibility.astype(np.float32),
        final_cell_expression_compatibility=final_cell_compatibility.astype(np.float32),
        final_expression_compatibility=final_compatibility.astype(np.float32),
        final_combined_score=final_score.astype(np.float32),
        iterations=tuple(diagnostics),
        converged=bool(converged),
        n_disconnected_hard_islands=int(islands),
    )
    result.validate()
    return result


def compute_spatial_prior(
    *,
    pair_distance_um: np.ndarray,
    candidate_row_splits: np.ndarray,
    candidate_max_distance_um: float,
) -> np.ndarray:
    """Return ragged softmax of ``-(distance / existing MaxDis)^2``."""
    d_max = float(candidate_max_distance_um)
    if d_max <= 0.0:
        raise ValueError("candidate_max_distance_um must be > 0")
    distance = np.asarray(pair_distance_um, dtype=np.float64)
    if np.any(distance < 0.0) or np.any(distance > d_max + 1.0e-5):
        raise ValueError("candidate pair distance lies outside existing RL MaxDis")
    utility = -np.square(distance / d_max)
    return _ragged_softmax(utility, np.asarray(candidate_row_splits, dtype=np.int64))


def initialize_q_from_nuclear_seeds(em_input: SparseEMInput) -> np.ndarray:
    """Initialize one physical-cell type posterior from confident nuclear bins."""
    ll = np.asarray(em_input.ll, dtype=np.float64)
    conf = np.asarray(em_input.expression_confidence, dtype=np.float64)
    locked = np.asarray(em_input.is_nuclear_locked, dtype=bool)
    owner = np.asarray(em_input.locked_owner_cell_index, dtype=np.int64)
    scores = np.asarray(em_input.log_prior, dtype=np.float64).copy()
    if scores.shape != (len(em_input.cell_ids), ll.shape[1]):
        raise ValueError("log_prior must have shape (C, K)")
    locked_bins = np.flatnonzero(locked)
    if locked_bins.size:
        contribution = conf[locked_bins, None] * ll[locked_bins]
        np.add.at(scores, owner[locked_bins], contribution)
    return _softmax_rows(scores)


def initialize_cell_expression_profiles(
    *,
    em_input: SparseEMInput,
    bin_gene_counts: sparse.spmatrix | np.ndarray,
    reference_theta: np.ndarray,
    cell_type_posterior: np.ndarray,
    relative_prior_strength: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Initialize physical-cell profiles from locked nuclei and scRNA shrinkage.

    The reference pseudocount is relative to the patch median positive nuclear
    count, so multiplying every observed count by the same capture-depth factor
    leaves both the normalized profiles and cell reliabilities unchanged.
    """
    counts = _validate_bin_gene_counts(
        bin_gene_counts,
        n_bins=len(em_input.barcode_ids),
    )
    q = _validate_cell_type_posterior(
        cell_type_posterior,
        n_cells=len(em_input.cell_ids),
        n_types=int(em_input.ll.shape[1]),
    )
    theta = _normalize_reference_theta(
        reference_theta,
        n_types=int(em_input.ll.shape[1]),
        n_genes=int(counts.shape[1]),
        epsilon=float(epsilon),
    )
    strength = float(relative_prior_strength)
    if not np.isfinite(strength) or strength <= 0.0:
        raise ValueError("relative_prior_strength must be finite and > 0")

    locked = np.asarray(em_input.is_nuclear_locked, dtype=bool)
    locked_bins = np.flatnonzero(locked)
    locked_owner = np.asarray(
        em_input.locked_owner_cell_index,
        dtype=np.int64,
    )[locked_bins]
    seed_membership = sparse.coo_matrix(
        (
            np.ones((locked_bins.size,), dtype=np.float64),
            (locked_owner, locked_bins),
        ),
        shape=(len(em_input.cell_ids), len(em_input.barcode_ids)),
        dtype=np.float64,
    ).tocsr()
    nuclear_counts = np.asarray((seed_membership @ counts).toarray(), dtype=np.float64)
    nuclear_total = np.sum(nuclear_counts, axis=1)
    positive_total = nuclear_total[nuclear_total > 0.0]
    median_positive_total = (
        float(np.median(positive_total)) if positive_total.size else 1.0
    )
    reference_prior_count = strength * median_positive_total
    reliability = nuclear_total / (nuclear_total + reference_prior_count)
    reference_profile = q @ theta
    profile = _normalize_cell_gene_counts(
        nuclear_counts + reference_prior_count * reference_profile,
        epsilon=float(epsilon),
    )
    return profile, reliability, nuclear_total, float(reference_prior_count)


def estimate_nuclear_heldout_profile_reliability(
    *,
    em_input: SparseEMInput,
    bin_gene_counts: sparse.spmatrix | np.ndarray,
    reference_theta: np.ndarray,
    reference_prior_count: float,
    lambda_grid_size: int,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate per-cell profile weights using locked nuclear bins only.

    Each positive-count nuclear bin is held out once. The cell-type posterior
    and the cell-specific profile used to score that bin are rebuilt without
    the held-out bin. Predictive stacking then chooses the convex probability
    mixture of the scRNA reference profile and nuclear cell profile with the
    highest aggregate held-out multinomial log likelihood. Non-nuclear bins do
    not enter any fitted quantity in this calculation.
    """
    counts = _validate_bin_gene_counts(
        bin_gene_counts,
        n_bins=len(em_input.barcode_ids),
    )
    n_cells = len(em_input.cell_ids)
    n_types = int(em_input.ll.shape[1])
    n_genes = int(counts.shape[1])
    theta = _normalize_reference_theta(
        reference_theta,
        n_types=n_types,
        n_genes=n_genes,
        epsilon=float(epsilon),
    )
    prior_count = float(reference_prior_count)
    if not np.isfinite(prior_count) or prior_count <= 0.0:
        raise ValueError("reference_prior_count must be finite and > 0")
    grid_size = int(lambda_grid_size)
    if grid_size < 2:
        raise ValueError("lambda_grid_size must be >= 2")
    eps = float(epsilon)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("epsilon must be finite and > 0")

    locked = np.asarray(em_input.is_nuclear_locked, dtype=bool)
    owner = np.asarray(em_input.locked_owner_cell_index, dtype=np.int64)
    locked_bins = np.flatnonzero(locked)
    if np.any(owner[locked_bins] < 0) or np.any(owner[locked_bins] >= n_cells):
        raise ValueError("locked nuclear bins require a valid physical owner")
    seed_membership = sparse.coo_matrix(
        (
            np.ones((locked_bins.size,), dtype=np.float64),
            (owner[locked_bins], locked_bins),
        ),
        shape=(n_cells, len(em_input.barcode_ids)),
        dtype=np.float64,
    ).tocsr()
    nuclear_counts = np.asarray((seed_membership @ counts).toarray(), dtype=np.float64)
    nuclear_total = np.sum(nuclear_counts, axis=1)
    bin_total = np.asarray(counts.sum(axis=1), dtype=np.float64).reshape(-1)

    ll = np.asarray(em_input.ll, dtype=np.float64)
    confidence = np.asarray(em_input.expression_confidence, dtype=np.float64)
    seed_score = np.asarray(em_input.log_prior, dtype=np.float64).copy()
    if seed_score.shape != (n_cells, n_types):
        raise ValueError("log_prior must have shape (C, K)")
    if locked_bins.size:
        np.add.at(
            seed_score,
            owner[locked_bins],
            confidence[locked_bins, None] * ll[locked_bins],
        )

    lambda_grid = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    reliability = np.zeros((n_cells,), dtype=np.float64)
    heldout_gain = np.zeros((n_cells,), dtype=np.float64)
    heldout_nuclear_bins = np.zeros((n_cells,), dtype=np.int64)
    for cell_idx in range(n_cells):
        cell_bins = locked_bins[
            (owner[locked_bins] == int(cell_idx)) & (bin_total[locked_bins] > 0.0)
        ]
        heldout_nuclear_bins[cell_idx] = int(cell_bins.size)
        if cell_bins.size < 2:
            continue
        grid_log_likelihood = np.zeros((grid_size,), dtype=np.float64)
        heldout_total = 0.0
        for bin_idx in cell_bins.tolist():
            row = counts.getrow(int(bin_idx))
            gene_index = row.indices
            gene_count = row.data
            total = float(bin_total[int(bin_idx)])
            q_minus = _softmax_rows(
                (
                    seed_score[int(cell_idx)]
                    - confidence[int(bin_idx)] * ll[int(bin_idx)]
                )[None, :]
            )[0]
            reference_probability = np.asarray(
                q_minus @ theta[:, gene_index],
                dtype=np.float64,
            ).reshape(-1)
            remaining_gene_count = (
                nuclear_counts[int(cell_idx), gene_index] - gene_count
            )
            tolerance = 1.0e-8 * max(1.0, float(nuclear_total[int(cell_idx)]))
            if np.any(remaining_gene_count < -tolerance):
                raise ValueError(
                    "nuclear held-out subtraction produced negative gene counts"
                )
            remaining_gene_count = np.maximum(remaining_gene_count, 0.0)
            denominator = (
                float(nuclear_total[int(cell_idx)])
                - total
                + prior_count
                + eps * n_genes
            )
            cell_probability = (
                remaining_gene_count
                + prior_count * reference_probability
                + eps
            ) / denominator
            mixture_probability = (
                reference_probability[None, :]
                + lambda_grid[:, None]
                * (cell_probability - reference_probability)[None, :]
            )
            grid_log_likelihood += np.sum(
                gene_count[None, :]
                * np.log(np.maximum(mixture_probability, eps)),
                axis=1,
            )
            heldout_total += total
        best_score = float(np.max(grid_log_likelihood))
        tie_tolerance = 1.0e-10 * max(1.0, abs(best_score))
        best_indices = np.flatnonzero(
            grid_log_likelihood >= best_score - tie_tolerance
        )
        best_idx = int(best_indices[0])
        reliability[cell_idx] = float(lambda_grid[best_idx])
        heldout_gain[cell_idx] = max(
            0.0,
            float(grid_log_likelihood[best_idx] - grid_log_likelihood[0])
            / heldout_total,
        )
    if (
        not np.isfinite(reliability).all()
        or not np.isfinite(heldout_gain).all()
    ):
        raise ValueError("nuclear held-out reliability contains NaN or Inf")
    return reliability, heldout_gain, heldout_nuclear_bins


def compute_type_expression_compatibility(
    *,
    q: np.ndarray,
    ll: np.ndarray,
    pair_bin_index: np.ndarray,
    candidate_cell_index: np.ndarray,
) -> np.ndarray:
    """Vectorized posterior-weighted type compatibility for sparse pairs."""
    q_arr = np.asarray(q, dtype=np.float64)
    ll_arr = np.asarray(ll, dtype=np.float64)
    pair_bin = np.asarray(pair_bin_index, dtype=np.int64)
    pair_cell = np.asarray(candidate_cell_index, dtype=np.int64)
    return np.einsum(
        "pk,pk->p",
        q_arr[pair_cell],
        ll_arr[pair_bin],
        optimize=False,
    )


def compute_cell_expression_compatibility(
    *,
    bin_gene_counts: sparse.spmatrix | np.ndarray,
    cell_expression_profile: np.ndarray,
    pair_bin_index: np.ndarray,
    candidate_cell_index: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Return normalized multinomial LL for every valid bin-cell pair."""
    pair_bin = np.asarray(pair_bin_index, dtype=np.int64)
    pair_cell = np.asarray(candidate_cell_index, dtype=np.int64)
    n_bins = int(np.max(pair_bin)) + 1 if pair_bin.size else 0
    counts = _validate_bin_gene_counts(bin_gene_counts, n_bins=n_bins)
    profile = np.asarray(cell_expression_profile, dtype=np.float64)
    if profile.ndim != 2 or profile.shape[1] != counts.shape[1]:
        raise ValueError("cell_expression_profile must have shape (C, G)")
    if np.any(profile < 0.0) or not np.isfinite(profile).all():
        raise ValueError("cell_expression_profile must be finite and nonnegative")
    if np.any(pair_cell < 0) or np.any(pair_cell >= profile.shape[0]):
        raise ValueError("candidate_cell_index is outside the cell profile axis")
    normalized_profile = _normalize_cell_gene_counts(
        profile,
        epsilon=float(epsilon),
    )
    log_profile = np.log(np.maximum(normalized_profile, float(epsilon)))
    totals = np.asarray(counts.sum(axis=1), dtype=np.float64).reshape(-1)
    compatibility = np.zeros((pair_bin.size,), dtype=np.float64)
    # Group by physical cell so each sparse matrix-vector product evaluates
    # only valid candidate pairs. This avoids materializing a dense B x C
    # matrix while retaining vectorized accumulation over genes and bins.
    for cell_idx in np.unique(pair_cell).tolist():
        pair_positions = np.flatnonzero(pair_cell == int(cell_idx))
        bin_indices = pair_bin[pair_positions]
        raw = np.asarray(
            counts[bin_indices] @ log_profile[int(cell_idx)],
            dtype=np.float64,
        ).reshape(-1)
        positive = totals[bin_indices] > 0.0
        raw[positive] /= totals[bin_indices[positive]]
        raw[~positive] = 0.0
        compatibility[pair_positions] = raw
    if not np.isfinite(compatibility).all():
        raise ValueError("cell expression compatibility contains NaN or Inf")
    return compatibility


def compute_leave_one_out_cell_expression_compatibility(
    *,
    bin_gene_counts: sparse.spmatrix | np.ndarray,
    cell_profile_pseudocounts: np.ndarray,
    profile_pair_responsibility: np.ndarray,
    pair_bin_index: np.ndarray,
    candidate_cell_index: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Score each pair after removing that bin's own profile contribution.

    `cell_profile_pseudocounts` is the complete numerator used to construct
    the current physical-cell profile. `profile_pair_responsibility` records
    the responsibility with which each candidate pair entered that numerator.
    The implementation groups pairs by physical cell and operates only on CSR
    nonzero genes; it never materializes an `N_pairs x N_genes` array.
    """
    pair_bin = np.asarray(pair_bin_index, dtype=np.int64)
    pair_cell = np.asarray(candidate_cell_index, dtype=np.int64)
    pair_weight = np.asarray(profile_pair_responsibility, dtype=np.float64)
    if pair_bin.shape != pair_cell.shape or pair_bin.shape != pair_weight.shape:
        raise ValueError("pair indices and profile_pair_responsibility shapes differ")
    n_bins = int(np.max(pair_bin)) + 1 if pair_bin.size else 0
    counts = _validate_bin_gene_counts(bin_gene_counts, n_bins=n_bins)
    pseudocounts = np.asarray(cell_profile_pseudocounts, dtype=np.float64)
    if pseudocounts.ndim != 2 or pseudocounts.shape[1] != counts.shape[1]:
        raise ValueError("cell_profile_pseudocounts must have shape (C, G)")
    if np.any(pair_cell < 0) or np.any(pair_cell >= pseudocounts.shape[0]):
        raise ValueError("candidate_cell_index is outside the pseudocount axis")
    if np.any(pseudocounts < 0.0) or not np.isfinite(pseudocounts).all():
        raise ValueError("cell_profile_pseudocounts must be finite and nonnegative")
    if (
        np.any(pair_weight < 0.0)
        or np.any(pair_weight > 1.0)
        or not np.isfinite(pair_weight).all()
    ):
        raise ValueError("profile_pair_responsibility must be finite and in [0, 1]")
    eps = float(epsilon)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("epsilon must be finite and > 0")

    bin_total = np.asarray(counts.sum(axis=1), dtype=np.float64).reshape(-1)
    profile_total = np.sum(pseudocounts, axis=1)
    if np.any(profile_total <= 0.0):
        raise ValueError("every cell profile must have positive total pseudocounts")
    compatibility = np.zeros((pair_bin.size,), dtype=np.float64)
    for cell_idx in np.unique(pair_cell).tolist():
        pair_positions = np.flatnonzero(pair_cell == int(cell_idx))
        bin_indices = pair_bin[pair_positions]
        weight = pair_weight[pair_positions]
        selected = sparse.csr_matrix(counts[bin_indices], dtype=np.float64)
        nonzero_per_row = np.diff(selected.indptr)
        row_for_nonzero = np.repeat(
            np.arange(pair_positions.size, dtype=np.int64),
            nonzero_per_row,
        )
        repeated_weight = np.repeat(weight, nonzero_per_row)
        remaining_gene_count = (
            pseudocounts[int(cell_idx), selected.indices]
            - repeated_weight * selected.data
        )
        numeric_tolerance = 1.0e-8 * max(
            1.0,
            float(profile_total[int(cell_idx)]),
        )
        if np.any(remaining_gene_count < -numeric_tolerance):
            raise ValueError("leave-one-out subtraction produced negative gene counts")
        remaining_total = (
            float(profile_total[int(cell_idx)])
            - weight * bin_total[bin_indices]
        )
        if np.any(remaining_total < -numeric_tolerance):
            raise ValueError("leave-one-out subtraction produced negative total counts")
        log_gene_sum = np.bincount(
            row_for_nonzero,
            weights=selected.data
            * np.log(np.maximum(remaining_gene_count, eps)),
            minlength=pair_positions.size,
        )
        positive = bin_total[bin_indices] > 0.0
        raw = np.zeros((pair_positions.size,), dtype=np.float64)
        raw[positive] = (
            log_gene_sum[positive]
            - bin_total[bin_indices[positive]]
            * np.log(np.maximum(remaining_total[positive], eps))
        ) / bin_total[bin_indices[positive]]
        compatibility[pair_positions] = raw
    if not np.isfinite(compatibility).all():
        raise ValueError("leave-one-out cell compatibility contains NaN or Inf")
    return compatibility


def assign_spatial_crossfit_folds(
    *,
    barcode_xy_um: np.ndarray,
    is_nuclear_locked: np.ndarray,
    block_size_um: float,
) -> np.ndarray:
    """Assign each non-nuclear bin to an absolute-coordinate spatial block.

    Every block is one held-out fold. Locked nuclear bins receive fold ``-1``
    so their counts remain in every cross-fitted physical-cell profile.
    """
    xy = np.asarray(barcode_xy_um, dtype=np.float64)
    locked = np.asarray(is_nuclear_locked, dtype=bool)
    size = float(block_size_um)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("barcode_xy_um must have shape (B, 2)")
    if locked.shape != (xy.shape[0],):
        raise ValueError("is_nuclear_locked must have shape (B,)")
    if not np.isfinite(xy).all():
        raise ValueError("barcode_xy_um must be finite")
    if not np.isfinite(size) or size <= 0.0:
        raise ValueError("spatial cross-fit block size must be finite and > 0")
    fold_index = np.full((xy.shape[0],), -1, dtype=np.int64)
    non_nuclear = ~locked
    if np.any(non_nuclear):
        block_xy = np.floor(xy[non_nuclear] / size).astype(np.int64)
        _unique_blocks, inverse = np.unique(
            block_xy,
            axis=0,
            return_inverse=True,
        )
        fold_index[non_nuclear] = inverse.astype(np.int64, copy=False)
    return fold_index


def compute_spatial_block_crossfit_cell_expression_compatibility(
    *,
    bin_gene_counts: sparse.spmatrix | np.ndarray,
    cell_profile_pseudocounts: np.ndarray,
    profile_pair_responsibility: np.ndarray,
    pair_bin_index: np.ndarray,
    candidate_cell_index: np.ndarray,
    spatial_fold_index: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Score a bin after excluding its entire non-nuclear spatial fold.

    For each physical cell and spatial fold, all previous-iteration expected
    counts ``r[j, c] * x[j]`` from non-nuclear bins in that fold are removed
    together. Nuclear bins have fold ``-1`` and are never removed. Work stays
    sparse over valid candidate pairs and observed CSR genes.
    """
    pair_bin = np.asarray(pair_bin_index, dtype=np.int64)
    pair_cell = np.asarray(candidate_cell_index, dtype=np.int64)
    pair_weight = np.asarray(profile_pair_responsibility, dtype=np.float64)
    if pair_bin.shape != pair_cell.shape or pair_bin.shape != pair_weight.shape:
        raise ValueError("pair indices and profile_pair_responsibility shapes differ")
    n_bins = int(np.max(pair_bin)) + 1 if pair_bin.size else 0
    counts = _validate_bin_gene_counts(bin_gene_counts, n_bins=n_bins)
    fold = np.asarray(spatial_fold_index, dtype=np.int64)
    if fold.shape != (n_bins,):
        raise ValueError("spatial_fold_index must have shape (B,)")
    if np.any(fold < -1):
        raise ValueError("spatial_fold_index values must be -1 or nonnegative")
    pseudocounts = np.asarray(cell_profile_pseudocounts, dtype=np.float64)
    if pseudocounts.ndim != 2 or pseudocounts.shape[1] != counts.shape[1]:
        raise ValueError("cell_profile_pseudocounts must have shape (C, G)")
    if np.any(pair_cell < 0) or np.any(pair_cell >= pseudocounts.shape[0]):
        raise ValueError("candidate_cell_index is outside the pseudocount axis")
    if np.any(pseudocounts < 0.0) or not np.isfinite(pseudocounts).all():
        raise ValueError("cell_profile_pseudocounts must be finite and nonnegative")
    if (
        np.any(pair_weight < 0.0)
        or np.any(pair_weight > 1.0)
        or not np.isfinite(pair_weight).all()
    ):
        raise ValueError("profile_pair_responsibility must be finite and in [0, 1]")
    eps = float(epsilon)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("epsilon must be finite and > 0")

    bin_total = np.asarray(counts.sum(axis=1), dtype=np.float64).reshape(-1)
    profile_total = np.sum(pseudocounts, axis=1)
    if np.any(profile_total <= 0.0):
        raise ValueError("every cell profile must have positive total pseudocounts")
    # Fold -1 rows are locked anchors. Their compatibility is diagnostic only,
    # but scoring them against the complete profile keeps outputs well-defined.
    compatibility = compute_cell_expression_compatibility(
        bin_gene_counts=counts,
        cell_expression_profile=pseudocounts,
        pair_bin_index=pair_bin,
        candidate_cell_index=pair_cell,
        epsilon=eps,
    )
    for cell_idx in np.unique(pair_cell).tolist():
        cell_pair_positions = np.flatnonzero(pair_cell == int(cell_idx))
        fold_by_pair = fold[pair_bin[cell_pair_positions]]
        for fold_idx in np.unique(fold_by_pair[fold_by_pair >= 0]).tolist():
            pair_positions = cell_pair_positions[fold_by_pair == int(fold_idx)]
            bin_indices = pair_bin[pair_positions]
            weight = pair_weight[pair_positions]
            selected = sparse.csr_matrix(counts[bin_indices], dtype=np.float64)
            held_out = (
                sparse.csr_matrix(weight.reshape(1, -1), dtype=np.float64)
                @ selected
            ).tocsr()
            held_out.sum_duplicates()
            held_out.sort_indices()

            held_out_at_selected = np.zeros_like(selected.data, dtype=np.float64)
            if held_out.nnz:
                lookup = np.searchsorted(held_out.indices, selected.indices)
                valid = lookup < held_out.indices.size
                matched = np.zeros_like(valid, dtype=bool)
                matched[valid] = (
                    held_out.indices[lookup[valid]] == selected.indices[valid]
                )
                held_out_at_selected[matched] = held_out.data[lookup[matched]]
            remaining_gene_count = (
                pseudocounts[int(cell_idx), selected.indices]
                - held_out_at_selected
            )
            held_out_total = float(np.dot(weight, bin_total[bin_indices]))
            remaining_total = float(profile_total[int(cell_idx)]) - held_out_total
            numeric_tolerance = 1.0e-8 * max(
                1.0,
                float(profile_total[int(cell_idx)]),
            )
            if np.any(remaining_gene_count < -numeric_tolerance):
                raise ValueError(
                    "spatial block cross-fit subtraction produced negative gene counts"
                )
            if remaining_total < -numeric_tolerance:
                raise ValueError(
                    "spatial block cross-fit subtraction produced negative total counts"
                )

            nonzero_per_row = np.diff(selected.indptr)
            row_for_nonzero = np.repeat(
                np.arange(pair_positions.size, dtype=np.int64),
                nonzero_per_row,
            )
            log_gene_sum = np.bincount(
                row_for_nonzero,
                weights=selected.data
                * np.log(np.maximum(remaining_gene_count, eps)),
                minlength=pair_positions.size,
            )
            positive = bin_total[bin_indices] > 0.0
            raw = np.zeros((pair_positions.size,), dtype=np.float64)
            raw[positive] = (
                log_gene_sum[positive]
                - bin_total[bin_indices[positive]]
                * np.log(max(remaining_total, eps))
            ) / bin_total[bin_indices[positive]]
            compatibility[pair_positions] = raw
    if not np.isfinite(compatibility).all():
        raise ValueError("spatial block cross-fit compatibility contains NaN or Inf")
    return compatibility


def update_all_cell_expression_profiles(
    *,
    responsibilities: np.ndarray,
    bin_gene_counts: sparse.spmatrix | np.ndarray,
    pair_bin_index: np.ndarray,
    candidate_cell_index: np.ndarray,
    cell_type_posterior: np.ndarray,
    reference_theta: np.ndarray,
    reference_prior_count: float,
    epsilon: float,
) -> np.ndarray:
    """Batch M-step for all physical-cell expression profiles."""
    r = np.asarray(responsibilities, dtype=np.float64)
    pair_bin = np.asarray(pair_bin_index, dtype=np.int64)
    pair_cell = np.asarray(candidate_cell_index, dtype=np.int64)
    if r.shape != pair_bin.shape or r.shape != pair_cell.shape:
        raise ValueError("responsibility and pair-index shapes differ")
    n_bins = int(np.max(pair_bin)) + 1 if pair_bin.size else 0
    counts = _validate_bin_gene_counts(bin_gene_counts, n_bins=n_bins)
    q_raw = np.asarray(cell_type_posterior, dtype=np.float64)
    if q_raw.ndim != 2:
        raise ValueError("cell_type_posterior must have shape (C, K)")
    q = _validate_cell_type_posterior(
        q_raw,
        n_cells=int(q_raw.shape[0]),
        n_types=int(q_raw.shape[1]),
    )
    if np.any(pair_cell < 0) or np.any(pair_cell >= q.shape[0]):
        raise ValueError("candidate_cell_index is outside the cell posterior axis")
    theta = _normalize_reference_theta(
        reference_theta,
        n_types=int(q.shape[1]),
        n_genes=int(counts.shape[1]),
        epsilon=float(epsilon),
    )
    prior_count = float(reference_prior_count)
    if not np.isfinite(prior_count) or prior_count <= 0.0:
        raise ValueError("reference_prior_count must be finite and > 0")
    membership = sparse.coo_matrix(
        (r, (pair_cell, pair_bin)),
        shape=(q.shape[0], n_bins),
        dtype=np.float64,
    ).tocsr()
    weighted_counts = np.asarray((membership @ counts).toarray(), dtype=np.float64)
    reference_profile = q @ theta
    return _normalize_cell_gene_counts(
        weighted_counts + prior_count * reference_profile,
        epsilon=float(epsilon),
    )


def apply_cell_profile_update_damping(
    *,
    old_profile: np.ndarray,
    raw_profile: np.ndarray,
    damping: float,
) -> np.ndarray:
    """Damp one full-batch physical-cell profile M-step.

    A value of zero freezes the nuclear-initialized profile. A value of one
    returns the undamped batch M-step exactly.
    """
    old = np.asarray(old_profile, dtype=np.float64)
    raw = np.asarray(raw_profile, dtype=np.float64)
    if old.shape != raw.shape or old.ndim != 2:
        raise ValueError("old_profile and raw_profile must have matching (C, G) shape")
    if not np.isfinite(old).all() or not np.isfinite(raw).all():
        raise ValueError("cell expression profiles must be finite")
    value = float(damping)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("profile update damping must be finite and in [0, 1]")
    if value == 0.0:
        return old.copy()
    if value == 1.0:
        return raw.copy()
    updated = (1.0 - value) * old + value * raw
    row_sum = np.sum(updated, axis=1)
    if not np.allclose(row_sum, 1.0, rtol=0.0, atol=1.0e-10):
        raise ValueError("damped cell expression profile rows do not sum to one")
    return updated


def compute_all_responsibilities(
    *,
    q: np.ndarray,
    ll: np.ndarray,
    expression_confidence: np.ndarray,
    spatial_prior: np.ndarray,
    candidate_row_splits: np.ndarray,
    pair_bin_index: np.ndarray,
    candidate_cell_index: np.ndarray,
    spatial_weight: float,
    expression_weight: float,
    epsilon: float,
    owned_probability_by_bin: np.ndarray | None = None,
    fixed_pair_expression_compatibility: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One batch E-step using one fixed ``q`` for every bin in the patch."""
    q_arr = np.asarray(q, dtype=np.float64)
    ll_arr = np.asarray(ll, dtype=np.float64)
    pair_bin = np.asarray(pair_bin_index, dtype=np.int64)
    pair_cell = np.asarray(candidate_cell_index, dtype=np.int64)
    if fixed_pair_expression_compatibility is None:
        compatibility = compute_type_expression_compatibility(
            q=q_arr,
            ll=ll_arr,
            pair_bin_index=pair_bin,
            candidate_cell_index=pair_cell,
        )
    else:
        compatibility = np.asarray(
            fixed_pair_expression_compatibility,
            dtype=np.float64,
        )
        if compatibility.shape != pair_cell.shape:
            raise ValueError(
                "fixed_pair_expression_compatibility must have shape (N_pairs,)"
            )
    conf = np.asarray(expression_confidence, dtype=np.float64)
    pi = np.asarray(spatial_prior, dtype=np.float64)
    score = (
        float(spatial_weight) * np.log(np.maximum(pi, float(epsilon)))
        + float(expression_weight) * conf[pair_bin] * compatibility
    )
    responsibility = _ragged_softmax(
        score,
        np.asarray(candidate_row_splits, dtype=np.int64),
    )
    if owned_probability_by_bin is not None:
        owned = np.asarray(owned_probability_by_bin, dtype=np.float64)
        if owned.shape != (ll_arr.shape[0],):
            raise ValueError("owned_probability_by_bin must have shape (B,)")
        if np.any(owned < 0.0) or np.any(owned > 1.0):
            raise ValueError("owned_probability_by_bin must lie in [0, 1]")
        responsibility *= owned[pair_bin]
    return responsibility, compatibility, score


def compute_background_probability(
    *,
    expression_confidence: np.ndarray,
    is_nuclear_locked: np.ndarray,
    enabled: bool,
    owned_logit_intercept: float,
    expression_confidence_weight: float,
) -> np.ndarray:
    """Return a fixed non-GT background gate for each physical bin.

    The existing E-step remains the conditional distribution over physical
    candidate cells.  This gate estimates the probability that a bin belongs
    to any physical cell at all; the residual is the explicit background
    responsibility.  Locked nuclear anchors always have background mass zero.
    """
    confidence = np.asarray(expression_confidence, dtype=np.float64)
    locked = np.asarray(is_nuclear_locked, dtype=bool)
    if confidence.shape != locked.shape:
        raise ValueError("expression confidence and nuclear lock shapes differ")
    if not enabled:
        return np.zeros_like(confidence, dtype=np.float64)
    owned_logit = (
        float(owned_logit_intercept)
        + float(expression_confidence_weight) * confidence
    )
    owned = np.empty_like(owned_logit, dtype=np.float64)
    positive = owned_logit >= 0.0
    owned[positive] = 1.0 / (1.0 + np.exp(-owned_logit[positive]))
    exp_value = np.exp(owned_logit[~positive])
    owned[~positive] = exp_value / (1.0 + exp_value)
    background = 1.0 - owned
    background[locked] = 0.0
    return np.clip(background, 0.0, 1.0)


def update_all_cell_type_posteriors(
    *,
    responsibilities: np.ndarray,
    ll: np.ndarray,
    expression_confidence: np.ndarray,
    pair_bin_index: np.ndarray,
    candidate_cell_index: np.ndarray,
    log_prior: np.ndarray,
) -> np.ndarray:
    """One batch M-step recomputed from the complete current responsibility state."""
    r = np.asarray(responsibilities, dtype=np.float64)
    ll_arr = np.asarray(ll, dtype=np.float64)
    conf = np.asarray(expression_confidence, dtype=np.float64)
    pair_bin = np.asarray(pair_bin_index, dtype=np.int64)
    pair_cell = np.asarray(candidate_cell_index, dtype=np.int64)
    scores = np.asarray(log_prior, dtype=np.float64).copy()
    contribution = (
        r[:, None]
        * conf[pair_bin, None]
        * ll_arr[pair_bin]
    )
    np.add.at(scores, pair_cell, contribution)
    return _softmax_rows(scores)


def normalized_responsibility_entropy(
    responsibilities: np.ndarray,
    candidate_row_splits: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
    background_probability: np.ndarray | None = None,
) -> np.ndarray:
    """Compute entropy divided by log(candidate count), with singleton rows at zero."""
    r = np.asarray(responsibilities, dtype=np.float64)
    splits = np.asarray(candidate_row_splits, dtype=np.int64)
    counts = np.diff(splits)
    raw_pair = -r * np.log(np.maximum(r, float(epsilon)))
    raw = np.add.reduceat(raw_pair, splits[:-1])
    effective_counts = counts.copy()
    if background_probability is not None:
        background = np.asarray(background_probability, dtype=np.float64)
        if background.shape != (counts.shape[0],):
            raise ValueError("background_probability must have shape (B,)")
        raw += -background * np.log(np.maximum(background, float(epsilon)))
        effective_counts = effective_counts + 1
    out = np.zeros((counts.shape[0],), dtype=np.float64)
    multi = effective_counts > 1
    out[multi] = raw[multi] / np.log(effective_counts[multi].astype(np.float64))
    out = np.clip(out, 0.0, 1.0)
    out[~multi] = 0.0
    return out


def responsibility_top_statistics(
    *,
    responsibilities: np.ndarray,
    candidate_row_splits: np.ndarray,
    candidate_cell_index: np.ndarray,
    background_probability: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return top owner, top-1/top-2 probability, and margin for each ragged row."""
    r = np.asarray(responsibilities, dtype=np.float64)
    splits = np.asarray(candidate_row_splits, dtype=np.int64)
    cells = np.asarray(candidate_cell_index, dtype=np.int64)
    n_bins = int(splits.shape[0] - 1)
    top_cell = np.full((n_bins,), -1, dtype=np.int64)
    top1 = np.zeros((n_bins,), dtype=np.float64)
    top2 = np.zeros((n_bins,), dtype=np.float64)
    background = (
        None
        if background_probability is None
        else np.asarray(background_probability, dtype=np.float64)
    )
    if background is not None and background.shape != (n_bins,):
        raise ValueError("background_probability must have shape (B,)")
    for bin_idx in range(n_bins):
        start = int(splits[bin_idx])
        end = int(splits[bin_idx + 1])
        row = r[start:end]
        component_probability = (
            row
            if background is None
            else np.concatenate((row, background[bin_idx : bin_idx + 1]))
        )
        component_cell = (
            cells[start:end]
            if background is None
            else np.concatenate((cells[start:end], np.asarray([-1], dtype=np.int64)))
        )
        order = np.argsort(-component_probability, kind="stable")
        best = int(order[0])
        top_cell[bin_idx] = int(component_cell[best])
        top1[bin_idx] = float(component_probability[best])
        if component_probability.size > 1:
            top2[bin_idx] = float(component_probability[int(order[1])])
    return top_cell, top1, top2, top1 - top2


def initialize_patch_context_from_em(
    *,
    context: PatchContext,
    config: EMAssignmentConfig,
    candidate_max_distance_um: float,
    bin_gene_counts: sparse.spmatrix | np.ndarray | None = None,
    reference_theta: np.ndarray | None = None,
) -> PatchContext:
    """Run EM, align all candidate relations, and attach the result for RL reset."""
    em_input = build_sparse_patch_em_input(
        context=context,
        candidate_max_distance_um=float(candidate_max_distance_um),
        non_nuclear_bin_filter=config.non_nuclear_bin_filter,
    )
    augmented_cells = _augment_episode_contexts_for_em_pairs(context.cells, em_input)
    augmented = replace(
        context,
        cells=augmented_cells,
        candidate_max_distance_um=float(candidate_max_distance_um),
    )
    # Rebuild after augmentation so every sparse EM pair and every future RL
    # REPLACE target share an explicit local candidate-bin index.
    em_input = build_sparse_patch_em_input(
        context=augmented,
        candidate_max_distance_um=float(candidate_max_distance_um),
        non_nuclear_bin_filter=config.non_nuclear_bin_filter,
    )
    result = run_generalized_em(
        em_input,
        config,
        bin_gene_counts=bin_gene_counts,
        reference_theta=reference_theta,
    )
    output = replace(augmented, em_assignment=result)
    output = replace(
        output,
        em_refine_ambiguous_only=bool(config.refine_ambiguous_only),
    )
    if config.save_artifacts:
        if config.artifact_dir is None:
            raise ValueError("em_assignment.artifacts.directory is required when save=true")
        save_em_result_npz(result, config.artifact_dir / f"{result.patch_id}.em_assignment.npz")
    if str(result.patch_id) in set(config.debug_patch_ids):
        if config.artifact_dir is None:
            raise ValueError("debug EM export requires em_assignment.artifacts.directory")
        save_em_debug_csv(result, config.artifact_dir / f"{result.patch_id}.em_debug.csv")
    return output


def _augment_episode_contexts_for_em_pairs(
    cells: tuple[EpisodeContext, ...],
    em_input: SparseEMInput,
) -> tuple[EpisodeContext, ...]:
    """Add missing local candidate rows needed by all MaxDis-valid EM pairs."""
    pair_cells_by_bin: list[np.ndarray] = []
    for bin_idx in range(len(em_input.barcode_ids)):
        start = int(em_input.candidate_row_splits[bin_idx])
        end = int(em_input.candidate_row_splits[bin_idx + 1])
        pair_cells_by_bin.append(em_input.candidate_cell_index[start:end])

    centers = np.vstack(
        [np.asarray(ctx.nucleus_center_xy_um, dtype=np.float64) for ctx in cells]
    )
    out: list[EpisodeContext] = []
    for cell_idx, ctx in enumerate(cells):
        existing_index = {
            str(barcode): idx for idx, barcode in enumerate(ctx.candidate_bin_ids)
        }
        missing_bin_indices = [
            bin_idx
            for bin_idx, candidates in enumerate(pair_cells_by_bin)
            if int(cell_idx) in set(candidates.tolist())
            and em_input.barcode_ids[bin_idx] not in existing_index
        ]
        if not missing_bin_indices:
            out.append(ctx)
            continue
        if ctx.stcs_reward_scores is not None:
            raise ValueError(
                "EM candidate completion with STCS reward is not supported in version 1"
            )
        old_n = int(ctx.n_bins)
        new_ids = tuple(
            [*ctx.candidate_bin_ids, *[em_input.barcode_ids[idx] for idx in missing_bin_indices]]
        )
        new_xy = np.vstack(
            (
                np.asarray(ctx.candidate_bin_xy_um, dtype=np.float32),
                np.asarray(em_input.barcode_xy_um[missing_bin_indices], dtype=np.float32),
            )
        )
        new_ll = np.vstack(
            (
                np.asarray(ctx.ll, dtype=np.float32),
                np.asarray(em_input.ll[missing_bin_indices], dtype=np.float32),
            )
        )
        new_conf = np.concatenate(
            (
                np.asarray(ctx.expression_confidence, dtype=np.float32),
                np.asarray(em_input.expression_confidence[missing_bin_indices], dtype=np.float32),
            )
        )
        new_totals = _counts_from_confidence(
            confidence=new_conf,
            pseudocount=float(ctx.expression_confidence_pseudocount),
        )
        old_totals = np.asarray(ctx.bin_count_totals, dtype=np.float32)
        new_totals[:old_n] = old_totals
        new_seed = np.concatenate(
            (
                np.asarray(ctx.initial_membership_mask, dtype=np.uint8),
                np.asarray(
                    [
                        1
                        if bool(em_input.is_nuclear_locked[idx])
                        and int(em_input.locked_owner_cell_index[idx]) == int(cell_idx)
                        else 0
                        for idx in missing_bin_indices
                    ],
                    dtype=np.uint8,
                ),
            )
        )
        delta = new_xy.astype(np.float64) - centers[cell_idx][None, :]
        d_n = np.sqrt(np.sum(delta * delta, axis=1))
        other_centers = np.delete(centers, int(cell_idx), axis=0)
        if other_centers.shape[0] == 0:
            d_other = np.full((new_xy.shape[0],), np.inf, dtype=np.float64)
        else:
            other_delta = new_xy.astype(np.float64)[:, None, :] - other_centers[None, :, :]
            d_other = np.min(np.sqrt(np.sum(other_delta * other_delta, axis=2)), axis=1)
        p_dis = d_n / float(ctx.r_max_um)
        p_overlap = np.maximum(0.0, (d_n - d_other) / float(ctx.r_max_um))
        ll_mean = np.mean(new_ll, axis=1)
        ll_max = np.max(new_ll, axis=1)
        out.append(
            replace(
                ctx,
                candidate_bin_ids=new_ids,
                initial_membership_mask=new_seed.astype(np.uint8),
                candidate_bin_xy_um=new_xy.astype(np.float32),
                ll=new_ll.astype(np.float32),
                p_dis=p_dis.astype(np.float32),
                p_overlap=p_overlap.astype(np.float32),
                ll_mean_z=_zscore_1d(ll_mean).astype(np.float32),
                ll_max_z=_zscore_1d(ll_max).astype(np.float32),
                base_penalty=(
                    float(ctx.w2) * p_dis + float(ctx.w3) * p_overlap
                ).astype(np.float32),
                expression_confidence=new_conf.astype(np.float32),
                bin_count_totals=new_totals.astype(np.float32),
                neighbor_index=build_eight_neighbor_index(new_ids, new_xy).astype(np.int32),
            )
        )
    return tuple(out)


def _counts_from_confidence(*, confidence: np.ndarray, pseudocount: float) -> np.ndarray:
    conf = np.asarray(confidence, dtype=np.float64)
    if pseudocount <= 0.0:
        return np.where(conf > 0.0, 1.0, 0.0).astype(np.float32)
    clipped = np.clip(conf, 0.0, 1.0 - 1.0e-7)
    return (float(pseudocount) * clipped / np.maximum(1.0 - clipped, 1.0e-7)).astype(
        np.float32
    )


def _ragged_softmax(scores: np.ndarray, row_splits: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    splits = np.asarray(row_splits, dtype=np.int64)
    counts = np.diff(splits)
    if splits.ndim != 1 or splits.size < 2 or np.any(counts <= 0):
        raise ValueError("row_splits must describe non-empty ragged rows")
    if int(splits[0]) != 0 or int(splits[-1]) != int(values.size):
        raise ValueError("row_splits do not align with scores")
    maxima = np.maximum.reduceat(values, splits[:-1])
    shifted = values - np.repeat(maxima, counts)
    exp_values = np.exp(shifted)
    denominators = np.add.reduceat(exp_values, splits[:-1])
    if np.any(denominators <= 0.0) or not np.isfinite(denominators).all():
        raise ValueError("ragged softmax denominator is non-finite or non-positive")
    out = exp_values / np.repeat(denominators, counts)
    if not np.isfinite(out).all():
        raise ValueError("ragged softmax produced NaN or Inf")
    return out


def _validate_bin_gene_counts(
    value: sparse.spmatrix | np.ndarray,
    *,
    n_bins: int,
) -> sparse.csr_matrix:
    counts = sparse.csr_matrix(value, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[0] != int(n_bins) or counts.shape[1] <= 0:
        raise ValueError("bin_gene_counts must have shape (B, G) with G > 0")
    if counts.data.size and (
        not np.isfinite(counts.data).all() or np.any(counts.data < 0.0)
    ):
        raise ValueError("bin_gene_counts must be finite and nonnegative")
    counts.sum_duplicates()
    counts.eliminate_zeros()
    return counts


def _normalize_reference_theta(
    value: np.ndarray,
    *,
    n_types: int,
    n_genes: int,
    epsilon: float,
) -> np.ndarray:
    theta = np.asarray(value, dtype=np.float64)
    if theta.shape != (int(n_types), int(n_genes)):
        raise ValueError("reference_theta must have shape (K, G)")
    if not np.isfinite(theta).all() or np.any(theta < 0.0):
        raise ValueError("reference_theta must be finite and nonnegative")
    return _normalize_cell_gene_counts(theta, epsilon=float(epsilon))


def _normalize_cell_gene_counts(value: np.ndarray, *, epsilon: float) -> np.ndarray:
    counts = np.asarray(value, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] <= 0:
        raise ValueError("cell-gene values must have shape (rows, G) with G > 0")
    if not np.isfinite(counts).all() or np.any(counts < 0.0):
        raise ValueError("cell-gene values must be finite and nonnegative")
    smoothed = counts + float(epsilon)
    denominator = np.sum(smoothed, axis=1, keepdims=True)
    if np.any(denominator <= 0.0) or not np.isfinite(denominator).all():
        raise ValueError("cell-gene rows must have positive finite mass")
    return smoothed / denominator


def _softmax_rows(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    denominator = np.sum(exp_values, axis=1, keepdims=True)
    if np.any(denominator <= 0.0) or not np.isfinite(denominator).all():
        raise ValueError("cell-type softmax denominator is non-finite or non-positive")
    return exp_values / denominator


def _validate_cell_type_posterior(
    value: np.ndarray,
    *,
    n_cells: int,
    n_types: int,
) -> np.ndarray:
    posterior = np.asarray(value, dtype=np.float64)
    if posterior.shape != (int(n_cells), int(n_types)):
        raise ValueError("initial_cell_type_posterior must have shape (C, K)")
    if not np.isfinite(posterior).all() or np.any(posterior < 0.0):
        raise ValueError("initial_cell_type_posterior must be finite and nonnegative")
    row_sum = np.sum(posterior, axis=1, keepdims=True)
    if np.any(row_sum <= 0.0):
        raise ValueError("initial_cell_type_posterior rows must have positive mass")
    return posterior / row_sum


def _renormalize_rows(
    responsibilities: np.ndarray,
    row_splits: np.ndarray,
    *,
    target_mass: np.ndarray | None = None,
) -> None:
    counts = np.diff(row_splits)
    sums = np.add.reduceat(responsibilities, row_splits[:-1])
    if np.any(sums <= 0.0) or not np.isfinite(sums).all():
        raise ValueError("responsibility row sum is non-finite or non-positive")
    targets = (
        np.ones((counts.shape[0],), dtype=np.float64)
        if target_mass is None
        else np.asarray(target_mass, dtype=np.float64)
    )
    if targets.shape != (counts.shape[0],):
        raise ValueError("target_mass must have shape (B,)")
    responsibilities *= np.repeat(targets / sums, counts)


def _lock_nuclear_rows(
    *,
    responsibilities: np.ndarray,
    candidate_row_splits: np.ndarray,
    candidate_cell_index: np.ndarray,
    is_nuclear_locked: np.ndarray,
    locked_owner_cell_index: np.ndarray,
) -> None:
    for bin_idx in np.flatnonzero(is_nuclear_locked).tolist():
        start = int(candidate_row_splits[bin_idx])
        end = int(candidate_row_splits[bin_idx + 1])
        cells = candidate_cell_index[start:end]
        owner = int(locked_owner_cell_index[bin_idx])
        matches = cells == owner
        if int(np.sum(matches)) != 1:
            raise ValueError("nuclear owner must appear exactly once in its candidate row")
        row = responsibilities[start:end]
        row.fill(0.0)
        row[matches] = 1.0


def _top_owner_indices(
    responsibilities: np.ndarray,
    row_splits: np.ndarray,
    candidate_cell_index: np.ndarray,
    background_probability: np.ndarray | None = None,
) -> np.ndarray:
    top, _p1, _p2, _margin = responsibility_top_statistics(
        responsibilities=responsibilities,
        candidate_row_splits=row_splits,
        candidate_cell_index=candidate_cell_index,
        background_probability=background_probability,
    )
    return top


def count_disconnected_hard_islands(
    *,
    barcode_ids: tuple[str, ...],
    barcode_xy_um: np.ndarray,
    top1_cell_index: np.ndarray,
    is_nuclear_locked: np.ndarray,
    locked_owner_cell_index: np.ndarray,
    n_cells: int,
) -> int:
    """Count hardened components that are disconnected from their cell's seeds."""
    xy = np.asarray(barcode_xy_um, dtype=np.float64)
    owners = np.asarray(top1_cell_index, dtype=np.int64)
    locked = np.asarray(is_nuclear_locked, dtype=bool)
    locked_owner = np.asarray(locked_owner_cell_index, dtype=np.int64)
    island_count = 0
    for cell_idx in range(int(n_cells)):
        owned_global = np.flatnonzero(owners == int(cell_idx))
        if owned_global.size <= 1:
            continue
        owned_ids = tuple(str(barcode_ids[idx]) for idx in owned_global.tolist())
        neighbors = build_eight_neighbor_index(owned_ids, xy[owned_global])
        visited = np.zeros((owned_global.size,), dtype=bool)
        components: list[np.ndarray] = []
        for start in range(owned_global.size):
            if visited[start]:
                continue
            stack = [int(start)]
            visited[start] = True
            component: list[int] = []
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in neighbors[current].tolist():
                    if neighbor < 0 or visited[int(neighbor)]:
                        continue
                    visited[int(neighbor)] = True
                    stack.append(int(neighbor))
            components.append(np.asarray(component, dtype=np.int64))
        seeded_component = []
        for component in components:
            global_indices = owned_global[component]
            has_seed = np.any(
                locked[global_indices]
                & (locked_owner[global_indices] == int(cell_idx))
            )
            seeded_component.append(bool(has_seed))
        if any(seeded_component):
            island_count += int(sum(not item for item in seeded_component))
        else:
            island_count += max(0, len(components) - 1)
    return int(island_count)


def em_debug_rows(result: EMAssignmentResult) -> list[dict[str, Any]]:
    """Return a human-readable long table for a selected debug patch."""
    rows: list[dict[str, Any]] = []
    for bin_idx, barcode in enumerate(result.barcode_ids):
        start = int(result.candidate_row_splits[bin_idx])
        end = int(result.candidate_row_splits[bin_idx + 1])
        for pair_idx in range(start, end):
            cell_idx = int(result.candidate_cell_index[pair_idx])
            rows.append(
                {
                    "barcode": str(barcode),
                    "candidate_cell_id": str(result.cell_ids[cell_idx]),
                    "distance_um": float(result.pair_distance_um[pair_idx]),
                    "spatial_prior": float(result.spatial_prior[pair_idx]),
                    "expression_confidence": float(result.expression_confidence[bin_idx]),
                    "expression_compatibility": float(
                        result.final_expression_compatibility[pair_idx]
                    ),
                    "type_expression_compatibility": float(
                        result.final_type_expression_compatibility[pair_idx]
                    ),
                    "cell_expression_compatibility": float(
                        result.final_cell_expression_compatibility[pair_idx]
                    ),
                    "cell_profile_reliability": float(
                        result.cell_profile_reliability[cell_idx]
                    ),
                    "cell_profile_reliability_heldout_gain": float(
                        result.cell_profile_reliability_heldout_gain[cell_idx]
                    ),
                    "cell_profile_reliability_heldout_nuclear_bins": int(
                        result.cell_profile_reliability_heldout_nuclear_bins[cell_idx]
                    ),
                    "combined_score": float(result.final_combined_score[pair_idx]),
                    "responsibility": float(result.responsibility[pair_idx]),
                    "background_probability": float(
                        result.background_probability[bin_idx]
                    ),
                    "normalized_entropy": float(result.normalized_entropy[bin_idx]),
                    "is_nuclear_locked": bool(result.is_nuclear_locked[bin_idx]),
                    "is_ambiguous": bool(result.is_ambiguous[bin_idx]),
                    "is_unassigned": bool(result.is_unassigned[bin_idx]),
                }
            )
    return rows


def save_em_debug_csv(result: EMAssignmentResult, path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = em_debug_rows(result)
    if not rows:
        raise ValueError("cannot export an empty EM debug table")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output


def save_em_result_npz(result: EMAssignmentResult, path: str | Path) -> Path:
    """Persist the complete compact soft assignment without pickle objects."""
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "patch_id": result.patch_id,
        "candidate_max_distance_um": result.candidate_max_distance_um,
        "converged": result.converged,
        "n_disconnected_hard_islands": result.n_disconnected_hard_islands,
        "iterations": [asdict(item) for item in result.iterations],
    }
    np.savez_compressed(
        output,
        metadata_json=np.asarray(json.dumps(metadata)),
        barcode_ids=np.asarray(result.barcode_ids, dtype=np.str_),
        barcode_index=result.barcode_index,
        barcode_xy_um=result.barcode_xy_um,
        candidate_row_splits=result.candidate_row_splits,
        candidate_cell_index=result.candidate_cell_index,
        pair_bin_index=result.pair_bin_index,
        pair_distance_um=result.pair_distance_um,
        spatial_prior=result.spatial_prior,
        expression_confidence=result.expression_confidence,
        responsibility=result.responsibility,
        background_probability=result.background_probability,
        top1_cell_index=result.top1_cell_index,
        top1_probability=result.top1_probability,
        top2_probability=result.top2_probability,
        margin=result.margin,
        normalized_entropy=result.normalized_entropy,
        is_nuclear_locked=result.is_nuclear_locked,
        locked_owner_cell_index=result.locked_owner_cell_index,
        is_ambiguous=result.is_ambiguous,
        is_unassigned=result.is_unassigned,
        cell_ids=np.asarray(result.cell_ids, dtype=np.str_),
        cell_type_posterior=result.cell_type_posterior,
        cell_expression_profile=result.cell_expression_profile,
        cell_profile_reliability=result.cell_profile_reliability,
        cell_profile_reliability_heldout_gain=(
            result.cell_profile_reliability_heldout_gain
        ),
        cell_profile_reliability_heldout_nuclear_bins=(
            result.cell_profile_reliability_heldout_nuclear_bins
        ),
        nuclear_expression_count_total=result.nuclear_expression_count_total,
        relative_profile_prior_count=np.asarray(
            result.relative_profile_prior_count,
            dtype=np.float64,
        ),
        final_type_expression_compatibility=result.final_type_expression_compatibility,
        final_cell_expression_compatibility=result.final_cell_expression_compatibility,
        final_expression_compatibility=result.final_expression_compatibility,
        final_combined_score=result.final_combined_score,
    )
    return output


def evaluate_em_assignments(
    *,
    result: EMAssignmentResult,
    truth_owner_by_barcode: Mapping[str, str],
    rl_owner_by_barcode: Mapping[str, str] | None = None,
    rl_replace_action_count: int | None = None,
    entropy_bin_edges: np.ndarray | None = None,
    boundary_barcodes: set[str] | None = None,
) -> dict[str, Any]:
    """Evaluate EM/EM+RL without allowing ground truth into inference."""
    truth = {str(key): str(value) for key, value in truth_owner_by_barcode.items()}
    predicted = {
        barcode: (
            result.cell_ids[int(result.top1_cell_index[idx])]
            if int(result.top1_cell_index[idx]) >= 0
            else "__background__"
        )
        for idx, barcode in enumerate(result.barcode_ids)
        if barcode in truth
    }
    evaluated = tuple(sorted(predicted))
    if not evaluated:
        raise ValueError("no EM barcodes overlap the supplied ground truth")
    correct = np.asarray([predicted[b] == truth[b] for b in evaluated], dtype=bool)
    index_by_barcode = result.barcode_to_index()
    entropy = np.asarray(
        [result.normalized_entropy[index_by_barcode[b]] for b in evaluated],
        dtype=np.float64,
    )
    edges = (
        np.linspace(0.0, 1.0, 6, dtype=np.float64)
        if entropy_bin_edges is None
        else np.asarray(entropy_bin_edges, dtype=np.float64)
    )
    calibration = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        include_upper = bool(np.isclose(upper, edges[-1]))
        mask = (entropy >= lower) & (
            (entropy <= upper) if include_upper else (entropy < upper)
        )
        calibration.append(
            {
                "entropy_min": float(lower),
                "entropy_max": float(upper),
                "n_bins": int(np.sum(mask)),
                "accuracy": float(np.mean(correct[mask])) if np.any(mask) else None,
            }
        )
    output: dict[str, Any] = {
        "em_top1_ownership_accuracy": float(np.mean(correct)),
        "n_evaluated_bins": int(len(evaluated)),
        "accuracy_vs_entropy_bins": calibration,
        "em_per_cell_iou": _per_cell_iou(predicted, truth, evaluated),
    }
    if boundary_barcodes is not None:
        boundary = [b for b in evaluated if b in boundary_barcodes]
        output["em_boundary_accuracy"] = (
            float(np.mean([predicted[b] == truth[b] for b in boundary]))
            if boundary
            else None
        )
    if rl_owner_by_barcode is not None:
        rl = {str(key): str(value) for key, value in rl_owner_by_barcode.items()}
        comparable = [b for b in evaluated if b in rl]
        em_ok = np.asarray([predicted[b] == truth[b] for b in comparable], dtype=bool)
        rl_ok = np.asarray([rl[b] == truth[b] for b in comparable], dtype=bool)
        modified = np.asarray([rl[b] != predicted[b] for b in comparable], dtype=bool)
        ambiguous = np.asarray(
            [result.is_ambiguous[index_by_barcode[b]] for b in comparable],
            dtype=bool,
        )
        output.update(
            {
                "em_plus_rl_top1_ownership_accuracy": float(np.mean(rl_ok))
                if comparable
                else None,
                "em_plus_rl_per_cell_iou": _per_cell_iou(rl, truth, comparable),
                "number_of_rl_replace_actions": (
                    None
                    if rl_replace_action_count is None
                    else int(rl_replace_action_count)
                ),
                "number_of_final_owner_changes": int(np.sum(modified)),
                "fraction_of_ambiguous_bins_modified": float(
                    np.sum(modified & ambiguous) / max(1, np.sum(ambiguous))
                ),
                "fraction_of_em_errors_corrected_by_rl": float(
                    np.sum((~em_ok) & rl_ok) / max(1, np.sum(~em_ok))
                ),
                "fraction_of_correct_em_assignments_damaged_by_rl": float(
                    np.sum(em_ok & (~rl_ok)) / max(1, np.sum(em_ok))
                ),
            }
        )
        if boundary_barcodes is not None:
            rl_boundary = [b for b in comparable if b in boundary_barcodes]
            output["em_plus_rl_boundary_accuracy"] = (
                float(np.mean([rl[b] == truth[b] for b in rl_boundary]))
                if rl_boundary
                else None
            )
    return output


def _per_cell_iou(
    predicted: Mapping[str, str],
    truth: Mapping[str, str],
    barcodes: Any,
) -> dict[str, float]:
    cells = sorted(
        set(predicted[b] for b in barcodes if b in predicted)
        | set(truth[b] for b in barcodes if b in truth)
    )
    output: dict[str, float] = {}
    barcode_set = set(barcodes)
    for cell_id in cells:
        pred_set = {b for b in barcode_set if predicted.get(b) == cell_id}
        truth_set = {b for b in barcode_set if truth.get(b) == cell_id}
        union = pred_set | truth_set
        output[cell_id] = float(len(pred_set & truth_set) / len(union)) if union else 1.0
    return output
