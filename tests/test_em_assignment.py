from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse

from hd_cell_rl.em_assignment import (
    apply_cell_profile_update_damping,
    assign_spatial_crossfit_folds,
    build_sparse_patch_em_input,
    compute_background_probability,
    compute_all_responsibilities,
    compute_cell_expression_compatibility,
    compute_leave_one_out_cell_expression_compatibility,
    compute_spatial_block_crossfit_cell_expression_compatibility,
    estimate_nuclear_heldout_profile_reliability,
    initialize_cell_expression_profiles,
    initialize_q_from_nuclear_seeds,
    initialize_patch_context_from_em,
    normalized_responsibility_entropy,
    responsibility_top_statistics,
    run_generalized_em,
    save_em_debug_csv,
    save_em_result_npz,
    update_all_cell_expression_profiles,
    update_all_cell_type_posteriors,
)
from hd_cell_rl.em_types import EMAssignmentConfig, SparseEMInput
from hd_cell_rl.patch_training import PatchBounds, PatchContext, TorchJointPatchEnv
from hd_cell_rl.ppo_state import EpisodeContext
from hd_cell_rl.reward import build_eight_neighbor_index, compute_expression_confidence
from preprocessing.build_spatial_patch_episodes import _make_patch_row


def _episode(
    cell_id: str,
    center: tuple[float, float],
    bin_ids: list[str],
    xy: list[tuple[float, float]],
    ll: list[list[float]],
    seed: list[int],
    counts: list[float] | None = None,
) -> EpisodeContext:
    n = len(bin_ids)
    totals = np.ones((n,), dtype=np.float32) if counts is None else np.asarray(counts, dtype=np.float32)
    confidence = compute_expression_confidence(totals, pseudocount=5.0).astype(np.float32)
    ll_arr = np.asarray(ll, dtype=np.float32)
    xy_arr = np.asarray(xy, dtype=np.float32)
    d = np.sqrt(np.sum((xy_arr - np.asarray(center, dtype=np.float32)) ** 2, axis=1))
    return EpisodeContext(
        cell_id=cell_id,
        candidate_bin_ids=tuple(bin_ids),
        initial_membership_mask=np.asarray(seed, dtype=np.uint8),
        candidate_bin_xy_um=xy_arr,
        nucleus_center_xy_um=np.asarray(center, dtype=np.float32),
        ll=ll_arr,
        p_dis=(d / 20.0).astype(np.float32),
        p_overlap=np.zeros((n,), dtype=np.float32),
        ll_mean_z=np.zeros((n,), dtype=np.float32),
        ll_max_z=np.zeros((n,), dtype=np.float32),
        base_penalty=np.zeros((n,), dtype=np.float32),
        expression_confidence=confidence,
        bin_count_totals=totals,
        neighbor_index=build_eight_neighbor_index(tuple(bin_ids), xy_arr),
        max_steps=20,
        log_prior=-np.log(2.0),
        r_max_um=20.0,
        w1=1.0,
        w2=0.0,
        w3=0.0,
        w4=0.0,
        w5=0.0,
        stop_lambda=0.0,
        stop_stat="max",
        stop_top_k=1,
        expression_confidence_pseudocount=5.0,
        normalize_expression_zscore=False,
        zscore_delta=1.0e-8,
    )


def _config(**overrides) -> EMAssignmentConfig:
    base = EMAssignmentConfig(enabled=True)
    return replace(base, **overrides)


def _simple_sparse_input(
    *,
    ll: np.ndarray,
    confidence: np.ndarray,
    distances: np.ndarray,
    locked: np.ndarray | None = None,
    locked_owner: np.ndarray | None = None,
) -> SparseEMInput:
    n_bins = int(ll.shape[0])
    n_cells = 2
    row_splits = np.arange(0, 2 * n_bins + 1, 2, dtype=np.int64)
    is_locked = np.zeros((n_bins,), dtype=bool) if locked is None else np.asarray(locked, dtype=bool)
    owner = np.full((n_bins,), -1, dtype=np.int64) if locked_owner is None else np.asarray(locked_owner, dtype=np.int64)
    return SparseEMInput(
        patch_id="synthetic",
        candidate_max_distance_um=10.0,
        barcode_ids=tuple(f"b{i}" for i in range(n_bins)),
        barcode_xy_um=np.column_stack((np.arange(n_bins), np.zeros(n_bins))).astype(np.float32),
        ll=np.asarray(ll, dtype=np.float32),
        expression_confidence=np.asarray(confidence, dtype=np.float32),
        candidate_row_splits=row_splits,
        candidate_cell_index=np.tile(np.arange(n_cells, dtype=np.int64), n_bins),
        pair_bin_index=np.repeat(np.arange(n_bins, dtype=np.int64), n_cells),
        pair_distance_um=np.asarray(distances, dtype=np.float32).reshape(-1),
        cell_ids=("A", "B"),
        is_nuclear_locked=is_locked,
        locked_owner_cell_index=owner,
        log_prior=np.full((2, ll.shape[1]), -np.log(float(ll.shape[1])), dtype=np.float64),
    )


class GeneralizedEMTests(unittest.TestCase):
    def test_cell_specific_config_defaults_preserve_legacy_mode(self) -> None:
        legacy = EMAssignmentConfig.from_mapping({"enabled": True})
        self.assertFalse(legacy.cell_specific_expression_enabled)
        self.assertEqual(legacy.cell_profile_relative_prior_strength, 1.0)
        self.assertEqual(legacy.cell_profile_update_damping, 1.0)
        self.assertEqual(legacy.cell_profile_compatibility_mode, "standard")
        self.assertEqual(legacy.spatial_crossfit_block_size_um, 8.0)
        self.assertEqual(legacy.cell_profile_reliability_mode, "nuclear_depth")
        self.assertEqual(legacy.heldout_reliability_grid_size, 101)
        configured = EMAssignmentConfig.from_mapping(
            {
                "enabled": True,
                "convergence": {"mean_profile_total_variation": 0.002},
                "cell_specific_expression": {
                    "enabled": True,
                    "relative_prior_strength": 2.5,
                    "profile_update_damping": 0.25,
                    "spatial_crossfit_block_size_um": 12.0,
                    "reliability_mode": "nuclear_heldout_predictive",
                    "heldout_reliability_grid_size": 51,
                },
            }
        )
        self.assertTrue(configured.cell_specific_expression_enabled)
        self.assertEqual(configured.cell_profile_relative_prior_strength, 2.5)
        self.assertEqual(configured.cell_profile_update_damping, 0.25)
        self.assertEqual(configured.cell_profile_compatibility_mode, "standard")
        self.assertEqual(configured.spatial_crossfit_block_size_um, 12.0)
        self.assertEqual(
            configured.cell_profile_reliability_mode,
            "nuclear_heldout_predictive",
        )
        self.assertEqual(configured.heldout_reliability_grid_size, 51)
        self.assertEqual(configured.mean_profile_total_variation, 0.002)

        for invalid in (-0.01, 1.01, np.nan):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "profile_update_damping"):
                    EMAssignmentConfig.from_mapping(
                        {
                            "enabled": True,
                            "cell_specific_expression": {
                                "profile_update_damping": invalid,
                            },
                        }
                    )

        leave_one_out = EMAssignmentConfig.from_mapping(
            {
                "enabled": True,
                "cell_specific_expression": {
                    "enabled": True,
                    "profile_update_damping": 1.0,
                    "compatibility_mode": "leave_one_out",
                },
            }
        )
        self.assertEqual(
            leave_one_out.cell_profile_compatibility_mode,
            "leave_one_out",
        )
        with self.assertRaisesRegex(ValueError, "requires profile_update_damping=1"):
            EMAssignmentConfig.from_mapping(
                {
                    "enabled": True,
                    "cell_specific_expression": {
                        "enabled": True,
                        "profile_update_damping": 0.5,
                        "compatibility_mode": "leave_one_out",
                    },
                }
            )
        with self.assertRaisesRegex(ValueError, "reliability_mode"):
            EMAssignmentConfig.from_mapping(
                {
                    "enabled": True,
                    "cell_specific_expression": {
                        "reliability_mode": "unknown",
                    },
                }
            )
        with self.assertRaisesRegex(ValueError, "heldout_reliability_grid_size"):
            EMAssignmentConfig.from_mapping(
                {
                    "enabled": True,
                    "cell_specific_expression": {
                        "heldout_reliability_grid_size": 1,
                    },
                }
            )

        spatial_crossfit = EMAssignmentConfig.from_mapping(
            {
                "enabled": True,
                "cell_specific_expression": {
                    "enabled": True,
                    "profile_update_damping": 1.0,
                    "compatibility_mode": "spatial_block_crossfit",
                    "spatial_crossfit_block_size_um": 4.0,
                },
            }
        )
        self.assertEqual(
            spatial_crossfit.cell_profile_compatibility_mode,
            "spatial_block_crossfit",
        )
        self.assertEqual(spatial_crossfit.spatial_crossfit_block_size_um, 4.0)
        for invalid in (0.0, -1.0, np.nan):
            with self.subTest(spatial_crossfit_block_size_um=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    "spatial_crossfit_block_size_um",
                ):
                    EMAssignmentConfig.from_mapping(
                        {
                            "enabled": True,
                            "cell_specific_expression": {
                                "spatial_crossfit_block_size_um": invalid,
                            },
                        }
                    )
        with self.assertRaisesRegex(ValueError, "requires profile_update_damping=1"):
            EMAssignmentConfig.from_mapping(
                {
                    "enabled": True,
                    "cell_specific_expression": {
                        "enabled": True,
                        "profile_update_damping": 0.5,
                        "compatibility_mode": "spatial_block_crossfit",
                    },
                }
            )

    def test_nuclear_heldout_reliability_prefers_reproducible_profile(self) -> None:
        em_input = SparseEMInput(
            patch_id="heldout",
            candidate_max_distance_um=10.0,
            barcode_ids=("n0", "n1"),
            barcode_xy_um=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            ll=np.zeros((2, 1), dtype=np.float32),
            expression_confidence=np.ones((2,), dtype=np.float32),
            candidate_row_splits=np.asarray([0, 1, 2], dtype=np.int64),
            candidate_cell_index=np.zeros((2,), dtype=np.int64),
            pair_bin_index=np.arange(2, dtype=np.int64),
            pair_distance_um=np.zeros((2,), dtype=np.float32),
            cell_ids=("A",),
            is_nuclear_locked=np.ones((2,), dtype=bool),
            locked_owner_cell_index=np.zeros((2,), dtype=np.int64),
            log_prior=np.zeros((1, 1), dtype=np.float64),
        )
        reliability, gain, n_bins = estimate_nuclear_heldout_profile_reliability(
            em_input=em_input,
            bin_gene_counts=sparse.csr_matrix([[10.0, 0.0], [8.0, 0.0]]),
            reference_theta=np.asarray([[0.5, 0.5]], dtype=np.float64),
            reference_prior_count=2.0,
            lambda_grid_size=101,
            epsilon=1.0e-12,
        )
        np.testing.assert_array_equal(reliability, np.asarray([1.0]))
        self.assertGreater(float(gain[0]), 0.0)
        np.testing.assert_array_equal(n_bins, np.asarray([2], dtype=np.int64))

    def test_nuclear_heldout_reliability_rejects_nonpredictive_profile(self) -> None:
        em_input = SparseEMInput(
            patch_id="heldout",
            candidate_max_distance_um=10.0,
            barcode_ids=("n0", "n1"),
            barcode_xy_um=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            ll=np.zeros((2, 1), dtype=np.float32),
            expression_confidence=np.ones((2,), dtype=np.float32),
            candidate_row_splits=np.asarray([0, 1, 2], dtype=np.int64),
            candidate_cell_index=np.zeros((2,), dtype=np.int64),
            pair_bin_index=np.arange(2, dtype=np.int64),
            pair_distance_um=np.zeros((2,), dtype=np.float32),
            cell_ids=("A",),
            is_nuclear_locked=np.ones((2,), dtype=bool),
            locked_owner_cell_index=np.zeros((2,), dtype=np.int64),
            log_prior=np.zeros((1, 1), dtype=np.float64),
        )
        reliability, gain, _n_bins = estimate_nuclear_heldout_profile_reliability(
            em_input=em_input,
            bin_gene_counts=sparse.csr_matrix([[10.0, 0.0], [0.0, 10.0]]),
            reference_theta=np.asarray([[0.5, 0.5]], dtype=np.float64),
            reference_prior_count=2.0,
            lambda_grid_size=101,
            epsilon=1.0e-12,
        )
        np.testing.assert_array_equal(reliability, np.asarray([0.0]))
        np.testing.assert_array_equal(gain, np.asarray([0.0]))

    def test_nuclear_heldout_reliability_ignores_cytoplasmic_bins_and_depth_scale(self) -> None:
        em_input = SparseEMInput(
            patch_id="heldout",
            candidate_max_distance_um=10.0,
            barcode_ids=("n0", "n1", "cytoplasm"),
            barcode_xy_um=np.asarray(
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32
            ),
            ll=np.zeros((3, 1), dtype=np.float32),
            expression_confidence=np.ones((3,), dtype=np.float32),
            candidate_row_splits=np.asarray([0, 1, 2, 3], dtype=np.int64),
            candidate_cell_index=np.zeros((3,), dtype=np.int64),
            pair_bin_index=np.arange(3, dtype=np.int64),
            pair_distance_um=np.zeros((3,), dtype=np.float32),
            cell_ids=("A",),
            is_nuclear_locked=np.asarray([True, True, False]),
            locked_owner_cell_index=np.asarray([0, 0, -1], dtype=np.int64),
            log_prior=np.zeros((1, 1), dtype=np.float64),
        )
        common = {
            "em_input": em_input,
            "reference_theta": np.asarray([[0.5, 0.5]], dtype=np.float64),
            "lambda_grid_size": 101,
            "epsilon": 1.0e-12,
        }
        first = estimate_nuclear_heldout_profile_reliability(
            **common,
            bin_gene_counts=sparse.csr_matrix(
                [[10.0, 0.0], [8.0, 0.0], [0.0, 1000.0]]
            ),
            reference_prior_count=2.0,
        )
        changed_cytoplasm = estimate_nuclear_heldout_profile_reliability(
            **common,
            bin_gene_counts=sparse.csr_matrix(
                [[10.0, 0.0], [8.0, 0.0], [1000.0, 0.0]]
            ),
            reference_prior_count=2.0,
        )
        scaled = estimate_nuclear_heldout_profile_reliability(
            **common,
            bin_gene_counts=sparse.csr_matrix(
                [[70.0, 0.0], [56.0, 0.0], [0.0, 7000.0]]
            ),
            reference_prior_count=14.0,
        )
        for observed in (changed_cytoplasm, scaled):
            np.testing.assert_array_equal(observed[0], first[0])
            np.testing.assert_allclose(observed[1], first[1], atol=1.0e-12)
            np.testing.assert_array_equal(observed[2], first[2])

    def test_spatial_crossfit_folds_use_absolute_blocks_and_exclude_locks(self) -> None:
        xy = np.asarray(
            [[0.25, 0.25], [3.75, 0.25], [4.25, 0.25], [0.5, 0.5]],
            dtype=np.float64,
        )
        locked = np.asarray([False, False, False, True])
        fold = assign_spatial_crossfit_folds(
            barcode_xy_um=xy,
            is_nuclear_locked=locked,
            block_size_um=4.0,
        )
        self.assertEqual(int(fold[0]), int(fold[1]))
        self.assertNotEqual(int(fold[0]), int(fold[2]))
        self.assertEqual(int(fold[3]), -1)

        permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
        shuffled = assign_spatial_crossfit_folds(
            barcode_xy_um=xy[permutation],
            is_nuclear_locked=locked[permutation],
            block_size_um=4.0,
        )
        np.testing.assert_array_equal(fold, shuffled[np.argsort(permutation)])

    def test_spatial_block_crossfit_excludes_whole_fold_but_keeps_nuclear(self) -> None:
        counts = sparse.csr_matrix(
            np.asarray(
                [
                    [4.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 0.0, 5.0],
                ],
                dtype=np.float64,
            )
        )
        pair_bin = np.arange(4, dtype=np.int64)
        pair_cell = np.zeros((4,), dtype=np.int64)
        pair_weight = np.asarray([0.5, 0.25, 0.75, 1.0], dtype=np.float64)
        base = np.asarray([3.0, 3.0, 2.0], dtype=np.float64)
        pseudocounts = np.asarray(
            [base + pair_weight @ counts.toarray()],
            dtype=np.float64,
        )
        fold = np.asarray([0, 0, 1, -1], dtype=np.int64)
        actual = compute_spatial_block_crossfit_cell_expression_compatibility(
            bin_gene_counts=counts,
            cell_profile_pseudocounts=pseudocounts,
            profile_pair_responsibility=pair_weight,
            pair_bin_index=pair_bin,
            candidate_cell_index=pair_cell,
            spatial_fold_index=fold,
            epsilon=1.0e-12,
        )

        dense = counts.toarray()
        expected = []
        for bin_idx in range(4):
            if fold[bin_idx] >= 0:
                held_out = np.sum(
                    dense[fold == fold[bin_idx]]
                    * pair_weight[fold == fold[bin_idx], None],
                    axis=0,
                )
            else:
                held_out = np.zeros((dense.shape[1],), dtype=np.float64)
            remaining = pseudocounts[0] - held_out
            profile = remaining / np.sum(remaining)
            expected.append(
                np.sum(dense[bin_idx] * np.log(np.maximum(profile, 1.0e-12)))
                / np.sum(dense[bin_idx])
            )
        np.testing.assert_allclose(actual, np.asarray(expected), atol=1.0e-12)
        self.assertAlmostEqual(float(np.exp(actual[3])), 7.0 / 17.5, places=12)

        leave_one_out = compute_leave_one_out_cell_expression_compatibility(
            bin_gene_counts=counts,
            cell_profile_pseudocounts=pseudocounts,
            profile_pair_responsibility=pair_weight,
            pair_bin_index=pair_bin,
            candidate_cell_index=pair_cell,
            epsilon=1.0e-12,
        )
        self.assertNotAlmostEqual(float(actual[0]), float(leave_one_out[0]))

    def test_leave_one_out_subtracts_the_bins_own_fractional_counts(self) -> None:
        counts = sparse.csr_matrix([[4.0, 0.0]])
        pseudocounts = np.asarray([[6.0, 4.0]], dtype=np.float64)
        leave_one_out = compute_leave_one_out_cell_expression_compatibility(
            bin_gene_counts=counts,
            cell_profile_pseudocounts=pseudocounts,
            profile_pair_responsibility=np.asarray([0.5], dtype=np.float64),
            pair_bin_index=np.asarray([0], dtype=np.int64),
            candidate_cell_index=np.asarray([0], dtype=np.int64),
            epsilon=1.0e-12,
        )
        standard = compute_cell_expression_compatibility(
            bin_gene_counts=counts,
            cell_expression_profile=pseudocounts,
            pair_bin_index=np.asarray([0], dtype=np.int64),
            candidate_cell_index=np.asarray([0], dtype=np.int64),
            epsilon=1.0e-12,
        )

        self.assertAlmostEqual(float(leave_one_out[0]), np.log(0.5), places=11)
        self.assertAlmostEqual(float(standard[0]), np.log(0.6), places=11)
        self.assertLess(float(leave_one_out[0]), float(standard[0]))

    def test_leave_one_out_with_zero_pair_weight_matches_standard_profile(self) -> None:
        counts = sparse.csr_matrix(
            np.asarray([[3.0, 1.0], [1.0, 3.0]], dtype=np.float64)
        )
        pseudocounts = np.asarray(
            [[8.0, 2.0], [2.0, 8.0]],
            dtype=np.float64,
        )
        pair_bin = np.asarray([0, 0, 1, 1], dtype=np.int64)
        pair_cell = np.asarray([0, 1, 0, 1], dtype=np.int64)
        standard = compute_cell_expression_compatibility(
            bin_gene_counts=counts,
            cell_expression_profile=pseudocounts,
            pair_bin_index=pair_bin,
            candidate_cell_index=pair_cell,
            epsilon=1.0e-12,
        )
        leave_one_out = compute_leave_one_out_cell_expression_compatibility(
            bin_gene_counts=counts,
            cell_profile_pseudocounts=pseudocounts,
            profile_pair_responsibility=np.zeros((4,), dtype=np.float64),
            pair_bin_index=pair_bin,
            candidate_cell_index=pair_cell,
            epsilon=1.0e-12,
        )
        np.testing.assert_allclose(
            leave_one_out,
            standard,
            rtol=0.0,
            atol=1.0e-10,
        )

    def test_sparse_leave_one_out_matches_naive_pair_calculation(self) -> None:
        counts = sparse.csr_matrix(
            np.asarray(
                [
                    [4.0, 0.0, 1.0, 0.0],
                    [0.0, 2.0, 0.0, 3.0],
                    [1.0, 1.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        )
        pair_bin = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        pair_cell = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
        pair_weight = np.asarray(
            [0.7, 0.3, 0.2, 0.8, 0.6, 0.4],
            dtype=np.float64,
        )
        pseudocounts = np.asarray(
            [
                [8.0, 4.0, 2.0, 3.0],
                [5.0, 6.0, 2.0, 7.0],
            ],
            dtype=np.float64,
        )
        actual = compute_leave_one_out_cell_expression_compatibility(
            bin_gene_counts=counts,
            cell_profile_pseudocounts=pseudocounts,
            profile_pair_responsibility=pair_weight,
            pair_bin_index=pair_bin,
            candidate_cell_index=pair_cell,
            epsilon=1.0e-12,
        )
        dense_counts = counts.toarray()
        expected = []
        for bin_idx, cell_idx, weight in zip(
            pair_bin,
            pair_cell,
            pair_weight,
            strict=True,
        ):
            remaining = (
                pseudocounts[int(cell_idx)]
                - float(weight) * dense_counts[int(bin_idx)]
            )
            profile = remaining / np.sum(remaining)
            total = np.sum(dense_counts[int(bin_idx)])
            expected.append(
                np.sum(
                    dense_counts[int(bin_idx)]
                    * np.log(np.maximum(profile, 1.0e-12))
                )
                / total
            )
        np.testing.assert_allclose(
            actual,
            np.asarray(expected),
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_cell_profile_update_damping_has_exact_endpoints(self) -> None:
        old = np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)
        raw = np.asarray([[0.2, 0.8], [0.7, 0.3]], dtype=np.float64)

        np.testing.assert_array_equal(
            apply_cell_profile_update_damping(
                old_profile=old,
                raw_profile=raw,
                damping=0.0,
            ),
            old,
        )
        np.testing.assert_array_equal(
            apply_cell_profile_update_damping(
                old_profile=old,
                raw_profile=raw,
                damping=1.0,
            ),
            raw,
        )
        np.testing.assert_allclose(
            apply_cell_profile_update_damping(
                old_profile=old,
                raw_profile=raw,
                damping=0.25,
            ),
            0.75 * old + 0.25 * raw,
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_zero_profile_update_damping_freezes_nuclear_initial_profile(self) -> None:
        em_input = _simple_sparse_input(
            ll=np.asarray(
                [[4.0, -4.0], [-4.0, 4.0], [0.0, 0.0]], dtype=np.float32
            ),
            confidence=np.ones((3,), dtype=np.float32),
            distances=np.asarray(
                [[0.0, 4.0], [4.0, 0.0], [2.0, 2.0]], dtype=np.float32
            ),
            locked=np.asarray([True, True, False]),
            locked_owner=np.asarray([0, 1, -1]),
        )
        counts = sparse.csr_matrix(
            np.asarray([[20.0, 0.0], [0.0, 20.0], [9.0, 1.0]])
        )
        theta = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
        initial_q = initialize_q_from_nuclear_seeds(em_input)
        initial_profile = initialize_cell_expression_profiles(
            em_input=em_input,
            bin_gene_counts=counts,
            reference_theta=theta,
            cell_type_posterior=initial_q,
            relative_prior_strength=1.0,
            epsilon=1.0e-12,
        )[0]
        result = run_generalized_em(
            em_input,
            _config(
                cell_specific_expression_enabled=True,
                cell_profile_relative_prior_strength=1.0,
                cell_profile_update_damping=0.0,
                expression_weight=2.0,
                damping=1.0,
                max_iterations=4,
            ),
            bin_gene_counts=counts,
            reference_theta=theta,
        )

        np.testing.assert_array_equal(
            result.cell_expression_profile,
            initial_profile.astype(np.float32),
        )
        self.assertTrue(
            all(item.mean_profile_total_variation == 0.0 for item in result.iterations)
        )
        np.testing.assert_array_equal(
            result.responsibility[:2], np.asarray([1.0, 0.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            result.responsibility[2:4], np.asarray([0.0, 1.0], dtype=np.float32)
        )
        np.testing.assert_allclose(
            np.add.reduceat(
                result.responsibility,
                result.candidate_row_splits[:-1],
            ),
            1.0,
            rtol=0.0,
            atol=1.0e-6,
        )

    def test_relative_profile_shrinkage_is_depth_scale_invariant(self) -> None:
        em_input = _simple_sparse_input(
            ll=np.asarray([[3.0, -3.0], [-3.0, 3.0]], dtype=np.float32),
            confidence=np.ones((2,), dtype=np.float32),
            distances=np.asarray([[0.0, 2.0], [2.0, 0.0]], dtype=np.float32),
            locked=np.asarray([True, True]),
            locked_owner=np.asarray([0, 1]),
        )
        counts = sparse.csr_matrix(
            np.asarray([[2.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
        )
        theta = np.asarray(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], dtype=np.float64
        )
        q = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        profile, reliability, total, prior_count = initialize_cell_expression_profiles(
            em_input=em_input,
            bin_gene_counts=counts,
            reference_theta=theta,
            cell_type_posterior=q,
            relative_prior_strength=1.0,
            epsilon=1.0e-12,
        )
        scaled = initialize_cell_expression_profiles(
            em_input=em_input,
            bin_gene_counts=counts * 7.0,
            reference_theta=theta,
            cell_type_posterior=q,
            relative_prior_strength=1.0,
            epsilon=1.0e-12,
        )
        np.testing.assert_allclose(profile, scaled[0], atol=1.0e-12)
        np.testing.assert_allclose(reliability, scaled[1], atol=1.0e-12)
        np.testing.assert_allclose(scaled[2], total * 7.0, atol=1.0e-12)
        self.assertAlmostEqual(scaled[3], prior_count * 7.0)
        self.assertLess(float(reliability[0]), float(reliability[1]))

    def test_cell_profile_m_step_uses_fractional_responsibility(self) -> None:
        profile = update_all_cell_expression_profiles(
            responsibilities=np.asarray([0.8, 0.2], dtype=np.float64),
            bin_gene_counts=sparse.csr_matrix([[10.0, 0.0]]),
            pair_bin_index=np.asarray([0, 0], dtype=np.int64),
            candidate_cell_index=np.asarray([0, 1], dtype=np.int64),
            cell_type_posterior=np.ones((2, 1), dtype=np.float64),
            reference_theta=np.asarray([[0.5, 0.5]], dtype=np.float64),
            reference_prior_count=10.0,
            epsilon=1.0e-12,
        )
        np.testing.assert_allclose(
            profile,
            np.asarray([[13.0 / 18.0, 5.0 / 18.0], [7.0 / 12.0, 5.0 / 12.0]]),
            atol=1.0e-12,
        )

    def test_alternating_profile_requires_raw_counts_and_theta(self) -> None:
        em_input = _simple_sparse_input(
            ll=np.zeros((1, 2), dtype=np.float32),
            confidence=np.ones((1,), dtype=np.float32),
            distances=np.asarray([[1.0, 1.0]], dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, "bin_gene_counts and reference_theta"):
            run_generalized_em(
                em_input,
                _config(cell_specific_expression_enabled=True),
            )

    def test_alternating_profile_is_batch_order_independent_and_keeps_locks(self) -> None:
        ll = np.asarray(
            [[4.0, -4.0], [-4.0, 4.0], [0.0, 0.0]], dtype=np.float32
        )
        distances = np.asarray(
            [[0.0, 4.0], [4.0, 0.0], [2.0, 2.0]], dtype=np.float32
        )
        locked = np.asarray([True, True, False])
        owner = np.asarray([0, 1, -1])
        counts = np.asarray(
            [[20.0, 0.0], [0.0, 20.0], [2.0, 8.0]], dtype=np.float64
        )
        theta = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
        config = _config(
            cell_specific_expression_enabled=True,
            cell_profile_relative_prior_strength=1.0,
            expression_weight=2.0,
            damping=1.0,
            max_iterations=3,
        )
        first = _simple_sparse_input(
            ll=ll,
            confidence=np.ones((3,), dtype=np.float32),
            distances=distances,
            locked=locked,
            locked_owner=owner,
        )
        permutation = np.asarray([2, 0, 1], dtype=np.int64)
        second = _simple_sparse_input(
            ll=ll[permutation],
            confidence=np.ones((3,), dtype=np.float32),
            distances=distances[permutation],
            locked=locked[permutation],
            locked_owner=owner[permutation],
        )
        result_a = run_generalized_em(
            first,
            config,
            bin_gene_counts=sparse.csr_matrix(counts),
            reference_theta=theta,
        )
        result_b = run_generalized_em(
            second,
            config,
            bin_gene_counts=sparse.csr_matrix(counts[permutation]),
            reference_theta=theta,
        )
        inverse = np.argsort(permutation)
        np.testing.assert_allclose(
            result_a.responsibility.reshape(3, 2),
            result_b.responsibility.reshape(3, 2)[inverse],
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            result_a.cell_type_posterior,
            result_b.cell_type_posterior,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            result_a.cell_expression_profile,
            result_b.cell_expression_profile,
            atol=1.0e-6,
        )
        np.testing.assert_array_equal(
            result_a.responsibility[:2], np.asarray([1.0, 0.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            result_a.responsibility[2:4], np.asarray([0.0, 1.0], dtype=np.float32)
        )
        np.testing.assert_allclose(
            np.sum(result_a.cell_expression_profile, axis=1),
            1.0,
            atol=1.0e-6,
        )
        self.assertTrue(
            all(
                np.isfinite(item.mean_profile_total_variation)
                for item in result_a.iterations
            )
        )

    def test_leave_one_out_em_is_batch_order_independent_and_keeps_locks(self) -> None:
        ll = np.asarray(
            [[4.0, -4.0], [-4.0, 4.0], [0.0, 0.0]], dtype=np.float32
        )
        distances = np.asarray(
            [[0.0, 4.0], [4.0, 0.0], [2.0, 2.0]], dtype=np.float32
        )
        locked = np.asarray([True, True, False])
        owner = np.asarray([0, 1, -1])
        counts = np.asarray(
            [[20.0, 0.0], [0.0, 20.0], [2.0, 8.0]], dtype=np.float64
        )
        theta = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
        config = _config(
            cell_specific_expression_enabled=True,
            cell_profile_relative_prior_strength=1.0,
            cell_profile_update_damping=1.0,
            cell_profile_compatibility_mode="leave_one_out",
            expression_weight=2.0,
            damping=1.0,
            max_iterations=3,
        )
        first = _simple_sparse_input(
            ll=ll,
            confidence=np.ones((3,), dtype=np.float32),
            distances=distances,
            locked=locked,
            locked_owner=owner,
        )
        permutation = np.asarray([2, 0, 1], dtype=np.int64)
        second = _simple_sparse_input(
            ll=ll[permutation],
            confidence=np.ones((3,), dtype=np.float32),
            distances=distances[permutation],
            locked=locked[permutation],
            locked_owner=owner[permutation],
        )
        result_a = run_generalized_em(
            first,
            config,
            bin_gene_counts=sparse.csr_matrix(counts),
            reference_theta=theta,
        )
        result_b = run_generalized_em(
            second,
            config,
            bin_gene_counts=sparse.csr_matrix(counts[permutation]),
            reference_theta=theta,
        )
        inverse = np.argsort(permutation)
        np.testing.assert_allclose(
            result_a.responsibility.reshape(3, 2),
            result_b.responsibility.reshape(3, 2)[inverse],
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            result_a.cell_type_posterior,
            result_b.cell_type_posterior,
            atol=1.0e-6,
        )
        np.testing.assert_array_equal(
            result_a.responsibility[:2],
            np.asarray([1.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            result_a.responsibility[2:4],
            np.asarray([0.0, 1.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            np.add.reduceat(
                result_a.responsibility,
                result_a.candidate_row_splits[:-1],
            ),
            1.0,
            rtol=0.0,
            atol=1.0e-6,
        )

    def test_spatial_crossfit_em_is_order_independent_and_keeps_locks(self) -> None:
        ll = np.asarray(
            [
                [4.0, -4.0],
                [-4.0, 4.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )
        distances = np.asarray(
            [[0.0, 4.0], [4.0, 0.0], [2.0, 2.0], [2.2, 1.8]],
            dtype=np.float32,
        )
        locked = np.asarray([True, True, False, False])
        owner = np.asarray([0, 1, -1, -1])
        xy = np.asarray(
            [[0.0, 0.0], [4.0, 0.0], [0.5, 0.5], [1.5, 0.5]],
            dtype=np.float32,
        )
        counts = np.asarray(
            [[20.0, 0.0], [0.0, 20.0], [9.0, 1.0], [8.0, 2.0]],
            dtype=np.float64,
        )
        theta = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
        config = _config(
            cell_specific_expression_enabled=True,
            cell_profile_relative_prior_strength=1.0,
            cell_profile_update_damping=1.0,
            cell_profile_compatibility_mode="spatial_block_crossfit",
            spatial_crossfit_block_size_um=4.0,
            expression_weight=2.0,
            damping=1.0,
            max_iterations=3,
        )
        first = replace(
            _simple_sparse_input(
                ll=ll,
                confidence=np.ones((4,), dtype=np.float32),
                distances=distances,
                locked=locked,
                locked_owner=owner,
            ),
            barcode_xy_um=xy,
        )
        permutation = np.asarray([3, 1, 2, 0], dtype=np.int64)
        second = _simple_sparse_input(
            ll=ll[permutation],
            confidence=np.ones((4,), dtype=np.float32),
            distances=distances[permutation],
            locked=locked[permutation],
            locked_owner=owner[permutation],
        )
        second = replace(
            second,
            barcode_ids=tuple(first.barcode_ids[idx] for idx in permutation),
            barcode_xy_um=xy[permutation],
        )
        result_a = run_generalized_em(
            first,
            config,
            bin_gene_counts=sparse.csr_matrix(counts),
            reference_theta=theta,
        )
        result_b = run_generalized_em(
            second,
            config,
            bin_gene_counts=sparse.csr_matrix(counts[permutation]),
            reference_theta=theta,
        )
        inverse = np.argsort(permutation)
        np.testing.assert_allclose(
            result_a.responsibility.reshape(4, 2),
            result_b.responsibility.reshape(4, 2)[inverse],
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            result_a.cell_type_posterior,
            result_b.cell_type_posterior,
            atol=1.0e-6,
        )
        np.testing.assert_array_equal(
            result_a.responsibility[:2],
            np.asarray([1.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            result_a.responsibility[2:4],
            np.asarray([0.0, 1.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            np.add.reduceat(
                result_a.responsibility,
                result_a.candidate_row_splits[:-1],
            ),
            1.0,
            rtol=0.0,
            atol=1.0e-6,
        )

    def test_first_spatial_crossfit_e_step_matches_standard_profile(self) -> None:
        em_input = _simple_sparse_input(
            ll=np.asarray(
                [[4.0, -4.0], [-4.0, 4.0], [0.0, 0.0]],
                dtype=np.float32,
            ),
            confidence=np.ones((3,), dtype=np.float32),
            distances=np.asarray(
                [[0.0, 4.0], [4.0, 0.0], [2.0, 2.0]],
                dtype=np.float32,
            ),
            locked=np.asarray([True, True, False]),
            locked_owner=np.asarray([0, 1, -1]),
        )
        counts = sparse.csr_matrix(
            np.asarray([[20.0, 0.0], [0.0, 20.0], [8.0, 2.0]])
        )
        theta = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
        common = dict(
            cell_specific_expression_enabled=True,
            cell_profile_relative_prior_strength=1.0,
            cell_profile_update_damping=1.0,
            expression_weight=2.0,
            damping=1.0,
            max_iterations=1,
        )
        standard = run_generalized_em(
            em_input,
            _config(**common),
            bin_gene_counts=counts,
            reference_theta=theta,
        )
        crossfit = run_generalized_em(
            em_input,
            _config(
                **common,
                cell_profile_compatibility_mode="spatial_block_crossfit",
                spatial_crossfit_block_size_um=8.0,
            ),
            bin_gene_counts=counts,
            reference_theta=theta,
        )
        np.testing.assert_array_equal(
            crossfit.responsibility,
            standard.responsibility,
        )

    def test_disabled_cell_profile_is_exactly_legacy_em(self) -> None:
        em_input = _simple_sparse_input(
            ll=np.asarray([[2.0, -2.0], [-2.0, 2.0]], dtype=np.float32),
            confidence=np.asarray([0.7, 1.0], dtype=np.float32),
            distances=np.asarray([[1.0, 3.0], [3.0, 1.0]], dtype=np.float32),
        )
        config = _config(
            cell_specific_expression_enabled=False,
            damping=0.5,
            max_iterations=4,
        )
        legacy = run_generalized_em(em_input, config)
        extra_inputs = run_generalized_em(
            em_input,
            config,
            bin_gene_counts=sparse.csr_matrix([[4.0, 0.0], [0.0, 4.0]]),
            reference_theta=np.asarray([[0.9, 0.1], [0.1, 0.9]]),
        )
        np.testing.assert_array_equal(
            legacy.responsibility, extra_inputs.responsibility
        )
        np.testing.assert_array_equal(
            legacy.cell_type_posterior, extra_inputs.cell_type_posterior
        )
        self.assertEqual(legacy.cell_expression_profile.shape, (2, 0))

    def test_confidence_gated_background_and_nuclear_lock(self) -> None:
        em_input = _simple_sparse_input(
            ll=np.zeros((3, 2), dtype=np.float32),
            confidence=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            distances=np.asarray(
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                dtype=np.float32,
            ),
            locked=np.asarray([False, False, True]),
            locked_owner=np.asarray([-1, -1, 0]),
        )
        result = run_generalized_em(
            em_input,
            _config(
                expression_weight=0.0,
                damping=1.0,
                background_enabled=True,
                background_owned_logit_intercept=-6.0,
                background_confidence_weight=12.0,
            ),
        )
        self.assertTrue(bool(result.is_unassigned[0]))
        self.assertFalse(bool(result.is_unassigned[1]))
        self.assertEqual(float(result.background_probability[2]), 0.0)
        np.testing.assert_array_equal(
            result.responsibility[4:6],
            np.asarray([1.0, 0.0], dtype=np.float32),
        )
        row_sum = np.add.reduceat(
            result.responsibility,
            result.candidate_row_splits[:-1],
        )
        np.testing.assert_allclose(
            row_sum + result.background_probability,
            1.0,
            rtol=0.0,
            atol=1.0e-6,
        )

    def test_background_disabled_is_exact_zero(self) -> None:
        probability = compute_background_probability(
            expression_confidence=np.asarray([0.0, 0.5, 1.0]),
            is_nuclear_locked=np.asarray([False, False, True]),
            enabled=False,
            owned_logit_intercept=-6.0,
            expression_confidence_weight=12.0,
        )
        np.testing.assert_array_equal(probability, np.zeros((3,), dtype=np.float64))

    def test_fixed_oracle_q_is_not_changed_by_m_step(self) -> None:
        em_input = _simple_sparse_input(
            ll=np.asarray([[5.0, -5.0], [-5.0, 5.0]], dtype=np.float32),
            confidence=np.ones((2,), dtype=np.float32),
            distances=np.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32),
        )
        oracle_q = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        result = run_generalized_em(
            em_input,
            _config(damping=1.0, max_iterations=3),
            initial_cell_type_posterior=oracle_q,
            freeze_cell_type_posterior=True,
        )
        np.testing.assert_array_equal(result.cell_type_posterior, oracle_q.astype(np.float32))
        self.assertTrue(all(item.mean_q_change == 0.0 for item in result.iterations))

    def test_fixed_pair_compatibility_overrides_type_level_q_ll(self) -> None:
        em_input = _simple_sparse_input(
            ll=np.asarray([[4.0, -4.0]], dtype=np.float32),
            confidence=np.ones((1,), dtype=np.float32),
            distances=np.asarray([[1.0, 1.0]], dtype=np.float32),
        )
        result = run_generalized_em(
            em_input,
            _config(expression_weight=2.0, damping=1.0, max_iterations=2),
            fixed_pair_expression_compatibility=np.asarray([-5.0, 5.0]),
        )
        self.assertEqual(int(result.top1_cell_index[0]), 1)
        np.testing.assert_allclose(
            result.final_expression_compatibility,
            np.asarray([-5.0, 5.0], dtype=np.float32),
        )

    def test_positive_expression_filter_keeps_zero_count_nuclear_lock(self) -> None:
        cell = _episode(
            "A",
            (0.0, 0.0),
            ["zero_seed", "zero_non_nuclear", "positive_non_nuclear"],
            [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
            [1, 0, 0],
            [0.0, 0.0, 2.0],
        )
        patch = PatchContext(
            patch_id="positive_expression_filter",
            cells=(cell,),
            core_cell_ids=("A",),
            margin_cell_ids=(),
            outer_bounds=PatchBounds(-1, 3, -1, 1),
            core_bounds=PatchBounds(-1, 3, -1, 1),
            max_steps=5,
        )
        em_input = build_sparse_patch_em_input(
            context=patch,
            candidate_max_distance_um=5.0,
            non_nuclear_bin_filter="positive_expression",
        )
        self.assertEqual(
            em_input.barcode_ids,
            ("positive_non_nuclear", "zero_seed"),
        )
        seed_idx = em_input.barcode_ids.index("zero_seed")
        self.assertTrue(bool(em_input.is_nuclear_locked[seed_idx]))

    def test_patch_index_context_expands_outer_bounds_by_candidate_maxdis(self) -> None:
        cell_ids = np.asarray(["core", "outside_outer", "too_far"], dtype=object)
        xy = np.asarray([[0.0, 0.0], [14.0, 0.0], [31.0, 0.0]], dtype=np.float64)
        row = _make_patch_row(
            cx=0.0,
            cy=0.0,
            source="test",
            cell_ids=cell_ids,
            xy=xy,
            candidate_indices=np.arange(3, dtype=np.int64),
            patch_size=20.0,
            core_size=10.0,
            min_core_cells=1,
            min_patch_cells=1,
            max_patch_cells=0,
            candidate_max_distance_um=5.0,
            patch_index=0,
        )
        self.assertIsNotNone(row)
        self.assertIn("outside_outer", row["patch_cell_ids"])
        self.assertNotIn("too_far", row["patch_cell_ids"])
        self.assertEqual(float(row["context_x_max"]), 15.0)
        self.assertEqual(float(row["candidate_max_distance_um"]), 5.0)

    def test_maxdis_includes_all_cells_without_nearest_three_and_excludes_outside(self) -> None:
        cells = []
        for idx, center_x in enumerate((0.0, 1.0, 2.0, 3.0, 4.0, 11.0)):
            ids = [f"seed_{idx}"]
            xy = [(center_x, 0.0)]
            ll = [[1.0, 0.0]]
            seed = [1]
            if idx == 0:
                ids.append("shared")
                xy.append((0.0, 0.0))
                ll.append([0.0, 0.0])
                seed.append(0)
            cells.append(_episode(str(idx), (center_x, 0.0), ids, xy, ll, seed))
        patch = PatchContext(
            patch_id="maxdis",
            cells=tuple(cells),
            core_cell_ids=("0",),
            margin_cell_ids=("1", "2", "3", "4", "5"),
            outer_bounds=PatchBounds(-1, 1, -1, 1),
            core_bounds=PatchBounds(-1, 1, -1, 1),
            max_steps=10,
        )
        em_input = build_sparse_patch_em_input(
            context=patch,
            candidate_max_distance_um=10.0,
        )
        shared = em_input.barcode_ids.index("shared")
        start = int(em_input.candidate_row_splits[shared])
        end = int(em_input.candidate_row_splits[shared + 1])
        candidates = em_input.candidate_cell_index[start:end].tolist()
        self.assertEqual(candidates, [0, 1, 2, 3, 4])
        self.assertGreater(len(candidates), 3)
        self.assertNotIn(5, candidates)

    def test_margin_cell_can_beat_core_cell(self) -> None:
        core = _episode(
            "core",
            (0.0, 0.0),
            ["core_seed", "shared"],
            [(0.0, 0.0), (1.0, 0.0)],
            [[5.0, -5.0], [-5.0, 5.0]],
            [1, 0],
            [100.0, 100.0],
        )
        margin = _episode(
            "margin",
            (3.0, 0.0),
            ["margin_seed", "shared"],
            [(3.0, 0.0), (1.0, 0.0)],
            [[-5.0, 5.0], [-5.0, 5.0]],
            [1, 0],
            [100.0, 100.0],
        )
        patch = PatchContext(
            patch_id="margin",
            cells=(core, margin),
            core_cell_ids=("core",),
            margin_cell_ids=("margin",),
            outer_bounds=PatchBounds(-1, 2, -1, 1),
            core_bounds=PatchBounds(-1, 2, -1, 1),
            max_steps=10,
        )
        result = run_generalized_em(
            build_sparse_patch_em_input(context=patch, candidate_max_distance_um=5.0),
            _config(expression_weight=5.0, damping=1.0),
        )
        idx = result.barcode_ids.index("shared")
        self.assertEqual(result.cell_ids[int(result.top1_cell_index[idx])], "margin")

    def test_nuclear_lock_remains_exact_one_hot(self) -> None:
        em_input = _simple_sparse_input(
            ll=np.asarray([[3.0, -3.0], [0.0, 0.0]], dtype=np.float32),
            confidence=np.ones((2,), dtype=np.float32),
            distances=np.asarray([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
            locked=np.asarray([True, False]),
            locked_owner=np.asarray([0, -1]),
        )
        result = run_generalized_em(em_input, _config(damping=0.5, max_iterations=5))
        np.testing.assert_array_equal(result.responsibility[:2], np.asarray([1.0, 0.0], dtype=np.float32))

    def test_responsibility_rows_are_normalized(self) -> None:
        em_input = _simple_sparse_input(
            ll=np.asarray([[3.0, -3.0], [0.0, 0.0]], dtype=np.float32),
            confidence=np.ones((2,), dtype=np.float32),
            distances=np.asarray([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        )
        result = run_generalized_em(em_input, _config(damping=0.5, max_iterations=5))
        row_sums = np.add.reduceat(result.responsibility, result.candidate_row_splits[:-1])
        np.testing.assert_allclose(row_sums, 1.0, rtol=0.0, atol=1.0e-6)

    def test_spatial_only_responsibilities_equal_spatial_prior(self) -> None:
        em_input = _simple_sparse_input(
            ll=np.asarray([[10.0, -10.0]], dtype=np.float32),
            confidence=np.ones((1,), dtype=np.float32),
            distances=np.asarray([[1.0, 8.0]], dtype=np.float32),
        )
        result = run_generalized_em(
            em_input,
            _config(expression_weight=0.0, damping=1.0),
        )
        np.testing.assert_allclose(result.responsibility, result.spatial_prior, atol=1.0e-6)

    def test_expression_can_override_distance(self) -> None:
        ll = np.asarray([[5.0, -5.0], [-5.0, 5.0], [-5.0, 5.0]], dtype=np.float32)
        confidence = np.ones((3,), dtype=np.float32)
        distances = np.asarray([[0.0, 4.0], [4.0, 0.0], [1.0, 4.0]], dtype=np.float32)
        em_input = _simple_sparse_input(
            ll=ll,
            confidence=confidence,
            distances=distances,
            locked=np.asarray([True, True, False]),
            locked_owner=np.asarray([0, 1, -1]),
        )
        result = run_generalized_em(
            em_input,
            _config(expression_weight=3.0, damping=1.0),
        )
        self.assertEqual(int(result.top1_cell_index[2]), 1)

    def test_expression_confidence_weakens_low_count_evidence(self) -> None:
        counts = np.asarray([1.0, 100.0], dtype=np.float64)
        confidence = compute_expression_confidence(counts, pseudocount=5.0)
        q = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        ll = np.asarray([[3.0, -3.0], [3.0, -3.0]], dtype=np.float64)
        splits = np.asarray([0, 2, 4], dtype=np.int64)
        pair_bin = np.asarray([0, 0, 1, 1], dtype=np.int64)
        pair_cell = np.asarray([0, 1, 0, 1], dtype=np.int64)
        responsibilities, _, _ = compute_all_responsibilities(
            q=q,
            ll=ll,
            expression_confidence=confidence,
            spatial_prior=np.full((4,), 0.5, dtype=np.float64),
            candidate_row_splits=splits,
            pair_bin_index=pair_bin,
            candidate_cell_index=pair_cell,
            spatial_weight=1.0,
            expression_weight=1.0,
            epsilon=1.0e-12,
        )
        self.assertGreater(responsibilities[2] - 0.5, responsibilities[0] - 0.5)

    def test_m_step_uses_fractional_responsibility(self) -> None:
        q = update_all_cell_type_posteriors(
            responsibilities=np.asarray([0.8, 0.2], dtype=np.float64),
            ll=np.asarray([[2.0, -2.0]], dtype=np.float64),
            expression_confidence=np.asarray([1.0], dtype=np.float64),
            pair_bin_index=np.asarray([0, 0], dtype=np.int64),
            candidate_cell_index=np.asarray([0, 1], dtype=np.int64),
            log_prior=np.zeros((2, 2), dtype=np.float64),
        )
        expected_a = np.exp([1.6, -1.6]) / np.sum(np.exp([1.6, -1.6]))
        expected_b = np.exp([0.4, -0.4]) / np.sum(np.exp([0.4, -0.4]))
        np.testing.assert_allclose(q[0], expected_a, atol=1.0e-7)
        np.testing.assert_allclose(q[1], expected_b, atol=1.0e-7)

    def test_batch_iteration_is_order_independent(self) -> None:
        ll = np.asarray([[3.0, -3.0], [-3.0, 3.0], [1.0, -1.0]], dtype=np.float32)
        distances = np.asarray([[0.0, 4.0], [4.0, 0.0], [1.0, 2.0]], dtype=np.float32)
        locked = np.asarray([True, True, False])
        owner = np.asarray([0, 1, -1])
        first = _simple_sparse_input(
            ll=ll,
            confidence=np.ones(3),
            distances=distances,
            locked=locked,
            locked_owner=owner,
        )
        permutation = np.asarray([2, 0, 1], dtype=np.int64)
        second = _simple_sparse_input(
            ll=ll[permutation],
            confidence=np.ones(3),
            distances=distances[permutation],
            locked=locked[permutation],
            locked_owner=owner[permutation],
        )
        result_a = run_generalized_em(first, _config(damping=1.0, max_iterations=1))
        result_b = run_generalized_em(second, _config(damping=1.0, max_iterations=1))
        inverse = np.argsort(permutation)
        np.testing.assert_allclose(
            result_a.responsibility.reshape(3, 2),
            result_b.responsibility.reshape(3, 2)[inverse],
            atol=1.0e-6,
        )
        np.testing.assert_allclose(result_a.cell_type_posterior, result_b.cell_type_posterior, atol=1.0e-6)

    def test_normalized_entropy_and_entropy_only_gate(self) -> None:
        rows = np.asarray([0.99, 0.01, 0.5, 0.5, 1 / 3, 1 / 3, 1 / 3, 1.0])
        splits = np.asarray([0, 2, 4, 7, 8], dtype=np.int64)
        entropy = normalized_responsibility_entropy(rows, splits)
        self.assertLess(float(entropy[0]), 0.1)
        self.assertAlmostEqual(float(entropy[1]), 1.0, places=6)
        self.assertAlmostEqual(float(entropy[2]), 1.0, places=6)
        self.assertEqual(float(entropy[3]), 0.0)

        top_cell, top1, top2, margin = responsibility_top_statistics(
            responsibilities=rows,
            candidate_row_splits=splits,
            candidate_cell_index=np.asarray([0, 1, 0, 1, 0, 1, 2, 0], dtype=np.int64),
        )
        self.assertEqual(top_cell.shape, (4,))
        self.assertAlmostEqual(float(top1[0]), 0.99)
        self.assertAlmostEqual(float(top2[0]), 0.01)
        self.assertAlmostEqual(float(margin[0]), 0.98)
        gate = entropy > 0.5
        np.testing.assert_array_equal(gate, np.asarray([False, True, True, False]))

    def test_ambiguity_gate_does_not_or_top1_or_margin(self) -> None:
        em_input = _simple_sparse_input(
            ll=np.zeros((1, 2), dtype=np.float32),
            confidence=np.ones((1,), dtype=np.float32),
            distances=np.asarray([[1.0, 1.0]], dtype=np.float32),
        )
        result = run_generalized_em(
            em_input,
            _config(
                expression_weight=0.0,
                damping=1.0,
                entropy_threshold=1.0,
            ),
        )
        self.assertAlmostEqual(float(result.top1_probability[0]), 0.5, places=6)
        self.assertAlmostEqual(float(result.margin[0]), 0.0, places=6)
        self.assertAlmostEqual(float(result.normalized_entropy[0]), 1.0, places=6)
        self.assertFalse(bool(result.is_ambiguous[0]))

    def test_convergence_detected(self) -> None:
        em_input = _simple_sparse_input(
            ll=np.zeros((2, 2), dtype=np.float32),
            confidence=np.ones(2),
            distances=np.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32),
        )
        result = run_generalized_em(
            em_input,
            _config(expression_weight=0.0, damping=1.0, max_iterations=15),
        )
        self.assertTrue(result.converged)
        self.assertLess(len(result.iterations), 15)

    def test_sparse_result_and_debug_table_are_serializable(self) -> None:
        em_input = _simple_sparse_input(
            ll=np.zeros((1, 2), dtype=np.float32),
            confidence=np.ones(1),
            distances=np.asarray([[1.0, 2.0]], dtype=np.float32),
        )
        result = run_generalized_em(em_input, _config(expression_weight=0.0))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = save_em_result_npz(result, root / "assignment.npz")
            debug = save_em_debug_csv(result, root / "debug.csv")
            with np.load(artifact, allow_pickle=False) as data:
                self.assertIn("candidate_row_splits", data.files)
                self.assertIn("responsibility", data.files)
                self.assertEqual(data["candidate_cell_index"].dtype, np.int64)
            self.assertTrue(debug.exists())

    def test_rl_initialization_and_ambiguous_only_replace_mask(self) -> None:
        import torch

        ids = ["a_seed", "fixed", "shared", "b_seed"]
        xy = [(0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (4.0, 0.0)]
        ll = [[6.0, -6.0], [6.0, -6.0], [0.0, 0.0], [-6.0, 6.0]]
        a = _episode("A", (0.0, 0.0), ids, xy, ll, [1, 0, 0, 0], [100] * 4)
        b = _episode("B", (4.0, 0.0), ids, xy, ll, [0, 0, 0, 1], [100] * 4)
        patch = PatchContext(
            patch_id="rl_em",
            cells=(a, b),
            core_cell_ids=("A", "B"),
            margin_cell_ids=(),
            outer_bounds=PatchBounds(-1, 5, -1, 3),
            core_bounds=PatchBounds(-1, 5, -1, 3),
            max_steps=10,
            force_fill_expression_bins=True,
            force_fill_target_barcodes=tuple(ids),
            agent_mode="multi_cell_joint_global_delta",
            after_fill_actions="replace_only",
        )
        initialized = initialize_patch_context_from_em(
            context=patch,
            config=_config(
                expression_weight=3.0,
                damping=1.0,
                entropy_threshold=0.5,
            ),
            candidate_max_distance_um=5.0,
        )
        env = TorchJointPatchEnv(initialized, device=torch.device("cpu"))
        env.reset()
        result = initialized.em_assignment
        self.assertIsNotNone(result)
        for bin_idx, barcode in enumerate(result.barcode_ids):
            owner = result.cell_ids[int(result.top1_cell_index[bin_idx])]
            local = initialized.cells[result.cell_ids.index(owner)].candidate_bin_ids.index(barcode)
            self.assertEqual(int(env.final_masks()[owner][local]), 1)

        replace_barcodes: set[str] = set()
        for obs in env.joint_observations():
            barcode_indices = obs["joint_action_barcodes"].detach().cpu().numpy()
            replace_flags = obs["joint_action_is_replace"].detach().cpu().numpy()
            for barcode_idx, is_replace in zip(barcode_indices, replace_flags, strict=True):
                if int(barcode_idx) >= 0 and bool(is_replace):
                    replace_barcodes.add(env._barcode_keys[int(barcode_idx)])
        self.assertIn("shared", replace_barcodes)
        self.assertNotIn("fixed", replace_barcodes)
        self.assertNotIn("a_seed", replace_barcodes)
        self.assertNotIn("b_seed", replace_barcodes)

    def test_em_disabled_preserves_nuclear_only_reset(self) -> None:
        import torch

        a = _episode(
            "A",
            (0.0, 0.0),
            ["seed", "candidate"],
            [(0.0, 0.0), (2.0, 0.0)],
            [[1.0, 0.0], [0.0, 1.0]],
            [1, 0],
        )
        patch = PatchContext(
            patch_id="disabled",
            cells=(a,),
            core_cell_ids=("A",),
            margin_cell_ids=(),
            outer_bounds=PatchBounds(-1, 3, -1, 1),
            core_bounds=PatchBounds(-1, 3, -1, 1),
            max_steps=5,
        )
        env = TorchJointPatchEnv(patch, device=torch.device("cpu"))
        env.reset()
        np.testing.assert_array_equal(env.final_masks()["A"], np.asarray([1, 0], dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
