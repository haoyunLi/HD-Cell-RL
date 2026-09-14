from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from hd_cell_rl.em_types import SparseEMInput
from scripts.run_em_weight_sweep import (
    _nuclear_profile_pair_compatibility,
    _oracle_q_for_input,
    _variant_name,
)


class EMWeightSweepTests(unittest.TestCase):
    def test_variant_name_records_profile_update_damping(self) -> None:
        self.assertEqual(
            _variant_name(1.0, 1.5, 0.25, 0.1),
            "alpha_1p0__beta_1p5__kappa_0p25__profile_damping_0p1",
        )
        self.assertEqual(
            _variant_name(1.0, 1.5, 0.25),
            "alpha_1p0__beta_1p5__kappa_0p25",
        )
        self.assertEqual(
            _variant_name(1.0, 1.5, 0.25, 1.0, "leave_one_out"),
            (
                "alpha_1p0__beta_1p5__kappa_0p25__profile_damping_1p0"
                "__profile_compat_leave_one_out"
            ),
        )
        self.assertEqual(
            _variant_name(
                1.0,
                1.5,
                2.0,
                0.0,
                "standard",
                None,
                "nuclear_heldout_predictive",
                101,
            ),
            (
                "alpha_1p0__beta_1p5__kappa_2p0__profile_damping_0p0"
                "__reliability_nuclear_heldout_predictive__lambda_grid_101"
            ),
        )
        self.assertEqual(
            _variant_name(
                1.0,
                1.5,
                2.0,
                1.0,
                "spatial_block_crossfit",
                8.0,
            ),
            (
                "alpha_1p0__beta_1p5__kappa_2p0__profile_damping_1p0"
                "__profile_compat_spatial_block_crossfit__block_um_8p0"
            ),
        )

    def test_oracle_q_uses_the_reference_type_axis(self) -> None:
        q = _oracle_q_for_input(
            cell_ids=("cell_a", "cell_b"),
            type_by_cell={"cell_a": "type_2", "cell_b": "type_1"},
            reference_cell_types=("type_1", "type_2"),
        )
        np.testing.assert_array_equal(
            q,
            np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        )

    def test_nuclear_profile_compatibility_is_physical_cell_specific(self) -> None:
        em_input = SparseEMInput(
            patch_id="profile_test",
            candidate_max_distance_um=20.0,
            barcode_ids=("seed_a", "seed_b", "query"),
            barcode_xy_um=np.asarray(
                [[0.0, 0.0], [4.0, 0.0], [2.0, 0.0]], dtype=np.float32
            ),
            ll=np.asarray(
                [[4.0, -4.0], [-4.0, 4.0], [0.0, 0.0]], dtype=np.float32
            ),
            expression_confidence=np.ones((3,), dtype=np.float32),
            candidate_row_splits=np.asarray([0, 2, 4, 6], dtype=np.int64),
            candidate_cell_index=np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
            pair_bin_index=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
            pair_distance_um=np.asarray([0.0, 4.0, 4.0, 0.0, 2.0, 2.0], dtype=np.float32),
            cell_ids=("cell_a", "cell_b"),
            is_nuclear_locked=np.asarray([True, True, False]),
            locked_owner_cell_index=np.asarray([0, 1, -1], dtype=np.int64),
            log_prior=np.full((2, 2), -np.log(2.0), dtype=np.float64),
        )
        expression = sparse.csr_matrix(
            np.asarray([[10.0, 0.0], [0.0, 10.0], [0.0, 5.0]], dtype=np.float64)
        )
        compatibility = _nuclear_profile_pair_compatibility(
            em_input=em_input,
            bin_expression=expression,
            reference_theta=np.asarray(
                [[0.99, 0.01], [0.01, 0.99]], dtype=np.float64
            ),
            prior_umis=10.0,
            epsilon=1.0e-12,
        )
        self.assertEqual(compatibility.shape, (6,))
        self.assertTrue(np.all(np.isfinite(compatibility)))
        self.assertGreater(float(compatibility[5]), float(compatibility[4]))


if __name__ == "__main__":
    unittest.main()
