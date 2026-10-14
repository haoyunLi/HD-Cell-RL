from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from scripts.build_patch_real_background_stress_dataset import (
    calibrated_cell_retention,
    sample_sparse_mixture,
)


class RealBackgroundStressDatasetTests(unittest.TestCase):
    def test_calibrated_cell_retention_holds_expected_total(self) -> None:
        probability = calibrated_cell_retention(
            pseudo_total=1000, real_total=200, real_fraction=0.2
        )
        self.assertAlmostEqual(probability, 0.16)
        expected_total = probability * 1000 + 0.2 * 200
        self.assertAlmostEqual(expected_total, 200.0)

    def test_pure_endpoints_recover_input_matrices(self) -> None:
        pseudo = sparse.coo_matrix(
            np.asarray([[5, 0, 2], [0, 3, 0]], dtype=np.int32)
        )
        real = sparse.coo_matrix(
            np.asarray([[0, 7, 0], [1, 0, 4]], dtype=np.int32)
        )
        pseudo_only, pseudo_total, real_total = sample_sparse_mixture(
            pseudo_matrix=pseudo,
            real_matrix=real,
            pseudo_probability=1.0,
            real_probability=0.0,
            rng=np.random.default_rng(1),
        )
        np.testing.assert_array_equal(pseudo_only.toarray(), pseudo.toarray())
        self.assertEqual(pseudo_total, int(pseudo.sum()))
        self.assertEqual(real_total, 0)

        real_only, pseudo_total, real_total = sample_sparse_mixture(
            pseudo_matrix=pseudo,
            real_matrix=real,
            pseudo_probability=0.0,
            real_probability=1.0,
            rng=np.random.default_rng(1),
        )
        np.testing.assert_array_equal(real_only.toarray(), real.toarray())
        self.assertEqual(pseudo_total, 0)
        self.assertEqual(real_total, int(real.sum()))

    def test_rejects_impossible_calibration(self) -> None:
        with self.assertRaisesRegex(ValueError, "more pseudo molecules"):
            calibrated_cell_retention(
                pseudo_total=100, real_total=1000, real_fraction=0.0
            )


if __name__ == "__main__":
    unittest.main()
