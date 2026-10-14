import unittest
import numpy as np
from scipy import sparse
from hd_cell_rl.pseudo_realism import (
    thin_counts, move_counts, gene_reweight, aggregate_bins, soften_compartment_weights,
)


class PseudoRealismTests(unittest.TestCase):
    def test_thinning_preserves_zeros_and_never_creates_molecules(self):
        x = sparse.csr_matrix([[10, 0], [1, 8]])
        y = thin_counts(x, np.array([0., 1.]), np.random.default_rng(1))
        np.testing.assert_array_equal(y.toarray(), [[0, 0], [1, 8]])
        with self.assertRaises(ValueError):
            thin_counts(x, 1.1, np.random.default_rng(1))

    def test_transport_conserves_gene_counts_and_cannot_cross_patches(self):
        grid = np.array([[0, 0], [0, 1], [0, 0], [0, 1]])
        x = sparse.csr_matrix([[100, 1], [10, 0], [0, 0], [0, 0]])
        y, rejected = move_counts(x, grid, np.array([0, 0, 1, 1]), 1, 1,
                                  np.random.default_rng(1))
        np.testing.assert_array_equal(y.sum(0), x.sum(0))
        self.assertEqual(y[2:].sum(), 0)
        self.assertGreater(rejected, 0)

    def test_platform_resampling_conserves_each_bin_count(self):
        x = sparse.csr_matrix([[100, 1], [10, 20]])
        y = gene_reweight(x, [0.1, 10], np.random.default_rng(1))
        np.testing.assert_array_equal(x.sum(1), y.sum(1))
        self.assertGreater(y[:, 1].sum(), x[:, 1].sum())

    def test_partial_aggregate_blocks_are_excluded(self):
        grid = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]])
        y, _ = aggregate_bins(sparse.csr_matrix(np.ones((5, 2))), grid, np.zeros(5), 2)
        np.testing.assert_array_equal(y.toarray(), [[4, 4]])

    def test_relaxed_membrane_can_enter_cytoplasm_but_not_nucleus(self):
        _, _, m = soften_compartment_weights([1, 0, 0], [0, 1, 0], [0, 0, 1], 0.25)
        np.testing.assert_allclose(m, [0, 0.375, 0.625])


if __name__ == '__main__':
    unittest.main()
