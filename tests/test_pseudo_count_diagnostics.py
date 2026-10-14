import unittest
import numpy as np
from scipy import sparse
from hd_cell_rl.pseudo_count_diagnostics import rarefy_exact,expected_genes,summarize_matched,row_features


class CountConditionedTests(unittest.TestCase):
    def test_exact_counts_never_add_genes_or_molecules(self):
        x=sparse.csr_matrix([[10,0,5],[0,1,0],[0,0,0]])
        y=rarefy_exact(x,[7,1,0],np.random.default_rng(1))
        np.testing.assert_array_equal(np.asarray(y.sum(1)).ravel(),[7,1,0])
        self.assertTrue(np.all(y.toarray()<=x.toarray()))
        with self.assertRaises(ValueError):rarefy_exact(x,[16,1,0],np.random.default_rng(1))
        with self.assertRaises(ValueError):rarefy_exact(x,[1.5,1,0],np.random.default_rng(1))

    def test_identity_and_reproducibility(self):
        x=sparse.csr_matrix([[20,10],[2,4]])
        self.assertEqual((x-rarefy_exact(x,[30,6],np.random.default_rng(1))).nnz,0)
        a=rarefy_exact(x,[7,3],np.random.default_rng(5));b=rarefy_exact(x,[7,3],np.random.default_rng(5))
        self.assertEqual((a-b).nnz,0)

    def test_expected_richness_known_hypergeometric_case(self):
        x=sparse.csr_matrix([[2,2],[4,0],[1,1],[0,0]])
        np.testing.assert_allclose(expected_genes(x,[2,2,1,0]),[5/3,1,1,0])
        np.testing.assert_allclose(expected_genes(x,[4,4,2,0]),[2,1,2,0])

    def test_same_counts_can_have_different_richness(self):
        r=sparse.csr_matrix([[10,0,0,0]]);s=sparse.csr_matrix([[3,3,2,2]])
        rows,ev=summarize_matched(r,s,1)
        self.assertEqual(rows[0]['expected_genes_ratio'],4)
        self.assertTrue(ev['matched_exact_counts'])

    def test_zero_support_is_reported_not_hidden(self):
        r=sparse.csr_matrix([[4,0],[0,0],[1,1]]);s=sparse.csr_matrix([[0,0],[3,0],[1,1]])
        rows,e=summarize_matched(r,s,1)
        self.assertEqual(e['n_both_positive'],1);self.assertEqual(e['n_real_only'],1);self.assertEqual(e['n_sim_only'],1)
        self.assertEqual(rows[0]['expected_genes_ratio'],1)

    def test_reference_gap_uses_normalized_likelihood(self):
        x=sparse.csr_matrix([[10,0],[100,0]])
        f=row_features(x,np.array([[.9,.1],[.1,.9]]),np.eye(2))
        np.testing.assert_allclose(f['type_ll_gap'],np.log(9))
        np.testing.assert_allclose(f['max_marker_fraction'],1)

    def test_invalid_fractional_and_negative_counts_fail(self):
        for x in [[[.5]], [[-1]], [[float('nan')]]]:
            with self.assertRaises(ValueError):rarefy_exact(x,[0],np.random.default_rng(1))


if __name__=='__main__':unittest.main()
