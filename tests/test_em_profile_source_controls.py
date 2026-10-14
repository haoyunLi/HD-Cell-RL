"""Matched profile controls must not alter observed counts, depth or identity."""
import unittest
import numpy as np
from scripts.run_em_profile_source_controls import matched_profile_counts, shrink_counts
from hd_cell_rl.em_assignment import _normalize_reference_theta, _normalize_cell_gene_counts


class ProfileSourceTests(unittest.TestCase):
    def setUp(self):
        self.oracle=np.array([[.1,.2,.7],[.6,.3,.1],[.2,.3,.5]])
        self.totals=np.array([11.,29.,0.])
        self.ids=['cell_a','cell_b','cell_zero']

    def test_expected_counts_are_exact_budget_times_profile(self):
        counts=matched_profile_counts(self.oracle,self.totals,self.ids)
        np.testing.assert_allclose(counts,self.oracle*self.totals[:,None])
        np.testing.assert_allclose(counts.sum(1),self.totals)

    def test_sampled_counts_preserve_integer_budgets_and_zero_cells(self):
        counts=matched_profile_counts(self.oracle,self.totals,self.ids,seed=2026091601)
        np.testing.assert_array_equal(counts.sum(1),self.totals)
        np.testing.assert_array_equal(counts,np.rint(counts))
        np.testing.assert_array_equal(counts[-1],0)

    def test_sampled_counts_ignore_cell_processing_order(self):
        counts=matched_profile_counts(self.oracle,self.totals,self.ids,seed=2026091601)
        order=np.array([2,0,1])
        shuffled=matched_profile_counts(self.oracle[order],self.totals[order],np.asarray(self.ids)[order],seed=2026091601)
        np.testing.assert_array_equal(shuffled,counts[order])

    def test_same_physical_cell_matches_between_patch_contexts(self):
        all_counts=matched_profile_counts(self.oracle,self.totals,self.ids,seed=2026091602)
        one=matched_profile_counts(self.oracle[1:2],self.totals[1:2],self.ids[1:2],seed=2026091602)
        np.testing.assert_array_equal(one[0],all_counts[1])

    def test_profile_inputs_are_not_mutated(self):
        p=self.oracle.copy(); n=self.totals.copy()
        matched_profile_counts(self.oracle,self.totals,self.ids,seed=123)
        np.testing.assert_array_equal(p,self.oracle)
        np.testing.assert_array_equal(n,self.totals)

    def test_fractional_budget_and_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            matched_profile_counts(self.oracle,[1.1,2,3],self.ids)
        with self.assertRaises(ValueError):
            matched_profile_counts(self.oracle,self.totals,['a','a','b'])

    def test_shrinkage_matches_existing_initializer_equation(self):
        counts=self.oracle*self.totals[:,None]
        q=np.array([[.2,.8],[.5,.5],[1.,0.]])
        theta=np.array([[.1,.5,.4],[.6,.3,.1]])
        before_q=q.copy(); before_theta=theta.copy()
        result=shrink_counts(counts,q,theta,20.,1e-12)
        normalized_theta=_normalize_reference_theta(theta,n_types=2,n_genes=3,epsilon=1e-12)
        expected=_normalize_cell_gene_counts(counts+20*(q@normalized_theta),epsilon=1e-12)
        np.testing.assert_array_equal(result,expected)
        np.testing.assert_array_equal(q,before_q);np.testing.assert_array_equal(theta,before_theta)
        np.testing.assert_allclose(result.sum(1),1.)


if __name__=='__main__':
    unittest.main()
