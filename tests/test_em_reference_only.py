from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from hd_cell_rl import em_assignment as em
from test_em_assignment import _simple_sparse_input, _config


class ReferenceOnlyTests(unittest.TestCase):
    def fixture(self):
        inp=_simple_sparse_input(ll=np.array([[-1,-3],[-3,-1],[-2,-1]]),confidence=np.ones(3),
            distances=np.array([[1,3],[3,1],[2,2]]),locked=np.array([True,True,False]),locked_owner=np.array([0,1,-1]))
        cfg=_config(cell_specific_expression_enabled=True,cell_profile_relative_prior_strength=2.,cell_profile_update_damping=0.)
        return inp,cfg,sparse.csr_matrix([[5,1],[1,4],[2,3]]),np.array([[.8,.2],[.2,.8]])

    def test_reference_profile_is_q_mixture_without_counts(self):
        q=np.array([[.25,.75],[.8,.2]])
        theta=np.array([[.9,.1],[.1,.9]])
        profile=em.reference_only_cell_profile(reference_theta=theta,cell_type_posterior=q,reference_prior_count=20.,epsilon=1e-12)
        np.testing.assert_allclose(profile,q@theta,atol=1e-11)
        np.testing.assert_allclose(profile.sum(1),1.,atol=1e-14)

    def test_exact_legacy_initializer_intervention_equivalence(self):
        inp,cfg,x,theta=self.fixture()
        original=em.initialize_cell_expression_profiles
        def replacement(**kw):
            _,lam,n,m=original(**kw)
            profile=em.reference_only_cell_profile(reference_theta=kw['reference_theta'],cell_type_posterior=kw['cell_type_posterior'],reference_prior_count=m,epsilon=kw['epsilon'])
            return profile,lam,n,m
        new=em.run_generalized_em(inp,cfg,bin_gene_counts=x,reference_theta=theta,use_reference_only_cell_profile=True)
        with patch.object(em,'initialize_cell_expression_profiles',replacement):
            expected=em.run_generalized_em(inp,cfg,bin_gene_counts=x,reference_theta=theta)
        for field in ('responsibility','cell_type_posterior','cell_expression_profile','final_combined_score'):
            np.testing.assert_array_equal(getattr(new,field),getattr(expected,field))

    def test_lambda_q_initialization_and_locks_preserved(self):
        inp,cfg,x,theta=self.fixture()
        q0=em.initialize_q_from_nuclear_seeds(inp)
        baseline=em.run_generalized_em(inp,cfg,bin_gene_counts=x,reference_theta=theta)
        with patch.object(em,'initialize_q_from_nuclear_seeds',wraps=em.initialize_q_from_nuclear_seeds) as initialize:
            result=em.run_generalized_em(inp,cfg,bin_gene_counts=x,reference_theta=theta,use_reference_only_cell_profile=True)
            self.assertEqual(initialize.call_count,1)
        np.testing.assert_array_equal(result.responsibility[:4],[1,0,0,1])
        for field in ('cell_profile_reliability','nuclear_expression_count_total','spatial_prior','expression_confidence'):
            np.testing.assert_array_equal(getattr(result,field),getattr(baseline,field))
        self.assertEqual(result.relative_profile_prior_count,baseline.relative_profile_prior_count)
        self.assertGreater(np.max(np.abs(result.cell_type_posterior-q0)),1e-4)

    def test_profile_stays_initial_reference_even_when_q_updates(self):
        inp,cfg,x,theta=self.fixture()
        q0=em.initialize_q_from_nuclear_seeds(inp)
        result=em.run_generalized_em(inp,cfg,bin_gene_counts=x,reference_theta=theta,use_reference_only_cell_profile=True)
        np.testing.assert_allclose(result.cell_expression_profile,q0@theta,atol=1e-7)
        self.assertFalse(np.allclose(result.cell_expression_profile,result.cell_type_posterior@theta,atol=1e-5))

    def test_reference_mixture_ll_is_not_type_only_ll(self):
        theta=np.array([[.99,.01],[.01,.99]])
        q=np.array([[.5,.5]])
        profile=em.reference_only_cell_profile(reference_theta=theta,cell_type_posterior=q,reference_prior_count=10.,epsilon=1e-12)
        value=em.compute_cell_expression_compatibility(bin_gene_counts=sparse.csr_matrix([[1,0]]),cell_expression_profile=profile,
            pair_bin_index=np.array([0]),candidate_cell_index=np.array([0]),epsilon=1e-12)[0]
        self.assertGreater(value,(q@np.log(theta[:,0])).item()+.5)

    def test_false_flag_reproduces_default(self):
        inp,cfg,x,theta=self.fixture()
        a=em.run_generalized_em(inp,cfg,bin_gene_counts=x,reference_theta=theta)
        b=em.run_generalized_em(inp,cfg,bin_gene_counts=x,reference_theta=theta,use_reference_only_cell_profile=False)
        np.testing.assert_array_equal(a.responsibility,b.responsibility)
        np.testing.assert_array_equal(a.cell_type_posterior,b.cell_type_posterior)

    def test_invalid_config_and_score_override_rejected(self):
        inp,cfg,x,theta=self.fixture()
        for bad in (replace(cfg,cell_profile_update_damping=.1),replace(cfg,cell_specific_expression_enabled=False),
                    replace(cfg,cell_profile_compatibility_mode='leave_one_out'),replace(cfg,cell_profile_reliability_mode='nuclear_heldout_predictive')):
            with self.assertRaises(ValueError):em.run_generalized_em(inp,bad,bin_gene_counts=x,reference_theta=theta,use_reference_only_cell_profile=True)
        with self.assertRaises(ValueError):
            em.run_generalized_em(inp,cfg,bin_gene_counts=x,reference_theta=theta,use_reference_only_cell_profile=True,fixed_cell_expression_compatibility=np.zeros(6))

    def test_zero_nuclear_expression_has_reference_profile_and_zero_lambda(self):
        inp,cfg,x,theta=self.fixture()
        x=sparse.csr_matrix([[0,0],[0,0],[3,4]])
        inp=replace(inp,ll=np.zeros((3,2)),expression_confidence=np.array([0.,0.,.5]))
        result=em.run_generalized_em(inp,cfg,bin_gene_counts=x,reference_theta=theta,use_reference_only_cell_profile=True)
        np.testing.assert_array_equal(result.cell_profile_reliability,[0.,0.])
        np.testing.assert_allclose(result.cell_expression_profile,.5,atol=1e-7)
        np.testing.assert_array_equal(result.responsibility[:4],[1,0,0,1])


if __name__=='__main__':unittest.main()
