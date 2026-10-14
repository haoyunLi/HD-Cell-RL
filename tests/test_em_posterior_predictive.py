from dataclasses import replace
import unittest

import numpy as np
from scipy import sparse
from scipy.special import gammaln
from scipy.stats import dirichlet_multinomial

from hd_cell_rl import em_assignment as em
from hd_cell_rl.em_posterior_predictive import nuclear_dirichlet_posterior, posterior_predictive_compatibility
from test_em_assignment import _simple_sparse_input, _config


class PredictiveTests(unittest.TestCase):
    def score(self, x, alpha):
        x, alpha = np.asarray(x, dtype=float), np.asarray(alpha, dtype=float)
        return posterior_predictive_compatibility(bin_gene_counts=sparse.csr_matrix(x), posterior_alpha=alpha,
            pair_bin_index=np.repeat(np.arange(len(x)), len(alpha)), candidate_cell_index=np.tile(np.arange(len(alpha)), len(x)))

    def test_matches_independent_scipy_distribution(self):
        x = np.array([[1,2,3],[0,0,0],[5,0,1],[100,2,0]])
        alpha = np.array([[.001,2,9],[4,1,3]])
        expected = []
        for row in x:
            n = row.sum()
            for a in alpha:
                coefficient = gammaln(n+1)-gammaln(row+1).sum()
                expected.append((dirichlet_multinomial.logpmf(row,a,n)-coefficient)/n if n else 0.)
        np.testing.assert_allclose(self.score(x,alpha),expected,atol=1e-12)

    def test_one_umi_equals_plugin_and_zero_bin_zero(self):
        a = np.array([[.01,2,4],[1,9,3]])
        np.testing.assert_allclose(self.score(np.eye(3),a).reshape(3,2),np.log(a/a.sum(1)[:,None]).T,atol=1e-13)
        np.testing.assert_array_equal(self.score([[0,0,0]],a),[0.,0.])

    def test_large_concentration_plugin_limit(self):
        x=np.array([[2,1,3],[1,1,0]])
        a=np.array([[.2,.3,.5]])*1e14
        np.testing.assert_allclose(self.score(x,a),(x@np.log(a[0]/a.sum()))/x.sum(1),atol=1e-12)

    def test_order_independence_and_tiny_prior(self):
        x=np.array([[2,0,1],[1,1,1],[0,2,0]])
        a=np.array([[1e-12,2,4],[3,1e-10,1]])
        np.testing.assert_allclose(self.score(x[[2,0,1]],a).reshape(3,2),self.score(x,a).reshape(3,2)[[2,0,1]])

    def test_invalid_counts_and_alpha(self):
        for x,a in [([[.5,1]],[[1,2]]),([[-1,2]],[[1,2]]),([[1,2]],[[0,2]]),([[1,2]],[[np.nan,2]])]:
            with self.assertRaises(ValueError): self.score(x,a)

    def fixture(self):
        inp=_simple_sparse_input(ll=np.array([[-1,-3],[-3,-1],[-2,-1]]),confidence=np.ones(3),
            distances=np.array([[1,3],[3,1],[2,2]]),locked=np.array([True,True,False]),locked_owner=np.array([0,1,-1]))
        cfg=_config(cell_specific_expression_enabled=True,cell_profile_relative_prior_strength=2.,cell_profile_update_damping=0.)
        return inp,cfg,sparse.csr_matrix([[5,1],[1,4],[2,3]]),np.array([[.8,.2],[.2,.8]])

    def test_nuclear_only_posterior_and_exact_initial_mean(self):
        inp,cfg,x,theta=self.fixture()
        alpha,profile,lam,totals,prior=nuclear_dirichlet_posterior(inp,cfg,x,theta)
        altered=x.toarray();altered[2]=[999,0]
        np.testing.assert_array_equal(alpha,nuclear_dirichlet_posterior(inp,cfg,altered,theta)[0])
        result=em.run_generalized_em(inp,cfg,bin_gene_counts=x,reference_theta=theta)
        np.testing.assert_allclose(profile,result.cell_expression_profile,rtol=1e-7)
        np.testing.assert_allclose(alpha.sum(1),totals+prior+2*cfg.epsilon)
        identity=sparse.eye(2,format='csr')
        args=dict(bin_gene_counts=identity,pair_bin_index=np.repeat(np.arange(2),2),candidate_cell_index=np.tile(np.arange(2),2))
        np.testing.assert_allclose(posterior_predictive_compatibility(**args,posterior_alpha=alpha),
            em.compute_cell_expression_compatibility(**args,cell_expression_profile=profile,epsilon=cfg.epsilon),atol=1e-12)

    def test_solver_hook_preserves_existing_algorithm_and_nuclear_locks(self):
        inp,cfg,x,theta=self.fixture()
        alpha,profile,*_=nuclear_dirichlet_posterior(inp,cfg,x,theta)
        args=dict(bin_gene_counts=x,pair_bin_index=inp.pair_bin_index,candidate_cell_index=inp.candidate_cell_index)
        plugin=em.compute_cell_expression_compatibility(**args,cell_expression_profile=profile,epsilon=cfg.epsilon)
        baseline=em.run_generalized_em(inp,cfg,bin_gene_counts=x,reference_theta=theta)
        same=em.run_generalized_em(inp,cfg,bin_gene_counts=x,reference_theta=theta,fixed_cell_expression_compatibility=plugin)
        np.testing.assert_array_equal(same.responsibility,baseline.responsibility)
        np.testing.assert_array_equal(same.cell_type_posterior,baseline.cell_type_posterior)
        pred=posterior_predictive_compatibility(**args,posterior_alpha=alpha)
        result=em.run_generalized_em(inp,cfg,bin_gene_counts=x,reference_theta=theta,fixed_cell_expression_compatibility=pred)
        np.testing.assert_array_equal(result.responsibility[:4],[1,0,0,1])
        np.testing.assert_array_equal(result.cell_profile_reliability,baseline.cell_profile_reliability)
        for bad in (replace(cfg,cell_profile_update_damping=.1),replace(cfg,cell_specific_expression_enabled=False)):
            with self.assertRaises(ValueError):
                em.run_generalized_em(inp,bad,bin_gene_counts=x,reference_theta=theta,fixed_cell_expression_compatibility=pred)

    def test_whole_bin_not_independent_single_read_average(self):
        a=np.array([[.2,.8]])
        repeated=self.score([[2,0]],a)[0]
        self.assertGreater(repeated,np.log(.2))
        diverse=self.score([[1,1]],a)[0]
        self.assertLess(diverse,.5*(np.log(.2)+np.log(.8)))


if __name__=='__main__': unittest.main()
