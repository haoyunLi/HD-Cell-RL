import unittest
import numpy as np
from scipy import sparse
from hd_cell_rl.pseudo_components import (sample_captured_profiles, spatial_source_kernels,
    local_type_probabilities, place_source_profiles, fit_component_budgets, combine_components)


class PseudoComponentTests(unittest.TestCase):
    def test_capture_budget_and_gene_richness_are_not_independently_forced(self):
        patterns=sparse.csr_matrix(np.ones((2,100)))
        x=sample_captured_profiles(patterns,[10,1000],np.random.default_rng(1))
        np.testing.assert_array_equal(np.asarray(x.sum(1)).ravel(),[10,1000])
        self.assertLessEqual(x[0].nnz,10)
        self.assertGreater(x[1].nnz,x[0].nnz)

    def test_unknown_support_not_zero_filling_and_source_trace(self):
        xy=np.array([[0,0],[2,0],[4,0],[100,0]])
        kernels=spatial_source_kernels(xy,['a']*3+['b'],[[2,0]],['a'],sigma_um=3,tissue_support=[True]*4)
        self.assertEqual(kernels[0,3],0)
        self.assertGreater(kernels[0,0],0)
        p=sparse.csr_matrix([[500,500]])
        x,trace=place_source_profiles(p,kernels,np.random.default_rng(2))
        self.assertEqual(int(x.sum()),1000)
        np.testing.assert_array_equal(np.asarray(x.sum(1)).ravel(),np.asarray(trace.sum(0)).ravel())
        self.assertEqual((combine_components(x,x,x)-x*3).nnz,0)

    def test_no_audit_count_leakage_into_fit(self):
        a=fit_component_budgets([20,10,1,1],[1,1,0,0],[1,0,1,0],[.2,1,.2,1])
        b=fit_component_budgets([20,10,9999,9000],[1,1,0,0],[1,0,1,0],[.2,1,.2,1])
        self.assertEqual(a,b)
        self.assertEqual(a['modeled_calibration_counts'],18)

    def test_local_types_and_invalid_inputs(self):
        names,p=local_type_probabilities([[0,0]],[[0,0],[100,0]],['A','B'],scale_um=5)
        self.assertEqual(names[np.argmax(p[0])],'A')
        with self.assertRaises(ValueError):sample_captured_profiles([[0,0]],[1],np.random.default_rng(1))
        with self.assertRaises(ValueError):sample_captured_profiles([[1]],[1.2],np.random.default_rng(1))
        with self.assertRaises(ValueError):fit_component_budgets([1,10],[1,1],[1,0],[1,1])


if __name__=='__main__':unittest.main()
