"""Unit checks for diagnostic-only interventions and score accounting."""
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse

from scripts.diagnose_em_library_size import align_oracle, score_components, transition


class DiagnosticTests(unittest.TestCase):
    def test_oracle_aligns_physical_ids_and_selected_genes(self):
        mapping = pd.DataFrame({'cell_id':['a','b'], 'sc_cell_barcode':['s2','s1']})
        result = align_oracle(sparse.csr_matrix([[1,3,2],[4,2,6]]),
            ['s1','s2'], ['g1','g2','g3'], mapping, ['b','a'], ['g3','g1'], 1e-10)
        np.testing.assert_allclose(result, [[2/3,1/3],[.6,.4]])

    def test_missing_physical_id_fails(self):
        mapping = pd.DataFrame({'cell_id':['a'], 'sc_cell_barcode':['s1']})
        with self.assertRaises(KeyError):
            align_oracle(sparse.csr_matrix([[1]]), ['s1'], ['g1'], mapping, ['b'], ['g1'], 1e-10)

    def test_duplicate_source_axis_fails(self):
        with self.assertRaises(ValueError):
            align_oracle(sparse.csr_matrix([[1],[1]]), ['s1','s1'], ['g1'],
                pd.DataFrame({'cell_id':['a'],'sc_cell_barcode':['s1']}), ['a'], ['g1'], 1e-10)

    def test_components_reconstruct_cell_specific_weighted_score(self):
        result = SimpleNamespace(pair_bin_index=np.array([0,0]), candidate_cell_index=np.array([0,1]),
            cell_profile_reliability=np.array([.2,.7]), spatial_prior=np.array([.6,.4]),
            expression_confidence=np.array([.5]), final_type_expression_compatibility=np.array([-3.,-4.]),
            final_cell_expression_compatibility=np.array([-5.,-2.]))
        cfg = SimpleNamespace(spatial_weight=1., expression_weight=1.5, epsilon=1e-10)
        result.final_combined_score = np.log([.6,.4]) + .75 * np.array([.8*-3+.2*-5,.3*-4+.7*-2])
        parts = score_components(result, cfg)
        np.testing.assert_allclose(sum(parts), result.final_combined_score)

    def test_transitions_are_exclusive(self):
        self.assertEqual(transition('a','b','a'), 'damaged')
        self.assertEqual(transition('b','a','a'), 'corrected')
        self.assertEqual(transition('a','a','a'), 'both_correct')
        self.assertEqual(transition('b','c','a'), 'both_wrong')


if __name__ == '__main__':
    unittest.main()
