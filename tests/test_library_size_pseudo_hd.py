import unittest
import numpy as np
from scripts.build_library_size_pseudo_hd import select_matched_profiles, common_capture_probability
from scripts import run_source_separated_method_comparison as comparison


class LibraryPseudoTests(unittest.TestCase):
    def test_type_matched_unique_deterministic_selection(self):
        target=np.array(['b','a','a']);source=np.array(['a','b','a','a'])
        i=select_matched_profiles(target,source,19)
        np.testing.assert_array_equal(source[i],target)
        self.assertEqual(len(set(i)),3)
        np.testing.assert_array_equal(i,select_matched_profiles(target,source,19))
        with self.assertRaises(ValueError):select_matched_profiles(['c'],source,19)

    def test_capture_does_not_consult_audit_counts(self):
        mask=np.array([True,False])
        p=common_capture_probability(np.array([10,999]),np.array([100,1]),mask)
        self.assertEqual(p,.1)
        self.assertEqual(p,common_capture_probability(np.array([10,1]),np.array([100,999]),mask))
        with self.assertRaises(ValueError):common_capture_probability(np.array([110,1]),np.array([100,999]),mask)

    def test_evaluation_uses_current_profile_override(self):
        from unittest.mock import patch
        from pathlib import Path
        with patch.object(comparison,'PROFILE_ASSIGNMENT',Path('/new/mapping.csv')),patch.object(comparison,'PROFILE_H5',Path('/new/counts.h5')):
            args=comparison.evaluation_args(Path('/context'),Path('/evaluation'),'check')
        self.assertEqual(args[args.index('--gt_cell_assignments_csv')+1],Path('/new/mapping.csv'))
        self.assertEqual(args[args.index('--gt_sc_expression_h5')+1],Path('/new/counts.h5'))


if __name__=='__main__':unittest.main()
