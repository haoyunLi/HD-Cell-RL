import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

from scripts.run_source_separated_method_comparison import assemble_components, validate_exported_ownership, prepare_common_seeds


class SourceSeparatedComparisonTests(unittest.TestCase):
    def test_common_seeds_follow_inference_metadata_not_gt_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'modeled_only').mkdir(); (root/'subdiagnostic').mkdir()
            (root/'modeled_only/episode_build.yaml').write_text('inputs:\n  bins_path: unused.parquet\n')
            pd.DataFrame({'barcode':['a','b','ambiguous']}).to_parquet(root/'barcodes.parquet')
            external = root/'external.csv'
            pd.DataFrame({'barcode':['a','b','ambiguous'], 'cell_id':['1','3','4'],
                          'is_nuclear':[True,True,True]}).to_csv(external,index=False)
            confident = pd.DataFrame({'barcode':['a','b'], 'dominant_cell_id':[1.,2.]})
            with patch('hd_cell_rl.nuclear_seeds.load_confident_nuclear_rows', return_value=confident), \
                 patch('scripts.run_source_separated_method_comparison.NUCLEAR',external):
                prepare_common_seeds(root)
            actual = pd.read_csv(root/'common_confident_nuclear_bins.csv.gz',dtype={'cell_id':str})
            self.assertEqual(actual.set_index('barcode').cell_id.to_dict(),{'a':'1','b':'2'})
            differences=pd.read_csv(root/'subdiagnostic/selected_seed_owner_disagreement.csv')
            self.assertEqual(differences.barcode.tolist(),['b'])

    def test_export_unique_and_locked(self):
        frame = pd.DataFrame({'barcode':['a','b'], 'cell_id':['1','2']})
        validate_exported_ownership(frame, {'a':'1'})
        with self.assertRaisesRegex(ValueError, 'nuclear ownership'):
            validate_exported_ownership(frame, {'a':'2'})
        with self.assertRaisesRegex(ValueError, 'duplicate physical'):
            validate_exported_ownership(pd.concat([frame, frame]), {})

    def test_export_missing_identity_rejected(self):
        frame = pd.DataFrame({'barcode':['a'], 'cell_id':[None]})
        with self.assertRaisesRegex(ValueError, 'missing exported'):
            validate_exported_ownership(frame, {})

    def make_sources(self, root, bad=False):
        bins = pd.DataFrame({'barcode':['b3','b1','b2'], 'matrix_col_index':[3,1,2],
                             'patch_id':['patch_b','patch_a','patch_b']})
        for patch, counts in [('patch_a', [90,2,90]), ('patch_b', [3,90,4])]:
            directory = root/'realizations'/('leave_out_'+patch)/'uniform_budget'
            directory.mkdir(parents=True)
            modeled = sparse.csr_matrix(np.array(counts)[:,None])
            unknown = sparse.csr_matrix([[1],[1],[1]])
            technical = sparse.csr_matrix(modeled.shape, dtype=np.int64)
            observed = modeled+unknown
            if bad and patch=='patch_a': observed = observed+unknown
            for name, value in [('modeled_cell',modeled),('unmodeled_biological',unknown),
                                ('technical_background',technical),('observed',observed)]:
                sparse.save_npz(directory/(name+'.npz'),value)
        return bins

    def test_own_fold_and_original_axis(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); bins=self.make_sources(root)
            result,_=assemble_components(bins,root)
            np.testing.assert_array_equal(result['modeled_cell'].toarray().ravel(),[3,2,4])
            np.testing.assert_array_equal(result['observed'].toarray().ravel(),[4,3,5])

    def test_component_conservation_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); bins=self.make_sources(root,bad=True)
            with self.assertRaisesRegex(ValueError,'conserve'):
                assemble_components(bins,root)

    def test_duplicate_physical_identity_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); bins=self.make_sources(root)
            bins.loc[1,'barcode']='b3'
            with self.assertRaisesRegex(ValueError,'unique'):
                assemble_components(bins,root)


if __name__=='__main__': unittest.main()
