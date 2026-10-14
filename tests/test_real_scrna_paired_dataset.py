import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy import sparse
from scripts.build_real_scrna_paired_dataset import calibration_probability,check_pair,generate_complement_background
from scripts.run_real_scrna_paired_benchmark import summarize_supported
from scripts import run_source_separated_method_comparison as protocol
from scripts.activate_real_scrna_dataset import activate


class PairedRealDatasetTests(unittest.TestCase):
    def test_activation_preserves_old_dataset_and_selects_explicit_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset=root/'new';active=root/'colorectal_nucleus_based_multi_owner_v2'
            (dataset/'no_background/filtered_feature_bc_matrix').mkdir(parents=True)
            (dataset/'no_background/filtered_feature_bc_matrix/matrix.mtx.gz').write_bytes(b'test')
            (dataset/'completion.json').write_text(json.dumps(dict(complete=True,whole_section=True,covered_bins_unchanged=True)))
            active.mkdir();(active/'old.txt').write_text('preserve')
            receipt=activate(dataset,active,'no_background','test')
            self.assertTrue(active.is_symlink());self.assertEqual(active.resolve(),dataset/'no_background')
            self.assertEqual((Path(receipt['backup'])/'old.txt').read_text(),'preserve')
            self.assertEqual(receipt['deleted_files'],0)
            self.assertEqual(receipt['dataset_role'],'experimental_control')
            self.assertFalse(receipt['biological_realism_validated'])
            self.assertIn('did not pass all',(active/'EXPERIMENTAL_CONTROL.md').read_text())
            with self.assertRaises(FileExistsError):activate(dataset,active,'no_background','test')

    def test_activation_refuses_unvalidated_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset=root/'new';dataset.mkdir()
            (dataset/'completion.json').write_text(json.dumps(dict(complete=False)))
            active=root/'colorectal_nucleus_based_multi_owner_v2';active.mkdir()
            with self.assertRaises(ValueError):activate(dataset,active,'no_background','test')
            self.assertTrue(active.is_dir());self.assertFalse(active.is_symlink())

    def test_capture_ignores_audit_counts(self):
        p=calibration_probability([10,9000],[100,1],[True,False],[True,True])
        self.assertEqual(p,.1)
        self.assertEqual(p,calibration_probability([10,0],[100,999],[True,False],[True,True]))
        with self.assertRaises(ValueError):calibration_probability([110],[100],[True],[True])

    def test_pair_is_exact_and_complement_only(self):
        m=sparse.csr_matrix([[2,1],[0,0],[3,0]])
        b=sparse.csr_matrix([[0,0],[3,4],[0,0]])
        x=check_pair(m,b,np.array([True,False,True]))
        self.assertEqual((x-m-b).nnz,0)
        self.assertEqual((x[[0,2]]-m[[0,2]]).nnz,0)
        with self.assertRaises(ValueError):check_pair(m,b,np.array([True,True,True]))
        with self.assertRaises(ValueError):check_pair(m,b,np.array([False,False,True]))

    def test_background_uses_only_allowed_support_and_is_reproducible(self):
        xy=np.array([[0,0],[2,0],[4,0],[6,0]],float)
        kwargs=dict(xy=xy,allowed=np.array([False,True,True,False]),cell_xy=np.array([[0,0.]]),
            cell_types=np.array(['A']),raw_profiles=sparse.csr_matrix([[5,15]]),
            source_types=np.array(['A']),rate=30,seed=24)
        a,trace=generate_complement_background(**kwargs);b,_=generate_complement_background(**kwargs)
        self.assertEqual((a-b).nnz,0);self.assertEqual(a[[0,3]].nnz,0)
        self.assertEqual(a.sum(),trace.counts.sum());self.assertGreater(a.sum(),0)

    def test_background_far_from_modeled_cells_has_finite_fallback(self):
        a,trace=generate_complement_background(
            xy=np.array([[10000.,10000.],[10002.,10000.]]),allowed=np.ones(2,bool),
            cell_xy=np.array([[0.,0.]]),cell_types=np.array(['A']),
            raw_profiles=sparse.csr_matrix([[5,15]]),source_types=np.array(['A']),rate=30,seed=24)
        self.assertTrue(np.isfinite(a.data).all())
        self.assertEqual(a.sum(),trace.counts.sum())
        self.assertGreater(a.sum(),0)
        self.assertFalse(trace.local_context_supported.any())
        self.assertTrue((trace.nearest_modeled_cell_um>72).all())

    def test_evaluation_support_is_a_posthoc_stratum_not_a_cell_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'benchmark';data=root/'data'
            (out/'subdiagnostic').mkdir(parents=True);data.mkdir()
            (out/'subdiagnostic/evaluation_checks.json').write_text(json.dumps([
                dict(condition='modeled_only',method='bin2cell',missing_core_nuclear_bins=4)]))
            pd.DataFrame(dict(cell_id=['1','2','3'],cell_type=['A','B','C'],reference_supported=[True,False,False])).to_csv(
                data/'cell_reference_support.csv',index=False)
            rows=[]
            for condition in ['modeled_only','source_separated']:
                for cell in ['1','2']:
                    rows.append(dict(cell_id=cell,cell_type='inference_label',patch_id='p',condition=condition,method='distance_only',
                                     **{m:.5 for m in protocol.METRICS}))
            pd.DataFrame(rows).to_csv(out/'subdiagnostic/per_cell_comparison.csv',index=False)
            pd.DataFrame(dict(matrix_col_index=[0,1],barcode=['a','b'])).to_parquet(root/'barcodes.parquet',index=False)
            pd.DataFrame(dict(barcode_index=[0],cell_id=[1],cell_coverage_fraction=[1.])).to_parquet(root/'gt.parquet',index=False)
            components=dict(modeled_cell=sparse.csr_matrix([[1],[0]]),unmodeled_biological=sparse.csr_matrix([[0],[1]]))
            pd.DataFrame(dict(patch_id=['p'],cohort=['additional12'])).to_csv(root/'patches.csv',index=False)
            with patch.object(protocol,'AUDIT',root),patch.object(protocol,'FRACTIONAL',root/'gt.parquet'), \
                 patch.object(protocol,'PATCHES',root/'patches.csv'), \
                 patch.object(protocol,'assemble_components',return_value=(components,{})):
                summarize_supported(out,data)
            self.assertIn('not a fully seed-retention-matched comparison',(out/'report.md').read_text())
            main=pd.read_csv(out/'method_comparison_supported.csv');missing=pd.read_csv(out/'method_comparison_missing_reference.csv')
            self.assertTrue(main.n_cells.eq(1).all());self.assertTrue(missing.n_cells.eq(1).all())
            coverage=pd.read_csv(out/'evaluated_type_coverage.csv').set_index('cell_type')
            self.assertEqual(coverage.loc['C','evaluated_core_cells'],0)
            retained=pd.read_csv(out/'subdiagnostic/per_cell_with_reference_support.csv')
            self.assertEqual(len(retained),4)
            cohorts=pd.read_csv(out/'method_comparison_by_cohort.csv')
            self.assertTrue(cohorts.cohort.eq('additional12').all())
            self.assertTrue(cohorts.n_cells.eq(1).all())


if __name__=='__main__':unittest.main()
