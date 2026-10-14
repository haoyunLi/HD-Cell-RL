import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from hd_cell_rl.spatial_profile_assignment import assign_spatial_profiles, audit_spatial_assignment, plan_donor_split
from scripts.build_real_scrna_pseudo_hd import (
    eligible_sources, target_coordinates, selected_raw_counts, load_localization_module,
    place_real_profiles, main,
)


def targets(x, types=None):
    return pd.DataFrame(dict(cell_id=np.arange(len(x))+1, cell_type=types or ['A']*len(x),
                             x_um=x, y_um=np.zeros(len(x))))


def sources(n=4):
    return pd.DataFrame(dict(sc_cell_barcode=[f'raw_{i}' for i in range(n)], cell_type=['A']*n))


class SpatialAssignmentTests(unittest.TestCase):
    def test_within_patch_no_duplicates(self):
        result = assign_spatial_profiles(targets([1,2,3,4]), sources())
        self.assertEqual(result.sc_cell_barcode.nunique(), 4)

    def test_cross_boundary_is_not_independent_sampling(self):
        result = assign_spatial_profiles(targets([127,129]), sources(2))
        self.assertEqual(result.patch_id.nunique(), 2)
        self.assertEqual(result.sc_cell_barcode.nunique(), 2)

    def test_distant_patch_reuse_is_allowed(self):
        result = assign_spatial_profiles(targets([1,300,600]), sources(1))
        self.assertEqual(result.sc_cell_barcode.nunique(), 1)
        self.assertEqual(len(result), 3)

    def test_radius_is_inclusive(self):
        with self.assertRaisesRegex(ValueError, 'exhausted'):
            assign_spatial_profiles(targets([100,164]), sources(1), exclusion_distance_um=64)
        result = assign_spatial_profiles(targets([100,164.01]), sources(1), exclusion_distance_um=64)
        self.assertEqual(result.sc_cell_barcode.nunique(), 1)

    def test_same_physical_cell_in_overlapping_contexts(self):
        t = targets([1,2]); t = pd.concat([t,t.iloc[[0]]])
        result = assign_spatial_profiles(t, sources())
        self.assertEqual(len(result), 2)
        t.iloc[-1,t.columns.get_loc('x_um')] = 10
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            assign_spatial_profiles(t, sources())

    def test_seed_and_input_order_stability(self):
        t, s = targets([1,2,129,140,1000]), sources(10)
        a = assign_spatial_profiles(t, s, seed=73)
        b = assign_spatial_profiles(t.sample(frac=1, random_state=9), s.iloc[::-1], seed=73)
        pd.testing.assert_frame_equal(a,b)

    def test_exact_type_matching_and_no_fallback(self):
        s = sources(); s.loc[3,'cell_type'] = 'B'
        a = assign_spatial_profiles(targets([1,2], ['A','B']), s)
        self.assertEqual(a.loc[1,'sc_cell_barcode'], 'raw_3')
        with self.assertRaisesRegex(ValueError,'no eligible'):
            assign_spatial_profiles(targets([1],['C']),s)
        with self.assertRaisesRegex(ValueError,'requires'):
            assign_spatial_profiles(targets([1,2]), sources(1))

    def test_invalid_inputs(self):
        for width, radius in [(0,64),(128,0),(np.nan,64),(128,np.inf)]:
            with self.assertRaises(ValueError):
                assign_spatial_profiles(targets([1]),sources(),patch_size_um=width,exclusion_distance_um=radius)
        for bad in [targets([np.nan]), targets([np.inf])]:
            with self.assertRaises(ValueError): assign_spatial_profiles(bad,sources())
        with self.assertRaises(ValueError):
            assign_spatial_profiles(targets([1]),pd.concat([sources(),sources()]))

    def test_independent_audit_detects_tampering(self):
        a = assign_spatial_profiles(targets([127,129]),sources())
        a.loc[1,'sc_cell_barcode'] = a.loc[0,'sc_cell_barcode']
        with self.assertRaisesRegex(ValueError,'boundary'):
            audit_spatial_assignment(a,sources(),exclusion_distance_um=64)

    def test_donor_disjoint_selection(self):
        s = pd.DataFrame(dict(cell=['a','b'],sct_cell_type=['A','A'],case_id=['d1','d2']))
        result = eligible_sources(s,generator_donors=['d1'],reference_donors=['d2'])
        self.assertEqual(result.sc_cell_barcode.tolist(),['a'])
        with self.assertRaises(ValueError): eligible_sources(s,generator_donors=['d1'],reference_donors=['d1'])
        with self.assertRaises(ValueError): eligible_sources(s,generator_donors=['missing'],reference_donors=['d1'])

    def test_donor_planner_reports_unavoidable_reference_gaps(self):
        s = pd.DataFrame(dict(cell=['a','b','c','d','e'],
            sct_cell_type=['rare','A','A','A','A'],case_id=['d1','d1','d2','d3','d4']))
        t = targets([1,2],['rare','A'])
        split, audit = plan_donor_split(t,s)
        self.assertIn('d1',split['generator_donors'])
        self.assertFalse(set(split['generator_donors']) & set(split['reference_donors']))
        self.assertEqual(split['reference_missing_types'],['rare'])
        self.assertFalse(split['all_types_supported_in_reference'])
        self.assertTrue((audit.generator_cells>=audit.max_cells_in_one_patch).all())
        shuffled,_ = plan_donor_split(t.iloc[::-1],s.iloc[::-1])
        self.assertEqual(split,shuffled)

    def test_donor_planner_does_not_hide_missing_generator_types(self):
        s=pd.DataFrame(dict(cell=['a','b'],sct_cell_type=['A','A'],case_id=['d1','d2']))
        with self.assertRaisesRegex(ValueError,'no donor-disjoint split'):
            plan_donor_split(targets([1],['missing']),s)

    def test_coordinate_conversion_is_not_hardcoded(self):
        t = pd.DataFrame(dict(cell_id=[1],cell_type=['A'],nucleus_area_um2=[25],nucleus_area_px=[100],
                              nucleus_centroid_col_fullres_px=[20],nucleus_centroid_row_fullres_px=[40]))
        out = target_coordinates(t)
        self.assertEqual(out.x_um.iloc[0],10);self.assertEqual(out.y_um.iloc[0],20)

    def test_raw_h5_reuse_and_count_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'raw.h5'
            with h5py.File(path,'w') as f:
                m=f.create_group('matrix')
                for key,value in dict(data=[2.,4.,7.],indices=[0,1,1],indptr=[0,2,3],shape=[2,2],
                                      barcodes=[b'a',b'b']).items():m.create_dataset(key,data=value)
                m.create_dataset('features/name',data=[b'g1',b'g2'])
            x,g=selected_raw_counts(path,['b','a','b'])
            np.testing.assert_array_equal(x.toarray(),[[0,7],[2,4],[0,7]])
            with self.assertRaises(ValueError):selected_raw_counts(path,['missing'])
            with h5py.File(path,'r+') as f:f['matrix/data'][0]=0.25
            with self.assertRaisesRegex(ValueError,'raw counts'):selected_raw_counts(path,['a'])


VALIDATION = Path(__file__).resolve().parents[2]/'Bin2Cell_Validation'


@unittest.skipUnless((VALIDATION/'scripts/simulation/create_colorectal_pseudo_hd.py').exists(),
                     'requires existing Bin2Cell_Validation localization implementation')
class ExistingLocalizationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod=load_localization_module(VALIDATION)

    def fixture(self):
        profiles=sparse.csr_matrix(np.array([[200,100],[300,400]],dtype=np.int64))
        a=assign_spatial_profiles(targets([0,2]),sources())
        bins=pd.DataFrame(dict(cell_id=[1,1,2,2],barcode_index=[0,1,1,2],
            array_row=[0,0,0,0],array_col=[0,1,1,2],cell_coverage_fraction=[1,.5,.5,1],
            nuclear_fraction=[.5,0,0,.5],cytoplasm_fraction=[.3,.3,.3,.3],membrane_fraction=[.2,.2,.2,.2]))
        return profiles,a,bins

    def run_placement(self,p,block=1000):
        profiles,a,bins=self.fixture()
        return place_real_profiles(profiles,a,bins,source_module=self.mod,p0=np.array([.1,.2]),
            membrane_flags=np.array([False,True]),n_bins=3,seed=24,capture_probability=p,block_cells=block)

    def test_counts_shared_bin_and_capture_conservation(self):
        x,trace,audit,summary=self.run_placement(1)
        self.assertEqual(x.sum(),1000)
        np.testing.assert_array_equal(np.asarray(trace.sum(1)).ravel(),[300,700])
        self.assertGreater(trace[0,1],0);self.assertGreater(trace[1,1],0)
        np.testing.assert_array_equal(np.asarray(x.sum(1)).ravel(),np.asarray(trace.sum(0)).ravel())
        self.assertAlmostEqual(summary['expected_nuclear_fraction'],self.mod.T_NUCLEAR_FRACTION)
        thinned,t,ad,_=self.run_placement(.2)
        self.assertLess(thinned.sum(),1000);self.assertGreater(thinned.sum(),0)
        self.assertEqual(thinned.sum()+ad.capture_rejected_counts.sum(),1000)

    def test_capture_field_zero_bins_and_zero_capture(self):
        x,_,_,_=self.run_placement(np.array([1,0,1]))
        self.assertEqual(x[1].sum(),0)
        x,_,a,_=self.run_placement(0)
        self.assertEqual(x.sum(),0);self.assertEqual(a.capture_rejected_counts.sum(),1000)
        with self.assertRaises(ValueError):self.run_placement([.2,.2])
        with self.assertRaises(ValueError):self.run_placement(np.nan)

    def test_aggregation_blocks_do_not_change_sampling(self):
        x,t,a,_=self.run_placement(.4,1);y,u,b,_=self.run_placement(.4,1000)
        self.assertEqual((x!=y).nnz,0);self.assertEqual((t!=u).nnz,0)
        pd.testing.assert_frame_equal(a,b)

    def test_shared_bin_source_reuse_is_rejected_even_beyond_configured_radius(self):
        profiles,a,bins=self.fixture();a['sc_cell_barcode']='same_source'
        with self.assertRaisesRegex(ValueError,'sharing a physical bin'):
            place_real_profiles(profiles,a,bins,source_module=self.mod,p0=np.array([.1,.2]),
                membrane_flags=np.array([False,True]),n_bins=3,seed=24,capture_probability=1)

    def test_end_to_end_tenx_output_and_no_overwrite(self):
        import gzip
        from scipy.io import mmread
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original=root/'original';original.mkdir();(original/'spatial').mkdir()
            (original/'spatial/scalefactors_json.json').write_text('{}')
            raw=root/'raw.h5'
            with h5py.File(raw,'w') as f:
                m=f.create_group('matrix')
                for k,v in dict(data=[200,100,300,400],indices=[0,1,0,1],indptr=[0,2,4],
                                shape=[2,2],barcodes=[b'a',b'b']).items():m.create_dataset(k,data=v)
                m.create_dataset('features/name',data=[b'g1',b'g2'])
            with h5py.File(original/'filtered_feature_bc_matrix.h5','w') as f:
                m=f.create_group('matrix');m.create_dataset('barcodes',data=[b'x',b'y',b'z'])
                for k,v in dict(id=[b'id1',b'id2'],name=[b'g1',b'g2'],feature_type=[b'Gene Expression']*2).items():
                    m.create_dataset('features/'+k,data=v)
            targets([0,2]).to_csv(root/'cells.csv',index=False)
            pd.DataFrame(dict(cell=['a','b','ref'],sct_cell_type=['A']*3,case_id=['g','g','r'])).to_csv(
                root/'meta.tsv',sep='\t',index=False)
            _,_,bins=self.fixture();bins.to_parquet(root/'bins.parquet',index=False)
            (root/'membrane.txt').write_text('g2\n')
            (root/'annotation.gtf').write_text(
                'chr1\ttest\tgene\t1\t100\t.\t+\t.\tgene_id "g1"; gene_name "g1";\n'
                'chr1\ttest\texon\t1\t50\t.\t+\t.\tgene_id "g1"; gene_name "g1";\n')
            args=['--ground-truth-cells',str(root/'cells.csv'),'--source-h5',str(raw),
                  '--source-metadata',str(root/'meta.tsv'),'--generator-donors','g','--reference-donors','r',
                  '--output-dir',str(root/'out'),'--bin-assignments',str(root/'bins.parquet'),
                  '--original-hd-dir',str(original),'--gtf',str(root/'annotation.gtf'),
                  '--membrane-genes',str(root/'membrane.txt'),'--capture-probability','0.5']
            main(args)
            with gzip.open(root/'out/filtered_feature_bc_matrix/matrix.mtx.gz','rb') as f:x=mmread(f)
            self.assertEqual(x.shape,(2,3));self.assertGreater(x.sum(),0)
            mapping=pd.read_csv(root/'out/ground_truth_cell_assignments.csv')
            self.assertEqual(set(mapping.expression_origin),{'real_scRNA'})
            self.assertNotIn('synthetic_scdesign3_barcode',mapping)
            self.assertTrue((root/'out/experiment_manifest.json').exists())
            with self.assertRaises(FileExistsError):main(args)


if __name__=='__main__':unittest.main()
