import unittest
import numpy as np
import pandas as pd
from scipy import sparse

from hd_cell_rl.spatial_profile_assignment import assign_background_profiles
from scripts.build_real_scrna_paired_dataset import render_background_sources


class UniqueBackgroundTests(unittest.TestCase):
    def fixture(self):
        sources=pd.DataFrame({'sc_cell_barcode':['a','b','c'],'cell_type':['A']*3})
        fixed=pd.DataFrame(dict(cell_id=[1],cell_type=['A'],x_um=[0.],y_um=[0.],sc_cell_barcode=['a']))
        proposed=pd.DataFrame(dict(source_center_barcode_index=[1,2,3,4],cell_type=['A']*4,
                                  x_um=[2.,4.,6.,8.],y_um=[0.]*4,counts=[10,20,30,40]))
        return proposed,fixed,sources

    def test_reduction_counts_types_and_fixed_identity(self):
        p,m,s=self.fixture();r,proposals,budgets,audit=assign_background_profiles(p,m,s)
        self.assertEqual(len(r),2);self.assertEqual(set(r.sc_cell_barcode),{'b','c'})
        self.assertEqual(r.counts.sum(),100)
        self.assertTrue((budgets.counts_before==budgets.counts_after).all())
        self.assertEqual(audit['within_patch_reuse'],0)
        self.assertEqual(audit['n_source_positions'],len(m)+len(r))
        self.assertEqual(audit['within_exclusion_distance_reuse'],0)
        self.assertEqual(m.sc_cell_barcode.tolist(),['a'])
        self.assertTrue(r.cell_type.eq('A').all())
        self.assertEqual(proposals.loc[proposals.retained,'sc_cell_barcode'].tolist(),r.sc_cell_barcode.tolist())
        self.assertTrue(proposals.loc[~proposals.retained,'sc_cell_barcode'].isna().all())

    def test_proposal_trace_distinguishes_old_and_new_identities(self):
        p,m,s=self.fixture();p['sc_cell_barcode']='a'
        r,proposals,_,_=assign_background_profiles(p,m,s)
        self.assertTrue(proposals.original_sc_cell_barcode.eq('a').all())
        self.assertNotIn('a',r.sc_cell_barcode.tolist())
        self.assertEqual(proposals.loc[proposals.retained,'sc_cell_barcode'].tolist(),r.sc_cell_barcode.tolist())

    def test_cross_boundary_and_reproducibility(self):
        p,m,s=self.fixture();p=p.iloc[:2].copy();p.x_um=[127.,129.];m.x_um=100.
        a,*_=assign_background_profiles(p,m,s,seed=18)
        b,*_=assign_background_profiles(p.iloc[::-1],m,s,seed=18)
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(a.sc_cell_barcode.nunique(),2)
        self.assertNotIn('a',a.sc_cell_barcode.tolist())

    def test_distant_reuse_allowed(self):
        p,m,s=self.fixture();p=p.iloc[:2].copy();p.x_um=[300.,600.]
        a,*_=assign_background_profiles(p,m,s.iloc[:1])
        self.assertEqual(a.sc_cell_barcode.tolist(),['a','a'])

    def test_no_capacity_does_not_change_type_or_borrow(self):
        p,m,s=self.fixture()
        with self.assertRaisesRegex(ValueError,'no feasible'):
            assign_background_profiles(p,m,s.iloc[:1])

    def test_render_conserves_budget_and_support(self):
        p,m,s=self.fixture();r,*_=assign_background_profiles(p,m,s)
        xy=np.c_[np.arange(5)*2.,np.zeros(5)]
        raw=sparse.csr_matrix([[10,0],[0,10],[4,6]])
        kwargs=dict(xy=xy,allowed=np.array([False,True,True,True,True]),raw_profiles=raw,records=r,seed=9)
        a=render_background_sources(**kwargs);b=render_background_sources(**kwargs)
        self.assertEqual(a.sum(),100);self.assertEqual(a[0].sum(),0)
        self.assertEqual((a-b).nnz,0)


if __name__=='__main__':unittest.main()
