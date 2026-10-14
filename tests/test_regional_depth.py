import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy import sparse

from hd_cell_rl.regional_depth import (spatial_blocks, fit_regional_depth, fit_residual_field,
    draw_depth_field, interpolate_block_values, geometry_type_features)
from hd_cell_rl.pseudo_components import place_source_profiles
from scripts.run_pseudo_realism_experiment import place_profiles


class RegionalDepthTests(unittest.TestCase):
    def test_global_model_is_calibration_mean_without_audit_leakage(self):
        x=np.zeros((6,0));y=np.array([10,20,30,40,1e9,np.nan]);keep=np.array([1,1,1,1,0,0],bool)
        model=fit_regional_depth(x,y,np.ones(6),keep,[])
        np.testing.assert_allclose(model.predict(x),25,rtol=1e-7)
        y[4:]=0
        other=fit_regional_depth(x,y,np.ones(6),keep,[])
        self.assertEqual(model.to_dict(),other.to_dict())

    def test_conditional_depth_predicts_synthetic_composition(self):
        x=np.linspace(-1,1,40)[:,None];y=20*np.exp(x[:,0]);keep=np.arange(40)<30
        model=fit_regional_depth(x,y,np.ones(40)*64,keep,['density'],ridge=.001)
        self.assertLess(np.mean(abs(model.predict(x)[~keep]/y[~keep]-1)),.02)
        self.assertGreater(model.coefficient[0],0)

    def test_residual_fit_does_not_read_holdout_depth(self):
        keys=np.array([[p,r,c] for p in range(3) for r in range(4) for c in range(4)])
        y=np.exp(np.linspace(2,4,len(keys)));mu=np.full(len(keys),20.);n=np.full(len(keys),64)
        keep=keys[:,0]<2
        first=fit_residual_field(y,mu,n,keys,keep,block_width_um=16)
        y[~keep]=np.nan;mu[~keep]=np.nan
        self.assertEqual(first,fit_residual_field(y,mu,n,keys,keep,block_width_um=16))

    def test_fit_is_row_order_independent(self):
        rng=np.random.default_rng(31);x=rng.normal(size=(60,2));y=20*np.exp(x@[.2,.3])
        n=rng.integers(10,65,size=60);keep=np.arange(60)<40;order=rng.permutation(60)
        a=fit_regional_depth(x,y,n,keep,['a','b'])
        b=fit_regional_depth(x[order],y[order],n[order],keep[order],['a','b'])
        np.testing.assert_allclose(a.predict(x),b.predict(x),rtol=1e-7)

    def test_extrapolation_rejected_without_allocating_profiles(self):
        from scripts.run_regional_depth_calibration import capture_realization,UnsafeDepthExtrapolation
        with self.assertRaises(UnsafeDepthExtrapolation):
            capture_realization({},np.array([1.,100.]),1)

    def test_single_audit_patch_has_all_five_metric_grains(self):
        from scripts.build_source_separated_pseudo import quality
        grid=np.array([[r,c] for r in range(8) for c in range(8)])
        bins=pd.DataFrame({'barcode':[f's_002um_{r:05}_{c:05}' for r,c in grid],
                           'patch_id':['a']*64,'partition':['audit']*64})
        x=sparse.csr_matrix(np.column_stack((np.arange(64)%5,np.ones(64))))
        region=sparse.csr_matrix(np.ones((1,64)))
        rows,cells=quality(x,x,bins,{'binary_cell_region':region,'fractional_region':region},
                           np.array(['audit']),np.array([True]),np.array(['x']),'identity',1)
        self.assertEqual(len(rows),5);self.assertEqual(len(cells),1)
        self.assertEqual({r['partition'] for r in rows},{'audit'})
        self.assertTrue(all(r['counts_mean_ratio']==1 and r['genes_mean_ratio']==1 for r in rows))

    def test_block_mapping_keeps_physical_patch_identity(self):
        grid=np.array([[r,c] for r in range(4) for c in range(4)]*2)
        m,inv,keys,patch,n=spatial_blocks(grid,['a']*16+['b']*16,2)
        self.assertEqual(m.shape,(8,32));np.testing.assert_array_equal(n,4)
        self.assertEqual(len(set(zip(patch,keys[:,1],keys[:,2]))),8)

    def test_source_budgets_and_generation_ignore_audit_counts(self):
        from scripts.run_regional_depth_calibration import base_source_realization
        cells=np.arange(10);groups=[];rr=[];cc=[]
        for c in cells:
            columns=np.arange((c//5)*90+(c%5)*12,(c//5)*90+(c%5+1)*12)
            rr.extend([c]*12);cc.extend(columns)
            groups.append(pd.DataFrame({'cell_id':c,'barcode_index':columns,'cell_coverage_fraction':1.,
                'nuclear_fraction':[1.]+[0.]*11,'cytoplasm_fraction':[0.]+[1.]*11,'membrane_fraction':0.}))
        region=sparse.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(10,180))
        coverage=np.asarray(region.sum(0)).ravel();cal=np.arange(180)<90
        bins=pd.DataFrame({'matrix_col_index':np.arange(180),'patch_id':['a']*90+['b']*90,
                           'x_um':np.arange(180)*2.,'y_um':np.zeros(180)})
        nuclei=pd.DataFrame({'center_x_um':[np.mean(bins.x_um[np.array(cc)[np.array(rr)==c]]) for c in cells],
                             'center_y_um':np.zeros(10)})
        real=sparse.csr_matrix(np.column_stack((np.where(coverage>0,100,5),np.ones(180))))
        data=dict(bins=bins,coverage=coverage,nuclei=nuclei,nucleus_types=np.array(['t']*10),real=real,
            region={'fractional_region':region},area=np.ones(10)*12,fullarea=np.ones(10)*12,
            types=np.array(['t']*10),cells=cells,gt=pd.concat(groups))
        profiles=(sparse.csr_matrix(np.ones((10,2))*100),sparse.csr_matrix([[100,100]]),
            pd.DataFrame({'Cell':['source'],'cell_type':['t']}),
            SimpleNamespace(nuclear_kernel=lambda g:(g.nuclear_fraction.to_numpy(),None)),
            np.array([.2,.2]),np.array([False,False]))
        cfg=dict(unknown_source_area_um2=64,type_context_scale_um=24,unknown_sigma_um=8,
                 nuclear_fraction=.25,membrane_strength=1.)
        a=base_source_realization(data,profiles,cal,41,cfg)
        changed=real.toarray();changed[~cal]*=10000;data['real']=sparse.csr_matrix(changed)
        b=base_source_realization(data,profiles,cal,41,cfg)
        self.assertEqual(a['fit'],b['fit'])
        for key in ['model_budget','unknown_budget']:np.testing.assert_array_equal(a[key],b[key])
        for key in ['modeled','unknown']:self.assertEqual((a[key]-b[key]).nnz,0)

    def test_spatial_field_reproducible_and_interpolation_never_crosses_patches(self):
        grid=np.array([[r,c] for r in range(12) for c in range(12)]*2);patch=np.array(['a']*144+['b']*144)
        params={'within_patch_log_sd':.5,'patch_log_sd':.2,'sigma_um':4}
        a=draw_depth_field(grid,patch,params,np.random.default_rng(5))
        b=draw_depth_field(grid,patch,params,np.random.default_rng(5))
        np.testing.assert_array_equal(a,b);self.assertTrue(np.all(a>0))
        self.assertAlmostEqual(a[:144].mean(),1);self.assertAlmostEqual(a[144:].mean(),1)
        x=interpolate_block_values([1,100],[[0,0],[0,0]],[[0,0],[0,0]],['a','b'],['a','b'],sigma_um=4)
        np.testing.assert_array_equal(x,[1,100])

    def test_source_capture_respects_shared_bin_probabilities_and_trace(self):
        profiles=sparse.csr_matrix([[500,500],[500,500]]);kernels=sparse.csr_matrix([[.5,.5],[.5,.5]])
        a,ta=place_source_profiles(profiles,kernels,np.random.default_rng(2))
        b,tb=place_source_profiles(profiles,kernels,np.random.default_rng(2),capture_probability=[1,1])
        self.assertEqual((a-b).nnz,0);self.assertEqual((ta-tb).nnz,0)
        diag={};x,trace=place_source_profiles(profiles,kernels,np.random.default_rng(2),capture_probability=[1,0],capture_diagnostics=diag)
        self.assertEqual(x[1].sum(),0)
        np.testing.assert_array_equal(np.asarray(x.sum(1)).ravel(),np.asarray(trace.sum(0)).ravel())
        self.assertEqual(diag['captured_counts']+diag['capture_rejected_counts'],2000)
        with self.assertRaises(ValueError):place_source_profiles(profiles,kernels,np.random.default_rng(2),capture_probability=[2,0])

    def test_compartment_capture_preserves_old_behavior_and_conservation(self):
        group=pd.DataFrame({'barcode_index':[0,1,2],'cell_coverage_fraction':[1,1,1],
            'nuclear_fraction':[1,0,0],'cytoplasm_fraction':[0,1,0],'membrane_fraction':[0,0,1]})
        args=dict(profiles=sparse.csr_matrix([[100,100]]),cells=[1],groups={1:group},local_by_global={0:0,1:1},
            n_bins=2,membrane_flags=np.array([False,True]),p0=np.array([.2,.2]),target_nuclear=.25,
            membrane_strength=1.,uniform=False,seed=1,source_module=SimpleNamespace(nuclear_kernel=lambda g:(np.array([1.,0,0]),None)))
        a,ta=place_profiles(**args);b,tb=place_profiles(**args,capture_probability=[1,1])
        self.assertEqual((a-b).nnz,0);self.assertEqual((ta-tb).nnz,0)
        diag={};z,t=place_profiles(**args,capture_probability=[0,0],capture_diagnostics=diag)
        self.assertEqual(z.sum(),0);self.assertEqual(t.sum(),0)
        self.assertEqual(diag['proposed_counts'],diag['outside_crop_counts']+diag['capture_rejected_counts'])


if __name__=='__main__':unittest.main()
