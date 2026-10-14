#!/usr/bin/env python3
"""Spatially blocked regional-depth controls on existing pseudo-HD contexts."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from scipy import sparse
import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hd_cell_rl.experiment_protocol import write_experiment_manifest
from hd_cell_rl.regional_depth import (spatial_blocks,geometry_type_features,fit_regional_depth,
    fit_residual_field,draw_depth_field,interpolate_block_values)
from hd_cell_rl.pseudo_components import (sample_captured_profiles,spatial_source_kernels,
    local_type_probabilities,place_source_profiles,fit_component_budgets,combine_components)
from scripts.build_source_separated_pseudo import align_genes,region_matrix,quality,acceptance,dump
from scripts.run_pseudo_realism_experiment import load_h5,place_profiles,vec


def prepare_inputs(audit,bench,validation, *, profile_assignment=None):
    bins=pd.read_parquet(audit/'barcodes.parquet').reset_index(drop=True)
    genes=pd.read_csv(audit/'genes.csv').gene.astype(str).tolist();real=sparse.load_npz(audit/'matrices/real.npz')
    if bins.barcode.duplicated().any() or bins.matrix_col_index.duplicated().any() or real.shape!=(len(bins),len(genes)):
        raise ValueError('invalid physical barcode/gene axes')
    if len(set(genes))!=len(genes):raise ValueError('gene panel must be unique')
    gt=pd.read_parquet(bench/'inputs/fractional_gt_contributing_cells.parquet')
    meta=pd.read_csv(bench/'inputs/ground_truth_cells.csv.gz').sort_values('cell_id');cells=meta.cell_id.to_numpy(int)
    assignment=pd.read_csv(profile_assignment or bench/'assignment/ground_truth_cell_to_synthetic_scdesign3.csv').set_index('cell_id').loc[cells]
    types=assignment.cell_type.to_numpy();glob=bins.matrix_col_index.to_numpy()
    region={name:region_matrix(gt,glob,cells,fractional=f) for name,f in [('binary_cell_region',False),('fractional_region',True)]}
    area=vec(region['fractional_region'].sum(1));fullarea=gt.groupby('cell_id').cell_coverage_fraction.sum().reindex(cells).to_numpy()
    cell_patch=np.full(len(cells),'',dtype=object)
    for p in sorted(bins.patch_id.unique()):
        visible=vec(region['fractional_region'][:,bins.patch_id.eq(p).to_numpy()].sum(1))>0
        if np.any(cell_patch[visible]!=''):raise ValueError('a physical cell crosses fold contexts')
        cell_patch[visible]=p
    core=np.isin(cells,pd.read_csv(bench/'selection/source_eval/per_episode.csv').cell_id)
    if np.any(core & (cell_patch=='')):raise ValueError('missing core-cell spatial support')
    nuclei=pd.read_parquet(bench/'episode_context/nuclei_patch_context.parquet')
    labels=nuclei.cell_id.astype(int).map(assignment.cell_type.to_dict())
    if labels.isna().any():raise ValueError('nuclei have missing simulation type annotations')
    local=gt[gt.barcode_index.isin(glob)]
    nuc_cov=local.groupby('barcode_index').nuclear_fraction.sum().reindex(glob,fill_value=0).to_numpy()
    return dict(bins=bins,genes=genes,real=real,gt=gt,cells=cells,types=types,region=region,area=area,
        fullarea=fullarea,cell_patch=cell_patch,core=core,nuclei=nuclei,nucleus_types=labels.to_numpy(),
        nuclear_coverage=nuc_cov,coverage=vec(region['fractional_region'].sum(0)),assignment=assignment)


def load_profiles(data,bench,validation):
    sy,sbar,sg=load_h5(bench/'simulation/scdesign3_generator_disjoint/scdesign3_synthetic_counts.h5')
    lookup={b:i for i,b in enumerate(sbar)}
    sy=align_genes(sy[[lookup[b] for b in data['assignment'].sc_cell_barcode]],sg,data['genes'])
    raw,rbar,rg=load_h5(validation/'data/single_cell/colorectal/CRC_GSE166555_raw_tumor.h5')
    rm=pd.read_csv(bench/'simulation/scdesign3_generator_disjoint/model_reference_cell_metadata.tsv',sep='\t')
    lookup={b:i for i,b in enumerate(rbar)};raw=align_genes(raw[[lookup[b] for b in rm.Cell]],rg,data['genes'])
    module_dir=validation/'scripts/simulation';sys.path.insert(0,str(module_dir))
    spec=importlib.util.spec_from_file_location('regional_localization',module_dir/'create_colorectal_pseudo_hd.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    p0,_,_=mod.localization_probabilities(np.array(data['genes']),sy.data,sy.indices,validation/'references/gene_annotation/gencode_v38.gtf')
    membrane=mod.load_membrane_genes(validation/'references/gene_localization/human_membrane_secretory_genes_uniprot.txt')
    return sy,raw,rm,mod,p0,np.isin(data['genes'],list(membrane))


def base_source_realization(data,profiles,calibration,seed,cfg):
    sy,raw,rm,mod,p0,flags=profiles;bins=data['bins'];patch=bins.patch_id.to_numpy()
    xy=bins[['x_um','y_um']].to_numpy();covered=data['coverage']>0;rng=np.random.default_rng(seed)
    source_bins=[]
    for p in sorted(set(patch)):
        eligible=np.flatnonzero((patch==p)&~covered)
        if not len(eligible):raise ValueError('source model requires complement support in each context')
        n=min(len(eligible),max(1,int(np.ceil(len(eligible)*4/cfg['unknown_source_area_um2']))))
        source_bins.extend(rng.choice(eligible,n,replace=False))
    source_bins=np.asarray(source_bins,int);source_patch=patch[source_bins]
    names,tp=local_type_probabilities(xy[source_bins],data['nuclei'][['center_x_um','center_y_um']].to_numpy(),
        data['nucleus_types'],scale_um=cfg['type_context_scale_um'])
    types=np.array([rng.choice(names,p=q) for q in tp])
    indices=np.array([rng.choice(np.flatnonzero(rm.cell_type.to_numpy()==t)) for t in types])
    kernels=spatial_source_kernels(xy,patch,xy[source_bins],source_patch,sigma_um=cfg['unknown_sigma_um'],tissue_support=np.ones(len(bins),bool))
    weights=np.ones(len(source_bins))
    for p in sorted(set(patch)):
        keep=source_patch==p;weights[keep]*=np.sum(patch==p)/keep.sum()
    fit=fit_component_budgets(vec(data['real'].sum(1)),calibration,covered,vec(kernels.T@weights),technical_fraction=0.)
    cal_area=vec(data['region']['fractional_region'][:,calibration].sum(1))
    cal_counts=vec((data['region']['fractional_region'][:,calibration]@data['real'][calibration]).sum(1))
    pool=(cal_area>0)&(data['area']/data['fullarea']>.8)&(data['area']>=10)
    if pool.sum()<5:raise ValueError('too few complete calibration cell-regions')
    density=cal_counts/np.maximum(cal_area,1e-12);drawn=[]
    for t in data['types']:
        own=pool&(data['types']==t);use=own if own.sum()>=5 else pool
        drawn.append(rng.choice(density[use]))
    target=np.asarray(drawn)*data['fullarea']
    arguments=dict(cells=data['cells'],groups={int(c):g for c,g in data['gt'].groupby('cell_id')},
        local_by_global={int(b):i for i,b in enumerate(bins.matrix_col_index)},n_bins=len(bins),membrane_flags=flags,p0=p0,
        target_nuclear=cfg['nuclear_fraction'],membrane_strength=cfg['membrane_strength'],uniform=False,seed=seed,source_module=mod)
    proposed=sample_captured_profiles(sy,np.rint(target),np.random.default_rng(seed+1))
    modeled,_=place_profiles(proposed,**arguments)
    scale=fit['modeled_calibration_counts']/float(modeled[calibration].sum())
    model_budget=target*scale;unknown_budget=weights*fit['unknown_amplitude']
    proposed=sample_captured_profiles(sy,np.rint(model_budget),np.random.default_rng(seed+1))
    modeled,trace=place_profiles(proposed,**arguments)
    unknown_profiles=sample_captured_profiles(raw[indices],np.rint(unknown_budget),np.random.default_rng(seed+2))
    unknown,unknown_trace=place_source_profiles(unknown_profiles,kernels,np.random.default_rng(seed+3))
    return dict(modeled=modeled,unknown=unknown,model_trace=trace,unknown_trace=unknown_trace,
        model_pattern=sy,unknown_pattern=raw[indices],model_budget=model_budget,unknown_budget=unknown_budget,
        arguments=arguments,kernels=kernels,source_bins=source_bins,source_patch=source_patch,
        source_types=types,source_profile_barcodes=rm.Cell.to_numpy()[indices],fit=fit)


class UnsafeDepthExtrapolation(ValueError):
    """A declared computational guard, recorded as a failed experiment arm."""


def capture_realization(base,multiplier,seed):
    """A finite reservoir implements depth multipliers, not physical efficiency inference."""
    if not np.isfinite(multiplier).all() or np.any(multiplier<=0):raise ValueError('invalid depth multiplier')
    reservoir=max(1.,float(np.ceil(multiplier.max()*1.05)))
    if reservoir>64:raise UnsafeDepthExtrapolation(f'model requires {reservoir:g}x reservoir (>64x guard); no clipping or generation')
    probability=multiplier/reservoir;md={};ud={}
    mp=sample_captured_profiles(base['model_pattern'],np.rint(base['model_budget']*reservoir),np.random.default_rng(seed+11))
    up=sample_captured_profiles(base['unknown_pattern'],np.rint(base['unknown_budget']*reservoir),np.random.default_rng(seed+12))
    modeled,mt=place_profiles(mp,**base['arguments'],capture_probability=probability,capture_diagnostics=md)
    unknown,ut=place_source_profiles(up,base['kernels'],np.random.default_rng(seed+13),capture_probability=probability,capture_diagnostics=ud)
    return modeled,unknown,mt,ut,{'reservoir_multiplier':reservoir,'modeled':md,'unmodeled':ud}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',type=Path,required=True);parser.add_argument('--benchmark-dir',type=Path,required=True)
    parser.add_argument('--validation-repo',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/pseudo_regional_depth.calibration.yaml')
    parser.add_argument('--predict-only',action='store_true',help='Fit and audit coarse-depth predictors without generation.')
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load(args.config.read_text())
    if cfg['technical_fraction']!=0:raise ValueError('technical component is intentionally fixed off in this experiment')
    data=prepare_inputs(args.audit_dir,args.benchmark_dir,args.validation_repo)
    bins=data['bins'];real=data['real'];patch=bins.patch_id.to_numpy();xy=bins[['x_um','y_um']].to_numpy()
    grid=bins.barcode.str.extract(r's_002um_(\d+)_(\d+)').astype(int).to_numpy();names=sorted(set(patch))
    if len(names)!=4:raise ValueError('the frozen protocol requires the four existing contexts')
    folds=[('original_2plus2',names[:2],names[2:])]+[('leave_out_'+p,[q for q in names if q!=p],[p]) for p in names]
    mapping,inverse,keys,block_patch,n=spatial_blocks(grid,patch,cfg['block_width_um']//2)
    features,feature_names=geometry_type_features(xy,patch,data['nuclei'][['center_x_um','center_y_um']].to_numpy(),
        data['nucleus_types'],coverage=data['coverage'],nuclear_coverage=data['nuclear_coverage'],density_sigma_um=cfg['density_sigma_um'])
    x=(mapping@features)/n[:,None];centers=(mapping@xy)/n[:,None];mean=vec(mapping@vec(real.sum(1)))/n
    block_frame=pd.DataFrame(x,columns=feature_names);block_frame.insert(0,'patch_id',block_patch)
    block_frame['grid_row']=keys[:,1];block_frame['grid_col']=keys[:,2];block_frame['n_bins']=n;block_frame['real_mean_counts']=mean
    block_frame.to_csv(out/'block_features_and_depth.csv',index=False)
    pd.DataFrame([dict(patch_id=p,n_bins=int(np.sum(patch==p)),real_mean_counts=float(real[patch==p].sum()/np.sum(patch==p)),
        n_context_nuclei=int(np.sum((data['nuclei'].center_x_um>=xy[patch==p,0].min())&(data['nuclei'].center_x_um<=xy[patch==p,0].max())&
                                   (data['nuclei'].center_y_um>=xy[patch==p,1].min())&(data['nuclei'].center_y_um<=xy[patch==p,1].max()))),
        core_cells=int(np.sum(data['core']&(data['cell_patch']==p)))) for p in names]).to_csv(out/'patch_inventory.csv',index=False)
    bins.to_parquet(out/'barcodes.parquet',index=False);pd.Series(data['genes'],name='gene').to_csv(out/'genes.csv',index=False)
    (out/'subdiagnostic').mkdir()
    for name in ['create_colorectal_pseudo_hd.py','gene_localization.py']:
        shutil.copy2(args.validation_repo/'scripts/simulation'/name,out/'subdiagnostic'/name)
    input_paths={'barcodes':args.audit_dir/'barcodes.parquet','real':args.audit_dir/'matrices/real.npz',
        'genes':args.audit_dir/'genes.csv','geometry':args.benchmark_dir/'inputs/fractional_gt_contributing_cells.parquet',
        'nuclei':args.benchmark_dir/'episode_context/nuclei_patch_context.parquet',
        'source_profiles':args.benchmark_dir/'simulation/scdesign3_generator_disjoint/scdesign3_synthetic_counts.h5',
        'generator_metadata':args.benchmark_dir/'simulation/scdesign3_generator_disjoint/model_reference_cell_metadata.tsv',
        'generator_raw_counts':args.validation_repo/'data/single_cell/colorectal/CRC_GSE166555_raw_tumor.h5',
        'source_assignment':args.benchmark_dir/'assignment/ground_truth_cell_to_synthetic_scdesign3.csv',
        'annotation':args.validation_repo/'references/gene_annotation/gencode_v38.gtf',
        'membrane_genes':args.validation_repo/'references/gene_localization/human_membrane_secretory_genes_uniprot.txt',
        'reference':args.benchmark_dir/'reference/reference_counts_donor_disjoint.npz'}
    write_experiment_manifest(out,config=cfg,inputs=input_paths,protocol={
        'folds':[dict(id=f,calibration=c,audit=a) for f,c,a in folds], 'no_EM_RL_training':True,
        'role':'development spatial validation on previously inspected contexts, one section',
        'type_label_caveat':'existing expression-derived simulation annotations, not morphology-only independent labels',
        'capture_caveat':'reservoir probability is a sampling device; technical capture efficiency is not identified',
        'predict_only':args.predict_only,'source_snapshot_timing':'before model fitting/generation'})
    predictions=[];models={};fits=[]
    for fold,cal_patches,audit_patches in folds:
        cal_blocks=np.isin(block_patch,cal_patches)&(n==(cfg['block_width_um']//2)**2)
        models[fold]={}
        for kind,columns in [('global',[]),('geometry',list(range(3))),('composition',list(range(x.shape[1])))]:
            model=fit_regional_depth(x[:,columns],mean,n,cal_blocks,[feature_names[i] for i in columns],ridge=cfg['ridge'])
            pred=model.predict(x[:,columns]);residual=fit_residual_field(mean,pred,n,keys,cal_blocks,
                block_width_um=cfg['block_width_um'],sigma_grid_um=cfg['residual_sigma_grid_um'])
            models[fold][kind]=(model,pred,residual)
            fits.append(dict(fold=fold,kind=kind,calibration_patches=cal_patches,audit_patches=audit_patches,
                model=model.to_dict(),residual=residual))
            for p in names:
                use=block_patch==p
                row=dict(fold=fold,kind=kind,patch_id=p,partition='calibration' if p in cal_patches else 'audit',
                    real_mean_counts=float(np.average(mean[use],weights=n[use])),
                    predicted_mean_counts=float(np.average(pred[use],weights=n[use])),
                    block_mean_absolute_error=float(np.average(abs(pred[use]-mean[use]),weights=n[use])),
                    n_complete_blocks=int(np.sum(use&(n==(cfg['block_width_um']//2)**2))))
                row['predicted_to_real_ratio']=row['predicted_mean_counts']/row['real_mean_counts'];predictions.append(row)
            block_out=block_frame[['patch_id','grid_row','grid_col','n_bins','real_mean_counts']].copy()
            block_out['predicted_mean_counts']=pred;block_out['is_calibration']=cal_blocks
            block_out.to_csv(out/'subdiagnostic'/f'{fold}.{kind}.predictions.csv',index=False)
            print('FIT',fold,kind,[(r['patch_id'],round(r['predicted_to_real_ratio'],3)) for r in predictions[-4:] if r['partition']=='audit'],flush=True)
    pd.DataFrame(predictions).to_csv(out/'depth_predictions.csv',index=False);dump(out/'fitted_models.json',fits)
    if args.predict_only:return
    profiles=load_profiles(data,args.benchmark_dir,args.validation_repo)
    all_metrics=[];all_cells=[];all_checks=[];capture_rows=[]
    for fold,cal_patches,audit_patches in folds:
        calibration=np.isin(patch,cal_patches)
        for seed in cfg['seeds']:
            base=base_source_realization(data,profiles,calibration,seed,cfg)
            baseline=base['modeled']+base['unknown'];block_base=vec(mapping@vec(baseline.sum(1)))/n
            smoothed_base=interpolate_block_values(np.maximum(block_base,1e-6),xy,centers,block_patch,patch,sigma_um=cfg['interpolation_sigma_um'])
            for variant in cfg['variants']:
                kind=variant['features'];model,pred,residual=models[fold][kind]
                if kind=='global':
                    multiplier=np.ones(len(bins))
                else:
                    expected=interpolate_block_values(pred,xy,centers,block_patch,patch,sigma_um=cfg['interpolation_sigma_um'])
                    multiplier=expected/smoothed_base
                    # Single calibration-only mass anchor. No audit depth is consulted.
                    multiplier*=float(real[calibration].sum())/float(np.dot(vec(baseline.sum(1))[calibration],multiplier[calibration]))
                multiplier*=draw_depth_field(grid,patch,residual,np.random.default_rng(seed+101),
                    spatial=variant['spatial'],patch_effect=variant['patch_effect'])
                try:
                    modeled,unknown,mt,ut,diagnostics=capture_realization(base,multiplier,seed)
                except UnsafeDepthExtrapolation as error:
                    capture_rows.append(dict(fold=fold,variant=variant['name'],seed=seed,
                        status='rejected_extrapolation',reason=str(error),
                        min_depth_multiplier=float(multiplier.min()),max_depth_multiplier=float(multiplier.max())))
                    for p in audit_patches:
                        all_checks.append(dict(fold=fold,patch_id=p,variant=variant['name'],seed=seed,
                            passed=False,failed_checks=['unsafe_depth_extrapolation'],reason=str(error)))
                    print('REJECTED',fold,variant['name'],seed,str(error),flush=True)
                    continue
                technical=sparse.csr_matrix(real.shape,dtype=np.int64);observed=combine_components(modeled,unknown,technical)
                capture_rows.append(dict(fold=fold,variant=variant['name'],seed=seed,
                    min_depth_multiplier=float(multiplier.min()),max_depth_multiplier=float(multiplier.max()),**diagnostics))
                for p in audit_patches:
                    evaluation_bins=bins.copy();evaluation_bins['partition']=np.where(patch==p,'audit','excluded')
                    parts=np.where(data['cell_patch']==p,'audit','excluded')
                    rows,cells=quality(observed,real,evaluation_bins,data['region'],parts,data['core'],data['types'],variant['name'],seed)
                    if len(rows)!=5 or not all(r['partition']=='audit' for r in rows):
                        raise ValueError('audit must contain 2/8/16 um and both cell-region metric rows')
                    rows=[dict(fold=fold,patch_id=p,**r) for r in rows];cells=[dict(fold=fold,patch_id=p,**r) for r in cells]
                    for row in cells:row['cell_id']=int(data['cells'][row['cell_index']])
                    verdict=acceptance(rows,cfg['acceptance']);all_checks.append(dict(fold=fold,patch_id=p,variant=variant['name'],seed=seed,**verdict))
                    all_metrics+=rows;all_cells+=cells
                    print('AUDIT',fold,variant['name'],seed,p,'failed',len(verdict['failed_checks']),flush=True)
                if seed==cfg['seeds'][0]:
                    d=out/'realizations'/fold/variant['name'];d.mkdir(parents=True)
                    for name,value in [('observed',observed),('modeled_cell',modeled),('unmodeled_biological',unknown),
                                       ('technical_background',technical),('modeled_source_bin_counts',mt),('unmodeled_source_bin_counts',ut)]:
                        sparse.save_npz(d/(name+'.npz'),value)
                    np.save(d/'depth_multiplier.npy',multiplier.astype(np.float32))
                    pd.DataFrame({'cell_id':data['cells'],'cell_type':data['types'],'source_budget_before_reservoir':base['model_budget'],
                        'captured_in_crop_counts':vec(mt.sum(1))}).to_csv(d/'modeled_sources.csv',index=False)
                    pd.DataFrame({'source_id':[f'unmodeled_{i:06d}' for i in range(len(base['source_bins']))],
                        'patch_id':base['source_patch'],'cell_type':base['source_types'],'profile_barcode':base['source_profile_barcodes'],
                        'x_um':xy[base['source_bins'],0],'y_um':xy[base['source_bins'],1],
                        'source_budget_before_reservoir':base['unknown_budget'],'captured_counts':vec(ut.sum(1))}).to_csv(d/'unmodeled_sources.csv',index=False)
            pd.DataFrame(all_metrics).to_csv(out/'multiscale_metrics.csv',index=False)
            pd.DataFrame(all_cells).to_csv(out/'cell_region_metrics.csv',index=False)
            dump(out/'capture_diagnostics.json',capture_rows)
            dump(out/'acceptance.json',{'passed':False,'complete':False,'checks':all_checks,'thresholds':cfg['acceptance']})
    dump(out/'capture_diagnostics.json',capture_rows)
    dump(out/'acceptance.json',{'passed':all(c['passed'] for c in all_checks),'complete':True,'checks':all_checks,
        'thresholds':cfg['acceptance'],'not_a_training_admission':'development controls; no model promoted or fresh test used'})
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
