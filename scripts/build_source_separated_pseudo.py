#!/usr/bin/env python3
"""Build and audit three-source pseudo-HD controls without fitting EM or RL.

Compact bin x gene outputs retain global barcode indices. No real expression
vector is replayed. This is a calibrated-budget sensitivity model, not a claim
to have identified the actual biological origins of mask-complement expression.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import ks_2samp, spearmanr
import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hd_cell_rl.experiment_protocol import write_experiment_manifest
from hd_cell_rl.pseudo_components import (sample_captured_profiles, spatial_source_kernels,
    local_type_probabilities, place_source_profiles, fit_component_budgets, combine_components)
from hd_cell_rl.pseudo_realism import aggregate_bins
from scripts.run_pseudo_realism_experiment import load_h5, place_profiles, vec


def dump(path, value):
    path.write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n')


def align_genes(matrix, source_genes, genes):
    lookup={str(g):i for i,g in enumerate(genes)}
    indices=[(i,lookup[str(g)]) for i,g in enumerate(source_genes) if str(g) in lookup]
    a,b=zip(*indices)
    transfer=sparse.csr_matrix((np.ones(len(a)),(a,b)),shape=(len(source_genes),len(genes)))
    return (matrix@transfer).astype(np.int64).tocsr()


def region_matrix(gt, global_ids, cell_ids, *, fractional):
    bins={int(b):i for i,b in enumerate(global_ids)}; cells={int(c):i for i,c in enumerate(cell_ids)}
    g=gt[gt.barcode_index.isin(bins) & gt.cell_id.isin(cells) & (gt.cell_coverage_fraction>0)]
    return sparse.csr_matrix((g.cell_coverage_fraction.to_numpy() if fractional else np.ones(len(g)),
        (g.cell_id.map(cells),g.barcode_index.map(bins))),shape=(len(cells),len(bins)))


def metric_row(a,b,**labels):
    a=a.tocsr();b=b.tocsr();a.eliminate_zeros();b.eliminate_zeros()
    x=vec(a.sum(1));y=vec(b.sum(1));g=np.diff(a.indptr);h=np.diff(b.indptr)
    cv=lambda v:float(np.std(v)/max(np.mean(v),1e-12))
    return dict(**labels,n_regions=len(x),counts_mean=float(x.mean()),real_counts_mean=float(y.mean()),
        counts_median=float(np.median(x)),real_counts_median=float(np.median(y)),
        genes_mean=float(g.mean()),real_genes_mean=float(h.mean()),genes_median=float(np.median(g)),
        real_genes_median=float(np.median(h)),zero_fraction=float(np.mean(x==0)),real_zero_fraction=float(np.mean(y==0)),
        counts_cv=cv(x),real_counts_cv=cv(y),counts_ks=float(ks_2samp(x,y).statistic),
        genes_ks=float(ks_2samp(g,h).statistic),counts_mean_ratio=float(x.mean()/max(y.mean(),1e-12)),
        genes_mean_ratio=float(g.mean()/max(h.mean(),1e-12)))


def quality(matrix,real,bins,region,cell_parts,core,cell_types,label,seed):
    grid=bins.barcode.str.extract(r's_002um_(\d+)_(\d+)').astype(int).to_numpy()
    patch=pd.Categorical(bins.patch_id,sorted(bins.patch_id.unique())).codes
    rows=[]; cell_rows=[]
    for part in ['calibration','audit']:
        mask=bins.partition.to_numpy()==part
        for width in [1,4,8]:
            a,keys=aggregate_bins(matrix[mask],grid[mask],patch[mask],width)
            b,_=aggregate_bins(real[mask],grid[mask],patch[mask],width)
            if not a.shape[0]:continue
            row=metric_row(a,b,variant=label,seed=seed,partition=part,scale_um=2*width,grain='spatial_block')
            lookup={tuple(k):i for i,k in enumerate(keys)}
            pairs=[(i,lookup[tuple(k+[0,0,1])]) for i,k in enumerate(keys) if tuple(k+[0,0,1]) in lookup]
            if pairs:
                i,j=np.asarray(pairs).T
                for prefix,x in [('sim',a),('real',b)]:
                    t=vec(x.sum(1));norm=np.sqrt(vec(x.power(2).sum(1)))
                    row[prefix+'_neighbor_cosine']=float(np.mean(vec(x[i].multiply(x[j]).sum(1))/np.maximum(norm[i]*norm[j],1e-12)))
                    row[prefix+'_neighbor_depth_corr']=float(spearmanr(t[i],t[j]).statistic) if min(np.std(t[i]),np.std(t[j]))>0 else np.nan
            rows.append(row)
        selected=(cell_parts==part)&core
        for grain,mapping in region.items():
            a=(mapping@matrix)[selected];b=(mapping@real)[selected]
            if a.shape[0]:rows.append(metric_row(a,b,variant=label,seed=seed,partition=part,scale_um=0,grain=grain))
        a=region['binary_cell_region']@matrix;b=region['binary_cell_region']@real
        for i in np.flatnonzero(selected):
            cell_rows.append(dict(variant=label,seed=seed,partition=part,cell_index=int(i),cell_type=cell_types[i],
                counts=float(a[i].sum()),real_counts=float(b[i].sum()),genes=int(a[i].nnz),real_genes=int(b[i].nnz)))
    return rows,cell_rows


def acceptance(rows,limits):
    failures=[]
    for row in rows:
        if row['partition']!='audit':continue
        scope=f"{row['grain']}:{row['scale_um']}um"
        for key in ['counts_mean_ratio','genes_mean_ratio']:
            if not limits['mean_ratio_min']<=row[key]<=limits['mean_ratio_max']:failures.append(scope+':'+key)
        for key in ['counts_ks','genes_ks']:
            if row[key]>limits['distribution_ks_max']:failures.append(scope+':'+key)
        if abs(row['zero_fraction']-row['real_zero_fraction'])>limits['zero_fraction_absolute_error_max']:
            failures.append(scope+':zero_fraction')
        if row['grain']=='binary_cell_region':
            ratio=row['counts_cv']/max(row['real_counts_cv'],1e-12)
            if not limits['cell_cv_ratio_min']<=ratio<=limits['cell_cv_ratio_max']:failures.append(scope+':counts_cv')
        if row['grain']=='spatial_block':
            a=row.get('sim_neighbor_depth_corr',np.nan);b=row.get('real_neighbor_depth_corr',np.nan)
            if not np.isfinite(a+b) or abs(a-b)>limits['neighbor_correlation_absolute_error_max']:
                failures.append(scope+':neighbor_depth_correlation')
    return {'passed':not failures,'failed_checks':failures}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-dir',type=Path,required=True);p.add_argument('--benchmark-dir',type=Path,required=True)
    p.add_argument('--validation-repo',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/pseudo_components.calibration.yaml')
    args=p.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load(args.config.read_text());audit=args.audit_dir;bench=args.benchmark_dir;v=args.validation_repo
    bins=pd.read_parquet(audit/'barcodes.parquet');genes=pd.read_csv(audit/'genes.csv').gene.astype(str).tolist()
    real=sparse.load_npz(audit/'matrices/real.npz');cal=bins.partition.eq('calibration').to_numpy()
    xy=bins[['x_um','y_um']].to_numpy();patch=bins.patch_id.to_numpy();glob=bins.matrix_col_index.to_numpy()
    if len(set(bins.barcode))!=len(bins) or real.shape!=(len(bins),len(genes)):raise ValueError('unmatched unique axes')
    # These selected contexts are inside tissue. Require an explicit assumption in the manifest.
    tissue=np.ones(len(bins),bool)
    gt=pd.read_parquet(bench/'inputs/fractional_gt_contributing_cells.parquet')
    meta=pd.read_csv(bench/'inputs/ground_truth_cells.csv.gz').sort_values('cell_id');cells=meta.cell_id.to_numpy(int)
    assignment=pd.read_csv(bench/'assignment/ground_truth_cell_to_synthetic_scdesign3.csv').set_index('cell_id').loc[cells]
    types=assignment.cell_type.to_numpy();groups={int(c):g for c,g in gt.groupby('cell_id')}
    region={name:region_matrix(gt,glob,cells,fractional=f) for name,f in [('binary_cell_region',False),('fractional_region',True)]}
    coverage=vec(region['fractional_region'].sum(0))>0
    area=vec(region['fractional_region'].sum(1));cal_area=vec(region['fractional_region'][:,cal].sum(1))
    full_area=gt.groupby('cell_id').cell_coverage_fraction.sum().reindex(cells).to_numpy()
    parts=np.where(cal_area>0,'calibration','audit');core=np.isin(cells,pd.read_csv(bench/'selection/source_eval/per_episode.csv').cell_id)
    regional=vec((region['fractional_region']@real).sum(1))
    pool=(cal_area>0)&(area/full_area>.8)&(area>=10)
    if pool.sum()<5:raise ValueError('not enough complete calibration regions for empirical density')
    densities=regional/np.maximum(area,1e-12)
    print('calibration regions',int(pool.sum()),'median density',float(np.median(densities[pool])),flush=True)
    sy,sbar,sg=load_h5(bench/'simulation/scdesign3_generator_disjoint/scdesign3_synthetic_counts.h5')
    sy=sy[[{b:i for i,b in enumerate(sbar)}[b] for b in assignment.sc_cell_barcode]];sy=align_genes(sy,sg,genes)
    raw,rbar,rg=load_h5(v/'data/single_cell/colorectal/CRC_GSE166555_raw_tumor.h5')
    rm=pd.read_csv(bench/'simulation/scdesign3_generator_disjoint/model_reference_cell_metadata.tsv',sep='\t')
    rlookup={b:i for i,b in enumerate(rbar)};raw=align_genes(raw[[rlookup[b] for b in rm.Cell]],rg,genes)
    module_dir=v/'scripts/simulation';sys.path.insert(0,str(module_dir))
    kernel_path=module_dir/'create_colorectal_pseudo_hd.py'
    spec=importlib.util.spec_from_file_location('source_localization',kernel_path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    p0,_,_=mod.localization_probabilities(np.array(genes),sy.data,sy.indices,v/'references/gene_annotation/gencode_v38.gtf')
    membrane=mod.load_membrane_genes(v/'references/gene_localization/human_membrane_secretory_genes_uniprot.txt');flags=np.isin(genes,list(membrane))
    nuclei=pd.read_parquet(bench/'episode_context/nuclei_patch_context.parquet');nmap=assignment.cell_type.to_dict()
    typed=nuclei[nuclei.cell_id.astype(int).isin(nmap)];centers=typed[['center_x_um','center_y_um']].to_numpy()
    center_types=typed.cell_id.astype(int).map(nmap).to_numpy()
    bg=pd.read_csv(audit/'tissue_background_panel_ids/panel_gene_background.csv').groupby('gene').off_tissue_counts.sum().reindex(genes,fill_value=0).to_numpy(float)
    bins.to_parquet(out/'barcodes.parquet',index=False);pd.Series(genes,name='gene').to_csv(out/'genes.csv',index=False)
    (out/'subdiagnostic').mkdir();shutil.copy2(kernel_path,out/'subdiagnostic/localization_source.py')
    shutil.copy2(module_dir/'gene_localization.py',out/'subdiagnostic/gene_localization.py')
    all_rows=[];all_cells=[];checks=[];budgets=[];source_reports=[]
    for baseline,filename in [('source_high_depth','source_high_depth.npz'),('global_thinning','depth_uniform.npz')]:
        rows,cr=quality(sparse.load_npz(audit/'matrices'/filename),real,bins,region,parts,core,types,baseline,0)
        all_rows+=rows;all_cells+=cr
    for seed in cfg['seeds']:
        rng=np.random.default_rng(seed)
        source_bins=[]
        for patch_id in sorted(set(patch)):
            eligible=np.flatnonzero((patch==patch_id)&~coverage&tissue)
            n=max(1,int(np.ceil(len(eligible)*4/cfg['unknown_source_area_um2'])))
            source_bins.extend(rng.choice(eligible,size=min(n,len(eligible)),replace=False))
        source_bins=np.asarray(source_bins,int);source_patch=patch[source_bins]
        names,tp=local_type_probabilities(xy[source_bins],centers,center_types,scale_um=cfg['type_context_scale_um'])
        stypes=np.array([rng.choice(names,p=q) for q in tp])
        unknown_indices=np.array([rng.choice(np.flatnonzero(rm.cell_type.to_numpy()==t)) for t in stypes])
        modeled_raw_indices=np.array([rng.choice(np.flatnonzero(rm.cell_type.to_numpy()==t)) for t in types])
        sampled_density=[]
        for t in types:
            own=pool&(types==t);eligible=own if own.sum()>=5 else pool
            sampled_density.append(float(rng.choice(densities[eligible])))
        for variant in cfg['variants']:
            label=variant['name'];vrng=np.random.default_rng(seed)
            kernels=spatial_source_kernels(xy,patch,xy[source_bins],source_patch,
                sigma_um=variant['sigma_um'],tissue_support=tissue)
            weights=np.ones(len(source_bins))
            for patch_id in sorted(set(patch)):
                keep=source_patch==patch_id;weights[keep]*=np.sum(patch==patch_id)/keep.sum()
            intensity=vec(kernels.T@weights)
            fit=fit_component_budgets(vec(real.sum(1)),cal,coverage,intensity,technical_fraction=variant['technical_fraction'])
            if not variant['unknown']:
                fit['unknown_amplitude']=0.;fit['modeled_calibration_counts']=float(real[cal&coverage].sum())
            patterns=sy if variant['profile']=='scdesign3' else raw[modeled_raw_indices]
            density=np.asarray(sampled_density) if variant['density']=='empirical' else np.full(len(cells),np.mean(densities[pool]))
            target=density*full_area
            profiles=sample_captured_profiles(patterns,np.rint(target),vrng)
            modeled,trace=place_profiles(profiles,cells,groups,{int(b):i for i,b in enumerate(glob)},len(bins),flags,p0,
                cfg['nuclear_fraction'],cfg['membrane_strength'],False,seed,mod)
            # Only calibration aggregate rescales the source budgets. Audit counts are unused.
            scale=fit['modeled_calibration_counts']/float(modeled[cal].sum())
            profiles=sample_captured_profiles(patterns,np.rint(target*scale),np.random.default_rng(seed+1))
            modeled,trace=place_profiles(profiles,cells,groups,{int(b):i for i,b in enumerate(glob)},len(bins),flags,p0,
                cfg['nuclear_fraction'],cfg['membrane_strength'],False,seed,mod)
            unknown_profiles=sample_captured_profiles(raw[unknown_indices],
                np.rint(weights*fit['unknown_amplitude']),np.random.default_rng(seed+2))
            unknown,unknown_trace=place_source_profiles(unknown_profiles,kernels,np.random.default_rng(seed+3))
            ntechnical=int(round(len(bins)*fit['technical_counts_per_bin']))
            technical_profiles=sample_captured_profiles(sparse.csr_matrix(bg[None,:]),[ntechnical],np.random.default_rng(seed+4))
            technical_kernel=sparse.csr_matrix(np.full((1,len(bins)),1/len(bins)))
            technical,_=place_source_profiles(technical_profiles,technical_kernel,np.random.default_rng(seed+5))
            combined=combine_components(modeled,unknown,technical)
            rows,cr=quality(combined,real,bins,region,parts,core,types,label,seed);all_rows+=rows;all_cells+=cr
            verdict=acceptance(rows,cfg['acceptance']);checks.append(dict(variant=label,seed=seed,**verdict))
            budgets.append(dict(variant=label,seed=seed,modeled_scale=scale,**fit))
            total=vec(combined.sum(1))
            source_reports.append(dict(variant=label,seed=seed,modeled_counts=int(modeled.sum()),
                unknown_counts=int(unknown.sum()),technical_counts=int(technical.sum()),total_counts=int(combined.sum()),
                unknown_inside_masks_fraction=float(unknown[coverage].sum()/max(unknown.sum(),1)),
                modeled_outside_masks_counts=int(modeled[~coverage].sum()),
                positive_support_BA=float((np.mean(total[coverage]>0)+np.mean(total[~coverage]==0))/2)))
            if seed==cfg['seeds'][0]:
                directory=out/'variants'/label;directory.mkdir(parents=True)
                for name,x in [('observed',combined),('modeled_cell',modeled),('unmodeled_biological',unknown),
                               ('technical_background',technical),('modeled_source_bin_counts',trace),
                               ('unmodeled_source_bin_counts',unknown_trace),('modeled_captured_profiles',profiles),
                               ('unmodeled_captured_profiles',unknown_profiles)]:sparse.save_npz(directory/(name+'.npz'),x)
                pd.DataFrame({'cell_id':cells,'cell_type':types,'captured_full_cell_budget':vec(profiles.sum(1)),
                    'captured_in_crop_counts':vec(trace.sum(1)),'area_fraction_in_crop':area/full_area}).to_csv(directory/'modeled_sources.csv',index=False)
                pd.DataFrame({'source_id':[f'unmodeled_{i:06d}' for i in range(len(source_bins))],
                    'source_kind':'unmodeled_biological','cell_type':stypes,'profile_barcode':rm.Cell.to_numpy()[unknown_indices],
                    'patch_id':source_patch,'x_um':xy[source_bins,0],'y_um':xy[source_bins,1],
                    'captured_budget':vec(unknown_profiles.sum(1))}).to_csv(directory/'unmodeled_sources.csv',index=False)
                dump(directory/'acceptance.json',verdict)
            print(label,seed,'passed',verdict['passed'],'failed checks',len(verdict['failed_checks']),flush=True)
            pd.DataFrame(all_rows).to_csv(out/'multiscale_metrics.csv',index=False)
    pd.DataFrame(all_cells).to_csv(out/'cell_region_metrics.csv',index=False)
    pd.DataFrame(budgets).to_csv(out/'calibrated_budgets.csv',index=False)
    pd.DataFrame(source_reports).to_csv(out/'component_counts.csv',index=False)
    dump(out/'acceptance.json',{'passed':all(c['passed'] for c in checks),'checks':checks,
        'thresholds':cfg['acceptance'],'role':'diagnostic audit, not untouched test',
        'not_a_training_admission':'no generator selected or independently validated by this sensitivity grid'})
    report=['# Source-separated pseudo-HD audit','',
        'No EM/RL fitting, reward changes or shape changes. Real expression is not replayed.',
        'Budgets use only calibration aggregates and calibration-region density distributions.',
        'Unknown profiles use generator-side scRNA and nearby modeled types; technical amplitude is hypothetical.',
        'All selected contexts are assumed tissue. Kernels are crop-normalized, not whole-slide transports.',
        'Mask-complement expression is not identified as purely biological by these data.',
        'Binary cell-region sums can double-count mixed bins; fractional summaries and traced source counts are separate.',
        'No tool scores enter acceptance. These previously inspected areas are not new held-out tests.','',
        '| Variant | Repeats passed | Total repeats |','|---|---:|---:|']
    for variant in cfg['variants']:
        selected=[c for c in checks if c['variant']==variant['name']]
        report.append(f"| {variant['name']} | {sum(c['passed'] for c in selected)} | {len(selected)} |")
    report+=['','See `multiscale_metrics.csv`, `cell_region_metrics.csv` and `component_counts.csv` for measured outcomes.',
        'Use `barcodes.parquet.matrix_col_index` to align compact matrices back to the global HD axis.',
        'Source matrices are diagnostics; unknown sources must not be silently converted to nuclear GT cells.']
    (out/'report.md').write_text('\n'.join(report)+'\n')
    # Inspectable execution record for the data-quality audit; loads saved metrics only.
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'}},'cells':[
        {'cell_type':'markdown','metadata':{},'source':['# Source-separated pseudo-HD audit\n','See report.md for assumptions; all matrices are bin x gene.']},
        {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
         'source':['from pathlib import Path\n','import pandas as pd\n',
                   "pd.read_csv('multiscale_metrics.csv').query(\"partition == 'audit'\")\n"]}]}
    dump(out/'analysis.ipynb',notebook)
    write_experiment_manifest(out,config=cfg,inputs={'gene_panel':audit/'genes.csv','barcodes':audit/'barcodes.parquet',
        'real_calibration_audit':audit/'matrices/real.npz','fractional_geometry':bench/'inputs/fractional_gt_contributing_cells.parquet',
        'generator_profiles':bench/'simulation/scdesign3_generator_disjoint/scdesign3_synthetic_counts.h5',
        'generator_cells':bench/'simulation/scdesign3_generator_disjoint/model_reference_cell_metadata.tsv',
        'generator_raw_counts':v/'data/single_cell/colorectal/CRC_GSE166555_raw_tumor.h5',
        'gene_annotation':v/'references/gene_annotation/gencode_v38.gtf',
        'membrane_gene_list':v/'references/gene_localization/human_membrane_secretory_genes_uniprot.txt',
        'technical_profile':audit/'tissue_background_panel_ids/panel_gene_background.csv',
        'reference':bench/'reference/reference_counts_donor_disjoint.npz','external_localization':kernel_path,
        'external_gene_localization':module_dir/'gene_localization.py'},
        protocol={'role':'calibration/audit generator sensitivity; no fresh held-out test','no_EM_RL_training':True,
            'partition':dict(bins.groupby('patch_id').partition.first()),'cell_region_gt_warning':'same synthetic masks in real and pseudo; not real-cell GT',
            'source_components':['modeled_cell','unmodeled_biological','technical_background'],
            'tissue_support':'explicit assumption: selected contexts are tissue; whole-slide export requires real tissue support',
            'external_kernel_sha256':hashlib.sha256(kernel_path.read_bytes()).hexdigest()})
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
