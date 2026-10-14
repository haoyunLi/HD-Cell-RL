#!/usr/bin/env python3
"""Same-location, exact-count pseudo/real diagnostics; no generator/inference changes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hd_cell_rl.pseudo_count_diagnostics import summarize_matched,integer_csr,rarefy_exact
from hd_cell_rl.pseudo_realism import aggregate_bins
from hd_cell_rl.reward import compute_reference_distribution
from hd_cell_rl.experiment_protocol import write_experiment_manifest
from scripts.run_regional_depth_calibration import prepare_inputs


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n')


def spatial_summary(matrix,patch,regions,distances,component,variant):
    total=np.asarray(matrix.sum(1)).ravel();mean=float(total.mean());rows=[]
    for name,keep in regions.items():
        if keep.any():rows.append(dict(patch_id=patch,variant=variant,component=component,region=name,n_bins=int(keep.sum()),
            mean_counts=float(total[keep].mean()),fraction_counts=float(total[keep].sum()/max(total.sum(),1)),
            relative_to_patch_mean=float(total[keep].mean()/max(mean,1e-12)),zero_fraction=float((total[keep]==0).mean())))
    for lo,hi in [(0,4),(4,8),(8,12),(12,20),(20,np.inf)]:
        keep=(distances>=lo)&(distances<hi)
        if keep.any():rows.append(dict(patch_id=patch,variant=variant,component=component,region=f'distance_{lo}_{hi}_um',n_bins=int(keep.sum()),
            mean_counts=float(total[keep].mean()),fraction_counts=float(total[keep].sum()/max(total.sum(),1)),
            relative_to_patch_mean=float(total[keep].mean()/max(mean,1e-12)),zero_fraction=float((total[keep]==0).mean())))
    return rows


def analyze_task(task):
    label,p,source,selected,real,grid,cellmap,refcols,theta,markers,seeds,region,distances=task
    sim=real.copy() if source is None else integer_csr(sparse.load_npz(source)[selected])
    results=[];evidence=[];space=[]
    if source is not None:
        space+=spatial_summary(real,p,region,distances,'observed','real')
        space+=spatial_summary(sim,p,region,distances,'observed',label)
        if source.name=='observed.npz':
            for name in ['modeled_cell','unmodeled_biological']:
                component=integer_csr(sparse.load_npz(source.parent/(name+'.npz'))[selected])
                space+=spatial_summary(component,p,region,distances,name,label)
    for width in [1,4,8,0]:
        if width:
            a,keys=aggregate_bins(real,grid,np.zeros(len(grid),int),width)
            b,bkeys=aggregate_bins(sim,grid,np.zeros(len(grid),int),width)
            if not np.array_equal(keys,bkeys):raise ValueError('physical aggregate axes differ')
        else:a=cellmap@real;b=cellmap@sim;keys=None
        grain='binary_core_cell_region' if not width else 'spatial_block';scale=width*2
        a=integer_csr(a);b=integer_csr(b)
        if keys is not None and source is not None:
            lookup={tuple(k):i for i,k in enumerate(keys)}
            edges=[(i,lookup[tuple(k+v)]) for i,k in enumerate(keys) for v in [[0,0,1],[0,1,0]] if tuple(k+v) in lookup]
            if edges:
                i,j=np.asarray(edges).T
                for name,x in [('real',a),(label,b)]:
                    n=np.asarray(x.sum(1)).ravel();rho=float(spearmanr(n[i],n[j]).statistic) if min(n[i].std(),n[j].std())>0 else np.nan
                    space.append(dict(patch_id=p,variant=name,component='observed',region=f'neighbor_{scale}um',n_bins=len(keys),
                                      neighbor_pairs=len(i),depth_spearman=rho))
        for panel,columns in [('all_genes',None),('reference_genes',refcols)]:
            ar=a if columns is None else a[:,columns];bs=b if columns is None else b[:,columns]
            for seed in seeds:
                r,s=ar,bs
                if source is None:
                    # Independent half-depth rarefactions are a Monte Carlo null,
                    # not independent biological samples or new synthetic benchmark data.
                    n=np.asarray(ar.sum(1)).ravel()//2
                    r=rarefy_exact(ar,n,np.random.default_rng(seed+10000));s=rarefy_exact(ar,n,np.random.default_rng(seed+20000))
                rows,ev=summarize_matched(r,s,seed,theta if columns is not None else None,markers if columns is not None else None)
                labels=dict(variant=label,patch_id=p,grain=grain,scale_um=scale,panel=panel,seed=seed)
                results += [dict(**labels,**row) for row in rows];evidence.append(dict(**labels,**ev))
    return results,evidence,space


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-dir',type=Path,default=ROOT/'runs/benchmarks/pseudo_realism_20260908T012500Z')
    p.add_argument('--benchmark-dir',type=Path,default=ROOT/'runs/benchmarks/em_donor_disjoint_unseen_patches_20260903T192649Z')
    p.add_argument('--regional-dir',type=Path,default=ROOT/'runs/benchmarks/regional_depth_blocked_20260909/experiment')
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--seeds',type=int,nargs='+',default=[20260914,20260915,20260916])
    args=p.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);(out/'subdiagnostic').mkdir()
    data=prepare_inputs(args.audit_dir,args.benchmark_dir,None);bins=data['bins'];real=integer_csr(data['real'])
    genes=data['genes'];patches=sorted(bins.patch_id.unique())
    reference=args.benchmark_dir/'reference/reference_counts_donor_disjoint.npz'
    ref=np.load(reference,allow_pickle=False);lookup={g:i for i,g in enumerate(genes)}
    valid=np.array([g in lookup for g in ref['genes']]);refcols=np.array([lookup[g] for g in ref['genes'][valid]])
    theta=compute_reference_distribution(ref['reference_counts'][:,valid],epsilon=1e-8)
    others=(theta.sum(0)[None,:]-theta)/(len(theta)-1)
    specificity=np.log(theta/np.maximum(others,1e-30));markers=np.zeros(theta.shape,dtype=np.float64);marker_rows=[]
    for k,t in enumerate(ref['cell_types']):
        eligible=np.flatnonzero((theta[k]>=1e-5)&(specificity[k]>=np.log(2)))
        chosen=eligible[np.argsort(-specificity[k,eligible],kind='stable')[:20]];markers[k,chosen]=1
        for g in chosen:marker_rows.append(dict(cell_type=str(t),gene=str(ref['genes'][valid][g]),reference_probability=float(theta[k,g]),log_specificity=float(specificity[k,g])))
    pd.DataFrame(marker_rows).to_csv(out/'subdiagnostic/reference_marker_sets.csv',index=False)
    pd.Series(genes,name='gene').to_csv(out/'genes.csv',index=False);bins.to_parquet(out/'barcodes.parquet',index=False)
    legacy=['source_high_depth','depth_uniform','regenerated_current','uniform_cell_localization','membrane_no_outer_bias']
    regional=['uniform_budget','geometry_mean','composition_mean','composition_spatial']
    sources={name:args.audit_dir/'matrices'/(name+'.npz') for name in legacy};tasks=[];skipped=[];inputs={
        'real':args.audit_dir/'matrices/real.npz','barcodes':args.audit_dir/'barcodes.parquet','genes':args.audit_dir/'genes.csv',
        'reference':reference,'geometry':args.benchmark_dir/'inputs/fractional_gt_contributing_cells.parquet',
        'nuclei':args.benchmark_dir/'episode_context/nuclei_patch_context.parquet',
        'core_cells':args.benchmark_dir/'selection/source_eval/per_episode.csv'}
    for name,path in sources.items():
        if not path.exists():raise FileNotFoundError(path)
        inputs[name]=path
    axes=pd.read_parquet(args.regional_dir/'barcodes.parquet');rgenes=pd.read_csv(args.regional_dir/'genes.csv').gene.astype(str).tolist()
    if not np.array_equal(axes.barcode,bins.barcode) or not np.array_equal(axes.matrix_col_index,bins.matrix_col_index) or rgenes!=genes:
        raise ValueError('regional artifact axes differ from audit axes')
    xy=bins[['x_um','y_um']].to_numpy();distance=cKDTree(data['nuclei'][['center_x_um','center_y_um']].to_numpy()).query(xy)[0]
    grid=bins.barcode.str.extract(r's_002um_(\d+)_(\d+)').astype(int).to_numpy()
    inventory=[]
    for patch in patches:
        selected=np.flatnonzero(bins.patch_id.eq(patch));core=data['core']&(data['cell_patch']==patch)
        cellmap=data['region']['binary_cell_region'][core][:,selected]
        coverage=data['coverage'][selected]>0;nuclear=data['nuclear_coverage'][selected]>0
        region={'nuclear_support':nuclear,'cell_nonnuclear_support':coverage&~nuclear,'mask_complement':~coverage}
        if not np.all(sum(v.astype(int) for v in region.values())==1):raise ValueError('geometry categories not exhaustive/disjoint')
        inventory.append(dict(patch_id=patch,n_bins=len(selected),n_core_cell_regions=int(core.sum()),real_counts=int(real[selected].sum())))
        local=dict(sources)
        for name in regional:
            path=args.regional_dir/'realizations'/('leave_out_'+patch)/name/'observed.npz'
            if not path.exists():skipped.append(dict(patch_id=patch,variant='regional_'+name,reason='prior extrapolation rejected; no generated matrix'));continue
            local['regional_'+name]=path;inputs[patch+'_'+name]=path
        local['real_half_depth_null']=None
        for name,path in local.items():
            tasks.append((name,patch,path,selected,real[selected],grid[selected],cellmap,refcols,theta,markers,args.seeds,region,distance[selected]))
    pd.DataFrame(inventory).to_csv(out/'inventory.csv',index=False);dump(out/'skipped.json',skipped)
    config=dict(seeds=args.seeds,workers=args.workers,legacy=legacy,regional=regional,matching='same physical region, min(real_counts,sim_counts), exact without replacement',
        panels={'all_genes':len(genes),'reference_genes':len(refcols)},scales_um=[2,8,16],core_cell_regions=True,
        marker_definition={'n_per_type':20,'min_theta':1e-5,'min_fold_over_other_types':2,'min_matched_counts_for_signal':20})
    write_experiment_manifest(out,config=config,inputs=inputs,protocol={'purpose':'diagnostic only; observed real depth conditioning is not independent validation',
        'scope':'four previously inspected development patches; no new test donor',
        'no_training':True,'no_generator_change':True,'partial_spatial_blocks':'excluded using current aggregate_bins',
        'source_seed':'regional retained seed 20260909 only; three seeds measure rarefaction variability',
        'gene_panel_warning':'reference metrics are separately rematched on common reference genes',
        'spatial_caveat':'synthetic mask regions are proxies, not real-cell GT; loaded nuclei define distance',
        'cell_region_warning':'binary core masks may share mixed bins; aggregation is diagnostic, not unique ownership'})
    results=[];evidence=[];space=[]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        jobs={executor.submit(analyze_task,t):(t[0],t[1]) for t in tasks}
        for job in as_completed(jobs):
            r,e,s=job.result();results+=r;evidence+=e;space+=s
            print('DONE',*jobs[job],len(r),flush=True)
    keys=['variant','patch_id','grain','scale_um','panel','seed']
    pd.DataFrame(results).sort_values(keys+['count_band']).to_csv(out/'matched_metrics.csv',index=False)
    pd.DataFrame(evidence).sort_values(keys).to_csv(out/'support_and_depth.csv',index=False)
    pd.DataFrame(space).drop_duplicates().sort_values(['variant','patch_id','component','region']).to_csv(out/'spatial_metrics.csv',index=False)
    total=pd.DataFrame(results);summary=total[total.count_band=='1-inf'].groupby(keys[:-1]).mean(numeric_only=True).reset_index()
    summary.to_csv(out/'per_patch_comparison.csv',index=False)
    dump(out/'completion.json',{'complete':True,'variant_patch_tasks':len(tasks),'matched_evaluations':len(evidence),
        'skipped_variant_patches':len(skipped),'same_counts_checks_passed':all(e['matched_exact_counts'] for e in evidence),
        'observations_are_development_diagnostics_not_training_data':True})
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
