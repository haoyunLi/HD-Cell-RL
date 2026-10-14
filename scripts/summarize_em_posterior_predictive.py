#!/usr/bin/env python3
"""Matched core-cell metrics, predeclared gate and distance-collapse diagnostics."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hd_cell_rl.experiment_protocol import paired_cluster_mean_ci
from scripts import run_source_separated_method_comparison as protocol
from scripts.run_em_posterior_predictive import ARMS,GENERATORS,GATE


def summarize(out,source,prior):
    diag=out/'subdiagnostic'
    arms=tuple(json.loads((out/'inference_complete.json').read_text()).get('arms',ARMS))
    if arms not in (ARMS,('distance_only','frozen_em','reference_only')):raise ValueError('unsupported matched run arms')
    alternative=arms[-1]
    prefix='predictive' if alternative=='posterior_predictive' else alternative
    gate=json.loads((out/'experiment_manifest.json').read_text())['evaluation_protocol']['gate']
    patch_map=pd.read_csv(protocol.BENCH/'selection/source_eval/per_episode.csv',dtype={'cell_id':str})[['cell_id','patch_id']]
    if patch_map.cell_id.duplicated().any():raise ValueError('core cell belongs to multiple scored patches')
    core_patch=patch_map.set_index('cell_id').patch_id.to_dict()
    gt=pd.read_csv(protocol.GT,dtype={'cell_id':str,'barcode':str},usecols=['cell_id','barcode'])
    if gt.barcode.duplicated().any():raise ValueError('dominant GT ownership duplicated')
    truth=gt.set_index('barcode').cell_id
    previous=pd.read_csv(prior/'method_comparison.csv')
    all_cells=[];metrics=[];all_bins=[];comparisons=[];strata=[];transitions=[];collapse=[];mechanism=[];gates=[];attenuation=[]
    for generator in GENERATORS:
        directory=out/generator
        cov=pd.read_csv(directory/'subdiagnostic/cell_covariates.csv',dtype={'cell_id':str})
        cov=cov[cov.is_core].drop(columns='generator')
        groups=[]
        for arm in arms:
            path,=(directory/'evaluations').glob(arm+'_*/per_episode.csv')
            frame=pd.read_csv(path,dtype={'cell_id':str}).drop(columns='patch_id',errors='ignore')
            frame=frame.merge(patch_map,on='cell_id',validate='one_to_one')
            if len(frame)!=43 or not np.isfinite(frame[list(protocol.METRICS)]).all().all():raise ValueError('unmatched/nonfinite core metrics')
            frame=frame.merge(cov,on=['cell_id','patch_id'],validate='one_to_one')
            frame['generator']=generator;frame['arm']=arm;groups.append(frame)
            means={m:float(frame[m].mean()) for m in protocol.METRICS}
            metrics.append(dict(generator=generator,arm=arm,n_cells=43,**means))
            if arm!=alternative:
                old=previous[(previous.generator==generator)&(previous.arm==arm)].iloc[0]
                for m in means:np.testing.assert_allclose(means[m],old[m],atol=1e-12,rtol=0.)
        cells=pd.concat(groups);all_cells.append(cells)
        pivot=cells.pivot(index=['cell_id','patch_id'],columns='arm',values='pred_iou').reset_index()
        joined=pivot.merge(cov,on=['cell_id','patch_id'],validate='one_to_one')
        perpatch=pivot.groupby('patch_id')[list(arms)].mean()
        for baseline in arms[:2]:
            delta=joined[alternative]-joined[baseline]
            low,high=paired_cluster_mean_ci(delta,joined.patch_id,rng=np.random.default_rng(20260916))
            comparisons.append(dict(generator=generator,baseline=baseline,mean_iou_delta=float(delta.mean()),
                patch_bootstrap_low=low,patch_bootstrap_high=high,n_improved_cells=int((delta>1e-12).sum()),
                n_damaged_cells=int((delta< -1e-12).sum()),n_nondecreasing_patches=int((perpatch[alternative]>=perpatch[baseline]-1e-12).sum())))
        for feature in ('nuclear_umi','nuclear_genes','profile_reliability'):
            cuts=pd.qcut(joined[feature],4,duplicates='drop')
            for label,group in joined.groupby(cuts,observed=True):
                strata.append(dict(generator=generator,feature=feature,quartile=str(label),n_cells=len(group),
                    feature_min=float(group[feature].min()),feature_max=float(group[feature].max()),
                    frozen_iou=float(group.frozen_em.mean()),**{prefix+'_iou':float(group[alternative].mean())},
                    distance_iou=float(group.distance_only.mean()),delta_vs_frozen=float((group[alternative]-group.frozen_em).mean())))
        low_cut=joined.nuclear_umi.quantile(.25);high_cut=joined.profile_reliability.quantile(.75)
        subset=joined[(joined.nuclear_umi<=low_cut)&(joined.profile_reliability>=high_cut)]
        strata.append(dict(generator=generator,feature='low_depth_AND_high_lambda',quartile='fixed marginal quartiles',n_cells=len(subset),
            feature_min=None,feature_max=None,frozen_iou=float(subset.frozen_em.mean()),
            **{prefix+'_iou':float(subset[alternative].mean())},distance_iou=float(subset.distance_only.mean()),
            delta_vs_frozen=float((subset[alternative]-subset.frozen_em).mean())))
        bins=pd.read_csv(directory/'subdiagnostic/bin_owners.csv.gz',dtype={a:str for a in arms})
        bins['gt_cell_id']=bins.barcode.map(truth)
        bins['gt_core']=bins.gt_cell_id.map(core_patch)==bins.patch_id
        all_bins.append(bins)
        scoped=bins[bins.gt_core&~bins.nuclear_locked].copy()
        if scoped.barcode.duplicated().any():raise ValueError('core-bin denominator double counted')
        for baseline in arms[:2]:
            before=scoped[baseline]==scoped.gt_cell_id;after=scoped[alternative]==scoped.gt_cell_id
            transitions.append(dict(generator=generator,baseline=baseline,n_evaluated=len(scoped),
                corrected=int((~before&after).sum()),damaged=int((before&~after).sum()),
                both_correct=int((before&after).sum()),both_wrong=int((~before&~after).sum())))
        for population,group in [('core_gt_nonnuclear',scoped),('all_patch_local_nonnuclear',bins[~bins.nuclear_locked])]:
            for arm,arm_prefix in [('frozen_em','frozen'),(alternative,prefix)]:
                collapse.append(dict(generator=generator,population=population,arm=arm,n_bins=len(group),
                    owner_agreement_distance=float((group[arm]==group.distance_only).mean()),
                    mean_probability_l1_distance=float(group[arm_prefix+'_distance_probability_l1'].mean())))
        scores=pd.read_csv(directory/'subdiagnostic/score_contrasts.csv.gz')
        scores=scores.merge(bins[['patch_id','barcode','gt_core']],on=['patch_id','barcode'],validate='one_to_one')
        scores=scores[scores.gt_core&~scores.nuclear_locked]
        for label,group in [('all_core_nonnuclear',scores),('one_umi',scores[scores.counts==1]),
                ('all_singleton_genes_multi_umi',scores[(scores.counts>1)&~scores.repeated_gene]),
                ('repeated_gene',scores[scores.repeated_gene])]:
            mechanism.append(dict(generator=generator,group=label,n_bins=len(group),
                mean_plugin_cell_score_range=float(group.plugin_cell_score_range.mean()),
                **{'mean_'+prefix+'_cell_score_range':float(group[prefix+'_cell_score_range'].mean())},
                mean_cell_compatibility_change=float(group.cell_compatibility_delta_mean.mean())))
        centered_old=[];centered_new=[];expression_ranges=[]
        selected=set(zip(scoped.patch_id,scoped.barcode))
        for path in (directory/'methods/frozen_em/em_assignment').glob('*.npz'):
            with np.load(path,allow_pickle=False) as old,np.load(directory/'methods'/alternative/'em_assignment'/path.name,allow_pickle=False) as new:
                # Includes type/cell mixing and final q; unlike the isolated
                # lambda*L_cell term this does not ignore cancellation by L_type.
                spatial=np.log(np.maximum(old['spatial_prior'].astype(float),1e-12))
                old_expression=old['final_combined_score'].astype(float)-spatial
                new_expression=new['final_combined_score'].astype(float)-spatial
                for bi,barcode in enumerate(old['barcode_ids'].astype(str)):
                    if (path.stem,barcode) not in selected:continue
                    lo,hi=old['candidate_row_splits'][bi:bi+2]
                    a=old_expression[lo:hi];b=new_expression[lo:hi]
                    centered_old.extend(a-a.mean());centered_new.extend(b-b.mean())
                    expression_ranges.append((np.ptp(a),np.ptp(b)))
        a=np.asarray(centered_old);b=np.asarray(centered_new)
        scale=float(np.dot(a,b)/np.dot(a,a))
        delta=b-a;residual=b-scale*a
        attenuation.append(dict(generator=generator,n_core_bins=len(scoped),n_pairs=len(a),
            mean_old_total_expression_range=float(np.mean(expression_ranges,axis=0)[0]),
            mean_new_total_expression_range=float(np.mean(expression_ranges,axis=0)[1]),
            best_uniform_score_scale=scale,
            change_energy_not_explained_by_uniform_scaling=float(np.dot(residual,residual)/np.dot(delta,delta)),
            interpretation='descriptive final-score fit only, not an additional weight experiment'))
        c=[r for r in comparisons if r['generator']==generator]
        tr=next(r for r in transitions if r['generator']==generator and r['baseline']=='frozen_em')
        gates.append(dict(generator=generator,beats_both_means=all(r['mean_iou_delta']>0 for r in c),
            patch_consistency=all(r['n_nondecreasing_patches']>=3 for r in c),
            corrected_exceeds_damaged=tr['corrected']>tr['damaged']))
    frames={'method_comparison':pd.DataFrame(metrics),'paired_effects':pd.DataFrame(comparisons),
        'depth_reliability_strata':pd.DataFrame(strata),'bin_correction_damage':pd.DataFrame(transitions),
        'distance_collapse':pd.DataFrame(collapse),prefix+'_mechanism':pd.DataFrame(mechanism),
        'expression_attenuation':pd.DataFrame(attenuation)}
    cells=pd.concat(all_cells);cells.to_csv(diag/'per_cell_results.csv',index=False)
    pd.concat(all_bins).to_csv(diag/'bin_evaluation.csv.gz',index=False)
    cells.groupby(['generator','patch_id','arm'])[list(protocol.METRICS)].mean().to_csv(out/'per_patch_comparison.csv')
    for name,frame in frames.items():frame.to_csv(out/(name+'.csv'),index=False)
    checks=pd.concat([pd.read_json(out/g/'subdiagnostic/solver_checks.json') for g in GENERATORS])
    checks.to_csv(diag/'solver_checks.csv',index=False)
    passed=all(all(v for k,v in g.items() if k!='generator') for g in gates) and checks.converged.all()
    protocol.dump(out/'development_gate.json',dict(predeclared=gate,per_generator=gates,
        passed=bool(passed),eligible_for_future_holdout=bool(passed),new_holdout_run=False,n_solves=len(checks),
        all_converged=bool(checks.converged.all()),GT_profile_inference=False))
    retained=diag/'analysis_source';retained.mkdir(exist_ok=True)
    for name in ('scripts/summarize_em_posterior_predictive.py','scripts/evaluate_em_posterior_predictive.py'):
        shutil.copy2(ROOT/name,retained/Path(name).name)
    protocol.dump(retained/'checksums.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in retained.glob('*.py')})
    print(frames['method_comparison'].to_string(index=False))
    print(frames['bin_correction_damage'].to_string(index=False))
    print('DEVELOPMENT GATE:',bool(passed))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','source-dir','prior-dir'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();summarize(a.run_dir.resolve(strict=True),a.source_dir.resolve(strict=True),a.prior_dir.resolve(strict=True))


if __name__=='__main__':main()
