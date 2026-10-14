#!/usr/bin/env python3
"""Three-arm development inference only. No generating profile/GT owners read.

Evaluation is a separate command, after all assignments have been sealed.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hd_cell_rl import em_assignment as em
from hd_cell_rl.em_posterior_predictive import nuclear_dirichlet_posterior, posterior_predictive_compatibility
from hd_cell_rl.experiment_protocol import write_experiment_manifest
from hd_cell_rl.patch_assignment import PatchOwnershipMerger, validate_unique_ownership
from hd_cell_rl.patch_dataset import PatchDataset
from hd_cell_rl.ppo_checkpoint import load_checkpoint_payload
from hd_cell_rl.ppo_config import load_ppo_training_config
from scripts import run_em_weight_sweep as sweep
from scripts import run_source_separated_method_comparison as protocol

ARMS=('distance_only','frozen_em','posterior_predictive')
GENERATORS=('A_current','B_library_offset')
GATE=dict(mean_iou='predictive > frozen and distance, both generators',
    patch_consistency='predictive >= frozen and distance on at least 3/4 patches each',
    bin_safety='corrected > damaged versus frozen, both generators; core-GT nonnuclear scored bins',
    numerics='all solves converged; locked nuclei exact; baseline full soft artifacts reproduced (float tolerance 1e-12)',
    failure_action='no held-out test, no weight search, no PPO')


def compare_artifact_value(new, old):
    if isinstance(new, dict):
        if new.keys()!=old.keys():raise AssertionError('metadata keys differ')
        for key in new:compare_artifact_value(new[key],old[key])
    elif isinstance(new,list):
        if len(new)!=len(old):raise AssertionError('metadata lengths differ')
        for a,b in zip(new,old,strict=True):compare_artifact_value(a,b)
    elif isinstance(new,float):np.testing.assert_allclose(new,old,atol=1e-12,rtol=1e-12)
    else:
        if new!=old:raise AssertionError('discrete metadata changed')


def generator_inference(args,generator,*,alternative='posterior_predictive'):
    if alternative not in ('posterior_predictive','reference_only'):
        raise ValueError('unsupported matched alternative')
    arms=('distance_only','frozen_em',alternative)
    diagnostic_prefix='predictive' if alternative=='posterior_predictive' else 'reference_only'
    source=args.source_dir/generator
    original=source/'benchmark/modeled_only'
    out=args.output_dir/generator
    out.mkdir();(out/'subdiagnostic').mkdir()
    context_path,=(original/'methods/nucleus_only').glob('*/evaluation_context')
    episodes,=(original/'episodes').glob('*/episodes_index.csv')
    base=replace(load_ppo_training_config(protocol.BASE),planner_enabled=False,episodes_index_path=episodes,
        nuclei_path=protocol.NUCLEI,reference_path=protocol.REFERENCE,reference_format='npz')
    settings=sweep._settings_from_checkpoint(patch_cfg=dict(load_checkpoint_payload(protocol.CHECKPOINT).get('patch_config')or{}),
        patches_index_path=protocol.PATCHES,mode='em_only',artifact_dir=out/'unused')
    cfg=replace(settings.em_assignment,save_artifacts=False,artifact_dir=None,debug_patch_ids=(),
        background_enabled=False,spatial_weight=1.,expression_weight=1.5,cell_specific_expression_enabled=True,
        cell_profile_relative_prior_strength=2.,cell_profile_update_damping=0.,
        non_nuclear_bin_filter='positive_expression',max_iterations=50)
    protocol.dump(out/'resolved_em_config.json',asdict(cfg))
    settings=replace(settings,cache_patch_contexts=True,em_assignment=replace(cfg,cell_specific_expression_enabled=False))
    target=set(sweep.load_eval_cell_ids(context_path/'per_episode.csv'))
    selected=sweep._select_patch_rows(pd.read_csv(protocol.PATCHES),target)
    dataset=PatchDataset(base_config=base,settings=settings,rng=np.random.default_rng(20260914))
    mergers={arm:PatchOwnershipMerger() for arm in arms}
    cells,bins,checks,reproduction,contrasts=[],[],[],[],[]
    try:
        for row in selected:
            context=dataset.load_patch_context(row)
            if context is None: raise ValueError('missing development context')
            inp=em.build_sparse_patch_em_input(context=context,candidate_max_distance_um=context.candidate_max_distance_um,
                non_nuclear_bin_filter='positive_expression')
            x=dataset.load_unique_patch_expression(context=context,barcode_ids=inp.barcode_ids)
            alpha,profile,reliability,nuclear_total,prior=nuclear_dirichlet_posterior(inp,cfg,x,dataset.reference_theta)
            pair_args=dict(bin_gene_counts=x,pair_bin_index=inp.pair_bin_index,candidate_cell_index=inp.candidate_cell_index)
            if alternative=='posterior_predictive':
                predicted=posterior_predictive_compatibility(**pair_args,posterior_alpha=alpha)
            else:
                theta=em._normalize_reference_theta(dataset.reference_theta,n_types=inp.ll.shape[1],
                    n_genes=x.shape[1],epsilon=cfg.epsilon)
                reference_profile=em.reference_only_cell_profile(reference_theta=theta,
                    cell_type_posterior=em.initialize_q_from_nuclear_seeds(inp),reference_prior_count=prior,epsilon=cfg.epsilon)
                predicted=em.compute_cell_expression_compatibility(**pair_args,cell_expression_profile=reference_profile,epsilon=cfg.epsilon)
            plugin=em.compute_cell_expression_compatibility(**pair_args,cell_expression_profile=profile,epsilon=cfg.epsilon)
            total=np.asarray(x.sum(1)).ravel()
            singleton=total[inp.pair_bin_index]==1
            if alternative=='posterior_predictive':
                np.testing.assert_allclose(predicted[singleton],plugin[singleton],atol=1e-12,rtol=0.)
            results={}
            for arm in arms:
                arm_cfg=replace(cfg,expression_weight=0. if arm=='distance_only' else 1.5)
                kwargs=dict(bin_gene_counts=x,reference_theta=dataset.reference_theta)
                if arm=='posterior_predictive':kwargs['fixed_cell_expression_compatibility']=predicted
                if arm=='reference_only':kwargs['use_reference_only_cell_profile']=True
                started=time.perf_counter()
                result=em.run_generalized_em(inp,arm_cfg,**kwargs)
                elapsed=time.perf_counter()-started
                result.validate()
                if not result.converged:raise AssertionError('convergence gate failed; do not select a GT iteration')
                results[arm]=result
                path=out/'methods'/arm/'em_assignment'/f'{inp.patch_id}.npz'
                em.save_em_result_npz(result,path)
                mergers[arm].add_patch(context=context,score=0.,
                    target_cell_ids=[c for c in context.core_cell_ids if c in target],
                    rows=sweep._hard_assignment_rows(context=context,result=result,target_cell_ids=target))
                checks.append(dict(generator=generator,patch_id=inp.patch_id,arm=arm,n_bins=result.n_bins,
                    n_cells=len(inp.cell_ids),converged=result.converged,iterations=len(result.iterations),
                    candidate_max_distance_um=inp.candidate_max_distance_um,n_locked=int(inp.is_nuclear_locked.sum()),
                    solver_elapsed_seconds=elapsed))
                if arm!=alternative:
                    prior_path=args.prior_dir/generator/'methods'/arm/'em_assignment'/path.name
                    with np.load(path,allow_pickle=False) as new,np.load(prior_path,allow_pickle=False) as old:
                        if set(new.files)!=set(old.files):raise AssertionError('baseline artifact fields changed')
                        all_exact=True
                        for field in new.files:
                            all_exact=all_exact and np.array_equal(new[field],old[field])
                            if field=='metadata_json':compare_artifact_value(json.loads(str(new[field])),json.loads(str(old[field])))
                            elif np.issubdtype(new[field].dtype,np.floating):
                                np.testing.assert_allclose(new[field],old[field],atol=1e-12,rtol=1e-12,err_msg=field)
                            else:np.testing.assert_array_equal(new[field],old[field],err_msg=field)
                    reproduction.append(dict(generator=generator,patch_id=inp.patch_id,arm=arm,
                        all_npz_fields_exact=all_exact,all_fields_reproduced_at_1e12_tolerance=True))
                print(generator,inp.patch_id,arm,'done',flush=True)
            frozen=results['frozen_em'];predictive=results[alternative];distance=results['distance_only']
            for field in ('cell_profile_reliability','nuclear_expression_count_total',
                          'spatial_prior','expression_confidence','is_nuclear_locked','pair_distance_um'):
                np.testing.assert_array_equal(getattr(frozen,field),getattr(predictive,field))
            if alternative=='posterior_predictive':
                np.testing.assert_array_equal(frozen.cell_expression_profile,predictive.cell_expression_profile)
            else:
                np.testing.assert_allclose(predictive.cell_expression_profile,reference_profile,atol=1e-9,rtol=1e-7)
            np.testing.assert_allclose(predicted,predictive.final_cell_expression_compatibility,atol=1e-6)
            locked_bins=np.flatnonzero(inp.is_nuclear_locked)
            member=sparse.coo_matrix((np.ones(len(locked_bins)),(inp.locked_owner_cell_index[locked_bins],locked_bins)),
                shape=(len(inp.cell_ids),len(inp.barcode_ids))).tocsr()
            nuclear=(member@x).tocsr()
            for ci,cell in enumerate(inp.cell_ids):
                cells.append(dict(generator=generator,patch_id=inp.patch_id,cell_id=cell,
                    is_core=cell in context.core_cell_ids and cell in target,nuclear_umi=nuclear_total[ci],
                    nuclear_genes=nuclear[ci].count_nonzero(),profile_reliability=reliability[ci],prior_count=prior,
                    posterior_concentration=alpha[ci].sum()))
            owner={arm:np.asarray(r.cell_ids)[r.top1_cell_index] for arm,r in results.items()}
            scales=cfg.expression_weight*inp.expression_confidence[inp.pair_bin_index]*reliability[inp.candidate_cell_index]
            plugin_score=scales*plugin;predictive_score=scales*predicted
            for bi,barcode in enumerate(inp.barcode_ids):
                lo,hi=inp.candidate_row_splits[bi:bi+2];indices=slice(lo,hi)
                nonzero=x[bi].data
                rec=dict(generator=generator,patch_id=inp.patch_id,barcode=barcode,
                    nuclear_locked=bool(inp.is_nuclear_locked[bi]),counts=total[bi],detected_genes=len(nonzero),
                    repeated_gene=bool(np.any(nonzero>1)),n_candidates=hi-lo,
                    frozen_distance_probability_l1=float(np.abs(frozen.responsibility[indices]-distance.responsibility[indices]).sum()),
                    **{diagnostic_prefix+'_distance_probability_l1':float(np.abs(predictive.responsibility[indices]-distance.responsibility[indices]).sum())})
                rec.update({a:str(v[bi]) for a,v in owner.items()});bins.append(rec)
                # Within-bin contrasts, not raw negative LL levels, determine ownership.
                def centered_range(v):return float(np.ptp(v[indices]))
                contrasts.append(dict(generator=generator,patch_id=inp.patch_id,barcode=barcode,
                    nuclear_locked=rec['nuclear_locked'],counts=total[bi],repeated_gene=rec['repeated_gene'],n_candidates=hi-lo,
                    plugin_cell_score_range=centered_range(plugin_score),
                    **{diagnostic_prefix+'_cell_score_range':centered_range(predictive_score)},
                    cell_compatibility_delta_mean=float((predicted-plugin)[indices].mean()),
                    cell_compatibility_delta_range=centered_range(predicted-plugin)))
    finally:dataset.close()
    sealed=[]
    for arm,merger in mergers.items():
        rows,conflicts=merger.finalize();validate_unique_ownership(rows,nuclear_owners=merger.nuclear_owners)
        path=out/'methods'/arm/'assignments.csv';frame=pd.DataFrame(rows);frame.to_csv(path,index=False)
        pd.DataFrame(conflicts).to_csv(out/'subdiagnostic'/f'{arm}_merge_conflicts.csv',index=False)
        sealed.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    pd.DataFrame(cells).to_csv(out/'subdiagnostic/cell_covariates.csv',index=False)
    pd.DataFrame(bins).to_csv(out/'subdiagnostic/bin_owners.csv.gz',index=False)
    pd.DataFrame(contrasts).to_csv(out/'subdiagnostic/score_contrasts.csv.gz',index=False)
    protocol.dump(out/'subdiagnostic/solver_checks.json',checks)
    protocol.dump(out/'subdiagnostic/baseline_reproduction.json',reproduction)
    protocol.dump(out/'inference_complete.json',{'assignments':sealed,'GT_profile_used':False})


def main(*,alternative='posterior_predictive'):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--prior-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    if alternative not in ('posterior_predictive','reference_only'):raise ValueError('unsupported alternative')
    arms=('distance_only','frozen_em',alternative)
    gate={k:v.replace('predictive',alternative) for k,v in GATE.items()}
    args.source_dir=args.source_dir.resolve(strict=True);args.prior_dir=args.prior_dir.resolve(strict=True)
    args.output_dir=args.output_dir.resolve();args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'subdiagnostic').mkdir()
    write_experiment_manifest(args.output_dir,config=vars(args),inputs={
        'reference':protocol.REFERENCE,'patches':protocol.PATCHES,'base':protocol.BASE,
        'frozen_prior_report':args.prior_dir/'method_comparison.csv'},protocol={
            'phase':'inference only; no GT profile or generating-profile mapping inputs',
            'arms':arms,'gate':gate,'scope':'A/B modeled-only, four existing development patches, 43 core cells',
            'settings':'alpha=1 beta=1.5 kappa=2 gamma=0; max_iterations=50 all arms; existing stopping criteria',
            'scoring':('log integrated ordered multinomial likelihood per UMI, same omitted coefficient as plugin'
                if alternative=='posterior_predictive' else 'existing plugin LL, replacing only frozen profile with q_initial @ independent reference; outer lambda unchanged'),
            'future_validation':'development only in this run; no weights, PPO or new held-out selection'})
    # Python and HDF5 safeguards supplement the deliberately GT-free call graph.
    import h5py
    original_h5=h5py.File
    def guard(path):
        value=str(path)
        if ('profile_assignment.csv' in value or 'scdesign3_library_size_ab_' in value
                or 'ground_truth_cell_to_synthetic' in value or 'scdesign3_synthetic_counts.h5' in value):
            raise RuntimeError('GT profile input forbidden during matched inference: '+value)
    def audit(event,args):
        if event=='open':guard(args[0])
    sys.addaudithook(audit)
    def checked_h5(path,*a,**kw):guard(path);return original_h5(path,*a,**kw)
    h5py.File=checked_h5
    try:
        for generator in GENERATORS:generator_inference(args,generator,alternative=alternative)
    finally:h5py.File=original_h5
    protocol.dump(args.output_dir/'inference_complete.json',{'generators':GENERATORS,'arms':arms,'GT_profile_guard_active':True})


if __name__=='__main__':main()
