#!/usr/bin/env python3
"""Reuse the frozen five-method benchmark on full-section real-scRNA pairs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_source_separated_method_comparison as protocol
from hd_cell_rl.experiment_protocol import paired_cluster_mean_ci,write_experiment_manifest


def summarize_supported(out, dataset):
    """Apply reference support only after inference; never filter competing cells."""
    support=pd.read_csv(dataset/'cell_reference_support.csv',dtype={'cell_id':str})
    if support.cell_id.duplicated().any():raise ValueError('duplicate physical support identity')
    frame=pd.read_csv(out/'subdiagnostic/per_cell_comparison.csv',dtype={'cell_id':str})
    frame=frame.merge(support,on='cell_id',how='left',validate='many_to_one')
    if frame.reference_supported.isna().any():raise ValueError('missing reference-support classification')
    # evaluator frames can already contain their own cell_type label; source
    # annotation is used only for post-hoc stratification, not inference.
    if 'cell_type_y' in frame:frame['cell_type']=frame.cell_type_y
    frame['evaluation_scope']=np.where(frame.reference_supported,'supported_main','missing_reference_diagnostic')
    selection_path=protocol.PATCHES
    if selection_path.exists():
        selection=pd.read_csv(selection_path)
        if 'cohort' in selection:
            frame=frame.merge(selection[['patch_id','cohort']],on='patch_id',how='left',validate='many_to_one')
            if frame.cohort.isna().any():raise ValueError('evaluated patch missing frozen cohort')
    rows=[];patch_rows=[];effects=[]
    for (scope,condition,method),g in frame.groupby(['evaluation_scope','condition','method']):
        rows.append(dict(scope=scope,condition=condition,method=method,n_cells=len(g),n_patches=g.patch_id.nunique(),
            n_cell_types=g.cell_type.nunique(),**{'mean_'+m:float(g[m].mean()) for m in protocol.METRICS},
            **{'n_finite_'+m:int(np.isfinite(g[m]).sum()) for m in protocol.METRICS}))
        for patch,h in g.groupby('patch_id'):
            patch_rows.append(dict(scope=scope,condition=condition,method=method,patch_id=patch,n_cells=len(h),
                                  **{'mean_'+m:float(h[m].mean()) for m in protocol.METRICS},
                                  **{'n_finite_'+m:int(np.isfinite(h[m]).sum()) for m in protocol.METRICS}))
    for (scope,method),g in frame.groupby(['evaluation_scope','method']):
        a=g[g.condition=='modeled_only'].set_index('cell_id')
        b=g[g.condition=='source_separated'].set_index('cell_id').loc[a.index]
        if not a.patch_id.equals(b.patch_id):raise ValueError('paired patches differ')
        for metric in protocol.METRICS:
            delta=b[metric].to_numpy()-a[metric].to_numpy()
            lo,hi=paired_cluster_mean_ci(delta,a.patch_id.to_numpy(),rng=np.random.default_rng(20260922))
            effects.append(dict(scope=scope,method=method,metric=metric,n_cells=len(a),n_patches=a.patch_id.nunique(),
                added_background_delta=float(np.nanmean(delta)),patch_bootstrap_low=lo,patch_bootstrap_high=hi))
    result=pd.DataFrame(rows)
    result[result.scope=='supported_main'].to_csv(out/'method_comparison_supported.csv',index=False)
    result[result.scope=='missing_reference_diagnostic'].to_csv(out/'method_comparison_missing_reference.csv',index=False)
    pd.DataFrame(patch_rows).to_csv(out/'per_patch_supported_comparison.csv',index=False)
    pd.DataFrame(effects).to_csv(out/'paired_background_effect_by_support.csv',index=False)
    if 'cohort' in frame:
        cohort_rows=[]
        for (cohort,scope,condition,method),g in frame.groupby(['cohort','evaluation_scope','condition','method']):
            cohort_rows.append(dict(cohort=cohort,scope=scope,condition=condition,method=method,
                n_cells=len(g),n_patches=g.patch_id.nunique(),
                **{'mean_'+m:float(g[m].mean()) for m in protocol.METRICS}))
        pd.DataFrame(cohort_rows).to_csv(out/'method_comparison_by_cohort.csv',index=False)
    frame.to_csv(out/'subdiagnostic/per_cell_with_reference_support.csv',index=False)
    coverage=support.groupby(['cell_type','reference_supported']).size().rename('whole_section_cells').reset_index()
    unique=frame[['cell_id','cell_type']].drop_duplicates()
    coverage['evaluated_core_cells']=coverage.cell_type.map(unique.cell_type.value_counts()).fillna(0).astype(int)
    coverage.to_csv(out/'evaluated_type_coverage.csv',index=False)
    # A geometry-complement uptake diagnostic should be zero for no-background.
    full=pd.read_parquet(protocol.AUDIT/'barcodes.parquet')
    components,_=protocol.assemble_components(full,protocol.REGIONAL)
    gt=pd.read_parquet(protocol.FRACTIONAL,columns=['barcode_index','cell_id','cell_coverage_fraction'])
    covered=full.matrix_col_index.isin(gt.loc[(gt.cell_id>0)&(gt.cell_coverage_fraction>0),'barcode_index']).to_numpy()
    if components['unmodeled_biological'][covered].nnz or components['modeled_cell'][~covered].nnz:
        raise ValueError('paired mask-complement guarantee failed')
    protocol.dump(out/'subdiagnostic/paired_support_checks.json',dict(covered_bins_unchanged=True,
        modeled_outside_masks=0,background_inside_masks=0,
        evaluated_supported_cells=int(unique.cell_id.isin(set(support.loc[support.reference_supported,'cell_id'])).sum()),
        evaluated_missing_reference_cells=int(unique.cell_id.isin(set(support.loc[~support.reference_supported,'cell_id'])).sum()),
        unrepresented_reference_missing_types=coverage.loc[(~coverage.reference_supported)&(coverage.evaluated_core_cells==0),'cell_type'].tolist()))
    lines=['# Real-scRNA paired method comparison','',
        f'Formal dataset: `{dataset}`. Data are stored in Bin2Cell_Validation, not this results directory.','',
        f'This comparison contains {unique.cell_id.nunique()} core cells across {frame.patch_id.nunique()} patches. '
        'Patch selection is recorded separately; this is not independent-donor validation. The main table includes only cells '
        'with an independent donor-disjoint reference. All cells, including missing-reference types, competed during inference.','',
        '`modeled_only` means no background; `source_separated` adds structured RNA only outside all modeled masks. '
        'Both arms have identical modeled-cell counts. The selected default is no background.','',
        '| Condition | Method | Core cells | Mean IoU | Fractional IoU | Gene Spearman |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in result[result.scope=='supported_main'].itertuples():
        lines.append(f'| {row.condition} | {row.method} | {row.n_cells} | {row.mean_pred_iou:.4f} | '
                     f'{row.mean_pred_fractional_iou:.4f} | {row.mean_gene_spearman_r:.4f} |')
    lines += ['', 'Missing-reference results are diagnostic only: see `method_comparison_missing_reference.csv` '
        'and `evaluated_type_coverage.csv` for which types were actually evaluated. Gene correlation is against '
        'the original generating raw scRNA profile, not capture-matched GT.','',
        'No PPO training, Smurf, or weight selection was performed. EM keeps alpha=1, beta=1.5, kappa=2, '
        'profile update damping=0. See `subdiagnostic/em_convergence.csv` for numerical convergence.','',
        'Dataset integrity is distinct from biological realism. See `data_quality_acceptance.json` and '
        '`data_quality_multiscale.csv` for the unchanged 2/8/16 µm quality checks. Background is a sensitivity '
        'model, not a validated ambient model; distant type fallbacks are recorded in '
        '`subdiagnostic/background_source_context.json`.','',
        'Old datasets and training/evolution runs are preserved. Paired uncertainty intervals resample patches '
        'and are undefined for a stratum with only one patch.']
    if 'cohort' in frame:
        lines += ['', 'Original four and twelve additional patches are reported separately in '
            '`method_comparison_by_cohort.csv`. The expression extraction halo is wider than the previous '
            'four-patch run, so historical score differences are not solely a background-source intervention. '
            'Within this run both conditions and every method use the same frozen expression contexts. '
            'The additional patches are a same-section extension, not an independent donor test.']
    checks_path=out/'subdiagnostic/evaluation_checks.json'
    if checks_path.exists():
        missing=[row for row in json.loads(checks_path.read_text()) if row['missing_core_nuclear_bins']]
        if missing:
            lines += ['', '## Native seed-retention caveat', '',
                'The methods received the same confident seed inputs, but native output filtering did not retain '
                'all seeds for the entries below. These are unmodified native results, not a fully seed-retention-matched comparison.']
            lines += [f'- {row["condition"]} / {row["method"]}: {row["missing_core_nuclear_bins"]} core nuclear bins omitted.'
                      for row in missing]
            lines += ['', 'Retained nuclear owners are unchanged. See `subdiagnostic/evaluation_checks.json`. '
                'No bins were silently backfilled and no native algorithm was changed.']
            omitted_path=out/'subdiagnostic/missing_bin2cell_core_seeds.csv'
            if omitted_path.exists():
                omitted=pd.read_csv(omitted_path)
                if len(omitted) and omitted.input_counts.eq(0).all():
                    lines += ['', 'The omitted Bin2Cell bins in `subdiagnostic/missing_bin2cell_core_seeds.csv` '
                        'all have zero input counts; the existing `min_counts=1` filter removes them.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(result.to_string(index=False),flush=True)


def audit_dataset_quality(out,dataset):
    """Distribution checks, not a gate selected using method performance."""
    from scripts.run_regional_depth_calibration import prepare_inputs
    from scripts.build_source_separated_pseudo import quality,acceptance
    import yaml
    data=prepare_inputs(protocol.AUDIT,protocol.BENCH,ROOT.parent/'Bin2Cell_Validation',
                        profile_assignment=dataset/'ground_truth_cell_assignments.csv')
    components,_=protocol.assemble_components(data['bins'],protocol.REGIONAL)
    calibration=json.loads((dataset/'calibration.json').read_text())['calibration_patches']
    bins=data['bins'].copy();bins['partition']=np.where(bins.patch_id.isin(calibration),'calibration','audit')
    cell_parts=np.where(np.isin(data['cell_patch'],calibration),'calibration','audit')
    limits=yaml.safe_load((ROOT/'configs/pseudo_components.calibration.yaml').read_text())['acceptance']
    rows=[];cells=[];checks=[]
    for condition,key in [('modeled_only','modeled_cell'),('source_separated','observed')]:
        r,c=quality(components[key],data['real'],bins,data['region'],cell_parts,data['core'],data['types'],condition,20260922)
        rows+=r;cells+=c;checks.append(dict(condition=condition,**acceptance(r,limits)))
    pd.DataFrame(rows).to_csv(out/'data_quality_multiscale.csv',index=False)
    pd.DataFrame(cells).to_csv(out/'subdiagnostic/cell_region_quality.csv',index=False)
    protocol.dump(out/'data_quality_acceptance.json',dict(checks=checks,threshold_source='unchanged pseudo_components.calibration.yaml'))
    sources=pd.read_csv(dataset/'subdiagnostic/unmodeled_sources.csv.gz')
    distant=~sources.local_context_supported
    protocol.dump(out/'subdiagnostic/background_source_context.json',dict(
        virtual_sources=len(sources),background_counts=int(sources.counts.sum()),
        distant_fallback_sources=int(distant.sum()),distant_fallback_fraction=float(distant.mean()),
        distant_fallback_count_fraction=float(sources.loc[distant,'counts'].sum()/max(1,sources.counts.sum())),
        max_nearest_cell_distance_um=float(sources.nearest_modeled_cell_um.max()),
        limitation='Beyond 72 um, nearest-cell type is a heuristic, not locally supported biological evidence.'))


def preserve_dataset_validation(out,dataset):
    """Keep the data's validation evidence and generation dependencies with it."""
    destination=dataset/'validation';destination.mkdir(exist_ok=False)
    for name in ('data_quality_acceptance.json','data_quality_multiscale.csv','evaluated_type_coverage.csv'):
        shutil.copy2(out/name,destination/name)
    shutil.copy2(out/'subdiagnostic/background_source_context.json',destination/'background_source_context.json')
    dependencies=destination/'source_dependencies';dependencies.mkdir()
    source=ROOT.parent/'Bin2Cell_Validation/scripts/simulation'
    for name in ('create_colorectal_pseudo_hd.py','gene_localization.py'):
        shutil.copy2(source/name,dependencies/name)
    write_experiment_manifest(dataset/'delivery_provenance',config=dict(condition_default='no_background',
        dataset_role='experimental_control',biological_realism_validated=False),
        inputs=dict(initial_manifest=dataset/'experiment_manifest.json',completion=dataset/'completion.json',
            assignment=dataset/'ground_truth_cell_assignments.csv',reference=dataset/'reference/reference_counts.npz',
            modeled_counts=dataset/'components/modeled_cell.npz',background_counts=dataset/'components/unmodeled_biological.npz',
            validation=destination/'data_quality_acceptance.json',localization=dependencies/'create_colorectal_pseudo_hd.py',
            gene_localization=dependencies/'gene_localization.py'),
        protocol=dict(paired_integrity_passed=True,realism_thresholds_all_passed=False,
            activation='User explicitly approved no-background experimental control despite incomplete realism.',
            no_old_data_deleted=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','internal','external','summarize','all'],default='all')
    p.add_argument('--external-methods',nargs='+',choices=['bin2cell','stcs'],default=['bin2cell','stcs'])
    p.add_argument('--runtime-root',type=Path)
    p.add_argument('--evaluation-context',type=Path,help='Frozen expanded patch selection and compact component axes.')
    args=p.parse_args();dataset=args.dataset_dir.resolve();out=args.output_dir.resolve()
    complete=json.loads((dataset/'completion.json').read_text())
    if not complete['complete'] or not complete['whole_section']:raise ValueError('whole-section generation is not complete')
    protocol.REGIONAL=dataset/'patch_components'
    protocol.REFERENCE=dataset/'reference/reference_counts.npz'
    protocol.PROFILE_ASSIGNMENT=dataset/'ground_truth_cell_assignments.csv'
    protocol.PROFILE_H5=ROOT.parent/'Bin2Cell_Validation/data/single_cell/colorectal/CRC_GSE166555_raw_tumor.h5'
    protocol.EM_MAX_ITERATIONS=50
    if args.evaluation_context is not None:
        context=args.evaluation_context.resolve(strict=True)
        protocol.BENCH=context;protocol.AUDIT=context
        protocol.REGIONAL=context/'patch_components'
        protocol.PATCHES=context/'selection/test_patches_index.csv'
        protocol.NUCLEI=context/'episode_context/nuclei_patch_context.parquet'
        protocol.FRACTIONAL=context/'inputs/fractional_gt_contributing_cells.parquet'
        protocol.AXIS_SOURCE=dataset/'no_background'
        variant_gt=context/'ground_truth'
        if variant_gt.is_dir():
            protocol.GT=variant_gt/'dominant_cell_bins.csv.gz'
            protocol.NUCLEAR=variant_gt/'dominant_nuclear_bins.csv.gz'
            protocol.FRACTIONAL_CELL_BINS=variant_gt/'fractional_cell_bins.csv.gz'
    if args.stage in ('prepare','all'):
        protocol.prepare(out,dataset_root=dataset/'patch_test_datasets',protocol_overrides=dict(
            whole_section_source=str(dataset),component_calibration='fixed first two development patches; no per-fold refit',
            role='fixed same-section patch comparison; original four plus twelve additional patches' if args.evaluation_context else 'four previously inspected development patches',
            selected_cells='all core cells in the frozen patch index; context cells compete',
            background_support='strict geometric mask complement within tissue',
            reference_scope='28 donor-disjoint types; main metrics split after inference',
            unsupported_types=complete['missing_types'],em_iteration_cap=50))
        audit_dataset_quality(out,dataset)
    if args.stage in ('internal','all'):protocol.run_internal(out)
    if args.stage in ('external','all'):protocol.run_external(out,args.external_methods,args.runtime_root)
    if args.stage in ('summarize','all'):
        protocol.summarize(out);summarize_supported(out,dataset)
        protocol.finalize(out)
        write_experiment_manifest(out/'subdiagnostic/final_provenance',config=dict(dataset=str(dataset),em_max_iterations=50),
            inputs=dict(dataset_manifest=dataset/'experiment_manifest.json',reference=protocol.REFERENCE,
                        supported_results=out/'method_comparison_supported.csv'),
            protocol=dict(no_training=True,no_weight_selection=True,data_not_stored_in_benchmark=True))
        preserve_dataset_validation(out,dataset)


if __name__=='__main__':main()
