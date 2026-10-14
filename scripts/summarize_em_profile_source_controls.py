#!/usr/bin/env python3
"""Four-group profile-source contrasts; average seeds within cells, not as replicates."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hd_cell_rl.experiment_protocol import paired_cluster_mean_ci
from scripts import summarize_em_library_diagnostics as base
from scripts.run_em_profile_source_controls import ARMS,SAMPLED_ARMS,EXPECTED_ARM


def summarize(run, source, prior):
    with patch.object(base,'ARMS',ARMS):
        metrics=base.summarize(run,source)
    old=pd.read_csv(prior/'method_comparison.csv')
    joined=metrics.merge(old,on=['generator','arm'],suffixes=('_new','_old'),validate='one_to_one')
    if len(joined)!=6:
        raise AssertionError('six prior baseline evaluations must be comparable')
    checked=['pred_iou','pred_fractional_iou','pred_precision','pred_recall','gene_spearman_r']
    for metric in checked:
        np.testing.assert_allclose(joined[metric+'_new'],joined[metric+'_old'],atol=1e-12,rtol=0)
    soft_checks=[]
    for generator in ('A_current','B_library_offset'):
        for arm in ('distance_only','frozen_em','oracle_profile_diagnostic'):
            for new_path in (run/generator/'methods'/arm/'em_assignment').glob('*.npz'):
                old_path=prior/generator/'methods'/arm/'em_assignment'/new_path.name
                with np.load(new_path) as new, np.load(old_path) as old:
                    if set(new.files)!=set(old.files):
                        raise AssertionError('baseline soft artifact schemas differ')
                    for key in new.files:
                        np.testing.assert_array_equal(new[key],old[key],err_msg=f'{generator}/{arm}/{new_path.stem}/{key}')
                soft_checks.append(dict(generator=generator,arm=arm,patch_id=new_path.stem,exact_all_fields=True))
    if len(soft_checks)!=24:
        raise AssertionError('expected 24 fully reproduced baseline soft artifacts')
    pd.DataFrame(soft_checks).to_csv(run/'subdiagnostic/soft_baseline_reproduction.csv',index=False)
    groups={'current_nuclear':('frozen_em',),'sampled_same_depth':SAMPLED_ARMS,
        'expected_same_depth':(EXPECTED_ARM,),'full_oracle':('oracle_profile_diagnostic',)}
    summary=[]
    for generator,frame in metrics.groupby('generator'):
        for group,arms in groups.items():
            selected=frame[frame.arm.isin(arms)]
            if len(selected)!=len(arms):
                raise ValueError('missing predeclared arm')
            row=dict(generator=generator,group=group,n_cells=43,n_realizations=len(arms))
            for metric in checked:
                row[metric+'_mean']=float(selected[metric].mean())
                row[metric+'_min']=float(selected[metric].min())
                row[metric+'_max']=float(selected[metric].max())
                row[metric+'_seed_sd']=float(selected[metric].std()) if len(arms)>1 else np.nan
            summary.append(row)
    pd.DataFrame(summary).to_csv(run/'four_group_comparison.csv',index=False)
    cells=pd.read_csv(run/'subdiagnostic/per_cell_results.csv',dtype={'cell_id':str})
    contrasts=[]; per_cell=[]; stratified=[]
    contrast_pairs=[('sampled_minus_current','sampled_same_depth','current_nuclear'),
        ('expected_minus_sampled','expected_same_depth','sampled_same_depth'),
        ('oracle_minus_expected','full_oracle','expected_same_depth'),
        ('oracle_minus_current','full_oracle','current_nuclear')]
    for generator,frame in cells.groupby('generator'):
        pivot=frame.pivot(index=['cell_id','patch_id'],columns='arm',values='pred_iou')
        grouped=pd.DataFrame({g:pivot[list(arms)].mean(axis=1) for g,arms in groups.items()}).reset_index()
        frozen=frame[frame.arm=='frozen_em'][['cell_id','patch_id','profile_reliability','nuclear_umi']]
        grouped=grouped.merge(frozen,on=['cell_id','patch_id'],validate='one_to_one')
        grouped['generator']=generator
        grouped['high_lambda']=grouped.profile_reliability>=grouped.profile_reliability.quantile(.75)
        for name,after,before in contrast_pairs:
            delta=grouped[after]-grouped[before]
            low,high=paired_cluster_mean_ci(delta,grouped.patch_id,replicates=10000,rng=np.random.default_rng(20260916))
            contrasts.append(dict(generator=generator,contrast=name,mean_iou_delta=float(delta.mean()),
                patch_bootstrap_low=low,patch_bootstrap_high=high,n_cells=len(delta),
                improved=int((delta>1e-12).sum()),damaged=int((delta < -1e-12).sum()),unchanged=int((abs(delta)<=1e-12).sum())))
            for high_lambda,subset in grouped.groupby('high_lambda'):
                stratified.append(dict(generator=generator,contrast=name,high_lambda=bool(high_lambda),
                    n_cells=len(subset),mean_iou_delta=float((subset[after]-subset[before]).mean()),
                    min_lambda=float(subset.profile_reliability.min()),max_lambda=float(subset.profile_reliability.max())))
        telescoped=(grouped.sampled_same_depth-grouped.current_nuclear)+(grouped.expected_same_depth-grouped.sampled_same_depth)+(grouped.full_oracle-grouped.expected_same_depth)
        np.testing.assert_allclose(telescoped,grouped.full_oracle-grouped.current_nuclear,atol=1e-15)
        per_cell.append(grouped)
    pd.concat(per_cell).to_csv(run/'subdiagnostic/four_group_per_cell.csv',index=False)
    pd.DataFrame(contrasts).to_csv(run/'contrast_effects.csv',index=False)
    pd.DataFrame(stratified).to_csv(run/'high_lambda_contrasts.csv',index=False)
    pd.concat(per_cell).groupby(['generator','patch_id'])[list(groups)].mean().to_csv(run/'four_group_per_patch.csv')
    transitions=pd.read_csv(run/'bin_correction_damage.csv')
    bin_groups=[]
    for generator,frame in transitions[transitions.baseline=='frozen'].groupby('generator'):
        for group,arms in groups.items():
            if group=='current_nuclear':
                continue
            selected=frame[frame.arm.isin(arms)]
            bin_groups.append(dict(generator=generator,group=group,n_realizations=len(selected),
                n_evaluated=int(selected.n_evaluated.iloc[0]),
                mean_corrected=float(selected.corrected.mean()),min_corrected=int(selected.corrected.min()),max_corrected=int(selected.corrected.max()),
                mean_damaged=float(selected.damaged.mean()),min_damaged=int(selected.damaged.min()),max_damaged=int(selected.damaged.max())))
    pd.DataFrame(bin_groups).to_csv(run/'four_group_bin_transitions.csv',index=False)
    profile=[]
    for generator in ('A_current','B_library_offset'):
        p=pd.read_csv(run/generator/'subdiagnostic/profile_interventions.csv',dtype={'cell_id':str})
        core=pd.read_csv(run/generator/'subdiagnostic/cell_covariates.csv',dtype={'cell_id':str})
        p=p.merge(core[['patch_id','cell_id','is_core']],on=['patch_id','cell_id'],validate='many_to_one')
        np.testing.assert_allclose(p.observed_nuclear_umi,p.intervention_count_total,atol=1e-8,rtol=0)
        if (p.groupby(['patch_id','cell_id']).q_initial_sha256.nunique()!=1).any():
            raise ValueError('initial q not fixed')
        p['generator']=generator;profile.append(p)
    pd.concat(profile).to_csv(run/'subdiagnostic/all_profile_interventions.csv',index=False)
    pd.concat(profile).query('is_core').groupby(['generator','arm'])[['profile_tv_to_oracle','observed_nuclear_umi','intervention_detected_genes','profile_reliability']].mean().to_csv(run/'profile_quality.csv')
    (run/'subdiagnostic/validation.json').write_text(json.dumps(dict(
        six_prior_evaluations_reproduced=True,baseline_soft_artifacts_exact=len(soft_checks),matched_nuclear_budgets=True,q_initialization_fixed=True,
        lambda_and_prior_fixed=True,arithmetic_contrasts_reconcile=True,
        seed_statistics='five profile redraws on same cells, not independent tissue replicates',
        causality='sequential path contrasts are not additive independent biological causes'),indent=2)+'\n')
    retained=run/'subdiagnostic/analysis_source'/Path(__file__).name
    shutil.copy2(Path(__file__),retained)
    (retained.with_suffix('.sha256')).write_text(hashlib.sha256(retained.read_bytes()).hexdigest()+'\n')
    return pd.DataFrame(summary)


def write_notebook(run):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    run=run.resolve()
    nb=nbformat.v4.new_notebook()
    nb.cells=[nbformat.v4.new_markdown_cell('# Same-depth EM profile controls\n\n## tl;dr\n'
        'See `report.md` for findings. These are diagnostic controls using generating profiles, not deployable methods.'),
        nbformat.v4.new_markdown_cell('## Context & Methods\n'
        'Same observed A/B modeled-only data and 43 core cells. Nuclear UMI budgets, initial q, lambda, prior strength and weights remain fixed. '
        'Only the profile initializer changes. Five predeclared multinomial seeds are averaged within each cell before patch resampling.\n\n'
        '### Key Assumptions\nExpected counts are fractional N times true generating proportions, not an observable count realization. '
        'q updates may diverge downstream. Four existing development patches do not establish donor generalization. '
        'Numerical contrasts telescope but are not independent biological error components.'),
        nbformat.v4.new_markdown_cell('## Data\nManifest and source archive retain exact inputs and code.'),
        nbformat.v4.new_code_cell('from pathlib import Path\nimport json\nimport pandas as pd\n'+f'run=Path({str(run)!r})\n'+
        'json.loads((run/"subdiagnostic/validation.json").read_text())'),
        nbformat.v4.new_markdown_cell('## Results\n### Four-group comparison\nCompact exact-value tables preserve the small comparison and seed ranges.'),
        nbformat.v4.new_code_cell('pd.read_csv(run/"four_group_comparison.csv")[["generator","group","n_realizations","pred_iou_mean","pred_iou_min","pred_iou_max","pred_iou_seed_sd"]].round(6)'),
        nbformat.v4.new_markdown_cell('### Paired contrasts\nBootstrap resamples whole patches, not cells or profile redraws.'),
        nbformat.v4.new_code_cell('pd.read_csv(run/"contrast_effects.csv").round(6)'),
        nbformat.v4.new_markdown_cell('### Corrected/damaged core nonnuclear bins\nLocked and filtered zero-count bins are excluded; official IoU uses its original evaluator.'),
        nbformat.v4.new_code_cell('pd.read_csv(run/"four_group_bin_transitions.csv").round(2)'),
        nbformat.v4.new_markdown_cell('## Takeaways\nRead `report.md` for interpretation, convergence checks and the next diagnostic. '
        'No production weights or PPO model were trained. Oracle and matched profile controls must not enter formal rankings.')]
    nb.metadata.kernelspec=dict(name='python3',display_name='Python diagnostic environment',language='python')
    km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    try:
        NotebookClient(nb,km=km,timeout=300,resources={'metadata':{'path':str(run)}}).execute()
    finally:
        if km.has_kernel:km.shutdown_kernel(now=True)
    nbformat.validate(nb);nbformat.write(nb,run/'profile_controls.ipynb')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--prior-diagnostic-dir',type=Path,required=True)
    p.add_argument('--notebook',action='store_true')
    a=p.parse_args()
    print(summarize(a.run_dir.resolve(),a.source_dir.resolve(),a.prior_diagnostic_dir.resolve()).to_string(index=False))
    if a.notebook:write_notebook(a.run_dir.resolve())


if __name__=='__main__':main()
