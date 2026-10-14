#!/usr/bin/env python3
"""Independent artifact checks and an executed, compact result companion."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hd_cell_rl.provenance import file_fingerprint
from scripts import run_source_separated_method_comparison as protocol
from scripts.run_em_posterior_predictive import ARMS,GENERATORS


def notebook(run):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    arms=tuple(json.loads((run/'inference_complete.json').read_text()).get('arms',ARMS))
    alternative=arms[-1]
    prefix='predictive' if alternative=='posterior_predictive' else alternative
    gate=json.loads((run/'development_gate.json').read_text())
    effects=pd.read_csv(run/'paired_effects.csv')
    headline='; '.join(f"{r.generator}: mean IoU change versus frozen EM {r.mean_iou_delta:+.6f}"
        for r in effects[effects.baseline=='frozen_em'].itertuples())
    mechanism=('The nuclear Dirichlet posterior models uncertainty around a nuclear profile, not a nuclear-to-cytoplasmic shift. '
        if alternative=='posterior_predictive' else
        'Reference-only freezes q_initial @ independent scRNA reference, omitting direct nuclear counts from the profile. '
        'Actual nuclear counts still initialize q and determine outer lambda; q updates each iteration. ')
    md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
    nb=nbformat.v4.new_notebook(cells=[
        md(f'# {alternative} EM development experiment\n\n## tl;dr\n'+headline+
           f'. Development gate passed: {gate["passed"]}. No new holdout, weight search, or PPO was started.'),
        md('## Context & Methods\nSame four development patches and 43 core cells per A/B condition. '
           'Alpha=1, beta=1.5, kappa=2, profile update damping=0. Only cell compatibility changed.\n\n'
           '### Key Assumptions\nA/B share geometry and are not independent tissue replicates. '
           +mechanism+
           'Inference used observed counts and independent scRNA only; GT profiles entered the separate evaluation phase. '
           'Whole-patch bootstrap is descriptive with only four patches.'),
        md('## Data\nSaved official evaluations, merged assignments and per-cell diagnostics. '
           'Exact commands, source archives, and input fingerprints are retained beside this notebook.'),
        code('from pathlib import Path\nimport json\nimport pandas as pd\n'+f'run=Path({str(run)!r})\n'+
             'metrics=pd.read_csv(run/"method_comparison.csv")\n'
             'cells=pd.read_csv(run/"subdiagnostic/per_cell_results.csv")\n'
             'assert cells.groupby(["generator","arm"]).size().eq(43).all()\n'
             'gate=json.loads((run/"development_gate.json").read_text())\n'
             f'assert gate["passed"] == {gate["passed"]!r}\n'),
        md('## Results\n### Same-population method comparison\nSix rows; exact-value tables are sufficient for this compact companion.'),
        code('metrics[["generator","arm","pred_iou","pred_fractional_iou","gene_spearman_r"]].round(6)'),
        md('### Paired changes\nMean IoU changes and whole-patch bootstrap intervals; not cell-independent confidence intervals.'),
        code('pd.read_csv(run/"paired_effects.csv").round(6)'),
        md('### Corrected and damaged bins\nOnly scored core-GT nonnuclear bins. This denominator differs from per-cell IoU.'),
        code('pd.read_csv(run/"bin_correction_damage.csv")'),
        md('### Low-depth and high-reliability cells\nGroups fixed from the baseline nuclear measurements. '
           'The low-depth AND high-lambda intersection contains zero cells in both datasets.'),
        code('strata=pd.read_csv(run/"depth_reliability_strata.csv")\n'
             'pd.concat([strata[strata.feature=="nuclear_umi"].groupby("generator").head(1),'
             'strata[strata.feature=="profile_reliability"].groupby("generator").tail(1)])'
             f'[["generator","feature","n_cells","frozen_iou","{prefix}_iou","delta_vs_frozen"]].round(6)'),
        md('### Distance collapse\nCompare hard owner agreement and soft responsibility distance against the geometry-only control.'),
        code('collapse=pd.read_csv(run/"distance_collapse.csv")\n'
             'collapse[collapse.population=="core_gt_nonnuclear"].drop(columns="population").round(6)'),
        md('## Takeaways\nDevelopment evidence is not independent validation. See report.md for the interpretation and next decision. '
           'This notebook is executed; browser-level layout inspection is not available in this environment.')])
    km=KernelManager(kernel_name='python3')
    km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    try:NotebookClient(nb,km=km,timeout=300,resources={'metadata':{'path':str(run)}}).execute()
    finally:
        if km.has_kernel:km.shutdown_kernel(now=True)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code':
            assert cell.execution_count is not None
            assert not any(o.output_type=='error' for o in cell.outputs)
    nbformat.write(nb,run/(alternative+'.ipynb'))
    return sum(c.cell_type=='code' for c in nb.cells)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','source-dir','profile-dir'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();run=args.run_dir.resolve(strict=True);source=args.source_dir.resolve(strict=True)
    profiles=args.profile_dir.resolve(strict=True);diag=run/'subdiagnostic'
    arms=tuple(json.loads((run/'inference_complete.json').read_text()).get('arms',ARMS))
    if arms not in (ARMS,('distance_only','frozen_em','reference_only')):raise ValueError('unsupported matched run arms')
    metrics=pd.read_csv(run/'method_comparison.csv')
    percell=pd.read_csv(diag/'per_cell_results.csv',dtype={'cell_id':str})
    checkmean=percell.groupby(['generator','arm'])[list(protocol.METRICS)].mean()
    reported=metrics.set_index(['generator','arm'])[list(protocol.METRICS)]
    np.testing.assert_allclose(checkmean.sort_index(),reported.sort_index(),atol=1e-12,rtol=0.)
    solver_checks=pd.concat([pd.read_json(run/g/'subdiagnostic/solver_checks.json') for g in GENERATORS])
    if 'solver_elapsed_seconds' in solver_checks:
        solver_checks.groupby(['generator','arm']).agg(n_patches=('patch_id','size'),
            total_solver_seconds=('solver_elapsed_seconds','sum'),
            median_patch_seconds=('solver_elapsed_seconds','median'),
            max_patch_seconds=('solver_elapsed_seconds','max')).to_csv(run/'solver_runtime.csv')
    n_soft=0;all_exact=0;iterations=[];fingerprints={};statuses=[]
    for generator in GENERATORS:
        directory=run/generator
        oldchecks=json.loads((directory/'subdiagnostic/baseline_reproduction.json').read_text())
        all_exact+=sum(r['all_npz_fields_exact'] for r in oldchecks)
        for arm in arms:
            assignments=pd.read_csv(directory/'methods'/arm/'assignments.csv')
            assert not assignments.barcode.duplicated().any()
            status=json.loads((directory/'subdiagnostic'/f'{arm}_evaluation.status.json').read_text())
            assert status['returncode']==0;statuses.append(status)
            for path in (directory/'methods'/arm/'em_assignment').glob('*.npz'):
                with np.load(path,allow_pickle=False) as data:
                    splits=data['candidate_row_splits'];r=data['responsibility'];q=data['cell_type_posterior']
                    assert np.isfinite(r).all() and np.isfinite(q).all()
                    np.testing.assert_allclose(np.add.reduceat(r,splits[:-1]),1.,atol=1e-6)
                    np.testing.assert_allclose(q.sum(1),1.,atol=1e-6)
                    for bi in np.flatnonzero(data['is_nuclear_locked']):
                        lo,hi=splits[bi:bi+2]
                        np.testing.assert_array_equal(r[lo:hi],(data['candidate_cell_index'][lo:hi]==data['locked_owner_cell_index'][bi]).astype(float))
                    assert np.all((data['normalized_entropy']>=0)&(data['normalized_entropy']<=1))
                    meta=json.loads(str(data['metadata_json']));assert meta['converged']
                    iterations.append(len(meta['iterations']));n_soft+=1
        original=source/generator/'benchmark/modeled_only'
        episode,=(original/'episodes').glob('*/episodes_index.csv')
        for label,path in [('observed_compact',original/'observed_compact.npz'),('episode_build',original/'episode_build.yaml'),
                ('episodes_index',episode),('common_nuclear_seeds',source/generator/'benchmark/common_confident_nuclear_bins.csv.gz'),
                ('evaluation_only_profile',profiles/f'{generator}_20260916.h5'),
                ('evaluation_only_profile_mapping',source/generator/'profile_assignment.csv')]:
            fingerprints[generator+'_'+label]=file_fingerprint(path)
    assert n_soft==24
    reproduced=sum(sum(r['all_fields_reproduced_at_1e12_tolerance'] for r in json.loads(
        (run/g/'subdiagnostic/baseline_reproduction.json').read_text())) for g in GENERATORS)
    assert reproduced==16
    tests=(diag/'tests.log').read_text()
    test_count=int(re.search(r'Ran (\d+) tests',tests).group(1))
    assert test_count >= (68 if arms[-1]=='reference_only' else 60) and tests.rstrip().endswith('OK')
    protocol.dump(diag/'input_fingerprints.json',fingerprints)
    n_code=notebook(run)
    retained=diag/'analysis_source';retained.mkdir(exist_ok=True)
    document='docs/EM_REFERENCE_ONLY_EXPERIMENT.md' if arms[-1]=='reference_only' else 'docs/EM_POSTERIOR_PREDICTIVE_EXPERIMENT.md'
    for name in ('scripts/audit_em_posterior_predictive.py',document):
        shutil.copy2(ROOT/name,retained/Path(name).name)
    protocol.dump(retained/'checksums.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in retained.iterdir() if p.name!='checksums.json'})
    protocol.dump(diag/'final_audit.json',dict(soft_artifacts_checked=n_soft,exact_baseline_soft_reproduction=all_exact,
        baseline_soft_reproduction_at_1e12_tolerance=reproduced,
        iterations_min=min(iterations),iterations_max=max(iterations),all_converged=True,
        nuclear_locks_exact=True,unique_merged_ownership=True,finite_normalized_probabilities=True,
        official_evaluations_passed=len(statuses),per_cell_means_reconcile=True,tests_passed=test_count,
        notebook_code_cells_executed=n_code,notebook_visual_inspection=False,
        no_new_holdout=True,no_PPO=True,no_weight_search=True,
        advancement_gate=json.loads((run/'development_gate.json').read_text())['passed']))
    print((diag/'final_audit.json').read_text())


if __name__=='__main__':main()
