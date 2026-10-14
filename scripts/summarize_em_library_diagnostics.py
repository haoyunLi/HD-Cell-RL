#!/usr/bin/env python3
"""Summarize frozen EM controls with explicit core-cell and bin denominators."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hd_cell_rl.experiment_protocol import paired_cluster_mean_ci
from scripts.diagnose_em_library_size import ARMS
from scripts import run_source_separated_method_comparison as comparison


def add_incremental_score_decomposition(bins, directory):
    """Also express the blend as type + lambda * (cell - type).

    This avoids treating the large cancelling weighted log-likelihood terms as
    separate causal effects. q is still the last fitted E-step q, not the q from
    a separately fitted type-only control.
    """
    records = []
    for path in (directory/'methods/frozen_em/em_assignment').glob('*.npz'):
        with np.load(path) as data:
            barcode_ids = data['barcode_ids'].astype(str)
            cell_ids = data['cell_ids'].astype(str)
            cell_index = {c:i for i,c in enumerate(cell_ids)}
            selected = bins[bins.patch_id == path.stem].set_index('barcode')
            pair_cell, pair_bin = data['candidate_cell_index'], data['pair_bin_index']
            scale = 1.5*data['expression_confidence'][pair_bin].astype(float)
            type_base = scale*data['final_type_expression_compatibility'].astype(float)
            correction = scale*data['cell_profile_reliability'][pair_cell].astype(float)*(data['final_cell_expression_compatibility'].astype(float)-data['final_type_expression_compatibility'].astype(float))
            for bi, barcode in enumerate(barcode_ids):
                if barcode not in selected.index:
                    continue
                gt = selected.loc[barcode,'gt_cell_id']
                start,stop = data['candidate_row_splits'][bi:bi+2]
                pairs=np.arange(start,stop)
                gt_pair=pairs[pair_cell[pairs] == cell_index.get(gt,-1)]
                if not len(gt_pair):
                    continue
                chosen=pairs[pair_cell[pairs] == data['top1_cell_index'][bi]][0]
                records.append(dict(patch_id=path.stem,barcode=barcode,
                    chosen_minus_gt_type_base=float(type_base[chosen]-type_base[gt_pair[0]]),
                    chosen_minus_gt_cell_correction=float(correction[chosen]-correction[gt_pair[0]])))
    result=bins.merge(pd.DataFrame(records),on=['patch_id','barcode'],how='left',validate='one_to_one')
    valid=result.gt_in_candidates
    np.testing.assert_allclose(result.loc[valid,'chosen_minus_gt_spatial']+result.loc[valid,'chosen_minus_gt_type_base']+
        result.loc[valid,'chosen_minus_gt_cell_correction'],result.loc[valid,'chosen_minus_gt_combined'],atol=2e-5,rtol=1e-5)
    return result


def summarize(out, source):
    diagnostic = out/'subdiagnostic'
    diagnostic.mkdir(exist_ok=True)
    all_cells, all_bins, metrics, reproduction = [], [], [], []
    expected = pd.read_csv(source/'subdiagnostic/per_cell_comparison.csv', dtype={'cell_id':str})
    expected = expected[expected.condition == 'modeled_only']
    patch_map = pd.read_csv(comparison.BENCH/'selection/source_eval/per_episode.csv', dtype={'cell_id':str})[['cell_id','patch_id']]
    if patch_map.cell_id.duplicated().any():
        raise ValueError('evaluation core cell must have one patch')
    for generator in ('A_current','B_library_offset'):
        directory = out/generator
        covariates = pd.read_csv(directory/'subdiagnostic/cell_covariates.csv', dtype={'cell_id':str})
        covariates = covariates[covariates.is_core]
        per_arm = []
        for arm in ARMS:
            path, = (directory/'evaluations').glob(arm+'_*/per_episode.csv')
            frame = pd.read_csv(path, dtype={'cell_id':str})
            if 'patch_id' in frame:
                frame = frame.drop(columns='patch_id')
            frame = frame.merge(patch_map, on='cell_id', validate='one_to_one')
            if len(frame) != 43 or frame.cell_id.nunique() != 43:
                raise ValueError('expected exactly 43 core evaluation cells')
            selected_metrics = ['pred_iou','pred_fractional_iou','pred_precision','pred_recall','gene_spearman_r']
            if not np.isfinite(frame[selected_metrics].to_numpy()).all():
                raise ValueError('nonfinite evaluation metric')
            if arm in ('frozen_em','distance_only'):
                old = expected[(expected.generator == generator)&(expected.method == arm)]
                joined = frame.merge(old[['cell_id']+selected_metrics], on='cell_id', suffixes=('_new','_old'), validate='one_to_one')
                errors = {m:float(np.max(np.abs(joined[m+'_new']-joined[m+'_old']))) for m in selected_metrics}
                if max(errors.values()) > 1e-10:
                    raise AssertionError(f'published metric reproduction failed: {errors}')
                reproduction.append(dict(generator=generator,arm=arm,**errors))
            frame['arm'] = arm
            frame['generator'] = generator
            per_arm.append(frame)
            metrics.append(dict(generator=generator, arm=arm, n_cells=len(frame),
                **{m:float(frame[m].mean()) for m in selected_metrics}))
        cells = pd.concat(per_arm)
        cells = cells.merge(covariates.drop(columns='generator'), on=['cell_id','patch_id'], validate='many_to_one')
        all_cells.append(cells)
        bins = pd.read_csv(directory/'subdiagnostic/bin_transitions.csv.gz', dtype={c:str for c in ['gt_cell_id', *ARMS]})
        bins = add_incremental_score_decomposition(bins, directory)
        all_bins.append(bins)
    cells = pd.concat(all_cells, ignore_index=True)
    bins = pd.concat(all_bins, ignore_index=True)
    cells.to_csv(diagnostic/'per_cell_results.csv', index=False)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(out/'method_comparison.csv', index=False)
    cells.groupby(['generator','patch_id','arm'])[['pred_iou','pred_precision','pred_recall','gene_spearman_r']].mean().to_csv(out/'per_patch_comparison.csv')
    comparison.dump(diagnostic/'baseline_reproduction.json', reproduction)
    differences, strata, correlations, bin_summaries, attribution = [], [], [], [], []
    for generator, group in cells.groupby('generator'):
        pivot = group.pivot(index=['cell_id','patch_id'], columns='arm', values='pred_iou').reset_index()
        frozen = group[group.arm == 'frozen_em'].copy()
        joined = frozen.merge(pivot, on=['cell_id','patch_id'], validate='one_to_one')
        joined['em_minus_distance'] = joined.frozen_em-joined.distance_only
        joined.to_csv(diagnostic/f'{generator}_cell_drivers.csv', index=False)
        for control in ARMS[2:]:
            delta = pivot[control]-pivot.frozen_em
            low, high = paired_cluster_mean_ci(delta, pivot.patch_id, replicates=10000, rng=np.random.default_rng(20260916))
            differences.append(dict(generator=generator,control=control,mean_iou_delta_vs_frozen=float(delta.mean()),
                patch_bootstrap_low=low,patch_bootstrap_high=high,n_improved=int((delta>1e-12).sum()),
                n_damaged=int((delta < -1e-12).sum()),n_equal=int((np.abs(delta)<=1e-12).sum())))
        for feature in ['nuclear_umi','nuclear_genes','gt_area_um2','competitor_depth_ratio','profile_reliability','oracle_profile_total_variation']:
            cut = pd.qcut(joined[feature], q=4, duplicates='drop')
            for label, subset in joined.groupby(cut, observed=True):
                strata.append(dict(generator=generator, feature=feature, quartile=str(label), n_cells=len(subset),
                    feature_min=float(subset[feature].min()),feature_max=float(subset[feature].max()),
                    mean_iou_delta=float(subset.em_minus_distance.mean()),
                    mean_em_iou=float(subset.frozen_em.mean()),mean_distance_iou=float(subset.distance_only.mean())))
            correlations.append(dict(generator=generator,feature=feature,
                spearman_with_em_minus_distance=float(joined[[feature,'em_minus_distance']].corr(method='spearman').iloc[0,1])))
        scoped = bins[(bins.generator == generator)&bins.gt_core&(~bins.nuclear_locked)]
        if scoped.barcode.duplicated().any():
            raise ValueError('core-bin diagnostic population duplicated across patches')
        for control in ARMS[1:]:
            for baseline in ('distance', 'frozen'):
                column = control+'_vs_'+baseline
                if column not in scoped:
                    continue
                values = scoped[column].value_counts()
                if int(values.sum()) != len(scoped):
                    raise ValueError('transition counts do not reconcile')
                bin_summaries.append(dict(generator=generator,arm=control,baseline=baseline,n_evaluated=len(scoped),
                    **{v:int(values.get(v,0)) for v in ['corrected','damaged','both_correct','both_wrong']}))
        damaged = scoped[(scoped.frozen_em_vs_distance == 'damaged')&scoped.gt_in_candidates].copy()
        for name in ['spatial','type','cell','type_base','cell_correction','combined']:
            values = damaged['chosen_minus_gt_'+name]
            attribution.append(dict(generator=generator, component=name, n_damaged=len(damaged),
                mean_chosen_minus_gt=float(values.mean()), median_chosen_minus_gt=float(values.median()),
                n_favor_wrong=int((values > 0).sum()),n_favor_gt=int((values < 0).sum()),
                n_last_score_argmax_mismatch=int((~damaged.top_equals_last_score_max).sum())))
        scoped.to_csv(diagnostic/f'{generator}_core_nonnuclear_bin_diagnosis.csv.gz',index=False)
    pd.DataFrame(differences).to_csv(out/'control_effects.csv',index=False)
    pd.DataFrame(strata).to_csv(out/'driver_strata.csv',index=False)
    pd.DataFrame(correlations).to_csv(diagnostic/'driver_correlations.csv',index=False)
    pd.DataFrame(bin_summaries).to_csv(out/'bin_correction_damage.csv',index=False)
    pd.DataFrame(attribution).to_csv(out/'damaged_score_decomposition.csv',index=False)
    checks = pd.concat([pd.read_json(out/g/'subdiagnostic/solver_checks.json') for g in ('A_current','B_library_offset')])
    checks.to_csv(diagnostic/'solver_checks.csv',index=False)
    source_record = diagnostic/'analysis_source'
    source_record.mkdir(exist_ok=True)
    source_path = Path(__file__).resolve()
    retained = source_record/source_path.name
    if retained.exists() and retained.read_bytes() != source_path.read_bytes():
        raise FileExistsError('refusing to replace a different retained analysis source')
    shutil.copy2(source_path, retained)
    comparison.dump(source_record/'manifest.json',dict(script=str(source_path),
        sha256=hashlib.sha256(retained.read_bytes()).hexdigest(),
        input_run=str(out),frozen_experiment_manifest=str(out/'experiment_manifest.json')))
    return metric_frame


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--source-dir',type=Path,required=True)
    a=p.parse_args()
    print(summarize(a.run_dir.resolve(),a.source_dir.resolve()).to_string(index=False))


def write_notebook(run_dir):
    """Executed compact-table companion; heavy experiments stay in the runner."""
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    run_dir = Path(run_dir).resolve()
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell('# Frozen EM failure diagnosis\n\n## tl;dr\n'
            'This companion reproduces the retained diagnostic tables. Read `report.md` for interpretation. '
            'The oracle uses true generating profiles and is **not a deployable method**. '
            'These four patches are development evidence, not new held-out validation.'),
        nbformat.v4.new_markdown_cell('## Context & Methods\n'
            'Modeled-only A/B pseudo-HD; same 43 core cells. Fixed alpha=1, beta=1.5, kappa=2, '
            'profile damping=0. The type-only control disables cell-specific expression. '
            'The oracle replaces the shrunken nuclear profile with the normalized generating profile, '
            'while preserving nuclear-derived lambda and q updates.\n\n'
            '### Key Assumptions\n'
            'Official IoU uses the existing merged candidate-clipped evaluator. '
            'Bin transitions use analyzed nonnuclear bins whose GT owner is a core cell. '
            'Excluded zero-count bins are not in that transition denominator. '
            'Correlations and quartiles are descriptive, not independent biological replicates.'),
        nbformat.v4.new_markdown_cell('## Data\nInput hashes and recoverable code: `experiment_manifest.json` and `source_snapshot.tar.gz`.'),
        nbformat.v4.new_code_cell('from pathlib import Path\nimport json\nimport pandas as pd\n'
            f'run = Path({str(run_dir)!r})\n'
            'checks = json.loads((run / "subdiagnostic/baseline_reproduction.json").read_text())\n'
            'assert all(max(v for k,v in row.items() if k not in ("generator","arm")) < 1e-10 for row in checks)\n'
            'print("All four frozen baseline evaluations reproduced.")'),
        nbformat.v4.new_markdown_cell('## Results\n### Core-cell comparison\n'
            'Exact values are kept in compact tables; no smoothed curves or cell-independent confidence intervals.'),
        nbformat.v4.new_code_cell('pd.read_csv(run / "method_comparison.csv").round(6)'),
        nbformat.v4.new_markdown_cell('### Control effect versus frozen EM\n'
            'Paired bootstrap resamples the four whole patches. With only four patches, these intervals are descriptive.'),
        nbformat.v4.new_code_cell('pd.read_csv(run / "control_effects.csv").round(6)'),
        nbformat.v4.new_markdown_cell('### Corrected and damaged bins\nThe denominator excludes locked nuclei and unanalyzed zero-count bins.'),
        nbformat.v4.new_code_cell('pd.read_csv(run / "bin_correction_damage.csv")'),
        nbformat.v4.new_markdown_cell('### Score accounting on bins damaged by frozen EM\n'
            'Positive chosen-minus-GT favors the wrong owner. This is the last raw E-step score, '
            'not the damped responsibility; mismatches are counted explicitly.'),
        nbformat.v4.new_code_cell('pd.read_csv(run / "damaged_score_decomposition.csv").round(5)'),
        nbformat.v4.new_markdown_cell('## Takeaways\n'
            'The controls isolate changes within this frozen simulator and inference protocol. '
            'They do not establish performance on real HD data. No PPO training or production weight selection occurred. '
            'See `report.md` for the evidence-backed next experiment.')]
    notebook.metadata.kernelspec = dict(name='python3', display_name='Python diagnostic environment', language='python')
    nbformat.validate(notebook)
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(notebook, km=manager, timeout=300, resources={'metadata':{'path':str(run_dir)}})
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    nbformat.write(notebook, run_dir/'diagnosis.ipynb')
    if any(output.output_type == 'error' for cell in notebook.cells if cell.cell_type == 'code' for output in cell.outputs):
        raise AssertionError('notebook contains errors')


if __name__ == '__main__':
    main()
