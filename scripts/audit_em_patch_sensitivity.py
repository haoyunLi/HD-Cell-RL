#!/usr/bin/env python3
"""Frozen EM regression and context-normalization diagnostic; no RL or tuning."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hd_cell_rl.em_assignment import build_sparse_patch_em_input, run_generalized_em
from hd_cell_rl.em_types import EMAssignmentConfig
from hd_cell_rl.experiment_protocol import write_experiment_manifest
from hd_cell_rl.patch_assignment import PatchOwnershipMerger
from hd_cell_rl.patch_dataset import PatchDataset
from hd_cell_rl.ppo_checkpoint import load_checkpoint_payload
from hd_cell_rl.ppo_config import load_ppo_training_config
from scripts.evaluate_patch_initialization import _settings_from_checkpoint
from scripts.run_em_weight_sweep import _hard_assignment_rows


def compare_results(base, changed):
    """Align real IDs; absent pairs have probability zero, never index equality."""
    a, b = base.barcode_to_index(), changed.barcode_to_index()
    common = sorted(a.keys() & b.keys())
    l1, owner = [], []
    for barcode in common:
        i, j = a[barcode], b[barcode]
        def row(result, index):
            s, e = result.candidate_row_splits[index:index+2]
            return {str(result.cell_ids[c]): float(p) for c, p in zip(
                result.candidate_cell_index[s:e], result.responsibility[s:e])}
        x, y = row(base, i), row(changed, j)
        l1.append(sum(abs(x.get(c, 0)-y.get(c, 0)) for c in x.keys() | y.keys()))
        owner.append(str(base.cell_ids[base.top1_cell_index[i]]) != str(changed.cell_ids[changed.top1_cell_index[j]]))
    return {'n_common_bins':len(common), 'mean_row_l1':float(np.mean(l1)),
            'hard_owner_changed_fraction':float(np.mean(owner))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--em-config', type=Path, default=ROOT/'configs/em_assignment.frozen_baseline.yaml')
    args = parser.parse_args()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    old = json.loads((args.benchmark_dir/'methods/em_frozen_and_distance/summary.json').read_text())
    em = EMAssignmentConfig.from_mapping(yaml.safe_load(args.em_config.read_text())['em_assignment'])
    config = replace(load_ppo_training_config(old['base_config']), planner_enabled=False,
        episodes_index_path=Path(old['episodes_index_path_used']), nuclei_path=Path(old['nuclei_path_used']),
        reference_path=Path(old['reference_path_used']), reference_format='npz')
    settings = _settings_from_checkpoint(patch_cfg=load_checkpoint_payload(old['checkpoint_protocol'])['patch_config'],
        patches_index_path=Path(old['patches_index_path']), mode='em_only', artifact_dir=out/'unused')
    # Keep candidate-completeness checks, but make the loader's preliminary solve cheap.
    settings = replace(settings, cache_patch_contexts=False, em_assignment=replace(em,
        cell_specific_expression_enabled=False, max_iterations=1, save_artifacts=False, artifact_dir=None))
    frame = pd.read_csv(old['patches_index_path'])
    dataset = PatchDataset(base_config=config, settings=settings, rng=np.random.default_rng(7))
    rows, reliability_rows, regression = [], [], []
    mergers = {}
    try:
        theta = np.asarray(dataset.reference_theta)
        for original in frame.to_dict('records'):
            base = None
            for variant, shrink, shift in [('baseline',0,0), ('halo_minus_4',4,0), ('halo_minus_8',8,0),
                                           ('inset_shift_left',4,-4), ('inset_shift_right',4,4)]:
                modified = dict(original)
                for axis in ['x','y']:
                    modified[f'outer_{axis}_min'] += shrink + (shift if axis=='x' else 0)
                    modified[f'outer_{axis}_max'] -= shrink - (shift if axis=='x' else 0)
                if any(modified[f'outer_{a}_min'] > original[f'core_{a}_min'] or
                       modified[f'outer_{a}_max'] < original[f'core_{a}_max'] for a in ['x','y']):
                    raise ValueError('inset must preserve the complete fixed core rectangle')
                # Let the normal MaxDis loader choose the necessary margins anew.
                modified['patch_cell_ids'] = original['core_cell_ids']
                modified['margin_cell_ids'] = '[]'
                modified['patch_id'] = str(original['patch_id'])+'__'+variant
                ctx = dataset.load_patch_context(SimpleNamespace(**modified))
                if ctx is None: raise ValueError('missing regression context')
                inp = build_sparse_patch_em_input(context=ctx, candidate_max_distance_um=ctx.candidate_max_distance_um,
                    non_nuclear_bin_filter=em.non_nuclear_bin_filter)
                counts = dataset.load_unique_patch_expression(context=ctx, barcode_ids=inp.barcode_ids)
                result = run_generalized_em(em_input=inp, config=em, bin_gene_counts=counts, reference_theta=theta)
                if base is None:
                    base = result
                merger = mergers.setdefault(variant, PatchOwnershipMerger())
                merger.add_patch(context=ctx, rows=_hard_assignment_rows(context=ctx, result=result,
                    target_cell_ids=set(ctx.core_cell_ids)), target_cell_ids=ctx.core_cell_ids)
                common_cells = sorted(set(base.cell_ids) & set(result.cell_ids))
                ai = {c:i for i,c in enumerate(base.cell_ids)}; bi = {c:i for i,c in enumerate(result.cell_ids)}
                for c in common_cells:
                    i, j = ai[c], bi[c]
                    reliability_rows.append(dict(patch_id=original['patch_id'], variant=variant, cell_id=c,
                        is_core=c in ctx.core_cell_ids, nuclear_umi_before=float(base.nuclear_expression_count_total[i]),
                        nuclear_umi_after=float(result.nuclear_expression_count_total[j]),
                        reliability_before=float(base.cell_profile_reliability[i]),
                        reliability_after=float(result.cell_profile_reliability[j])))
                record = dict(patch_id=original['patch_id'], variant=variant, n_cells=result.n_cells,
                    converged=result.converged, iterations=len(result.iterations),
                    candidate_max_distance_um=ctx.candidate_max_distance_um,
                    reference_prior_count=result.relative_profile_prior_count, **compare_results(base,result))
                # Context evidence is unchanged between these two solves; only prior scaling differs.
                if variant != 'baseline':
                    anchored = run_generalized_em(em_input=inp, config=replace(em,
                        cell_profile_relative_prior_strength=em.cell_profile_relative_prior_strength *
                            base.relative_profile_prior_count/result.relative_profile_prior_count),
                        bin_gene_counts=counts, reference_theta=theta)
                    record.update({'normalization_only_'+k:v for k,v in compare_results(anchored,result).items()})
                    control = mergers.setdefault(variant+'__fixed_prior_control', PatchOwnershipMerger())
                    control.add_patch(context=ctx, rows=_hard_assignment_rows(context=ctx, result=anchored,
                        target_cell_ids=set(ctx.core_cell_ids)), target_cell_ids=ctx.core_cell_ids)
                rows.append(record)
                print(original['patch_id'], variant, record, flush=True)
    finally:
        dataset.close()
    assignments, conflicts = mergers['baseline'].finalize()
    pd.DataFrame(assignments).to_csv(out/'assignments.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'context_sensitivity.csv',index=False)
    pd.DataFrame(reliability_rows).to_csv(out/'cell_reliability.csv',index=False)
    (out/'subdiagnostic').mkdir()
    merged_comparison=[]
    baseline_owners={str(r['barcode']):str(r['cell_id']) for r in assignments}
    for variant,merger in mergers.items():
        result,conflicts_variant=merger.finalize()
        owners={str(r['barcode']):str(r['cell_id']) for r in result}
        shared=baseline_owners.keys() & owners.keys()
        merged_comparison.append(dict(variant=variant,n_unique_barcodes=len(owners),n_common_owned_barcodes=len(shared),
            n_common_owner_changed=sum(baseline_owners[b]!=owners[b] for b in shared),
            n_assigned_support_changed=len(baseline_owners.keys()^owners.keys()),n_conflict_rows=len(conflicts_variant)))
        pd.DataFrame(result).to_csv(out/'subdiagnostic'/f'{variant}.assignments.csv',index=False)
    pd.DataFrame(merged_comparison).to_csv(out/'merged_context_sensitivity.csv',index=False)
    pd.DataFrame(conflicts, columns=['barcode','candidate_cell_id','source_patch_id','source_patch_score',
        'nuclear_owner','final_owner','rule']).to_csv(out/'subdiagnostic/merge_conflicts.csv',index=False)
    previous = pd.read_csv(args.benchmark_dir/'methods/em_frozen_and_distance/variants/alpha_1p0__beta_1p5__kappa_2p0__profile_damping_0p0/assignments.csv', dtype={'cell_id':str})
    current = pd.DataFrame(assignments)
    for c in sorted(current.cell_id.unique()):
        a=set(previous.loc[previous.cell_id==c,'barcode']); b=set(current.loc[current.cell_id==c,'barcode'])
        regression.append(dict(cell_id=c,old_n_bins=len(a),new_n_bins=len(b),changed_bins=len(a^b),
                               old_new_assignment_iou=len(a&b)/max(len(a|b),1)))
    pd.DataFrame(regression).to_csv(out/'frozen_code_regression.csv',index=False)
    write_experiment_manifest(out, config={'em_assignment':asdict(em), 'base':config.to_serializable_dict()},
        inputs={'baseline_config':args.em_config, 'episodes':config.episodes_index_path,
                'reference':config.reference_path,'nuclei':config.nuclei_path,'patches':settings.patches_index_path},
        protocol={'role':'development/audit regression; not fresh test', 'GT_used_for_inference':False,
            'geometry':'fixed core; shrink halo by 4/8 um; inset grid shift +/-4 um, all within original context',
            'normalization_control':'hold baseline prior count fixed by compensating kappa; diagnostic only',
            'previous_results':'same physical data and retained parameters; current correctness/convergence implementation'})


if __name__ == '__main__': main()
