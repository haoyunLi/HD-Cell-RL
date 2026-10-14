#!/usr/bin/env python3
"""Frozen modeled-only EM controls; generating-profile oracle is diagnostic only.

No production solver changes. The oracle replaces only the initializer's profile
in a process-local context; nuclear reliability, prior strength, q updates,
geometry, confidence and all damping/convergence settings remain unchanged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hd_cell_rl import em_assignment as em
from hd_cell_rl.experiment_protocol import write_experiment_manifest
from hd_cell_rl.patch_assignment import PatchOwnershipMerger, validate_unique_ownership
from hd_cell_rl.patch_dataset import PatchDataset
from hd_cell_rl.ppo_checkpoint import load_checkpoint_payload
from hd_cell_rl.ppo_config import load_ppo_training_config
from scripts import run_em_weight_sweep as sweep
from scripts import run_source_separated_method_comparison as comparison
from scripts.run_pseudo_realism_experiment import load_h5

ARMS = ('distance_only', 'frozen_em', 'type_only', 'oracle_profile_diagnostic')


def align_oracle(counts, source_ids, source_genes, mapping, cell_ids, genes, epsilon):
    """Strict physical-cell and reference-gene alignment; never infer missing IDs."""
    if len(set(source_ids)) != len(source_ids) or len(set(source_genes)) != len(source_genes):
        raise ValueError('duplicate oracle axes')
    if mapping.cell_id.duplicated().any():
        raise ValueError('duplicate physical cell mapping')
    rows = {str(v): i for i, v in enumerate(source_ids)}
    cols = {str(v): i for i, v in enumerate(source_genes)}
    physical = mapping.set_index('cell_id').sc_cell_barcode.to_dict()
    values = counts[[rows[physical[str(c)]] for c in cell_ids]][:, [cols[str(g)] for g in genes]].toarray().astype(float)
    if np.any(values.sum(1) <= 0):
        raise ValueError('oracle profile has no selected-gene expression')
    return em._normalize_cell_gene_counts(values, epsilon=epsilon)


def score_components(result, config):
    pair = result.pair_bin_index
    reliability = result.cell_profile_reliability[result.candidate_cell_index].astype(float)
    spatial = config.spatial_weight * np.log(np.maximum(result.spatial_prior.astype(float), config.epsilon))
    scale = config.expression_weight * result.expression_confidence[pair].astype(float)
    type_score = scale * (1 - reliability) * result.final_type_expression_compatibility
    cell_score = scale * reliability * result.final_cell_expression_compatibility
    np.testing.assert_allclose(spatial + type_score + cell_score, result.final_combined_score, atol=1e-5, rtol=1e-6)
    return spatial, type_score, cell_score


def transition(before, after, truth):
    if before == truth and after != truth:
        return 'damaged'
    if before != truth and after == truth:
        return 'corrected'
    return 'both_correct' if after == truth else 'both_wrong'


def run_generator(args, generator, *, profile_controls=None, evaluation_workers=1, max_iterations=None):
    # Optional diagnostic-only profile builders; the original four-arm default
    # and production solver are unchanged. Parallelism is evaluation-only.
    controls = profile_controls or {}
    arms = ARMS if not controls else ('distance_only', 'frozen_em', *controls, 'oracle_profile_diagnostic')
    if len(set(arms)) != len(arms):
        raise ValueError('diagnostic arm names must be unique')
    source = args.source_dir / generator
    original = source / 'benchmark/modeled_only'
    out = args.output_dir / generator
    out.mkdir()
    (out / 'subdiagnostic').mkdir()
    context_path, = (original/'methods/nucleus_only').glob('*/evaluation_context')
    episodes, = (original/'episodes').glob('*/episodes_index.csv')
    config = replace(load_ppo_training_config(comparison.BASE), planner_enabled=False,
        episodes_index_path=episodes, nuclei_path=comparison.NUCLEI,
        reference_path=comparison.REFERENCE, reference_format='npz')
    settings = sweep._settings_from_checkpoint(
        patch_cfg=dict(load_checkpoint_payload(comparison.CHECKPOINT).get('patch_config') or {}),
        patches_index_path=comparison.PATCHES, mode='em_only', artifact_dir=out/'unused')
    frozen = replace(settings.em_assignment, save_artifacts=False, artifact_dir=None,
        debug_patch_ids=(), background_enabled=False, spatial_weight=1., expression_weight=1.5,
        cell_specific_expression_enabled=True, cell_profile_relative_prior_strength=2.,
        cell_profile_update_damping=0., non_nuclear_bin_filter='positive_expression')
    if max_iterations is not None:
        frozen = replace(frozen, max_iterations=int(max_iterations))
    settings = replace(settings, cache_patch_contexts=True, em_assignment=replace(frozen,
        cell_specific_expression_enabled=False))
    target = set(sweep.load_eval_cell_ids(context_path/'per_episode.csv'))
    selected = sweep._select_patch_rows(pd.read_csv(comparison.PATCHES), target)
    mapping = pd.read_csv(source/'profile_assignment.csv', dtype={'cell_id':str})
    profile_h5 = args.profile_dir / f'{generator}_20260916.h5'
    counts, profile_ids, profile_genes = load_h5(profile_h5)
    with np.load(comparison.REFERENCE) as reference:
        genes = reference[config.reference_genes_key].astype(str)
    comparison.PROFILE_ASSIGNMENT = source/'profile_assignment.csv'
    comparison.PROFILE_H5 = profile_h5
    comparison.dump(out/'resolved_em_config.json', asdict(frozen))
    dataset = PatchDataset(base_config=config, settings=settings, rng=np.random.default_rng(20260914))
    mergers = {arm: PatchOwnershipMerger() for arm in arms}
    cell_records, bin_records, pair_records, run_checks = [], [], [], []
    truth = sweep._load_gt(comparison.GT, set(mapping.cell_id))
    if truth.barcode.duplicated().any():
        raise ValueError('dominant GT must have unique physical barcodes')
    truth_owner = dict(zip(truth.barcode.astype(str), truth.cell_id.astype(str)))
    area = truth.groupby('cell_id').size().to_dict()
    try:
        for patch_row in selected:
            context = dataset.load_patch_context(patch_row)
            if context is None:
                raise ValueError('missing selected context')
            inp = em.build_sparse_patch_em_input(context=context,
                candidate_max_distance_um=context.candidate_max_distance_um,
                non_nuclear_bin_filter='positive_expression')
            expression = dataset.load_unique_patch_expression(context=context, barcode_ids=inp.barcode_ids)
            oracle = align_oracle(counts, profile_ids, profile_genes, mapping, inp.cell_ids, genes, frozen.epsilon)
            nuclear_bins = np.flatnonzero(inp.is_nuclear_locked)
            membership = sparse.coo_matrix((np.ones(len(nuclear_bins)),
                (inp.locked_owner_cell_index[nuclear_bins], nuclear_bins)),
                shape=(len(inp.cell_ids), len(inp.barcode_ids))).tocsr()
            nuclear = (membership @ expression).tocsr()
            observed_total = np.asarray(expression.sum(1)).ravel()
            original_initializer = em.initialize_cell_expression_profiles
            def oracle_initializer(**kwargs):
                _, reliability, totals, prior = original_initializer(**kwargs)
                return oracle.copy(), reliability, totals, prior
            results = {}
            for arm in arms:
                arm_cfg = replace(frozen, expression_weight=0. if arm == 'distance_only' else 1.5,
                    cell_specific_expression_enabled=arm != 'type_only')
                kwargs = dict(bin_gene_counts=expression, reference_theta=dataset.reference_theta)
                if arm in controls:
                    def controlled_initializer(**initialization):
                        original_values = original_initializer(**initialization)
                        profile = controls[arm](oracle, initialization, original_values)
                        if profile.shape != original_values[0].shape:
                            raise ValueError('diagnostic profile changed gene/cell axes')
                        np.testing.assert_allclose(profile.sum(1), 1., atol=1e-12)
                        return profile, *original_values[1:]
                    with patch.object(em, 'initialize_cell_expression_profiles', controlled_initializer):
                        result = em.run_generalized_em(inp, arm_cfg, **kwargs)
                elif arm == 'oracle_profile_diagnostic':
                    with patch.object(em, 'initialize_cell_expression_profiles', oracle_initializer):
                        result = em.run_generalized_em(inp, arm_cfg, **kwargs)
                else:
                    result = em.run_generalized_em(inp, arm_cfg, **kwargs)
                result.validate()
                results[arm] = result
                path = out/'methods'/arm
                em.save_em_result_npz(result, path/'em_assignment'/f'{context.patch_id}.npz')
                mergers[arm].add_patch(context=context, score=0., target_cell_ids=[c for c in context.core_cell_ids if c in target],
                    rows=sweep._hard_assignment_rows(context=context, result=result, target_cell_ids=target))
                score_components(result, arm_cfg)
                run_checks.append(dict(generator=generator, patch_id=context.patch_id, arm=arm,
                    n_bins=result.n_bins, n_cells=len(result.cell_ids), n_locked=int(result.is_nuclear_locked.sum()),
                    converged=result.converged, iterations=len(result.iterations),
                    candidate_max_distance_um=context.candidate_max_distance_um))
                print(generator, context.patch_id, arm, 'done', flush=True)
            baseline = results['frozen_em']
            for controlled_arm in controls:
                np.testing.assert_array_equal(baseline.cell_profile_reliability, results[controlled_arm].cell_profile_reliability)
                np.testing.assert_array_equal(baseline.nuclear_expression_count_total, results[controlled_arm].nuclear_expression_count_total)
                assert baseline.relative_profile_prior_count == results[controlled_arm].relative_profile_prior_count
            np.testing.assert_array_equal(baseline.cell_profile_reliability, results['oracle_profile_diagnostic'].cell_profile_reliability)
            np.testing.assert_allclose(results['oracle_profile_diagnostic'].cell_expression_profile, oracle, rtol=1e-6, atol=1e-9)
            spatial, type_score, cell_score = score_components(baseline, frozen)
            owners = {a: np.asarray(r.cell_ids)[r.top1_cell_index] for a, r in results.items()}
            for ci, cell in enumerate(inp.cell_ids):
                rows = np.unique(inp.pair_bin_index[inp.candidate_cell_index == ci])
                neighbors = np.unique(inp.candidate_cell_index[np.isin(inp.pair_bin_index, rows)])
                neighbors = neighbors[neighbors != ci]
                nuclear_totals = baseline.nuclear_expression_count_total
                cell_records.append(dict(generator=generator, patch_id=context.patch_id, cell_id=cell,
                    is_core=cell in context.core_cell_ids and cell in target,
                    nuclear_umi=float(nuclear_totals[ci]), nuclear_genes=int(nuclear[ci].count_nonzero()),
                    nuclear_bins=int(np.sum(inp.locked_owner_cell_index[nuclear_bins] == ci)),
                    profile_reliability=float(baseline.cell_profile_reliability[ci]),
                    prior_count=float(baseline.relative_profile_prior_count), gt_area_um2=4*area.get(cell, 0),
                    n_competitors=len(neighbors), competitor_mean_nuclear_umi=float(nuclear_totals[neighbors].mean()) if len(neighbors) else np.nan,
                    competitor_depth_ratio=float(nuclear_totals[neighbors].mean()/(nuclear_totals[ci]+1)) if len(neighbors) else np.nan,
                    oracle_profile_total_variation=float(.5*np.abs(baseline.cell_expression_profile[ci]-oracle[ci]).sum())))
            cell_indices = {c:i for i,c in enumerate(inp.cell_ids)}
            for bi, barcode in enumerate(inp.barcode_ids):
                gt = truth_owner.get(barcode)
                start, stop = inp.candidate_row_splits[bi:bi+2]
                pairs = np.arange(start, stop)
                candidate_cells = inp.candidate_cell_index[pairs]
                gt_pairs = pairs[candidate_cells == cell_indices.get(gt, -1)]
                chosen = pairs[candidate_cells == baseline.top1_cell_index[bi]][0]
                rec = dict(generator=generator, patch_id=context.patch_id, barcode=barcode,
                    gt_cell_id=gt, gt_core=gt in context.core_cell_ids and gt in target,
                    gt_in_candidates=len(gt_pairs)==1, nuclear_locked=bool(inp.is_nuclear_locked[bi]),
                    counts=float(observed_total[bi]), n_candidates=len(pairs),
                    confidence=float(baseline.expression_confidence[bi]),
                    entropy=float(baseline.normalized_entropy[bi]),
                    top_equals_last_score_max=chosen == pairs[np.argmax(baseline.final_combined_score[pairs])])
                rec.update({a: str(v[bi]) for a,v in owners.items()})
                if gt is not None:
                    for arm in arms[1:]:
                        rec[arm+'_vs_distance'] = transition(owners['distance_only'][bi], owners[arm][bi], gt)
                    for arm in arms[2:]:
                        rec[arm+'_vs_frozen'] = transition(owners['frozen_em'][bi], owners[arm][bi], gt)
                if len(gt_pairs):
                    gp = gt_pairs[0]
                    for name, values in [('spatial',spatial),('type',type_score),('cell',cell_score),('combined',baseline.final_combined_score)]:
                        rec['chosen_minus_gt_'+name] = float(values[chosen]-values[gp])
                bin_records.append(rec)
            pair_records.append(pd.DataFrame(dict(generator=generator, patch_id=context.patch_id,
                barcode=np.asarray(inp.barcode_ids)[inp.pair_bin_index],
                candidate_cell_id=np.asarray(inp.cell_ids)[inp.candidate_cell_index],
                distance_um=inp.pair_distance_um, spatial_score=spatial, type_score=type_score,
                cell_score=cell_score, combined_score=baseline.final_combined_score,
                responsibility=baseline.responsibility)))
    finally:
        dataset.close()
    evaluation_tasks = []
    for arm, merger in mergers.items():
        rows, conflicts = merger.finalize()
        validate_unique_ownership(rows, nuclear_owners=merger.nuclear_owners)
        path = out/'methods'/arm
        frame = pd.DataFrame(rows)
        frame.to_csv(path/'assignments.csv', index=False)
        pd.DataFrame(conflicts).to_csv(out/'subdiagnostic'/f'{arm}_merge_conflicts.csv', index=False)
        if arm in ('frozen_em','distance_only'):
            beta = '1p5' if arm == 'frozen_em' else '0p0'
            old = pd.read_csv(original/f'methods/em_and_distance/variants/alpha_1p0__beta_{beta}__kappa_2p0__profile_damping_0p0/assignments.csv', dtype={'cell_id':str})
            if set(zip(frame.barcode, frame.cell_id.astype(str))) != set(zip(old.barcode, old.cell_id)):
                raise AssertionError(f'{generator} {arm} does not reproduce frozen ownership')
        evaluation_tasks.append((arm, [
            '--assignments-csv', path/'assignments.csv', '--method-name', arm, '--method-label', arm,
            '--external-nuclear-bins-path', source/'benchmark/common_confident_nuclear_bins.csv.gz',
            *comparison.evaluation_args(context_path, out, arm)]))
    pd.DataFrame(cell_records).to_csv(out/'subdiagnostic/cell_covariates.csv', index=False)
    pd.DataFrame(bin_records).to_csv(out/'subdiagnostic/bin_transitions.csv.gz', index=False)
    pd.concat(pair_records).to_parquet(out/'subdiagnostic/frozen_pair_scores.parquet', index=False)
    comparison.dump(out/'subdiagnostic/solver_checks.json', run_checks)
    def evaluate(task):
        arm, command_args = task
        comparison.command(out, arm+'_evaluation', 'scripts/evaluate_assignment_table.py', command_args)
    with ThreadPoolExecutor(max_workers=int(evaluation_workers)) as executor:
        list(executor.map(evaluate, evaluation_tasks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--profile-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.source_dir = args.source_dir.resolve(strict=True)
    args.profile_dir = args.profile_dir.resolve(strict=True)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_experiment_manifest(args.output_dir, config=vars(args),
        inputs={'source_comparison':args.source_dir/'method_comparison.csv', 'reference':comparison.REFERENCE,
            'A_profiles':args.profile_dir/'A_current_20260916.h5', 'B_profiles':args.profile_dir/'B_library_offset_20260916.h5',
            'A_mapping':args.source_dir/'A_current/profile_assignment.csv', 'B_mapping':args.source_dir/'B_library_offset/profile_assignment.csv'},
        protocol={'arms':ARMS,'oracle':'replace initial physical-cell profile only; preserve nuclear-derived lambda and q updates',
            'scope':'modeled_only, same four development patches, 43 core cells; no held-out claim',
            'frozen':'alpha=1 beta=1.5 kappa=2 profile_damping=0; no PPO or tuning',
            'diagnostic_grain':'patch-local physical bins including margin owners; official IoU uses existing merged candidate-clipped evaluator'})
    for generator in ('A_current','B_library_offset'):
        run_generator(args, generator)


if __name__ == '__main__':
    main()
