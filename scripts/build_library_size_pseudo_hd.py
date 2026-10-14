#!/usr/bin/env python3
"""Paired A/B pseudo-HD: preserve source-cell totals through common binomial capture."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import sparse
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_regional_depth_calibration import prepare_inputs, load_profiles
from scripts.run_pseudo_realism_experiment import load_h5, place_profiles, vec
from scripts.build_source_separated_pseudo import align_genes, quality, acceptance, dump
from hd_cell_rl.pseudo_components import combine_components
from hd_cell_rl.experiment_protocol import write_experiment_manifest


def select_matched_profiles(target_types, source_types, seed):
    """Select unique source cells within each type, using identical rows for A and B."""
    rng = np.random.default_rng(seed)
    selected = np.empty(len(target_types), dtype=np.int64)
    for label in sorted(set(target_types)):
        targets = np.flatnonzero(np.asarray(target_types) == label)
        pool = np.flatnonzero(np.asarray(source_types) == label)
        if len(pool) < len(targets):
            raise ValueError(f'insufficient unique profiles for {label}')
        selected[targets] = rng.choice(pool, len(targets), replace=False)
    return selected


def common_capture_probability(target_counts, proposed_counts, calibration):
    target = float(target_counts[calibration].sum())
    source = float(proposed_counts[calibration].sum())
    if not np.isfinite(target + source) or source <= 0 or target <= 0 or target > source:
        raise ValueError('calibration requires positive counts and genuine thinning (0 < p <= 1)')
    return target / source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ab-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20260916)
    args = p.parse_args()
    from scripts import run_source_separated_method_comparison as prior
    validation = ROOT.parent / 'Bin2Cell_Validation'
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    (out/'subdiagnostic').mkdir()
    cfg = yaml.safe_load((ROOT/'configs/pseudo_regional_depth.calibration.yaml').read_text())
    data = prepare_inputs(prior.AUDIT, prior.BENCH, validation)
    _, _, _, mod, p0, flags = load_profiles(data, prior.BENCH, validation)
    bins = data['bins']; patch = bins.patch_id.to_numpy(); names = sorted(set(patch))
    data_sources = {}; profiles = {}; paths = {}
    for method in ('A_current', 'B_library_offset'):
        path = args.ab_dir/f'{method}_{args.seed}.h5'
        matrix, barcodes, genes = load_h5(path)
        metadata = pd.read_csv(args.ab_dir/f'{method}_{args.seed}_per_cell.csv')
        if not np.array_equal(barcodes, metadata.cell_id):
            raise ValueError('source barcode order differs from per-cell metadata')
        if method == 'A_current':
            selected = select_matched_profiles(data['types'], metadata.cell_type.to_numpy(), args.seed)
            source_types = metadata.cell_type.to_numpy(); source_barcodes = barcodes.copy()
        elif not (np.array_equal(source_types, metadata.cell_type) and np.array_equal(source_barcodes, barcodes)):
            raise ValueError('A/B source identities/types do not match')
        matrix = align_genes(matrix[selected], genes, data['genes'])
        profiles[method] = matrix; paths[method] = path.resolve()
        target = out/method; target.mkdir()
        assignment = data['assignment'].reset_index().copy()
        assignment['sc_cell_barcode'] = barcodes[selected]
        assignment['synthetic_scdesign3_barcode'] = barcodes[selected]
        assignment['expression_origin'] = method
        assignment.to_csv(target/'profile_assignment.csv', index=False)
        assignment[['cell_id','cell_type','sc_cell_barcode']].assign(
            source_counts=vec(matrix.sum(1)), source_genes=np.diff(matrix.indptr)
        ).to_csv(target/'source_cell_counts.csv', index=False)
        data_sources[method] = path
    arguments = dict(cells=data['cells'], groups={int(c):g for c,g in data['gt'].groupby('cell_id')},
        local_by_global={int(b):i for i,b in enumerate(bins.matrix_col_index)}, n_bins=len(bins),
        membrane_flags=flags, p0=p0, target_nuclear=cfg['nuclear_fraction'],
        membrane_strength=cfg['membrane_strength'], uniform=False, seed=args.seed, source_module=mod)
    proposed_a, _ = place_profiles(profiles['A_current'], **arguments)
    rows, cell_rows, checks, capture = [], [], [], []
    for heldout in names:
        fold = 'leave_out_' + heldout
        calibration = patch != heldout
        source = prior.REGIONAL/'realizations'/fold/'uniform_budget'
        old_modeled = sparse.load_npz(source/'modeled_cell.npz')
        unknown = sparse.load_npz(source/'unmodeled_biological.npz')
        technical = sparse.load_npz(source/'technical_background.npz')
        if technical.nnz or unknown.shape != proposed_a.shape:
            raise ValueError('fixed source components do not match the declared protocol')
        probability = common_capture_probability(vec(old_modeled.sum(1)), vec(proposed_a.sum(1)), calibration)
        print('CAPTURE', heldout, probability, flush=True)
        for method, matrix in profiles.items():
            diagnostics = {}
            modeled, trace = place_profiles(matrix, **arguments,
                capture_probability=np.full(len(bins), probability), capture_diagnostics=diagnostics)
            observed = combine_components(modeled, unknown, technical)
            directory = out/method/'regional/realizations'/fold/'uniform_budget'
            directory.mkdir(parents=True)
            for name, value in [('modeled_cell',modeled),('unmodeled_biological',unknown),
                                ('technical_background',technical),('observed',observed),('modeled_source_bin_counts',trace)]:
                sparse.save_npz(directory/(name+'.npz'), value)
            capture.append(dict(method=method,fold=fold,probability=probability,**diagnostics))
            data['assignment'].reset_index()[['cell_id','cell_type']].assign(
                source_counts=vec(matrix.sum(1)),captured_in_crop_counts=vec(trace.sum(1))
            ).to_csv(directory/'modeled_sources.csv',index=False)
            audit_bins = bins.copy(); audit_bins['partition'] = np.where(patch==heldout,'audit','excluded')
            parts = np.where(data['cell_patch']==heldout,'audit','excluded')
            for condition, observed_part in [('modeled_only',modeled),('source_separated',observed)]:
                metrics, cells = quality(observed_part, data['real'], audit_bins, data['region'], parts,
                    data['core'], data['types'], method, args.seed)
                rows += [dict(condition=condition,patch_id=heldout,**r) for r in metrics]
                cell_rows += [dict(condition=condition,patch_id=heldout,cell_id=int(data['cells'][r['cell_index']]),**r) for r in cells]
                checks.append(dict(method=method,condition=condition,patch_id=heldout,**acceptance(metrics,cfg['acceptance'])))
    bins.to_parquet(out/'barcodes.parquet',index=False)
    pd.DataFrame(rows).to_csv(out/'multiscale_realism.csv',index=False)
    pd.DataFrame(cell_rows).to_csv(out/'subdiagnostic/cell_region_realism.csv',index=False)
    dump(out/'subdiagnostic/capture_checks.json',capture)
    dump(out/'realism_acceptance.json',dict(checks=checks,passed=all(c['passed'] for c in checks)))
    write_experiment_manifest(out,config=dict(seed=args.seed,source_selection_seed=args.seed,
        nuclear_fraction=cfg['nuclear_fraction'],membrane_strength=cfg['membrane_strength']),
        inputs={**data_sources,'old_regional_manifest':prior.REGIONAL/'experiment_manifest.json',
                'geometry':prior.FRACTIONAL,'reference':prior.REFERENCE,'ab_protocol':args.ab_dir/'protocol.json'},
        protocol=dict(role='paired development benchmark, four previously inspected patches, one source seed',
            source_profiles='unique type-matched rows shared by A/B; full source totals preserved before capture',
            capture='per leave-out fold: scalar probability calibrated on A and previous modeled calibration counts; identical p for A and B',
            no_per_cell_target_normalization=True,unknown='exact saved component retained for both A/B',
            technical_background=False,no_EM_RL_changes=True,
            comparison_to_old_run='not a library-only causal comparison; capture budgets changed. Use newly generated matched A/B.',
            limitations='source-depth independent of physical area; unmodeled RNA budget remains a stress-test assumption'))
    print('COMPLETE',out,flush=True)


if __name__ == '__main__':
    main()
