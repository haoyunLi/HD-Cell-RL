#!/usr/bin/env python3
"""Package saved observations for the slide demos; never run inference or training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'runs/benchmarks/em_donor_disjoint_unseen_patches_20260903T192649Z'
AUDIT = ROOT / 'runs/benchmarks/pseudo_realism_20260908T012500Z'
REWARD = ROOT / ('runs/benchmarks/colorectal_nucleus_based_multi_owner_v2_43cells_20260826T210246Z/'
    'evaluations/hd_cell_rl_em_plus_frozen_rl_v2_43cells_20260826T212325Z/'
    'diagnostics/reward_attribution/replace_reward_attribution.csv')


def decode(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def panel_totals(path, selected_barcodes, panel):
    """Read only requested H5 cell columns and sum the exact aligned gene panel."""
    with h5py.File(path) as f:
        m = f['matrix']
        genes = decode(m['features/name'][:])
        lookup = {b: i for i, b in enumerate(decode(m['barcodes'][:]))}
        keep = np.array([g in panel for g in genes])
        ptr = m['indptr'][:]
        totals = []
        for barcode in selected_barcodes:
            i = lookup[barcode]
            start, end = int(ptr[i]), int(ptr[i + 1])
            totals.append(int(m['data'][start:end][keep[m['indices'][start:end]]].sum()))
    return np.asarray(totals, dtype=np.int64)


def build_data(validation_repo):
    bins = pd.read_parquet(AUDIT / 'barcodes.parquet')
    assert bins.barcode.is_unique and bins.matrix_col_index.is_unique
    gt = pd.read_parquet(BENCH / 'inputs/fractional_gt_contributing_cells.parquet')
    coverage = gt.groupby('barcode_index').cell_coverage_fraction.sum().reindex(bins.matrix_col_index, fill_value=0).to_numpy()
    dominant = (gt.sort_values(['barcode_index', 'cell_coverage_fraction', 'cell_id'], ascending=[True, False, True])
                .drop_duplicates('barcode_index').set_index('barcode_index').cell_id
                .reindex(bins.matrix_col_index, fill_value=-1).to_numpy(dtype=np.int64))
    seeds = (gt[gt.nuclear_fraction > 0].sort_values(['barcode_index', 'nuclear_fraction', 'cell_id'], ascending=[True, False, True])
             .drop_duplicates('barcode_index').set_index('barcode_index').cell_id
             .reindex(bins.matrix_col_index, fill_value=-1).to_numpy(dtype=np.int64))
    names = ['source_high_depth', 'depth_uniform', 'real']
    counts = {}
    for name in names:
        matrix = sparse.load_npz(AUDIT / f'matrices/{name}.npz')
        assert matrix.shape == (len(bins), 18132)
        counts[name] = np.asarray(matrix.sum(axis=1)).ravel().astype(np.int64)
    grid = bins.barcode.str.extract(r's_002um_(\d+)_(\d+)').astype(int).to_numpy()
    patches = []
    for patch_id in sorted(bins.patch_id.unique()):
        idx = np.flatnonzero(bins.patch_id.to_numpy() == patch_id)
        origin = grid[idx].min(axis=0)
        local = grid[idx] - origin
        data = [[int(local[j, 0]), int(local[j, 1]), round(float(coverage[i]), 5),
                 int(dominant[i]), int(seeds[i]), *[int(counts[n][i]) for n in names]] for j, i in enumerate(idx)]
        ba = {}
        for name in names:
            inside = coverage[idx] > 0
            nonzero = counts[name][idx] > 0
            ba[name] = float((nonzero[inside].mean() + (~nonzero[~inside]).mean()) / 2)
        patches.append(dict(patch_id=patch_id, partition=str(bins.iloc[idx[0]].partition),
                            rows=int(local[:, 0].max() + 1), cols=int(local[:, 1].max() + 1),
                            n_bins=len(idx), bins=data, balanced_accuracy=ba))

    rewards = pd.read_csv(REWARD)
    fields = ['expression_confidence', 'expression_compatibility', 'distance_penalty',
              'overlap_penalty', 'neighbor_support', 'shape_prior',
              'penalty_rounding_residual', 'numeric_residual']
    cases = []
    for outcome in ['corrected', 'damaged']:
        valid = rewards[(rewards.outcome == outcome) & rewards.inside_core
                        & rewards.old_matched_gt_cell_id.notna() & rewards.new_matched_gt_cell_id.notna()]
        row = valid.iloc[0]  # Saved CSV order; do not select the largest effect.
        components = {name: float(row[name]) for name in fields}
        assert abs(sum(components.values()) - float(row.reward)) < 1e-10
        cases.append(dict(outcome=outcome, patch_id=str(row.patch_id), step=int(row.step_index),
                          barcode=str(row.barcode), old_cell_id=int(row.old_cell_id), new_cell_id=int(row.new_cell_id),
                          gt_owner=int(row.gt_owner_cell_id), old_gt=int(row.old_matched_gt_cell_id),
                          new_gt=int(row.new_matched_gt_cell_id), reward=float(row.reward), components=components,
                          rank=int(row.chosen_action_reward_rank), available=int(row.available_action_count)))

    # An actual within-type distribution, not a Gaussian reconstructed from CVs.
    reference = pd.read_csv(BENCH / 'simulation/scdesign3_generator_disjoint/model_reference_cell_metadata.tsv', sep='\t')
    assignment = pd.read_csv(BENCH / 'assignment/ground_truth_cell_to_synthetic_scdesign3.csv')
    panel = set(pd.read_csv(AUDIT / 'genes.csv').gene)
    raw_bars = reference.loc[reference.cell_type == 'Cycling T', 'Cell'].tolist()
    syn_bars = assignment.loc[assignment.cell_type == 'Cycling T', 'sc_cell_barcode'].tolist()
    raw = panel_totals(validation_repo / 'data/single_cell/colorectal/CRC_GSE166555_raw_tumor.h5', raw_bars, panel)
    syn = panel_totals(BENCH / 'simulation/scdesign3_generator_disjoint/scdesign3_synthetic_counts.h5', syn_bars, panel)
    profile = dict(cell_type='Cycling T', raw=sorted(raw.tolist()), synthetic=sorted(syn.tolist()), panel_genes=len(panel))
    summary = pd.read_csv(AUDIT / 'comparison/source_profile_depth.csv')
    for label, totals in [('raw_generator_scRNA', raw), ('scdesign3', syn)]:
        ref = summary[(summary.cell_type == 'Cycling T') & (summary.source == label)].iloc[0]
        assert len(totals) == int(ref.n_cells)
        assert np.isclose(totals.mean(), ref.mean_umi)
        assert np.isclose(totals.std() / totals.mean(), ref.umi_cv)

    support_rows = pd.read_csv(AUDIT / 'comparison/all_realizations_support.csv')
    support_realization = {name: float(support_rows[(support_rows.variant == name)
                           & (support_rows.seed == (20260907 if name == 'depth_uniform' else 0))]
                           .iloc[0].support_detects_gt_coverage_balanced_accuracy) for name in names}
    methods = pd.read_csv(BENCH / 'comparison/method_comparison.csv')
    comparison = methods[['condition', 'method', 'n_cells', 'mean_pred_iou', 'mean_gene_spearman_r']].to_dict('records')
    return dict(version='20260908-explanation', patches=patches, reward_cases=cases, profile=profile, comparison=comparison,
                support_realization=support_realization,
                sources=dict(support=str(AUDIT.relative_to(ROOT)), rewards=str(REWARD.relative_to(ROOT)),
                             comparison=str((BENCH / 'comparison/method_comparison.csv').relative_to(ROOT))),
                conventions=dict(bin_fields=['array_row_offset', 'array_col_offset', 'pseudo_coverage', 'pseudo_dominant_owner',
                                             'nuclear_dominant_owner', *names],
                                 support='All four saved contexts; barcode-array coordinates; absent array bins are missing, not zero.',
                                 reward_selection='First CSV row per outcome with inside_core=true and both matched GT IDs present.',
                                 reward_geometry='Transfer diagram is schematic; IDs and weighted deltas are measured historical values.',
                                 profile='Actual panel UMI totals; donor-disjoint generator only; unequal sample sizes retained.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-repo', type=Path, default=ROOT.parent / 'Bin2Cell_Validation')
    parser.add_argument('--output', type=Path, default=ROOT / 'web/progress-report/assets/story-data.js')
    args = parser.parse_args()
    data = build_data(args.validation_repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('// Derived from saved results; regenerate with scripts/build_progress_report_story_data.py\n'
                           + 'window.PROGRESS_STORY_DATA = ' + json.dumps(data, separators=(',', ':'), allow_nan=False) + ';\n')
    print(f'Packaged {len(data["patches"])} contexts, two historical actions and actual profile distributions: {args.output}')


if __name__ == '__main__':
    main()
