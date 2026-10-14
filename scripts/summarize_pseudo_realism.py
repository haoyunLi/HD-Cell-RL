#!/usr/bin/env python3
"""Consolidate data-only realism controls and draw an exploratory overview."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.experiment_dir
    main_metrics = pd.read_csv(out / 'comparison/metrics.csv')
    extra_metrics = pd.read_csv(out / 'profile_controls/metrics.csv')
    metrics = pd.concat([main_metrics, extra_metrics], ignore_index=True)
    support = pd.concat([pd.read_csv(out / p) for p in ('comparison/support_and_compartments.csv', 'profile_controls/support_and_compartments.csv')], ignore_index=True)
    if support.duplicated(['variant', 'seed']).any():
        raise ValueError('duplicate realizations')
    if metrics.duplicated(['variant', 'seed', 'partition', 'scale_um']).any():
        raise ValueError('duplicate measurement rows')
    if not (metrics.groupby(['variant', 'seed']).size() == 6).all():
        raise ValueError('missing scale or partition measurements')
    expected_shape = (len(pd.read_parquet(out / 'barcodes.parquet')), len(pd.read_csv(out / 'genes.csv')))
    for path in list((out / 'matrices').glob('*.npz')) + list((out / 'profile_controls').glob('*.npz')):
        x = sparse.load_npz(path)
        if x.shape != expected_shape or not np.isfinite(x.data).all() or np.any(x.data < 0) or np.any(x.data != np.rint(x.data)):
            raise ValueError(f'invalid saved matrix: {path}')
    metrics.to_csv(out / 'comparison/all_realizations_metrics.csv', index=False)
    support.to_csv(out / 'comparison/all_realizations_support.csv', index=False)
    values = ['depth_ks', 'genes_ks', 'gene_js', 'mean_ratio_to_real', 'sim_neighbor_cosine', 'reference_positive_residual_fraction']
    summary = metrics.groupby(['variant', 'partition', 'scale_um'])[values].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(out / 'comparison/configuration_summary.csv', index=False)
    support_summary = support.groupby('variant').agg(support_association=('support_detects_gt_coverage_balanced_accuracy', 'mean'),
        outside_mask_umi_fraction=('fraction_counts_outside_pseudo_masks', 'mean'))
    compact = summary.xs(('audit', 2), level=('partition', 'scale_um'))[['depth_ks_mean', 'gene_js_mean']].join(support_summary)
    compact.to_csv(out / 'comparison/audit_2um_summary.csv')
    print(compact.round(5).to_string())
    print('\nProfile library controls:')
    profiles = pd.read_csv(out / 'profile_controls/profile_depth.csv')
    print(profiles[profiles.n_cells >= 10].groupby(['variant', 'cell_type']).umi_cv.mean().to_string())
    colors = ['#202020', '#dc6b3b', '#4284b5', '#479c78', '#9874ab']
    variants = [('real', 'Real HD'), ('source_high_depth', 'Current pseudo'), ('depth_uniform', 'Depth matched'),
                ('local_transport_2um_0.1', '10% local transport'), ('capture_plus_unassigned_transport', 'Unassigned-tissue control')]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    paths = list((out / 'matrices').glob('*.npz'))
    for (variant, label), color in zip(variants, colors):
        path = next(p for p in paths if p.stem == variant or p.stem == variant + '.seed20260907')
        totals = np.asarray(sparse.load_npz(path).sum(1)).ravel()
        totals = np.sort(totals)
        axes[0, 0].plot(np.log1p(totals), np.arange(1, len(totals)+1)/len(totals), label=label, color=color)
    axes[0, 0].set(xlabel='log(1 + counts per 2 µm bin)', ylabel='Cumulative fraction', title='A  Matching total depth does not match the distribution')
    axes[0, 0].legend(fontsize=9, loc='lower right')
    depth = pd.read_csv(out / 'comparison/source_profile_depth.csv')
    selected = depth[(depth.source == 'scdesign3') & (depth.n_cells >= 10)].cell_type.tolist()
    positions = np.arange(len(selected))
    for i, (source, label, color) in enumerate([('raw_generator_scRNA', 'Generator scRNA', '#4284b5'), ('scdesign3', 'Current scDesign3', '#dc6b3b')]):
        d = depth[depth.source == source].set_index('cell_type').loc[selected]
        axes[0, 1].bar(positions + (i-.5)*.36, d.umi_cv, width=.36, label=label, color=color)
    axes[0, 1].set_xticks(positions, [s.replace('Inflammatory ', 'Inflammatory\n') for s in selected], rotation=15)
    axes[0, 1].set(ylabel='Within-type library-size CV', title='B  Simulated cell libraries are much less variable')
    axes[0, 1].legend()
    chosen = [v for v, _ in variants]
    labels = [l for _, l in variants]
    axes[1, 0].barh(labels, support_summary.loc[chosen, 'support_association'], color=colors)
    axes[1, 0].axvline(.5, color='gray', linestyle='--')
    axes[1, 0].set(xlim=(0, 1.03), xlabel='Balanced accuracy: positive counts → pseudo-mask coverage', title='C  Positive support is a geometric shortcut')
    for (variant, label), color in zip(variants, colors):
        d = metrics[(metrics.variant == variant) & (metrics.partition == 'audit')].groupby('scale_um').sim_neighbor_cosine.mean()
        axes[1, 1].plot(d.index, d, marker='o', color=color, label=label)
    axes[1, 1].set(xlabel='Bin width (µm); complete blocks only', ylabel='Adjacent-bin expression cosine', title='D  Residual multiscale spatial mismatch', xticks=[2, 8, 16])
    fig.suptitle('Pseudo HD realism audit — diagnostic patches, not a held-out method benchmark', fontsize=16)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.title.set_fontsize(11)
    figure_dir = out / 'comparison/figures'
    figure_dir.mkdir(exist_ok=True)
    fig.savefig(figure_dir / 'realism_overview.png', dpi=160)
    fig.savefig(figure_dir / 'realism_overview.svg')
    source_files = ['hd_cell_rl/pseudo_realism.py', 'scripts/run_pseudo_realism_experiment.py',
                    'scripts/run_pseudo_profile_controls.py', 'scripts/audit_hd_tissue_background.py',
                    'scripts/audit_pseudo_profile_separation.py', 'scripts/summarize_pseudo_realism.py', 'tests/test_pseudo_realism.py']
    provenance = dict(n_configurations=int(support.variant.nunique()), n_realizations=len(support),
                      no_em_or_ppo_training=True, source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_files})
    (out / 'comparison/experiment_manifest.json').write_text(json.dumps(provenance, indent=2)+'\n')


if __name__ == '__main__':
    main()
