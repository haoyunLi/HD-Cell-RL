#!/usr/bin/env python3
"""Frozen five-method comparison of saved source-separated pseudo-HD counts.

Each development patch uses its own leave-out calibration realization. The
paired control removes unmodeled RNA without redrawing modeled-cell counts.
No generator, EM, policy, or method hyperparameters are fitted here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BENCH = ROOT / 'runs/benchmarks/em_donor_disjoint_unseen_patches_20260903T192649Z'
AUDIT = ROOT / 'runs/benchmarks/pseudo_realism_20260908T012500Z'
REGIONAL = ROOT / 'runs/benchmarks/regional_depth_blocked_20260909/experiment'
INTERMEDIATE = ROOT / 'workspace_outputs/colorectal_nucleus_based_multi_owner_v2/intermediate'
GT = INTERMEDIATE / 'ground_truth/dominant_owner/colorectal_nucleus_based_multi_owner_v2_square_002um_dominant_cell_bins.csv.gz'
NUCLEAR = INTERMEDIATE / 'ground_truth/dominant_owner/colorectal_nucleus_based_multi_owner_v2_square_002um_dominant_nuclear_bins.csv.gz'
FRACTIONAL = BENCH / 'inputs/fractional_gt_contributing_cells.parquet'
FRACTIONAL_CELL_BINS = INTERMEDIATE / 'ground_truth/colorectal_nucleus_based_multi_owner_v2_square_002um_cell_bins.csv.gz'
CHECKPOINT = ROOT / 'runs/colorectal_nucleus_based_patch_overfit4_w5_validated_20260825T114854Z/checkpoints/best_model.pt'
BASE = ROOT / 'configs/ppo_training.patch_overfit4_w5_validated.multi_owner_v2.resolved.yaml'
REFERENCE = BENCH / 'reference/reference_counts_donor_disjoint.npz'
PATCHES = BENCH / 'selection/test_patches_index.csv'
NUCLEI = BENCH / 'episode_context/nuclei_patch_context.parquet'
PROFILE_ASSIGNMENT = BENCH / 'assignment/ground_truth_cell_to_synthetic_scdesign3.csv'
PROFILE_H5 = BENCH / 'simulation/scdesign3_generator_disjoint/scdesign3_synthetic_counts.h5'
IMAGE = ROOT / 'workspace_outputs/colorectal_nucleus_based/intermediate/cropped_visium_hd_colorectal_nucleus_based.png'
CONDITIONS = ('modeled_only', 'source_separated')
METRICS = ('pred_iou', 'pred_fractional_iou', 'pred_precision', 'pred_recall', 'gene_spearman_r')
GPU_OFFSET = 0
EM_MAX_ITERATIONS = 15
AXIS_SOURCE = None


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')


def common_seed_path(out):
    return out/'common_confident_nuclear_bins.csv.gz'


def prepare_common_seeds(out):
    """Use inference's confidence mask, not all dominant-owner nuclear bins."""
    import pandas as pd
    from hd_cell_rl.nuclear_seeds import load_confident_nuclear_rows
    cfg = yaml.safe_load((out/'modeled_only/episode_build.yaml').read_text())
    metadata = Path(cfg['inputs']['bins_path'])
    confident = load_confident_nuclear_rows(metadata)
    confident = confident.dropna(subset=['dominant_cell_id']).copy()
    confident['cell_id'] = confident.dominant_cell_id.astype('int64').astype(str)
    external = pd.read_csv(NUCLEAR, dtype={'cell_id':str})
    comparison = confident[['barcode','cell_id']].merge(
        external[['barcode','cell_id']].rename(columns={'cell_id':'external_cell_id'}),
        on='barcode', how='left', validate='one_to_one')
    differences = comparison[comparison.cell_id != comparison.external_cell_id]
    # Episode metadata is the inference authority. Do not let a GT-derived
    # dominant-owner export replace its owner or expand its confidence mask.
    merged = confident[['barcode','cell_id']].copy()
    merged['is_nuclear'] = True
    if common_seed_path(out).exists():
        raise FileExistsError(common_seed_path(out))
    merged.to_csv(common_seed_path(out),index=False)
    selected = set(pd.read_parquet(out/'barcodes.parquet').barcode)
    differences[differences.barcode.isin(selected)].to_csv(
        out/'subdiagnostic/selected_seed_owner_disagreement.csv',index=False)
    dump(out/'subdiagnostic/common_seed_audit.json',dict(metadata=str(metadata),
        n_external_nuclear=len(external), n_confident_nuclear=len(merged),
        n_selected_external_nuclear=int(external.barcode.isin(selected).sum()),
        n_selected_confident_nuclear=int(merged.barcode.isin(selected).sum()),
        n_confident_owner_disagreement=len(differences),
        n_selected_confident_owner_disagreement=int(differences.barcode.isin(selected).sum()),
        n_selected_missing_from_external=int((differences.barcode.isin(selected) & differences.external_cell_id.isna()).sum()),
        n_selected_different_existing_owner=int((differences.barcode.isin(selected) & differences.external_cell_id.notna()).sum()),
        rule='has_nuclear_annotation and not ambiguous_nuclear_assignment',
        note='All methods use confident seeds; ambiguous nuclear bins are not locked.'))


def assemble_components(bins, regional):
    """Keep stable row order and never mix a patch's calibration/audit outputs."""
    import numpy as np
    from scipy import sparse
    from hd_cell_rl.pseudo_count_diagnostics import integer_csr
    if bins.barcode.duplicated().any() or bins.matrix_col_index.duplicated().any():
        raise ValueError('physical barcodes and global indices must be unique')
    chunks = {name: [] for name in ('modeled_cell', 'unmodeled_biological', 'technical_background', 'observed')}
    order = []
    sources = {}
    for patch in sorted(bins.patch_id.unique()):
        indices = np.flatnonzero(bins.patch_id.to_numpy() == patch)
        directory = regional / 'realizations' / ('leave_out_' + patch) / 'uniform_budget'
        local = {}
        for name in chunks:
            path = directory / (name + '.npz')
            x = integer_csr(sparse.load_npz(path))
            if x.shape[0] != len(bins):
                raise ValueError('source matrix does not match saved barcode axis')
            local[name] = x[indices]
            chunks[name].append(local[name])
            sources[patch + '_' + name] = path
        if (local['observed'] - local['modeled_cell'] - local['unmodeled_biological'] - local['technical_background']).nnz:
            raise ValueError('source components do not conserve observed counts')
        if local['technical_background'].nnz:
            raise ValueError('this protocol expects technical background disabled')
        order.extend(indices)
    inverse = np.argsort(order)
    return {name: sparse.vstack(values, format='csr')[inverse] for name, values in chunks.items()}, sources


def prepare(out, *, dataset_root=None, protocol_overrides=None):
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from hd_cell_rl.experiment_protocol import write_experiment_manifest
    from scripts.build_patch_capture_stress_dataset import write_dataset
    out.mkdir(parents=True, exist_ok=False)
    diag = out / 'subdiagnostic'
    diag.mkdir()
    bins = pd.read_parquet(AUDIT / 'barcodes.parquet')
    genes = pd.read_csv(AUDIT / 'genes.csv').gene.astype(str).tolist()
    components, sources = assemble_components(bins, REGIONAL)
    bins.to_parquet(out / 'barcodes.parquet', index=False)
    pd.Series(genes, name='gene').to_csv(out / 'genes.csv', index=False)
    template = yaml.safe_load((BENCH / 'episode_context/episode_build.capture_thinned.yaml').read_text())
    metadata = pd.read_parquet(ROOT / template['inputs']['bins_path'], columns=['matrix_col_index', 'barcode'])
    source = AXIS_SOURCE or BENCH / 'datasets/source_pseudo_hd_generator_disjoint'
    features = pd.read_csv(source / 'filtered_feature_bc_matrix/features.tsv.gz', sep='\t', header=None)
    barcodes = pd.read_csv(source / 'filtered_feature_bc_matrix/barcodes.tsv.gz', sep='\t', header=None)[0].astype(str)
    positions = pd.read_parquet(source / 'spatial/tissue_positions.parquet')
    global_bins = bins.matrix_col_index.to_numpy(np.int64)
    if not np.array_equal(barcodes.iloc[global_bins].to_numpy(), bins.barcode.to_numpy()):
        raise ValueError('global barcode mapping changed')
    if not np.array_equal(metadata.set_index('matrix_col_index').loc[global_bins, 'barcode'].to_numpy(), bins.barcode.to_numpy()):
        raise ValueError('episode metadata barcode mapping changed')
    if features[1].astype(str).duplicated().any():
        raise ValueError('source gene symbols are not unique')
    gene_indices = pd.Index(features[1].astype(str)).get_indexer(genes)
    if np.any(gene_indices < 0):
        raise ValueError('missing source genes')
    counts = []
    for condition, component in [('modeled_only', 'modeled_cell'), ('source_separated', 'observed')]:
        x = components[component]
        if x.shape != (len(bins), len(genes)):
            raise ValueError('gene axis length mismatch')
        coo = x.tocoo()
        full = sparse.coo_matrix((coo.data, (gene_indices[coo.col], global_bins[coo.row])),
                                 shape=(len(features), len(barcodes)), dtype=np.int64)
        directory = out / condition
        directory.mkdir()
        dataset_path = directory/'dataset' if dataset_root is None else Path(dataset_root)/condition/'dataset'
        write_dataset(output_dir=dataset_path, source_dir=source, positions=positions,
                      selected_barcodes=set(bins.barcode), matrix=full)
        if dataset_root is not None:
            (directory/'dataset').symlink_to(dataset_path.resolve(),target_is_directory=True)
        # Keep full global axes for episode construction and compact axes for diagnostics.
        compact_path = directory/'observed_compact.npz' if dataset_root is None else Path(dataset_root)/condition/'observed_compact.npz'
        sparse.save_npz(compact_path, x)
        if dataset_root is not None:
            (directory/'observed_compact.npz').symlink_to(compact_path.resolve())
        cfg = json.loads(json.dumps(template))
        cfg['run'].update(name=condition + '_episode_build', output_root=str(directory / 'episodes'))
        cfg['inputs']['bins_path'] = str((ROOT / template['inputs']['bins_path']).resolve())
        cfg['inputs']['expression'].update(matrix_path=str(directory / 'dataset/filtered_feature_bc_matrix'),
            reference_npz_path=str(REFERENCE),
            n_workers=12, filter_empty_nuclear_cells=False, min_nuclear_expression_sum=0.)
        (directory / 'episode_build.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        for patch, group in bins.groupby('patch_id'):
            ix = group.index.to_numpy()
            counts.append(dict(condition=condition, patch_id=patch, n_bins=len(ix),
                counts=int(x[ix].sum()), positive_bins=int(np.count_nonzero(np.asarray(x[ix].sum(1)))),
                unmodeled_counts=int(components['unmodeled_biological'][ix].sum()) if condition=='source_separated' else 0))
    pd.DataFrame(counts).to_csv(out / 'dataset_summary.csv', index=False)
    prepare_common_seeds(out)
    dump(diag / 'axis_checks.json', dict(n_bins=len(bins), n_genes=len(genes), n_patches=bins.patch_id.nunique(),
         barcode_unique=True, global_axis_exact=True, components_conserved=True, technical_counts=0))
    write_experiment_manifest(out, config=dict(conditions=CONDITIONS, regional_variant='uniform_budget',
        seed=20260909 if REGIONAL == ROOT/'runs/benchmarks/regional_depth_blocked_20260909/experiment' else None,
        component_source=str(REGIONAL), alpha=1., beta=1.5, kappa=2., profile_update_damping=0.,
        bin2cell_max_bin_distance=5, stcs_search_radius=5, stcs_lambda=.5),
        inputs={**sources, 'barcodes': AUDIT/'barcodes.parquet', 'genes': AUDIT/'genes.csv',
                'reference': REFERENCE, 'nuclei': NUCLEI, 'patches': PATCHES, 'nuclear_seeds': NUCLEAR,
                'checkpoint_protocol_only': CHECKPOINT, 'ground_truth': GT, 'fractional_gt': FRACTIONAL,
                'common_confident_nuclear_seeds':common_seed_path(out),
                'gt_profile_mapping':PROFILE_ASSIGNMENT, 'gt_profiles':PROFILE_H5},
        protocol=dict(role='four previously inspected development patches; not independent validation',
            selected_cells='same 43 core cells; context cells compete',
            paired_control='exact modeled counts retained; only unmodeled component removed',
            gene_correlation='reconstructed observed expression versus original generating source profile, not captured-count GT',
            no_training=True, no_weight_selection=True, smurf_excluded=True,
            tissue_support='all selected contexts assumed tissue; kernel crop-normalized') | (protocol_overrides or {}))
    print('PREPARED', out, flush=True)


def command(out, name, script, args, gpu='', python=None, pythonpath=None, pythonhome=None, work_root=None):
    log = out / 'subdiagnostic' / (name + '.log')
    receipt = log.with_suffix('.command.json')
    if receipt.exists():
        raise FileExistsError(receipt)
    cmd = [str(python or sys.executable), '-B', str(ROOT / script), *map(str, args)]
    work = (Path(work_root) if work_root is not None else out/'subdiagnostic') / (name + '_runtime')
    work.mkdir(parents=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               NUMBA_NUM_THREADS='4', TF_NUM_INTRAOP_THREADS='4', TF_NUM_INTEROP_THREADS='2',
               CUDA_VISIBLE_DEVICES=gpu, MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1',
               MPLCONFIGDIR=str(work/'matplotlib'), NUMBA_CACHE_DIR=str(work/'numba_cache'))
    if python is not None:
        env['PYTHONNOUSERSITE'] = '1'
    if pythonpath is not None:
        env['PYTHONPATH'] = str(pythonpath)
    if pythonhome is not None:
        env['PYTHONHOME'] = str(pythonhome)
        env['LD_LIBRARY_PATH'] = str(Path(pythonhome)/'lib') + ':' + env.get('LD_LIBRARY_PATH','')
    dump(receipt, dict(command=cmd, cwd=str(work), gpu=gpu, pythonpath=pythonpath, pythonhome=pythonhome))
    start = time.monotonic()
    print('START', name, flush=True)
    with log.open('x') as handle:
        result = subprocess.run(cmd, cwd=work, env=env, stdout=handle, stderr=subprocess.STDOUT)
    dump(log.with_suffix('.status.json'), dict(returncode=result.returncode, elapsed_seconds=time.monotonic()-start))
    if result.returncode:
        raise RuntimeError(f'{name} failed; inspect {log}')
    print('DONE', name, round(time.monotonic()-start, 1), 's', flush=True)


def evaluation_args(context, directory, name):
    return ['--ppo_eval_run_dir', context, '--eval_output_root', directory / 'evaluations',
            '--eval_run_name', name, '--overlay_max_cells', 0, '--eval_seed', 20260914,
            '--gt_cell_bins_path', GT,
            '--gt_fractional_cell_bins_path', FRACTIONAL_CELL_BINS,
            '--gt_nuclear_bins_path', NUCLEAR,
            '--gt_cell_assignments_csv', PROFILE_ASSIGNMENT,
            '--gt_sc_expression_h5', PROFILE_H5]


def build_and_em(out, condition):
    directory = out / condition
    command(out, condition+'_episodes', 'scripts/run_episode_build.py', ['--config', directory/'episode_build.yaml'])
    indices = list((directory / 'episodes').glob('*/episodes_index.csv'))
    if len(indices) != 1:
        raise ValueError('expected one episode build')
    index = indices[0]
    common = ['--checkpoint', CHECKPOINT, '--base-config', BASE,
              '--episodes-index-path-override', index, '--nuclei-path-override', NUCLEI,
              '--reference-npz-path-override', REFERENCE, '--patches-index-path', PATCHES]
    command(out, condition+'_nucleus', 'scripts/evaluate_patch_initialization.py', common + [
        '--mode', 'nucleus_only', '--device', 'cpu', '--output-dir', directory/'methods/nucleus_only',
        '--external_nuclear_bins_path', common_seed_path(out),
        *evaluation_args(BENCH/'selection/source_eval', directory, condition+'_nucleus_only')])
    contexts = list((directory/'methods/nucleus_only').glob('*/evaluation_context'))
    if len(contexts) != 1:
        raise ValueError('expected one nucleus evaluation context')
    context = contexts[0]
    command(out, condition+'_em', 'scripts/run_em_weight_sweep.py', common + [
        '--source-eval-dir', context, '--gt-cell-bins-path', GT, '--output-dir', directory/'methods/em_and_distance',
        '--spatial-weights', 1, '--expression-weights', 0, 1.5, '--non-nuclear-bin-filter', 'positive_expression',
        '--alternating-cell-profile', '--cell-profile-relative-prior-strengths', 2,
        '--profile-update-dampings', 0, '--no-background-enabled', '--seed', 20260914,
        '--max-iterations', EM_MAX_ITERATIONS])
    for method, variant in [('distance_only', 'alpha_1p0__beta_0p0__kappa_2p0__profile_damping_0p0'),
                            ('frozen_em', 'alpha_1p0__beta_1p5__kappa_2p0__profile_damping_0p0')]:
        assignment = directory/'methods/em_and_distance/variants'/variant/'assignments.csv'
        command(out, condition+'_'+method+'_eval', 'scripts/evaluate_assignment_table.py', [
            '--assignments-csv', assignment, '--method-name', 'hd_cell_rl_'+method, '--method-label', method,
            '--external-nuclear-bins-path', common_seed_path(out), *evaluation_args(context, directory, condition+'_'+method)])
    return context


def external_method(out, condition, method, context, runtime_root=None):
    directory = out / condition
    args = ['--pseudo_hd_dir', directory/'dataset', '--output_dir', directory/'methods'/method,
            '--dataset_name', condition, '--external_nuclear_bins_path', common_seed_path(out),
            *evaluation_args(context, directory, condition+'_'+method)]
    if method == 'bin2cell':
        scales = json.loads((directory/'dataset/spatial/scalefactors_json.json').read_text())
        mpp = float(scales['microns_per_pixel'])
        if abs(float(scales['spot_diameter_fullres'])*mpp-2.0) > 0.01:
            raise ValueError('Pseudo-HD spatial scale does not describe 2 um bins')
        args += ['--image_path', IMAGE, '--mpp', mpp, '--max_bin_distance', 5,
                 '--gaussian_sigma', 5, '--prob_thresh_gex', .05, '--nms_thresh_gex', .5]
    else:
        args += ['--stcs_repo_dir', ROOT/'external/STCS', '--search_radius', 5, '--lambda_spatial', .5,
                 '--restrict_to_eval_cells', '--eval_context_radius_bins', 10]
    python = ROOT / (method+'_env') / 'bin/python'
    if not python.is_file():
        raise FileNotFoundError(python)
    prefix = None
    suffix = '_native_env'
    if method == 'bin2cell':
        # Expose only the missing evaluator dependency, not a second entire
        # site-packages tree (which could shadow the tool's native packages).
        prefix = out/'subdiagnostic/bin2cell_pyarrow_bridge' if runtime_root is None else runtime_root/'pyarrow_bridge'
        prefix.mkdir(exist_ok=True)
        source = ROOT/'stcs_env/lib/python3.12/site-packages' if runtime_root is None else runtime_root/'stcs_site'
        for path in [source/'pyarrow', *source.glob('pyarrow-*.dist-info')]:
            if not path.exists():
                raise FileNotFoundError(path)
            target = prefix/path.name
            if target.is_symlink():
                if target.resolve() != path.resolve():
                    raise ValueError('unexpected existing PyArrow bridge')
            elif target.exists():
                raise FileExistsError(target)
            else:
                # Both paired method jobs can reach this tiny shared bridge.
                try:
                    target.symlink_to(path.resolve(), target_is_directory=True)
                except FileExistsError:
                    if target.resolve() != path.resolve():
                        raise
        suffix = '_compatible_env'
    home = None
    if runtime_root is not None:
        home = runtime_root/'base'
        python = home/'bin/python'
        site = runtime_root/(method+'_site')
        prefix = str(site) + (':' + str(prefix) if prefix is not None else '')
        suffix = '_node_local'
    suffix += '_confident_seeds'
    command(out, condition+'_'+method+suffix, 'preprocessing/run_'+method+'_analysis.py', args,
            gpu=str(GPU_OFFSET + (0 if condition=='modeled_only' else 1)), python=python, pythonpath=prefix, pythonhome=home,
            work_root=runtime_root/'work' if runtime_root is not None else None)


def run(out, runtime_root=None):
    run_internal(out)
    run_external(out, runtime_root=runtime_root)


def run_internal(out):
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda c: build_and_em(out, c), CONDITIONS))


def run_external(out, methods=('bin2cell', 'stcs'), runtime_root=None):
    contexts = {}
    for condition in CONDITIONS:
        candidates = list((out/condition/'methods/nucleus_only').glob('*/evaluation_context'))
        if len(candidates) != 1:
            raise ValueError('expected one completed nucleus context')
        contexts[condition] = candidates[0]
    # Two methods at a time; isolate diagnostics and GPU devices by condition.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for method in methods:
            list(pool.map(lambda c: external_method(out, c, method, contexts[c], runtime_root), CONDITIONS))


def validate_exported_ownership(assignments, seed_owners):
    """Table-only counterpart of the Torch-coupled patch export contract."""
    if assignments[['barcode', 'cell_id']].isna().any().any():
        raise ValueError('missing exported physical identity')
    if assignments.barcode.duplicated().any():
        raise ValueError('duplicate physical barcode in assignments')
    owners = assignments.barcode.map(seed_owners)
    if (owners.notna() & (owners != assignments.cell_id)).any():
        raise ValueError('nuclear ownership changed in assignments')


def summarize(out):
    import numpy as np
    import pandas as pd
    from hd_cell_rl.experiment_protocol import paired_cluster_mean_ci
    mapping = pd.read_csv(BENCH/'selection/source_eval/per_episode.csv', dtype={'cell_id':str})
    mapping = mapping[['cell_id', 'patch_id']]
    bins = pd.read_parquet(out/'barcodes.parquet')
    components, _ = assemble_components(bins, REGIONAL)
    lookup = {b:i for i,b in enumerate(bins.barcode)}
    sources = []
    seed_frame = pd.read_csv(common_seed_path(out), usecols=['barcode','cell_id','is_nuclear'], dtype={'cell_id':str})
    seed_frame = seed_frame[seed_frame.is_nuclear.astype(str).str.lower().isin(['true','1']) & seed_frame.barcode.isin(lookup)]
    if seed_frame.groupby('barcode').cell_id.nunique().max() > 1:
        raise ValueError('conflicting external nuclear owners')
    seed_owners = seed_frame.drop_duplicates('barcode').set_index('barcode').cell_id.to_dict()
    seed_rows = []
    for patch, group in bins.groupby('patch_id'):
        for scope, selected_bins in [('all_context_nuclear', group[group.barcode.isin(seed_owners)]),
                                     ('core_nuclear', group[group.barcode.isin({b for b,c in seed_owners.items() if c in set(mapping.cell_id)})])]:
            ix = selected_bins.index.to_numpy()
            n = int(components['observed'][ix].sum())
            u = int(components['unmodeled_biological'][ix].sum())
            seed_rows.append(dict(patch_id=patch,scope=scope,n_bins=len(ix),observed_counts=n,
                unmodeled_counts=u,unmodeled_fraction=u/max(n,1)))
    pd.DataFrame(seed_rows).to_csv(out/'subdiagnostic/nuclear_source_mixing.csv',index=False)
    rows, frames, checks, owner_maps = [], [], [], {}
    for condition in CONDITIONS:
        summaries = list((out/condition/'evaluations').glob('*/summary.json'))
        if len(summaries) != 5:
            raise ValueError(f'{condition}: expected five completed method evaluations, got {len(summaries)}')
        for path in summaries:
            summary = json.loads(path.read_text())
            method = summary['method'].removeprefix('hd_cell_rl_')
            cells = pd.read_csv(path.parent/'per_episode.csv', dtype={'cell_id':str})
            if set(cells.cell_id) != set(mapping.cell_id) or cells.cell_id.duplicated().any():
                raise ValueError('methods did not evaluate the same unique core cells')
            cells = cells.merge(mapping, on='cell_id', validate='one_to_one')
            cells['condition'] = condition
            cells['method'] = method
            assignment_path = next(Path(value) for key,value in summary.items() if key.endswith('_assignments_path'))
            assignments = pd.read_csv(assignment_path, dtype={'cell_id':str, 'barcode':str})
            validate_exported_ownership(assignments, seed_owners)
            owner_maps[condition, method] = assignments.set_index('barcode').cell_id
            # This diagnostic uses full exported core-cell assignments in the saved
            # crop, not the evaluator's candidate-clipped membership masks.
            selected = assignments[assignments.cell_id.isin(cells.matched_pred_cell_id.dropna().astype(int).astype(str))]
            index = np.array([lookup[b] for b in selected.barcode if b in lookup], dtype=np.int64)
            observed = components['observed'] if condition=='source_separated' else components['modeled_cell']
            assigned_counts = int(observed[index].sum())
            unknown_counts = int(components['unmodeled_biological'][index].sum()) if condition=='source_separated' else 0
            sources.append(dict(condition=condition,method=method,n_exported_core_bins=len(index),
                n_exported_core_zero_count_bins=int(np.count_nonzero(np.asarray(observed[index].sum(1)).ravel()==0)),
                assigned_counts=assigned_counts,assigned_unmodeled_counts=unknown_counts,
                unmodeled_fraction_of_assigned_umis=unknown_counts/max(assigned_counts,1)))
            frames.append(cells)
            rows.append(dict(condition=condition, method=method, n_cells=len(cells),
                n_matched_pred=int(summary['n_matched_pred']),
                **{'mean_'+m:float(cells[m].mean()) for m in METRICS},
                **{'n_finite_'+m:int(np.isfinite(cells[m]).sum()) for m in METRICS},
                evaluation_dir=str(path.parent)))
            checks.append(dict(condition=condition, method=method, core_cells_exact=True, n_cells=len(cells),
                exported_barcodes_unique=True, retained_nuclear_owners_exact=True,
                missing_core_nuclear_bins=sum(b not in set(assignments.barcode)
                    for b,c in seed_owners.items() if c in set(mapping.cell_id)),
                n_exported_bins=len(assignments)))
    combined = pd.concat(frames, ignore_index=True)
    if combined.gene_common_count.nunique(dropna=False) != 1:
        raise ValueError('expression evaluation gene axes differ across methods or conditions')
    pd.DataFrame(rows).sort_values(['condition','mean_pred_iou'], ascending=[True,False]).to_csv(out/'method_comparison.csv', index=False)
    combined.to_csv(out/'subdiagnostic/per_cell_comparison.csv', index=False)
    combined.groupby(['condition','patch_id','method'])[list(METRICS)].mean().reset_index().to_csv(out/'per_patch_comparison.csv',index=False)
    paired = []
    for method, group in combined.groupby('method'):
        a = group[group.condition=='modeled_only'].set_index('cell_id')
        b = group[group.condition=='source_separated'].set_index('cell_id').loc[a.index]
        for metric in METRICS:
            delta = b[metric].to_numpy()-a[metric].to_numpy()
            lo, hi = paired_cluster_mean_ci(delta, a.patch_id.to_numpy(), rng=np.random.default_rng(20260914))
            paired.append(dict(method=method, metric=metric, delta_added_unmodeled=float(np.nanmean(delta)),
                patch_bootstrap_low=lo, patch_bootstrap_high=hi, n_patches=int(a.patch_id.nunique())))
    pd.DataFrame(paired).to_csv(out/'paired_component_effect.csv', index=False)
    pd.DataFrame(sources).to_csv(out/'unmodeled_uptake.csv', index=False)
    dump(out/'subdiagnostic/evaluation_checks.json', checks)
    support = []
    for method in sorted({m for _, m in owner_maps}):
        a, b = owner_maps['modeled_only', method], owner_maps['source_separated', method]
        shared = a.index.intersection(b.index)
        support.append(dict(method=method, modeled_only_exported_bins=len(a),
            source_separated_exported_bins=len(b), shared_exported_bins=len(shared),
            shared_bins_owner_changed=int((a.loc[shared] != b.loc[shared]).sum()),
            newly_assigned_bins=len(b.index.difference(a.index)),
            no_longer_assigned_bins=len(a.index.difference(b.index))))
    pd.DataFrame(support).to_csv(out/'subdiagnostic/paired_assignment_support.csv',index=False)
    convergence = []
    for condition in CONDITIONS:
        for path in sorted((out/condition/'methods/em_and_distance/variants').glob('*/em_assignment/*.npz')):
            with np.load(path, allow_pickle=False) as saved:
                metadata = json.loads(str(saved['metadata_json'].item()))
            convergence.append(dict(condition=condition, variant=path.parent.parent.name,
                patch_id=metadata['patch_id'], converged=metadata['converged'],
                iterations=len(metadata['iterations']),
                n_disconnected_hard_islands=metadata['n_disconnected_hard_islands']))
    pd.DataFrame(convergence).to_csv(out/'subdiagnostic/em_convergence.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def finalize(out):
    """Archive the delivery source separately from the preparation snapshot."""
    from hd_cell_rl.experiment_protocol import write_experiment_manifest
    initial = json.loads((out/'experiment_manifest.json').read_text())
    inputs = {name: value['path'] for name, value in initial['inputs'].items()}
    episode_config = yaml.safe_load((out/'modeled_only/episode_build.yaml').read_text())
    inputs.update(fractional_gt_evaluator=INTERMEDIATE/'ground_truth/colorectal_nucleus_based_multi_owner_v2_square_002um_cell_bins.csv.gz',
                  common_confident_nuclear_seeds=common_seed_path(out),
                  seed_and_candidate_metadata=episode_config['inputs']['bins_path'],
                  modeled_episode_config=out/'modeled_only/episode_build.yaml',
                  source_separated_episode_config=out/'source_separated/episode_build.yaml',
                  stcs_implementation=ROOT/'external/STCS/STCS/STCS_main.py',
                  gt_profile_mapping=PROFILE_ASSIGNMENT,
                  gt_profiles=PROFILE_H5,
                  image=IMAGE, comparison=out/'method_comparison.csv')
    write_experiment_manifest(out/'subdiagnostic/delivery_provenance',
        config=initial['resolved_config'], inputs=inputs,
        protocol={**initial['evaluation_protocol'],
                  'runtime_receipts':str(out/'subdiagnostic'),
                  'delivery_snapshot':'includes launcher compatibility fixes; algorithms unchanged'})


def main():
    global REGIONAL, PROFILE_ASSIGNMENT, PROFILE_H5, GPU_OFFSET
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=('prepare', 'seeds', 'run', 'internal', 'external', 'summarize', 'finalize', 'all'), default='all')
    parser.add_argument('--external-methods', nargs='+', choices=('bin2cell','stcs'), default=['bin2cell','stcs'])
    parser.add_argument('--runtime-root', type=Path, help='Optional node-local copy of the unchanged native environments.')
    parser.add_argument('--regional-dir', type=Path, help='Alternative saved component realizations with the same physical axes.')
    parser.add_argument('--profile-assignment-csv', type=Path, help='Generating cell-to-profile mapping; evaluation only.')
    parser.add_argument('--profile-h5', type=Path, help='Matching generating expression profiles; evaluation only.')
    parser.add_argument('--gpu-offset', type=int, default=0, help='External paired jobs use this GPU and the next GPU.')
    args = parser.parse_args()
    if args.gpu_offset < 0:
        parser.error('gpu-offset must be nonnegative')
    GPU_OFFSET = args.gpu_offset
    if (args.profile_assignment_csv is None) != (args.profile_h5 is None):
        parser.error('profile mapping and H5 must be supplied together')
    if args.regional_dir is not None:
        REGIONAL = args.regional_dir.resolve(strict=True)
    if args.profile_h5 is not None:
        PROFILE_ASSIGNMENT = args.profile_assignment_csv.resolve(strict=True)
        PROFILE_H5 = args.profile_h5.resolve(strict=True)
    out = args.output_dir.resolve()
    if args.stage in ('prepare','all'): prepare(out)
    if args.stage == 'seeds': prepare_common_seeds(out)
    if args.stage in ('run','all'): run(out, args.runtime_root)
    if args.stage == 'internal': run_internal(out)
    if args.stage == 'external': run_external(out, args.external_methods, args.runtime_root)
    if args.stage in ('summarize','all'): summarize(out)
    if args.stage in ('finalize','all'): finalize(out)


if __name__ == '__main__':
    main()
