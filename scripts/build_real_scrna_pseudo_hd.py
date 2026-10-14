#!/usr/bin/env python3
"""Fill existing pseudo-HD masks with locally non-repeated real scRNA cells.

Reuse Bin2Cell_Validation's localization kernels and 10x writers. No scDesign3,
EM, PPO, shape changes, or new background model are involved.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hd_cell_rl.spatial_profile_assignment import assign_spatial_profiles, audit_spatial_assignment, plan_donor_split
from hd_cell_rl.experiment_protocol import write_experiment_manifest


def load_localization_module(validation_repo):
    directory = Path(validation_repo) / 'scripts/simulation'
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location('real_scrna_localization', directory/'create_colorectal_pseudo_hd.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eligible_sources(metadata, *, generator_donors, reference_donors):
    """Use an explicit disjoint donor split; never borrow from reference donors."""
    generator, reference = set(generator_donors), set(reference_donors)
    if not generator or not reference or generator & reference:
        raise ValueError('generator/reference donor sets must be nonempty and disjoint')
    if metadata[['cell', 'sct_cell_type', 'case_id']].isna().any().any():
        raise ValueError('source metadata has missing identity/type/donor')
    if metadata.cell.duplicated().any():
        raise ValueError('duplicate physical scRNA identity in source metadata')
    missing = (generator | reference) - set(metadata.case_id)
    if missing:
        raise ValueError(f'donors absent from metadata: {sorted(missing)}')
    return metadata.loc[metadata.case_id.isin(generator)].rename(
        columns={'cell': 'sc_cell_barcode', 'sct_cell_type': 'cell_type'}).copy()


def target_coordinates(metadata):
    """Use existing physical coordinates; infer image scale from recorded areas."""
    if {'x_um', 'y_um'}.issubset(metadata):
        return metadata[['cell_id', 'cell_type', 'x_um', 'y_um']].copy()
    ratio = metadata.nucleus_area_um2.to_numpy(float) / metadata.nucleus_area_px.to_numpy(float)
    if not np.isfinite(ratio).all() or np.any(ratio <= 0) or not np.allclose(ratio, ratio[0], rtol=1e-6):
        raise ValueError('inconsistent recorded pixel-to-micrometre area conversion')
    result = metadata[['cell_id', 'cell_type']].copy()
    result['x_um'] = metadata.nucleus_centroid_col_fullres_px.to_numpy(float)*np.sqrt(ratio[0])
    result['y_um'] = metadata.nucleus_centroid_row_fullres_px.to_numpy(float)*np.sqrt(ratio[0])
    return result


def selected_raw_counts(path, selected_barcodes):
    """Read only assigned raw H5 columns, in physical-cell order, allowing reuse."""
    with h5py.File(path, 'r') as handle:
        m = handle['matrix']
        decode = lambda values: np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in values])
        bars = decode(m['barcodes'][:]); genes = decode(m['features/name'][:])
        if len(set(bars)) != len(bars) or len(set(genes)) != len(genes):
            raise ValueError('raw H5 barcodes/genes must be unique')
        lookup = {barcode: i for i, barcode in enumerate(bars)}
        missing = set(selected_barcodes) - set(lookup)
        if missing:
            raise ValueError(f'assigned raw cells absent from H5: {sorted(missing)[:5]}')
        shape = tuple(m['shape'][:]); pointers = m['indptr'][:]
        if shape != (len(genes), len(bars)) or len(pointers) != len(bars)+1:
            raise ValueError('raw H5 is not gene-by-cell CSC')
        # Load once; avoid thousands of random shared-filesystem HDF5 reads.
        raw = sparse.csr_matrix((m['data'][:], m['indices'][:], pointers), shape=(len(bars), len(genes)))
    if not np.isfinite(raw.data).all() or np.any(raw.data < 0) or np.any(raw.data != np.floor(raw.data)):
        raise ValueError('source must contain nonnegative integer raw counts, not normalized expression')
    raw.sum_duplicates(); raw.eliminate_zeros()
    return raw[[lookup[v] for v in selected_barcodes]].astype(np.int64), genes


def place_real_profiles(profiles, assignment, bins, *, source_module, p0, membrane_flags,
                        n_bins, seed, capture_probability, block_cells=1000, report_progress=False):
    """Existing compartment allocation, then independent binomial capture.

    No per-cell library-size normalization. Global nuclear calibration includes
    each assigned physical cell once, including deliberate distant source reuse.
    Capture changes observed depth, not the underlying raw expression profile.
    """
    if block_cells <= 0:
        raise ValueError('block_cells must be positive')
    probability = np.asarray(capture_probability, dtype=float)
    if probability.ndim not in (0, 1) or (probability.ndim == 1 and probability.shape != (n_bins,)):
        raise ValueError('capture must be a scalar or one probability per original barcode')
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('capture probabilities must be finite and in [0,1]')
    bins = bins.sort_values(['cell_id', 'barcode_index']).reset_index(drop=True)
    ids, starts = np.unique(bins.cell_id, return_index=True)
    if not np.array_equal(ids, assignment.cell_id) or profiles.shape[0] != len(ids):
        raise ValueError('profiles, physical cells, and complete mask geometry must align')
    contacts = bins[['barcode_index', 'cell_id']].copy()
    contacts['source'] = contacts.cell_id.map(assignment.set_index('cell_id').sc_cell_barcode)
    if contacts.duplicated(['barcode_index', 'source']).any():
        raise ValueError('cells sharing a physical bin reuse a source; increase spatial exclusion or source pool')
    ends = np.r_[starts[1:], len(bins)]
    mod = source_module
    rng = np.random.default_rng(seed)
    capture_rng = np.random.default_rng(np.random.SeedSequence([seed, 1729]))
    factors = rng.lognormal(0, mod.SIGMA_CELL_FACTOR, len(ids))
    alpha, expected_fraction, panel_total = mod.calibrate_global_alpha(
        profiles.data, profiles.indices, profiles.indptr, p0, factors,
        np.ones(profiles.shape[1], bool), 1.0)
    destinations, genes = [], []
    rows, cols, values = [], [], []
    trace_rows, trace_cols, trace_values, audits = [], [], [], []
    for i, cell_id in enumerate(ids):
        begin, end = profiles.indptr[i:i+2]
        gene_index, counts = profiles.indices[begin:end], profiles.data[begin:end]
        nuclear = rng.binomial(counts, np.clip(alpha*p0[gene_index]*factors[i]*mod.C_CAPTURE_EFFICIENCY,
                                             mod.CLIP_EPS, 1-mod.CLIP_EPS))
        membrane = np.where(membrane_flags[gene_index], counts-nuclear, 0)
        cytoplasm = counts-nuclear-membrane
        group = bins.iloc[starts[i]:ends[i]]
        coverage = group.cell_coverage_fraction.to_numpy()
        weights = [mod.nuclear_kernel(group)[0],
                   mod.normalized_weights(group.cytoplasm_fraction.to_numpy(), coverage)[0],
                   mod.normalized_weights(group.membrane_fraction.to_numpy(), coverage)[0]]
        local_dest, local_genes = [], []
        placed, retained = [], []
        for amounts, weight in zip((nuclear, cytoplasm, membrane), weights):
            d, g = [], []
            n = mod.add_molecules(d, g, rng, group.barcode_index.to_numpy(), weight, gene_index, amounts)
            placed.append(n)
            if n:
                p = probability if probability.ndim == 0 else probability[d[0]]
                keep = capture_rng.random(n) < p
                local_dest.append(d[0][keep]); local_genes.append(g[0][keep])
                retained.append(int(keep.sum()))
            else:
                retained.append(0)
        if local_dest:
            d = np.concatenate(local_dest); g = np.concatenate(local_genes)
            destinations.append(d); genes.append(g)
            unique, totals = np.unique(d, return_counts=True)
            trace_rows.append(np.full(len(unique), i)); trace_cols.append(unique); trace_values.append(totals)
        audits.append(dict(cell_id=int(cell_id), source_on_panel_counts=int(counts.sum()),
                           nuclear_before_capture=placed[0], cytoplasm_before_capture=placed[1],
                           membrane_before_capture=placed[2], nuclear_counts=retained[0],
                           cytoplasm_counts=retained[1], membrane_counts=retained[2],
                           captured_counts=sum(retained), capture_rejected_counts=int(counts.sum())-sum(retained)))
        if (i+1) % block_cells == 0 or i+1 == len(ids):
            mod.flush_entry_block(destinations, genes, profiles.shape[1], rows, cols, values)
            if report_progress:
                print('PLACED',i+1,'/',len(ids),'physical cells',flush=True)
    def matrix(rs, cs, vs, shape):
        if not vs:
            return sparse.csr_matrix(shape, dtype=np.int64)
        return sparse.coo_matrix((np.concatenate(vs), (np.concatenate(rs), np.concatenate(cs))), shape=shape).tocsr()
    observed = matrix(rows, cols, values, (n_bins, profiles.shape[1]))
    trace = matrix(trace_rows, trace_cols, trace_values, (len(ids), n_bins))
    audit = pd.DataFrame(audits)
    if int(observed.sum()) != int(trace.sum()) or int(observed.sum()+audit.capture_rejected_counts.sum()) != panel_total:
        raise AssertionError('source/capture/placement count conservation failed')
    return observed, trace, audit, dict(global_alpha=alpha, expected_nuclear_fraction=expected_fraction,
                                       source_on_panel_counts=int(panel_total), captured_counts=int(observed.sum()),
                                       count_conservation_passed=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation-repo', type=Path, default=ROOT.parent/'Bin2Cell_Validation')
    p.add_argument('--ground-truth-cells', type=Path, required=True)
    p.add_argument('--source-h5', type=Path, required=True)
    p.add_argument('--source-metadata', type=Path, required=True)
    p.add_argument('--generator-donors', nargs='+')
    p.add_argument('--reference-donors', nargs='+')
    p.add_argument('--plan-donor-split', action='store_true', help='Choose a new disjoint split from type coverage and patch capacity only.')
    p.add_argument('--patch-size-um', type=float, default=128)
    p.add_argument('--exclusion-distance-um', type=float, default=64)
    p.add_argument('--seed', type=int, default=20260922)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--assignment-only', action='store_true')
    p.add_argument('--bin-assignments', type=Path)
    p.add_argument('--original-hd-dir', type=Path)
    p.add_argument('--gtf', type=Path)
    p.add_argument('--membrane-genes', type=Path)
    capture = p.add_mutually_exclusive_group()
    capture.add_argument('--capture-probability', type=float)
    capture.add_argument('--capture-probabilities-npy', type=Path,
                         help='Existing resolved capture field, aligned to original H5 barcode order.')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.assignment_only:
        if not args.bin_assignments or not args.original_hd_dir or (
            args.capture_probability is None and args.capture_probabilities_npy is None
        ):
            raise ValueError('generation requires bin assignments, original HD directory and explicit capture settings')
    if args.output_dir.exists():
        raise FileExistsError(f'refusing to overwrite output: {args.output_dir}')
    metadata = pd.read_csv(args.source_metadata, sep='\t', usecols=['cell', 'sct_cell_type', 'case_id'], dtype=str)
    targets = target_coordinates(pd.read_csv(args.ground_truth_cells))
    if args.plan_donor_split:
        if args.generator_donors or args.reference_donors:
            raise ValueError('automatic split cannot be combined with explicit donor lists')
        split, split_audit = plan_donor_split(targets, metadata, patch_size_um=args.patch_size_um)
        args.generator_donors, args.reference_donors = split['generator_donors'], split['reference_donors']
        print('DONOR SPLIT', json.dumps(split), flush=True)
    else:
        if not args.generator_donors or not args.reference_donors:
            raise ValueError('provide both donor lists or --plan-donor-split')
        split, split_audit = None, None
    source = eligible_sources(metadata, generator_donors=args.generator_donors, reference_donors=args.reference_donors)
    # Validate every eligible source against real H5 identities before sampling.
    with h5py.File(args.source_h5, 'r') as f:
        bars = {b.decode() if isinstance(b, bytes) else str(b) for b in f['matrix/barcodes'][:]}
    if set(source.sc_cell_barcode)-bars:
        raise ValueError('eligible source metadata contains cells absent from raw H5')
    assignment = assign_spatial_profiles(targets, source, patch_size_um=args.patch_size_um,
        exclusion_distance_um=args.exclusion_distance_um, seed=args.seed)
    summary = audit_spatial_assignment(assignment, source, exclusion_distance_um=args.exclusion_distance_um)
    assignment['source_donor'] = assignment.sc_cell_barcode.map(source.set_index('sc_cell_barcode').case_id)
    args.output_dir.mkdir(parents=True)
    debug = args.output_dir/'subdiagnostic'; debug.mkdir()
    if split is not None:
        (args.output_dir/'donor_split.json').write_text(json.dumps(split,indent=2)+'\n')
        split_audit.to_csv(debug/'donor_split_type_coverage.csv',index=False)
    reference_meta = metadata.loc[metadata.case_id.isin(args.reference_donors)]
    reference_meta.to_csv(args.output_dir/'reference_metadata.tsv',sep='\t',index=False)
    assignment.to_csv(args.output_dir/'ground_truth_cell_assignments.csv', index=False)
    source.to_csv(debug/'eligible_sources.tsv', sep='\t', index=False)
    assignment.groupby(['patch_id','cell_type']).size().rename('n_cells').to_csv(debug/'patch_inventory.csv')
    inputs = dict(ground_truth_cells=args.ground_truth_cells, source_h5=args.source_h5, source_metadata=args.source_metadata)
    summary.update(generator_donors=args.generator_donors, reference_donors=args.reference_donors,
                   patch_size_um=args.patch_size_um, exclusion_distance_um=args.exclusion_distance_um,
                   generator_reference_donor_overlap=[], expression_origin='real_scRNA', seed=args.seed)
    summary['reference_missing_types'] = sorted(set(targets.cell_type)-set(reference_meta.sct_cell_type))
    summary['benchmark_ready_for_all_types'] = not summary['reference_missing_types']
    (args.output_dir/'assignment_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    if not args.assignment_only:
        mod = load_localization_module(args.validation_repo)
        profiles, source_genes = selected_raw_counts(args.source_h5, assignment.sc_cell_barcode)
        features, barcodes = mod.load_original_features_and_barcodes(args.original_hd_dir/'filtered_feature_bc_matrix.h5')
        lookup = {g:i for i,g in enumerate(features['name'])}
        mapped = np.array([lookup.get(g,-1) for g in source_genes])
        keep = mapped >= 0
        mapping = sparse.csr_matrix((np.ones(keep.sum(), dtype=np.int64), (np.flatnonzero(keep), mapped[keep])),
                                    shape=(len(source_genes),len(features['name'])))
        total_raw = int(profiles.sum()); profiles = (profiles @ mapping).tocsr()
        profiles.sort_indices()
        gtf = args.gtf or args.validation_repo/'references/gene_annotation/gencode_v38.gtf'
        membrane_path = args.membrane_genes or args.validation_repo/'references/gene_localization/human_membrane_secretory_genes_uniprot.txt'
        p0, _, _ = mod.localization_probabilities(features['name'], profiles.data, profiles.indices, gtf)
        flags = np.isin(features['name'], list(mod.load_membrane_genes(membrane_path)))
        bins = mod.validate_fractional_bin_assignments(pd.read_parquet(args.bin_assignments), n_barcodes=len(barcodes))
        probability = args.capture_probability if args.capture_probabilities_npy is None else np.load(args.capture_probabilities_npy, allow_pickle=False)
        observed, trace, audit, generation = place_real_profiles(profiles, assignment, bins, source_module=mod,
            p0=p0, membrane_flags=flags, n_bins=len(barcodes), seed=args.seed, capture_probability=probability)
        matrix_dir = args.output_dir/'filtered_feature_bc_matrix'; matrix_dir.mkdir()
        mod.write_matrix_market_gzip(observed.T.tocsc(), matrix_dir/'matrix.mtx.gz')
        mod.write_10x_tsvs(matrix_dir, features, barcodes)
        shutil.copytree(args.original_hd_dir/'spatial', args.output_dir/'spatial')
        sparse.save_npz(args.output_dir/'modeled_source_bin_counts.npz', trace)
        audit.to_csv(debug/'per_cell_capture.csv',index=False)
        counts_per_bin = bins.groupby('barcode_index').cell_id.nunique()
        generation.update(total_raw_counts=total_raw, counts_outside_hd_gene_panel=total_raw-int(profiles.sum()),
            n_multi_cell_bins=int((counts_per_bin>1).sum()), n_physical_cells=len(assignment),
            source_profile_h5=str(args.source_h5.resolve()), assignment_file='ground_truth_cell_assignments.csv',
            technical_background=False, unmodeled_biological=False, capture_explicit=True)
        (args.output_dir/'pseudo_hd_summary.json').write_text(json.dumps(generation,indent=2)+'\n')
        inputs.update(bin_assignments=args.bin_assignments, gtf=gtf, membrane_genes=membrane_path,
                      original_hd=args.original_hd_dir/'filtered_feature_bc_matrix.h5')
        if args.capture_probabilities_npy:
            inputs['capture_probabilities'] = args.capture_probabilities_npy
        for name in ('create_colorectal_pseudo_hd.py', 'gene_localization.py'):
            path = args.validation_repo/'scripts/simulation'/name
            shutil.copy2(path, debug/name); inputs[name] = path
    write_experiment_manifest(args.output_dir, config=vars(args), inputs=inputs, protocol=dict(
        mode='real scRNA patch-exclusive source assignment', assignment_only=args.assignment_only,
        no_EM_RL_changes=True, geometry_unchanged=True, no_automatic_depth_fit=True,
        boundary_exclusion='nucleus-centre Euclidean distance, inclusive; not a morphology-contact guarantee',
        caution='Source reuse across distant patches is allowed; patch replicates are not independent donors.'))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
