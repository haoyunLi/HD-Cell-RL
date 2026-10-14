#!/usr/bin/env python3
"""Whole-section paired real-scRNA pseudo HD; data live in Bin2Cell_Validation.

Noise here means structured unmodeled RNA in the geometric mask complement,
not measurement noise or a claim of an identified ambient mechanism.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_real_scrna_pseudo_hd import selected_raw_counts, load_localization_module, place_real_profiles
from scripts.build_source_separated_pseudo import align_genes, dump
from scripts.run_regional_depth_calibration import prepare_inputs
from scripts import run_source_separated_method_comparison as protocol
from hd_cell_rl.experiment_protocol import write_experiment_manifest
from hd_cell_rl.spatial_profile_assignment import assign_background_profiles
from preprocessing.build_reference_counts_from_scrna import build_reference_counts, write_reference_counts_npz, write_summary_json


def calibration_probability(real_counts, proposed_counts, calibration, covered):
    """One capture parameter from declared calibration bins, never audit counts."""
    keep = np.asarray(calibration,bool) & np.asarray(covered,bool)
    real, proposed = np.asarray(real_counts,float), np.asarray(proposed_counts,float)
    if real.shape != keep.shape or proposed.shape != keep.shape or not keep.any():
        raise ValueError('calibration count axes are invalid')
    p = float(real[keep].sum()/proposed[keep].sum())
    if not np.isfinite(p) or not 0 < p <= 1:
        raise ValueError(f'raw-cell reservoir cannot support thinning at required capture: {p}')
    return p


def check_pair(modeled, background, covered):
    if modeled.shape != background.shape or len(covered) != modeled.shape[0]:
        raise ValueError('component axes differ')
    for x in (modeled,background):
        if not np.isfinite(x.data).all() or np.any(x.data<0) or np.any(x.data!=np.floor(x.data)):
            raise ValueError('components must contain nonnegative finite integer counts')
    if background[np.asarray(covered,bool)].nnz:
        raise ValueError('background entered a modeled mask')
    if modeled[~np.asarray(covered,bool)].nnz:
        raise ValueError('modeled counts escaped their geometry')
    return (modeled+background).tocsr()


def expected_counts_at_bins(profiles, assignment, gt, selected_ids, mod, p0, flags, seed):
    """Full-section localization expectation on selected calibration/audit bins."""
    rng=np.random.default_rng(seed)
    factors=rng.lognormal(0,mod.SIGMA_CELL_FACTOR,len(assignment))
    alpha,_,_=mod.calibrate_global_alpha(profiles.data,profiles.indices,profiles.indptr,p0,factors,
                                        np.ones(profiles.shape[1],bool),1.)
    lookup={int(b):i for i,b in enumerate(selected_ids)}
    local_ids=set(gt.loc[gt.barcode_index.isin(lookup),'cell_id'])
    row_of={int(c):i for i,c in enumerate(assignment.cell_id)}
    expected=np.zeros(len(selected_ids))
    for c,g in gt[gt.cell_id.isin(local_ids)].groupby('cell_id',sort=True):
        i=row_of[int(c)];s,e=profiles.indptr[i:i+2]
        genes,counts=profiles.indices[s:e],profiles.data[s:e]
        nuclear=counts*np.clip(alpha*p0[genes]*factors[i]*mod.C_CAPTURE_EFFICIENCY,mod.CLIP_EPS,1-mod.CLIP_EPS)
        membrane=np.where(flags[genes],counts-nuclear,0);cytoplasm=counts-nuclear-membrane
        coverage=g.cell_coverage_fraction.to_numpy()
        weights=[mod.nuclear_kernel(g)[0],
                 mod.normalized_weights(g.cytoplasm_fraction.to_numpy(),coverage)[0],
                 mod.normalized_weights(g.membrane_fraction.to_numpy(),coverage)[0]]
        indices=np.array([lookup.get(int(b),-1) for b in g.barcode_index]);valid=indices>=0
        for amount,w in zip((nuclear.sum(),cytoplasm.sum(),membrane.sum()),weights):
            np.add.at(expected,indices[valid],amount*w[valid])
    return expected


def generate_complement_background(xy, allowed, cell_xy, cell_types, raw_profiles, source_types,
                                   *, rate, seed, sigma_um=8., source_area_um2=64., tile_um=128.):
    """Sample generator-donor profiles into sparse Gaussian complement fields.

    Source budgets are Poisson, calibrated by mean real complement depth only.
    Gaussian fields are truncated at 3 sigma and tile boundaries. Virtual RNA
    sources are not additional ground-truth cells; keep their audit separate.
    """
    if not np.isfinite(rate) or rate<0 or sigma_um<=0 or source_area_um2<=0:
        raise ValueError('invalid complement model parameters')
    rng=np.random.default_rng(seed);all_indices=np.flatnonzero(allowed)
    grid=np.floor(xy[all_indices]/tile_um).astype(np.int64)
    _,tile=np.unique(grid,axis=0,return_inverse=True)
    order=np.argsort(tile,kind='stable');bounds=np.r_[0,np.flatnonzero(np.diff(tile[order]))+1,len(order)]
    tree=cKDTree(cell_xy);names=np.unique(cell_types)
    pools={label:np.flatnonzero(source_types==label) for label in names}
    if any(not len(pool) for pool in pools.values()):raise ValueError('background source type missing from generator donors')
    rr,cc,vv,records=[],[],[],[]
    for t,(start,end) in enumerate(zip(bounds[:-1],bounds[1:])):
        indices=all_indices[order[start:end]];coords=xy[indices];local_tree=cKDTree(coords)
        n=min(len(indices),max(1,int(np.ceil(len(indices)*4/source_area_um2))))
        centers=rng.choice(len(indices),n,replace=False);destinations=[];genes=[]
        for center in centers:
            point=coords[center];near=tree.query_ball_point(point,72.)
            if not near:near=[int(tree.query(point)[1])]
            distance2=np.sum((cell_xy[near]-point)**2,axis=1)
            # Relative type weights, including a finite nearest-cell fallback
            # when all modeled nuclei are far from an allowed tissue bin.
            weights=np.exp(-(distance2-distance2.min())/(2*24.**2))
            q=np.array([weights[cell_types[near]==label].sum() for label in names]);q/=q.sum()
            label=str(rng.choice(names,p=q));source=int(rng.choice(pools[label]))
            count=int(rng.poisson(rate*len(indices)/n))
            s,e=raw_profiles.indptr[source:source+2];g=raw_profiles.indices[s:e];c=raw_profiles.data[s:e]
            if not len(c) or c.sum()<=0:raise ValueError('empty generator profile for background')
            local=np.asarray(sorted(local_tree.query_ball_point(point,3*sigma_um)),dtype=np.int64)
            w=np.exp(-np.sum((coords[local]-point)**2,axis=1)/(2*sigma_um**2));w/=w.sum()
            destinations.append(indices[rng.choice(local,count,p=w)])
            genes.append(rng.choice(g,count,p=c/c.sum()))
            records.append(dict(source_index=source,cell_type=label,x_um=point[0],y_um=point[1],
                                source_center_barcode_index=int(indices[center]),counts=count,tile=t,
                                nearest_modeled_cell_um=float(np.sqrt(distance2.min())),
                                local_context_supported=bool(distance2.min()<=72.**2)))
        if destinations:
            key=np.concatenate(destinations)*raw_profiles.shape[1]+np.concatenate(genes)
            keys,counts=np.unique(key,return_counts=True)
            rr.append(keys//raw_profiles.shape[1]);cc.append(keys%raw_profiles.shape[1]);vv.append(counts)
        if t%200==0:print('BACKGROUND tiles',t,'/',len(bounds)-1,flush=True)
    if not vv:return sparse.csr_matrix((len(xy),raw_profiles.shape[1]),dtype=np.int64),pd.DataFrame(records)
    result=sparse.coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),
                             shape=(len(xy),raw_profiles.shape[1])).tocsr()
    return result,pd.DataFrame(records)


def finish_pair(*,out,seed,master,gen,genes,bins,bin_meta,covered,tissue,modeled,mod,features,
                barcodes,original,raw_h5,rate,split,background_assignment_seed=None,background_render_seed=None):
    diag=out/'subdiagnostic';components=out/'components';reference_dir=out/'reference'
    selected=bins.matrix_col_index.to_numpy(int);names=sorted(bins.patch_id.unique())
    if (components/'unmodeled_biological.npz').exists() or (out/'no_background').exists():
        raise FileExistsError('refusing to overwrite existing background/export artifacts')
    raw,rg=selected_raw_counts(raw_h5,gen.cell);raw=align_genes(raw,rg,genes)
    background,records=generate_complement_background(bin_meta[['x_um','y_um']].to_numpy(),tissue & ~covered,
        master[['x_um','y_um']].to_numpy(),master.cell_type.to_numpy(),raw,gen.sct_cell_type.to_numpy(),rate=rate,seed=seed+1)
    sources=gen.rename(columns={'cell':'sc_cell_barcode','sct_cell_type':'cell_type'})
    records,proposals,budgets,audit=assign_background_profiles(records,master,sources,
        seed=seed+2 if background_assignment_seed is None else background_assignment_seed)
    background=render_background_sources(bin_meta[['x_um','y_um']].to_numpy(),tissue & ~covered,
                                         raw,records,seed=seed+3 if background_render_seed is None else background_render_seed)
    proposals.to_csv(diag/'background_source_proposals.csv.gz',index=False)
    budgets.to_csv(diag/'background_source_budgets.csv.gz',index=False)
    dump(diag/'source_exclusivity.json',audit)
    records.to_csv(diag/'unmodeled_sources.csv.gz',index=False)
    sparse.save_npz(components/'unmodeled_biological.npz',background)
    observed=check_pair(modeled,background,covered)
    gene_indices=pd.Index(features['name']).get_indexer(genes)
    if np.any(gene_indices<0):raise ValueError('missing original HD genes')
    for name,matrix in [('no_background',modeled),('with_background',observed)]:
        directory=out/name;directory.mkdir();md=directory/'filtered_feature_bc_matrix';md.mkdir()
        coo=matrix.tocoo();full=sparse.coo_matrix((coo.data,(gene_indices[coo.col],coo.row)),shape=(len(features['name']),len(barcodes)))
        mod.write_matrix_market_gzip(full,md/'matrix.mtx.gz');mod.write_10x_tsvs(md,features,barcodes)
        (directory/'spatial').symlink_to((original/'spatial').resolve(),target_is_directory=True)
        (directory/'reference').symlink_to(reference_dir.resolve(),target_is_directory=True)
        (directory/'ground_truth_cell_assignments.csv').symlink_to((out/'ground_truth_cell_assignments.csv').resolve())
        dump(directory/'pseudo_hd_summary.json',dict(expression_origin='real_scRNA',condition=name,n_cells=len(master),
            dataset_role='experimental_control',biological_realism_validated=False,
            counts=int(matrix.sum()),n_barcodes=len(barcodes),reference=str(reference_dir/'reference_counts.npz'),
            source_h5=str(raw_h5.resolve()),paired_dataset=str(out),background_counts=int(background.sum()) if name=='with_background' else 0))
        print('DATASET WRITTEN',directory,flush=True)
    regional=out/'patch_components';shared=regional/'shared';shared.mkdir(parents=True)
    compact={'modeled_cell':modeled[selected],'unmodeled_biological':background[selected],
             'technical_background':sparse.csr_matrix((len(selected),len(genes)),dtype=np.int64),'observed':observed[selected]}
    for name,matrix in compact.items():sparse.save_npz(shared/(name+'.npz'),matrix)
    for patch in names:
        directory=regional/'realizations'/('leave_out_'+patch)/'uniform_budget';directory.mkdir(parents=True)
        for name in compact:(directory/(name+'.npz')).symlink_to((shared/(name+'.npz')).resolve())
    bins.to_parquet(out/'barcodes.parquet',index=False);pd.Series(genes,name='gene').to_csv(out/'genes.csv',index=False)
    dump(out/'completion.json',dict(complete=True,whole_section=True,n_cells=len(master),n_barcodes=len(barcodes),
        modeled_counts=int(modeled.sum()),background_counts=int(background.sum()),covered_bins_unchanged=True,
        no_background_counts_outside_masks=int(modeled[~covered].sum()),background_counts_inside_masks=int(background[covered].sum()),
        n_reference_types=28,missing_types=split['reference_missing_types']))
    print('COMPLETE',out,flush=True)


def render_background_sources(xy, allowed, raw_profiles, records, *, seed,
                              sigma_um=8., tile_um=128.):
    """Place real-source expression with the unchanged Gaussian spatial kernel."""
    xy=np.asarray(xy,float);allowed=np.asarray(allowed,bool)
    if allowed.shape!=(len(xy),) or sigma_um<=0 or tile_um<=0:
        raise ValueError('invalid background support')
    indices=np.flatnonzero(allowed)
    grid=np.floor(xy[indices]/tile_um).astype(np.int64)
    keys=np.array([f'x{x}_y{y}' for x,y in grid],dtype=object)
    groups=pd.Series(indices).groupby(keys,sort=True)
    by_patch={patch:group.to_numpy() for patch,group in groups}
    rr,cc,vv=[],[],[];rng=np.random.default_rng(seed)
    for number,(patch,group) in enumerate(records.groupby('patch_id',sort=True)):
        if patch not in by_patch:raise ValueError('source outside allowed tile')
        target=by_patch[patch];coords=xy[target];tree=cKDTree(coords)
        destinations=[];genes=[]
        for row in group.itertuples():
            point=np.array([row.x_um,row.y_um]);count=int(row.counts)
            start,end=raw_profiles.indptr[int(row.source_index):int(row.source_index)+2]
            g=raw_profiles.indices[start:end];c=raw_profiles.data[start:end]
            if not len(c) or c.sum()<=0:raise ValueError('empty real background profile')
            local=np.asarray(sorted(tree.query_ball_point(point,3*sigma_um)),dtype=np.int64)
            if not len(local):raise ValueError('source has no local allowed support')
            w=np.exp(-np.sum((coords[local]-point)**2,axis=1)/(2*sigma_um**2));w/=w.sum()
            destinations.append(target[rng.choice(local,count,p=w)])
            genes.append(rng.choice(g,count,p=c/c.sum()))
        if destinations:
            key=np.concatenate(destinations)*raw_profiles.shape[1]+np.concatenate(genes)
            unique,counts=np.unique(key,return_counts=True)
            rr.append(unique//raw_profiles.shape[1]);cc.append(unique%raw_profiles.shape[1]);vv.append(counts)
        if number%200==0:print('RENDER BACKGROUND',number,'/',records.patch_id.nunique(),flush=True)
    result=sparse.coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),
                             shape=(len(xy),raw_profiles.shape[1])).tocsr() if vv else sparse.csr_matrix((len(xy),raw_profiles.shape[1]),dtype=np.int64)
    if int(result.sum())!=int(records.counts.sum()) or result[~allowed].nnz:
        raise AssertionError('background count/support conservation failed')
    return result


def resume_background(args):
    """Resume only the failed background stage; existing modeled counts are read-only."""
    out=args.output_dir.resolve();manifest=json.loads((out/'experiment_manifest.json').read_text())
    if manifest['resolved_config']['seed']!=args.seed:raise ValueError('resume seed differs')
    if (out/'completion.json').exists():raise FileExistsError('dataset already completed')
    from hd_cell_rl.provenance import file_fingerprint
    for key in ('raw_h5','bin_axes','geometry'):
        old=manifest['inputs'][key];current=file_fingerprint(old['path'])
        if current['sha256']!=old['sha256']:raise ValueError(f'resume input changed: {key}')
    master=pd.read_csv(out/'ground_truth_cell_assignments.csv')
    old_assignment=pd.read_csv(args.assignment_dir/'ground_truth_cell_assignments.csv').sort_values('cell_id').reset_index(drop=True)
    pd.testing.assert_frame_equal(master,old_assignment)
    gen=pd.read_csv(out/'generator_metadata.tsv',sep='\t');genes=(out/'genes.txt').read_text().splitlines()
    bins=pd.read_parquet(protocol.AUDIT/'barcodes.parquet')
    bin_meta=pd.read_parquet(manifest['inputs']['bin_axes']['path'],columns=['matrix_col_index','barcode','x_um','y_um']).sort_values('matrix_col_index')
    v=ROOT.parent/'Bin2Cell_Validation';original=v/'data/visium_hd/colorectal/output/binned_outputs/square_002um'
    mod=load_localization_module(v);features,barcodes=mod.load_original_features_and_barcodes(original/'filtered_feature_bc_matrix.h5')
    if not np.array_equal(bin_meta.barcode,barcodes):raise ValueError('resume barcode axes changed')
    gt=pd.read_parquet(manifest['inputs']['geometry']['path'],columns=['barcode_index','cell_id','cell_coverage_fraction'])
    covered=np.zeros(len(barcodes),bool);covered[gt.loc[(gt.cell_id>0)&(gt.cell_coverage_fraction>0),'barcode_index'].to_numpy(int)]=True
    positions=pd.read_parquet(original/'spatial/tissue_positions.parquet',columns=['barcode','in_tissue'])
    tissue=pd.Series(barcodes).map(positions.set_index('barcode').in_tissue).fillna(0).to_numpy()==1
    modeled=sparse.load_npz(out/'components/modeled_cell.npz')
    audit=json.loads((out/'subdiagnostic/localization.json').read_text())
    if modeled.shape!=(len(barcodes),len(genes)) or int(modeled.sum())!=audit['captured_counts']:
        raise ValueError('saved modeled counts failed resume validation')
    finish_pair(out=out,seed=args.seed,master=master,gen=gen,genes=genes,bins=bins,bin_meta=bin_meta,
        covered=covered,tissue=tissue,modeled=modeled,mod=mod,features=features,barcodes=barcodes,original=original,
        raw_h5=Path(manifest['inputs']['raw_h5']['path']),rate=manifest['resolved_config']['background_rate'],
        split=json.loads((out/'donor_split.json').read_text()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assignment-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260922)
    p.add_argument('--resume-background',action='store_true')
    p.add_argument('--geometry-path',type=Path,
        help='Fractional bin GT override for a paired geometry experiment; default preserves the existing dataset.')
    p.add_argument('--fixed-calibration-json',type=Path,
        help='Reuse an existing capture probability and complement rate instead of recalibrating each geometry arm.')
    p.add_argument('--reuse-reference-dir',type=Path,
        help='Reuse an already built donor-disjoint scRNA reference without rebuilding it.')
    p.add_argument('--background-assignment-seed',type=int,
        help='Explicit source-assignment seed for paired geometry arms.')
    p.add_argument('--background-render-seed',type=int,
        help='Explicit background-placement seed for paired geometry arms.')
    args=p.parse_args()
    if args.resume_background:return resume_background(args)
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    diag=out/'subdiagnostic';diag.mkdir();components=out/'components';components.mkdir()
    v=ROOT.parent/'Bin2Cell_Validation';raw_h5=v/'data/single_cell/colorectal/CRC_GSE166555_raw_tumor.h5'
    original=v/'data/visium_hd/colorectal/output/binned_outputs/square_002um'
    gt_path=(args.geometry_path or
        v/'outputs/ground_truth/colorectal_nucleus_based/bins_multi_owner_v2/ground_truth_bin_assignments.parquet').resolve(strict=True)
    master=pd.read_csv(args.assignment_dir/'ground_truth_cell_assignments.csv').sort_values('cell_id').reset_index(drop=True)
    split=json.loads((args.assignment_dir/'donor_split.json').read_text())
    metadata=pd.read_csv(v/'data/single_cell/colorectal/GSE166555_meta_data_tumor.tsv',sep='\t',
                         usecols=['cell','case_id','sct_cell_type'],dtype=str)
    gen=metadata[metadata.case_id.isin(split['generator_donors'])].copy()
    ref=metadata[metadata.case_id.isin(split['reference_donors']) & metadata.sct_cell_type.isin(master.cell_type)].copy()
    if set(gen.cell)&set(ref.cell) or set(gen.case_id)&set(ref.case_id):raise ValueError('generator/reference leakage')
    if not set(master.sc_cell_barcode)<=set(gen.cell):raise ValueError('assigned source is outside generator donors')
    master.to_csv(out/'ground_truth_cell_assignments.csv',index=False)
    gen.to_csv(out/'generator_metadata.tsv',sep='\t',index=False);ref.to_csv(out/'reference_metadata.tsv',sep='\t',index=False)
    dump(out/'donor_split.json',split)
    data=prepare_inputs(protocol.AUDIT,protocol.BENCH,v);genes=data['genes'];bins=data['bins']
    (out/'genes.txt').write_text('\n'.join(genes)+'\n')
    reference_dir=out/'reference'
    if args.reuse_reference_dir:
        source_reference=args.reuse_reference_dir.resolve(strict=True)
        with np.load(source_reference/'reference_counts.npz') as saved:
            types=saved['cell_types'].astype(str)
            rgenes=saved['genes'].astype(str)
        original_genes=(source_reference.parent/'genes.txt').read_text().splitlines()
        if len(types)!=28 or original_genes!=genes or not set(rgenes)<=set(genes):
            raise ValueError('reused reference does not match the fixed cell-type/gene axes')
        reference_dir.symlink_to(source_reference,target_is_directory=True)
    else:
        reference_dir.mkdir()
        counts,types,rgenes,n,summary=build_reference_counts(raw_h5,out/'reference_metadata.tsv',cell_id_column='cell',
            cell_type_column='sct_cell_type',min_cells_per_type=1,gene_allowlist_path=out/'genes.txt')
        if len(types)!=28:raise ValueError(f'expected declared 28 reference types, got {len(types)}')
        write_reference_counts_npz(reference_dir/'reference_counts.npz',reference_counts=counts,cell_types=types,genes=rgenes,n_cells_per_type=n)
        write_summary_json(reference_dir/'summary.json',summary,cell_types=types,n_cells_per_type=n)
    supported=set(types);master[['cell_id','cell_type']].assign(reference_supported=master.cell_type.isin(supported)).to_csv(
        out/'cell_reference_support.csv',index=False)
    print('REFERENCE complete',len(types),len(rgenes),flush=True)
    mod=load_localization_module(v);features,barcodes=mod.load_original_features_and_barcodes(original/'filtered_feature_bc_matrix.h5')
    template=yaml.safe_load((protocol.BENCH/'episode_context/episode_build.capture_thinned.yaml').read_text())
    bin_meta=pd.read_parquet(ROOT/template['inputs']['bins_path'],columns=['matrix_col_index','barcode','x_um','y_um'])
    bin_meta=bin_meta.sort_values('matrix_col_index').reset_index(drop=True)
    if not np.array_equal(bin_meta.matrix_col_index,np.arange(len(barcodes))) or not np.array_equal(bin_meta.barcode,barcodes):
        raise ValueError('original physical bin axes differ')
    gt=mod.validate_fractional_bin_assignments(pd.read_parquet(gt_path),n_barcodes=len(barcodes))
    covered=np.zeros(len(barcodes),bool);covered[gt.barcode_index.to_numpy(int)]=True
    positions=pd.read_parquet(original/'spatial/tissue_positions.parquet',columns=['barcode','in_tissue'])
    tissue=pd.Series(barcodes).map(positions.set_index('barcode').in_tissue).fillna(0).to_numpy()==1
    profiles,pg=selected_raw_counts(raw_h5,master.sc_cell_barcode);profiles=align_genes(profiles,pg,genes)
    gtf=v/'references/gene_annotation/gencode_v38.gtf';membrane=v/'references/gene_localization/human_membrane_secretory_genes_uniprot.txt'
    p0,_,_=mod.localization_probabilities(np.array(genes),profiles.data,profiles.indices,gtf)
    flags=np.isin(genes,list(mod.load_membrane_genes(membrane)))
    names=sorted(bins.patch_id.unique());calibration=bins.patch_id.isin(names[:2]).to_numpy()
    selected=bins.matrix_col_index.to_numpy(int)
    expected=expected_counts_at_bins(profiles,master,gt,selected,mod,p0,flags,args.seed)
    real_totals=np.asarray(data['real'].sum(1)).ravel()
    capture=calibration_probability(real_totals,expected,calibration,covered[selected]) if not args.fixed_calibration_json else None
    cal_out=calibration & ~covered[selected] & tissue[selected]
    if not cal_out.any():raise ValueError('no calibration complement bins')
    rate=float(real_totals[cal_out].mean())
    if args.fixed_calibration_json:
        fixed=json.loads(args.fixed_calibration_json.read_text())
        capture=float(fixed['capture_probability'])
        rate=float(fixed['complement_counts_per_bin'])
        if not np.isfinite(capture) or not 0<capture<=1 or not np.isfinite(rate) or rate<0:
            raise ValueError('fixed calibration values are invalid')
    dump(out/'calibration.json',dict(calibration_patches=names[:2],audit_patches=names[2:],capture_probability=capture,
        expected_calibration_modeled_counts=float(expected[calibration & covered[selected]].sum()),
        target_calibration_modeled_counts=float(real_totals[calibration & covered[selected]].sum()),
        complement_counts_per_bin=rate,background_sigma_um=8,background_source_area_um2=64,
        background_support='tissue and zero geometric cell coverage',technical_background=False,
        fixed_calibration_source=str(args.fixed_calibration_json.resolve()) if args.fixed_calibration_json else None,
        note='One section; mean-depth calibration only, not calibrated biological ambient. Audit counts unused.'))
    write_experiment_manifest(out,config=dict(seed=args.seed,calibration_patches=names[:2],capture_probability=capture,
        background_rate=rate,reference_types=types.tolist(),
        background_assignment_seed=args.background_assignment_seed if args.background_assignment_seed is not None else args.seed+2,
        background_render_seed=args.background_render_seed if args.background_render_seed is not None else args.seed+3),inputs=dict(raw_h5=raw_h5,assignment=args.assignment_dir/'ground_truth_cell_assignments.csv',
        reference_metadata=out/'reference_metadata.tsv',geometry=gt_path,real_counts=protocol.AUDIT/'matrices/real.npz',
        bin_axes=ROOT/template['inputs']['bins_path'],gtf=gtf,membrane=membrane,
        **({'fixed_calibration':args.fixed_calibration_json} if args.fixed_calibration_json else {}),
        **({'reused_reference':args.reuse_reference_dir/'reference_counts.npz'} if args.reuse_reference_dir else {})),protocol=dict(
        reference_main='28 supported types; remaining 8 diagnostic only, no removal from inference',
        paired='identical modeled counts, complement-only structured RNA added',no_PPO=True,no_weight_search=True))
    print('CAPTURE',capture,'BACKGROUND RATE',rate,'PLACING FULL SECTION',flush=True)
    modeled,trace,audit,generation=place_real_profiles(profiles,master,gt,source_module=mod,p0=p0,membrane_flags=flags,
        n_bins=len(barcodes),seed=args.seed,capture_probability=capture,report_progress=True)
    sparse.save_npz(components/'modeled_cell.npz',modeled);sparse.save_npz(components/'modeled_source_bin_counts.npz',trace)
    audit.to_csv(diag/'per_cell_capture.csv.gz',index=False);dump(diag/'localization.json',generation)
    del profiles,trace
    print('MODELED COMPLETE',int(modeled.sum()),flush=True)
    finish_pair(out=out,seed=args.seed,master=master,gen=gen,genes=genes,bins=bins,bin_meta=bin_meta,
        covered=covered,tissue=tissue,modeled=modeled,mod=mod,features=features,barcodes=barcodes,original=original,
        raw_h5=raw_h5,rate=rate,split=split,
        background_assignment_seed=args.background_assignment_seed,
        background_render_seed=args.background_render_seed)


if __name__=='__main__':main()
