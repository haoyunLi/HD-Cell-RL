#!/usr/bin/env python3
"""Version an existing real-scRNA pair with jointly exclusive background sources."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hd_cell_rl.spatial_profile_assignment import assign_background_profiles
from hd_cell_rl.experiment_protocol import write_experiment_manifest
from scripts.build_real_scrna_pseudo_hd import selected_raw_counts,load_localization_module
from scripts.build_source_separated_pseudo import align_genes,dump
from scripts.build_real_scrna_paired_dataset import render_background_sources,check_pair


def assign(parent,out,seed):
    out.mkdir(parents=True,exist_ok=False)
    (out/'subdiagnostic').mkdir();(out/'components').mkdir()
    for name in ('ground_truth_cell_assignments.csv','generator_metadata.tsv','reference_metadata.tsv',
                 'donor_split.json','cell_reference_support.csv','genes.txt','genes.csv','reference','calibration.json'):
        (out/name).symlink_to((parent/name).resolve(),target_is_directory=(parent/name).is_dir())
    for name in ('modeled_cell.npz','modeled_source_bin_counts.npz'):
        (out/'components'/name).symlink_to((parent/'components'/name).resolve())
    for name in ('per_cell_capture.csv.gz','localization.json'):
        (out/'subdiagnostic'/name).symlink_to((parent/'subdiagnostic'/name).resolve())
    m=pd.read_csv(parent/'ground_truth_cell_assignments.csv')
    g=pd.read_csv(parent/'generator_metadata.tsv',sep='\t').rename(columns={'cell':'sc_cell_barcode','sct_cell_type':'cell_type'})
    proposals=pd.read_csv(parent/'subdiagnostic/unmodeled_sources.csv.gz')
    kept,all_rows,budgets,audit=assign_background_profiles(proposals,m,g,seed=seed)
    kept.to_csv(out/'subdiagnostic/unmodeled_sources.csv.gz',index=False)
    all_rows.to_csv(out/'subdiagnostic/background_source_proposals.csv.gz',index=False)
    budgets.to_csv(out/'subdiagnostic/background_source_budgets.csv.gz',index=False)
    dump(out/'subdiagnostic/source_exclusivity.json',audit)
    write_experiment_manifest(out,config=dict(parent_dataset=str(parent),seed=seed,patch_size_um=128,
        exclusion_distance_um=64,background_sigma_um=8,background_budget='exact per tile/type conservation'),
        inputs=dict(parent_manifest=parent/'experiment_manifest.json',modeled_assignment=parent/'ground_truth_cell_assignments.csv',
                    original_background_sources=parent/'subdiagnostic/unmodeled_sources.csv.gz',reference=parent/'reference/reference_counts.npz'),
        protocol=dict(no_training=True,no_parameter_selection=True,modeled_counts_unchanged=True,
                      reference_unchanged=True,background_raw_profiles=True,source_exclusivity='modeled plus background',
                      background_interpretation='virtual unmodeled RNA sources, not new GT cells',
                      biological_realism_validated=False))
    print('SOURCE AUDIT',json.dumps(audit),flush=True)


def axes(parent):
    manifest=json.loads((parent/'experiment_manifest.json').read_text())
    meta=pd.read_parquet(manifest['inputs']['bin_axes']['path'],columns=['matrix_col_index','barcode','x_um','y_um']).sort_values('matrix_col_index')
    v=ROOT.parent/'Bin2Cell_Validation';original=v/'data/visium_hd/colorectal/output/binned_outputs/square_002um'
    mod=load_localization_module(v);features,bars=mod.load_original_features_and_barcodes(original/'filtered_feature_bc_matrix.h5')
    if not np.array_equal(meta.matrix_col_index,np.arange(len(bars))) or not np.array_equal(meta.barcode,bars):
        raise ValueError('global bin axes differ')
    gt=pd.read_parquet(manifest['inputs']['geometry']['path'],columns=['barcode_index','cell_id','cell_coverage_fraction'])
    covered=np.zeros(len(bars),bool)
    covered[gt.loc[(gt.cell_id>0)&(gt.cell_coverage_fraction>0),'barcode_index'].to_numpy(int)]=True
    pos=pd.read_parquet(original/'spatial/tissue_positions.parquet',columns=['barcode','in_tissue'])
    tissue=pd.Series(bars).map(pos.set_index('barcode').in_tissue).fillna(0).eq(1).to_numpy()
    return meta,covered,tissue,mod,features,bars,original


def render(parent,out,seed):
    path=out/'components/unmodeled_biological.npz'
    if path.exists():raise FileExistsError(path)
    meta,covered,tissue,_,_,_,_=axes(parent)
    gen=pd.read_csv(out/'generator_metadata.tsv',sep='\t')
    genes=(out/'genes.txt').read_text().splitlines()
    raw,gn=selected_raw_counts(ROOT.parent/'Bin2Cell_Validation/data/single_cell/colorectal/CRC_GSE166555_raw_tumor.h5',gen.cell)
    raw=align_genes(raw,gn,genes)
    records=pd.read_csv(out/'subdiagnostic/unmodeled_sources.csv.gz')
    if not np.array_equal(gen.cell.to_numpy()[records.source_index],records.sc_cell_barcode):
        raise ValueError('raw source index/barcode mapping changed')
    x=render_background_sources(meta[['x_um','y_um']].to_numpy(),tissue & ~covered,raw,records,seed=seed+1)
    sparse.save_npz(path,x)
    dump(out/'subdiagnostic/background_render_checks.json',dict(counts=int(x.sum()),
        positive_bins=int(np.count_nonzero(np.asarray(x.sum(1)))),counts_inside_masks=int(x[covered].sum()),
        counts_outside_tissue=int(x[~tissue].sum()),source_axis_exact=True))


def export(parent,out):
    if (out/'completion.json').exists():raise FileExistsError('already completed')
    _,covered,_,mod,features,bars,original=axes(parent)
    m=sparse.load_npz(out/'components/modeled_cell.npz');b=sparse.load_npz(out/'components/unmodeled_biological.npz')
    observed=check_pair(m,b,covered);del m
    genes=(out/'genes.txt').read_text().splitlines()
    gene_indices=pd.Index(features['name']).get_indexer(genes)
    if np.any(gene_indices<0):raise ValueError('gene axis mismatch')
    for name in ('no_background','with_background'):
        directory=out/name;directory.mkdir(exist_ok=False)
        if name=='no_background':
            (directory/'filtered_feature_bc_matrix').symlink_to((parent/name/'filtered_feature_bc_matrix').resolve(),target_is_directory=True)
        else:
            md=directory/'filtered_feature_bc_matrix';md.mkdir()
            coo=observed.tocoo();full=sparse.coo_matrix((coo.data,(gene_indices[coo.col],coo.row)),shape=(len(features['name']),len(bars)))
            mod.write_matrix_market_gzip(full,md/'matrix.mtx.gz');mod.write_10x_tsvs(md,features,bars)
            del coo,full
        (directory/'spatial').symlink_to((original/'spatial').resolve(),target_is_directory=True)
        (directory/'reference').symlink_to((out/'reference').resolve(),target_is_directory=True)
        (directory/'ground_truth_cell_assignments.csv').symlink_to((out/'ground_truth_cell_assignments.csv').resolve())
        summary=json.loads((parent/name/'pseudo_hd_summary.json').read_text())
        summary.update(paired_dataset=str(out),joint_source_exclusivity=True,
                       modeled_counts_unchanged=True,background_counts=0 if name=='no_background' else int(b.sum()))
        dump(directory/'pseudo_hd_summary.json',summary)
        print('EXPORTED',directory,flush=True)
    complete=json.loads((parent/'completion.json').read_text())
    complete.update(parent_dataset=str(parent),joint_source_exclusivity=True,
                    background_counts=int(b.sum()),modeled_counts_unchanged=True,
                    biological_realism_validated=False)
    dump(out/'completion.json',complete)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-dataset',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--stage',choices=['assign','render','export','all'],default='all');p.add_argument('--seed',type=int,default=20260923)
    a=p.parse_args();parent=a.parent_dataset.resolve(strict=True);out=a.output_dir.resolve()
    if a.stage in ('assign','all'):assign(parent,out,a.seed)
    if a.stage in ('render','all'):render(parent,out,a.seed)
    if a.stage in ('export','all'):export(parent,out)


if __name__=='__main__':main()
