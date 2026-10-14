#!/usr/bin/env python3
"""Freeze 4 existing + 12 additional patches before method evaluation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hd_cell_rl.experiment_protocol import write_experiment_manifest
from scripts import run_source_separated_method_comparison as protocol
from scripts.build_source_separated_pseudo import align_genes,dump


def select_patches(all_patches, original, *, n_patches=16, seed=20260923, excluded_ids=()):
    """Geometry and seed only; never read a method outcome or GT accuracy."""
    if n_patches<len(original) or original.patch_id.duplicated().any():raise ValueError('invalid patch target')
    selected=original.copy();selected['cohort']='original4'
    excluded=set(excluded_ids)|set(original.patch_id)
    candidates=all_patches[(all_patches.n_core_cells>=5)&(all_patches.n_core_cells<=20)&
                           ~all_patches.patch_id.isin(excluded)].sort_values('patch_id')
    order=np.random.default_rng(seed).permutation(len(candidates))
    rows=selected.to_dict('records')
    for idx in order:
        if len(rows)==n_patches:break
        row=candidates.iloc[idx].to_dict()
        if any(abs(row['center_x_um']-r['center_x_um'])<200 and
               abs(row['center_y_um']-r['center_y_um'])<200 for r in rows):continue
        row['cohort']='additional12';rows.append(row)
    if len(rows)!=n_patches:raise ValueError('insufficient spatially separated patches')
    result=pd.DataFrame(rows)
    seen=set()
    for raw in result.core_cell_ids:
        cells=set(map(str,json.loads(raw)))
        if seen&cells:raise ValueError('core cell repeated across selected patches')
        seen|=cells
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260923);p.add_argument('--n-patches',type=int,default=16)
    a=p.parse_args();dataset=a.dataset_dir.resolve(strict=True);out=a.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    for sub in ('selection/source_eval','episode_context','inputs','matrices','patch_components/shared'): (out/sub).mkdir(parents=True)
    full_index=ROOT/'runs/colorectal_nucleus_based_patch_episode_build/colorectal_nucleus_based_patches_index.csv'
    original=pd.read_csv(protocol.PATCHES);all_patches=pd.read_csv(full_index)
    excluded=[]
    for path in sorted((ROOT/'runs/subdiagnostic').glob('em_disjoint_test_patch_selection*/test_patches_index.csv')):
        excluded.extend(pd.read_csv(path).patch_id.astype(str))
    old_overfit=ROOT/'runs/colorectal_nucleus_based_patch_episode_build/colorectal_nucleus_based_patches_overfit4_64um_index.csv'
    excluded.extend(pd.read_csv(old_overfit).patch_id.astype(str))
    patches=select_patches(all_patches,original,n_patches=a.n_patches,seed=a.seed,excluded_ids=excluded)
    template=yaml.safe_load((protocol.BENCH/'episode_context/episode_build.capture_thinned.yaml').read_text())
    radius=float(template['environment']['max_center_distance_um'])
    if not np.isfinite(radius) or radius<=0:raise ValueError('invalid candidate distance')
    meta_path=ROOT/template['inputs']['bins_path']
    meta=pd.read_parquet(meta_path,columns=['matrix_col_index','barcode','x_um','y_um']).sort_values('matrix_col_index').reset_index(drop=True)
    if not np.array_equal(meta.matrix_col_index,np.arange(len(meta))):raise ValueError('noncanonical barcode axis')
    nuclei_path=protocol.INTERMEDIATE/'nuclei/colorectal_nucleus_based_multi_owner_v2_nuclei.parquet'
    nuclei=pd.read_parquet(nuclei_path);nuclei['cell_id']=nuclei.cell_id.astype(str)
    all_context=set();core_rows=[];bin_patch=np.full(len(meta),'',dtype=object)
    x=meta.x_um.to_numpy();y=meta.y_um.to_numpy();nx=nuclei.center_x_um.to_numpy();ny=nuclei.center_y_um.to_numpy()
    for idx,row in patches.iterrows():
        core=list(map(str,json.loads(row.core_cell_ids)))
        cell_mask=(nx>=row.outer_x_min-radius)&(nx<=row.outer_x_max+radius)&(ny>=row.outer_y_min-radius)&(ny<=row.outer_y_max+radius)
        context=set(nuclei.loc[cell_mask,'cell_id'])|set(core);all_context|=context
        for name,values in [('patch',sorted(context,key=int)),('core',core),('margin',sorted(context-set(core),key=int))]:
            patches.at[idx,name+'_cell_ids']=json.dumps(values);patches.at[idx,'n_'+name+'_cells']=len(values)
        for direction,bound in [('x_min',row.outer_x_min-radius),('x_max',row.outer_x_max+radius),
                                ('y_min',row.outer_y_min-radius),('y_max',row.outer_y_max+radius)]:
            patches.at[idx,'context_'+direction]=bound
        patches.at[idx,'candidate_max_distance_um']=radius
        # Include complete candidate expression around margin cells too.
        mask=(x>=row.outer_x_min-2*radius)&(x<=row.outer_x_max+2*radius)&(y>=row.outer_y_min-2*radius)&(y<=row.outer_y_max+2*radius)
        if np.any(bin_patch[mask]!=''):raise ValueError('extracted expression contexts overlap')
        bin_patch[mask]=row.patch_id
        core_rows.extend(dict(cell_id=c,patch_id=row.patch_id,cohort=row.cohort) for c in core)
    selected=bin_patch!='';bins=meta[selected].copy();bins['patch_id']=bin_patch[selected];bins=bins.reset_index(drop=True)
    patches.to_csv(out/'selection/test_patches_index.csv',index=False)
    pd.DataFrame(core_rows).to_csv(out/'selection/source_eval/per_episode.csv',index=False)
    nuclei=nuclei[nuclei.cell_id.isin(all_context)].copy()
    if set(nuclei.cell_id)!=all_context:raise ValueError('missing physical nuclei')
    nuclei.to_parquet(out/'episode_context/nuclei_patch_context.parquet',index=False)
    template['inputs']['nuclei_path']=str(out/'episode_context/nuclei_patch_context.parquet')
    template['inputs']['expression']['reference_npz_path']=str(dataset/'reference/reference_counts.npz')
    (out/'episode_context/episode_build.capture_thinned.yaml').write_text(yaml.safe_dump(template,sort_keys=False))
    bins.to_parquet(out/'barcodes.parquet',index=False)
    genes=(dataset/'genes.txt').read_text().splitlines();pd.Series(genes,name='gene').to_csv(out/'genes.csv',index=False)
    global_indices=bins.matrix_col_index.to_numpy(np.int64)
    compact={}
    for key in ('modeled_cell','unmodeled_biological'):
        matrix=sparse.load_npz(dataset/'components'/f'{key}.npz');compact[key]=matrix[global_indices];del matrix
    compact['observed']=compact['modeled_cell']+compact['unmodeled_biological']
    compact['technical_background']=sparse.csr_matrix(compact['observed'].shape,dtype=np.int64)
    shared=out/'patch_components/shared'
    for name,matrix in compact.items():sparse.save_npz(shared/f'{name}.npz',matrix)
    for patch in patches.patch_id:
        directory=out/'patch_components/realizations'/('leave_out_'+patch)/'uniform_budget';directory.mkdir(parents=True)
        for name in compact:(directory/f'{name}.npz').symlink_to(shared/f'{name}.npz')
    v=ROOT.parent/'Bin2Cell_Validation'
    gt=pd.read_parquet(v/'outputs/ground_truth/colorectal_nucleus_based/bins_multi_owner_v2/ground_truth_bin_assignments.parquet')
    contributing=set(gt.loc[gt.barcode_index.isin(global_indices),'cell_id'].astype(int))|set(map(int,all_context))
    gt=gt[gt.cell_id.isin(contributing)].copy();gt.to_parquet(out/'inputs/fractional_gt_contributing_cells.parquet',index=False)
    master=pd.read_csv(dataset/'ground_truth_cell_assignments.csv')
    master[master.cell_id.isin(contributing)].to_csv(out/'inputs/ground_truth_cells.csv.gz',index=False)
    with h5py.File(v/'data/visium_hd/colorectal/output/binned_outputs/square_002um/filtered_feature_bc_matrix.h5') as f:
        m=f['matrix'];bars=m['barcodes'].asstr()[:]
        if not np.array_equal(bars[global_indices],bins.barcode):raise ValueError('real expression barcode mismatch')
        gn=m['features/name'].asstr()[:]
        real=sparse.csr_matrix((m['data'][:],m['indices'][:],m['indptr'][:]),shape=(len(bars),len(gn)))[global_indices]
    real=align_genes(real,gn,genes);sparse.save_npz(out/'matrices/real.npz',real)
    dump(out/'selection/summary.json',dict(seed=a.seed,n_patches=len(patches),n_core_cells=len(core_rows),
        n_context_cells=len(nuclei),n_expression_bins=len(bins),original_patches=original.patch_id.tolist(),
        additional_patches=patches.loc[patches.cohort.eq('additional12'),'patch_id'].tolist(),
        selected_without_method_outcomes=True,excluded_known_patch_ids=sorted(set(excluded)),
        expression_context_extra_um=2*radius,
        patch_outer_width_um=float(patches.outer_x_max.iloc[0]-patches.outer_x_min.iloc[0]),
        core_width_um=float(patches.core_x_max.iloc[0]-patches.core_x_min.iloc[0]),
        candidate_distance_um=radius,same_section_not_independent_donor=True))
    write_experiment_manifest(out,config=dict(seed=a.seed,n_patches=a.n_patches),
        inputs=dict(full_patch_index=full_index,original_patch_index=protocol.PATCHES,selected_patch_index=out/'selection/test_patches_index.csv',
                    dataset=dataset/'experiment_manifest.json',bin_metadata=meta_path,nuclei=nuclei_path),
        protocol=dict(selection='frozen geometry-only random selection; no method outcomes',
                      preserve_original_four=True,no_weight_selection=True,expanded_margin_expression=True))
    print('CONTEXT READY',json.dumps(json.loads((out/'selection/summary.json').read_text())),flush=True)


if __name__=='__main__':main()
