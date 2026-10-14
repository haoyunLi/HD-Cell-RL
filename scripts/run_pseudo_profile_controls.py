#!/usr/bin/env python3
"""Follow up reduced within-type library variability and weak diffuse background."""
import argparse
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_pseudo_realism_experiment import load_h5, vec, measure, place_profiles
from hd_cell_rl.pseudo_realism import thin_counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-dir',type=Path,required=True)
    p.add_argument('--benchmark-dir',type=Path,required=True)
    p.add_argument('--validation-repo',type=Path,required=True)
    args=p.parse_args();out=args.experiment_dir;bench=args.benchmark_dir;v=args.validation_repo
    result=out/'profile_controls';result.mkdir(exist_ok=False)
    bins=pd.read_parquet(out/'barcodes.parquet');genes=pd.read_csv(out/'genes.csv').gene.tolist()
    grid=bins.barcode.str.extract(r's_002um_(\d+)_(\d+)').astype(int).to_numpy()
    patch=pd.Categorical(bins.patch_id,sorted(bins.patch_id.unique())).codes
    glob=bins.matrix_col_index.to_numpy();lookup={int(b):i for i,b in enumerate(glob)}
    coords=bins[['x_um','y_um']].to_numpy()
    real=sparse.load_npz(out/'matrices/real.npz');source=sparse.load_npz(out/'matrices/source_high_depth.npz')
    gt=pd.read_parquet(bench/'inputs/fractional_gt_contributing_cells.parquet')
    local=gt[gt.barcode_index.isin(glob)]
    coverage=local.groupby('barcode_index').cell_coverage_fraction.sum().reindex(glob,fill_value=0).to_numpy()
    background=coverage<=0
    nuclear=local.groupby('barcode_index').nuclear_fraction.sum().reindex(glob,fill_value=0).to_numpy()>0
    dominant=local.sort_values(['barcode_index','cell_coverage_fraction','cell_id'],ascending=[True,False,True]).drop_duplicates('barcode_index').set_index('barcode_index').cell_id
    owner=dominant.reindex(glob,fill_value=-1).to_numpy()
    meta=pd.read_csv(bench/'inputs/ground_truth_cells.csv.gz').sort_values('cell_id');cells=meta.cell_id.to_numpy(int)
    assign=pd.read_csv(bench/'assignment/ground_truth_cell_to_synthetic_scdesign3.csv').set_index('cell_id').loc[cells]
    types=assign.cell_type.to_numpy();groups={int(c):g for c,g in gt.groupby('cell_id')}
    nuc=pd.read_parquet(bench/'episode_context/nuclei_patch_context.parquet')
    core=set(pd.read_csv(bench/'selection/source_eval/per_episode.csv').cell_id)
    sy,sbar,sg=load_h5(bench/'simulation/scdesign3_generator_disjoint/scdesign3_synthetic_counts.h5')
    sidx={b:i for i,b in enumerate(sbar)};sy=sy[[sidx[b] for b in assign.sc_cell_barcode]]
    raw,rbar,rg=load_h5(v/'data/single_cell/colorectal/CRC_GSE166555_raw_tumor.h5')
    rm=pd.read_csv(bench/'simulation/scdesign3_generator_disjoint/model_reference_cell_metadata.tsv',sep='\t')
    ridx={b:i for i,b in enumerate(rbar)};raw=raw[[ridx[b] for b in rm.Cell]]
    def align(x,g):
        gm={name:i for i,name in enumerate(g)}
        transfer=sparse.csr_matrix((np.ones(sum(name in gm for name in genes)),
                       ([gm[name] for name in genes if name in gm],[i for i,name in enumerate(genes) if name in gm])),
                                  shape=(len(g),len(genes)))
        return (x@transfer).astype(np.int64).tocsr()
    sy=align(sy,sg);raw=align(raw,rg)
    rawtot=vec(raw.sum(1));syntot=vec(sy.sum(1))
    module_dir=v/'scripts/simulation';sys.path.insert(0,str(module_dir))
    spec=importlib.util.spec_from_file_location('localization',module_dir/'create_colorectal_pseudo_hd.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    p0,_,_=mod.localization_probabilities(np.array(genes),sy.data,sy.indices,v/'references/gene_annotation/gencode_v38.gtf')
    membrane=mod.load_membrane_genes(v/'references/gene_localization/human_membrane_secretory_genes_uniprot.txt')
    flags=np.array([g in membrane for g in genes])
    ref=np.load(bench/'reference/reference_counts_donor_disjoint.npz',allow_pickle=True)
    refg={str(g):i for i,g in enumerate(ref['genes'])}
    rindices=np.array([i for i,g in enumerate(genes) if g in refg]);rindices=rindices[np.argsort(vec(real[patch<2].sum(0))[rindices])[-500:]]
    reference=ref['reference_counts'][:,[refg[genes[i]] for i in rindices]].astype(float)+1e-8
    reference/=reference.sum(1,keepdims=True)
    allrows=[];diags=[];cellrows=[];profile_rows=[]
    def record(label,x,seed):
        a,b,c=measure(x,real,grid,patch,background,owner,nuclear,nuc.cell_id.to_numpy(),
                      nuc[['center_x_um','center_y_um']].to_numpy(),coords,reference,rindices,label,seed)
        allrows.extend(a);diags.append(b);cellrows.extend([v for v in c if v['cell_id'] in core])
        print(label,seed,b['support_detects_gt_coverage_balanced_accuracy'],flush=True)
        if seed==20260907:sparse.save_npz(result/f'{label}.npz',x)
    # Compare expression variation at equal depth using normalized log profiles.
    for seed in (20260907,20260908,20260909):
        rng=np.random.default_rng(seed)
        donor_indices=np.array([rng.choice(np.flatnonzero(rm.cell_type.to_numpy()==t)) for t in types])
        bootstrap=raw[donor_indices]
        empirical=sy.copy()
        for i,target in enumerate(rawtot[donor_indices]):
            s,e=empirical.indptr[i:i+2]
            empirical.data[s:e]=rng.multinomial(int(target),sy.data[s:e]/syntot[i])
        empirical.eliminate_zeros()
        for label,used in [('empirical_library_scdesign3',empirical),('raw_scRNA_profile_resampling',bootstrap)]:
            for typ in np.unique(types):
                totals=vec(used[types==typ].sum(1))
                profile_rows.append(dict(variant=label,seed=seed,cell_type=typ,n_cells=len(totals),
                                          mean_umi=totals.mean(),umi_cv=totals.std()/max(totals.mean(),1)))
            x,_=place_profiles(used,cells,groups,lookup,len(bins),flags,p0,.25,.25,False,seed,mod)
            captured=thin_counts(x,min(real.sum()/x.sum(),1),np.random.default_rng(seed+91))
            record(label,captured,seed)
        # Actual off-tissue-derived gene profile, with explicitly hypothetical mass fractions.
        bg=pd.read_csv(out/'tissue_background_panel_ids/panel_gene_background.csv').groupby('gene').off_tissue_counts.sum().reindex(genes,fill_value=0).to_numpy(float)
        bg/=bg.sum()
        for frac in (.01,.05):
            n=int(round(real.sum()*frac))
            sampled=sparse.csr_matrix((np.ones(n,dtype=np.int64),
                          (rng.integers(0,len(bins),n),rng.choice(len(genes),n,p=bg))),shape=real.shape)
            x=thin_counts(source,(1-frac)*real.sum()/source.sum(),rng)+sampled
            record(f'off_tissue_profile_fraction_{frac:g}_sensitivity',x,seed)
    pd.DataFrame(allrows).to_csv(result/'metrics.csv',index=False)
    pd.DataFrame(diags).to_csv(result/'support_and_compartments.csv',index=False)
    pd.DataFrame(cellrows).to_csv(result/'support_nearest_per_cell.csv',index=False)
    pd.DataFrame(profile_rows).to_csv(result/'profile_depth.csv',index=False)
    print('COMPLETE',result,flush=True)


if __name__=='__main__':main()
