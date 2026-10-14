#!/usr/bin/env python3
"""Measure actual off-tissue counts on the targeted HD gene panel.

Off-tissue burden is an observable, not an estimate of all within-tissue ambient.
It must not be substituted for the fraction outside synthetic cell masks.
"""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hd-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    positions=pd.read_parquet(args.hd_dir/'spatial/tissue_positions.parquet',
                              columns=['barcode','in_tissue','array_row','array_col'])
    grid=positions[['array_row','array_col']].to_numpy(int)
    tissue=np.zeros(tuple(grid.max(0)+1),bool)
    tissue[grid[:,0],grid[:,1]]=positions.in_tissue.to_numpy()==1
    distances=distance_transform_edt(~tissue)*2.0
    with h5py.File(args.hd_dir/'filtered_feature_bc_matrix.h5') as f:
        panel=set(v.decode() for v in f['matrix/features/id'][:])
    with h5py.File(args.hd_dir/'raw_feature_bc_matrix.h5') as f:
        m=f['matrix'];genes=np.array([v.decode() for v in m['features/name'][:]])
        ids=np.array([v.decode() for v in m['features/id'][:]])
        mask=np.array([v in panel for v in ids])
        if int(mask.sum()) != len(panel):
            raise ValueError('target gene ID panel did not align uniquely')
        indices=pd.Index(positions.barcode).get_indexer(m['barcodes'][:].astype(str))
        if np.any(indices<0):raise ValueError('raw barcodes absent from spatial table')
        in_tissue=positions.in_tissue.to_numpy()[indices]==1
        dd=distances[grid[indices,0],grid[indices,1]]
        ptr=m['indptr'][:].astype(np.int64)
        totals=np.zeros(len(ptr)-1,np.int64)
        gene_tissue=np.zeros(len(genes),np.int64);gene_bg=np.zeros(len(genes),np.int64)
        for start in range(0,len(totals),200000):
            end=min(start+200000,len(totals));s,e=int(ptr[start]),int(ptr[end])
            g=m['indices'][s:e];v=m['data'][s:e].astype(np.int64)
            bins=np.repeat(np.arange(start,end),np.diff(ptr[start:end+1]))
            keep=mask[g]
            totals[start:end]=np.bincount(bins[keep]-start,weights=v[keep],minlength=end-start).astype(np.int64)
            on=in_tissue[bins]&keep;off=(~in_tissue[bins])&keep
            gene_tissue+=np.bincount(g[on],weights=v[on],minlength=len(genes)).astype(np.int64)
            gene_bg+=np.bincount(g[off],weights=v[off],minlength=len(genes)).astype(np.int64)
        rows=[]
        selections={'in_tissue':in_tissue,'off_tissue_all':~in_tissue,
                    'off_tissue_0_10um':(~in_tissue)&(dd<=10),
                    'off_tissue_10_50um':(~in_tissue)&(dd>10)&(dd<=50),
                    'off_tissue_over_50um':(~in_tissue)&(dd>50)}
        for name,k in selections.items():
            x=totals[k]
            rows.append(dict(region=name,n_bins=len(x),total_umi=int(x.sum()),
                             mean_umi=float(x.mean()),median_umi=float(np.median(x)),
                             positive_fraction=float(np.mean(x>0)),p99_umi=float(np.quantile(x,.99))))
    pd.DataFrame(rows).to_csv(args.output_dir/'tissue_background_counts.csv',index=False)
    pd.DataFrame(dict(gene_id=ids[mask],gene=genes[mask],in_tissue_counts=gene_tissue[mask],
                     off_tissue_counts=gene_bg[mask])).to_csv(args.output_dir/'panel_gene_background.csv',index=False)
    result=dict(off_tissue_umi_fraction=float(totals[~in_tissue].sum()/totals.sum()),
                n_target_genes=int(mask.sum()),total_panel_umis=int(totals.sum()),
                warning='off-tissue expression is not total ambient and includes tissue-mask errors; mask-complement inside tissue is a different quantity')
    (args.output_dir/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
