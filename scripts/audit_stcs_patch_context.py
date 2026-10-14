#!/usr/bin/env python3
"""Compare saved STCS candidates with full-section confident nuclear support."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_source_separated_method_comparison import GT


def coordinates(barcodes):
    frame=barcodes.str.extract(r'^s_002um_(\d+)_(\d+)-\d+$')
    if frame.isna().any().any():raise ValueError('unrecognized physical barcode coordinates')
    return frame.to_numpy(np.int64)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark-dir',type=Path,required=True);p.add_argument('--evaluation-context',type=Path,required=True)
    a=p.parse_args();out=a.benchmark_dir.resolve()
    target=pd.read_csv(a.evaluation_context/'selection/source_eval/per_episode.csv',dtype={'cell_id':str})
    truth=pd.read_csv(GT,usecols=['barcode','cell_id'],dtype={'cell_id':str})
    truth=truth[truth.cell_id.isin(target.cell_id)].drop_duplicates('barcode')
    seed=pd.read_csv(out/'common_confident_nuclear_bins.csv.gz',dtype={'cell_id':str})
    seed=seed[seed.is_nuclear.astype(str).str.lower().isin(['true','1'])]
    if seed.barcode.duplicated().any():raise ValueError('nonunique confident seeds')
    tree=cKDTree(coordinates(seed.barcode));xy=coordinates(truth.barcode)
    owners=seed.cell_id.to_numpy();rows=[];missing=[]
    for condition in ('modeled_only','source_separated'):
        directory=out/condition/'methods/stcs'
        summary=json.loads((directory/f'{condition}_stcs_pipeline_summary.json').read_text())
        candidate_files=list(directory.rglob('barcode_candidates.json'))
        if len(candidate_files)!=1:raise ValueError('expected one native candidate artifact')
        candidates=json.loads(candidate_files[0].read_text())
        neighbor_indices=tree.query_ball_point(xy,summary['search_radius'])
        expected_pairs=0;missing_pairs=0;missing_bins=0
        for barcode,indices in zip(truth.barcode,neighbor_indices):
            expected=set(owners[indices]);actual=set(map(str,candidates.get(barcode,[])))
            absent=expected-actual;expected_pairs+=len(expected);missing_pairs+=len(absent)
            missing_bins+=bool(absent)
            missing.extend(dict(condition=condition,barcode=barcode,missing_cell_id=c) for c in sorted(absent))
        restriction=summary['eval_context_restriction']
        rows.append(dict(condition=condition,n_core_cells=len(target),n_core_gt_bins=len(truth),
            search_radius_bins=summary['search_radius'],expected_pairs_from_full_section=expected_pairs,
            missing_candidate_pairs=missing_pairs,core_gt_bins_with_missing_candidates=missing_bins,
            included_nuclear_cells=restriction['n_included_cells'],
            surrounding_nuclear_cells=restriction['n_included_cells']-len(target),
            interpretation='Native spatial nuclear-pseudobulk reference; not scRNA cell-type reference.'))
    diagnostic=out/'subdiagnostic'
    pd.DataFrame(rows).to_csv(diagnostic/'stcs_full_section_candidate_audit.csv',index=False)
    pd.DataFrame(missing,columns=['condition','barcode','missing_cell_id']).to_csv(diagnostic/'stcs_missing_candidate_pairs.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
