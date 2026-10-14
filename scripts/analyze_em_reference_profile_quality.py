#!/usr/bin/env python3
"""Post-inference profile diagnostics; oracle artifacts are evaluation-only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_source_separated_method_comparison as protocol


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--oracle-dir',type=Path,required=True)
    a=p.parse_args();run=a.run_dir.resolve(strict=True);oracle=a.oracle_dir.resolve(strict=True)
    if not (run/'inference_complete.json').is_file():raise RuntimeError('finish and seal inference first')
    current_manifest=json.loads((run/'experiment_manifest.json').read_text())
    oracle_manifest=json.loads((oracle/'experiment_manifest.json').read_text())
    if current_manifest['inputs']['reference']['sha256']!=oracle_manifest['inputs']['reference']['sha256']:
        raise ValueError('oracle and inference reference axes are not verified equal')
    diag=run/'subdiagnostic'
    bins=pd.read_csv(diag/'bin_evaluation.csv.gz',dtype={'gt_cell_id':str,'frozen_em':str,'reference_only':str})
    core=pd.read_csv(diag/'per_cell_results.csv',dtype={'cell_id':str})
    core=core[core.arm=='frozen_em']
    cell_records=[];bin_records=[];scope=[]
    for generator in ('A_current','B_library_offset'):
        scoped=bins[(bins.generator==generator)&bins.gt_core&~bins.nuclear_locked]
        for path in (run/generator/'methods/frozen_em/em_assignment').glob('*.npz'):
            patch_id=path.stem
            with (np.load(path,allow_pickle=False) as frozen,
                  np.load(run/generator/'methods/reference_only/em_assignment'/path.name,allow_pickle=False) as reference,
                  np.load(oracle/generator/'methods/oracle_profile_diagnostic/em_assignment'/path.name,allow_pickle=False) as truth):
                for key in ('cell_ids','barcode_ids','candidate_cell_index','candidate_row_splits'):
                    np.testing.assert_array_equal(frozen[key],reference[key])
                    np.testing.assert_array_equal(frozen[key],truth[key])
                cell_ids=frozen['cell_ids'].astype(str);cell_index={c:i for i,c in enumerate(cell_ids)}
                barcode_index={b:i for i,b in enumerate(frozen['barcode_ids'].astype(str))}
                core_ids=set(core[(core.generator==generator)&(core.patch_id==patch_id)].cell_id)
                for ci,cell in enumerate(cell_ids):
                    if cell not in core_ids:continue
                    ideal=truth['cell_expression_profile'][ci].astype(float);ideal/=ideal.sum()
                    rec=dict(generator=generator,patch_id=patch_id,cell_id=cell,
                        nuclear_umi=float(frozen['nuclear_expression_count_total'][ci]),
                        profile_reliability=float(frozen['cell_profile_reliability'][ci]))
                    for name,data in [('frozen_em',frozen),('reference_only',reference)]:
                        profile=data['cell_expression_profile'][ci].astype(float);profile/=profile.sum()
                        rec[name+'_tv_to_generating_profile']=float(.5*np.abs(profile-ideal).sum())
                        rec[name+'_profile_entropy']=float(-np.sum(profile*np.log(np.maximum(profile,1e-12))))
                    cell_records.append(rec)
                local=scoped[scoped.patch_id==patch_id]
                missing=0
                for row in local.itertuples():
                    bi=barcode_index[row.barcode];lo,hi=frozen['candidate_row_splits'][bi:bi+2]
                    pairs=np.arange(lo,hi);gt_pairs=pairs[frozen['candidate_cell_index'][pairs]==cell_index.get(row.gt_cell_id,-1)]
                    if len(gt_pairs)!=1:missing+=1;continue
                    gp=gt_pairs[0];other=pairs[pairs!=gp]
                    rec=dict(generator=generator,patch_id=patch_id,barcode=row.barcode,gt_cell_id=row.gt_cell_id,
                        n_candidates=len(pairs),counts=row.counts,
                        frozen_correct=row.frozen_em==row.gt_cell_id,reference_correct=row.reference_only==row.gt_cell_id)
                    for name,data in [('frozen_em',frozen),('reference_only',reference)]:
                        ll=data['final_cell_expression_compatibility'].astype(float)
                        score=data['final_combined_score'].astype(float)
                        rec[name+'_gt_cell_ll']=float(ll[gp])
                        rec[name+'_gt_cell_ll_margin']=float(ll[gp]-ll[other].max()) if len(other) else np.nan
                        rec[name+'_gt_total_score_margin']=float(score[gp]-score[other].max()) if len(other) else np.nan
                    bin_records.append(rec)
                scope.append(dict(generator=generator,patch_id=patch_id,core_gt_nonnuclear_scored=len(local),gt_owner_missing_from_candidates=missing))
    cells=pd.DataFrame(cell_records);details=pd.DataFrame(bin_records)
    assert cells.groupby('generator').size().eq(43).all()
    cells.to_csv(diag/'profile_quality_per_cell.csv',index=False)
    details.to_csv(diag/'profile_quality_per_bin.csv.gz',index=False)
    pd.DataFrame(scope).to_csv(diag/'profile_quality_population.csv',index=False)
    cells.groupby('generator').agg(n_cells=('cell_id','size'),
        frozen_tv=('frozen_em_tv_to_generating_profile','mean'),reference_only_tv=('reference_only_tv_to_generating_profile','mean'),
        frozen_entropy=('frozen_em_profile_entropy','mean'),reference_only_entropy=('reference_only_profile_entropy','mean')).to_csv(run/'profile_quality.csv')
    metrics=[c for c in details.columns if c.endswith(('_ll','_margin'))]
    summaries=[]
    for generator,group in details.groupby('generator'):
        summary=dict(generator=generator,n_bins=len(group),n_competing_bins=int((group.n_candidates>1).sum()))
        summary.update({m:float(group[m].mean()) for m in metrics})
        summaries.append(summary)
    pd.DataFrame(summaries).to_csv(run/'nonnuclear_profile_prediction.csv',index=False)
    source=diag/'analysis_source';source.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__),source/Path(__file__).name)
    protocol.dump(diag/'profile_quality_provenance.json',dict(phase='post-inference evaluation only',oracle_dir=str(oracle),
        reference_sha256=current_manifest['inputs']['reference']['sha256'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),GT_profiles_used_for_inference=False))


if __name__=='__main__':main()
