#!/usr/bin/env python3
"""Audit a versioned background update without consulting method outcomes."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hd_cell_rl.spatial_profile_assignment import audit_spatial_assignment


def audit(dataset, output):
    output.mkdir(parents=True,exist_ok=False)
    complete=json.loads((dataset/'completion.json').read_text())
    parent=Path(complete['parent_dataset'])
    modeled=pd.read_csv(dataset/'ground_truth_cell_assignments.csv')
    sources=pd.read_csv(dataset/'generator_metadata.tsv',sep='\t').rename(
        columns={'cell':'sc_cell_barcode','sct_cell_type':'cell_type'})
    reference=pd.read_csv(dataset/'reference_metadata.tsv',sep='\t')
    background=pd.read_csv(dataset/'subdiagnostic/unmodeled_sources.csv.gz')
    proposals=pd.read_csv(dataset/'subdiagnostic/background_source_proposals.csv.gz')
    old=pd.read_csv(parent/'subdiagnostic/unmodeled_sources.csv.gz')
    if set(sources.sc_cell_barcode)&set(reference.cell) or set(sources.case_id)&set(reference.case_id):
        raise ValueError('generator/reference identity or donor leakage')
    if not np.array_equal(sources.sc_cell_barcode.to_numpy()[background.source_index],background.sc_cell_barcode):
        raise ValueError('background raw profile axis mismatch')
    joint=pd.concat([modeled,background],ignore_index=True)
    result=audit_spatial_assignment(joint,sources,exclusion_distance_um=64.)
    result['n_source_positions']=result.pop('n_physical_cells')
    for name in ('ground_truth_cell_assignments.csv','components/modeled_cell.npz','reference/reference_counts.npz'):
        if (dataset/name).resolve()!=(parent/name).resolve():
            raise ValueError(f'expected unchanged linked input: {name}')
    for table in (old,background):
        grid=np.floor(table[['x_um','y_um']].to_numpy()/128).astype(int)
        table['tile_id']=[f'x{x}_y{y}' for x,y in grid]
    before=old.groupby(['tile_id','cell_type']).counts.sum().sort_index()
    after=background.groupby(['tile_id','cell_type']).counts.sum().sort_index()
    pd.testing.assert_series_equal(before,after)
    removed=proposals[~proposals.retained]
    patches=pd.read_csv(dataset/'evaluation16/selection/test_patches_index.csv')
    rows=[]
    for patch in patches.itertuples():
        inside=(removed.x_um.between(patch.outer_x_min-40,patch.outer_x_max+40)&
                removed.y_um.between(patch.outer_y_min-40,patch.outer_y_max+40))
        rows.append(dict(patch_id=patch.patch_id,cohort=patch.cohort,removed_background_centers_in_expression_context=int(inside.sum())))
    pd.DataFrame(rows).to_csv(output/'reduction_region_coverage.csv',index=False)
    budgets=pd.read_csv(dataset/'subdiagnostic/background_source_budgets.csv.gz')
    reduced=budgets[budgets.proposed_sources>budgets.retained_sources]
    reduced.to_csv(output/'reduced_source_tiles.csv',index=False)
    result.update(n_modeled_cells=len(modeled),n_background_before=len(old),n_background_after=len(background),
        removed_background_sources=len(removed),affected_generation_tiles=int(reduced.patch_id.nunique()),
        n_generation_tiles=int(joint.patch_id.nunique()),background_counts=int(background.counts.sum()),
        all_tile_type_budgets_identical=True,modeled_counts_reference_and_mapping_identical=True,
        generator_reference_cells_and_donors_disjoint=True,background_profiles_from_raw_generator_axis=True,
        old_background_mean_umi=float(old.counts.mean()),new_background_mean_umi=float(background.counts.mean()),
        old_background_max_umi=int(old.counts.max()),new_background_max_umi=int(background.counts.max()),
        n_evaluation_contexts_containing_removed_centers=sum(r['removed_background_centers_in_expression_context']>0 for r in rows),
        limitation='All background identities and RNA draws were renewed; this is not a density-only intervention. '
                   'The sixteen geometry-selected evaluation contexts do not specifically target source-limited regions. '
                   'Conserved tile/type UMI budgets may concentrate RNA at remaining sources.')
    (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();audit(a.dataset_dir.resolve(strict=True),a.output_dir.resolve())
