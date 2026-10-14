"""Type-matched real-cell sampling with patch and cross-boundary exclusions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def plan_donor_split(targets, metadata, *, patch_size_um=128.0):
    """Enumerate donor splits by coverage/capacity only, never method outcomes.

    Require generator coverage and the within-patch source-pool lower bound.
    Then maximize reference type coverage, types with >=20 reference cells,
    and closeness to a 50:50 donor-count split, in that order. Ties minimize
    source-cell-count imbalance and finally use sorted donor IDs. Twenty is a
    diagnostic support threshold, not a guarantee of reliable type estimation.
    Spatial exclusion feasibility is checked separately by the actual sampler.
    """
    if not np.isfinite(patch_size_um) or patch_size_um <= 0:
        raise ValueError('patch size must be finite and positive')
    donors = sorted(metadata.case_id.unique())
    if not 2 <= len(donors) <= 16:
        raise ValueError('exhaustive donor planner supports 2 to 16 donors')
    labels = sorted(targets.cell_type.unique())
    counts = pd.crosstab(metadata.sct_cell_type, metadata.case_id).reindex(index=labels, columns=donors, fill_value=0)
    t = targets.drop_duplicates('cell_id').copy()
    xy = np.floor(t[['x_um','y_um']].to_numpy()/patch_size_um).astype(np.int64)
    t['px'], t['py'] = xy[:,0], xy[:,1]
    demand = t.groupby(['cell_type','px','py']).size().groupby(level=0).max().reindex(labels).to_numpy()
    values = counts.to_numpy(); total = values.sum(1)
    best = None
    for mask in range(1, (1 << len(donors))-1):
        select = np.array([bool(mask & (1 << i)) for i in range(len(donors))])
        gen = values[:,select].sum(1); ref = total-gen
        if np.any(gen < demand):
            continue
        gen_ids = tuple(np.asarray(donors)[select])
        score = (-int(np.sum(ref>0)), -int(np.sum(ref>=20)),
                 abs(int(select.sum())*2-len(donors)), abs(int(gen.sum())*2-int(total.sum())), gen_ids)
        if best is None or score < best[0]:
            best = (score, select, gen, ref)
    if best is None:
        raise ValueError('no donor-disjoint split covers every target type with sufficient within-patch capacity')
    _, select, gen, ref = best
    audit = pd.DataFrame(dict(cell_type=labels, generator_cells=gen, reference_cells=ref,
                              max_cells_in_one_patch=demand, reference_supported=ref>0))
    return dict(generator_donors=list(np.asarray(donors)[select]),
                reference_donors=list(np.asarray(donors)[~select]),
                reference_missing_types=audit.loc[ref==0,'cell_type'].tolist(),
                reference_below_20_types=audit.loc[ref<20,'cell_type'].tolist(),
                all_types_supported_in_reference=bool(np.all(ref>0))), audit


def assign_spatial_profiles(targets, sources, *, patch_size_um=128.0,
                            exclusion_distance_um=64.0, seed=42):
    """Assign one source per physical cell, independent of input order.

    Targets: cell_id, cell_type, x_um, y_um. Sources: sc_cell_barcode, cell_type.
    Reuse requires different patches AND separation beyond the exclusion radius.
    Distinct real cells can still have similar expression. Exhaustion raises;
    the deterministic greedy sampler never silently samples with replacement.
    Greedy failure does not prove that no feasible assignment exists.
    """
    if not np.isfinite([patch_size_um, exclusion_distance_um]).all() or min(
        patch_size_um, exclusion_distance_um
    ) <= 0:
        raise ValueError('patch size and exclusion distance must be finite and positive')
    target = targets[['cell_id', 'cell_type', 'x_um', 'y_um']].copy()
    source = sources[['sc_cell_barcode', 'cell_type']].copy()
    if target.empty or source.empty or target.isna().any().any() or source.isna().any().any():
        raise ValueError('target/source tables must be nonempty with no missing values')
    if source.sc_cell_barcode.duplicated().any():
        raise ValueError('source cell barcodes must be globally unique')
    # Repeated appearances in overlapping contexts have a single physical identity.
    target = target.drop_duplicates()
    if target.cell_id.duplicated().any():
        raise ValueError('conflicting repeated physical cell metadata')
    ids = pd.to_numeric(target.cell_id, errors='raise').to_numpy()
    if not np.isfinite(ids).all() or np.any(ids != np.floor(ids)) or np.any(ids <= 0):
        raise ValueError('physical cell IDs must be positive integers')
    target['cell_id'] = ids.astype(np.int64)
    if target.cell_id.duplicated().any():
        raise ValueError('physical IDs collide after integer conversion')
    target['cell_type'] = target.cell_type.astype(str)
    source = source.astype(str).sort_values('sc_cell_barcode').reset_index(drop=True)
    if source.sc_cell_barcode.duplicated().any():
        raise ValueError('source cell barcodes collide after string conversion')
    target = target.sort_values('cell_id').reset_index(drop=True)
    xy = target[['x_um', 'y_um']].to_numpy(float)
    if not np.isfinite(xy).all():
        raise ValueError('physical coordinates must be finite micrometres')
    grid = np.floor(xy / patch_size_um).astype(np.int64)
    target['patch_x'], target['patch_y'] = grid[:, 0], grid[:, 1]
    target['patch_id'] = [f'x{x}_y{y}' for x, y in grid]
    chosen = np.full(len(target), '', dtype=object)
    rng = np.random.default_rng(seed)
    for label, group in target.groupby('cell_type', sort=True):
        positions = group.index.to_numpy()
        pool = source.loc[source.cell_type.eq(label), 'sc_cell_barcode'].to_numpy()
        if not len(pool):
            raise ValueError(f'no eligible real scRNA cells for type {label!r}')
        demand = group.groupby('patch_id').size()
        if demand.max() > len(pool):
            raise ValueError(f'type {label!r}: patch {demand.idxmax()} requires '
                             f'{demand.max()} unique sources but only {len(pool)} are eligible')
        tree = cKDTree(xy[positions])
        used_by_patch = {}
        for pos in positions:
            used = used_by_patch.setdefault(target.at[pos, 'patch_id'], set())
            neighbors = tree.query_ball_point(xy[pos], exclusion_distance_um)
            forbidden = used | set(chosen[positions[neighbors]])
            available = pool[~np.isin(pool, list(forbidden))]
            if not len(available):
                raise ValueError(f'type {label!r}, cell {target.at[pos, "cell_id"]}: '
                                 'pool exhausted by patch/boundary exclusions; no replacement fallback')
            chosen[pos] = rng.choice(available)
            used.add(chosen[pos])
    target['sc_cell_barcode'] = chosen
    target['expression_origin'] = 'real_scRNA'
    audit_spatial_assignment(target, source, exclusion_distance_um=exclusion_distance_um)
    return target


def audit_spatial_assignment(assignment, sources, *, exclusion_distance_um):
    """Independently check identity, type, within-patch and boundary constraints."""
    if assignment.cell_id.duplicated().any():
        raise ValueError('multiple assignments for one physical cell')
    expected = assignment.sc_cell_barcode.map(sources.set_index('sc_cell_barcode').cell_type)
    if expected.isna().any() or not np.array_equal(expected, assignment.cell_type):
        raise ValueError('source identity/type mismatch')
    if assignment.duplicated(['patch_id', 'sc_cell_barcode']).any():
        raise ValueError('a source cell was repeated inside a patch')
    closest = np.inf
    for _, group in assignment.groupby('sc_cell_barcode', sort=False):
        if len(group) > 1:
            xy = group[['x_um', 'y_um']].to_numpy()
            distances, _ = cKDTree(xy).query(xy, k=2)
            closest = min(closest, float(distances[:, 1].min()))
    if closest <= exclusion_distance_um:
        raise ValueError('a source cell was repeated within the boundary exclusion distance')
    return dict(n_physical_cells=len(assignment), n_patches=assignment.patch_id.nunique(),
                n_unique_source_cells=assignment.sc_cell_barcode.nunique(),
                n_reuses=len(assignment)-assignment.sc_cell_barcode.nunique(),
                closest_reuse_distance_um=None if not np.isfinite(closest) else closest,
                within_patch_reuse=0, within_exclusion_distance_reuse=0)


def assign_background_profiles(records, modeled, sources, *, seed=42,
                               patch_size_um=128., exclusion_distance_um=64.):
    """Reserve modeled identities; reduce background density on pool exhaustion.

    Counts are conserved per spatial patch and cell type, never by changing a
    source's type. A first pass reserves one feasible source per budget group.
    Failure to retain any source for a positive budget raises explicitly.
    Background IDs here are virtual source IDs, not additional GT cell labels.
    """
    if not np.isfinite([patch_size_um, exclusion_distance_um]).all() or min(
            patch_size_um, exclusion_distance_um) <= 0:
        raise ValueError('invalid source exclusion geometry')
    source = sources[['sc_cell_barcode', 'cell_type']].astype(str).reset_index(drop=True)
    if source.sc_cell_barcode.duplicated().any():
        raise ValueError('duplicate source identity')
    fixed = modeled[['cell_id','cell_type','x_um','y_um','sc_cell_barcode']].copy()
    proposed = records.copy().sort_values('source_center_barcode_index').reset_index(drop=True)
    if proposed.source_center_barcode_index.duplicated().any():
        raise ValueError('duplicate background center')
    if not np.isfinite(proposed[['x_um','y_um','counts']].to_numpy(float)).all():
        raise ValueError('nonfinite background proposal')
    if (proposed.counts < 0).any() or (proposed.counts != np.floor(proposed.counts)).any():
        raise ValueError('background budgets must be nonnegative integers')
    for frame in (fixed, proposed):
        grid = np.floor(frame[['x_um','y_um']].to_numpy(float)/patch_size_um).astype(np.int64)
        frame['patch_id'] = [f'x{x}_y{y}' for x,y in grid]
    audit_spatial_assignment(fixed, source, exclusion_distance_um=exclusion_distance_um)
    proposed['cell_id'] = np.arange(len(proposed), dtype=np.int64)+int(fixed.cell_id.max())+1
    joint = pd.concat([fixed, proposed[['cell_id','cell_type','x_um','y_um','patch_id']]], ignore_index=True)
    xy = joint[['x_um','y_um']].to_numpy(float)
    chosen = np.full(len(joint), -1, dtype=np.int64)
    lookup = dict(zip(source.sc_cell_barcode, source.index))
    chosen[:len(fixed)] = fixed.sc_cell_barcode.map(lookup).to_numpy(np.int64)
    rng = np.random.default_rng(seed)
    for label, group in joint.groupby('cell_type', sort=True):
        positions = group.index.to_numpy()
        pool = source.index[source.cell_type.eq(label)].to_numpy()
        if not len(pool):
            raise ValueError(f'no real source cells for background type {label}')
        tree = cKDTree(xy[positions])
        used_by_patch = {}
        for pos in positions[positions < len(fixed)]:
            used_by_patch.setdefault(joint.at[pos,'patch_id'], set()).add(int(chosen[pos]))
        pending = group[group.index >= len(fixed)]
        groups = [(patch, rng.permutation(g.index.to_numpy())) for patch,g in pending.groupby('patch_id',sort=True)]

        def attempt(pos):
            used = used_by_patch.setdefault(joint.at[pos,'patch_id'], set())
            neighbors = positions[tree.query_ball_point(xy[pos], exclusion_distance_um)]
            forbidden = used | set(chosen[neighbors])
            available = pool[~np.isin(pool, list(forbidden))]
            if not len(available):
                return False
            chosen[pos] = int(rng.choice(available)); used.add(int(chosen[pos]))
            return True

        # Protect small budget groups from a neighboring dense group's second pass.
        for patch, positions_in_patch in groups:
            if not any(attempt(int(pos)) for pos in positions_in_patch):
                budget = proposed.loc[positions_in_patch-len(fixed),'counts'].sum()
                if budget:
                    raise ValueError(f'no feasible background source for {patch}/{label}; budget={budget}')
        for _, positions_in_patch in groups:
            for pos in positions_in_patch:
                if chosen[pos] < 0:
                    attempt(int(pos))
    proposed['source_index'] = chosen[len(fixed):]
    proposed['retained'] = proposed.source_index >= 0
    if 'sc_cell_barcode' in proposed:
        proposed['original_sc_cell_barcode'] = proposed.sc_cell_barcode
    proposed['sc_cell_barcode'] = None
    keep = proposed.retained
    proposed.loc[keep,'sc_cell_barcode'] = source.sc_cell_barcode.to_numpy()[proposed.loc[keep,'source_index']]
    proposed['original_counts'] = proposed.counts.astype(np.int64)
    budgets = []
    for (patch,label), group in proposed.groupby(['patch_id','cell_type'],sort=True):
        kept = group[group.retained]
        total = int(group.original_counts.sum())
        if not len(kept):
            if total:
                raise ValueError('positive budget lost all source cells')
            continue
        extra = total-int(kept.original_counts.sum())
        weights = kept.original_counts.to_numpy(float)
        if weights.sum() == 0:
            weights = np.ones(len(kept))
        shares = extra*weights/weights.sum()
        addition = np.floor(shares).astype(np.int64)
        order = np.argsort(-(shares-addition), kind='stable')
        addition[order[:extra-int(addition.sum())]] += 1
        proposed.loc[kept.index,'counts'] = kept.original_counts.to_numpy()+addition
        budgets.append(dict(patch_id=patch,cell_type=label,proposed_sources=len(group),
                            retained_sources=len(kept),counts_before=total,
                            counts_after=int(proposed.loc[kept.index,'counts'].sum())))
    retained = proposed[proposed.retained].copy()
    retained['expression_origin'] = 'real_scRNA'
    joined = pd.concat([fixed,retained[fixed.columns]],ignore_index=True)
    audit = audit_spatial_assignment(joined, source, exclusion_distance_um=exclusion_distance_um)
    audit['n_source_positions'] = audit.pop('n_physical_cells')
    if int(retained.counts.sum()) != int(records.counts.sum()):
        raise AssertionError('background source reduction changed the UMI budget')
    audit.update(n_modeled_cells=len(fixed),n_background_proposed=len(proposed),
                 n_background_retained=len(retained),n_background_removed=int((~proposed.retained).sum()),
                 background_counts=int(retained.counts.sum()),budgets_conserved=True,
                 patch_size_um=patch_size_um,exclusion_distance_um=exclusion_distance_um)
    return retained, proposed, pd.DataFrame(budgets), audit
