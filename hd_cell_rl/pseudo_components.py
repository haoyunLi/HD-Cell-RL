"""Source-separated pseudo-HD generation primitives.

Real expression is a calibration target, never a replay source. Unknown-source
IDs are not physical GT cell IDs. Capture budgets count observed UMIs, not
uncaptured whole-cell molecules. Spatial support is independent of zero counts.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


def sample_captured_profiles(patterns, budgets, rng):
    """Draw integer observed counts at new budgets without fixing detected genes."""
    patterns = sparse.csr_matrix(patterns, dtype=np.float64)
    budgets = np.asarray(budgets)
    if budgets.shape != (patterns.shape[0],) or not np.isfinite(budgets).all() or np.any(budgets < 0):
        raise ValueError('one finite nonnegative captured budget is required per source')
    if np.any(budgets != np.floor(budgets)) or not np.isfinite(patterns.data).all() or np.any(patterns.data < 0):
        raise ValueError('budgets must be integers and expression patterns nonnegative')
    result = patterns.astype(np.int64)
    for i, budget in enumerate(budgets.astype(np.int64)):
        s, e = patterns.indptr[i:i+2]
        mass = patterns.data[s:e].sum()
        if budget and mass <= 0:
            raise ValueError('positive budget cannot be drawn from an empty profile')
        result.data[s:e] = rng.multinomial(int(budget), patterns.data[s:e]/mass) if mass > 0 else 0
    result.eliminate_zeros()
    return result


def spatial_source_kernels(coordinates, patch_ids, source_centers, source_patch_ids, *, sigma_um, tissue_support):
    """Sparse normalized source x bin kernels; never bridge disjoint patches.

    Gaussian cytoplasmic-like fields are a sensitivity model, not measured cell
    extensions. Crop truncation is normalized locally and explicitly documented.
    """
    xy = np.asarray(coordinates, float); centers = np.asarray(source_centers, float)
    patches = np.asarray(patch_ids).astype(str); source_patches = np.asarray(source_patch_ids).astype(str)
    support = np.asarray(tissue_support, bool)
    if not np.isfinite(sigma_um) or sigma_um <= 0 or xy.shape != (len(patches),2) or support.shape != patches.shape:
        raise ValueError('invalid source geometry or sigma')
    if centers.shape != (len(source_patches),2) or not np.isfinite(xy).all() or not np.isfinite(centers).all():
        raise ValueError('finite aligned 2D coordinates required')
    rr, cc, vv = [], [], []
    for patch in sorted(set(source_patches)):
        allowed = np.flatnonzero((patches == patch) & support)
        if not len(allowed): raise ValueError('source patch has no allowed tissue bins')
        tree = cKDTree(xy[allowed])
        for i in np.flatnonzero(source_patches == patch):
            local = tree.query_ball_point(centers[i], 3*sigma_um)
            if not local: raise ValueError('source has no spatial support')
            bins = allowed[np.sort(local)]
            weight = np.exp(-np.sum((xy[bins]-centers[i])**2,1)/(2*sigma_um**2))
            weight /= weight.sum()
            rr.extend([i]*len(bins)); cc.extend(bins); vv.extend(weight)
    return sparse.csr_matrix((vv,(rr,cc)), shape=(len(centers),len(xy)))


def local_type_probabilities(source_centers, cell_centers, cell_types, *, scale_um=24.):
    """Use nearby modeled types as context, not real per-bin expression labels."""
    centers = np.asarray(cell_centers,float); types=np.asarray(cell_types).astype(str)
    names = np.unique(types)
    if centers.shape != (len(types),2) or not len(types) or not np.isfinite(scale_um) or scale_um <= 0:
        raise ValueError('valid typed centers and positive scale are required')
    tree=cKDTree(centers); probabilities=[]
    for source in np.asarray(source_centers,float):
        indices=tree.query_ball_point(source,3*scale_um)
        if not indices: indices=[int(tree.query(source)[1])]
        weights=np.exp(-np.sum((centers[indices]-source)**2,1)/(2*scale_um**2))
        counts=np.array([weights[types[indices]==t].sum() for t in names])
        probabilities.append(counts/counts.sum())
    return names, np.asarray(probabilities)


def place_source_profiles(profiles, kernels, rng, *, capture_probability=None, capture_diagnostics=None):
    """Allocate integer captured counts and trace every source's bin totals."""
    profiles=sparse.csr_matrix(profiles); kernels=sparse.csr_matrix(kernels)
    if profiles.shape[0] != kernels.shape[0] or np.any(profiles.data < 0) or np.any(profiles.data != np.floor(profiles.data)):
        raise ValueError('source profiles must be aligned nonnegative integer counts')
    if not np.isfinite(kernels.data).all() or np.any(kernels.data < 0) or not np.allclose(np.asarray(kernels.sum(1)).ravel(),1):
        raise ValueError('source kernels must be normalized')
    probability=None if capture_probability is None else np.asarray(capture_probability,float)
    if probability is not None and (probability.shape!=(kernels.shape[1],) or not np.isfinite(probability).all() or
                                    np.any((probability<0)|(probability>1))):
        raise ValueError('capture probabilities must align to bins and lie in [0,1]')
    if probability is not None and np.all(probability==1):probability=None
    capture_rejected=0
    rows=[]; cols=[]; vals=[]; sr=[]; sb=[]; sv=[]
    for i in range(profiles.shape[0]):
        s,e=profiles.indptr[i:i+2]; ks,ke=kernels.indptr[i:i+2]
        bins=kernels.indices[ks:ke]; probabilities=kernels.data[ks:ke]
        # Simulation can allocate count batches. No inference algorithm expands reads.
        for gene, count in zip(profiles.indices[s:e],profiles.data[s:e]):
            draws=rng.multinomial(int(count),probabilities)
            if probability is not None:
                captured=rng.binomial(draws,probability[bins]);capture_rejected+=int((draws-captured).sum());draws=captured
            keep=draws>0
            rows.extend(bins[keep]); cols.extend([gene]*int(keep.sum())); vals.extend(draws[keep])
            sr.extend([i]*int(keep.sum())); sb.extend(bins[keep]); sv.extend(draws[keep])
    matrix=sparse.csr_matrix((vals,(rows,cols)),shape=(kernels.shape[1],profiles.shape[1]),dtype=np.int64)
    trace=sparse.csr_matrix((sv,(sr,sb)),shape=kernels.shape,dtype=np.int64)
    if matrix.sum()+capture_rejected!=profiles.sum() or trace.sum()!=matrix.sum(): raise AssertionError('source count conservation failed')
    if capture_diagnostics is not None:
        capture_diagnostics.update(proposed_counts=int(profiles.sum()),capture_rejected_counts=capture_rejected,
                                   captured_counts=int(matrix.sum()))
    return matrix,trace


def fit_component_budgets(real_totals, calibration, modeled_coverage, unknown_intensity, *, technical_fraction=0.):
    """Fit only two calibration aggregates, not gene vectors or audit counts.

    A zero technical fraction assigns complement signal to the unknown source
    hypothesis. This does not establish that all complement signal is biological.
    """
    real=np.asarray(real_totals,float); cal=np.asarray(calibration,bool)
    covered=np.asarray(modeled_coverage,bool); intensity=np.asarray(unknown_intensity,float)
    if not (real.shape==cal.shape==covered.shape==intensity.shape) or not cal.any():
        raise ValueError('aligned calibration vectors required')
    if not np.isfinite(real).all() or np.any(real<0) or not np.isfinite(intensity).all() or np.any(intensity<0):
        raise ValueError('invalid counts or source intensity')
    if not 0 <= technical_fraction < 1: raise ValueError('technical fraction must be in [0,1)')
    outside=cal & ~covered; inside=cal & covered
    if not outside.any() or not inside.any(): raise ValueError('calibration needs inside and outside regions')
    tech_per_bin=technical_fraction*real[cal].sum()/cal.sum()
    outside_target=real[outside].sum()-tech_per_bin*outside.sum()
    if intensity[outside].sum() <= 0 or outside_target < 0:
        raise ValueError('unknown source support/technical budget cannot explain calibration complement')
    amplitude=outside_target/intensity[outside].sum()
    modeled_target=real[inside].sum()-tech_per_bin*inside.sum()-amplitude*intensity[inside].sum()
    if modeled_target <= 0: raise ValueError('unknown kernel leaks too much into modeled masks for this budget model')
    return {'modeled_calibration_counts':float(modeled_target), 'unknown_amplitude':float(amplitude),
            'technical_counts_per_bin':float(tech_per_bin), 'calibration_total_counts':float(real[cal].sum())}


def combine_components(modeled, unmodeled, technical):
    components=[sparse.csr_matrix(x) for x in [modeled,unmodeled,technical]]
    if any(x.shape!=components[0].shape or not np.isfinite(x.data).all() or
           np.any(x.data<0) or np.any(x.data != np.floor(x.data)) for x in components):
        raise ValueError('components must be nonnegative integer matrices on the same axes')
    components=[x.astype(np.int64) for x in components]
    return sum(components[1:],components[0].copy()).tocsr()
