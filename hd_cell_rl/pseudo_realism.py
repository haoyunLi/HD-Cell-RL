"""Count-conserving controls for pseudo spatial data, independent of inference."""
from __future__ import annotations

import numpy as np
from scipy import sparse


def thin_counts(matrix, probability, rng):
    """Binomial thinning; scalar or per-bin probability on a bin x gene matrix."""
    x = sparse.coo_matrix(matrix)
    p = np.asarray(probability)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("retention probabilities must be finite and in [0,1]")
    if p.ndim > 1 or (p.ndim == 1 and len(p) != x.shape[0]):
        raise ValueError("probability axis must match bins")
    if np.any(x.data < 0) or not np.allclose(x.data, np.rint(x.data)):
        raise ValueError("counts must be nonnegative integers")
    n = rng.binomial(x.data.astype(np.int64), p if p.ndim == 0 else p[x.row])
    result = sparse.csr_matrix((n, (x.row, x.col)), shape=x.shape)
    result.eliminate_zeros()
    return result


def move_counts(matrix, grid, patch, sigma_bins, fraction, rng):
    """Displace sampled molecules on the grid; unavailable destinations stay put.

    This is a simulator transport control, not a read-level inference algorithm.
    Moving molecules preserves each gene's total. Rejected moves are counted so
    a cropped boundary cannot silently act as a biological barrier.
    """
    x = sparse.coo_matrix(matrix)
    if sigma_bins < 0 or not 0 <= fraction <= 1:
        raise ValueError("invalid transport parameters")
    if sigma_bins == 0 or fraction == 0:
        return x.tocsr(), 0.0
    lookup = {(int(p), int(a), int(b)): i for i, (p, (a, b)) in enumerate(zip(patch, grid))}
    moving = rng.binomial(x.data.astype(np.int64), fraction)
    sources = np.repeat(x.row, moving)
    genes = np.repeat(x.col, moving)
    offsets = np.rint(rng.normal(0, sigma_bins, (len(sources), 2))).astype(np.int64)
    targets = grid[sources] + offsets
    destinations = np.fromiter(
        (lookup.get((int(patch[s]), int(t[0]), int(t[1])), -1)
         for s, t in zip(sources, targets)), dtype=np.int64, count=len(sources)
    )
    rejected = destinations < 0
    destinations[rejected] = sources[rejected]
    rest = sparse.csr_matrix((x.data - moving, (x.row, x.col)), shape=x.shape)
    transported = sparse.csr_matrix(
        (np.ones(len(sources), dtype=np.int64), (destinations, genes)), shape=x.shape
    )
    result = rest + transported
    result.eliminate_zeros()
    return result, float(rejected.mean()) if len(rejected) else 0.0


def gene_reweight(matrix, efficiencies, rng):
    """Resample gene identities at fixed per-bin counts (platform sensitivity control)."""
    x = sparse.csr_matrix(matrix)
    efficiencies = np.asarray(efficiencies, dtype=float)
    if efficiencies.shape != (x.shape[1],) or np.any(efficiencies <= 0):
        raise ValueError("gene efficiencies must be positive and match genes")
    result = x.copy()
    for b in range(x.shape[0]):
        s, e = x.indptr[b:b + 2]
        if s == e:
            continue
        weights = x.data[s:e] * efficiencies[x.indices[s:e]]
        result.data[s:e] = rng.multinomial(int(x.data[s:e].sum()), weights / weights.sum())
    result.eliminate_zeros()
    return result


def aggregate_bins(matrix, grid, patch, width):
    keys = np.column_stack((patch, grid // width))
    unique, inverse, n = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    mapping = sparse.csr_matrix(
        (np.ones(len(grid)), (inverse, np.arange(len(grid)))), shape=(len(unique), len(grid))
    )
    # Exclude partial 8/16-um blocks so cropped edges do not alter resolution statistics.
    keep = n == width * width
    return (mapping @ matrix)[keep].tocsr(), unique[keep]


def soften_compartment_weights(nuclear, cytoplasm, membrane, membrane_strength=1.0):
    """Relax the hard outer-membrane RNA rule into the non-nuclear area."""
    arrays = [np.asarray(v, dtype=float) for v in (nuclear, cytoplasm, membrane)]
    if not 0 <= membrane_strength <= 1 or any(np.any(v < 0) for v in arrays):
        raise ValueError("invalid compartment weights")
    coverage = sum(arrays)
    def norm(x):
        return x / x.sum() if x.sum() > 0 else coverage / coverage.sum()
    nw, cw, mw = map(norm, arrays)
    nonnuclear = norm(arrays[1] + arrays[2])
    return nw, cw, membrane_strength * mw + (1 - membrane_strength) * nonnuclear
