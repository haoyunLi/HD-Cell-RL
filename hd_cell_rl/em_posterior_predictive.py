"""Experimental nuclear-posterior compatibility; no GT or file access.

Integrate the *whole bin* under a Dirichlet posterior rather than evaluating
its expression at one posterior mean. Scores omit the multinomial coefficient,
as does the existing plug-in LL, and use the same per-UMI scale. Including that
coefficient only in L_cell would introduce a spurious cell-dependent term when
mixed with the cell-varying lambda. This does not turn generalized EM into an
exact finite-mixture MLE algorithm.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.special import gammaln

from . import em_assignment as em


def nuclear_dirichlet_posterior(em_input, config, bin_gene_counts, reference_theta):
    """Match the existing initializer AND scorer's epsilon regularization.

    Before roundoff regularization, alpha = nuclear counts + m*(q0 @ theta),
    where m is the existing kappa-scaled positive nuclear-UMI patch median.
    The final mean is the same regularized mean used by the plug-in scorer;
    total concentration is observed nuclear UMI + m + G*epsilon. Neither
    ambiguous bins nor generating profiles enter this posterior.
    """
    if (not config.cell_specific_expression_enabled
            or config.cell_profile_update_damping != 0.0
            or config.cell_profile_reliability_mode != 'nuclear_depth'
            or config.cell_profile_compatibility_mode != 'standard'):
        raise ValueError('predictive experiment requires frozen standard nuclear-depth profiles')
    counts = em._validate_bin_gene_counts(bin_gene_counts, n_bins=len(em_input.barcode_ids))
    theta = em._normalize_reference_theta(reference_theta, n_types=em_input.ll.shape[1],
        n_genes=counts.shape[1], epsilon=config.epsilon)
    profile, reliability, totals, prior = em.initialize_cell_expression_profiles(
        em_input=em_input, bin_gene_counts=counts, reference_theta=theta,
        cell_type_posterior=em.initialize_q_from_nuclear_seeds(em_input),
        relative_prior_strength=config.cell_profile_relative_prior_strength, epsilon=config.epsilon)
    mean = em._normalize_cell_gene_counts(profile, epsilon=config.epsilon)
    if np.any(mean < config.epsilon):
        raise ValueError('epsilon clipping would change the matched posterior mean')
    concentration = totals + prior + counts.shape[1]*config.epsilon
    return mean * concentration[:, None], profile, reliability, totals, prior


def _log_rising_factorial(a, count):
    """Stable log Gamma(a+n)-log Gamma(a); counts stay aggregated by gene.

    Small integer count groups use a finite product to avoid cancellation at
    large concentration. This never expands bin counts into synthetic reads.
    """
    a, count = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(count, dtype=float))
    result = gammaln(a + count) - gammaln(a)
    small = count <= 64
    for n in np.unique(count[small]).astype(int):
        selected = small & (count == n)
        value = np.zeros(np.count_nonzero(selected))
        for offset in range(n):
            value += np.log(a[selected] + offset)
        result[selected] = value
    return result


def posterior_predictive_compatibility(*, bin_gene_counts, posterior_alpha,
                                     pair_bin_index, candidate_cell_index):
    """Sparse valid-pair scoring, O(candidate pairs × detected genes), no B×C×G.

    L = [sum_g log rising(alpha_cg, x_bg) - log rising(A_c, N_b)] / N_b.
    N_b=0 gives zero. With N_b=1 this is exactly log posterior mean.
    """
    counts = sparse.csr_matrix(bin_gene_counts, dtype=float, copy=True)
    counts.sum_duplicates()
    counts.eliminate_zeros()
    if (not np.isfinite(counts.data).all() or np.any(counts.data < 0)
            or np.any(counts.data != np.rint(counts.data))):
        raise ValueError('posterior predictive requires nonnegative integer observed counts')
    alpha = np.asarray(posterior_alpha, dtype=float)
    if (alpha.ndim != 2 or alpha.shape[1] != counts.shape[1]
            or not np.isfinite(alpha).all() or np.any(alpha <= 0)):
        raise ValueError('posterior_alpha must be strictly positive finite (C,G)')
    bins = np.asarray(pair_bin_index, dtype=np.int64)
    cells = np.asarray(candidate_cell_index, dtype=np.int64)
    if (bins.ndim != 1 or bins.shape != cells.shape or np.any(bins < 0)
            or np.any(bins >= counts.shape[0]) or np.any(cells < 0) or np.any(cells >= alpha.shape[0])):
        raise ValueError('invalid sparse pair axes')
    totals = np.asarray(counts.sum(1)).ravel()
    concentration = alpha.sum(1)
    result = np.zeros(len(bins))
    for cell in np.unique(cells):
        positions = np.flatnonzero(cells == cell)
        x = counts[bins[positions]].tocsr()
        row = np.repeat(np.arange(len(positions)), np.diff(x.indptr))
        terms = _log_rising_factorial(alpha[cell, x.indices], x.data)
        numerator = np.bincount(row, weights=terms, minlength=len(positions))
        n = totals[bins[positions]]
        raw = numerator - _log_rising_factorial(concentration[cell], n)
        result[positions] = np.divide(raw, n, out=np.zeros_like(raw), where=n > 0)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite predictive compatibility')
    return result
