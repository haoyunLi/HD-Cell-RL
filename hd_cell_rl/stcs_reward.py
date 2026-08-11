"""STCS-style patch reward precomputation for controlled patch experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StcsRewardConfig:
    """Small STCS-style reward configuration used by the patch test backend."""

    search_radius_bins: int = 5
    lambda_spatial: float = 0.5
    normalize_distances: bool = True
    target_sum: float = 10_000.0
    n_top_genes: int = 5_000
    pseudobulk_mode: str = "mean"
    n_pcs: int = 50
    spatial_bin_size_um: float = 2.0
    epsilon: float = 1.0e-8

    @classmethod
    def from_mapping(cls, raw: dict[str, object] | None) -> "StcsRewardConfig":
        data = dict(raw or {})
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unsupported patch_training.stcs_reward keys: {unknown}")
        config = cls(**data)
        if int(config.search_radius_bins) <= 0:
            raise ValueError("patch_training.stcs_reward.search_radius_bins must be > 0")
        if float(config.lambda_spatial) < 0.0:
            raise ValueError("patch_training.stcs_reward.lambda_spatial must be >= 0")
        if float(config.target_sum) <= 0.0:
            raise ValueError("patch_training.stcs_reward.target_sum must be > 0")
        if int(config.n_top_genes) <= 0:
            raise ValueError("patch_training.stcs_reward.n_top_genes must be > 0")
        if int(config.n_pcs) <= 0:
            raise ValueError("patch_training.stcs_reward.n_pcs must be > 0")
        if float(config.spatial_bin_size_um) <= 0.0:
            raise ValueError("patch_training.stcs_reward.spatial_bin_size_um must be > 0")
        if str(config.pseudobulk_mode).strip().lower() != "mean":
            raise ValueError("patch_training.stcs_reward.pseudobulk_mode currently supports only 'mean'")
        return config


@dataclass(frozen=True)
class StcsCellPayload:
    """Per-cell inputs for STCS-style score precomputation."""

    cell_id: str
    candidate_bin_ids: tuple[str, ...]
    candidate_expression: np.ndarray
    candidate_bin_xy_um: np.ndarray
    nucleus_center_xy_um: np.ndarray
    initial_membership_mask: np.ndarray


def compute_stcs_reward_scores(
    cells: tuple[StcsCellPayload, ...] | list[StcsCellPayload],
    config: StcsRewardConfig,
) -> tuple[np.ndarray, ...]:
    """Return one fixed score vector per cell, aligned to each candidate bin.

    The score follows the STCS distance idea: transcriptomic PCA distance plus
    fixed spatial distance to the cell's initial seed bins. Higher is better,
    so the returned reward score is ``-(transcriptomic + lambda * spatial)``.
    """
    payloads = tuple(cells)
    if not payloads:
        return tuple()

    gene_idx = _select_top_gene_indices(
        [np.asarray(item.candidate_expression, dtype=np.float32) for item in payloads],
        n_top_genes=int(config.n_top_genes),
    )
    candidate_blocks = [
        np.asarray(item.candidate_expression[:, gene_idx], dtype=np.float32)
        for item in payloads
    ]
    pseudo_blocks = [
        _pseudobulk_mean(block, np.asarray(item.initial_membership_mask, dtype=np.uint8) > 0)
        for block, item in zip(candidate_blocks, payloads, strict=False)
    ]
    all_counts = np.vstack([*candidate_blocks, *[row.reshape(1, -1) for row in pseudo_blocks]]).astype(
        np.float32,
        copy=False,
    )
    normalized = _normalize_log1p_counts(all_counts, target_sum=float(config.target_sum))
    pca = _pca_scores(normalized, n_components=int(config.n_pcs))
    gene_cos_sim = _cosine_similarity_pca_dimensions(pca, epsilon=float(config.epsilon))

    candidate_slices: list[np.ndarray] = []
    start = 0
    for block in candidate_blocks:
        stop = start + int(block.shape[0])
        candidate_slices.append(pca[start:stop])
        start = stop
    pseudo_pca = pca[start : start + len(payloads)]

    t_log_blocks = []
    for cell_idx, candidate_pca in enumerate(candidate_slices):
        delta = candidate_pca - pseudo_pca[int(cell_idx)].reshape(1, -1)
        t_raw = np.einsum("bp,pq,bq->b", delta, gene_cos_sim, delta, optimize=True)
        t_log_blocks.append(np.log(np.maximum(t_raw, float(config.epsilon))).astype(np.float32, copy=False))
    t_log_all = np.concatenate(t_log_blocks, axis=0) if t_log_blocks else np.zeros((0,), dtype=np.float32)

    if bool(config.normalize_distances) and t_log_all.size > 0:
        t_min = float(np.min(t_log_all))
        t_max = float(np.max(t_log_all))
        denom = max(t_max - t_min, float(config.epsilon))
        transcriptomic_blocks = [((values - t_min) / denom).astype(np.float32, copy=False) for values in t_log_blocks]
    else:
        transcriptomic_blocks = [values.astype(np.float32, copy=False) for values in t_log_blocks]

    scores: list[np.ndarray] = []
    for item, transcriptomic in zip(payloads, transcriptomic_blocks, strict=False):
        spatial = _fixed_spatial_distance_bins(
            candidate_xy_um=np.asarray(item.candidate_bin_xy_um, dtype=np.float32),
            seed_mask=np.asarray(item.initial_membership_mask, dtype=np.uint8) > 0,
            nucleus_center_xy_um=np.asarray(item.nucleus_center_xy_um, dtype=np.float32),
            bin_size_um=float(config.spatial_bin_size_um),
        )
        combined = transcriptomic + float(config.lambda_spatial) * spatial.astype(np.float32, copy=False)
        scores.append((-combined).astype(np.float32, copy=False))
    return tuple(scores)


def _select_top_gene_indices(expressions: list[np.ndarray], *, n_top_genes: int) -> np.ndarray:
    if not expressions:
        return np.zeros((0,), dtype=np.int64)
    n_genes = int(expressions[0].shape[1])
    if any(int(item.shape[1]) != n_genes for item in expressions):
        raise ValueError("all STCS expression blocks must have the same gene dimension")
    n_select = min(int(n_top_genes), n_genes)
    total_n = 0
    sums = np.zeros((n_genes,), dtype=np.float64)
    sums_sq = np.zeros((n_genes,), dtype=np.float64)
    for block in expressions:
        arr = np.asarray(block, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("STCS candidate_expression must have shape (B, G)")
        total_n += int(arr.shape[0])
        sums += np.sum(arr, axis=0, dtype=np.float64)
        sums_sq += np.sum(arr.astype(np.float64, copy=False) * arr.astype(np.float64, copy=False), axis=0)
    if total_n <= 0:
        return np.arange(n_select, dtype=np.int64)
    mean = sums / float(total_n)
    variances = np.maximum(sums_sq / float(total_n) - mean * mean, 0.0)
    if n_select >= n_genes:
        return np.arange(n_genes, dtype=np.int64)
    top = np.argpartition(variances, -n_select)[-n_select:]
    return np.asarray(top[np.argsort(top)], dtype=np.int64)


def _pseudobulk_mean(block: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(block, dtype=np.float32)
    mask = np.asarray(seed_mask, dtype=bool)
    if arr.ndim != 2:
        raise ValueError("STCS candidate_expression must have shape (B, G)")
    if arr.shape[0] == 0:
        return np.zeros((arr.shape[1],), dtype=np.float32)
    if mask.shape[0] == arr.shape[0] and np.any(mask):
        return np.mean(arr[mask], axis=0, dtype=np.float32)
    return np.mean(arr, axis=0, dtype=np.float32)


def _normalize_log1p_counts(counts: np.ndarray, *, target_sum: float) -> np.ndarray:
    arr = np.asarray(counts, dtype=np.float32)
    totals = np.sum(arr, axis=1, dtype=np.float64).astype(np.float32)
    scale = np.zeros_like(totals, dtype=np.float32)
    positive = totals > 0.0
    scale[positive] = float(target_sum) / totals[positive]
    return np.log1p(arr * scale.reshape(-1, 1)).astype(np.float32, copy=False)


def _pca_scores(matrix: np.ndarray, *, n_components: int) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("PCA input must have shape (N, G)")
    n_rows, n_cols = int(arr.shape[0]), int(arr.shape[1])
    n_components_i = min(int(n_components), max(1, n_rows - 1), n_cols)
    if n_rows <= 1 or n_cols == 0 or n_components_i <= 0:
        return np.zeros((n_rows, 1), dtype=np.float32)

    centered = arr - np.mean(arr, axis=0, dtype=np.float32, keepdims=True)
    try:
        if min(n_rows, n_cols) <= n_components_i + 1:
            u, s, _vt = np.linalg.svd(centered, full_matrices=False)
            return (u[:, :n_components_i] * s[:n_components_i]).astype(np.float32, copy=False)
        from scipy.sparse.linalg import svds

        u, s, _vt = svds(centered, k=n_components_i)
        order = np.argsort(s)[::-1]
        return (u[:, order] * s[order]).astype(np.float32, copy=False)
    except Exception:
        u, s, _vt = np.linalg.svd(centered, full_matrices=False)
        return (u[:, :n_components_i] * s[:n_components_i]).astype(np.float32, copy=False)


def _cosine_similarity_pca_dimensions(pca: np.ndarray, *, epsilon: float) -> np.ndarray:
    arr = np.asarray(pca, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return np.ones((1, 1), dtype=np.float32)
    dims = arr.T.astype(np.float32, copy=False)
    norms = np.sqrt(np.sum(dims * dims, axis=1, keepdims=True))
    normalized = dims / np.maximum(norms, float(epsilon))
    return (normalized @ normalized.T).astype(np.float32, copy=False)


def _fixed_spatial_distance_bins(
    *,
    candidate_xy_um: np.ndarray,
    seed_mask: np.ndarray,
    nucleus_center_xy_um: np.ndarray,
    bin_size_um: float,
) -> np.ndarray:
    xy = np.asarray(candidate_xy_um, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("candidate_bin_xy_um must have shape (B, 2)")
    mask = np.asarray(seed_mask, dtype=bool)
    if mask.shape[0] == xy.shape[0] and np.any(mask):
        anchors = xy[mask]
    else:
        anchors = np.asarray(nucleus_center_xy_um, dtype=np.float32).reshape(1, 2)
    delta = (xy[:, None, :] - anchors[None, :, :]) / max(float(bin_size_um), 1.0e-8)
    return np.sqrt(np.sum(delta * delta, axis=2, dtype=np.float32)).min(axis=1).astype(np.float32, copy=False)
