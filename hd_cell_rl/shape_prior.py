"""Unsupervised morphology shape-prior model and scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from scipy.spatial import ConvexHull, QhullError
except Exception:  # pragma: no cover - scipy is available in normal project envs.
    ConvexHull = None  # type: ignore[assignment]
    QhullError = Exception  # type: ignore[assignment]

SHAPE_FEATURE_NAMES: tuple[str, ...] = ("log_area", "compactness", "solidity", "anisotropy")
FEATURE_PREFIX_COLUMNS: tuple[str, ...] = tuple(f"feature_{name}" for name in SHAPE_FEATURE_NAMES)
SUFFIX_Z_COLUMNS: tuple[str, ...] = tuple(f"{name}_z" for name in SHAPE_FEATURE_NAMES)
PREFIX_Z_COLUMNS: tuple[str, ...] = tuple(f"z_{name}" for name in SHAPE_FEATURE_NAMES)
DEFAULT_EPSILON = 1.0e-4
_LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class ShapeFeatureMatrix:
    """Resolved shape feature matrix and optional raw-to-z scaler."""

    values: np.ndarray
    feature_cols: tuple[str, ...]
    feature_names: tuple[str, ...]
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    zscored_input: bool


@dataclass(frozen=True)
class ShapePriorModel:
    """Gaussian mixture prior over z-scored morphology shape features."""

    feature_names: tuple[str, ...]
    cluster_labels: tuple[str, ...]
    n_cells: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    inv_covariances: np.ndarray
    log_determinants: np.ndarray
    priors: np.ndarray
    epsilon: float
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    zscored_input: bool

    @property
    def n_clusters(self) -> int:
        return int(len(self.cluster_labels))

    @property
    def n_features(self) -> int:
        return int(len(self.feature_names))

    def transform_raw_features(self, raw_features: np.ndarray) -> np.ndarray:
        """Map raw shape features into the z-space used by this model."""
        arr = np.asarray(raw_features, dtype=np.float64)
        return ((arr - self.scaler_mean) / np.maximum(self.scaler_std, 1.0e-12)).astype(np.float64, copy=False)


@dataclass(frozen=True)
class _GridShapeState:
    """Reusable exact shape state for one current membership mask."""

    coords: np.ndarray
    occupied: set[tuple[int, int]]
    area: int
    perimeter: int
    hull_area: float
    hull: Any | None
    sum_x: float
    sum_y: float
    sum_xx: float
    sum_yy: float
    sum_xy: float
    raw_features: np.ndarray


def cluster_reference_shapes(
    feature_df: pd.DataFrame,
    n_clusters: int = 4,
    feature_cols: Iterable[str] | None = None,
    *,
    random_state: int = 7,
    cluster_col: str = "shape_cluster",
) -> pd.DataFrame:
    """Assign unsupervised morphology clusters from z-scored shape features.

    The returned table is a copy of ``feature_df`` with ``cluster_col`` added.
    These labels are morphology groups only; biological cell type is ignored.
    """
    if n_clusters <= 1:
        raise ValueError("n_clusters must be > 1")
    from sklearn.cluster import KMeans

    matrix = resolve_shape_feature_matrix(feature_df, feature_cols=feature_cols)
    if matrix.values.shape[0] < int(n_clusters):
        raise ValueError("n_clusters cannot exceed the number of reference cells")
    kmeans = KMeans(n_clusters=int(n_clusters), random_state=int(random_state), n_init=10)
    labels = kmeans.fit_predict(matrix.values)
    out = feature_df.copy()
    out[cluster_col] = [f"cluster_{idx:02d}" for idx in labels]
    return out


def select_best_cluster_number(
    feature_df: pd.DataFrame,
    k_range: Iterable[int] = range(2, 9),
    feature_cols: Iterable[str] | None = None,
    *,
    random_state: int = 7,
    sample_size: int = 20000,
) -> pd.DataFrame:
    """Evaluate candidate KMeans cluster counts using silhouette score."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    matrix = resolve_shape_feature_matrix(feature_df, feature_cols=feature_cols)
    x = matrix.values
    if x.shape[0] < 3:
        raise ValueError("at least 3 cells are required for silhouette scoring")
    if sample_size > 0 and x.shape[0] > sample_size:
        rng = np.random.default_rng(int(random_state))
        keep = np.sort(rng.choice(x.shape[0], size=int(sample_size), replace=False))
        x_eval = x[keep]
    else:
        x_eval = x

    rows: list[dict[str, Any]] = []
    for k_raw in k_range:
        k = int(k_raw)
        if k <= 1 or k >= x_eval.shape[0]:
            continue
        labels = KMeans(n_clusters=k, random_state=int(random_state), n_init=10).fit_predict(x_eval)
        score = float(silhouette_score(x_eval, labels))
        rows.append({"n_clusters": k, "silhouette_score": score})
    if not rows:
        raise ValueError("no valid k values for silhouette scoring")
    out = pd.DataFrame(rows).sort_values("silhouette_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=np.int32)
    return out


def fit_shape_cluster_distributions(
    feature_df: pd.DataFrame,
    cluster_col: str = "shape_cluster",
    feature_cols: Iterable[str] | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> ShapePriorModel:
    """Fit one regularized Gaussian distribution per morphology cluster."""
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if cluster_col not in feature_df.columns:
        raise ValueError(f"feature table is missing cluster column {cluster_col!r}")
    matrix = resolve_shape_feature_matrix(feature_df, feature_cols=feature_cols)
    clusters = feature_df[cluster_col].fillna("unknown").astype(str).to_numpy()
    finite_cluster = clusters != "unknown"
    x = matrix.values[finite_cluster]
    clusters = clusters[finite_cluster]
    if x.shape[0] == 0:
        raise ValueError("no clustered reference cells available")

    labels = tuple(sorted(set(clusters.tolist())))
    n_features = int(x.shape[1])
    global_cov = _safe_covariance(x, epsilon=epsilon)
    if not np.isfinite(global_cov).all():
        global_cov = np.eye(n_features, dtype=np.float64)

    means: list[np.ndarray] = []
    covs: list[np.ndarray] = []
    inv_covs: list[np.ndarray] = []
    logdets: list[float] = []
    counts: list[int] = []
    total = float(x.shape[0])

    for label in labels:
        group = x[clusters == label]
        counts.append(int(group.shape[0]))
        mu = np.mean(group, axis=0, dtype=np.float64)
        cov = _cluster_covariance(group, global_cov=global_cov, epsilon=epsilon)
        inv_cov, logdet = _stable_inverse_and_logdet(cov, epsilon=epsilon)
        means.append(mu)
        covs.append(cov)
        inv_covs.append(inv_cov)
        logdets.append(float(logdet))

    n_cells = np.asarray(counts, dtype=np.int64)
    priors = n_cells.astype(np.float64) / max(total, 1.0)
    priors = priors / np.sum(priors)
    return ShapePriorModel(
        feature_names=matrix.feature_names,
        cluster_labels=labels,
        n_cells=n_cells,
        means=np.stack(means, axis=0).astype(np.float64),
        covariances=np.stack(covs, axis=0).astype(np.float64),
        inv_covariances=np.stack(inv_covs, axis=0).astype(np.float64),
        log_determinants=np.asarray(logdets, dtype=np.float64),
        priors=priors.astype(np.float64),
        epsilon=float(epsilon),
        scaler_mean=matrix.scaler_mean.astype(np.float64),
        scaler_std=matrix.scaler_std.astype(np.float64),
        zscored_input=bool(matrix.zscored_input),
    )


def gaussian_log_likelihood(z: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    """Compute stable log N(z | mu, cov) for one vector."""
    arr = np.asarray(z, dtype=np.float64)
    mean = np.asarray(mu, dtype=np.float64)
    cov_arr = np.asarray(cov, dtype=np.float64)
    inv_cov, logdet = _stable_inverse_and_logdet(cov_arr, epsilon=DEFAULT_EPSILON)
    return float(_gaussian_log_likelihood_from_inverse(arr, mean, inv_cov, logdet))


def score_shape_against_clusters(z_new: np.ndarray, shape_model: ShapePriorModel) -> dict[str, Any]:
    """Score one z-scored shape vector against every morphology cluster."""
    z = np.asarray(z_new, dtype=np.float64).reshape(-1)
    if z.shape != (shape_model.n_features,):
        raise ValueError(f"z_new must have shape ({shape_model.n_features},), got {z.shape}")
    ll = np.asarray(
        [
            _gaussian_log_likelihood_from_inverse(
                z,
                shape_model.means[i],
                shape_model.inv_covariances[i],
                float(shape_model.log_determinants[i]),
            )
            for i in range(shape_model.n_clusters)
        ],
        dtype=np.float64,
    )
    best_idx = int(np.argmax(ll))
    log_mix = _logsumexp(np.log(np.maximum(shape_model.priors, 1.0e-300)) + ll)
    return {
        "cluster_log_likelihoods": dict(zip(shape_model.cluster_labels, ll.tolist(), strict=False)),
        "log_likelihoods": ll,
        "best_cluster": shape_model.cluster_labels[best_idx],
        "best_cluster_index": best_idx,
        "best_log_likelihood": float(ll[best_idx]),
        "mixture_log_likelihood": float(log_mix),
    }


def compute_shape_reward(z_new: np.ndarray, shape_model: ShapePriorModel, mode: str = "mixture") -> float:
    """Convert shape likelihood into a scalar reward value."""
    score = score_shape_against_clusters(z_new, shape_model)
    value = str(mode).strip().lower()
    if value == "mixture":
        return float(score["mixture_log_likelihood"])
    if value in {"max", "best"}:
        return float(score["best_log_likelihood"])
    raise ValueError("mode must be 'mixture' or 'max'")


def _compute_shape_reward_values(z_values: np.ndarray, shape_model: ShapePriorModel, mode: str = "mixture") -> np.ndarray:
    """Vectorized shape reward for a matrix of z-scored shape vectors."""
    z = np.asarray(z_values, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != shape_model.n_features:
        raise ValueError(f"z_values must have shape (N, {shape_model.n_features})")
    delta = z[:, None, :] - shape_model.means[None, :, :]
    mahal = np.einsum("nkf,kfg,nkg->nk", delta, shape_model.inv_covariances, delta, optimize=True)
    ll = -0.5 * (
        float(shape_model.n_features) * _LOG_2PI
        + shape_model.log_determinants[None, :]
        + mahal
    )
    value = str(mode).strip().lower()
    if value == "mixture":
        return _logsumexp_axis1(np.log(np.maximum(shape_model.priors, 1.0e-300))[None, :] + ll)
    if value in {"max", "best"}:
        return np.max(ll, axis=1).astype(np.float64, copy=False)
    raise ValueError("mode must be 'mixture' or 'max'")


def compute_delta_shape_reward(
    z_before: np.ndarray,
    z_after: np.ndarray,
    shape_model: ShapePriorModel,
    mode: str = "mixture",
) -> float:
    """Return shape reward improvement after a candidate add action."""
    return float(
        compute_shape_reward(z_after, shape_model, mode=mode)
        - compute_shape_reward(z_before, shape_model, mode=mode)
    )


def compute_grid_shape_features_from_mask(
    *,
    membership_mask: np.ndarray,
    candidate_bin_xy_um: np.ndarray,
    bin_size_um: float = 2.0,
    epsilon: float = 1.0e-8,
) -> np.ndarray:
    """Compute raw [log_area, compactness, solidity, anisotropy] from assigned bins.

    Convex hull uses square corners around grid cells, matching the GT reference
    builder. Coordinates are inferred on a regular grid from candidate XY values.
    """
    mask = np.asarray(membership_mask, dtype=np.uint8).astype(bool, copy=False)
    xy = np.asarray(candidate_bin_xy_um, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("candidate_bin_xy_um must have shape (B, 2)")
    if mask.shape != (xy.shape[0],):
        raise ValueError("membership_mask and candidate_bin_xy_um row counts differ")
    coords = _grid_coords_from_xy(xy[mask], bin_size_um=bin_size_um)
    return shape_features_from_grid_coords(coords, epsilon=epsilon)


def shape_features_from_grid_coords(coords: np.ndarray, *, epsilon: float = 1.0e-8) -> np.ndarray:
    """Compute raw shape features from integer grid coordinates."""
    return _build_grid_shape_state(coords, epsilon=epsilon).raw_features


def compute_delta_shape_rewards_for_candidates(
    *,
    membership_mask: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_bin_xy_um: np.ndarray,
    shape_model: ShapePriorModel,
    mode: str = "mixture",
    bin_size_um: float = 2.0,
    epsilon: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute exact-hull delta shape reward for candidate ADD bins."""
    mask = np.asarray(membership_mask, dtype=np.uint8)
    candidate_idx = np.asarray(candidate_indices, dtype=np.int64)
    out = np.zeros((mask.shape[0],), dtype=np.float32)
    xy = np.asarray(candidate_bin_xy_um, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("candidate_bin_xy_um must have shape (B, 2)")
    if mask.shape != (xy.shape[0],):
        raise ValueError("membership_mask and candidate_bin_xy_um row counts differ")
    grid_coords = _grid_coords_from_xy(xy, bin_size_um=bin_size_um)
    current_state = _build_grid_shape_state(grid_coords[mask.astype(bool, copy=False)], epsilon=epsilon)
    current_raw = current_state.raw_features
    current_z = shape_model.transform_raw_features(current_raw)
    current_reward = compute_shape_reward(current_z, shape_model, mode=mode)
    valid_idx = candidate_idx[(candidate_idx >= 0) & (candidate_idx < mask.shape[0])]
    valid_idx = valid_idx[mask[valid_idx] == 0]
    if valid_idx.size == 0:
        return out, current_raw.astype(np.float32), float(current_reward)

    candidate_coords = grid_coords[valid_idx]
    is_new = np.asarray(
        [(int(x), int(y)) not in current_state.occupied for x, y in candidate_coords.tolist()],
        dtype=bool,
    )
    if np.any(is_new):
        new_idx = valid_idx[is_new]
        after_raw = _shape_features_after_adding_grid_coords(
            current_state,
            candidate_coords=candidate_coords[is_new],
            epsilon=epsilon,
        )
        after_z = shape_model.transform_raw_features(after_raw)
        out[new_idx] = (_compute_shape_reward_values(after_z, shape_model, mode=mode) - current_reward).astype(
            np.float32,
            copy=False,
        )
    return out, current_raw.astype(np.float32), float(current_reward)


def resolve_shape_feature_matrix(feature_df: pd.DataFrame, feature_cols: Iterable[str] | None = None) -> ShapeFeatureMatrix:
    """Resolve shape features, preferring existing z-scored columns."""
    if feature_cols is not None:
        cols = tuple(str(c) for c in feature_cols)
        _require_columns(feature_df, cols)
        names = tuple(_canonical_feature_name(c) for c in cols)
        values = _finite_matrix(feature_df, cols)
        if cols == SHAPE_FEATURE_NAMES:
            mu = np.mean(values, axis=0, dtype=np.float64)
            sigma = np.std(values, axis=0, ddof=0, dtype=np.float64)
            sigma = np.where(sigma > 1.0e-12, sigma, 1.0)
            values = ((values - mu) / sigma).astype(np.float64, copy=False)
            return ShapeFeatureMatrix(
                values=values,
                feature_cols=cols,
                feature_names=names,
                scaler_mean=mu.astype(np.float64),
                scaler_std=sigma.astype(np.float64),
                zscored_input=False,
            )
        return ShapeFeatureMatrix(
            values=values,
            feature_cols=cols,
            feature_names=names,
            scaler_mean=np.zeros((len(cols),), dtype=np.float64),
            scaler_std=np.ones((len(cols),), dtype=np.float64),
            zscored_input=True,
        )

    for cols in (FEATURE_PREFIX_COLUMNS, SUFFIX_Z_COLUMNS, PREFIX_Z_COLUMNS):
        if all(col in feature_df.columns for col in cols):
            return ShapeFeatureMatrix(
                values=_finite_matrix(feature_df, cols),
                feature_cols=tuple(cols),
                feature_names=SHAPE_FEATURE_NAMES,
                scaler_mean=np.zeros((len(cols),), dtype=np.float64),
                scaler_std=np.ones((len(cols),), dtype=np.float64),
                zscored_input=True,
            )

    raw_cols = SHAPE_FEATURE_NAMES
    _require_columns(feature_df, raw_cols)
    raw = _finite_matrix(feature_df, raw_cols)
    mu = np.mean(raw, axis=0, dtype=np.float64)
    sigma = np.std(raw, axis=0, ddof=0, dtype=np.float64)
    sigma = np.where(sigma > 1.0e-12, sigma, 1.0)
    return ShapeFeatureMatrix(
        values=((raw - mu) / sigma).astype(np.float64, copy=False),
        feature_cols=tuple(raw_cols),
        feature_names=SHAPE_FEATURE_NAMES,
        scaler_mean=mu.astype(np.float64),
        scaler_std=sigma.astype(np.float64),
        zscored_input=False,
    )


def save_shape_prior_model(model: ShapePriorModel, path: str | Path) -> Path:
    """Save a shape prior model to compressed NPZ."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        feature_names=np.asarray(model.feature_names, dtype="U"),
        cluster_labels=np.asarray(model.cluster_labels, dtype="U"),
        n_cells=model.n_cells.astype(np.int64),
        means=model.means.astype(np.float64),
        covariances=model.covariances.astype(np.float64),
        inv_covariances=model.inv_covariances.astype(np.float64),
        log_determinants=model.log_determinants.astype(np.float64),
        priors=model.priors.astype(np.float64),
        epsilon=np.asarray([model.epsilon], dtype=np.float64),
        scaler_mean=model.scaler_mean.astype(np.float64),
        scaler_std=model.scaler_std.astype(np.float64),
        zscored_input=np.asarray([bool(model.zscored_input)], dtype=np.bool_),
    )
    return out


def load_shape_prior_model(path: str | Path) -> ShapePriorModel:
    """Load a shape prior model saved by ``save_shape_prior_model``."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"shape prior model not found: {model_path}")
    with np.load(model_path, allow_pickle=False) as data:
        return ShapePriorModel(
            feature_names=tuple(data["feature_names"].astype(str).tolist()),
            cluster_labels=tuple(data["cluster_labels"].astype(str).tolist()),
            n_cells=np.asarray(data["n_cells"], dtype=np.int64),
            means=np.asarray(data["means"], dtype=np.float64),
            covariances=np.asarray(data["covariances"], dtype=np.float64),
            inv_covariances=np.asarray(data["inv_covariances"], dtype=np.float64),
            log_determinants=np.asarray(data["log_determinants"], dtype=np.float64),
            priors=np.asarray(data["priors"], dtype=np.float64),
            epsilon=float(np.asarray(data["epsilon"], dtype=np.float64).reshape(-1)[0]),
            scaler_mean=np.asarray(data["scaler_mean"], dtype=np.float64),
            scaler_std=np.asarray(data["scaler_std"], dtype=np.float64),
            zscored_input=bool(np.asarray(data["zscored_input"]).reshape(-1)[0]),
        )


def _cluster_covariance(group: np.ndarray, *, global_cov: np.ndarray, epsilon: float) -> np.ndarray:
    n, d = group.shape
    if n <= d:
        diag = np.var(group, axis=0, ddof=0) if n > 1 else np.diag(global_cov)
        cov = np.diag(np.maximum(diag, np.diag(global_cov) * 0.25))
    else:
        cov = np.cov(group, rowvar=False, bias=False)
    cov = np.asarray(cov, dtype=np.float64)
    if cov.shape != (d, d) or not np.isfinite(cov).all():
        cov = np.asarray(global_cov, dtype=np.float64).copy()
    return _regularize_covariance(cov, epsilon=epsilon)


def _safe_covariance(x: np.ndarray, *, epsilon: float) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape[0] <= 1:
        return np.eye(arr.shape[1], dtype=np.float64)
    cov = np.cov(arr, rowvar=False, bias=False)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]], dtype=np.float64)
    return _regularize_covariance(np.asarray(cov, dtype=np.float64), epsilon=epsilon)


def _regularize_covariance(cov: np.ndarray, *, epsilon: float) -> np.ndarray:
    arr = np.asarray(cov, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("covariance must be square")
    arr = 0.5 * (arr + arr.T)
    arr = arr + float(epsilon) * np.eye(arr.shape[0], dtype=np.float64)
    return arr.astype(np.float64, copy=False)


def _stable_inverse_and_logdet(cov: np.ndarray, *, epsilon: float) -> tuple[np.ndarray, float]:
    arr = _regularize_covariance(cov, epsilon=epsilon)
    sign, logdet = np.linalg.slogdet(arr)
    if sign <= 0 or not np.isfinite(logdet):
        arr = arr + max(float(epsilon), 1.0e-6) * np.eye(arr.shape[0], dtype=np.float64)
        sign, logdet = np.linalg.slogdet(arr)
    inv = np.linalg.pinv(arr, rcond=1.0e-12)
    return inv.astype(np.float64), float(logdet if np.isfinite(logdet) else 0.0)


def _gaussian_log_likelihood_from_inverse(z: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray, logdet: float) -> float:
    delta = np.asarray(z, dtype=np.float64) - np.asarray(mu, dtype=np.float64)
    d = int(delta.shape[0])
    mahal = float(delta @ np.asarray(inv_cov, dtype=np.float64) @ delta)
    return float(-0.5 * (d * _LOG_2PI + float(logdet) + mahal))


def _logsumexp(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    m = float(np.max(arr))
    if not np.isfinite(m):
        return m
    return float(m + np.log(np.sum(np.exp(arr - m))))


def _logsumexp_axis1(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("values must have shape (N, K)")
    m = np.max(arr, axis=1)
    out = m + np.log(np.sum(np.exp(arr - m[:, None]), axis=1))
    out[~np.isfinite(m)] = m[~np.isfinite(m)]
    return out.astype(np.float64, copy=False)


def _finite_matrix(df: pd.DataFrame, cols: Iterable[str]) -> np.ndarray:
    arr = df.loc[:, list(cols)].to_numpy(dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise ValueError("shape feature matrix must have at least one column")
    if not np.isfinite(arr).all():
        raise ValueError("shape feature matrix contains non-finite values")
    return arr


def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [str(col) for col in cols if str(col) not in df.columns]
    if missing:
        raise ValueError(f"shape feature table missing columns: {missing}")


def _canonical_feature_name(col: str) -> str:
    text = str(col)
    if text.startswith("feature_"):
        text = text[len("feature_"):]
    if text.startswith("z_"):
        text = text[len("z_"):]
    if text.endswith("_z"):
        text = text[:-2]
    return text


def _grid_coords_from_xy(xy: np.ndarray, *, bin_size_um: float) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float64)
    if arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    step = max(float(bin_size_um), 1.0e-8)
    # Candidate bins use official 2um centers. Rounding to the 2um lattice keeps
    # exposed-edge and hull calculations aligned with the GT reference builder.
    return np.rint(arr / step).astype(np.int64)


def _build_grid_shape_state(coords: np.ndarray, *, epsilon: float) -> _GridShapeState:
    arr = np.asarray(coords, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    if arr.shape[0] == 0:
        empty = np.zeros((0, 2), dtype=np.int64)
        return _GridShapeState(
            coords=empty,
            occupied=set(),
            area=0,
            perimeter=0,
            hull_area=0.0,
            hull=None,
            sum_x=0.0,
            sum_y=0.0,
            sum_xx=0.0,
            sum_yy=0.0,
            sum_xy=0.0,
            raw_features=np.zeros((4,), dtype=np.float64),
        )

    arr = np.unique(arr, axis=0)
    area = int(arr.shape[0])
    perimeter = int(_count_exposed_edges(arr))
    hull_area, hull = _convex_hull_area_and_hull_for_grid_cells(arr, epsilon=float(epsilon))
    sums = _grid_coord_sums(arr)
    raw = _shape_features_from_components(
        area=area,
        perimeter=perimeter,
        hull_area=hull_area,
        sums=sums,
        epsilon=float(epsilon),
    )
    return _GridShapeState(
        coords=arr,
        occupied={(int(x), int(y)) for x, y in arr.tolist()},
        area=area,
        perimeter=perimeter,
        hull_area=hull_area,
        hull=hull,
        sum_x=sums[0],
        sum_y=sums[1],
        sum_xx=sums[2],
        sum_yy=sums[3],
        sum_xy=sums[4],
        raw_features=raw,
    )


def _shape_features_after_adding_grid_coord(
    state: _GridShapeState,
    *,
    candidate_coord: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    x = int(candidate_coord[0])
    y = int(candidate_coord[1])
    if (x, y) in state.occupied:
        return state.raw_features

    area = int(state.area + 1)
    n_neighbors = int(
        ((x + 1, y) in state.occupied)
        + ((x - 1, y) in state.occupied)
        + ((x, y + 1) in state.occupied)
        + ((x, y - 1) in state.occupied)
    )
    perimeter = int(state.perimeter + 4 - 2 * n_neighbors)
    hull_area = _updated_hull_area_after_adding_grid_cell(
        state,
        candidate_coord=np.asarray([x, y], dtype=np.int64),
        epsilon=float(epsilon),
    )
    sums = (
        float(state.sum_x + x),
        float(state.sum_y + y),
        float(state.sum_xx + x * x),
        float(state.sum_yy + y * y),
        float(state.sum_xy + x * y),
    )
    return _shape_features_from_components(
        area=area,
        perimeter=perimeter,
        hull_area=hull_area,
        sums=sums,
        epsilon=float(epsilon),
    )


def _shape_features_after_adding_grid_coords(
    state: _GridShapeState,
    *,
    candidate_coords: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    coords = np.asarray(candidate_coords, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("candidate_coords must have shape (N, 2)")
    n_candidates = int(coords.shape[0])
    if n_candidates == 0:
        return np.zeros((0, 4), dtype=np.float64)

    xs = coords[:, 0].astype(np.float64)
    ys = coords[:, 1].astype(np.float64)
    n_neighbors = np.asarray(
        [
            int(
                ((int(x) + 1, int(y)) in state.occupied)
                + ((int(x) - 1, int(y)) in state.occupied)
                + ((int(x), int(y) + 1) in state.occupied)
                + ((int(x), int(y) - 1) in state.occupied)
            )
            for x, y in coords.tolist()
        ],
        dtype=np.float64,
    )
    area = np.full((n_candidates,), float(state.area + 1), dtype=np.float64)
    perimeter = float(state.perimeter) + 4.0 - 2.0 * n_neighbors
    hull_area = _updated_hull_areas_after_adding_grid_cells(state, candidate_coords=coords, epsilon=float(epsilon))
    sums = np.column_stack(
        (
            float(state.sum_x) + xs,
            float(state.sum_y) + ys,
            float(state.sum_xx) + xs * xs,
            float(state.sum_yy) + ys * ys,
            float(state.sum_xy) + xs * ys,
        )
    )
    return _shape_features_from_component_arrays(
        area=area,
        perimeter=perimeter,
        hull_area=hull_area,
        sums=sums,
        epsilon=float(epsilon),
    )


def _shape_features_from_components(
    *,
    area: int,
    perimeter: int,
    hull_area: float,
    sums: tuple[float, float, float, float, float],
    epsilon: float,
) -> np.ndarray:
    if area <= 0:
        return np.zeros((4,), dtype=np.float64)
    compactness = float((4.0 * math.pi * float(area)) / max(float(perimeter * perimeter), float(epsilon)))
    hull = float(max(float(hull_area), float(area)))
    solidity = float(area / hull) if hull > epsilon else 1.0
    anisotropy = float(_anisotropy_from_sums(int(area), sums=sums, epsilon=float(epsilon)))
    return np.asarray([math.log(float(area) + 1.0), compactness, solidity, anisotropy], dtype=np.float64)


def _shape_features_from_component_arrays(
    *,
    area: np.ndarray,
    perimeter: np.ndarray,
    hull_area: np.ndarray,
    sums: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    area_arr = np.asarray(area, dtype=np.float64)
    if area_arr.ndim != 1:
        raise ValueError("area must have shape (N,)")
    perimeter_arr = np.asarray(perimeter, dtype=np.float64)
    hull_arr = np.asarray(hull_area, dtype=np.float64)
    sums_arr = np.asarray(sums, dtype=np.float64)
    if perimeter_arr.shape != area_arr.shape or hull_arr.shape != area_arr.shape:
        raise ValueError("area, perimeter, and hull_area must have the same shape")
    if sums_arr.shape != (area_arr.shape[0], 5):
        raise ValueError("sums must have shape (N, 5)")

    out = np.zeros((area_arr.shape[0], 4), dtype=np.float64)
    valid = area_arr > 0
    if not np.any(valid):
        return out
    compactness = (4.0 * math.pi * area_arr[valid]) / np.maximum(perimeter_arr[valid] * perimeter_arr[valid], float(epsilon))
    hull = np.maximum(hull_arr[valid], area_arr[valid])
    solidity = np.where(hull > float(epsilon), area_arr[valid] / hull, 1.0)
    anisotropy = _anisotropy_from_sums_batch(area_arr[valid].astype(np.int64), sums=sums_arr[valid], epsilon=float(epsilon))
    out[valid, 0] = np.log(area_arr[valid] + 1.0)
    out[valid, 1] = compactness
    out[valid, 2] = solidity
    out[valid, 3] = anisotropy
    return out


def _grid_coord_sums(coords: np.ndarray) -> tuple[float, float, float, float, float]:
    arr = np.asarray(coords, dtype=np.float64)
    if arr.shape[0] == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    x = arr[:, 0]
    y = arr[:, 1]
    return (
        float(np.sum(x)),
        float(np.sum(y)),
        float(np.sum(x * x)),
        float(np.sum(y * y)),
        float(np.sum(x * y)),
    )


def _anisotropy_from_sums(
    n: int,
    *,
    sums: tuple[float, float, float, float, float],
    epsilon: float,
) -> float:
    if int(n) < 3:
        return 1.0
    sum_x, sum_y, sum_xx, sum_yy, sum_xy = sums
    inv_n = 1.0 / float(n)
    mean_x = sum_x * inv_n
    mean_y = sum_y * inv_n
    cov = np.asarray(
        [
            [sum_xx * inv_n - mean_x * mean_x, sum_xy * inv_n - mean_x * mean_y],
            [sum_xy * inv_n - mean_x * mean_y, sum_yy * inv_n - mean_y * mean_y],
        ],
        dtype=np.float64,
    )
    cov = 0.5 * (cov + cov.T)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return 1.0
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    major = float(max(eigvals[0], 0.0))
    minor = float(max(eigvals[1], 0.0))
    if major <= epsilon:
        return 1.0
    return float(major / max(minor, epsilon))


def _anisotropy_from_sums_batch(
    n: np.ndarray,
    *,
    sums: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    n_arr = np.asarray(n, dtype=np.float64)
    sums_arr = np.asarray(sums, dtype=np.float64)
    if n_arr.ndim != 1 or sums_arr.shape != (n_arr.shape[0], 5):
        raise ValueError("n must have shape (N,) and sums must have shape (N, 5)")
    out = np.ones((n_arr.shape[0],), dtype=np.float64)
    valid = n_arr >= 3
    if not np.any(valid):
        return out

    s = sums_arr[valid]
    inv_n = 1.0 / n_arr[valid]
    mean_x = s[:, 0] * inv_n
    mean_y = s[:, 1] * inv_n
    cov_xx = s[:, 2] * inv_n - mean_x * mean_x
    cov_yy = s[:, 3] * inv_n - mean_y * mean_y
    cov_xy = s[:, 4] * inv_n - mean_x * mean_y
    cov = np.empty((s.shape[0], 2, 2), dtype=np.float64)
    cov[:, 0, 0] = cov_xx
    cov[:, 0, 1] = cov_xy
    cov[:, 1, 0] = cov_xy
    cov[:, 1, 1] = cov_yy
    finite = np.isfinite(cov).all(axis=(1, 2))
    if not np.any(finite):
        return out
    eigvals = np.linalg.eigvalsh(cov[finite])
    major = np.maximum(eigvals[:, 1], 0.0)
    minor = np.maximum(eigvals[:, 0], 0.0)
    values = np.where(major <= float(epsilon), 1.0, major / np.maximum(minor, float(epsilon)))
    valid_positions = np.flatnonzero(valid)
    out[valid_positions[finite]] = values
    return out


def _updated_hull_area_after_adding_grid_cell(
    state: _GridShapeState,
    *,
    candidate_coord: np.ndarray,
    epsilon: float,
) -> float:
    if state.hull is not None and _grid_cell_inside_hull(candidate_coord, state.hull, epsilon=float(epsilon)):
        return float(max(state.hull_area, float(state.area + 1)))
    if state.coords.shape[0] == 0:
        after = np.asarray([candidate_coord], dtype=np.int64)
    else:
        after = np.vstack((state.coords, np.asarray(candidate_coord, dtype=np.int64).reshape(1, 2)))
    return float(_convex_hull_area_for_grid_cells(after, epsilon=float(epsilon)))


def _updated_hull_areas_after_adding_grid_cells(
    state: _GridShapeState,
    *,
    candidate_coords: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    coords = np.asarray(candidate_coords, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("candidate_coords must have shape (N, 2)")
    out = np.full((coords.shape[0],), float(max(state.hull_area, float(state.area + 1))), dtype=np.float64)
    if coords.shape[0] == 0:
        return out

    if state.hull is not None:
        needs_hull = ~_grid_cells_inside_hull(coords, state.hull, epsilon=float(epsilon))
    else:
        needs_hull = np.ones((coords.shape[0],), dtype=bool)
    if not np.any(needs_hull):
        return out

    for idx in np.flatnonzero(needs_hull):
        coord = coords[idx]
        if state.coords.shape[0] == 0:
            after = np.asarray([coord], dtype=np.int64)
        else:
            after = np.vstack((state.coords, coord.reshape(1, 2)))
        out[idx] = float(_convex_hull_area_for_grid_cells(after, epsilon=float(epsilon)))
    return out


def _grid_cell_inside_hull(coord: np.ndarray, hull: Any, *, epsilon: float) -> bool:
    equations = getattr(hull, "equations", None)
    if equations is None:
        return False
    eq = np.asarray(equations, dtype=np.float64)
    if eq.ndim != 2 or eq.shape[1] < 3:
        return False
    corners = _grid_cell_corners(coord)
    signed = corners @ eq[:, :-1].T + eq[:, -1]
    return bool(np.all(signed <= max(float(epsilon) * 10.0, 1.0e-9)))


def _grid_cells_inside_hull(coords: np.ndarray, hull: Any, *, epsilon: float) -> np.ndarray:
    equations = getattr(hull, "equations", None)
    arr = np.asarray(coords, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    if equations is None:
        return np.zeros((arr.shape[0],), dtype=bool)
    eq = np.asarray(equations, dtype=np.float64)
    if eq.ndim != 2 or eq.shape[1] < 3:
        return np.zeros((arr.shape[0],), dtype=bool)
    corners = _grid_cell_corners_batch(arr)
    signed = np.einsum("ncd,hd->nch", corners, eq[:, :-1], optimize=True) + eq[:, -1][None, None, :]
    return np.all(signed <= max(float(epsilon) * 10.0, 1.0e-9), axis=(1, 2))


def _grid_cell_corners(coord: np.ndarray) -> np.ndarray:
    x = int(coord[0])
    y = int(coord[1])
    return np.asarray(
        (
            (x, y),
            (x + 1, y),
            (x, y + 1),
            (x + 1, y + 1),
        ),
        dtype=np.float64,
    )


def _grid_cell_corners_batch(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    offsets = np.asarray(((0, 0), (1, 0), (0, 1), (1, 1)), dtype=np.float64)
    return arr.astype(np.float64)[:, None, :] + offsets[None, :, :]


def _count_exposed_edges(coords: np.ndarray) -> int:
    occupied = {(int(x), int(y)) for x, y in np.asarray(coords, dtype=np.int64).tolist()}
    exposed = 0
    for x, y in occupied:
        exposed += int((x + 1, y) not in occupied)
        exposed += int((x - 1, y) not in occupied)
        exposed += int((x, y + 1) not in occupied)
        exposed += int((x, y - 1) not in occupied)
    return int(exposed)


def _convex_hull_area_and_hull_for_grid_cells(coords: np.ndarray, *, epsilon: float) -> tuple[float, Any | None]:
    arr = np.asarray(coords, dtype=np.int64)
    area = float(arr.shape[0])
    if arr.shape[0] <= 2 or ConvexHull is None:
        return area, None
    corners = np.vstack(
        (
            arr,
            arr + np.asarray([1, 0], dtype=np.int64),
            arr + np.asarray([0, 1], dtype=np.int64),
            arr + np.asarray([1, 1], dtype=np.int64),
        )
    ).astype(np.float64)
    corners = np.unique(corners, axis=0)
    if corners.shape[0] < 3:
        return area, None
    try:
        hull = ConvexHull(corners)
        hull_area = float(hull.volume)
    except (QhullError, ValueError):
        return area, None
    if not np.isfinite(hull_area) or hull_area <= epsilon:
        return area, None
    return float(max(hull_area, area)), hull


def _convex_hull_area_for_grid_cells(coords: np.ndarray, *, epsilon: float) -> float:
    hull_area, _ = _convex_hull_area_and_hull_for_grid_cells(coords, epsilon=epsilon)
    return float(hull_area)
