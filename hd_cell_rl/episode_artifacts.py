"""Episode artifact and expression-loading helpers shared by PPO tools."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import yaml

from .matrix_io import resolve_matrix_csc_h5_path
from .ppo_config import ConfigError
from .reward import compute_bin_log_likelihood_by_type

_ARTIFACT_SHARD_CACHE: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
_ARTIFACT_SHARD_CACHE_SIZE = 8


@dataclass(frozen=True)
class PreparedEpisodeArtifact:
    """Episode payload with static per-bin terms loaded once."""

    cell_id: str
    candidate_bin_ids: tuple[str, ...]
    initial_membership_mask: np.ndarray
    candidate_bin_xy_um: np.ndarray
    nucleus_center_xy_um: np.ndarray
    bin_count_totals: np.ndarray
    precomputed_ll: np.ndarray
    precomputed_d_other_um: np.ndarray
    matched_gt_cell_id: str | None = None
    gt_candidate_mask: np.ndarray | None = None
    gt_candidate_bin_count: int = 0
    gt_full_bin_count: int = 0
    eval_size_ratio: float | None = None
    candidate_expression: np.ndarray | None = None
    candidate_matrix_col_index: np.ndarray | None = None


@dataclass(frozen=True)
class EpisodeArtifactLocator:
    """Resolved storage reference for one episode artifact."""

    path: Path
    member_index: int | None


@dataclass(frozen=True)
class NucleiSpatialIndex:
    """Fast nearest-neighbor lookup over nucleus centers."""

    centers_xy_um: np.ndarray
    cell_id_to_index: dict[str, int]
    tree: cKDTree


def _build_nuclei_spatial_index(nuclei_centers_by_cell: dict[str, np.ndarray]) -> NucleiSpatialIndex:
    """Build a KD-tree over nucleus centers for nearest-other distance queries."""
    if not nuclei_centers_by_cell:
        raise ValueError("nuclei centers table is empty")

    cell_ids = list(nuclei_centers_by_cell.keys())
    centers = np.vstack([np.asarray(nuclei_centers_by_cell[cell_id], dtype=np.float64) for cell_id in cell_ids])
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("nuclei center table must have shape (N, 2)")
    if not np.isfinite(centers).all():
        raise ValueError("nuclei centers contain non-finite values")

    return NucleiSpatialIndex(
        centers_xy_um=centers,
        cell_id_to_index={cell_id: idx for idx, cell_id in enumerate(cell_ids)},
        tree=cKDTree(centers),
    )


def _nearest_other_nucleus_distances(
    candidate_bin_xy_um: np.ndarray,
    cell_id: str,
    nuclei_spatial_index: NucleiSpatialIndex,
) -> np.ndarray:
    """Return distance from each candidate bin to the nearest other nucleus center."""
    bin_xy = np.asarray(candidate_bin_xy_um, dtype=np.float64)
    if bin_xy.ndim != 2 or bin_xy.shape[1] != 2:
        raise ValueError("candidate_bin_xy_um must have shape (B, 2)")
    if bin_xy.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)

    if cell_id not in nuclei_spatial_index.cell_id_to_index:
        raise ValueError(f"cell_id {cell_id!r} not found in nuclei spatial index")

    n_nuclei = int(nuclei_spatial_index.centers_xy_um.shape[0])
    if n_nuclei <= 1:
        return np.full(bin_xy.shape[0], np.inf, dtype=np.float64)

    own_idx = int(nuclei_spatial_index.cell_id_to_index[cell_id])
    query_k = int(min(8, n_nuclei))
    dists, nn_idx = nuclei_spatial_index.tree.query(bin_xy, k=query_k)

    if query_k == 1:
        dists = np.asarray(dists, dtype=np.float64)[:, None]
        nn_idx = np.asarray(nn_idx, dtype=np.int64)[:, None]
    else:
        dists = np.asarray(dists, dtype=np.float64)
        nn_idx = np.asarray(nn_idx, dtype=np.int64)

    out = np.full(bin_xy.shape[0], np.inf, dtype=np.float64)
    unresolved = np.ones(bin_xy.shape[0], dtype=bool)
    for rank in range(query_k):
        choose = unresolved & (nn_idx[:, rank] != own_idx)
        if np.any(choose):
            out[choose] = dists[choose, rank]
            unresolved[choose] = False
        if not np.any(unresolved):
            break

    if np.any(unresolved):
        other_centers = np.delete(nuclei_spatial_index.centers_xy_um, own_idx, axis=0)
        deltas = bin_xy[unresolved, None, :] - other_centers[None, :, :]
        brute = np.sqrt(np.sum(deltas * deltas, axis=2))
        out[unresolved] = np.min(brute, axis=1)

    return out


def _load_one_episode_artifact(
    artifact_path: str | Path,
    cell_id: str,
    expression_loader: "_MatrixOnDemandExpressionLoader | None",
    theta: np.ndarray,
    log_theta: np.ndarray,
    nuclei_spatial_index: NucleiSpatialIndex,
    include_candidate_bin_ids: bool = True,
) -> PreparedEpisodeArtifact | None:
    locator = _parse_episode_artifact_locator(artifact_path)
    if not locator.path.exists():
        raise FileNotFoundError(f"episode artifact not found: {locator.path}")

    if locator.member_index is None:
        candidate_bin_ids, candidate_bin_xy_um, nucleus_center_xy_um, candidate_expression, col_index = (
            _load_legacy_episode_artifact_payload(
                artifact_path=locator.path,
                include_candidate_bin_ids=include_candidate_bin_ids,
            )
        )
    else:
        candidate_bin_ids, candidate_bin_xy_um, nucleus_center_xy_um, candidate_expression, col_index = (
            _load_sharded_episode_artifact_payload(
                locator=locator,
                cell_id=cell_id,
                include_candidate_bin_ids=include_candidate_bin_ids,
            )
        )

    if candidate_expression is not None:
        expr = np.asarray(candidate_expression, dtype=np.float64)
        ll = compute_bin_log_likelihood_by_type(bin_counts=expr, theta=theta)
        bin_count_totals = np.sum(expr, axis=1, dtype=np.float64)
    elif col_index is not None:
        if expression_loader is None:
            raise ValueError(
                f"artifact {locator.path} stores candidate_matrix_col_index but no matrix loader is available. "
                "Ensure episodes_index.csv comes from a run with config/config_resolved.yaml and reference format is npz."
            )
        ll, bin_count_totals = expression_loader.compute_ll_and_bin_counts_for_columns(
            col_index=col_index,
            log_theta=log_theta,
        )
    else:
        raise ValueError(
            f"artifact {locator.path} must contain either candidate_expression or candidate_matrix_col_index"
        )

    if ll.ndim != 2:
        raise ValueError(f"precomputed ll in {artifact_path} must have shape (B, K)")
    if candidate_bin_xy_um.shape != (ll.shape[0], 2):
        raise ValueError(f"candidate_bin_xy_um in {artifact_path} must have shape (B, 2)")
    if include_candidate_bin_ids and len(candidate_bin_ids) != ll.shape[0]:
        raise ValueError(f"candidate_bin_ids length mismatch in {artifact_path}")
    if nucleus_center_xy_um.shape != (2,):
        raise ValueError(f"nucleus_center_xy_um in {artifact_path} must have shape (2,)")
    if ll.shape[0] == 0 or ll.shape[1] == 0:
        return None

    d_other = _nearest_other_nucleus_distances(
        candidate_bin_xy_um=candidate_bin_xy_um,
        cell_id=cell_id,
        nuclei_spatial_index=nuclei_spatial_index,
    )

    return PreparedEpisodeArtifact(
        cell_id=cell_id,
        candidate_bin_ids=candidate_bin_ids,
        initial_membership_mask=np.zeros((ll.shape[0],), dtype=np.int8),
        candidate_bin_xy_um=candidate_bin_xy_um,
        nucleus_center_xy_um=nucleus_center_xy_um,
        bin_count_totals=np.asarray(bin_count_totals, dtype=np.float64),
        precomputed_ll=ll,
        precomputed_d_other_um=d_other,
        candidate_expression=None if candidate_expression is None else np.asarray(candidate_expression, dtype=np.float32),
        candidate_matrix_col_index=None if col_index is None else np.asarray(col_index, dtype=np.int64),
    )


def _parse_episode_artifact_locator(artifact_path: str | Path) -> EpisodeArtifactLocator:
    raw = str(artifact_path)
    if "::" in raw:
        path_str, member_str = raw.rsplit("::", 1)
        try:
            member_index = int(member_str)
        except ValueError as exc:
            raise ValueError(f"invalid artifact locator member index in {raw!r}") from exc
        return EpisodeArtifactLocator(
            path=Path(path_str).expanduser().resolve(),
            member_index=member_index,
        )
    return EpisodeArtifactLocator(
        path=Path(raw).expanduser().resolve(),
        member_index=None,
    )


def _load_legacy_episode_artifact_payload(
    *,
    artifact_path: Path,
    include_candidate_bin_ids: bool,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    with np.load(artifact_path, allow_pickle=True) as data:
        if include_candidate_bin_ids:
            candidate_bin_ids = tuple(str(x) for x in np.asarray(data["candidate_bin_ids"], dtype=object).tolist())
        else:
            candidate_bin_ids = tuple()
        candidate_bin_xy_um = np.asarray(data["candidate_bin_xy_um"], dtype=np.float64)
        nucleus_center_xy_um = np.asarray(data["nucleus_center_xy_um"], dtype=np.float64)
        candidate_expression = None
        col_index = None
        if "candidate_expression" in data:
            candidate_expression = np.asarray(data["candidate_expression"], dtype=np.float64)
        elif "candidate_matrix_col_index" in data:
            col_index = np.asarray(data["candidate_matrix_col_index"], dtype=np.int64)
    return candidate_bin_ids, candidate_bin_xy_um, nucleus_center_xy_um, candidate_expression, col_index


def _load_sharded_episode_artifact_payload(
    *,
    locator: EpisodeArtifactLocator,
    cell_id: str,
    include_candidate_bin_ids: bool,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    shard = _open_episode_artifact_shard(locator.path)
    member_index = int(locator.member_index)
    n_members = int(shard["nucleus_center_xy_um"].shape[0])
    if member_index < 0 or member_index >= n_members:
        raise IndexError(
            f"artifact locator member index {member_index} is outside shard range [0, {n_members}) for {locator.path}"
        )

    row_splits = np.asarray(shard["candidate_row_splits"], dtype=np.int64)
    start = int(row_splits[member_index])
    end = int(row_splits[member_index + 1])
    candidate_bin_xy_um = np.asarray(shard["candidate_bin_xy_um"][start:end], dtype=np.float64)
    nucleus_center_xy_um = np.asarray(shard["nucleus_center_xy_um"][member_index], dtype=np.float64)

    candidate_ids_arr = shard.get("candidate_bin_ids")
    if include_candidate_bin_ids and candidate_ids_arr is not None:
        candidate_bin_ids = tuple(str(x) for x in np.asarray(candidate_ids_arr[start:end]).tolist())
    elif include_candidate_bin_ids:
        candidate_bin_ids = tuple(f"{cell_id}::bin::{i}" for i in range(end - start))
    else:
        candidate_bin_ids = tuple()

    candidate_expression = None
    if "candidate_expression" in shard:
        candidate_expression = np.asarray(shard["candidate_expression"][start:end], dtype=np.float64)

    col_index = None
    if "candidate_matrix_col_index" in shard:
        col_index = np.asarray(shard["candidate_matrix_col_index"][start:end], dtype=np.int64)

    return candidate_bin_ids, candidate_bin_xy_um, nucleus_center_xy_um, candidate_expression, col_index


def _open_episode_artifact_shard(shard_path: Path) -> dict[str, np.ndarray]:
    key = str(shard_path)
    cached = _ARTIFACT_SHARD_CACHE.get(key)
    if cached is not None:
        _ARTIFACT_SHARD_CACHE.move_to_end(key)
        return cached

    if not shard_path.is_dir():
        raise FileNotFoundError(f"episode artifact shard directory not found: {shard_path}")

    required = {
        "nucleus_center_xy_um": "nucleus_center_xy_um.npy",
        "candidate_row_splits": "candidate_row_splits.npy",
        "candidate_bin_xy_um": "candidate_bin_xy_um.npy",
    }
    optional = {
        "candidate_bin_ids": "candidate_bin_ids.npy",
        "candidate_expression": "candidate_expression.npy",
        "candidate_matrix_col_index": "candidate_matrix_col_index.npy",
    }

    shard: dict[str, np.ndarray] = {}
    for name, filename in required.items():
        file_path = shard_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"artifact shard is missing required array: {file_path}")
        shard[name] = np.load(file_path, allow_pickle=False)

    for name, filename in optional.items():
        file_path = shard_path / filename
        if file_path.exists():
            shard[name] = np.load(file_path, allow_pickle=False)

    _ARTIFACT_SHARD_CACHE[key] = shard
    _ARTIFACT_SHARD_CACHE.move_to_end(key)
    while len(_ARTIFACT_SHARD_CACHE) > _ARTIFACT_SHARD_CACHE_SIZE:
        _ARTIFACT_SHARD_CACHE.popitem(last=False)
    return shard


class _MatrixOnDemandExpressionLoader:
    """Load selected-gene expression vectors for matrix column indices from 10x H5."""

    def __init__(
        self,
        matrix_path: Path,
        reference_npz_path: Path,
        reference_genes_key: str,
        cache_size: int = 20000,
    ) -> None:
        if not matrix_path.exists():
            raise FileNotFoundError(f"matrix source not found: {matrix_path}")
        if not reference_npz_path.exists():
            raise FileNotFoundError(f"reference NPZ file not found: {reference_npz_path}")
        if cache_size < 0:
            raise ValueError("cache_size must be >= 0")

        resolved_matrix_h5_path = resolve_matrix_csc_h5_path(matrix_path)
        self._h5 = h5py.File(resolved_matrix_h5_path, "r")
        if "matrix" not in self._h5:
            raise ValueError(f"H5 file does not contain 'matrix' group: {resolved_matrix_h5_path}")
        mg = self._h5["matrix"]
        for key in ("data", "indices", "indptr", "shape", "features"):
            if key not in mg:
                raise ValueError(f"H5 matrix group missing key: matrix/{key}")
        fg = mg["features"]
        if "name" not in fg:
            raise ValueError("H5 matrix/features missing 'name' dataset")

        shape = tuple(int(v) for v in mg["shape"][:].tolist())
        if len(shape) != 2:
            raise ValueError(f"matrix/shape in {resolved_matrix_h5_path} is invalid: {shape}")
        n_features, n_cols = int(shape[0]), int(shape[1])
        feature_names = np.asarray([x.decode("utf-8") for x in fg["name"][:]], dtype="U")
        selected_feature_indices = _resolve_matrix_feature_indices_from_reference(
            feature_names=feature_names,
            reference_npz_path=reference_npz_path,
            reference_genes_key=reference_genes_key,
        )
        if selected_feature_indices.size == 0:
            raise ValueError("selected zero genes for on-demand expression loader")
        if selected_feature_indices.max() >= n_features:
            raise ValueError("selected feature index exceeds matrix row count")

        self._n_cols = int(n_cols)
        self._expression_dim = int(selected_feature_indices.size)
        self._data_ds = mg["data"]
        self._indices_ds = mg["indices"]
        self._indptr = np.asarray(mg["indptr"][:], dtype=np.int64)
        if self._indptr.shape[0] != self._n_cols + 1:
            raise ValueError("matrix indptr length mismatch with matrix column count")

        feature_lookup = np.full(n_features, -1, dtype=np.int32)
        feature_lookup[selected_feature_indices] = np.arange(self._expression_dim, dtype=np.int32)
        self._feature_lookup = feature_lookup
        self._cache_size = int(cache_size)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._ll_cache_signature: tuple[int, tuple[int, ...]] | None = None
        self._ll_cache: OrderedDict[int, tuple[np.ndarray, float]] = OrderedDict()

    @property
    def expression_dim(self) -> int:
        return self._expression_dim

    def close(self) -> None:
        try:
            self._h5.close()
        except Exception:
            pass

    def load_columns(self, col_indices: np.ndarray) -> np.ndarray:
        col_indices = np.asarray(col_indices, dtype=np.int64)
        if col_indices.ndim != 1:
            raise ValueError("candidate_matrix_col_index must be a 1D array")
        if col_indices.size == 0:
            return np.zeros((0, self._expression_dim), dtype=np.float64)
        if (col_indices < 0).any():
            raise ValueError("candidate_matrix_col_index contains negative values")
        if (col_indices >= self._n_cols).any():
            bad = int(col_indices[col_indices >= self._n_cols][0])
            raise ValueError(
                f"candidate_matrix_col_index contains value {bad} outside matrix range [0, {self._n_cols})"
            )

        out = np.zeros((col_indices.size, self._expression_dim), dtype=np.float64)
        for i, col in enumerate(col_indices.tolist()):
            out[i] = self._load_one_column(int(col))
        return out

    def compute_ll_for_columns(self, col_index: np.ndarray, log_theta: np.ndarray) -> np.ndarray:
        """Compute LL[B, K] directly from sparse matrix columns without dense BxG arrays."""
        ll, _ = self.compute_ll_and_bin_counts_for_columns(col_index=col_index, log_theta=log_theta)
        return ll

    def compute_ll_and_bin_counts_for_columns(
        self,
        col_index: np.ndarray,
        log_theta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute LL[B, K] and total selected-gene counts per bin from sparse columns."""
        cols = np.asarray(col_index, dtype=np.int64)
        if cols.ndim != 1:
            raise ValueError("candidate_matrix_col_index must be a 1D array")
        if cols.size == 0:
            return (
                np.zeros((0, int(np.asarray(log_theta).shape[0])), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
            )
        if (cols < 0).any():
            raise ValueError("candidate_matrix_col_index contains negative values")
        if (cols >= self._n_cols).any():
            bad = int(cols[cols >= self._n_cols][0])
            raise ValueError(
                f"candidate_matrix_col_index contains value {bad} outside matrix range [0, {self._n_cols})"
            )

        lt = np.asarray(log_theta, dtype=np.float64)
        if lt.ndim != 2:
            raise ValueError("log_theta must have shape (K, G)")
        if lt.shape[1] != self._expression_dim:
            raise ValueError(
                "log_theta gene dimension mismatch: %d != %d" % (lt.shape[1], self._expression_dim)
            )
        ll_cache_signature = (id(lt), tuple(int(x) for x in lt.shape))
        if self._ll_cache_signature != ll_cache_signature:
            self._ll_cache.clear()
            self._ll_cache_signature = ll_cache_signature

        out = np.zeros((cols.size, lt.shape[0]), dtype=np.float64)
        bin_count_totals = np.zeros((cols.size,), dtype=np.float64)
        for i, col in enumerate(cols.tolist()):
            cached = self._ll_cache.get(int(col))
            if cached is not None:
                self._ll_cache.move_to_end(int(col))
                out[i, :] = cached[0]
                bin_count_totals[i] = float(cached[1])
                continue

            start = int(self._indptr[col])
            end = int(self._indptr[col + 1])
            ll_row = np.zeros((lt.shape[0],), dtype=np.float64)
            nb = 0.0
            if end > start:
                col_feature_idx = np.asarray(self._indices_ds[start:end], dtype=np.int64)
                col_values = np.asarray(self._data_ds[start:end], dtype=np.float64)
                selected_pos = self._feature_lookup[col_feature_idx]
                keep = selected_pos >= 0
                if np.any(keep):
                    pos = selected_pos[keep].astype(np.int64, copy=False)
                    vals = col_values[keep]
                    nb = float(np.sum(vals))
                    if nb > 0:
                        weighted = np.sum(lt[:, pos] * vals[None, :], axis=1)
                        ll_row = (weighted / nb).astype(np.float64, copy=False)
            out[i, :] = ll_row
            bin_count_totals[i] = nb
            if self._cache_size > 0:
                self._ll_cache[int(col)] = (ll_row.copy(), float(nb))
                self._ll_cache.move_to_end(int(col))
                while len(self._ll_cache) > self._cache_size:
                    self._ll_cache.popitem(last=False)

        return out, bin_count_totals

    def _load_one_column(self, col_index: int) -> np.ndarray:
        cached = self._cache.get(col_index)
        if cached is not None:
            self._cache.move_to_end(col_index)
            return cached

        start = int(self._indptr[col_index])
        end = int(self._indptr[col_index + 1])
        expr = np.zeros(self._expression_dim, dtype=np.float64)
        if end > start:
            col_feature_idx = np.asarray(self._indices_ds[start:end], dtype=np.int64)
            col_values = np.asarray(self._data_ds[start:end], dtype=np.float64)
            selected_pos = self._feature_lookup[col_feature_idx]
            keep = selected_pos >= 0
            if keep.any():
                expr[selected_pos[keep]] = col_values[keep]

        if self._cache_size > 0:
            self._cache[col_index] = expr
            self._cache.move_to_end(col_index)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return expr


def _resolve_matrix_feature_indices_from_reference(
    feature_names: np.ndarray,
    reference_npz_path: Path,
    reference_genes_key: str,
) -> np.ndarray:
    with np.load(reference_npz_path) as data:
        if reference_genes_key not in data:
            raise ConfigError(
                f"inputs.reference.genes_key {reference_genes_key!r} is not present in {reference_npz_path}"
            )
        ordered_genes = data[reference_genes_key].astype(str)

    first_index_by_name: dict[str, int] = {}
    for i, name in enumerate(feature_names.astype(str)):
        if name not in first_index_by_name:
            first_index_by_name[name] = i

    missing = [g for g in ordered_genes if g not in first_index_by_name]
    if missing:
        preview = missing[:5]
        raise ValueError(
            f"{len(missing)} reference genes are missing in matrix features; first missing: {preview}"
        )
    return np.asarray([first_index_by_name[g] for g in ordered_genes], dtype=np.int64)


def _load_episode_build_expression_context(
    episodes_index_path: Path,
) -> dict[str, Any] | None:
    run_dir = episodes_index_path.parent
    cfg_path = run_dir / "config" / "config_resolved.yaml"
    if not cfg_path.exists():
        return None
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        return None
    inputs = raw.get("inputs")
    if not isinstance(inputs, dict):
        return None
    expression = inputs.get("expression")
    if not isinstance(expression, dict):
        return None
    if str(expression.get("mode", "")).strip() != "matrix_h5":
        return None
    matrix_value = expression.get("matrix_path", expression.get("matrix_h5_path", None))
    if matrix_value is None:
        return None
    bins_value = inputs.get("bins_path")
    return {
        "matrix_path": Path(str(matrix_value)).expanduser().resolve(),
        "cache_size": int(expression.get("cache_size", 20000)),
        "bins_path": None if bins_value is None else Path(str(bins_value)).expanduser().resolve(),
    }


def _build_nuclei_centers(df: pd.DataFrame, columns: dict[str, str]) -> dict[str, np.ndarray]:
    required = [columns["cell_id"], columns["center_x_um"], columns["center_y_um"]]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"missing columns in nuclei table: {missing}")

    cell_ids = df[columns["cell_id"]].astype(str)
    if cell_ids.isna().any():
        raise ValueError("nuclei cell_id column contains missing values")
    if cell_ids.duplicated().any():
        dup_value = str(cell_ids.loc[cell_ids.duplicated()].iloc[0])
        raise ValueError(f"duplicate cell_id found in nuclei table: {dup_value!r}")

    center_x = pd.to_numeric(df[columns["center_x_um"]], errors="raise").to_numpy(dtype=np.float64)
    center_y = pd.to_numeric(df[columns["center_y_um"]], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(center_x).all() or not np.isfinite(center_y).all():
        raise ValueError("nuclei center coordinates must be finite")

    centers: dict[str, np.ndarray] = {}
    for idx, cell_id in enumerate(cell_ids.to_numpy()):
        centers[str(cell_id)] = np.asarray([center_x[idx], center_y[idx]], dtype=np.float64)
    return centers


def _default_nuclei_center_columns(overrides: dict[str, Any]) -> dict[str, str]:
    cols = {
        "cell_id": "cell_id",
        "center_x_um": "center_x_um",
        "center_y_um": "center_y_um",
    }
    cols.update(overrides)

    for key in ("cell_id", "center_x_um", "center_y_um"):
        if cols.get(key) is None:
            raise ConfigError(f"inputs.nuclei.columns.{key} must not be null")
        cols[key] = str(cols[key])

    return cols
