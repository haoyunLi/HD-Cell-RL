"""Episode dataset and artifact loading helpers for PPO/GRPO training."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd
from scipy import sparse

from .ppo_config import ConfigError, PPOTrainingConfig
from .nuclear_seeds import load_confident_nuclear_rows
from .ppo_state import EpisodeContext, _zscore_1d
from .shape_prior import load_shape_prior_model
from .reward import (
    build_eight_neighbor_index,
    compute_expression_confidence,
    compute_reference_distribution,
)
from .episode_artifacts import (
    _MatrixOnDemandExpressionLoader,
    _build_nuclei_centers,
    _build_nuclei_spatial_index,
    _load_episode_build_candidate_geometry_context,
    _load_episode_build_expression_context,
    _load_one_episode_artifact,
    _parse_episode_artifact_locator,
    _load_legacy_episode_artifact_payload,
    _load_sharded_episode_artifact_payload,
)


_CACHE_MISS = object()


class EpisodeDataset:
    """On-demand loader for episode artifacts, optimized for many-cell datasets."""

    def __init__(self, config: PPOTrainingConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng

        if not config.episodes_index_path.exists():
            raise FileNotFoundError(f"episodes index file not found: {config.episodes_index_path}")
        index_df = pd.read_csv(config.episodes_index_path)
        required = ["cell_id", "artifact_path"]
        missing = [col for col in required if col not in index_df.columns]
        if missing:
            raise ValueError(f"episodes index missing required columns: {missing}")
        self._index_df = index_df.reset_index(drop=True)

        self._reference_counts = _load_reference_counts(
            path=config.reference_path,
            reference_format=config.reference_format,
            array_key=config.reference_array_key,
        )
        self._theta = compute_reference_distribution(self._reference_counts, epsilon=config.epsilon)
        self._log_theta = np.log(self._theta)
        self._context_cache_size = int(config.episode_context_cache_size)
        self._context_cache: OrderedDict[tuple[str, str, int | None], EpisodeContext | None] = OrderedDict()
        self._shape_prior_model = (
            load_shape_prior_model(config.shape_prior_model_path)
            if bool(config.shape_prior_enabled) and config.shape_prior_model_path is not None
            else None
        )

        nuclei_df = _load_table(config.nuclei_path, config.nuclei_format)
        centers = _build_nuclei_centers(df=nuclei_df, columns=config.nuclei_columns)
        self._nuclei_spatial_index = _build_nuclei_spatial_index(centers)
        candidate_geometry = _load_episode_build_candidate_geometry_context(
            config.episodes_index_path
        )
        self._candidate_max_distance_um = (
            None
            if candidate_geometry is None
            else float(candidate_geometry["max_center_distance_um"])
        )
        self._candidate_radius_band_um = (
            None
            if candidate_geometry is None
            else candidate_geometry["radius_band_um"]
        )

        expression_ctx = _load_episode_build_expression_context(config.episodes_index_path)
        self._expression_loader: _MatrixOnDemandExpressionLoader | None = None
        self._nuclear_barcode_to_cell: dict[str, str] = {}
        if expression_ctx is not None and config.reference_format == "npz":
            cache_size = (
                int(config.expression_cache_size)
                if config.expression_cache_size is not None
                else int(expression_ctx["cache_size"])
            )
            self._expression_loader = _MatrixOnDemandExpressionLoader(
                matrix_path=Path(expression_ctx["matrix_path"]),
                reference_npz_path=config.reference_path,
                reference_genes_key=config.reference_genes_key,
                cache_size=cache_size,
            )
            bins_path = expression_ctx.get("bins_path")
            if bins_path is not None:
                self._nuclear_barcode_to_cell = _load_nuclear_barcode_assignment_lookup(Path(bins_path))

    @property
    def n_cells(self) -> int:
        return int(len(self._index_df))

    @property
    def candidate_max_distance_um(self) -> float | None:
        """Hard episode-build distance used to include/exclude candidate bins."""
        return self._candidate_max_distance_um

    @property
    def candidate_radius_band_um(self) -> float | None:
        """Optional secondary radial-band filter from episode construction."""
        return self._candidate_radius_band_um

    @property
    def reference_theta(self) -> np.ndarray:
        """Reference cell-type distributions aligned to loaded candidate expression."""
        return self._theta

    def candidate_cell_ids_for_expanded_bounds(
        self,
        *,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        expansion_um: float,
    ) -> tuple[str, ...]:
        """Return all nuclei in an axis-aligned patch box expanded by MaxDis."""
        expansion = float(expansion_um)
        if expansion < 0.0:
            raise ValueError("expansion_um must be >= 0")
        lower_x = float(x_min) - expansion
        upper_x = float(x_max) + expansion
        lower_y = float(y_min) - expansion
        upper_y = float(y_max) + expansion
        center = np.asarray(
            [(lower_x + upper_x) / 2.0, (lower_y + upper_y) / 2.0],
            dtype=np.float64,
        )
        radius = float(
            np.sqrt(
                ((upper_x - lower_x) / 2.0) ** 2
                + ((upper_y - lower_y) / 2.0) ** 2
            )
        )
        candidate_indices = np.asarray(
            self._nuclei_spatial_index.tree.query_ball_point(center, r=radius),
            dtype=np.int64,
        )
        if candidate_indices.size == 0:
            return ()
        xy = self._nuclei_spatial_index.centers_xy_um[candidate_indices]
        keep = (
            (xy[:, 0] >= lower_x)
            & (xy[:, 0] <= upper_x)
            & (xy[:, 1] >= lower_y)
            & (xy[:, 1] <= upper_y)
        )
        selected = candidate_indices[keep]
        return tuple(
            self._nuclei_spatial_index.cell_ids[int(idx)]
            for idx in selected.tolist()
        )

    def close(self) -> None:
        if self._expression_loader is not None:
            self._expression_loader.close()

    def sample_rows(self, n_rows: int) -> pd.DataFrame:
        """Sample rows from training set; with replacement if needed."""
        if n_rows <= 0:
            raise ValueError("n_rows must be > 0")
        if len(self._index_df) == 0:
            raise ValueError("episodes index is empty")

        replace = n_rows > len(self._index_df)
        sampled = self._rng.choice(len(self._index_df), size=n_rows, replace=replace)
        return self._index_df.iloc[np.asarray(sampled, dtype=np.int64)].reset_index(drop=True)

    def load_episode_context(
        self,
        cell_id: str,
        artifact_path: Path,
        max_steps_per_episode: int | None,
        *,
        include_candidate_bin_ids: bool = False,
    ) -> EpisodeContext | None:
        """Load one episode artifact and convert it into training-ready static context."""
        cache_key = (str(cell_id), str(Path(artifact_path).resolve()), max_steps_per_episode)
        cached = self._get_cached_context(cache_key)
        if cached is not _CACHE_MISS:
            return cached

        prepared = _load_one_episode_artifact(
            artifact_path=artifact_path,
            cell_id=cell_id,
            expression_loader=self._expression_loader,
            theta=self._theta,
            log_theta=self._log_theta,
            nuclei_spatial_index=self._nuclei_spatial_index,
            include_candidate_bin_ids=True,
        )
        if prepared is None:
            self._put_cached_context(cache_key, None)
            return None

        ll = np.asarray(prepared.precomputed_ll, dtype=np.float32)
        if ll.ndim != 2 or ll.shape[0] == 0 or ll.shape[1] == 0:
            self._put_cached_context(cache_key, None)
            return None

        bin_xy = np.asarray(prepared.candidate_bin_xy_um, dtype=np.float32)
        nucleus_center = np.asarray(prepared.nucleus_center_xy_um, dtype=np.float32)
        delta = bin_xy - nucleus_center[None, :]
        d_n = np.sqrt(np.sum(delta * delta, axis=1, dtype=np.float32))
        p_dis = (d_n / float(self._config.r_max_um)).astype(np.float32, copy=False)
        d_other = np.asarray(prepared.precomputed_d_other_um, dtype=np.float32)
        p_overlap = np.maximum(0.0, (d_n - d_other) / float(self._config.r_max_um)).astype(np.float32, copy=False)

        ll_mean = np.mean(ll, axis=1)
        ll_max = np.max(ll, axis=1)
        ll_mean_z = _zscore_1d(ll_mean).astype(np.float32, copy=False)
        ll_max_z = _zscore_1d(ll_max).astype(np.float32, copy=False)
        w2 = float(self._config.w2)
        w3 = float(self._config.w3)
        base_penalty = (w2 * p_dis + w3 * p_overlap).astype(np.float32, copy=False)
        expression_confidence = compute_expression_confidence(
            bin_count_totals=np.asarray(prepared.bin_count_totals, dtype=np.float64),
            pseudocount=float(self._config.expression_confidence_pseudocount),
        ).astype(np.float32, copy=False)
        neighbor_index = build_eight_neighbor_index(prepared.candidate_bin_ids, bin_xy).astype(np.int32, copy=False)
        initial_membership_mask = _build_initial_membership_mask(
            candidate_bin_ids=prepared.candidate_bin_ids,
            cell_id=str(prepared.cell_id),
            nuclear_barcode_to_cell=self._nuclear_barcode_to_cell,
        )

        max_steps = int(max_steps_per_episode) if max_steps_per_episode is not None else max(1, ll.shape[0] * 3)
        context = EpisodeContext(
            cell_id=str(prepared.cell_id),
            candidate_bin_ids=tuple(prepared.candidate_bin_ids),
            initial_membership_mask=initial_membership_mask,
            candidate_bin_xy_um=bin_xy.astype(np.float32, copy=False),
            nucleus_center_xy_um=nucleus_center.astype(np.float32, copy=False),
            ll=ll,
            p_dis=p_dis,
            p_overlap=p_overlap,
            ll_mean_z=ll_mean_z,
            ll_max_z=ll_max_z,
            base_penalty=base_penalty,
            expression_confidence=expression_confidence,
            bin_count_totals=np.asarray(prepared.bin_count_totals, dtype=np.float32),
            neighbor_index=neighbor_index,
            max_steps=max_steps,
            log_prior=-np.log(float(ll.shape[1])),
            r_max_um=float(self._config.r_max_um),
            w1=float(self._config.w1),
            w2=w2,
            w3=w3,
            w4=float(self._config.w4),
            w5=float(self._config.w5),
            stop_lambda=float(self._config.stop_lambda),
            stop_stat=str(self._config.stop_stat),
            stop_top_k=int(self._config.stop_top_k),
            expression_confidence_pseudocount=float(self._config.expression_confidence_pseudocount),
            normalize_expression_zscore=bool(self._config.normalize_expression_zscore),
            zscore_delta=float(self._config.zscore_delta),
            competition_margin_weight=float(self._config.competition_margin_weight),
            competition_margin_radius_um=float(self._config.competition_margin_radius_um),
            competition_margin_clip=float(self._config.competition_margin_clip),
            competition_margin_affects_stop=bool(self._config.competition_margin_affects_stop),
            shape_prior_model=self._shape_prior_model,
            shape_prior_weight=float(self._config.shape_prior_weight) if self._shape_prior_model is not None else 0.0,
            shape_prior_mode=str(self._config.shape_prior_mode),
            shape_prior_reward_mode=str(self._config.shape_prior_reward_mode),
            shape_prior_normalize_over_frontier=bool(self._config.shape_prior_normalize_over_frontier),
            shape_prior_clip=self._config.shape_prior_clip,
            shape_prior_bin_size_um=float(self._config.shape_prior_bin_size_um),
        )
        self._put_cached_context(cache_key, context)
        return context

    def load_episode_candidate_expression(
        self,
        *,
        cell_id: str,
        artifact_path: Path,
    ) -> tuple[tuple[str, ...], np.ndarray] | None:
        """Load selected-gene candidate expression aligned to one episode artifact."""
        prepared = _load_one_episode_artifact(
            artifact_path=artifact_path,
            cell_id=cell_id,
            expression_loader=self._expression_loader,
            theta=self._theta,
            log_theta=self._log_theta,
            nuclei_spatial_index=self._nuclei_spatial_index,
            include_candidate_bin_ids=True,
        )
        if prepared is None:
            return None
        if prepared.candidate_expression is not None:
            expression = np.asarray(prepared.candidate_expression, dtype=np.float32)
        elif prepared.candidate_matrix_col_index is not None:
            if self._expression_loader is None:
                raise ValueError(
                    f"artifact {artifact_path} stores candidate_matrix_col_index but no matrix loader is available"
                )
            expression = self._expression_loader.load_columns(prepared.candidate_matrix_col_index).astype(
                np.float32,
                copy=False,
            )
        else:
            raise ValueError(f"artifact {artifact_path} has no candidate expression payload")
        return tuple(prepared.candidate_bin_ids), expression

    def load_episode_sparse_expression(self, *, cell_id: str, artifact_path: Path,
                                       wanted_barcodes: set[str]) -> tuple[tuple[str, ...], sparse.csr_matrix]:
        """Read only requested physical bins; do not recompute LL or densify H5 counts."""
        locator = _parse_episode_artifact_locator(artifact_path)
        if locator.member_index is None:
            payload = _load_legacy_episode_artifact_payload(artifact_path=locator.path, include_candidate_bin_ids=True)
        else:
            payload = _load_sharded_episode_artifact_payload(locator=locator, cell_id=cell_id, include_candidate_bin_ids=True)
        barcodes, _, _, expression, columns = payload
        selected = np.asarray([i for i, barcode in enumerate(barcodes) if barcode in wanted_barcodes], dtype=np.int64)
        ids = tuple(barcodes[int(i)] for i in selected)
        if expression is not None:
            counts = sparse.csr_matrix(expression[selected], dtype=np.float64)
        elif columns is not None and self._expression_loader is not None:
            counts = self._expression_loader.load_sparse_columns(columns[selected])
        else:
            raise ValueError(f"artifact {artifact_path} has no usable candidate expression payload")
        return ids, counts

    def _get_cached_context(self, key: tuple[str, str, int | None]) -> EpisodeContext | None | object:
        if self._context_cache_size <= 0:
            return _CACHE_MISS
        try:
            value = self._context_cache.pop(key)
        except KeyError:
            return _CACHE_MISS
        self._context_cache[key] = value
        return value

    def _put_cached_context(self, key: tuple[str, str, int | None], value: EpisodeContext | None) -> None:
        if self._context_cache_size <= 0:
            return
        self._context_cache[key] = value
        while len(self._context_cache) > self._context_cache_size:
            self._context_cache.popitem(last=False)


def _normalize_cell_id(value: Any) -> str | None:
    """Normalize mixed int/float/string cell IDs into one stable string form."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def _load_nuclear_barcode_assignment_lookup(bins_path: Path) -> dict[str, str]:
    """Load confident nuclear barcode -> cell_id assignments from merged bins metadata."""
    if not bins_path.exists():
        raise FileNotFoundError(f"episode-build bins metadata not found: {bins_path}")

    df = load_confident_nuclear_rows(bins_path)
    if len(df) == 0:
        return {}

    out = df.loc[:, ["barcode", "dominant_cell_id"]].copy()
    if out.empty:
        return {}

    out["cell_id"] = out["dominant_cell_id"].map(_normalize_cell_id)
    out = out.loc[out["cell_id"].notna()].copy()
    if out.empty:
        return {}

    return {
        str(barcode): str(cell_id)
        for barcode, cell_id in zip(out["barcode"].astype(str), out["cell_id"].astype(str), strict=False)
    }


def _build_initial_membership_mask(
    *,
    candidate_bin_ids: tuple[str, ...],
    cell_id: str,
    nuclear_barcode_to_cell: dict[str, str],
) -> np.ndarray:
    """Seed episode state with all confident nuclear bins for this cell."""
    if not candidate_bin_ids:
        return np.zeros((0,), dtype=np.uint8)
    normalized_cell_id = _normalize_cell_id(cell_id)
    return np.asarray(
        [
            1 if normalized_cell_id is not None and nuclear_barcode_to_cell.get(str(bin_id)) == normalized_cell_id else 0
            for bin_id in candidate_bin_ids
        ],
        dtype=np.uint8,
    )


def _load_reference_counts(path: Path, reference_format: str, array_key: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"reference file not found: {path}")
    if reference_format == "npy":
        arr = np.load(path)
        return _validate_reference_matrix(arr)
    if reference_format == "npz":
        with np.load(path) as data:
            if array_key not in data:
                raise ConfigError(f"reference array key {array_key!r} is not present in {path}")
            arr = data[array_key]
        return _validate_reference_matrix(arr)

    table = _load_table(path, reference_format)
    numeric_cols = [c for c in table.columns if pd.api.types.is_numeric_dtype(table[c])]
    if not numeric_cols:
        raise ValueError(f"reference table {path} has no numeric columns to form C[K,G]")
    arr = table.loc[:, numeric_cols].to_numpy(dtype=np.float64, copy=True)
    return _validate_reference_matrix(arr)


def _validate_reference_matrix(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("reference matrix must have shape (K, G)")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("reference matrix must have positive shape in both dimensions")
    if not np.isfinite(arr).all():
        raise ValueError("reference matrix contains non-finite values")
    if (arr < 0).any():
        raise ValueError("reference matrix must be non-negative")
    return arr


def _load_table(path: Path, table_format: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"table file not found: {path}")
    if table_format == "parquet":
        return pd.read_parquet(path)
    if table_format == "csv":
        return pd.read_csv(path)
    if table_format == "tsv":
        return pd.read_csv(path, sep="\t")
    raise ConfigError(f"unsupported table format for loader: {table_format!r}")
