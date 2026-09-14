"""Patch index loading and PatchContext construction."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
from scipy import sparse

from .em_assignment import build_sparse_patch_em_input, initialize_patch_context_from_em
from .em_types import EMAssignmentConfig
from .patch_types import PatchBounds, PatchContext, PatchTrainingSettings
from .ppo_config import PPOTrainingConfig
from .ppo_dataset import EpisodeDataset
from .ppo_state import EpisodeContext
from .stcs_reward import StcsCellPayload, StcsRewardConfig, compute_stcs_reward_scores


class PatchDataset:
    """Load patch rows and convert their cells into EpisodeContext objects."""

    def __init__(
        self,
        *,
        base_config: PPOTrainingConfig,
        settings: PatchTrainingSettings,
        rng: np.random.Generator,
    ) -> None:
        if not settings.patches_index_path.exists():
            raise FileNotFoundError(f"patches index not found: {settings.patches_index_path}")
        self._base_config = base_config
        self._settings = settings
        self._rng = rng
        self._patches_df = pd.read_csv(settings.patches_index_path)
        if self._patches_df.empty:
            raise ValueError(f"patches index is empty: {settings.patches_index_path}")
        self._cell_dataset = EpisodeDataset(config=base_config, rng=rng)
        self._artifact_by_cell = _load_episode_artifact_map(base_config.episodes_index_path)
        self._context_cache: dict[str, PatchContext | None] = {}

    @property
    def n_patches(self) -> int:
        return int(len(self._patches_df))

    def close(self) -> None:
        self._cell_dataset.close()

    @property
    def reference_theta(self) -> np.ndarray:
        return self._cell_dataset.reference_theta

    def load_cell_candidate_expression(
        self,
        cell_id: str,
    ) -> tuple[tuple[str, ...], np.ndarray] | None:
        """Load selected-gene candidate expression for one physical cell."""
        artifact_path = self._artifact_by_cell.get(str(cell_id))
        if artifact_path is None:
            raise KeyError(f"no episode artifact for cell {cell_id!r}")
        return self._cell_dataset.load_episode_candidate_expression(
            cell_id=str(cell_id),
            artifact_path=artifact_path,
        )

    def load_unique_patch_expression(
        self,
        *,
        context: PatchContext,
        barcode_ids: tuple[str, ...],
    ) -> sparse.csr_matrix:
        """Load selected-gene counts once per stable physical barcode.

        Episode artifacts repeat a physical barcode for every candidate cell.
        This method mirrors the patch environment's unique-owner representation
        and aligns the sparse matrix exactly to ``barcode_ids``.
        """
        target_index = {
            str(barcode): idx for idx, barcode in enumerate(barcode_ids)
        }
        if len(target_index) != len(barcode_ids):
            raise ValueError("EM barcode_ids must be unique")
        remaining = set(target_index)
        rows: list[sparse.csr_matrix | None] = [None] * len(barcode_ids)
        n_genes: int | None = None
        for cell in context.cells:
            artifact_path = self._artifact_by_cell.get(str(cell.cell_id))
            if artifact_path is None:
                raise KeyError(f"no episode artifact for cell {cell.cell_id!r}")
            loaded = self._cell_dataset.load_episode_sparse_expression(
                cell_id=str(cell.cell_id), artifact_path=artifact_path, wanted_barcodes=remaining)
            loaded_ids, expression = loaded
            values = sparse.csr_matrix(expression, dtype=np.float64)
            if values.ndim != 2 or values.shape[0] != len(loaded_ids):
                raise ValueError(
                    f"candidate expression shape mismatch for cell {cell.cell_id!r}"
                )
            if n_genes is None:
                n_genes = int(values.shape[1])
            elif int(values.shape[1]) != n_genes:
                raise ValueError(
                    "candidate expression gene dimensions differ across cells"
                )
            for local_idx, raw_barcode in enumerate(loaded_ids):
                barcode = str(raw_barcode)
                if barcode not in remaining:
                    continue
                rows[target_index[barcode]] = sparse.csr_matrix(
                    values[local_idx : local_idx + 1]
                )
                remaining.remove(barcode)
            if not remaining:
                break
        if remaining:
            raise ValueError(
                f"missing expression for {len(remaining)} EM bins in patch "
                f"{context.patch_id!r} (first: {sorted(remaining)[:5]})"
            )
        if n_genes is None or any(row is None for row in rows):
            raise ValueError(
                f"no complete candidate expression was loaded for patch {context.patch_id!r}"
            )
        return sparse.vstack(rows, format="csr", dtype=np.float64)

    def sample_rows(self, n_rows: int) -> pd.DataFrame:
        if n_rows <= 0:
            raise ValueError("n_rows must be > 0")
        replace = int(n_rows) > len(self._patches_df)
        idx = self._rng.choice(len(self._patches_df), size=int(n_rows), replace=replace)
        return self._patches_df.iloc[np.asarray(idx, dtype=np.int64)].reset_index(drop=True)

    def load_patch_context(self, row: Any) -> PatchContext | None:
        patch_id = str(getattr(row, "patch_id"))
        if self._settings.cache_patch_contexts and patch_id in self._context_cache:
            return self._context_cache[patch_id]

        outer_bounds = PatchBounds(
            x_min=float(getattr(row, "outer_x_min")),
            x_max=float(getattr(row, "outer_x_max")),
            y_min=float(getattr(row, "outer_y_min")),
            y_max=float(getattr(row, "outer_y_max")),
        )
        em_config = getattr(self._settings, "em_assignment", EMAssignmentConfig())
        candidate_max_distance_um: float | None = None
        patch_cell_ids = _json_cell_list(getattr(row, "patch_cell_ids"))
        core_cell_ids = tuple(_json_cell_list(getattr(row, "core_cell_ids")))
        margin_cell_ids = tuple(_json_cell_list(getattr(row, "margin_cell_ids")))
        if em_config.enabled:
            if not self._settings.margin_cells_compete:
                raise ValueError(
                    "em_assignment.enabled requires patch_training.margin_cells_compete=true"
                )
            if self._cell_dataset.candidate_radius_band_um is not None:
                raise ValueError(
                    "EM requires complete distance-only episode candidates, but the "
                    "episode-build config has environment.radius_band_um set. Rebuild "
                    "EM-ready episodes with radius_band_um=null; the EM radius still "
                    "comes from environment.max_center_distance_um."
                )
            candidate_max_distance_um = self._resolve_candidate_max_distance(row)
            complete_ids = self._cell_dataset.candidate_cell_ids_for_expanded_bounds(
                x_min=outer_bounds.x_min,
                x_max=outer_bounds.x_max,
                y_min=outer_bounds.y_min,
                y_max=outer_bounds.y_max,
                expansion_um=candidate_max_distance_um,
            )
            missing_artifacts = sorted(
                cell_id
                for cell_id in complete_ids
                if str(cell_id) not in self._artifact_by_cell
            )
            if missing_artifacts:
                raise ValueError(
                    "patch candidate context is incomplete: "
                    f"{len(missing_artifacts)} nuclei within outer_bounds + MaxDis have no "
                    "episode artifact (first: "
                    f"{missing_artifacts[:5]}). Rebuild episodes with "
                    "inputs.expression.filter_empty_nuclear_cells=false before enabling EM."
                )
            existing = set(patch_cell_ids)
            patch_cell_ids.extend(
                cell_id for cell_id in sorted(complete_ids) if cell_id not in existing
            )
        if not self._settings.margin_cells_compete:
            patch_cell_ids = list(core_cell_ids)

        cells: list[EpisodeContext] = []
        loaded_cell_ids: set[str] = set()
        for cell_id in patch_cell_ids:
            artifact_path = self._artifact_by_cell.get(str(cell_id))
            if artifact_path is None:
                continue
            ctx = self._cell_dataset.load_episode_context(
                cell_id=str(cell_id),
                artifact_path=artifact_path,
                max_steps_per_episode=self._base_config.max_steps_per_episode,
                include_candidate_bin_ids=True,
            )
            if ctx is None or ctx.n_bins <= 0:
                continue
            if not em_config.enabled and int(np.sum(ctx.initial_membership_mask)) <= 0:
                continue
            cells.append(ctx)
            loaded_cell_ids.add(str(cell_id))

        loaded_core = tuple(cell_id for cell_id in core_cell_ids if cell_id in loaded_cell_ids)
        if not cells or not loaded_core:
            if self._settings.cache_patch_contexts:
                self._context_cache[patch_id] = None
            return None

        reward_backend = str(getattr(self._settings, "reward_backend", "standard")).strip().lower()
        if em_config.enabled and reward_backend != "standard":
            raise ValueError(
                "generalized EM v1 supports the standard LL reward backend only"
            )
        if reward_backend == "stcs":
            cells = self._attach_stcs_reward_scores(cells)
        elif reward_backend != "standard":
            raise ValueError("patch_training.reward_backend must be one of: standard, stcs")

        force_fill_enabled = bool(getattr(self._settings, "force_fill_expression_bins", False))
        score_normalization = str(self._settings.score_normalization)
        should_build_expression_target = force_fill_enabled or score_normalization in {
            "mean_expression_bins",
            "sqrt_expression_bins",
        }
        fill_target = str(getattr(self._settings, "fill_target", "reachable_expression_bins"))
        force_fill_target_barcodes = (
            _reachable_expression_barcodes(cells=cells, outer_bounds=outer_bounds)
            if should_build_expression_target and fill_target == "reachable_expression_bins"
            else ()
        )

        loaded_margin = (
            tuple(
                str(ctx.cell_id)
                for ctx in cells
                if str(ctx.cell_id) not in set(loaded_core)
            )
            if em_config.enabled
            else tuple(cell_id for cell_id in margin_cell_ids if cell_id in loaded_cell_ids)
        )
        context = PatchContext(
            patch_id=patch_id,
            cells=tuple(cells),
            core_cell_ids=loaded_core,
            margin_cell_ids=loaded_margin,
            outer_bounds=outer_bounds,
            core_bounds=PatchBounds(
                x_min=float(getattr(row, "core_x_min")),
                x_max=float(getattr(row, "core_x_max")),
                y_min=float(getattr(row, "core_y_min")),
                y_max=float(getattr(row, "core_y_max")),
            ),
            max_steps=int(self._settings.max_steps_per_patch),
            score_normalization=score_normalization,
            reward_backend=reward_backend,
            competition_margin_enabled=bool(getattr(self._settings, "competition_margin_enabled", True)),
            force_fill_expression_bins=force_fill_enabled,
            fill_target=fill_target,
            stop_action_mode=str(getattr(self._settings, "stop_action_mode", "enabled")),
            force_fill_target_barcodes=force_fill_target_barcodes,
            agent_mode=str(getattr(self._settings, "agent_mode", "multi_cell")),
            after_fill_actions=str(getattr(self._settings, "after_fill_actions", "add_or_stop")),
            global_delta_epsilon=float(getattr(self._settings, "global_delta_epsilon", 1.0e-6)),
            candidate_max_distance_um=candidate_max_distance_um,
        )
        if em_config.enabled:
            em_expression: sparse.csr_matrix | None = None
            reference_theta: np.ndarray | None = None
            if em_config.cell_specific_expression_enabled:
                preliminary_em_input = build_sparse_patch_em_input(
                    context=context,
                    candidate_max_distance_um=float(candidate_max_distance_um),
                    non_nuclear_bin_filter=em_config.non_nuclear_bin_filter,
                )
                em_expression = self.load_unique_patch_expression(
                    context=context,
                    barcode_ids=preliminary_em_input.barcode_ids,
                )
                reference_theta = np.asarray(self.reference_theta, dtype=np.float64)
            context = initialize_patch_context_from_em(
                context=context,
                config=em_config,
                candidate_max_distance_um=float(candidate_max_distance_um),
                bin_gene_counts=em_expression,
                reference_theta=reference_theta,
            )
        if self._settings.cache_patch_contexts:
            self._context_cache[patch_id] = context
        return context

    def _resolve_candidate_max_distance(self, row: Any) -> float:
        """Resolve MaxDis from episode construction, never from reward.r_max_um."""
        episode_value = self._cell_dataset.candidate_max_distance_um
        row_value_raw = getattr(row, "candidate_max_distance_um", None)
        row_value = (
            None
            if row_value_raw is None or bool(pd.isna(row_value_raw))
            else float(row_value_raw)
        )
        if episode_value is None:
            if row_value is None:
                raise ValueError(
                    "em_assignment requires environment.max_center_distance_um from the "
                    "episode-build config/config_resolved.yaml"
                )
            episode_value = row_value
        if row_value is not None and not np.isclose(
            float(row_value),
            float(episode_value),
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError(
                "patch index candidate_max_distance_um does not match episode-build "
                "environment.max_center_distance_um"
            )
        if float(episode_value) <= 0.0:
            raise ValueError("episode candidate max distance must be > 0")
        return float(episode_value)

    def _attach_stcs_reward_scores(self, cells: list[EpisodeContext]) -> list[EpisodeContext]:
        payloads: list[StcsCellPayload] = []
        for ctx in cells:
            artifact_path = self._artifact_by_cell.get(str(ctx.cell_id))
            if artifact_path is None:
                raise ValueError(f"missing episode artifact for STCS reward cell {ctx.cell_id!r}")
            loaded = self._cell_dataset.load_episode_candidate_expression(
                cell_id=str(ctx.cell_id),
                artifact_path=artifact_path,
            )
            if loaded is None:
                raise ValueError(f"failed to load STCS expression payload for cell {ctx.cell_id!r}")
            loaded_bin_ids, expression = loaded
            aligned_expression = _align_expression_to_context_bins(
                loaded_bin_ids=loaded_bin_ids,
                expression=expression,
                target_bin_ids=ctx.candidate_bin_ids,
                cell_id=str(ctx.cell_id),
            )
            payloads.append(
                StcsCellPayload(
                    cell_id=str(ctx.cell_id),
                    candidate_bin_ids=ctx.candidate_bin_ids,
                    candidate_expression=aligned_expression,
                    candidate_bin_xy_um=np.asarray(ctx.candidate_bin_xy_um, dtype=np.float32),
                    nucleus_center_xy_um=np.asarray(ctx.nucleus_center_xy_um, dtype=np.float32),
                    initial_membership_mask=np.asarray(ctx.initial_membership_mask, dtype=np.uint8),
                )
            )

        stcs_config = StcsRewardConfig.from_mapping(getattr(self._settings, "stcs_reward_config", None))
        scores = compute_stcs_reward_scores(payloads, stcs_config)
        return [
            replace(ctx, stcs_reward_scores=np.asarray(score, dtype=np.float32))
            for ctx, score in zip(cells, scores, strict=True)
        ]

def _load_episode_artifact_map(episodes_index_path: Path) -> dict[str, Path]:
    df = pd.read_csv(episodes_index_path, usecols=["cell_id", "artifact_path"])
    out: dict[str, Path] = {}
    for row in df.itertuples(index=False):
        cell_id = str(getattr(row, "cell_id"))
        if cell_id not in out:
            out[cell_id] = Path(str(getattr(row, "artifact_path"))).expanduser().resolve()
    return out


def _json_cell_list(value: Any) -> list[str]:
    if isinstance(value, str):
        raw = json.loads(value)
    else:
        raw = value
    return [str(x) for x in raw]


def _align_expression_to_context_bins(
    *,
    loaded_bin_ids: tuple[str, ...],
    expression: np.ndarray,
    target_bin_ids: tuple[str, ...],
    cell_id: str,
) -> np.ndarray:
    expr = np.asarray(expression, dtype=np.float32)
    if tuple(loaded_bin_ids) == tuple(target_bin_ids):
        return expr
    index_by_bin = {str(bin_id): idx for idx, bin_id in enumerate(loaded_bin_ids)}
    try:
        order = [index_by_bin[str(bin_id)] for bin_id in target_bin_ids]
    except KeyError as exc:
        raise ValueError(f"STCS expression bins do not match EpisodeContext bins for cell {cell_id!r}") from exc
    return expr[np.asarray(order, dtype=np.int64)]


def _reachable_expression_barcodes(*, cells: list[EpisodeContext], outer_bounds: PatchBounds) -> tuple[str, ...]:
    out: set[str] = set()
    for ctx in cells:
        outer = np.asarray(outer_bounds.contains_xy(ctx.candidate_bin_xy_um), dtype=bool)
        reachable = _reachable_mask_from_seed(
            seed_mask=np.asarray(ctx.initial_membership_mask, dtype=np.uint8) > 0,
            neighbor_index=np.asarray(ctx.neighbor_index, dtype=np.int64),
            allowed_mask=outer,
        )
        expressed = np.asarray(ctx.bin_count_totals, dtype=np.float32) > 0.0
        for bin_idx in np.flatnonzero(reachable & expressed & outer).tolist():
            out.add(str(ctx.candidate_bin_ids[int(bin_idx)]))
    return tuple(sorted(out))


def _reachable_mask_from_seed(*, seed_mask: np.ndarray, neighbor_index: np.ndarray, allowed_mask: np.ndarray) -> np.ndarray:
    n_bins = int(seed_mask.shape[0])
    reachable = np.zeros((n_bins,), dtype=bool)
    stack = [int(idx) for idx in np.flatnonzero(seed_mask & allowed_mask).tolist()]
    for idx in stack:
        reachable[int(idx)] = True
    while stack:
        current = int(stack.pop())
        for raw_neighbor in np.asarray(neighbor_index[current], dtype=np.int64).tolist():
            neighbor = int(raw_neighbor)
            if neighbor < 0 or neighbor >= n_bins:
                continue
            if reachable[neighbor] or not bool(allowed_mask[neighbor]):
                continue
            reachable[neighbor] = True
            stack.append(neighbor)
    return reachable
