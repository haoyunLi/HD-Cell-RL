"""Configuration and sparse result types for patch-level generalized EM."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EMAssignmentConfig:
    """Resolved generalized-EM settings used before patch-level RL."""

    enabled: bool = False
    use_existing_rl_candidate_max_distance: bool = True
    spatial_weight: float = 1.0
    expression_weight: float = 0.25
    damping: float = 0.5
    max_iterations: int = 15
    top_owner_change_fraction: float = 0.01
    mean_probability_change: float = 0.001
    max_q_change: float = 0.001
    mean_profile_total_variation: float = 0.001
    entropy_threshold: float = 0.5
    non_nuclear_bin_filter: str = "all"
    background_enabled: bool = False
    background_owned_logit_intercept: float = -6.0
    background_confidence_weight: float = 12.0
    cell_specific_expression_enabled: bool = False
    cell_profile_relative_prior_strength: float = 1.0
    cell_profile_update_damping: float = 1.0
    cell_profile_compatibility_mode: str = "standard"
    spatial_crossfit_block_size_um: float = 8.0
    cell_profile_reliability_mode: str = "nuclear_depth"
    heldout_reliability_grid_size: int = 101
    initialize_from_em: bool = True
    lock_nuclear_bins: bool = True
    refine_ambiguous_only: bool = True
    epsilon: float = 1.0e-12
    save_artifacts: bool = False
    debug_patch_ids: tuple[str, ...] = ()
    artifact_dir: Path | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> "EMAssignmentConfig":
        """Parse the nested ``em_assignment`` YAML section."""
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("em_assignment must be a mapping")
        convergence = value.get("convergence", {})
        ambiguity = value.get("ambiguity", {})
        rl_integration = value.get("rl_integration", {})
        artifacts = value.get("artifacts", {})
        bin_universe = value.get("bin_universe", {})
        background = value.get("background", {})
        cell_specific_expression = value.get("cell_specific_expression", {})
        for name, section in (
            ("em_assignment.convergence", convergence),
            ("em_assignment.ambiguity", ambiguity),
            ("em_assignment.rl_integration", rl_integration),
            ("em_assignment.artifacts", artifacts),
            ("em_assignment.bin_universe", bin_universe),
            ("em_assignment.background", background),
            ("em_assignment.cell_specific_expression", cell_specific_expression),
        ):
            if not isinstance(section, dict):
                raise ValueError(f"{name} must be a mapping")

        debug_patch_ids_raw = artifacts.get("debug_patch_ids", [])
        if debug_patch_ids_raw is None:
            debug_patch_ids_raw = []
        if not isinstance(debug_patch_ids_raw, (list, tuple)):
            raise ValueError("em_assignment.artifacts.debug_patch_ids must be a list")
        artifact_dir_raw = artifacts.get("directory")
        config = cls(
            enabled=bool(value.get("enabled", False)),
            use_existing_rl_candidate_max_distance=bool(
                value.get("use_existing_rl_candidate_max_distance", True)
            ),
            spatial_weight=float(value.get("spatial_weight", 1.0)),
            expression_weight=float(value.get("expression_weight", 0.25)),
            damping=float(value.get("damping", 0.5)),
            max_iterations=int(value.get("max_iterations", 15)),
            top_owner_change_fraction=float(
                convergence.get("top_owner_change_fraction", 0.01)
            ),
            mean_probability_change=float(
                convergence.get("mean_probability_change", 0.001)
            ),
            max_q_change=float(convergence.get("max_q_change", 0.001)),
            mean_profile_total_variation=float(
                convergence.get("mean_profile_total_variation", 0.001)
            ),
            entropy_threshold=float(ambiguity.get("entropy_threshold", 0.5)),
            non_nuclear_bin_filter=str(
                bin_universe.get("non_nuclear_filter", "all")
            ).strip().lower(),
            background_enabled=bool(background.get("enabled", False)),
            background_owned_logit_intercept=float(
                background.get("owned_logit_intercept", -6.0)
            ),
            background_confidence_weight=float(
                background.get("expression_confidence_weight", 12.0)
            ),
            cell_specific_expression_enabled=bool(
                cell_specific_expression.get("enabled", False)
            ),
            cell_profile_relative_prior_strength=float(
                cell_specific_expression.get("relative_prior_strength", 1.0)
            ),
            cell_profile_update_damping=float(
                cell_specific_expression.get("profile_update_damping", 1.0)
            ),
            cell_profile_compatibility_mode=str(
                cell_specific_expression.get("compatibility_mode", "standard")
            )
            .strip()
            .lower(),
            spatial_crossfit_block_size_um=float(
                cell_specific_expression.get("spatial_crossfit_block_size_um", 8.0)
            ),
            cell_profile_reliability_mode=str(
                cell_specific_expression.get("reliability_mode", "nuclear_depth")
            )
            .strip()
            .lower(),
            heldout_reliability_grid_size=int(
                cell_specific_expression.get("heldout_reliability_grid_size", 101)
            ),
            initialize_from_em=bool(rl_integration.get("initialize_from_em", True)),
            lock_nuclear_bins=bool(rl_integration.get("lock_nuclear_bins", True)),
            refine_ambiguous_only=bool(
                rl_integration.get("refine_ambiguous_only", True)
            ),
            epsilon=float(value.get("epsilon", 1.0e-12)),
            save_artifacts=bool(artifacts.get("save", False)),
            debug_patch_ids=tuple(str(item) for item in debug_patch_ids_raw),
            artifact_dir=(
                None
                if artifact_dir_raw in (None, "")
                else Path(str(artifact_dir_raw)).expanduser().resolve()
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not np.isfinite(self.max_q_change) or self.max_q_change < 0.0:
            raise ValueError("em_assignment.convergence.max_q_change must be finite and >= 0")
        if self.enabled and not self.use_existing_rl_candidate_max_distance:
            raise ValueError(
                "generalized EM v1 requires use_existing_rl_candidate_max_distance=true"
            )
        if self.spatial_weight < 0.0:
            raise ValueError("em_assignment.spatial_weight must be >= 0")
        if self.expression_weight < 0.0:
            raise ValueError("em_assignment.expression_weight must be >= 0")
        if not (0.0 < self.damping <= 1.0):
            raise ValueError("em_assignment.damping must be in (0, 1]")
        if self.max_iterations <= 0:
            raise ValueError("em_assignment.max_iterations must be > 0")
        if not (0.0 <= self.top_owner_change_fraction <= 1.0):
            raise ValueError(
                "em_assignment.convergence.top_owner_change_fraction must be in [0, 1]"
            )
        if self.mean_probability_change < 0.0:
            raise ValueError(
                "em_assignment.convergence.mean_probability_change must be >= 0"
            )
        if self.mean_profile_total_variation < 0.0:
            raise ValueError(
                "em_assignment.convergence.mean_profile_total_variation must be >= 0"
            )
        if not (0.0 <= self.entropy_threshold <= 1.0):
            raise ValueError("em_assignment.ambiguity.entropy_threshold must be in [0, 1]")
        if self.non_nuclear_bin_filter not in {"all", "positive_expression"}:
            raise ValueError(
                "em_assignment.bin_universe.non_nuclear_filter must be one of: "
                "all, positive_expression"
            )
        if not np.isfinite(self.background_owned_logit_intercept):
            raise ValueError(
                "em_assignment.background.owned_logit_intercept must be finite"
            )
        if not np.isfinite(self.background_confidence_weight):
            raise ValueError(
                "em_assignment.background.expression_confidence_weight must be finite"
            )
        if (
            not np.isfinite(self.cell_profile_relative_prior_strength)
            or self.cell_profile_relative_prior_strength <= 0.0
        ):
            raise ValueError(
                "em_assignment.cell_specific_expression.relative_prior_strength "
                "must be finite and > 0"
            )
        if (
            not np.isfinite(self.cell_profile_update_damping)
            or not 0.0 <= self.cell_profile_update_damping <= 1.0
        ):
            raise ValueError(
                "em_assignment.cell_specific_expression.profile_update_damping "
                "must be finite and in [0, 1]"
            )
        if self.cell_profile_compatibility_mode not in {
            "standard",
            "leave_one_out",
            "spatial_block_crossfit",
        }:
            raise ValueError(
                "em_assignment.cell_specific_expression.compatibility_mode "
                "must be one of: standard, leave_one_out, spatial_block_crossfit"
            )
        if (
            not np.isfinite(self.spatial_crossfit_block_size_um)
            or self.spatial_crossfit_block_size_um <= 0.0
        ):
            raise ValueError(
                "em_assignment.cell_specific_expression."
                "spatial_crossfit_block_size_um must be finite and > 0"
            )
        if self.cell_profile_reliability_mode not in {
            "nuclear_depth",
            "nuclear_heldout_predictive",
        }:
            raise ValueError(
                "em_assignment.cell_specific_expression.reliability_mode must be "
                "one of: nuclear_depth, nuclear_heldout_predictive"
            )
        if self.heldout_reliability_grid_size < 2:
            raise ValueError(
                "em_assignment.cell_specific_expression."
                "heldout_reliability_grid_size must be >= 2"
            )
        if (
            self.cell_specific_expression_enabled
            and self.cell_profile_compatibility_mode
            in {"leave_one_out", "spatial_block_crossfit"}
            and self.cell_profile_update_damping != 1.0
        ):
            raise ValueError(
                f"{self.cell_profile_compatibility_mode} compatibility requires "
                "profile_update_damping=1 "
                "so the excluded pseudocount state exactly matches the profile M-step"
            )
        if self.epsilon <= 0.0:
            raise ValueError("em_assignment.epsilon must be > 0")
        if self.enabled and not self.initialize_from_em:
            raise ValueError("generalized EM v1 requires rl_integration.initialize_from_em=true")
        if self.enabled and not self.lock_nuclear_bins:
            raise ValueError("generalized EM v1 requires rl_integration.lock_nuclear_bins=true")


@dataclass(frozen=True)
class EMIterationDiagnostics:
    iteration: int
    candidate_max_distance_used: float
    number_of_unique_bins: int
    number_of_cells: int
    number_of_candidate_pairs: int
    mean_candidates_per_bin: float
    max_candidates_per_bin: int
    mean_responsibility_change: float
    fraction_top_owner_changed: float
    mean_normalized_entropy: float
    median_normalized_entropy: float
    fraction_ambiguous: float
    number_nuclear_locked: int
    mean_q_change: float
    max_q_change: float
    mean_profile_total_variation: float
    max_profile_total_variation: float
    mean_cell_profile_reliability: float


@dataclass(frozen=True)
class SparseEMInput:
    """Patch-independent sparse input for the generalized-EM solver."""

    patch_id: str
    candidate_max_distance_um: float
    barcode_ids: tuple[str, ...]
    barcode_xy_um: np.ndarray
    ll: np.ndarray
    expression_confidence: np.ndarray
    candidate_row_splits: np.ndarray
    candidate_cell_index: np.ndarray
    pair_bin_index: np.ndarray
    pair_distance_um: np.ndarray
    cell_ids: tuple[str, ...]
    is_nuclear_locked: np.ndarray
    locked_owner_cell_index: np.ndarray
    log_prior: np.ndarray


@dataclass(frozen=True)
class EMAssignmentResult:
    """Compact ragged soft ownership state aligned by barcode and cell IDs."""

    patch_id: str
    candidate_max_distance_um: float
    barcode_ids: tuple[str, ...]
    barcode_index: np.ndarray
    barcode_xy_um: np.ndarray
    candidate_row_splits: np.ndarray
    candidate_cell_index: np.ndarray
    pair_bin_index: np.ndarray
    pair_distance_um: np.ndarray
    spatial_prior: np.ndarray
    expression_confidence: np.ndarray
    responsibility: np.ndarray
    background_probability: np.ndarray
    top1_cell_index: np.ndarray
    top1_probability: np.ndarray
    top2_probability: np.ndarray
    margin: np.ndarray
    normalized_entropy: np.ndarray
    is_nuclear_locked: np.ndarray
    locked_owner_cell_index: np.ndarray
    is_ambiguous: np.ndarray
    is_unassigned: np.ndarray
    cell_ids: tuple[str, ...]
    cell_type_posterior: np.ndarray
    cell_expression_profile: np.ndarray
    cell_profile_reliability: np.ndarray
    cell_profile_reliability_heldout_gain: np.ndarray
    cell_profile_reliability_heldout_nuclear_bins: np.ndarray
    nuclear_expression_count_total: np.ndarray
    relative_profile_prior_count: float
    final_type_expression_compatibility: np.ndarray
    final_cell_expression_compatibility: np.ndarray
    final_expression_compatibility: np.ndarray
    final_combined_score: np.ndarray
    iterations: tuple[EMIterationDiagnostics, ...] = field(default_factory=tuple)
    converged: bool = False
    n_disconnected_hard_islands: int = 0

    @property
    def n_bins(self) -> int:
        return int(len(self.barcode_ids))

    @property
    def n_cells(self) -> int:
        return int(len(self.cell_ids))

    @property
    def n_pairs(self) -> int:
        return int(self.candidate_cell_index.shape[0])

    @property
    def initial_owner_cell_index(self) -> np.ndarray:
        """Discrete top-1 state used to initialize the existing RL environment."""
        return self.top1_cell_index

    def barcode_to_index(self) -> dict[str, int]:
        return {barcode: idx for idx, barcode in enumerate(self.barcode_ids)}

    def cell_id_to_index(self) -> dict[str, int]:
        return {cell_id: idx for idx, cell_id in enumerate(self.cell_ids)}

    def validate(self, *, atol: float = 1.0e-5) -> None:
        """Validate sparse normalization, numeric safety, and exact nuclear locks."""
        b = self.n_bins
        p = self.n_pairs
        if self.barcode_index.shape != (b,) or self.barcode_index.dtype != np.int64:
            raise ValueError("barcode_index must have shape (B,) and dtype int64")
        if self.candidate_row_splits.shape != (b + 1,):
            raise ValueError("candidate_row_splits must have shape (B+1,)")
        if self.candidate_row_splits.dtype != np.int64:
            raise ValueError("candidate_row_splits must use int64")
        if int(self.candidate_row_splits[0]) != 0 or int(self.candidate_row_splits[-1]) != p:
            raise ValueError("candidate_row_splits endpoints are invalid")
        counts = np.diff(self.candidate_row_splits)
        if np.any(counts <= 0):
            raise ValueError("every EM bin must have at least one physical candidate cell")
        if self.candidate_cell_index.shape != (p,) or self.candidate_cell_index.dtype != np.int64:
            raise ValueError("candidate_cell_index must have shape (N_pairs,) and dtype int64")
        if self.pair_bin_index.shape != (p,) or self.pair_bin_index.dtype != np.int64:
            raise ValueError("pair_bin_index must have shape (N_pairs,) and dtype int64")
        if self.responsibility.shape != (p,):
            raise ValueError("responsibility must have shape (N_pairs,)")
        numeric_arrays = (
            self.pair_distance_um,
            self.spatial_prior,
            self.expression_confidence,
            self.responsibility,
            self.background_probability,
            self.top1_probability,
            self.top2_probability,
            self.margin,
            self.normalized_entropy,
            self.cell_type_posterior,
            self.cell_expression_profile,
            self.cell_profile_reliability,
            self.cell_profile_reliability_heldout_gain,
            self.nuclear_expression_count_total,
            self.final_type_expression_compatibility,
            self.final_cell_expression_compatibility,
            self.final_expression_compatibility,
            self.final_combined_score,
        )
        if any(not np.isfinite(np.asarray(arr)).all() for arr in numeric_arrays):
            raise ValueError("EM result contains NaN or Inf")
        profile = np.asarray(self.cell_expression_profile, dtype=np.float64)
        reliability = np.asarray(self.cell_profile_reliability, dtype=np.float64)
        heldout_gain = np.asarray(
            self.cell_profile_reliability_heldout_gain,
            dtype=np.float64,
        )
        heldout_bins = np.asarray(
            self.cell_profile_reliability_heldout_nuclear_bins
        )
        nuclear_total = np.asarray(
            self.nuclear_expression_count_total,
            dtype=np.float64,
        )
        if profile.ndim != 2 or profile.shape[0] != self.n_cells:
            raise ValueError("cell_expression_profile must have shape (C, G)")
        if reliability.shape != (self.n_cells,):
            raise ValueError("cell_profile_reliability must have shape (C,)")
        if heldout_gain.shape != (self.n_cells,):
            raise ValueError(
                "cell_profile_reliability_heldout_gain must have shape (C,)"
            )
        if heldout_bins.shape != (self.n_cells,) or heldout_bins.dtype != np.int64:
            raise ValueError(
                "cell_profile_reliability_heldout_nuclear_bins must have shape "
                "(C,) and dtype int64"
            )
        if nuclear_total.shape != (self.n_cells,):
            raise ValueError("nuclear_expression_count_total must have shape (C,)")
        if np.any(reliability < -atol) or np.any(reliability > 1.0 + atol):
            raise ValueError("cell_profile_reliability is outside [0, 1]")
        if np.any(heldout_gain < -atol):
            raise ValueError("cell_profile_reliability_heldout_gain must be nonnegative")
        if np.any(heldout_bins < 0):
            raise ValueError(
                "cell_profile_reliability_heldout_nuclear_bins must be nonnegative"
            )
        if np.any(nuclear_total < 0.0):
            raise ValueError("nuclear_expression_count_total must be nonnegative")
        if not np.isfinite(self.relative_profile_prior_count):
            raise ValueError("relative_profile_prior_count must be finite")
        if profile.shape[1] > 0 and not np.allclose(
            np.sum(profile, axis=1),
            1.0,
            rtol=0.0,
            atol=float(atol),
        ):
            raise ValueError("cell_expression_profile rows do not sum to one")
        for name, values in (
            ("final_type_expression_compatibility", self.final_type_expression_compatibility),
            ("final_cell_expression_compatibility", self.final_cell_expression_compatibility),
            ("final_expression_compatibility", self.final_expression_compatibility),
        ):
            if np.asarray(values).shape != (p,):
                raise ValueError(f"{name} must have shape (N_pairs,)")
        row_sums = np.add.reduceat(
            np.asarray(self.responsibility, dtype=np.float64),
            self.candidate_row_splits[:-1],
        )
        background = np.asarray(self.background_probability, dtype=np.float64)
        if background.shape != (b,):
            raise ValueError("background_probability must have shape (B,)")
        if np.any(background < -atol) or np.any(background > 1.0 + atol):
            raise ValueError("background_probability is outside [0, 1]")
        if not np.allclose(row_sums + background, 1.0, rtol=0.0, atol=float(atol)):
            raise ValueError("physical plus background responsibility rows do not sum to one")
        if self.is_unassigned.shape != (b,):
            raise ValueError("is_unassigned must have shape (B,)")
        if not np.array_equal(self.is_unassigned, self.top1_cell_index < 0):
            raise ValueError("is_unassigned must match negative top1_cell_index rows")
        if np.any(self.top1_cell_index < -1) or np.any(self.top1_cell_index >= self.n_cells):
            raise ValueError("top1_cell_index contains an invalid component index")
        if np.any(self.normalized_entropy < -atol) or np.any(self.normalized_entropy > 1.0 + atol):
            raise ValueError("normalized entropy is outside [0, 1]")
        for bin_idx in np.flatnonzero(self.is_nuclear_locked).tolist():
            if float(background[bin_idx]) != 0.0:
                raise ValueError("nuclear background responsibility must be exactly zero")
            start = int(self.candidate_row_splits[bin_idx])
            end = int(self.candidate_row_splits[bin_idx + 1])
            owner = int(self.locked_owner_cell_index[bin_idx])
            cells = self.candidate_cell_index[start:end]
            row = self.responsibility[start:end]
            expected = (cells == owner).astype(np.float32)
            if not np.array_equal(row, expected):
                raise ValueError("nuclear responsibility row is not exact one-hot")
