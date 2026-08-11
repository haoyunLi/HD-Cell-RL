"""Public API for the HD spatial-cell RL scaffold."""

from __future__ import annotations

__all__ = [
    "Action",
    "ActionType",
    "BinRecord",
    "CellAssignmentEnv",
    "CellAssignmentState",
    "CellEpisodeData",
    "EnvironmentConfig",
    "EpisodeBuildConfig",
    "NucleusRecord",
    "Policy",
    "PosteriorAddBinReward",
    "RandomPolicy",
    "RewardFunction",
    "ZeroReward",
    "build_episode_for_cell",
    "build_episodes",
    "run_episode_build",
    "run_episode_build_from_config",
    "compute_bin_log_likelihood_by_type",
    "compute_reference_distribution",
    "PPOTrainingConfig",
    "ActorCritic",
    "run_ppo_training",
    "run_ppo_training_from_config",
]

_LAZY_EXPORTS = {
    "Action": (".actions", "Action"),
    "ActionType": (".actions", "ActionType"),
    "build_episode_for_cell": (".builder", "build_episode_for_cell"),
    "build_episodes": (".builder", "build_episodes"),
    "EnvironmentConfig": (".config", "EnvironmentConfig"),
    "EpisodeBuildConfig": (".episode_build", "EpisodeBuildConfig"),
    "run_episode_build": (".episode_build", "run_episode_build"),
    "run_episode_build_from_config": (".episode_build", "run_episode_build_from_config"),
    "CellAssignmentEnv": (".environment", "CellAssignmentEnv"),
    "BinRecord": (".models", "BinRecord"),
    "CellEpisodeData": (".models", "CellEpisodeData"),
    "NucleusRecord": (".models", "NucleusRecord"),
    "Policy": (".policy", "Policy"),
    "RandomPolicy": (".policy", "RandomPolicy"),
    "PosteriorAddBinReward": (".reward", "PosteriorAddBinReward"),
    "RewardFunction": (".reward", "RewardFunction"),
    "ZeroReward": (".reward", "ZeroReward"),
    "compute_bin_log_likelihood_by_type": (".reward", "compute_bin_log_likelihood_by_type"),
    "compute_reference_distribution": (".reward", "compute_reference_distribution"),
    "CellAssignmentState": (".state", "CellAssignmentState"),
    "PPOTrainingConfig": (".ppo_config", "PPOTrainingConfig"),
    "ActorCritic": (".ppo_model", "ActorCritic"),
    "run_ppo_training": (".ppo_training", "run_ppo_training"),
    "run_ppo_training_from_config": (".ppo_training", "run_ppo_training_from_config"),
}


def __getattr__(name: str):
    """Load public objects on demand instead of during package import."""
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    from importlib import import_module

    module = import_module(module_name, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
