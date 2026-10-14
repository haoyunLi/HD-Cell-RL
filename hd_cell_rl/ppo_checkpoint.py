"""Actor-critic construction and checkpoint loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import tempfile
import warnings

import torch

from .ppo_config import PPOTrainingConfig
from .ppo_feature_schema import ACTION_FEATURE_DIM, GLOBAL_FEATURE_DIM, ACTION_FEATURE_NAMES, GLOBAL_FEATURE_NAMES
from .ppo_model import ActorCritic


def checkpoint_contract(*, training_unit: str = "cell", agent_mode: str = "single_cell", reward_backend: str = "standard", rollout_backend: str = "legacy_cpu") -> dict[str, Any]:
    """Version feature semantics, not just tensor dimensions. No feature changes."""
    return {"version": 1, "global_features": list(GLOBAL_FEATURE_NAMES),
            "action_features": list(ACTION_FEATURE_NAMES), "training_unit": training_unit,
            "agent_mode": agent_mode, "reward_backend": reward_backend,
            "rollout_backend": rollout_backend}


def validate_checkpoint_contract(payload: dict[str, Any], *, expected: dict[str, Any], allow_semantic_transfer: bool = False) -> None:
    actual = payload.get("feature_contract")
    if actual is None:
        warnings.warn("Legacy checkpoint has no feature contract; semantic compatibility cannot be verified", RuntimeWarning)
        return
    if actual != expected:
        message = f"Checkpoint feature/environment contract mismatch: saved={actual}, expected={expected}"
        if not allow_semantic_transfer:
            raise ValueError(message)
        warnings.warn(message + "; explicit warm-start transfer enabled", RuntimeWarning)


def atomic_save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    """Publish a complete checkpoint atomically; preserve the old file on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint_payload(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a checkpoint payload and validate its top-level shape."""
    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"invalid checkpoint payload in {path}")
    return payload


def build_actor_critic_from_config(
    config: PPOTrainingConfig,
    *,
    device: torch.device | None = None,
) -> ActorCritic:
    """Build the policy model shape implied by a resolved PPO config."""
    model = ActorCritic(
        global_dim=GLOBAL_FEATURE_DIM,
        action_dim=ACTION_FEATURE_DIM,
        hidden_dim=int(config.hidden_dim),
        planner_enabled=bool(config.planner_enabled),
        planner_mode_count=len(config.planner_modes),
    )
    if device is not None:
        model = model.to(device)
    return model


def load_actor_critic_checkpoint(
    checkpoint_path: str | Path,
    config: PPOTrainingConfig,
    *,
    device: torch.device,
    payload: dict[str, Any] | None = None,
) -> tuple[ActorCritic, dict[str, Any]]:
    """Load an actor-critic checkpoint using the model shape from ``config``."""
    path = Path(checkpoint_path).expanduser().resolve()
    if payload is None:
        payload = load_checkpoint_payload(path)
    if "model_state_dict" not in payload:
        raise ValueError(f"checkpoint missing 'model_state_dict': {path}")

    validate_checkpoint_contract(payload, expected=checkpoint_contract())

    model = build_actor_critic_from_config(config, device=device)
    try:
        model.load_state_dict(payload["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"checkpoint model shape does not match config/state feature schema: {path}"
        ) from exc
    model.eval()
    return model, payload
