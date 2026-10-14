"""Recoverable temporary checkpoints for long formula-rollout diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Iterable

import pandas as pd


def _validated_checkpoint_dir(raw_path: Path) -> Path:
    path = raw_path.expanduser().resolve()
    tmp_root = Path("/tmp").resolve()
    if path == tmp_root or tmp_root not in path.parents:
        raise ValueError("formula checkpoint directory must be a child of /tmp")
    return path


def _paths(directory: Path, formula_name: str) -> dict[str, Path]:
    safe_name = str(formula_name).strip()
    if not safe_name or not safe_name.replace("_", "").isalnum():
        raise ValueError(f"unsafe formula checkpoint name: {formula_name!r}")
    prefix = directory / f"formula_{safe_name}"
    return {
        "manifest": prefix.with_suffix(".json"),
        "patch": directory / f"formula_{safe_name}_patch.parquet",
        "action": directory / f"formula_{safe_name}_action.parquet",
        "cell": directory / f"formula_{safe_name}_cell.parquet",
    }


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def save_formula_checkpoint(
    *,
    directory: Path,
    split: str,
    formula: dict[str, Any],
    patch_ids: Iterable[str],
    patch_rows: Iterable[dict[str, Any]],
    action_rows: Iterable[dict[str, Any]],
    cell_rows: Iterable[dict[str, Any]],
) -> None:
    checkpoint_dir = _validated_checkpoint_dir(directory)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    formula_name = str(formula["name"])
    paths = _paths(checkpoint_dir, formula_name)
    patch_frame = pd.DataFrame(tuple(patch_rows))
    action_frame = pd.DataFrame(tuple(action_rows))
    cell_frame = pd.DataFrame(tuple(cell_rows))
    if patch_frame.empty:
        raise ValueError("cannot checkpoint a formula without patch rows")
    if action_frame.empty:
        action_frame = pd.DataFrame(columns=["split", "formula", "patch_id"])
    if cell_frame.empty:
        cell_frame = pd.DataFrame(columns=["split", "formula", "patch_id"])
    _atomic_parquet(patch_frame, paths["patch"])
    _atomic_parquet(action_frame, paths["action"])
    _atomic_parquet(cell_frame, paths["cell"])
    payload = {
        "split": str(split),
        "formula": formula,
        "patch_ids": [str(value) for value in patch_ids],
        "n_patch_rows": int(len(patch_frame)),
        "n_action_rows": int(len(action_frame)),
        "n_cell_rows": int(len(cell_frame)),
    }
    temporary_manifest = paths["manifest"].with_name(
        f".{paths['manifest'].name}.tmp"
    )
    temporary_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(paths["manifest"])


def load_formula_checkpoint(
    *,
    directory: Path,
    split: str,
    formula: dict[str, Any],
    patch_ids: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    checkpoint_dir = _validated_checkpoint_dir(directory)
    paths = _paths(checkpoint_dir, str(formula["name"]))
    if not paths["manifest"].exists():
        return None
    payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    expected_patch_ids = [str(value) for value in patch_ids]
    if payload.get("split") != str(split):
        raise RuntimeError("formula checkpoint split does not match")
    if payload.get("formula") != formula:
        raise RuntimeError("formula checkpoint weights do not match")
    if payload.get("patch_ids") != expected_patch_ids:
        raise RuntimeError("formula checkpoint patch IDs do not match")
    for key in ("patch", "action", "cell"):
        if not paths[key].exists():
            raise RuntimeError(f"formula checkpoint is incomplete: {paths[key]}")
    patch_frame = pd.read_parquet(paths["patch"])
    action_frame = pd.read_parquet(paths["action"])
    cell_frame = pd.read_parquet(paths["cell"])
    if len(patch_frame) != int(payload["n_patch_rows"]):
        raise RuntimeError("formula checkpoint patch-row count mismatch")
    if len(action_frame) != int(payload["n_action_rows"]):
        raise RuntimeError("formula checkpoint action-row count mismatch")
    if len(cell_frame) != int(payload["n_cell_rows"]):
        raise RuntimeError("formula checkpoint cell-row count mismatch")
    return patch_frame, action_frame, cell_frame


def remove_formula_checkpoint(directory: Path) -> None:
    checkpoint_dir = _validated_checkpoint_dir(directory)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
