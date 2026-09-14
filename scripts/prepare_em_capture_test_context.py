#!/usr/bin/env python
"""Prepare patch-complete nuclei and an episode config for capture testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-index", required=True)
    parser.add_argument("--full-nuclei-path", required=True)
    parser.add_argument("--episode-config-template", required=True)
    parser.add_argument("--capture-matrix-path", required=True)
    parser.add_argument(
        "--reference-npz-path",
        default=None,
        help="Optional donor-specific reference bundle replacing the template reference.",
    )
    parser.add_argument("--episode-output-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--n-workers", type=int, default=48)
    return parser.parse_args()


def _cell_ids(values: pd.Series) -> set[str]:
    result: set[str] = set()
    for raw in values.astype(str):
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("patch_cell_ids must contain JSON lists")
        result.update(str(value) for value in parsed)
    return result


def prepare_context(args: argparse.Namespace) -> dict[str, Any]:
    patch_index = Path(args.patch_index).expanduser().resolve()
    full_nuclei_path = Path(args.full_nuclei_path).expanduser().resolve()
    template_path = Path(args.episode_config_template).expanduser().resolve()
    capture_matrix_path = Path(args.capture_matrix_path).expanduser().resolve()
    reference_npz_path = (
        None
        if args.reference_npz_path is None
        else Path(args.reference_npz_path).expanduser().resolve()
    )
    episode_output_root = Path(args.episode_output_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    patches = pd.read_csv(patch_index)
    if "patch_cell_ids" not in patches:
        raise ValueError("patch index is missing patch_cell_ids")
    requested_ids = _cell_ids(patches["patch_cell_ids"])
    nuclei = pd.read_parquet(full_nuclei_path)
    if "cell_id" not in nuclei:
        raise ValueError("nuclei table is missing cell_id")
    normalized_ids = nuclei["cell_id"].astype(str)
    if bool(normalized_ids.duplicated().any()):
        raise ValueError("full nuclei table contains duplicate cell IDs")
    selected = nuclei.loc[normalized_ids.isin(requested_ids)].copy()
    found_ids = set(selected["cell_id"].astype(str))
    if found_ids != requested_ids:
        missing = sorted(requested_ids.difference(found_ids))
        raise ValueError(f"patch context nuclei are missing: {missing[:10]}")
    selected = selected.sort_values("cell_id").reset_index(drop=True)
    nuclei_path = output_dir / "nuclei_patch_context.parquet"
    selected.to_parquet(nuclei_path, index=False)
    (output_dir / "patch_context_cell_ids.txt").write_text(
        "\n".join(sorted(requested_ids)) + "\n", encoding="utf-8"
    )

    with template_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("episode config template root must be a mapping")
    config["run"]["name"] = str(args.run_name)
    config["run"]["output_root"] = str(episode_output_root)
    config["inputs"]["nuclei_path"] = str(nuclei_path)
    config["inputs"]["expression"]["matrix_path"] = str(capture_matrix_path)
    if reference_npz_path is not None:
        if not reference_npz_path.exists():
            raise FileNotFoundError(reference_npz_path)
        config["inputs"]["expression"]["reference_npz_path"] = str(
            reference_npz_path
        )
    config["inputs"]["expression"]["filter_empty_nuclear_cells"] = False
    config["inputs"]["expression"]["min_nuclear_expression_sum"] = 0.0
    config["inputs"]["expression"]["n_workers"] = int(args.n_workers)
    config_path = output_dir / "episode_build.capture_thinned.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    summary = {
        "patch_index": str(patch_index),
        "full_nuclei_path": str(full_nuclei_path),
        "n_patches": int(len(patches)),
        "n_patch_context_cells": int(len(requested_ids)),
        "n_selected_nuclei": int(len(selected)),
        "cell_id_set_exact": True,
        "capture_matrix_path": str(capture_matrix_path),
        "reference_npz_path": (
            None if reference_npz_path is None else str(reference_npz_path)
        ),
        "episode_output_root": str(episode_output_root),
        "episode_config": str(config_path),
        "filter_empty_nuclear_cells": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    print(json.dumps(prepare_context(_parse_args()), indent=2))


if __name__ == "__main__":
    main()
