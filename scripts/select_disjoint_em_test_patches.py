#!/usr/bin/env python
"""Select metadata-matched EM test patches without consulting ground truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "patch_id",
    "source",
    "context_x_min",
    "context_x_max",
    "context_y_min",
    "context_y_max",
    "candidate_max_distance_um",
    "n_patch_cells",
    "n_core_cells",
    "patch_cell_ids",
    "core_cell_ids",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-patch-index", required=True)
    parser.add_argument("--reference-patch-index", required=True)
    parser.add_argument(
        "--exclude-patch-index",
        action="append",
        default=[],
        help=(
            "Optional patch index to exclude from selection. May be repeated; "
            "excluded patches contribute both context rectangles and patch cell IDs."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def _cell_ids(raw: Any) -> set[str]:
    values = json.loads(str(raw))
    if not isinstance(values, list):
        raise ValueError("patch cell IDs must be a JSON list")
    return {str(value) for value in values}


def _context_overlaps(left: pd.Series, right: pd.Series) -> bool:
    return not (
        float(left["context_x_max"]) < float(right["context_x_min"])
        or float(left["context_x_min"]) > float(right["context_x_max"])
        or float(left["context_y_max"]) < float(right["context_y_min"])
        or float(left["context_y_min"]) > float(right["context_y_max"])
    )


def _hash_rank(patch_id: str, seed: int) -> str:
    return hashlib.sha256(f"{int(seed)}:{patch_id}".encode("utf-8")).hexdigest()


def select_disjoint_test_patches(
    *,
    full: pd.DataFrame,
    reference: pd.DataFrame,
    seed: int,
    excluded: Sequence[pd.DataFrame] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match reference metadata while enforcing cell and context disjointness."""
    named_frames = [("full", full), ("reference", reference)] + [
        (f"excluded[{index}]", frame) for index, frame in enumerate(excluded)
    ]
    for name, frame in named_frames:
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} patch index is missing columns: {sorted(missing)}")
        if bool(frame["patch_id"].astype(str).duplicated().any()):
            raise ValueError(f"{name} patch index contains duplicate patch_id values")
    reference_rows = [row for _, row in reference.iterrows()]
    excluded_rows = [
        row for frame in excluded for _, row in frame.iterrows()
    ]
    forbidden_rows = reference_rows + excluded_rows
    forbidden_patch_ids = set(reference["patch_id"].astype(str))
    for frame in excluded:
        forbidden_patch_ids.update(frame["patch_id"].astype(str))
    reference_cells: set[str] = set()
    for row in forbidden_rows:
        raw = row["patch_cell_ids"]
        reference_cells.update(_cell_ids(raw))

    selected_rows: list[pd.Series] = []
    selected_cells: set[str] = set()
    audit_rows: list[dict[str, Any]] = []
    for template in reference_rows:
        exact = full.loc[
            (full["source"].astype(str) == str(template["source"]))
            & (full["n_core_cells"].astype(int) == int(template["n_core_cells"]))
            & (full["n_patch_cells"].astype(int) == int(template["n_patch_cells"]))
            & np.isclose(
                full["candidate_max_distance_um"].astype(float),
                float(template["candidate_max_distance_um"]),
                rtol=0.0,
                atol=1.0e-6,
            )
        ].copy()
        exact = exact.loc[
            ~exact["patch_id"].astype(str).isin(forbidden_patch_ids)
        ]
        exact["_hash_rank"] = exact["patch_id"].astype(str).map(
            lambda patch_id: _hash_rank(patch_id, seed)
        )
        exact = exact.sort_values(["_hash_rank", "patch_id"])

        chosen: pd.Series | None = None
        eligible_count = 0
        for _, candidate in exact.iterrows():
            candidate_cells = _cell_ids(candidate["patch_cell_ids"])
            if candidate_cells.intersection(reference_cells):
                continue
            if any(_context_overlaps(candidate, row) for row in forbidden_rows):
                continue
            if candidate_cells.intersection(selected_cells):
                continue
            if any(_context_overlaps(candidate, row) for row in selected_rows):
                continue
            eligible_count += 1
            if chosen is None:
                chosen = candidate
        if chosen is None:
            raise RuntimeError(
                "no disjoint exact metadata match for reference patch "
                f"{template['patch_id']}"
            )
        chosen_cells = _cell_ids(chosen["patch_cell_ids"])
        selected_rows.append(chosen)
        selected_cells.update(chosen_cells)
        audit_rows.append(
            {
                "reference_patch_id": str(template["patch_id"]),
                "test_patch_id": str(chosen["patch_id"]),
                "source": str(chosen["source"]),
                "n_core_cells": int(chosen["n_core_cells"]),
                "n_patch_cells": int(chosen["n_patch_cells"]),
                "candidate_max_distance_um": float(
                    chosen["candidate_max_distance_um"]
                ),
                "n_exact_candidates_before_disjoint_filter": int(len(exact)),
                "n_eligible_disjoint_candidates": int(eligible_count),
                "selection_hash": str(chosen["_hash_rank"]),
                "context_overlap_with_reference": False,
                "cell_overlap_with_reference": 0,
                "context_overlap_with_other_test_patches": False,
                "cell_overlap_with_other_test_patches": 0,
            }
        )
    selected = pd.DataFrame(selected_rows).drop(columns=["_hash_rank"])
    selected = selected.loc[:, full.columns]
    return selected.reset_index(drop=True), pd.DataFrame(audit_rows)


def main() -> None:
    args = _parse_args()
    full_path = Path(args.full_patch_index).expanduser().resolve()
    reference_path = Path(args.reference_patch_index).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    full = pd.read_csv(full_path)
    reference = pd.read_csv(reference_path)
    excluded_paths = [Path(value).expanduser().resolve() for value in args.exclude_patch_index]
    excluded = [pd.read_csv(path) for path in excluded_paths]
    selected, audit = select_disjoint_test_patches(
        full=full,
        reference=reference,
        seed=int(args.seed),
        excluded=excluded,
    )
    selected.to_csv(output_dir / "test_patches_index.csv", index=False)
    audit.to_csv(output_dir / "selection_audit.csv", index=False)

    target_rows: list[dict[str, str]] = []
    for row in selected.itertuples(index=False):
        for cell_id in sorted(_cell_ids(getattr(row, "core_cell_ids"))):
            target_rows.append(
                {"cell_id": str(cell_id), "patch_id": str(getattr(row, "patch_id"))}
            )
    target = pd.DataFrame(target_rows)
    if bool(target["cell_id"].duplicated().any()):
        raise RuntimeError("selected test patches contain duplicate core cell IDs")
    source_eval_dir = output_dir / "source_eval"
    source_eval_dir.mkdir()
    target.to_csv(source_eval_dir / "per_episode.csv", index=False)

    summary = {
        "created_from_ground_truth": False,
        "selection_uses_performance_metrics": False,
        "seed": int(args.seed),
        "full_patch_index": str(full_path),
        "reference_patch_index": str(reference_path),
        "excluded_patch_indices": [str(path) for path in excluded_paths],
        "n_explicitly_excluded_patches": int(sum(len(frame) for frame in excluded)),
        "n_reference_patches": int(len(reference)),
        "n_test_patches": int(len(selected)),
        "n_reference_core_cells": int(reference["n_core_cells"].sum()),
        "n_test_core_cells": int(selected["n_core_cells"].sum()),
        "n_test_context_cells": int(
            len(
                set().union(
                    *(_cell_ids(raw) for raw in selected["patch_cell_ids"])
                )
            )
        ),
        "patch_id_overlap": 0,
        "context_cell_id_overlap": 0,
        "context_rectangle_overlap": False,
        "matching_fields": [
            "source",
            "n_core_cells",
            "n_patch_cells",
            "candidate_max_distance_um",
        ],
        "test_patch_ids": selected["patch_id"].astype(str).tolist(),
        "test_core_cell_ids": target["cell_id"].astype(str).tolist(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
