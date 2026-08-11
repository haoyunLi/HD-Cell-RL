#!/usr/bin/env python
"""Backfill patch-level debug JSON and plots for an existing evaluation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.patch_debug_export import (
    export_patch_debug_bundle,
    update_evaluation_summary_with_patch_debug,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-run-dir", required=True)
    parser.add_argument("--patch-assignments-csv", default=None)
    parser.add_argument("--patches-index-path", default=None)
    parser.add_argument("--merge-candidates-csv", default=None)
    parser.add_argument("--gt-cell-bins-path", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    eval_run_dir = Path(args.eval_run_dir).expanduser().resolve()
    summary_path = eval_run_dir / "summary.json"
    per_episode_csv = eval_run_dir / "per_episode.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not per_episode_csv.exists():
        raise FileNotFoundError(per_episode_csv)

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    merged_assignments = Path(str(summary["patch_rl_assignments_path"])).expanduser().resolve()
    intermediate_dir = merged_assignments.parent
    pipeline_path = intermediate_dir / "patch_eval_pipeline_summary.json"
    if not pipeline_path.exists():
        raise FileNotFoundError(pipeline_path)
    with pipeline_path.open("r", encoding="utf-8") as handle:
        pipeline = json.load(handle)

    default_patch_assignments = intermediate_dir / "patch_rl_patch_assignments.csv"
    if not default_patch_assignments.exists():
        default_patch_assignments = merged_assignments
    patch_assignments_csv = _resolve_path(args.patch_assignments_csv, default_patch_assignments)
    patches_index_path = _resolve_path(args.patches_index_path, Path(str(pipeline["patches_index_path"])))
    merge_candidates_csv = _resolve_optional_path(
        args.merge_candidates_csv,
        pipeline.get("merge_candidates_csv"),
    )
    gt_cell_bins_path = _resolve_path(
        args.gt_cell_bins_path,
        Path(str(summary["gt_cell_bins_path"])),
    )
    output_dir = _resolve_path(args.output_dir, eval_run_dir / "patch_debug")

    patch_ids = pd.read_csv(patch_assignments_csv, usecols=["patch_id"])["patch_id"].astype(str).drop_duplicates().tolist()
    manifest_path = export_patch_debug_bundle(
        patch_assignments_csv=patch_assignments_csv,
        patches_index_path=patches_index_path,
        per_episode_csv=per_episode_csv,
        gt_cell_bins_path=gt_cell_bins_path,
        output_dir=output_dir,
        patch_ids=patch_ids,
        merge_candidates_csv=merge_candidates_csv,
        source_eval_run_dir=eval_run_dir,
    )
    update_evaluation_summary_with_patch_debug(
        summary_path=summary_path,
        manifest_path=manifest_path,
    )
    print(f"Patch debug manifest: {manifest_path}")


def _resolve_path(raw: str | None, fallback: Path) -> Path:
    return Path(str(fallback if raw in (None, "") else raw)).expanduser().resolve()


def _resolve_optional_path(raw: str | None, fallback: str | None) -> Path | None:
    value = fallback if raw in (None, "") else raw
    return None if value in (None, "") else Path(str(value)).expanduser().resolve()


if __name__ == "__main__":
    main()
