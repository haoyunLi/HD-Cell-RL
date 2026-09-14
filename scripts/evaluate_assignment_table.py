#!/usr/bin/env python3
"""Evaluate an existing barcode-to-cell assignment table on a fixed cell set."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.ppo_format_assignment_eval import (
    add_ppo_format_assignment_eval_args,
    run_ppo_format_assignment_evaluation,
    validate_ppo_format_assignment_eval_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments-csv", required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--method-label", required=True)
    parser.add_argument("--nuclear-source", default="external_ppo_aligned")
    parser.add_argument("--external-nuclear-bins-path", default=None)
    add_ppo_format_assignment_eval_args(
        parser,
        default_eval_run_name="assignment_table_eval",
        method_label="assignment table",
    )
    args = parser.parse_args()
    if args.ppo_eval_run_dir is None:
        raise ValueError("--ppo_eval_run_dir is required")
    validate_ppo_format_assignment_eval_args(args)
    return args


def main() -> None:
    args = parse_args()
    assignments = Path(args.assignments_csv).expanduser().resolve()
    if not assignments.is_file():
        raise FileNotFoundError(assignments)
    external = (
        None
        if args.external_nuclear_bins_path is None
        else Path(args.external_nuclear_bins_path).expanduser().resolve()
    )
    run_dir = run_ppo_format_assignment_evaluation(
        assignments_csv=assignments,
        method_name=str(args.method_name),
        method_label=str(args.method_label),
        nuclear_source=str(args.nuclear_source),
        external_nuclear_bins_path=external,
        args=args,
        pipeline_config={
            "source": "existing_assignment_table",
            "assignments_csv": str(assignments),
        },
    )
    print(run_dir)


if __name__ == "__main__":
    main()
