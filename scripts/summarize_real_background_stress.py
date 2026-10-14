#!/usr/bin/env python3
"""Summarize the formal real-background stress-test method matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONDITIONS = ("real_background_20pct", "real_background_50pct")
METHOD_ORDER = (
    "nucleus_only",
    "distance_only",
    "em_only",
    "em_plus_frozen_rl",
    "bin2cell",
    "stcs",
    "smurf",
)
METRICS = (
    "mean_pred_iou",
    "mean_pred_fractional_iou",
    "mean_pred_precision",
    "mean_pred_recall",
    "mean_gene_spearman_r",
    "mean_gene_rmse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress-root", required=True, type=Path)
    parser.add_argument(
        "--clean-comparison-csv",
        required=True,
        type=Path,
        help="Capture-depth long table containing the high_depth_local baseline.",
    )
    return parser.parse_args()


def method_from_evaluation_name(name: str) -> str | None:
    matches = {
        "hd_cell_rl_nucleus_only": "nucleus_only",
        "hd_cell_rl_distance_only": "distance_only",
        "hd_cell_rl_em_only": "em_only",
        "hd_cell_rl_em_plus_frozen_rl": "em_plus_frozen_rl",
        "bin2cell": "bin2cell",
        "stcs": "stcs",
    }
    for token, method in matches.items():
        if token in name:
            return method
    return None


def assignment_path(summary: dict, method: str) -> Path | None:
    key_by_method = {
        "nucleus_only": "hd_cell_rl_nucleus_only_assignments_path",
        "distance_only": "hd_cell_rl_distance_only_assignments_path",
        "em_only": "hd_cell_rl_em_only_assignments_path",
        "bin2cell": "bin2cell_assignments_path",
        "stcs": "stcs_assignments_path",
    }
    if method == "em_plus_frozen_rl":
        value = summary.get("patch_rl_assignments_path")
    else:
        value = summary.get(key_by_method.get(method, ""))
    return Path(value) if value else None


def audit_unique_owner(path: Path) -> dict:
    frame = pd.read_csv(path, usecols=["barcode", "cell_id"])
    duplicated_rows = frame.duplicated(["barcode", "cell_id"]).sum()
    owners = frame.drop_duplicates(["barcode", "cell_id"]).groupby("barcode")[
        "cell_id"
    ].nunique()
    return {
        "n_rows": int(len(frame)),
        "n_unique_barcodes": int(frame["barcode"].nunique()),
        "n_duplicate_rows": int(duplicated_rows),
        "n_barcodes_with_multiple_cell_owners": int((owners > 1).sum()),
        "max_cell_owners_per_barcode": int(owners.max()) if len(owners) else 0,
    }


def main() -> None:
    args = parse_args()
    stress_root = args.stress_root.resolve()
    evaluation_root = stress_root / "evaluations"
    output_dir = stress_root / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    audits: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for summary_path in sorted(evaluation_root.glob("*/summary.json")):
        evaluation_name = summary_path.parent.name
        condition = next(
            (item for item in CONDITIONS if evaluation_name.startswith(item + "_")),
            None,
        )
        method = method_from_evaluation_name(evaluation_name)
        if condition is None or method is None:
            continue
        key = (condition, method)
        if key in seen:
            raise ValueError(f"Duplicate completed evaluation for {key}")
        seen.add(key)
        summary = json.loads(summary_path.read_text())
        row = {
            "condition": condition,
            "method": method,
            "status": "completed",
            "evaluation_dir": str(summary_path.parent),
            **{metric: summary.get(metric) for metric in METRICS},
            "n_matched_pred": summary.get("n_matched_pred"),
            "n_matched_gt": summary.get("n_matched_gt"),
            "failure_stage": None,
            "failure_reason": None,
            "n_cells_after_smurf_filters": None,
        }
        rows.append(row)
        path = assignment_path(summary, method)
        if path is None or not path.exists():
            audits.append(
                {
                    "condition": condition,
                    "method": method,
                    "status": "missing",
                    "assignment_path": str(path) if path else None,
                }
            )
        else:
            audits.append(
                {
                    "condition": condition,
                    "method": method,
                    "status": "audited",
                    "assignment_path": str(path),
                    **audit_unique_owner(path),
                }
            )

    filter_path = stress_root / "diagnostics" / "smurf_filtering.json"
    filtering = json.loads(filter_path.read_text())
    filter_by_condition = {
        item["dataset"]: item for item in filtering["conditions"]
    }
    failures = {
        "real_background_20pct": {
            "failure_stage": "post_iterative_arrangement_output_contract",
            "failure_reason": (
                "SMURF completed 30 iterative-arrangement passes but did not "
                "produce the required adatas.h5ad artifact"
            ),
        },
        "real_background_50pct": {
            "failure_stage": "singlecellanalysis_neighbors",
            "failure_reason": "X_pca does not have enough PCs",
        },
    }
    failure_records = []
    for condition in CONDITIONS:
        details = filter_by_condition[condition]
        record = {
            "condition": condition,
            "method": "smurf",
            "status": "failed",
            "evaluation_dir": None,
            **{metric: None for metric in METRICS},
            "n_matched_pred": None,
            "n_matched_gt": None,
            **failures[condition],
            "n_cells_after_smurf_filters": details[
                "n_cells_passing_min_counts_500"
            ],
        }
        rows.append(record)
        failure_records.append(
            {
                **record,
                "n_genes_after_min_counts_100": details[
                    "n_genes_after_min_counts_100"
                ],
                "nuclear_selected_gene_umi_median": details[
                    "nuclear_selected_gene_umi_median"
                ],
                "method_output_dir": str(
                    stress_root / "method_outputs" / condition / "smurf"
                ),
                "protocol_changed": False,
            }
        )

    long = pd.DataFrame(rows)
    expected = {(condition, method) for condition in CONDITIONS for method in METHOD_ORDER}
    actual = set(zip(long["condition"], long["method"]))
    if actual != expected:
        raise ValueError(
            f"Incomplete method matrix; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    order = {
        (condition, method): index
        for index, (condition, method) in enumerate(
            (pair for condition in CONDITIONS for pair in [(condition, m) for m in METHOD_ORDER])
        )
    }
    long["_order"] = [order[(c, m)] for c, m in zip(long.condition, long.method)]
    long = long.sort_values("_order").drop(columns="_order")

    clean = pd.read_csv(args.clean_comparison_csv)
    clean = clean.loc[clean["condition"] == "high_depth_local"].copy()
    clean = clean.set_index("method", drop=False)
    paired_rows = []
    for row in long.itertuples(index=False):
        clean_row = clean.loc[row.method] if row.method in clean.index else None
        output = {
            "condition": row.condition,
            "method": row.method,
            "background_status": row.status,
            "clean_status": clean_row["status"] if clean_row is not None else None,
        }
        for metric in METRICS:
            background_value = getattr(row, metric)
            clean_value = clean_row[metric] if clean_row is not None else np.nan
            output[f"clean_{metric}"] = clean_value
            output[f"background_{metric}"] = background_value
            output[f"delta_background_minus_clean_{metric}"] = (
                float(background_value) - float(clean_value)
                if pd.notna(background_value) and pd.notna(clean_value)
                else np.nan
            )
        paired_rows.append(output)
    paired = pd.DataFrame(paired_rows)

    audit = pd.DataFrame(audits)
    numeric_audit_columns = (
        "n_rows",
        "n_unique_barcodes",
        "n_duplicate_rows",
        "n_barcodes_with_multiple_cell_owners",
        "max_cell_owners_per_barcode",
    )
    for column in numeric_audit_columns:
        if column not in audit:
            audit[column] = np.nan
    audit = audit[
        ["condition", "method", "status", "assignment_path", *numeric_audit_columns]
    ].sort_values(["condition", "method"])

    long_path = output_dir / "real_background_all_methods_long.csv"
    paired_path = output_dir / "real_background_vs_clean.csv"
    audit_path = output_dir / "unique_owner_audit.csv"
    failure_path = stress_root / "diagnostics" / "smurf_failures.json"
    summary_path = output_dir / "summary.json"
    long.to_csv(long_path, index=False)
    paired.to_csv(paired_path, index=False)
    audit.to_csv(audit_path, index=False)
    failure_path.write_text(json.dumps(failure_records, indent=2) + "\n")

    audited = audit.loc[audit["status"] == "audited"]
    all_unique = bool(
        (audited["n_barcodes_with_multiple_cell_owners"] == 0).all()
        and (audited["max_cell_owners_per_barcode"] <= 1).all()
    )
    summary = {
        "comparison_long_csv": str(long_path),
        "comparison_vs_clean_csv": str(paired_path),
        "unique_owner_audit_csv": str(audit_path),
        "smurf_failures_json": str(failure_path),
        "n_completed_evaluations": int((long["status"] == "completed").sum()),
        "n_failed_evaluations": int((long["status"] == "failed").sum()),
        "all_audited_assignments_unique_owner": all_unique,
        "clean_reference_condition": "high_depth_local",
        "clean_reference_csv": str(args.clean_comparison_csv.resolve()),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
