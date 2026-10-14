#!/usr/bin/env python3
"""Validate and summarize the formal seven-method pseudo-HD benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METHOD_ORDER = (
    "nucleus_only",
    "distance_only",
    "em_only",
    "em_plus_frozen_rl",
    "bin2cell",
    "stcs",
    "smurf",
)
METHOD_TOKENS = {
    "hd_cell_rl_nucleus_only": "nucleus_only",
    "hd_cell_rl_distance_only": "distance_only",
    "hd_cell_rl_em_only": "em_only",
    "hd_cell_rl_em_plus_frozen_rl": "em_plus_frozen_rl",
    "bin2cell": "bin2cell",
    "stcs": "stcs",
    "smurf": "smurf",
}
ASSIGNMENT_KEYS = {
    "nucleus_only": "hd_cell_rl_nucleus_only_assignments_path",
    "distance_only": "hd_cell_rl_distance_only_assignments_path",
    "em_only": "hd_cell_rl_em_only_assignments_path",
    "em_plus_frozen_rl": "patch_rl_assignments_path",
    "bin2cell": "bin2cell_assignments_path",
    "stcs": "stcs_assignments_path",
    "smurf": "smurf_assignments_path",
}
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
    parser.add_argument("--benchmark-root", required=True, type=Path)
    return parser.parse_args()


def method_from_evaluation_name(name: str) -> str | None:
    for token, method in METHOD_TOKENS.items():
        if token in name:
            return method
    return None


def audit_unique_owner(path: Path) -> dict[str, int]:
    frame = pd.read_csv(path, usecols=["barcode", "cell_id"])
    duplicated_rows = int(frame.duplicated(["barcode", "cell_id"]).sum())
    unique_pairs = frame.drop_duplicates(["barcode", "cell_id"])
    owner_counts = unique_pairs.groupby("barcode", sort=False)["cell_id"].nunique()
    return {
        "n_rows": int(len(frame)),
        "n_unique_barcodes": int(frame["barcode"].nunique()),
        "n_duplicate_rows": duplicated_rows,
        "n_barcodes_with_multiple_cell_owners": int((owner_counts > 1).sum()),
        "max_cell_owners_per_barcode": (
            int(owner_counts.max()) if len(owner_counts) else 0
        ),
    }


def main() -> None:
    args = parse_args()
    benchmark_root = args.benchmark_root.resolve()
    evaluation_root = benchmark_root / "evaluations"
    output_dir = benchmark_root / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, tuple[Path, dict]] = {}
    for summary_path in sorted(evaluation_root.glob("*/summary.json")):
        method = method_from_evaluation_name(summary_path.parent.name)
        if method is None:
            continue
        if method in summaries:
            raise ValueError(
                f"Duplicate formal evaluation for {method}: "
                f"{summaries[method][0]} and {summary_path}"
            )
        summaries[method] = (summary_path, json.loads(summary_path.read_text()))

    missing = [method for method in METHOD_ORDER if method not in summaries]
    if missing:
        raise ValueError(f"Incomplete formal method matrix; missing={missing}")

    rows: list[dict] = []
    audit_rows: list[dict] = []
    for display_order, method in enumerate(METHOD_ORDER):
        summary_path, summary = summaries[method]
        assignment_value = summary.get(ASSIGNMENT_KEYS[method])
        if not assignment_value:
            raise ValueError(
                f"{summary_path} lacks assignment key {ASSIGNMENT_KEYS[method]}"
            )
        assignment_path = Path(assignment_value).resolve()
        if not assignment_path.exists():
            raise FileNotFoundError(assignment_path)
        rows.append(
            {
                "method": method,
                "display_order": display_order,
                "evaluation_dir": str(summary_path.parent.resolve()),
                "assignment_path": str(assignment_path),
                **{metric: summary.get(metric) for metric in METRICS},
                "n_matched_pred": summary.get("n_matched_pred"),
                "n_matched_gt": summary.get("n_matched_gt"),
            }
        )
        audit_rows.append(
            {
                "method": method,
                "assignment_path": str(assignment_path),
                **audit_unique_owner(assignment_path),
            }
        )

    comparison = pd.DataFrame(rows).sort_values("display_order")
    comparison["iou_rank"] = comparison["mean_pred_iou"].rank(
        ascending=False, method="min"
    ).astype("Int64")
    comparison["fractional_iou_rank"] = comparison[
        "mean_pred_fractional_iou"
    ].rank(ascending=False, method="min").astype("Int64")
    comparison["gene_spearman_rank"] = comparison["mean_gene_spearman_r"].rank(
        ascending=False, method="min"
    ).astype("Int64")
    audit = pd.DataFrame(audit_rows)
    all_unique = bool(
        (audit["n_barcodes_with_multiple_cell_owners"] == 0).all()
        and (audit["max_cell_owners_per_barcode"] <= 1).all()
    )
    if not all_unique:
        failed = audit.loc[
            (audit["n_barcodes_with_multiple_cell_owners"] > 0)
            | (audit["max_cell_owners_per_barcode"] > 1),
            "method",
        ].tolist()
        raise ValueError(f"Final assignments violate unique ownership: {failed}")

    comparison_path = output_dir / "all_methods.csv"
    audit_path = output_dir / "unique_owner_audit.csv"
    summary_path = output_dir / "summary.json"
    comparison.to_csv(comparison_path, index=False)
    audit.to_csv(audit_path, index=False)
    best_iou = comparison.sort_values("mean_pred_iou", ascending=False).iloc[0]
    result = {
        "benchmark_root": str(benchmark_root),
        "comparison_csv": str(comparison_path),
        "unique_owner_audit_csv": str(audit_path),
        "n_completed_methods": len(comparison),
        "expected_methods": list(METHOD_ORDER),
        "all_assignments_unique_owner": all_unique,
        "best_mean_iou_method": str(best_iou["method"]),
        "best_mean_iou": float(best_iou["mean_pred_iou"]),
    }
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
