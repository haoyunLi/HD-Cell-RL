#!/usr/bin/env python3
"""Summarize frozen-EM generalization across disjoint matched patch splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


METHOD_ORDER = ("nucleus_only", "distance_only", "bin2cell", "stcs", "frozen_em")
EVAL_METHODS = {
    "hd_cell_rl_nucleus_only": "nucleus_only",
    "bin2cell": "bin2cell",
    "stcs": "stcs",
}
METRIC_COLUMNS = ("iou", "precision", "recall", "fractional_iou")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--validation-benchmark-root", required=True, type=Path)
    parser.add_argument("--validation-patch-index", required=True, type=Path)
    parser.add_argument("--validation-em-run", required=True, type=Path)
    parser.add_argument("--validation-distance-run", required=True, type=Path)
    parser.add_argument(
        "--heldout",
        action="append",
        nargs=4,
        required=True,
        metavar=("NAME", "SELECTION_DIR", "EM_RUN", "DISTANCE_RUN"),
    )
    parser.add_argument("--capture-validation-em-run", type=Path, default=None)
    parser.add_argument("--capture-heldout-em-run", type=Path, default=None)
    parser.add_argument("--capture-dataset-summary", type=Path, default=None)
    return parser.parse_args()


def _normalise_cell_id(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = values.astype("string")
    integer = numeric.notna() & np.isclose(numeric, np.round(numeric))
    result.loc[integer] = numeric.loc[integer].round().astype("Int64").astype("string")
    return result


def standardize_cell_metrics(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Return one row per physical cell with common metric names."""
    if "cell_id" not in frame.columns:
        raise ValueError(f"{source} lacks cell_id")
    prefix = "pred_" if "pred_iou" in frame.columns else ""
    if f"{prefix}iou" not in frame.columns:
        raise ValueError(f"{source} lacks an IoU column")
    out = pd.DataFrame({"cell_id": _normalise_cell_id(frame["cell_id"])})
    for metric in METRIC_COLUMNS:
        candidate = f"{prefix}{metric}"
        out[metric] = (
            pd.to_numeric(frame[candidate], errors="coerce")
            if candidate in frame.columns
            else np.nan
        )
    out = out.loc[out["cell_id"].notna()].copy()
    if bool(out["cell_id"].duplicated().any()):
        raise ValueError(f"{source} contains duplicate cell IDs")
    return out.reset_index(drop=True)


def _load_sweep_cells(
    run_dir: Path,
    *,
    expression_weight: float,
    reliability_mode: str | None = None,
) -> pd.DataFrame:
    path = run_dir.resolve() / "per_cell.csv"
    frame = pd.read_csv(path)
    keep = np.isclose(
        pd.to_numeric(frame["expression_weight"], errors="coerce"),
        float(expression_weight),
        rtol=0.0,
        atol=1.0e-12,
    )
    if reliability_mode is not None:
        if "cell_profile_reliability_mode" not in frame.columns:
            raise ValueError(f"{path} lacks cell_profile_reliability_mode")
        keep &= frame["cell_profile_reliability_mode"].astype(str).eq(
            str(reliability_mode)
        )
    selected = frame.loc[keep].copy()
    variants = selected["variant"].astype(str).unique().tolist()
    if len(variants) != 1:
        raise ValueError(
            f"expected one variant in {path}, found {variants} for "
            f"expression_weight={expression_weight}, reliability={reliability_mode}"
        )
    return standardize_cell_metrics(selected, source=str(path))


def _load_eval_methods(evaluation_root: Path) -> dict[str, pd.DataFrame]:
    methods: dict[str, pd.DataFrame] = {}
    for summary_path in sorted(evaluation_root.resolve().glob("*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        method = EVAL_METHODS.get(str(payload.get("method")))
        if method is None:
            continue
        if method in methods:
            raise ValueError(f"duplicate {method} evaluations under {evaluation_root}")
        per_cell_path = summary_path.parent / "per_episode.csv"
        methods[method] = standardize_cell_metrics(
            pd.read_csv(per_cell_path), source=str(per_cell_path)
        )
    return methods


def _require_same_cells(
    *, split: str, method_frames: dict[str, pd.DataFrame]
) -> None:
    expected: set[str] | None = None
    for method in METHOD_ORDER:
        if method not in method_frames:
            raise ValueError(f"split {split} is missing method {method}")
        observed = set(method_frames[method]["cell_id"].astype(str))
        if expected is None:
            expected = observed
        elif observed != expected:
            raise ValueError(
                f"split {split} has mismatched cells for {method}: "
                f"expected={len(expected)}, observed={len(observed)}, "
                f"missing={len(expected - observed)}, extra={len(observed - expected)}"
            )


def _cell_summary_rows(
    *, split: str, role: str, method_frames: dict[str, pd.DataFrame]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        frame = method_frames[method]
        row: dict[str, Any] = {
            "split": split,
            "split_role": role,
            "method": method,
            "n_cells": int(len(frame)),
        }
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(frame[metric], errors="coerce")
            finite = values[np.isfinite(values.to_numpy(dtype=np.float64))]
            row[f"mean_{metric}"] = float(finite.mean()) if len(finite) else np.nan
            row[f"median_{metric}"] = float(finite.median()) if len(finite) else np.nan
            row[f"std_cell_{metric}"] = (
                float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
            )
        rows.append(row)
    return rows


def _paired_rows(
    *, split: str, method_frames: dict[str, pd.DataFrame]
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    em = method_frames["frozen_em"].set_index("cell_id")
    rows: list[dict[str, Any]] = []
    wide = pd.DataFrame(index=em.index)
    wide.index.name = "cell_id"
    for method in METHOD_ORDER:
        frame = method_frames[method].set_index("cell_id")
        wide[f"{method}_iou"] = frame.loc[wide.index, "iou"]
    for comparator in METHOD_ORDER:
        if comparator == "frozen_em":
            continue
        delta = wide["frozen_em_iou"] - wide[f"{comparator}_iou"]
        finite = delta[np.isfinite(delta.to_numpy(dtype=np.float64))]
        rows.append(
            {
                "split": split,
                "comparison": f"frozen_em_minus_{comparator}",
                "n_cells": int(len(finite)),
                "mean_iou_delta": float(finite.mean()),
                "median_iou_delta": float(finite.median()),
                "fraction_em_better": float((finite > 1.0e-12).mean()),
                "fraction_tied": float((np.abs(finite) <= 1.0e-12).mean()),
                "fraction_em_worse": float((finite < -1.0e-12).mean()),
            }
        )
    wide.insert(0, "split", split)
    return rows, wide.reset_index()


def _cell_ids(raw: Any) -> set[str]:
    parsed = json.loads(str(raw))
    if not isinstance(parsed, list):
        raise ValueError("patch cell IDs must be JSON arrays")
    return {str(value) for value in parsed}


def _load_em_barcodes(run_dir: Path) -> set[str]:
    paths = sorted(run_dir.resolve().glob("variants/*/em_assignment/*.npz"))
    if not paths:
        raise FileNotFoundError(f"no EM artifacts under {run_dir}")
    # A frozen run must contain one selected variant. Reliability experiments can
    # contain several, so retain only the baseline nuclear-depth variant.
    preferred = [path for path in paths if "reliability_" not in str(path.parent.parent.name)]
    use_paths = preferred or paths
    result: set[str] = set()
    for path in use_paths:
        with np.load(path, allow_pickle=False) as data:
            result.update(str(value) for value in data["barcode_ids"].tolist())
    return result


def _rectangles_overlap(left: pd.Series, right: pd.Series) -> bool:
    return not (
        float(left["context_x_max"]) < float(right["context_x_min"])
        or float(left["context_x_min"]) > float(right["context_x_max"])
        or float(left["context_y_max"]) < float(right["context_y_min"])
        or float(left["context_y_min"]) > float(right["context_y_max"])
    )


def audit_patch_groups(
    groups: dict[str, tuple[pd.DataFrame, set[str]]]
) -> pd.DataFrame:
    """Audit pairwise patch, context-cell, barcode, and rectangle overlap."""
    rows: list[dict[str, Any]] = []
    names = list(groups)
    for left_index, left_name in enumerate(names):
        left, left_barcodes = groups[left_name]
        left_patches = set(left["patch_id"].astype(str))
        left_cells = set().union(*(_cell_ids(value) for value in left["patch_cell_ids"]))
        for right_name in names[left_index + 1 :]:
            right, right_barcodes = groups[right_name]
            right_patches = set(right["patch_id"].astype(str))
            right_cells = set().union(
                *(_cell_ids(value) for value in right["patch_cell_ids"])
            )
            rectangle_overlap = any(
                _rectangles_overlap(left_row, right_row)
                for _, left_row in left.iterrows()
                for _, right_row in right.iterrows()
            )
            rows.append(
                {
                    "left_split": left_name,
                    "right_split": right_name,
                    "patch_id_overlap": int(len(left_patches & right_patches)),
                    "context_cell_id_overlap": int(len(left_cells & right_cells)),
                    "em_barcode_overlap": int(len(left_barcodes & right_barcodes)),
                    "context_rectangle_overlap": bool(rectangle_overlap),
                }
            )
    return pd.DataFrame(rows)


def _load_capture_row(run_dir: Path) -> dict[str, Any]:
    payload = json.loads((run_dir.resolve() / "summary.json").read_text(encoding="utf-8"))
    row = dict(payload["best_by_validation_mean_core_cell_iou"])
    return {
        "run_dir": str(run_dir.resolve()),
        "mean_core_cell_iou": float(row["mean_core_cell_iou"]),
        "owner_accuracy": float(row["owner_accuracy"]),
        "mean_boundary_f1": float(row["mean_boundary_f1"]),
        "mean_normalized_entropy": float(row["mean_normalized_entropy"]),
    }


def _markdown_table(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    selected = frame.loc[:, list(columns)].copy()
    for column in selected.columns:
        if pd.api.types.is_float_dtype(selected[column]):
            selected[column] = selected[column].map(
                lambda value: "" if not np.isfinite(value) else f"{value:.4f}"
            )
    headers = [str(column) for column in selected.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    experiment_root = args.experiment_root.resolve()
    comparison_dir = experiment_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    validation_methods = _load_eval_methods(
        args.validation_benchmark_root.resolve() / "evaluations"
    )
    validation_methods["distance_only"] = _load_sweep_cells(
        args.validation_distance_run,
        expression_weight=0.0,
    )
    validation_methods["frozen_em"] = _load_sweep_cells(
        args.validation_em_run,
        expression_weight=1.5,
        reliability_mode="nuclear_depth",
    )

    all_methods: dict[str, dict[str, pd.DataFrame]] = {
        "validation": validation_methods
    }
    selections: dict[str, tuple[Path, Path]] = {
        "validation": (args.validation_patch_index.resolve(), args.validation_em_run.resolve())
    }
    for name, selection_dir, em_run, distance_run in args.heldout:
        if name in all_methods:
            raise ValueError(f"duplicate split name: {name}")
        methods = _load_eval_methods(experiment_root / "evaluations" / name)
        methods["distance_only"] = _load_sweep_cells(
            Path(distance_run), expression_weight=0.0
        )
        methods["frozen_em"] = _load_sweep_cells(
            Path(em_run),
            expression_weight=1.5,
            reliability_mode="nuclear_depth",
        )
        all_methods[name] = methods
        selections[name] = (
            Path(selection_dir).resolve() / "test_patches_index.csv",
            Path(em_run).resolve(),
        )

    summary_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    paired_cells: list[pd.DataFrame] = []
    for split, methods in all_methods.items():
        _require_same_cells(split=split, method_frames=methods)
        role = "tuning_reference" if split == "validation" else "heldout"
        summary_rows.extend(
            _cell_summary_rows(split=split, role=role, method_frames=methods)
        )
        if role == "heldout":
            split_pairs, split_cells = _paired_rows(
                split=split, method_frames=methods
            )
            paired_rows.extend(split_pairs)
            paired_cells.append(split_cells)

    by_split = pd.DataFrame(summary_rows)
    by_split["iou_rank_within_split"] = by_split.groupby("split")[
        "mean_iou"
    ].rank(ascending=False, method="min").astype("Int64")
    by_split.to_csv(comparison_dir / "method_comparison_by_split.csv", index=False)

    heldout = by_split.loc[by_split["split_role"] == "heldout"].copy()
    heldout_summary = heldout.groupby("method", as_index=False).agg(
        n_splits=("split", "nunique"),
        total_cells=("n_cells", "sum"),
        mean_of_split_mean_iou=("mean_iou", "mean"),
        std_across_split_mean_iou=("mean_iou", "std"),
        min_split_mean_iou=("mean_iou", "min"),
        max_split_mean_iou=("mean_iou", "max"),
        mean_of_split_mean_precision=("mean_precision", "mean"),
        mean_of_split_mean_recall=("mean_recall", "mean"),
    )
    heldout_summary["rank"] = heldout_summary["mean_of_split_mean_iou"].rank(
        ascending=False, method="min"
    ).astype("Int64")
    heldout_summary = heldout_summary.sort_values("rank")
    heldout_summary.to_csv(comparison_dir / "heldout_method_summary.csv", index=False)

    validation_iou = by_split.loc[by_split["split"] == "validation"].set_index(
        "method"
    )["mean_iou"]
    generalization = heldout_summary.copy()
    generalization["validation_mean_iou"] = generalization["method"].map(
        validation_iou
    )
    generalization["heldout_minus_validation_iou"] = (
        generalization["mean_of_split_mean_iou"]
        - generalization["validation_mean_iou"]
    )
    generalization.to_csv(comparison_dir / "validation_to_heldout.csv", index=False)

    em_diagnostic_rows = [
        {
            "split": "validation",
            "split_role": "tuning_reference",
            **_load_capture_row(args.validation_em_run),
        }
    ]
    em_diagnostic_rows.extend(
        {
            "split": str(name),
            "split_role": "heldout",
            **_load_capture_row(Path(em_run)),
        }
        for name, _, em_run, _ in args.heldout
    )
    em_diagnostics = pd.DataFrame(em_diagnostic_rows)
    em_diagnostics.to_csv(
        comparison_dir / "frozen_em_diagnostics_by_split.csv", index=False
    )

    paired = pd.DataFrame(paired_rows)
    paired.to_csv(comparison_dir / "em_paired_comparisons_by_split.csv", index=False)
    paired_cell_frame = pd.concat(paired_cells, ignore_index=True)
    paired_cell_frame.to_csv(comparison_dir / "paired_cell_iou.csv", index=False)
    pooled_pair_rows: list[dict[str, Any]] = []
    for comparator in METHOD_ORDER:
        if comparator == "frozen_em":
            continue
        delta = (
            paired_cell_frame["frozen_em_iou"]
            - paired_cell_frame[f"{comparator}_iou"]
        )
        finite = delta[np.isfinite(delta.to_numpy(dtype=np.float64))]
        pooled_pair_rows.append(
            {
                "comparison": f"frozen_em_minus_{comparator}",
                "n_cells": int(len(finite)),
                "mean_iou_delta": float(finite.mean()),
                "median_iou_delta": float(finite.median()),
                "fraction_em_better": float((finite > 1.0e-12).mean()),
                "fraction_tied": float((np.abs(finite) <= 1.0e-12).mean()),
                "fraction_em_worse": float((finite < -1.0e-12).mean()),
            }
        )
    pooled_pairs = pd.DataFrame(pooled_pair_rows)
    pooled_pairs.to_csv(
        comparison_dir / "em_paired_comparisons_pooled.csv", index=False
    )

    group_payload: dict[str, tuple[pd.DataFrame, set[str]]] = {}
    for name, (index_path, em_run) in selections.items():
        group_payload[name] = (pd.read_csv(index_path), _load_em_barcodes(em_run))
    leakage = audit_patch_groups(group_payload)
    leakage.to_csv(comparison_dir / "leakage_audit.csv", index=False)
    leakage_clear = bool(
        (leakage["patch_id_overlap"] == 0).all()
        and (leakage["context_cell_id_overlap"] == 0).all()
        and (leakage["em_barcode_overlap"] == 0).all()
        and (~leakage["context_rectangle_overlap"]).all()
    )

    capture_rows: list[dict[str, Any]] = []
    if args.capture_validation_em_run is not None:
        capture_rows.append(
            {"split": "validation", **_load_capture_row(args.capture_validation_em_run)}
        )
    if args.capture_heldout_em_run is not None:
        capture_rows.append(
            {"split": str(args.heldout[0][0]), **_load_capture_row(args.capture_heldout_em_run)}
        )
    capture = pd.DataFrame(capture_rows)
    capture.to_csv(comparison_dir / "capture_thinned_frozen_em.csv", index=False)
    capture_dataset: dict[str, Any] | None = None
    if args.capture_dataset_summary is not None:
        capture_dataset = json.loads(
            args.capture_dataset_summary.resolve().read_text(encoding="utf-8")
        )

    em_heldout = heldout_summary.loc[
        heldout_summary["method"] == "frozen_em"
    ].iloc[0]
    em_validation = float(validation_iou["frozen_em"])
    em_rank_one_all = bool(
        (
            by_split.loc[
                by_split["split_role"] == "heldout"
            ].groupby("split")["iou_rank_within_split"].min()
            == 1
        ).all()
        and (
            by_split.loc[
                (by_split["split_role"] == "heldout")
                & (by_split["method"] == "frozen_em"),
                "iou_rank_within_split",
            ]
            == 1
        ).all()
    )
    result = {
        "experiment_root": str(experiment_root),
        "frozen_em_protocol": {
            "spatial_weight": 1.0,
            "expression_weight": 1.5,
            "cell_profile_relative_prior_strength": 2.0,
            "profile_update_damping": 0.0,
            "cell_profile_compatibility_mode": "standard",
            "cell_profile_reliability_mode": "nuclear_depth",
            "non_nuclear_bin_filter": "positive_expression",
            "background_enabled": False,
        },
        "methods_compared": list(METHOD_ORDER),
        "methods_excluded_by_request": ["smurf"],
        "validation_patch_index": str(args.validation_patch_index.resolve()),
        "validation_em_run": str(args.validation_em_run.resolve()),
        "validation_distance_run": str(args.validation_distance_run.resolve()),
        "heldout_inputs": {
            str(name): {
                "selection_dir": str(Path(selection_dir).resolve()),
                "em_run": str(Path(em_run).resolve()),
                "distance_run": str(Path(distance_run).resolve()),
            }
            for name, selection_dir, em_run, distance_run in args.heldout
        },
        "n_heldout_splits": int(heldout["split"].nunique()),
        "n_heldout_cells_total": int(
            heldout.loc[heldout["method"] == "frozen_em", "n_cells"].sum()
        ),
        "same_pseudo_slide": True,
        "independent_dataset_tested": False,
        "all_patch_cell_barcode_rectangle_overlap_checks_clear": leakage_clear,
        "frozen_em_validation_mean_iou": em_validation,
        "frozen_em_heldout_mean_iou": float(em_heldout["mean_of_split_mean_iou"]),
        "frozen_em_heldout_split_sd": float(em_heldout["std_across_split_mean_iou"]),
        "frozen_em_heldout_minus_validation_iou": float(
            em_heldout["mean_of_split_mean_iou"] - em_validation
        ),
        "frozen_em_rank_one_on_every_heldout_split": em_rank_one_all,
        "capture_dataset": capture_dataset,
    }
    (comparison_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    report_lines = [
        "# Frozen EM multi-split generalization report",
        "",
        "## Result",
        "",
        (
            f"The frozen EM reached mean IoU {float(em_heldout['mean_of_split_mean_iou']):.4f} "
            f"across {int(em_heldout['n_splits'])} held-out splits "
            f"({int(em_heldout['total_cells'])} cells), compared with {em_validation:.4f} on "
            "the four patches used during method development."
        ),
        "",
        (
            "Patch IDs, context cells, EM barcodes, and context rectangles are disjoint "
            "between the development and held-out splits."
            if leakage_clear
            else "At least one leakage audit failed; do not interpret this as a clean held-out result."
        ),
        "",
        "These are new patches and physical cells from the same pseudo slide. The experiment tests "
        "four-patch memorization, not transfer to an independent donor, slide, or generator.",
        "SMURF was excluded from this experiment by request.",
        "",
        "## Held-out method comparison",
        "",
        _markdown_table(
            heldout_summary,
            (
                "rank",
                "method",
                "mean_of_split_mean_iou",
                "std_across_split_mean_iou",
                "min_split_mean_iou",
                "max_split_mean_iou",
            ),
        ),
        "",
        "## Validation versus held-out",
        "",
        _markdown_table(
            generalization,
            (
                "method",
                "validation_mean_iou",
                "mean_of_split_mean_iou",
                "heldout_minus_validation_iou",
            ),
        ),
        "",
        "## Frozen EM diagnostics",
        "",
        _markdown_table(
            em_diagnostics,
            (
                "split",
                "mean_core_cell_iou",
                "owner_accuracy",
                "mean_boundary_f1",
                "mean_normalized_entropy",
            ),
        ),
        "",
        "## Paired cell comparison",
        "",
        _markdown_table(
            pooled_pairs,
            (
                "comparison",
                "mean_iou_delta",
                "fraction_em_better",
                "fraction_tied",
                "fraction_em_worse",
            ),
        ),
    ]
    if len(capture):
        report_lines.extend(
            [
                "",
                "## Capture-thinned check",
                "",
                _markdown_table(
                    capture,
                    (
                        "split",
                        "mean_core_cell_iou",
                        "owner_accuracy",
                        "mean_boundary_f1",
                        "mean_normalized_entropy",
                    ),
                ),
            ]
        )
    report_lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The same-slide result is strong enough to reject the narrow claim that the frozen formula "
            "only works on the original four patches. It is not evidence of donor-level or "
            "dataset-level generalization. That requires a completed independent pseudo dataset with "
            "a separately generated expression realization and a predeclared test protocol.",
            "",
        ]
    )
    (comparison_dir / "report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
