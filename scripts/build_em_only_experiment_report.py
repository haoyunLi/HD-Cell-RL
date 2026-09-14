#!/usr/bin/env python
"""Build the canonical artifact for the colorectal EM-only experiment report."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RUN_PATHS = {
    "bin_universe": "runs/subdiagnostic/em_bin_universe_ablation_20260901T211757Z/method_comparison.csv",
    "type_high": "runs/subdiagnostic/em_weight_sweep_current_positive_expression_20260901T220003Z/weight_sweep.csv",
    "type_capture": "runs/subdiagnostic/em_weight_sweep_capture_thinned_positive_expression_20260901T220003Z/weight_sweep.csv",
    "background_high": "runs/subdiagnostic/em_weight_sweep_current_background_local_20260901T215713Z/weight_sweep.csv",
    "background_capture": "runs/subdiagnostic/em_weight_sweep_capture_thinned_background_20260901T215839Z/weight_sweep.csv",
    "oracle_high": "runs/subdiagnostic/em_oracle_q_current_positive_expression_20260901T220250Z/weight_sweep.csv",
    "profile_high": "runs/subdiagnostic/em_nuclear_profile_current_prior5000_20260901T221534Z/weight_sweep.csv",
    "profile_capture": "runs/subdiagnostic/em_nuclear_profile_capture_thinned_prior5000_extended_20260901T221534Z/weight_sweep.csv",
    "external_methods": "runs/benchmarks/colorectal_nucleus_based_multi_owner_v2_43cells_20260826T210246Z/stress_tests/capture_depth_20260826T230115Z/comparison/capture_depth_all_methods_long.csv",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    return parser.parse_args()


def _read(repo_root: Path, key: str) -> pd.DataFrame:
    path = repo_root / RUN_PATHS[key]
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _at_beta(frame: pd.DataFrame, beta: float) -> pd.Series:
    selected = frame.loc[
        np.isclose(frame["spatial_weight"].astype(float), 1.0)
        & np.isclose(frame["expression_weight"].astype(float), float(beta))
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one alpha=1, beta={beta} row; found {len(selected)}")
    return selected.iloc[0]


def _metric_row(
    *,
    method: str,
    condition: str,
    row: pd.Series,
    model_scope: str,
) -> dict[str, Any]:
    return {
        "method": method,
        "condition": condition,
        "model_scope": model_scope,
        "expression_weight": float(row.get("expression_weight", np.nan)),
        "mean_iou": float(row["mean_core_cell_iou"]),
        "mean_precision": float(row["mean_core_cell_precision"]),
        "mean_recall": float(row["mean_core_cell_recall"]),
        "mean_boundary_f1": float(row.get("mean_boundary_f1", np.nan)),
        "boundary_owner_accuracy": float(row.get("gt_boundary_owner_accuracy", np.nan)),
        "mean_absolute_area_bias": float(
            row.get("mean_absolute_area_bias_fraction", np.nan)
        ),
        "hard_assignment_coverage": float(row.get("hard_assignment_coverage", 1.0)),
    }


def _source(
    source_id: str,
    label: str,
    path: str,
    description: str,
    *,
    executed_at: str,
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "description": description,
        "query": {
            "engine": "DuckDB",
            "language": "sql",
            "sql": f"SELECT * FROM read_csv_auto('{path}', header = true);",
            "description": description,
            "executed_at": executed_at,
            "tables_used": [path],
            "filters": [
                "Four fixed colorectal nucleus-based multi-owner v2 patches",
                "43 core cells for macro cell metrics",
                "Margin cells compete but are not included in macro averages",
                "GT is joined only after normal EM inference except in the named oracle-q diagnostic",
            ],
            "metric_definitions": [
                "Mean IoU is the unweighted mean of per-core-cell intersection over union.",
                "Hard-assignment coverage is the fraction of the stated physical-bin universe with a non-background hard owner.",
                "Boundary owner accuracy is micro accuracy on dominant-owner GT bins marked as boundary.",
                "Boundary F1 is the macro exact-bin F1 between predicted 8-neighbor boundaries and GT boundary bins.",
                "Mean absolute area bias is the macro mean of abs((predicted bins - GT bins) / GT bins).",
            ],
        },
    }


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return JSON-safe records while preserving missing optional metrics as null."""
    cleaned = frame.astype(object).where(pd.notna(frame), None)
    return cleaned.to_dict(orient="records")


def _card(
    card_id: str,
    label: str,
    field: str,
    description: str,
    source_id: str,
    *,
    comparison_field: str | None = None,
    comparison_label: str | None = None,
    comparison_format: str = "percent",
    comparison_unit: str | None = None,
) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = [
        {"label": label, "field": field, "format": "percent"}
    ]
    if comparison_field is not None:
        comparison: dict[str, Any] = {
            "label": str(comparison_label),
            "field": comparison_field,
            "format": comparison_format,
            "signed": True,
        }
        if comparison_unit is not None:
            comparison["unit"] = comparison_unit
        metrics.append(comparison)
    return {
        "id": card_id,
        "description": description,
        "dataset": "headline",
        "sourceId": source_id,
        "metrics": metrics,
    }


def main() -> None:
    args = _parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    data_dir = output_dir / "data"
    data_dir.mkdir()

    ablation = _read(repo_root, "bin_universe")
    type_high = _read(repo_root, "type_high")
    type_capture = _read(repo_root, "type_capture")
    background_high = _read(repo_root, "background_high")
    background_capture = _read(repo_root, "background_capture")
    oracle_high = _read(repo_root, "oracle_high")
    profile_high = _read(repo_root, "profile_high")
    profile_capture = _read(repo_root, "profile_capture")
    external = _read(repo_root, "external_methods")

    variant_labels = {
        "A_forced_owner": "A · forced owner",
        "B_positive_expression": "B · positive-expression bins",
        "C_background_default": "C · confidence-gated background",
    }
    ablation_out = ablation.copy()
    ablation_out.insert(
        1,
        "variant_label",
        ablation_out["variant"].map(variant_labels).fillna(ablation_out["variant"]),
    )
    ablation_out.to_csv(data_dir / "bin_universe_ablation.csv", index=False)
    ablation_long = ablation_out[
        ["variant_label", "mean_core_cell_iou", "hard_assignment_coverage"]
    ].melt(
        id_vars="variant_label",
        var_name="metric_key",
        value_name="value",
    )
    ablation_long["metric"] = ablation_long["metric_key"].map(
        {
            "mean_core_cell_iou": "Mean core-cell IoU",
            "hard_assignment_coverage": "Hard-assignment coverage",
        }
    )

    sweep_frames: list[pd.DataFrame] = []
    for condition, frame in (
        ("Current high depth", type_high),
        ("Capture-thinned", type_capture),
    ):
        selected = frame.loc[np.isclose(frame["spatial_weight"].astype(float), 1.0)].copy()
        selected.insert(0, "condition", condition)
        sweep_frames.append(selected)
    type_sweep = pd.concat(sweep_frames, ignore_index=True)
    type_sweep.to_csv(data_dir / "type_level_weight_sweep.csv", index=False)

    background_rows: list[dict[str, Any]] = []
    for condition, rule, frame in (
        ("Current high depth", "Positive-expression universe", type_high),
        ("Current high depth", "Confidence-gated background", background_high),
        ("Capture-thinned", "Positive-expression universe", type_capture),
        ("Capture-thinned", "Confidence-gated background", background_capture),
    ):
        row = _at_beta(frame, 1.5)
        background_rows.append(
            {
                "condition": condition,
                "assignment_rule": rule,
                "expression_weight": 1.5,
                "mean_iou": float(row["mean_core_cell_iou"]),
                "hard_assignment_coverage": float(row["hard_assignment_coverage"]),
                "n_unassigned": int(row["n_unassigned"]),
                "mean_background_probability": float(row["mean_background_probability"]),
            }
        )
    background_comparison = pd.DataFrame(background_rows)
    background_comparison.to_csv(data_dir / "background_depth_sensitivity.csv", index=False)

    depth_rows: list[dict[str, Any]] = []
    for condition, type_frame, profile_frame, external_condition in (
        ("Current high depth", type_high, profile_high, "high_depth_local"),
        ("Capture-thinned", type_capture, profile_capture, "capture_thinned"),
    ):
        depth_rows.extend(
            [
                _metric_row(
                    method="Distance only",
                    condition=condition,
                    row=_at_beta(type_frame, 0.0),
                    model_scope="Positive-expression universe",
                ),
                _metric_row(
                    method="Type-level EM",
                    condition=condition,
                    row=_at_beta(type_frame, 1.5),
                    model_scope="alpha=1, beta=1.5",
                ),
                _metric_row(
                    method="Cell-specific nuclear profile",
                    condition=condition,
                    row=_at_beta(profile_frame, 1.5),
                    model_scope="Fixed profile diagnostic; prior_umis=5000",
                ),
            ]
        )
        for method, label in (("stcs", "STCS"), ("bin2cell", "Bin2Cell")):
            rows = external.loc[
                (external["condition"].astype(str) == external_condition)
                & (external["method"].astype(str) == method)
                & (external["status"].astype(str) == "completed")
            ]
            if len(rows) != 1:
                raise ValueError(
                    f"expected one completed external row for {external_condition}/{method}"
                )
            row = rows.iloc[0]
            depth_rows.append(
                {
                    "method": label,
                    "condition": condition,
                    "model_scope": "Native method output on the same 43-cell benchmark",
                    "expression_weight": np.nan,
                    "mean_iou": float(row["mean_pred_iou"]),
                    "mean_precision": float(row["mean_pred_precision"]),
                    "mean_recall": float(row["mean_pred_recall"]),
                    "mean_boundary_f1": np.nan,
                    "boundary_owner_accuracy": np.nan,
                    "mean_absolute_area_bias": np.nan,
                    "hard_assignment_coverage": np.nan,
                }
            )
    architecture_depth = pd.DataFrame(depth_rows)
    architecture_depth.to_csv(data_dir / "architecture_depth_comparison.csv", index=False)

    diagnostic_rows = [
        _metric_row(
            method="Learned type-level q",
            condition="Current high depth",
            row=_at_beta(type_high, 1.5),
            model_scope="Standard alternating generalized EM",
        ),
        _metric_row(
            method="Oracle one-hot type q",
            condition="Current high depth",
            row=_at_beta(oracle_high, 1.5),
            model_scope="GT type used only for a fixed-q diagnostic",
        ),
        _metric_row(
            method="Cell-specific nuclear profile",
            condition="Current high depth",
            row=_at_beta(profile_high, 1.5),
            model_scope="Fixed profile diagnostic; prior_umis=5000",
        ),
    ]
    ext_bin2cell = architecture_depth.loc[
        (architecture_depth["condition"] == "Current high depth")
        & (architecture_depth["method"] == "Bin2Cell")
    ].iloc[0]
    diagnostic_rows.append(ext_bin2cell.to_dict())
    architecture_diagnostic = pd.DataFrame(diagnostic_rows)
    architecture_diagnostic.to_csv(
        data_dir / "architecture_bottleneck_diagnostic.csv", index=False
    )

    robust_metrics = architecture_depth.loc[
        architecture_depth["method"].isin(
            ["Type-level EM", "Cell-specific nuclear profile", "Bin2Cell", "STCS"]
        )
    ].copy()
    robust_metrics.to_csv(data_dir / "robust_depth_metrics.csv", index=False)

    b_row = ablation_out.loc[ablation_out["variant"] == "B_positive_expression"].iloc[0]
    a_row = ablation_out.loc[ablation_out["variant"] == "A_forced_owner"].iloc[0]
    high_type_best = type_high.sort_values("mean_core_cell_iou", ascending=False).iloc[0]
    capture_type_best = type_capture.sort_values("mean_core_cell_iou", ascending=False).iloc[0]
    high_profile_robust = _at_beta(profile_high, 1.5)
    capture_profile_robust = _at_beta(profile_capture, 1.5)
    capture_bin2cell = architecture_depth.loc[
        (architecture_depth["condition"] == "Capture-thinned")
        & (architecture_depth["method"] == "Bin2Cell")
    ].iloc[0]
    headline = [
        {
            "forced_owner_iou": float(a_row["mean_core_cell_iou"]),
            "positive_expression_iou": float(b_row["mean_core_cell_iou"]),
            "bin_universe_gain_pp": float(
                100.0 * (b_row["mean_core_cell_iou"] - a_row["mean_core_cell_iou"])
            ),
            "type_level_best_high_iou": float(high_type_best["mean_core_cell_iou"]),
            "type_level_best_capture_iou": float(
                capture_type_best["mean_core_cell_iou"]
            ),
            "profile_high_iou": float(high_profile_robust["mean_core_cell_iou"]),
            "profile_capture_iou": float(capture_profile_robust["mean_core_cell_iou"]),
            "profile_minus_bin2cell_high_pp": float(
                100.0
                * (
                    high_profile_robust["mean_core_cell_iou"]
                    - ext_bin2cell["mean_iou"]
                )
            ),
            "profile_minus_bin2cell_capture_pp": float(
                100.0
                * (
                    capture_profile_robust["mean_core_cell_iou"]
                    - capture_bin2cell["mean_iou"]
                )
            ),
        }
    ]
    pd.DataFrame(headline).to_csv(data_dir / "headline.csv", index=False)

    created_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    sources = [
        _source(
            "headline",
            "EM-only headline metrics",
            "data/headline.csv",
            "Selected headline values derived from the reviewed ablation and cross-depth tables.",
            executed_at=created_at,
        ),
        _source(
            "bin_universe",
            "EM bin-universe and background ablation",
            "data/bin_universe_ablation.csv",
            "A/B/C results on four fixed patches and 43 core cells.",
            executed_at=created_at,
        ),
        _source(
            "type_sweep",
            "Full type-level generalized-EM weight sweeps",
            "data/type_level_weight_sweep.csv",
            "Every weight reruns the complete batch E/M loop on current and capture-thinned expression.",
            executed_at=created_at,
        ),
        _source(
            "background_sensitivity",
            "Background depth-sensitivity comparison",
            "data/background_depth_sensitivity.csv",
            "Matched alpha=1 and beta=1.5 comparison of positive-expression and confidence-gated background rules.",
            executed_at=created_at,
        ),
        _source(
            "architecture_diagnostic",
            "EM architecture bottleneck diagnostic",
            "data/architecture_bottleneck_diagnostic.csv",
            "High-depth comparison of learned q, oracle type q, physical-cell profile, and Bin2Cell.",
            executed_at=created_at,
        ),
        _source(
            "depth_comparison",
            "Cross-depth EM and external-method comparison",
            "data/architecture_depth_comparison.csv",
            "Current high-depth and capture-thinned results on the same 43-cell benchmark.",
            executed_at=created_at,
        ),
        _source(
            "robust_metrics",
            "Exact cross-depth metrics",
            "data/robust_depth_metrics.csv",
            "IoU, precision, recall, boundary, area-bias, and coverage fields where available.",
            executed_at=created_at,
        ),
    ]

    cards = [
        _card(
            "bin_universe_gain",
            "IoU after fixing forced ownership",
            "positive_expression_iou",
            "Positive-expression bin universe; the companion value is the gain over forced ownership.",
            "headline",
            comparison_field="bin_universe_gain_pp",
            comparison_label="absolute IoU gain (pp)",
            comparison_format="number",
            comparison_unit=" pp",
        ),
        _card(
            "type_best_high",
            "Best type-level EM · high depth",
            "type_level_best_high_iou",
            "Validation maximum from the full alpha=1 expression-weight sweep.",
            "headline",
        ),
        _card(
            "profile_high",
            "Cell-specific profile · high depth",
            "profile_high_iou",
            "Fixed-profile diagnostic at alpha=1, beta=1.5, prior_umis=5000.",
            "headline",
            comparison_field="profile_minus_bin2cell_high_pp",
            comparison_label="IoU delta vs Bin2Cell (pp)",
            comparison_format="number",
            comparison_unit=" pp",
        ),
        _card(
            "profile_capture",
            "Cell-specific profile · capture-thinned",
            "profile_capture_iou",
            "Same fixed-profile setting under capture thinning.",
            "headline",
            comparison_field="profile_minus_bin2cell_capture_pp",
            comparison_label="IoU delta vs Bin2Cell (pp)",
            comparison_format="number",
            comparison_unit=" pp",
        ),
    ]

    charts = [
        {
            "id": "bin_universe_chart",
            "title": "Bin-universe ablation: IoU and assignment coverage",
            "subtitle": "Four fixed patches, 43 core cells; coverage is relative to 4,390 EM-scored physical bins.",
            "showDescription": True,
            "intent": "comparison",
            "question": "How much error comes from forcing every in-radius physical bin to have a cell owner?",
            "rationale": "Grouped bars show the accuracy-versus-coverage tradeoff for all three ownership-universe rules.",
            "comparisonContext": {
                "grain": "Three EM bin-universe variants",
                "unit": "fraction",
                "baseline": "A · forced owner",
            },
            "type": "bar",
            "dataset": "bin_universe_long",
            "sourceId": "bin_universe",
            "encodings": {
                "x": {"field": "variant_label", "type": "nominal", "label": "Variant"},
                "y": {"field": "value", "type": "quantitative", "format": "percent", "label": "Fraction"},
                "color": {"field": "metric", "type": "nominal", "label": "Metric"},
            },
            "valueFormat": "percent",
            "layout": "full",
        },
        {
            "id": "background_depth_chart",
            "title": "Assignment-universe rule under capture thinning",
            "subtitle": "Matched alpha=1 and beta=1.5; the confidence-only background gate rejects more bins as counts fall.",
            "showDescription": True,
            "intent": "comparison",
            "question": "Is the first confidence-gated background component stable across capture depth?",
            "rationale": "Grouped bars compare the same rule and weight across the two count-depth conditions.",
            "comparisonContext": {
                "grain": "Two assignment rules by two depth conditions",
                "unit": "mean core-cell IoU",
                "baseline": "Positive-expression universe",
            },
            "type": "bar",
            "dataset": "background_comparison",
            "sourceId": "background_sensitivity",
            "encodings": {
                "x": {"field": "condition", "type": "nominal", "label": "Expression condition"},
                "y": {"field": "mean_iou", "type": "quantitative", "format": "percent", "label": "Mean IoU"},
                "color": {"field": "assignment_rule", "type": "nominal", "label": "Assignment rule"},
                "tooltip": [
                    {"field": "hard_assignment_coverage", "type": "quantitative", "format": "percent", "label": "Coverage"},
                    {"field": "n_unassigned", "type": "quantitative", "format": "number", "label": "Unassigned bins"},
                ],
            },
            "valueFormat": "percent",
            "layout": "full",
        },
        {
            "id": "type_weight_chart",
            "title": "Type-level generalized-EM expression-weight sweep",
            "subtitle": "Full E/M rerun at every point; spatial weight fixed at 1.0.",
            "showDescription": True,
            "intent": "custom",
            "question": "Was the original expression weight 0.25 too small, and is one region stable across depth?",
            "rationale": "The ordered curves expose the optimum and the cross-depth shift without treating the scan as a time trend.",
            "comparisonContext": {
                "grain": "One converged full-EM run per weight and depth condition",
                "unit": "mean core-cell IoU",
                "baseline": "expression_weight=0",
            },
            "type": "line",
            "dataset": "type_sweep",
            "sourceId": "type_sweep",
            "encodings": {
                "x": {"field": "expression_weight", "type": "quantitative", "label": "Expression weight"},
                "y": {"field": "mean_core_cell_iou", "type": "quantitative", "format": "percent", "label": "Mean IoU"},
                "color": {"field": "condition", "type": "nominal", "label": "Expression condition"},
                "tooltip": [
                    {"field": "gt_boundary_owner_accuracy", "type": "quantitative", "format": "percent", "label": "Boundary owner accuracy"},
                    {"field": "mean_absolute_area_bias_fraction", "type": "quantitative", "format": "percent", "label": "Mean absolute area bias"},
                ],
            },
            "valueFormat": "percent",
            "layout": "full",
        },
        {
            "id": "architecture_diagnostic_chart",
            "title": "High-depth EM architecture diagnostic",
            "subtitle": "Oracle q uses GT type only as an evaluation diagnostic; the physical-cell profile uses no GT during inference.",
            "showDescription": True,
            "intent": "comparison",
            "question": "Is the remaining bottleneck cell-type posterior estimation or physical-cell discrimination?",
            "rationale": "A direct bar comparison isolates the effect of replacing type-level compatibility with a cell-specific nuclear profile.",
            "comparisonContext": {
                "grain": "Four methods on the same 43 core cells",
                "unit": "mean core-cell IoU",
                "baseline": "Learned type-level q",
            },
            "type": "bar",
            "dataset": "architecture_diagnostic",
            "sourceId": "architecture_diagnostic",
            "encodings": {
                "x": {"field": "method", "type": "nominal", "label": "Compatibility model"},
                "y": {"field": "mean_iou", "type": "quantitative", "format": "percent", "label": "Mean IoU"},
                "tooltip": [
                    {"field": "mean_precision", "type": "quantitative", "format": "percent", "label": "Precision"},
                    {"field": "mean_recall", "type": "quantitative", "format": "percent", "label": "Recall"},
                ],
            },
            "valueFormat": "percent",
            "layout": "full",
        },
        {
            "id": "depth_comparison_chart",
            "title": "Method performance across expression-depth conditions",
            "subtitle": "Same four patches and 43 core cells; external tools retain their native output rules.",
            "showDescription": True,
            "intent": "comparison",
            "question": "Does the strongest EM architecture remain competitive when expression is capture-thinned?",
            "rationale": "Grouped bars make the cross-depth robustness gap visible for each method.",
            "comparisonContext": {
                "grain": "Five methods by two expression-depth conditions",
                "unit": "mean core-cell IoU",
                "baseline": "Distance only",
            },
            "type": "bar",
            "dataset": "architecture_depth",
            "sourceId": "depth_comparison",
            "encodings": {
                "x": {"field": "method", "type": "nominal", "label": "Method"},
                "y": {"field": "mean_iou", "type": "quantitative", "format": "percent", "label": "Mean IoU"},
                "color": {"field": "condition", "type": "nominal", "label": "Expression condition"},
            },
            "valueFormat": "percent",
            "layout": "full",
        },
    ]

    tables = [
        {
            "id": "robust_metrics_table",
            "title": "Exact metrics for the cross-depth comparison",
            "subtitle": "Boundary and area metrics are available for the EM runners; external-method rows retain the matched benchmark metrics available in their evaluation table.",
            "showDescription": True,
            "dataset": "robust_metrics",
            "sourceId": "robust_metrics",
            "defaultSort": {"field": "condition", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "condition", "label": "Condition", "type": "text"},
                {"field": "method", "label": "Method", "type": "text"},
                {"field": "mean_iou", "label": "Mean IoU", "format": "percent"},
                {"field": "mean_precision", "label": "Precision", "format": "percent"},
                {"field": "mean_recall", "label": "Recall", "format": "percent"},
                {"field": "mean_boundary_f1", "label": "Boundary F1", "format": "percent"},
                {"field": "boundary_owner_accuracy", "label": "Boundary owner accuracy", "format": "percent"},
                {"field": "mean_absolute_area_bias", "label": "Mean absolute area bias", "format": "percent"},
            ],
        }
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# EM-only 实验：问题定位与下一版方向"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## 结论：先修 ownership universe，再补 physical-cell expression\n\n"
                "这轮实验把当前 EM 的问题分成了两层。第一层是 **强制归属**：389 个零 selected-gene expression、非 nuclear 的 physical bins 被迫分给附近细胞，使 mean core-cell IoU 只有 **0.7018**，平均面积偏差为 **+31.7%**。改为只在 positive-expression 非核 bins 上运行、同时保留全部 locked nuclear seeds 后，IoU 升到 **0.8064**，recall 不变，面积偏差降到 **+5.3%**。\n\n"
                "第二层是 **physical-cell discrimination**。完整 E/M weight sweep 证明原来的 `expression_weight=0.25` 偏小；type-level compatibility 在高深度约于 `beta=1.75` 达到 **0.8399**，capture-thinned 约于 `beta=1.5` 达到 **0.8159**。但把 q 固定成 GT cell type 的 oracle one-hot 后只有 **0.8047**，所以瓶颈不是 cell typing 不准。\n\n"
                "最强诊断结果来自 physical-cell-specific nuclear profile。用 locked nuclear counts 建 profile，并用 scRNA reference 做 shrinkage，在统一 `alpha=1, beta=1.5` 下，高深度 IoU 为 **0.8963**，capture-thinned 为 **0.8440**。这说明需要把 cell-specific expression evidence 加入正式 EM；它现在仍是 fixed-profile ablation，不能直接当作最终模型。"
            ),
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [
                "bin_universe_gain",
                "type_best_high",
                "profile_high",
                "profile_capture",
            ],
        },
        {
            "id": "bin_universe_finding",
            "type": "markdown",
            "sourceId": "bin_universe",
            "body": (
                "## 零表达 bins 的强制归属是第一处系统性误差\n\n"
                "A、B、C 使用相同的 MaxDis、candidate cells、locked nuclear seeds 和 cell-type likelihood。唯一变化是哪些 physical bins 必须获得 owner。B 去掉 389 个零置信度非核 bins；C 保留全部 bins，但允许 confidence-gated background。两者的 IoU 都约为 **0.8064**，而 A 只有 **0.7018**。因此，之前的细胞扩张并不主要来自 E/M 更新，而是来自 `sum_c r[b,c]=1` 对没有表达证据的行也强制成立。\n\n"
                "B 的 coverage 为 **91.14%**，C 的 hard-assignment coverage 为 **91.03%**。这两个结果不能靠只看 precision 解释，必须同时报告 coverage、recall 和 area bias。"
            ),
        },
        {"id": "bin_universe_visual", "type": "chart", "chartId": "bin_universe_chart"},
        {
            "id": "background_finding",
            "type": "markdown",
            "sourceId": "background_sensitivity",
            "body": (
                "## 第一版 confidence-only background 不够稳健\n\n"
                "在 `alpha=1, beta=1.5` 下，confidence-gated background 在高深度与 positive-expression universe 几乎相同（IoU **0.8378** 对 **0.8381**）。capture thinning 后，它把 unassigned bins 从 394 左右扩大到 **535**，IoU 降到 **0.7812**；positive-expression universe 仍为 **0.8159**。\n\n"
                "原因很直接：这个 gate 只看 expression confidence，而 confidence 本身随 capture depth 变化。它会把低 capture 当成 background。当前实现适合作为诊断开关，不适合作为默认 background model。"
            ),
        },
        {"id": "background_visual", "type": "chart", "chartId": "background_depth_chart"},
        {
            "id": "weight_finding",
            "type": "markdown",
            "sourceId": "type_sweep",
            "body": (
                "## 原始 expression weight 偏小，但调 weight 只能解决一部分问题\n\n"
                "在两个 depth conditions 中，`beta=0.25` 都明显低于 1.0–1.75 区间。高深度最佳点是 **1.75**，capture-thinned 最佳点是 **1.5**；两条曲线都在继续增大 beta 后下降。因此，更大的 expression weight 确实让表达证据发挥作用，但不存在越大越好的单调关系。\n\n"
                "`beta=1.5` 是当前较稳健的跨深度工作点，而不是生物学上已优化的默认值。所有点都重新初始化并运行完整 batch E-step/M-step 至收敛，不是 frozen-q screening。"
            ),
        },
        {"id": "weight_visual", "type": "chart", "chartId": "type_weight_chart"},
        {
            "id": "architecture_finding",
            "type": "markdown",
            "sourceId": "architecture_diagnostic",
            "body": (
                "## q 不是 ceiling；缺失的是同类型 physical cells 的表达区分\n\n"
                "把每个 cell 的 q 固定为 GT cell type one-hot 后，IoU 为 **0.8047**，反而低于 learned type-level q 的 **0.8381**。这说明继续训练或锐化 q 不是当前最有效的方向。一个 cell type 下的多个 physical cells 仍得到近似相同的 `q·LL`，E-step 最后只能依赖距离区分它们。\n\n"
                "用 locked nuclear bins 建立每个 physical cell 的表达 profile，再以 scRNA theta 对低证据 profile 做 shrinkage 后，IoU 达到 **0.8963**。这个增益直接回答了架构问题：下一版 EM 应保留 type-level q，同时加入可靠度可调的 cell-specific compatibility。"
            ),
        },
        {
            "id": "architecture_diagnostic_visual",
            "type": "chart",
            "chartId": "architecture_diagnostic_chart",
        },
        {
            "id": "depth_finding",
            "type": "markdown",
            "sourceId": "depth_comparison",
            "body": (
                "## Cell-specific profile 在两种深度下都有效，但 capture-thinned 仍未超过 Bin2Cell\n\n"
                "统一使用 `alpha=1, beta=1.5, prior_umis=5000` 时，cell-specific profile 在高深度为 **0.8963**，高于同一 paired benchmark 的 Bin2Cell **0.8599**；capture-thinned 为 **0.8440**，低于 Bin2Cell **0.8606**，但高于 STCS **0.7422** 和 type-level EM **0.8159**。\n\n"
                "高深度优势不能直接外推到真实数据。当前 pseudo expression 与 nuclear profile 来自高度相关的数据生成过程，可能把 cell-specific signature 做得过清楚；`prior_umis=5000` 也是绝对 count scale，不应直接写进默认配置。"
            ),
        },
        {"id": "depth_visual", "type": "chart", "chartId": "depth_comparison_chart"},
        {
            "id": "metrics_intro",
            "type": "markdown",
            "body": (
                "## 改善同时出现在 IoU、边界和面积偏差\n\n"
                "高深度 cell-specific profile 的 precision/recall 为 **0.9441/0.9474**，boundary F1 为 **0.5500**，mean absolute area bias 为 **0.0612**。capture-thinned 对应为 **0.9147/0.9184**、**0.4988** 和 **0.1012**。这些指标方向一致，说明改善不只是通过扩大或缩小细胞面积获得。"
            ),
        },
        {"id": "metrics_table", "type": "table", "tableId": "robust_metrics_table"},
        {
            "id": "scope",
            "type": "markdown",
            "body": (
                "## 评价范围与定义\n\n"
                "所有 EM 结果来自 colorectal nucleus-based multi-owner v2 的四个固定 patch，共 **43 个 core cells**。每个 patch 的 outer extent 为 **64 × 64 µm**；margin cells 参与 candidate competition，但不进入 macro cell metric。candidate eligibility 复用 episode construction 的 `environment.max_center_distance_um=20 µm`，没有 nearest-3 限制。当前数据中，GT owner 在 MaxDis candidate set 内的 recall 为 100%。\n\n"
                "Cell IoU、precision、recall、boundary F1 和 area bias 只对 43 个 core cells 做 macro average。GT 只用于实验评价、weight selection 和明确标记为 oracle 的 ceiling test，从未用于正常 EM E-step、M-step 或 candidate construction。"
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "body": (
                "## 这轮具体测试了什么\n\n"
                "1. **Bin universe A/B/C。** A 强制所有 valid-candidate bins 有 owner；B 只纳入 positive-expression 非核 bins，并始终保留 locked nuclear seeds；C 为所有 bins 增加显式 background mass。\n"
                "2. **完整 weight sweep。** 每个 `(alpha, beta)` 都从 nuclear initialization 开始，使用同一个 `q^(t)` 完成整 patch E-step，再用完整 `r^(t+1)` 更新全部 cells。旧 iteration evidence 不累积。\n"
                "3. **Oracle-q diagnostic。** 只在高深度条件把 q 固定成 GT cell type one-hot，用来判断 cell typing 是否是性能上限。\n"
                "4. **Cell-specific profile diagnostic。** 每个 physical cell 的 locked nuclear counts 与 `q_seed @ scRNA theta` 混合，形成固定 profile；E-step 使用 bin 对具体 physical cell 的 multinomial LL。GT 不参与 profile 建立。"
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 目前不能下的结论\n\n"
                "- 不能说 fixed nuclear-profile ablation 已经是正式 generalized EM。它在 E-step 中使用固定 cell profile，q 的 M-step 不再驱动这个兼容性项；它的作用是定位缺失信号。\n"
                "- 不能把 `positive_expression` 当作真实组织的最终 background rule。真实细胞内部可以有零 UMI 2 µm bins。\n"
                "- 不能把 `prior_umis=5000` 当作跨数据集常数。scDesign3 与真实 capture depth 的 count scale 不同，绝对 pseudo-count 会随测序深度改变 shrinkage 强度。\n"
                "- 不能根据这四个 patch 宣布 EM 普遍优于 Bin2Cell。当前 pseudo GT 由 nucleus-based expansion 产生，对几何方法有结构性偏好；同源 expression 生成也可能偏向 cell-specific profile。"
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## 推荐的下一版 EM 实验\n\n"
                "1. **把 cell-specific compatibility 正式并入 E-step。** 保留当前 `q·LL` type-level term，再加入 nuclear-profile term。两者的 mixing weight 应由每个 cell 的 nuclear evidence 决定，而不是统一的绝对 UMI pseudo-count。\n"
                "2. **让 shrinkage 对 depth 无量纲。** 用 nuclear selected-gene total、有效基因数或 posterior uncertainty 计算 per-cell reliability；在 high-depth 与 capture-thinned 上共享同一规则。\n"
                "3. **暂不启用 confidence-only background。** 下一版 background 应结合 ambient expression likelihood、到 nucleus/segmentation 的绝对几何和 tissue support；在此之前保留 B 作为诊断 universe，并始终报告 coverage。\n"
                "4. **在 held-out patches/donor 上一次性验证。** 固定候选公式与参数后再看 IoU、boundary、area bias 和 entropy calibration，避免继续在这四个 patch 上选择超参数。\n"
                "5. **最后再校准 entropy。** compatibility 和 background 改变后，responsibility entropy 的尺度也会改变；当前 threshold 不应先行定稿。"
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## 下一轮需要回答的问题\n\n"
                "- 同类型相邻 cells 的 nuclear profiles 在独立 donor 和真实 capture depth 下，是否仍有可重复的 discriminative signal？\n"
                "- per-cell reliability 应由 nuclear UMI、detected genes、q entropy，还是它们的组合决定？\n"
                "- background likelihood 应从 off-cell tissue bins、ambient profile，还是 segmentation uncertainty 学习？\n"
                "- 在统一 observed-bin universe 与 native-output 两种评价口径下，EM 与 Bin2Cell 的排序是否一致？"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "EM-only 实验：问题定位与下一版方向",
            "description": "Generalized-EM bin-universe, weight, depth-robustness, oracle-q, and physical-cell profile diagnostics.",
            "generatedAt": created_at,
            "sources": sources,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": created_at,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "bin_universe_long": _records(ablation_long),
                "background_comparison": _records(background_comparison),
                "type_sweep": _records(type_sweep),
                "architecture_diagnostic": _records(architecture_diagnostic),
                "architecture_depth": _records(architecture_depth),
                "robust_metrics": _records(robust_metrics),
            },
        },
        "sources": sources,
    }
    artifact_path = output_dir / "artifact.json"
    with artifact_path.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

    source_notes = [
        "# EM-only report source notes",
        "",
        "All source paths are repository-relative. The report contains no RL rollout or PPO result.",
        "",
        "## Input runs",
        "",
        *[f"- `{key}`: `{path}`" for key, path in RUN_PATHS.items()],
        "",
        "## Chart map",
        "",
        "- Bin universe: grouped bar, variant × {IoU, hard coverage}; supports the forced-owner diagnosis.",
        "- Background depth sensitivity: grouped bar, depth × assignment rule; supports the depth-confounding diagnosis.",
        "- Type-level weight sweep: two-series ordered line, beta × IoU; supports the weight-range conclusion.",
        "- Architecture diagnostic: categorical bar, compatibility model × IoU; separates q estimation from physical-cell discrimination.",
        "- Cross-depth comparison: grouped bar, method × depth; tests robustness and external-method context.",
        "",
        "## Selection notes",
        "",
        "- The report uses alpha=1, beta=1.5 as the shared cross-depth point where a fixed comparison is required.",
        "- The high-depth and capture-thinned type-level maxima are reported as validation maxima, not held-out estimates.",
        "- Oracle q is a GT-assisted diagnostic only. GT is otherwise joined after inference for evaluation.",
        "- Cell-specific nuclear-profile rows are fixed-profile ablations with scRNA-reference shrinkage, not the final alternating EM specification.",
    ]
    (output_dir / "source_notes.md").write_text("\n".join(source_notes) + "\n", encoding="utf-8")
    print(artifact_path)


if __name__ == "__main__":
    main()
