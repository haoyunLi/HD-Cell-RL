#!/usr/bin/env python3
"""Summarize the fixed donor-disjoint EM benchmark and paired depth control."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hd_cell_rl.experiment_protocol import paired_cluster_mean_ci


METHOD_NAMES = {
    "hd_cell_rl_nucleus_only": "nucleus_only",
    "hd_cell_rl_distance_only": "distance_only",
    "hd_cell_rl_frozen_em": "frozen_em",
    "bin2cell": "bin2cell",
    "stcs": "stcs",
}

CONDITIONS = {
    "capture_matched_ambient": "evaluations",
    "high_depth_no_ambient": "evaluations_high_depth",
}

SUMMARY_METRICS = (
    "mean_pred_iou",
    "mean_pred_fractional_iou",
    "mean_pred_precision",
    "mean_pred_recall",
    "mean_gene_spearman_r",
    "mean_gene_rmse",
)

CELL_METRICS = (
    "pred_iou",
    "pred_fractional_iou",
    "pred_precision",
    "pred_recall",
    "gene_spearman_r",
    "gene_rmse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _discover_runs(root: Path) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for summary_path in sorted(root.glob("*/summary.json")):
        summary = _read_json(summary_path)
        canonical = METHOD_NAMES.get(str(summary.get("method", "")))
        if canonical is not None:
            runs[canonical] = summary_path.parent
    missing = set(METHOD_NAMES.values()).difference(runs)
    if missing:
        raise FileNotFoundError(f"Missing evaluation methods in {root}: {sorted(missing)}")
    return runs


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    draws = rng.choice(values, size=(int(replicates), values.size), replace=True)
    means = draws.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _format(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    labels = {
        "method": "Method",
        "mean_pred_iou": "IoU",
        "mean_pred_fractional_iou": "Fractional IoU",
        "mean_pred_precision": "Precision",
        "mean_pred_recall": "Recall",
        "mean_gene_spearman_r": "Gene Spearman",
        "mean_gene_rmse": "Gene RMSE",
    }
    lines = [
        "| " + " | ".join(labels.get(column, column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.loc[:, columns].itertuples(index=False, name=None):
        values = []
        for column, value in zip(columns, row, strict=True):
            values.append(str(value) if column == "method" else _format(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    output_dir = (
        benchmark_dir / ('comparison_patch_bootstrap_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        if args.output_dir is None
        else Path(args.output_dir).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    source_cell_patch = pd.read_csv(benchmark_dir / "selection/source_eval/per_episode.csv")
    source_cell_patch["cell_id"] = source_cell_patch["cell_id"].astype(str)
    rng = np.random.default_rng(int(args.seed))

    comparison_rows: list[dict[str, Any]] = []
    per_cell_frames: dict[tuple[str, str], pd.DataFrame] = {}
    run_paths: dict[str, dict[str, str]] = {}

    for condition, relative_eval_dir in CONDITIONS.items():
        evaluation_root = benchmark_dir / relative_eval_dir
        runs = _discover_runs(evaluation_root)
        run_paths[condition] = {method: str(path) for method, path in runs.items()}
        for method, run_dir in runs.items():
            summary = _read_json(run_dir / "summary.json")
            row: dict[str, Any] = {
                "condition": condition,
                "method": method,
                "n_cells": int(summary["n_episodes_evaluated"]),
                "evaluation_dir": str(run_dir),
            }
            for metric in SUMMARY_METRICS:
                row[metric] = summary.get(metric, np.nan)
            comparison_rows.append(row)

            cells = pd.read_csv(run_dir / "per_episode.csv")
            cells["cell_id"] = cells["cell_id"].astype(str)
            cells = cells.merge(source_cell_patch, on="cell_id", how="left", validate="one_to_one")
            if cells["patch_id"].isna().any():
                raise ValueError(f"Missing patch IDs for {condition}/{method}")
            per_cell_frames[(condition, method)] = cells

    comparison = pd.DataFrame(comparison_rows)
    comparison["iou_rank"] = comparison.groupby("condition")["mean_pred_iou"].rank(
        method="min", ascending=False
    ).astype(int)
    comparison = comparison.sort_values(["condition", "iou_rank", "method"])
    comparison.to_csv(output_dir / "method_comparison.csv", index=False)

    per_patch_rows: list[dict[str, Any]] = []
    for (condition, method), cells in per_cell_frames.items():
        for patch_id, patch in cells.groupby("patch_id", sort=True):
            row = {
                "condition": condition,
                "method": method,
                "patch_id": patch_id,
                "n_cells": int(len(patch)),
            }
            for metric in CELL_METRICS:
                row[f"mean_{metric}"] = float(pd.to_numeric(patch[metric], errors="coerce").mean())
            per_patch_rows.append(row)
    pd.DataFrame(per_patch_rows).sort_values(
        ["condition", "patch_id", "method"]
    ).to_csv(output_dir / "per_patch_method_comparison.csv", index=False)

    paired_cell_rows: list[pd.DataFrame] = []
    paired_summary_rows: list[dict[str, Any]] = []
    competitors = ("distance_only", "bin2cell", "stcs", "nucleus_only")
    for condition in CONDITIONS:
        em = per_cell_frames[(condition, "frozen_em")].set_index("cell_id")
        for competitor in competitors:
            other = per_cell_frames[(condition, competitor)].set_index("cell_id")
            joined = em.loc[:, list(CELL_METRICS)].join(
                other.loc[:, list(CELL_METRICS)],
                how="inner",
                lsuffix="_em",
                rsuffix="_competitor",
                validate="one_to_one",
            )
            joined = joined.join(source_cell_patch.set_index("cell_id"), how="left")
            out = joined.loc[:, ["patch_id"]].copy()
            out.insert(0, "competitor", competitor)
            out.insert(0, "condition", condition)
            for metric in CELL_METRICS:
                out[f"{metric}_em"] = joined[f"{metric}_em"]
                out[f"{metric}_competitor"] = joined[f"{metric}_competitor"]
                out[f"{metric}_delta_em_minus_competitor"] = (
                    joined[f"{metric}_em"] - joined[f"{metric}_competitor"]
                )
                delta = pd.to_numeric(
                    out[f"{metric}_delta_em_minus_competitor"], errors="coerce"
                ).to_numpy(dtype=np.float64)
                finite = delta[np.isfinite(delta)]
                low, high = paired_cluster_mean_ci(
                    delta, out['patch_id'].to_numpy(), replicates=int(args.bootstrap_replicates), rng=rng
                )
                if metric == "gene_rmse":
                    fraction_better = float(np.mean(finite < 0.0)) if finite.size else np.nan
                else:
                    fraction_better = float(np.mean(finite > 0.0)) if finite.size else np.nan
                paired_summary_rows.append(
                    {
                        "condition": condition,
                        "competitor": competitor,
                        "metric": metric,
                        "n_pairs": int(finite.size),
                        "n_clusters": int(out.loc[np.isfinite(delta), 'patch_id'].nunique()),
                        "bootstrap_unit": "patch (paired, cell-weighted mean)",
                        "mean_delta_em_minus_competitor": (
                            float(finite.mean()) if finite.size else np.nan
                        ),
                        "median_delta_em_minus_competitor": (
                            float(np.median(finite)) if finite.size else np.nan
                        ),
                        "bootstrap_mean_delta_ci_low": low,
                        "bootstrap_mean_delta_ci_high": high,
                        "fraction_cells_em_better": fraction_better,
                    }
                )
            out.insert(2, "cell_id", out.index)
            paired_cell_rows.append(out.reset_index(drop=True))

    paired_cells = pd.concat(paired_cell_rows, ignore_index=True)
    paired_cells.to_csv(output_dir / "paired_cell_deltas.csv", index=False)
    paired_summary = pd.DataFrame(paired_summary_rows)
    paired_summary.to_csv(output_dir / "paired_summary.csv", index=False)

    condition_deltas: list[dict[str, Any]] = []
    for method in METHOD_NAMES.values():
        capture = per_cell_frames[("capture_matched_ambient", method)].set_index("cell_id")
        high = per_cell_frames[("high_depth_no_ambient", method)].set_index("cell_id")
        joined = high.loc[:, list(CELL_METRICS)].join(
            capture.loc[:, list(CELL_METRICS)],
            lsuffix="_high_depth",
            rsuffix="_capture",
            validate="one_to_one",
        )
        for metric in CELL_METRICS:
            delta = (
                joined[f"{metric}_high_depth"] - joined[f"{metric}_capture"]
            ).to_numpy(dtype=np.float64)
            clusters = source_cell_patch.set_index('cell_id').loc[joined.index, 'patch_id'].to_numpy()
            low, high_ci = paired_cluster_mean_ci(
                delta, clusters, replicates=int(args.bootstrap_replicates), rng=rng
            )
            n_clusters = len(set(clusters[np.isfinite(delta)]))
            delta = delta[np.isfinite(delta)]
            condition_deltas.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_pairs": int(delta.size),
                    "mean_delta_high_depth_minus_capture": float(delta.mean()),
                    "n_clusters": n_clusters,
                    "bootstrap_unit": "patch (paired, cell-weighted mean)",
                    "bootstrap_mean_delta_ci_low": low,
                    "bootstrap_mean_delta_ci_high": high_ci,
                }
            )
    condition_delta_df = pd.DataFrame(condition_deltas)
    condition_delta_df.to_csv(output_dir / "paired_condition_deltas.csv", index=False)

    selection = _read_json(benchmark_dir / "selection/summary.json")
    inputs = _read_json(benchmark_dir / "inputs/summary.json")
    capture_data = _read_json(
        benchmark_dir / "datasets/capture_matched_structured_ambient/pseudo_hd_summary.json"
    )
    leakage_audit = {
        "selection_created_from_ground_truth": selection["created_from_ground_truth"],
        "selection_uses_performance_metrics": selection["selection_uses_performance_metrics"],
        "patch_id_overlap_with_development": selection["patch_id_overlap"],
        "context_cell_id_overlap_with_development": selection["context_cell_id_overlap"],
        "context_rectangle_overlap_with_development": selection["context_rectangle_overlap"],
        "test_patch_ids": selection["test_patch_ids"],
        "n_test_core_cells": selection["n_test_core_cells"],
        "n_test_context_cells": selection["n_test_context_cells"],
        "generator_donors": inputs["generator_donors"],
        "reference_donors": inputs["reference_donors"],
        "generator_reference_donor_overlap": inputs["generator_reference_donor_overlap"],
        "generator_reference_cell_barcode_overlap": inputs[
            "generator_reference_cell_barcode_overlap"
        ],
        "geometry_changed": inputs["geometry_changed"],
        "same_source_slide_geometry": True,
        "capture_final_to_real_total_ratio": capture_data[
            "realized_final_to_real_total_ratio"
        ],
        "ambient_fraction_of_real_total": capture_data["ambient_target_fraction"],
        "frozen_em_parameters": {
            "spatial_weight": 1.0,
            "expression_weight": 1.5,
            "relative_profile_prior_strength_kappa": 2.0,
            "profile_update_damping": 0.0,
            "background_enabled": False,
        },
        "test_set_parameter_tuning": False,
        "ppo_or_rl_run": False,
        "entropy_recalibrated": False,
        "evaluation_run_paths": run_paths,
    }
    with (output_dir / "leakage_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(leakage_audit, handle, indent=2)

    tables = {
        condition: comparison.loc[comparison["condition"] == condition].sort_values("iou_rank")
        for condition in CONDITIONS
    }
    em_capture = tables["capture_matched_ambient"].set_index("method").loc["frozen_em"]
    em_high = tables["high_depth_no_ambient"].set_index("method").loc["frozen_em"]
    dist_capture = tables["capture_matched_ambient"].set_index("method").loc["distance_only"]
    dist_high = tables["high_depth_no_ambient"].set_index("method").loc["distance_only"]
    report = f"""# Donor-disjoint, spatially unseen EM benchmark

## 结论

这次结果不支持“EM 只是记住了原来的 4 个 training patches”。在同一组预先锁定、与开发区无空间重叠的 4 个 patches 上，high-depth 条件下 frozen EM 的 mean core-cell IoU 为 {_format(em_high['mean_pred_iou'])}，高于 distance-only 的 {_format(dist_high['mean_pred_iou'])}。但在匹配真实 capture depth 并加入 structured ambient 后，frozen EM 降到 {_format(em_capture['mean_pred_iou'])}，并略低于 distance-only 的 {_format(dist_capture['mean_pred_iou'])}。

因此当前最可信的诊断是：EM 的表达项在高计数、无 ambient 的 pseudo data 上有效，但没有对真实 capture 稀疏度和 ambient contamination 保持稳健。现在不应训练 PPO，也不应重新校准 entropy。先修正或重新标定 expression compatibility，尤其是低 UMI bins 的 evidence scale 与 ambient-aware likelihood。

## Capture-matched + structured ambient

{_markdown_table(tables['capture_matched_ambient'], ['method', *SUMMARY_METRICS])}

## High-depth, no ambient paired control

{_markdown_table(tables['high_depth_no_ambient'], ['method', *SUMMARY_METRICS])}

## 实验锁定与泄漏检查

- 这 4 个 patches 最初在运行方法前通过 metadata 选定，共 43 个 core cells、190 个 context cells；经多次检查后，当前重分析将其标记为 development/audit evidence，不再当作未见 test。
- 它们与原 4 个 development patches 及先前 A/B/C validation patches 没有 patch ID、context cell 或 context rectangle 重叠。
- scDesign3 generator donors 与 scRNA reference donors 完全分开；cell barcode overlap 为 0。
- EM 参数冻结为 alpha=1、beta=1.5、kappa=2、profile update damping=0。beta=0 是同批 distance-only control。
- 所有方法使用同一组 external nuclear seeds、core-cell evaluation set 和 ground truth；没有运行 PPO/RL，也没有重新校准 entropy。
- 这仍不是一个独立 tissue slide：expression donors 和 reference donors 独立，但细胞几何来自同一 source slide。报告不能把它称为跨组织外部验证。

## 需要谨慎解释的地方

Bin2Cell 在两个条件的 GEX StarDist 都检测到 0 个对象。当前 Bin2Cell 数字实际来自 external nuclear labels 加 `max_bin_distance=5` 的固定扩张，不是完整 GEX segmentation 成功后的 Bin2Cell 行为。

capture-matched 条件同时改变了 capture depth 和 ambient，因此单凭这两个条件还不能把下降完全归因于其中一项。下一项最小实验应只做两个 paired controls：`thinning only` 和 `ambient only`。保持 patch、synthetic cell realization、reference 和所有方法参数不变，即可分解两种因素。

## 输出

- `method_comparison.csv`: 两个条件下的总体方法表。
- `per_patch_method_comparison.csv`: 每个 patch 的方法表现。
- `paired_cell_deltas.csv`: frozen EM 与各方法的逐 cell 配对差值。
- `paired_summary.csv`: 配对均值、median、按完整 patch 配对重采样的 95% CI 和 EM 胜率。估计量是 cell-weighted 均值，但独立抽样单位是 patch；只有 4 个组织区域，不能把 43 个 cells 当成 43 个独立组织重复。
- `paired_condition_deltas.csv`: 每种方法从 capture-matched 到 high-depth 的逐 cell 配对变化。
- `leakage_audit.json`: patch、donor、参数冻结和运行路径审计。
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    machine_summary = {
        "benchmark_dir": str(benchmark_dir),
        "output_dir": str(output_dir),
        "n_conditions": len(CONDITIONS),
        "n_methods": len(METHOD_NAMES),
        "n_core_cells": int(selection["n_test_core_cells"]),
        "capture_frozen_em_iou": float(em_capture["mean_pred_iou"]),
        "capture_distance_only_iou": float(dist_capture["mean_pred_iou"]),
        "high_depth_frozen_em_iou": float(em_high["mean_pred_iou"]),
        "high_depth_distance_only_iou": float(dist_high["mean_pred_iou"]),
        "recommendation": "run paired thinning-only and ambient-only EM-only controls before PPO",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(machine_summary, handle, indent=2)
    print(output_dir)


if __name__ == "__main__":
    main()
