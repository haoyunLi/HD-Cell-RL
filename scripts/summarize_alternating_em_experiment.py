#!/usr/bin/env python
"""Summarize alternating physical-cell profile EM experiments without RL."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


JOINT_METRICS = (
    "mean_core_cell_iou",
    "mean_core_cell_precision",
    "mean_core_cell_recall",
    "mean_boundary_f1",
    "gt_boundary_owner_accuracy",
    "mean_area_bias_fraction",
    "mean_absolute_area_bias_fraction",
    "hard_assignment_coverage",
    "owner_accuracy",
    "mean_normalized_entropy",
    "fraction_entropy_above_0p5",
    "mean_iterations",
    "converged_all_patches",
    "mean_cell_profile_reliability",
    "mean_relative_profile_prior_count",
    "mean_final_profile_total_variation",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high-run", required=True)
    parser.add_argument("--capture-run", required=True)
    parser.add_argument("--high-iteration1-run", required=True)
    parser.add_argument("--capture-iteration1-run", required=True)
    parser.add_argument("--type-high-run", required=True)
    parser.add_argument("--type-capture-run", required=True)
    parser.add_argument("--fixed-profile-high-run", required=True)
    parser.add_argument("--fixed-profile-capture-run", required=True)
    parser.add_argument("--external-methods-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _sweep(path: str | Path) -> pd.DataFrame:
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        source = source / "weight_sweep.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError(f"empty EM sweep: {source}")
    return frame


def _metric_record(
    *,
    condition: str,
    method: str,
    selection: str,
    row: pd.Series,
) -> dict[str, Any]:
    return {
        "condition": condition,
        "method": method,
        "selection": selection,
        "spatial_weight": row.get("spatial_weight", np.nan),
        "expression_weight": row.get("expression_weight", np.nan),
        "relative_prior_strength": row.get("relative_prior_strength", np.nan),
        "mean_core_cell_iou": row.get("mean_core_cell_iou", np.nan),
        "mean_core_cell_precision": row.get("mean_core_cell_precision", np.nan),
        "mean_core_cell_recall": row.get("mean_core_cell_recall", np.nan),
        "mean_boundary_f1": row.get("mean_boundary_f1", np.nan),
        "gt_boundary_owner_accuracy": row.get(
            "gt_boundary_owner_accuracy", np.nan
        ),
        "mean_area_bias_fraction": row.get("mean_area_bias_fraction", np.nan),
        "mean_absolute_area_bias_fraction": row.get(
            "mean_absolute_area_bias_fraction", np.nan
        ),
        "hard_assignment_coverage": row.get("hard_assignment_coverage", np.nan),
        "owner_accuracy": row.get("owner_accuracy", np.nan),
        "mean_normalized_entropy": row.get("mean_normalized_entropy", np.nan),
        "fraction_entropy_above_0p5": row.get(
            "fraction_entropy_above_0p5", np.nan
        ),
        "mean_iterations": row.get("mean_iterations", np.nan),
        "converged_all_patches": row.get("converged_all_patches", np.nan),
    }


def _external_record(
    *,
    external: pd.DataFrame,
    condition: str,
    external_condition: str,
    method: str,
) -> dict[str, Any]:
    rows = external.loc[
        (external["condition"].astype(str) == external_condition)
        & (external["method"].astype(str) == method.lower())
        & (external["status"].astype(str) == "completed")
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one external row for {external_condition}/{method}; "
            f"found {len(rows)}"
        )
    source = rows.iloc[0]
    return {
        "condition": condition,
        "method": method,
        "selection": "native matched benchmark output",
        "spatial_weight": np.nan,
        "expression_weight": np.nan,
        "relative_prior_strength": np.nan,
        "mean_core_cell_iou": float(source["mean_pred_iou"]),
        "mean_core_cell_precision": float(source["mean_pred_precision"]),
        "mean_core_cell_recall": float(source["mean_pred_recall"]),
        "mean_boundary_f1": np.nan,
        "gt_boundary_owner_accuracy": np.nan,
        "mean_area_bias_fraction": np.nan,
        "mean_absolute_area_bias_fraction": np.nan,
        "hard_assignment_coverage": np.nan,
        "owner_accuracy": np.nan,
        "mean_normalized_entropy": np.nan,
        "fraction_entropy_above_0p5": np.nan,
        "mean_iterations": np.nan,
        "converged_all_patches": np.nan,
    }


def _best(frame: pd.DataFrame) -> pd.Series:
    return frame.sort_values(
        ["mean_core_cell_iou", "owner_accuracy"],
        ascending=False,
    ).iloc[0]


def _at_parameters(
    frame: pd.DataFrame,
    *,
    spatial_weight: float,
    expression_weight: float,
    relative_prior_strength: float,
) -> pd.Series:
    selected = frame.loc[
        np.isclose(frame["spatial_weight"].astype(float), spatial_weight)
        & np.isclose(frame["expression_weight"].astype(float), expression_weight)
        & np.isclose(
            frame["relative_prior_strength"].astype(float),
            relative_prior_strength,
        )
    ]
    if len(selected) != 1:
        raise ValueError(
            "expected one alternating-EM row at "
            f"alpha={spatial_weight}, beta={expression_weight}, "
            f"kappa={relative_prior_strength}; found {len(selected)}"
        )
    return selected.iloc[0]


def _at_beta(frame: pd.DataFrame, beta: float) -> pd.Series:
    selected = frame.loc[
        np.isclose(frame["spatial_weight"].astype(float), 1.0)
        & np.isclose(frame["expression_weight"].astype(float), beta)
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one alpha=1, beta={beta} row; found {len(selected)}")
    return selected.iloc[0]


def _write_diagnosis(
    *,
    output: Path,
    robust: pd.Series,
    high_best: pd.Series,
    capture_best: pd.Series,
    iteration_effect: pd.DataFrame,
) -> None:
    high_first = iteration_effect.loc[
        (iteration_effect["condition"] == "current_high_depth")
        & np.isclose(
            iteration_effect["relative_prior_strength"].astype(float),
            float(high_best["relative_prior_strength"]),
        )
    ].iloc[0]
    capture_first = iteration_effect.loc[
        (iteration_effect["condition"] == "capture_thinned")
        & np.isclose(
            iteration_effect["relative_prior_strength"].astype(float),
            float(capture_best["relative_prior_strength"]),
        )
    ].iloc[0]
    lines = [
        "# Alternating physical-cell profile EM: validation diagnosis",
        "",
        "This is an EM-only comparison on the four fixed validation patches and "
        "43 core cells. Margin cells participate in ownership competition. Ground "
        "truth is used only after inference.",
        "",
        "## Result",
        "",
        (
            "The best high-depth setting was "
            f"kappa={float(high_best['relative_prior_strength']):g}, "
            f"beta={float(high_best['expression_weight']):g} "
            f"(IoU {float(high_best['mean_core_cell_iou']):.6f}). The best "
            "capture-thinned setting was "
            f"kappa={float(capture_best['relative_prior_strength']):g}, "
            f"beta={float(capture_best['expression_weight']):g} "
            f"(IoU {float(capture_best['mean_core_cell_iou']):.6f})."
        ),
        "",
        (
            "Selecting one setting by mean IoU across both conditions gives "
            f"kappa={float(robust['relative_prior_strength']):g}, "
            f"beta={float(robust['expression_weight']):g}: high-depth IoU "
            f"{float(robust['mean_core_cell_iou_high']):.6f}, capture-thinned "
            f"IoU {float(robust['mean_core_cell_iou_capture']):.6f}."
        ),
        "",
        "## What the iteration diagnostic shows",
        "",
        (
            "At the high-depth optimum, the first ownership E-step had IoU "
            f"{float(high_first['mean_core_cell_iou_iteration1']):.6f}; the "
            "converged run had IoU "
            f"{float(high_first['mean_core_cell_iou_converged']):.6f} "
            f"(delta {float(high_first['iou_delta_after_alternation']):+.6f})."
        ),
        "",
        (
            "At the capture-thinned optimum, the corresponding values were "
            f"{float(capture_first['mean_core_cell_iou_iteration1']):.6f} and "
            f"{float(capture_first['mean_core_cell_iou_converged']):.6f} "
            f"(delta {float(capture_first['iou_delta_after_alternation']):+.6f})."
        ),
        "",
        "The high-depth drop is consistent with confirmation drift: early hard-to-"
        "assign boundary bins enter the soft M-step, shift the physical-cell profile, "
        "and then reinforce later ownership. Capture thinning needs stronger scRNA "
        "shrinkage, and its selected setting changes little after the first step. "
        "This evidence does not support training PPO yet.",
        "",
        "## Files",
        "",
        "- `joint_parameter_ranking.csv`: every matched kappa/beta setting across depth.",
        "- `architecture_comparison.csv`: distance, type-level EM, fixed profile, "
        "alternating profile, Bin2Cell, and STCS.",
        "- `iteration_effect.csv`: first E-step versus converged alternating result.",
        "- `summary.json`: selected settings and source paths.",
        "",
        "The next EM experiment should damp or cross-fit the physical-cell profile "
        "update, while keeping the existing responsibility damping and RL unchanged.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    high = _sweep(args.high_run)
    capture = _sweep(args.capture_run)
    high_iteration1 = _sweep(args.high_iteration1_run)
    capture_iteration1 = _sweep(args.capture_iteration1_run)
    type_high = _sweep(args.type_high_run)
    type_capture = _sweep(args.type_capture_run)
    fixed_high = _sweep(args.fixed_profile_high_run)
    fixed_capture = _sweep(args.fixed_profile_capture_run)
    external = pd.read_csv(Path(args.external_methods_csv).expanduser().resolve())

    keys = ["spatial_weight", "expression_weight", "relative_prior_strength"]
    high_selected = high[keys + list(JOINT_METRICS)].rename(
        columns={name: f"{name}_high" for name in JOINT_METRICS}
    )
    capture_selected = capture[keys + list(JOINT_METRICS)].rename(
        columns={name: f"{name}_capture" for name in JOINT_METRICS}
    )
    joint = high_selected.merge(capture_selected, on=keys, validate="one_to_one")
    joint["mean_iou_across_depth"] = joint[
        ["mean_core_cell_iou_high", "mean_core_cell_iou_capture"]
    ].mean(axis=1)
    joint["worst_condition_iou"] = joint[
        ["mean_core_cell_iou_high", "mean_core_cell_iou_capture"]
    ].min(axis=1)
    joint["capture_minus_high_iou"] = (
        joint["mean_core_cell_iou_capture"] - joint["mean_core_cell_iou_high"]
    )
    joint = joint.sort_values(
        ["mean_iou_across_depth", "worst_condition_iou"],
        ascending=False,
    ).reset_index(drop=True)
    joint.insert(0, "joint_rank", np.arange(1, len(joint) + 1, dtype=np.int64))
    joint.to_csv(output_dir / "joint_parameter_ranking.csv", index=False)
    robust = joint.iloc[0]

    high_best = _best(high)
    capture_best = _best(capture)
    robust_high = _at_parameters(
        high,
        spatial_weight=float(robust["spatial_weight"]),
        expression_weight=float(robust["expression_weight"]),
        relative_prior_strength=float(robust["relative_prior_strength"]),
    )
    robust_capture = _at_parameters(
        capture,
        spatial_weight=float(robust["spatial_weight"]),
        expression_weight=float(robust["expression_weight"]),
        relative_prior_strength=float(robust["relative_prior_strength"]),
    )

    architecture_rows: list[dict[str, Any]] = []
    for condition, type_frame, fixed_frame, alternate_best, alternate_robust in (
        (
            "current_high_depth",
            type_high,
            fixed_high,
            high_best,
            robust_high,
        ),
        (
            "capture_thinned",
            type_capture,
            fixed_capture,
            capture_best,
            robust_capture,
        ),
    ):
        architecture_rows.extend(
            [
                _metric_record(
                    condition=condition,
                    method="distance_only_em",
                    selection="alpha=1, beta=0",
                    row=_at_beta(type_frame, 0.0),
                ),
                _metric_record(
                    condition=condition,
                    method="type_level_em",
                    selection="condition-specific validation best",
                    row=_best(type_frame),
                ),
                _metric_record(
                    condition=condition,
                    method="fixed_nuclear_profile",
                    selection="condition-specific validation best; prior_umis=5000",
                    row=_best(fixed_frame),
                ),
                _metric_record(
                    condition=condition,
                    method="alternating_cell_profile_em",
                    selection="condition-specific validation best",
                    row=alternate_best,
                ),
                _metric_record(
                    condition=condition,
                    method="alternating_cell_profile_em",
                    selection="single cross-depth validation setting",
                    row=alternate_robust,
                ),
            ]
        )
        external_condition = (
            "high_depth_local"
            if condition == "current_high_depth"
            else "capture_thinned"
        )
        architecture_rows.extend(
            [
                _external_record(
                    external=external,
                    condition=condition,
                    external_condition=external_condition,
                    method="Bin2Cell",
                ),
                _external_record(
                    external=external,
                    condition=condition,
                    external_condition=external_condition,
                    method="STCS",
                ),
            ]
        )
    architecture = pd.DataFrame(architecture_rows)
    architecture.to_csv(output_dir / "architecture_comparison.csv", index=False)

    iteration_rows: list[pd.DataFrame] = []
    for condition, iteration1, converged in (
        ("current_high_depth", high_iteration1, high),
        ("capture_thinned", capture_iteration1, capture),
    ):
        metrics = [
            "mean_core_cell_iou",
            "mean_core_cell_precision",
            "mean_core_cell_recall",
            "mean_boundary_f1",
            "gt_boundary_owner_accuracy",
            "mean_absolute_area_bias_fraction",
            "mean_normalized_entropy",
            "fraction_entropy_above_0p5",
            "owner_accuracy",
        ]
        first = iteration1[keys + metrics].rename(
            columns={name: f"{name}_iteration1" for name in metrics}
        )
        final = converged[keys + metrics].rename(
            columns={name: f"{name}_converged" for name in metrics}
        )
        merged = first.merge(final, on=keys, validate="one_to_one")
        merged.insert(0, "condition", condition)
        merged["iou_delta_after_alternation"] = (
            merged["mean_core_cell_iou_converged"]
            - merged["mean_core_cell_iou_iteration1"]
        )
        iteration_rows.append(merged)
    iteration_effect = pd.concat(iteration_rows, ignore_index=True)
    iteration_effect.to_csv(output_dir / "iteration_effect.csv", index=False)

    summary = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": {
            "rl_used": False,
            "n_validation_patches": 4,
            "n_core_cells": 43,
            "bin_universe": "positive_expression_plus_locked_nuclear",
            "background_enabled": False,
            "candidate_max_distance_um": 20.0,
        },
        "condition_specific_best": {
            "current_high_depth": high_best.to_dict(),
            "capture_thinned": capture_best.to_dict(),
        },
        "single_cross_depth_selection": robust.to_dict(),
        "sources": {key: str(Path(value).expanduser().resolve()) for key, value in {
            "high_run": args.high_run,
            "capture_run": args.capture_run,
            "high_iteration1_run": args.high_iteration1_run,
            "capture_iteration1_run": args.capture_iteration1_run,
            "type_high_run": args.type_high_run,
            "type_capture_run": args.type_capture_run,
            "fixed_profile_high_run": args.fixed_profile_high_run,
            "fixed_profile_capture_run": args.fixed_profile_capture_run,
            "external_methods_csv": args.external_methods_csv,
        }.items()},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    _write_diagnosis(
        output=output_dir / "README.md",
        robust=robust,
        high_best=high_best,
        capture_best=capture_best,
        iteration_effect=iteration_effect,
    )
    print(joint.head(10).to_string(index=False))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
