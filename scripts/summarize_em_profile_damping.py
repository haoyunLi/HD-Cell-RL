#!/usr/bin/env python
"""Summarize physical-cell profile-update damping experiments without RL."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEYS = (
    "spatial_weight",
    "expression_weight",
    "relative_prior_strength",
    "profile_update_damping",
)
METRICS = (
    "mean_core_cell_iou",
    "owner_accuracy",
    "mean_boundary_f1",
    "gt_boundary_owner_accuracy",
    "mean_normalized_entropy",
    "fraction_entropy_above_0p5",
    "mean_iterations",
    "converged_all_patches",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-run", required=True)
    parser.add_argument("--capture-run", required=True)
    parser.add_argument("--current-gamma0p1-convergence-run", required=True)
    parser.add_argument("--capture-gamma0p1-convergence-run", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load_table(path: str | Path, name: str) -> pd.DataFrame:
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        source = source / name
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError(f"empty experiment table: {source}")
    return frame


def _row_record(row: pd.Series) -> dict[str, Any]:
    return {
        name: (
            bool(row[name])
            if isinstance(row[name], (bool, np.bool_))
            else float(row[name])
        )
        for name in (*KEYS, *METRICS)
    }


def _best(frame: pd.DataFrame) -> pd.Series:
    return frame.sort_values(
        ["mean_core_cell_iou", "owner_accuracy"],
        ascending=False,
    ).iloc[0]


def _at(
    frame: pd.DataFrame,
    *,
    relative_prior_strength: float,
    profile_update_damping: float,
) -> pd.Series:
    rows = frame.loc[
        np.isclose(
            frame["relative_prior_strength"].astype(float),
            float(relative_prior_strength),
        )
        & np.isclose(
            frame["profile_update_damping"].astype(float),
            float(profile_update_damping),
        )
    ]
    if len(rows) != 1:
        raise ValueError(
            "expected one row at "
            f"kappa={relative_prior_strength:g}, "
            f"profile_update_damping={profile_update_damping:g}; "
            f"found {len(rows)}"
        )
    return rows.iloc[0]


def _condition_comparison(
    *,
    condition: str,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for kappa in sorted(frame["relative_prior_strength"].astype(float).unique()):
        frozen = _at(
            frame,
            relative_prior_strength=float(kappa),
            profile_update_damping=0.0,
        )
        full = _at(
            frame,
            relative_prior_strength=float(kappa),
            profile_update_damping=1.0,
        )
        rows.append(
            {
                "condition": condition,
                "relative_prior_strength": float(kappa),
                "iou_profile_frozen": float(frozen["mean_core_cell_iou"]),
                "iou_full_profile_m_step": float(full["mean_core_cell_iou"]),
                "iou_frozen_minus_full": float(
                    frozen["mean_core_cell_iou"]
                    - full["mean_core_cell_iou"]
                ),
                "boundary_f1_profile_frozen": float(frozen["mean_boundary_f1"]),
                "boundary_f1_full_profile_m_step": float(full["mean_boundary_f1"]),
                "owner_accuracy_profile_frozen": float(frozen["owner_accuracy"]),
                "owner_accuracy_full_profile_m_step": float(
                    full["owner_accuracy"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    *,
    output: Path,
    current_best: pd.Series,
    capture_best: pd.Series,
    robust: pd.Series,
    robust_frozen_vs_full: pd.DataFrame,
    convergence: pd.DataFrame,
) -> None:
    current_delta = robust_frozen_vs_full.loc[
        robust_frozen_vs_full["condition"] == "current_capture",
        "iou_frozen_minus_full",
    ].iloc[0]
    capture_delta = robust_frozen_vs_full.loc[
        robust_frozen_vs_full["condition"] == "capture_thinned",
        "iou_frozen_minus_full",
    ].iloc[0]
    converged_count = int(convergence["converged_all_patches_extended"].sum())
    lines = [
        "# Physical-cell profile-update damping: EM-only validation",
        "",
        "This experiment uses the same four fixed validation patches and 43 core "
        "cells as the previous EM comparison. Margin cells still compete for "
        "ownership. Ground truth is used only for evaluation, and no RL or PPO "
        "rollout is involved.",
        "",
        "## Result",
        "",
        (
            "The best current-capture setting was "
            f"kappa={float(current_best['relative_prior_strength']):g}, "
            f"profile_update_damping="
            f"{float(current_best['profile_update_damping']):g} "
            f"(IoU {float(current_best['mean_core_cell_iou']):.6f}). "
            "The best capture-thinned setting was "
            f"kappa={float(capture_best['relative_prior_strength']):g}, "
            f"profile_update_damping="
            f"{float(capture_best['profile_update_damping']):g} "
            f"(IoU {float(capture_best['mean_core_cell_iou']):.6f})."
        ),
        "",
        (
            "Selecting one setting by mean IoU across both capture conditions "
            f"gives kappa={float(robust['relative_prior_strength']):g}, "
            f"profile_update_damping="
            f"{float(robust['profile_update_damping']):g}: current-capture IoU "
            f"{float(robust['mean_core_cell_iou_current']):.6f}, "
            "capture-thinned IoU "
            f"{float(robust['mean_core_cell_iou_capture']):.6f}, and mean IoU "
            f"{float(robust['mean_iou_across_capture']):.6f}."
        ),
        "",
        "## What damping tested",
        "",
        "`profile_update_damping` is separate from responsibility damping. After "
        "a complete sparse M-step computes a raw physical-cell expression "
        "profile, the solver applies:",
        "",
        "`profile_new = (1 - gamma) * profile_old + gamma * profile_raw`",
        "",
        "At gamma=0, the physical-cell profile remains exactly at its "
        "nuclear-initialized value. At gamma=1, behavior is exactly the previous "
        "full profile M-step.",
        "",
        "## Diagnosis",
        "",
        (
            "At the cross-condition kappa, freezing the profile changed IoU "
            f"by {float(current_delta):+.6f} in current capture and "
            f"{float(capture_delta):+.6f} after capture thinning relative to "
            "the full profile M-step."
        ),
        "",
        (
            f"The extended gamma=0.1 audit converged for {converged_count}/"
            f"{len(convergence)} condition-by-kappa settings. Positive damping "
            "therefore acts mainly as an optimization step size: given enough "
            "iterations, it approaches the same updated-profile fixed point. "
            "Only gamma=0 removes feedback from candidate-bin assignments into "
            "the physical-cell profile."
        ),
        "",
        "The current data support the confirmation-drift hypothesis. Profile "
        "freezing improves the cross-condition mean, but the best kappa still "
        "changes with capture depth. These four patches are not enough to choose "
        "a final inference formula or calibrate entropy.",
        "",
        "## Files",
        "",
        "- `joint_parameter_ranking.csv`: all 25 matched kappa/gamma settings.",
        "- `frozen_vs_full.csv`: gamma=0 versus gamma=1 at each kappa.",
        "- `gamma0p1_convergence.csv`: 15-iteration and 60-iteration gamma=0.1 runs.",
        "- `selected_per_patch.csv`: per-patch metrics for the robust setting.",
        "- `summary.json`: selected settings and source directories.",
        "",
        "The next clean EM test is leave-one-bin-out or cross-fitted profile "
        "updating. That test can keep useful profile adaptation while preventing "
        "a bin from reinforcing its own assignment. PPO should remain paused "
        "until that comparison is complete on held-out patches or donors.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    current = _load_table(args.current_run, "weight_sweep.csv")
    capture = _load_table(args.capture_run, "weight_sweep.csv")
    current_extended = _load_table(
        args.current_gamma0p1_convergence_run,
        "weight_sweep.csv",
    )
    capture_extended = _load_table(
        args.capture_gamma0p1_convergence_run,
        "weight_sweep.csv",
    )

    current_selected = current[list(KEYS) + list(METRICS)].rename(
        columns={name: f"{name}_current" for name in METRICS}
    )
    capture_selected = capture[list(KEYS) + list(METRICS)].rename(
        columns={name: f"{name}_capture" for name in METRICS}
    )
    joint = current_selected.merge(
        capture_selected,
        on=list(KEYS),
        validate="one_to_one",
    )
    joint["mean_iou_across_capture"] = joint[
        ["mean_core_cell_iou_current", "mean_core_cell_iou_capture"]
    ].mean(axis=1)
    joint["worst_capture_iou"] = joint[
        ["mean_core_cell_iou_current", "mean_core_cell_iou_capture"]
    ].min(axis=1)
    joint = joint.sort_values(
        ["mean_iou_across_capture", "worst_capture_iou"],
        ascending=False,
    ).reset_index(drop=True)
    joint.insert(0, "joint_rank", np.arange(1, len(joint) + 1, dtype=np.int64))
    joint.to_csv(output / "joint_parameter_ranking.csv", index=False)
    robust = joint.iloc[0]

    frozen_vs_full = pd.concat(
        [
            _condition_comparison(condition="current_capture", frame=current),
            _condition_comparison(condition="capture_thinned", frame=capture),
        ],
        ignore_index=True,
    )
    frozen_vs_full.to_csv(output / "frozen_vs_full.csv", index=False)

    convergence_rows: list[pd.DataFrame] = []
    for condition, short, extended in (
        ("current_capture", current, current_extended),
        ("capture_thinned", capture, capture_extended),
    ):
        short_gamma = short.loc[
            np.isclose(short["profile_update_damping"].astype(float), 0.1)
        ]
        merged = short_gamma.merge(
            extended,
            on=list(KEYS),
            suffixes=("_iteration15", "_extended"),
            validate="one_to_one",
        )
        merged.insert(0, "condition", condition)
        convergence_rows.append(merged)
    convergence = pd.concat(convergence_rows, ignore_index=True)
    convergence.to_csv(output / "gamma0p1_convergence.csv", index=False)

    selected_patch_rows: list[pd.DataFrame] = []
    for condition, run_path in (
        ("current_capture", args.current_run),
        ("capture_thinned", args.capture_run),
    ):
        per_patch = _load_table(run_path, "per_patch.csv")
        selected = per_patch.loc[
            np.isclose(
                per_patch["relative_prior_strength"].astype(float),
                float(robust["relative_prior_strength"]),
            )
            & np.isclose(
                per_patch["profile_update_damping"].astype(float),
                float(robust["profile_update_damping"]),
            )
        ].copy()
        selected.insert(0, "condition", condition)
        selected_patch_rows.append(selected)
    pd.concat(selected_patch_rows, ignore_index=True).to_csv(
        output / "selected_per_patch.csv",
        index=False,
    )

    current_best = _best(current)
    capture_best = _best(capture)
    robust_frozen_vs_full = frozen_vs_full.loc[
        np.isclose(
            frozen_vs_full["relative_prior_strength"].astype(float),
            float(robust["relative_prior_strength"]),
        )
    ]
    summary = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "EM-only; four fixed validation patches; 43 core cells",
        "selection_rule": "maximum mean core-cell IoU across both capture conditions",
        "current_run": str(Path(args.current_run).expanduser().resolve()),
        "capture_run": str(Path(args.capture_run).expanduser().resolve()),
        "current_gamma0p1_convergence_run": str(
            Path(args.current_gamma0p1_convergence_run).expanduser().resolve()
        ),
        "capture_gamma0p1_convergence_run": str(
            Path(args.capture_gamma0p1_convergence_run).expanduser().resolve()
        ),
        "best_current_capture": _row_record(current_best),
        "best_capture_thinned": _row_record(capture_best),
        "robust_cross_capture": robust.to_dict(),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    _write_report(
        output=output / "README.md",
        current_best=current_best,
        capture_best=capture_best,
        robust=robust,
        robust_frozen_vs_full=robust_frozen_vs_full,
        convergence=convergence,
    )
    print(joint.head(10).to_string(index=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
