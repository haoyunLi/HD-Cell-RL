#!/usr/bin/env python
"""Summarize exact leave-one-bin-out physical-cell EM experiments."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KAPPAS = (0.25, 0.5, 1.0, 2.0, 4.0)
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
    parser.add_argument("--loo-current-run", required=True)
    parser.add_argument("--loo-capture-run", required=True)
    parser.add_argument("--loo-current-iteration2-run", required=True)
    parser.add_argument("--loo-capture-iteration2-run", required=True)
    parser.add_argument("--iteration1-current-run", required=True)
    parser.add_argument("--iteration1-capture-run", required=True)
    parser.add_argument("--profile-damping-current-run", required=True)
    parser.add_argument("--profile-damping-capture-run", required=True)
    parser.add_argument("--external-comparison-csv", required=True)
    parser.add_argument("--gt-cell-bins-path", required=True)
    parser.add_argument("--source-eval-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _table(path: str | Path, name: str = "weight_sweep.csv") -> pd.DataFrame:
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        source = source / name
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError(f"empty experiment table: {source}")
    return frame


def _select(
    frame: pd.DataFrame,
    *,
    kappa: float,
    profile_update_damping: float | None = None,
) -> pd.Series:
    mask = np.isclose(
        frame["relative_prior_strength"].astype(float),
        float(kappa),
    )
    if profile_update_damping is not None:
        mask &= np.isclose(
            frame["profile_update_damping"].astype(float),
            float(profile_update_damping),
        )
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise ValueError(
            f"expected one row for kappa={kappa:g}; found {len(rows)}"
        )
    return rows.iloc[0]


def _metric_row(
    *,
    condition: str,
    method: str,
    kappa: float,
    row: pd.Series,
) -> dict[str, Any]:
    return {
        "condition": condition,
        "method": method,
        "relative_prior_strength": float(kappa),
        **{name: row.get(name, np.nan) for name in METRICS},
    }


def _assignment_path(run: str | Path, row: pd.Series) -> Path:
    path = (
        Path(run).expanduser().resolve()
        / "variants"
        / str(row["variant"])
        / "assignments.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _owner_map(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, dtype={"barcode": str, "cell_id": str})
    conflicts = frame.groupby("barcode")["cell_id"].nunique()
    if bool((conflicts > 1).any()):
        raise ValueError(f"conflicting predicted owners in {path}")
    return dict(
        zip(
            frame["barcode"].astype(str),
            frame["cell_id"].astype(str),
            strict=False,
        )
    )


def _core_truth_transitions(
    *,
    condition: str,
    iteration2_run: str | Path,
    converged_run: str | Path,
    iteration2_row: pd.Series,
    converged_row: pd.Series,
    gt_core: pd.DataFrame,
) -> dict[str, Any]:
    owner2 = _owner_map(_assignment_path(iteration2_run, iteration2_row))
    owner_final = _owner_map(_assignment_path(converged_run, converged_row))
    truth = dict(
        zip(
            gt_core["barcode"].astype(str),
            gt_core["cell_id"].astype(str),
            strict=False,
        )
    )
    corrected = 0
    damaged = 0
    wrong_to_wrong = 0
    changed = 0
    for barcode, cell_id in truth.items():
        before = owner2.get(barcode, "__noncore__")
        after = owner_final.get(barcode, "__noncore__")
        if before == after:
            continue
        changed += 1
        before_correct = before == cell_id
        after_correct = after == cell_id
        if before_correct and not after_correct:
            damaged += 1
        elif not before_correct and after_correct:
            corrected += 1
        else:
            wrong_to_wrong += 1
    return {
        "condition": condition,
        "relative_prior_strength": float(
            iteration2_row["relative_prior_strength"]
        ),
        "n_core_truth_bins": int(len(truth)),
        "n_owner_changed": int(changed),
        "n_corrected": int(corrected),
        "n_damaged": int(damaged),
        "n_wrong_to_wrong": int(wrong_to_wrong),
        "net_corrected_minus_damaged": int(corrected - damaged),
    }


def _per_cell_evolution(
    *,
    condition: str,
    iteration2_run: str | Path,
    converged_run: str | Path,
    kappa: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    first = _table(iteration2_run, "per_cell.csv")
    final = _table(converged_run, "per_cell.csv")
    first = first.loc[
        np.isclose(first["relative_prior_strength"].astype(float), kappa),
        ["cell_id", "iou"],
    ].rename(columns={"iou": "iou_iteration2"})
    final = final.loc[
        np.isclose(final["relative_prior_strength"].astype(float), kappa),
        ["cell_id", "iou"],
    ].rename(columns={"iou": "iou_converged"})
    merged = first.merge(final, on="cell_id", validate="one_to_one")
    merged.insert(0, "condition", condition)
    merged["iou_delta"] = merged["iou_converged"] - merged["iou_iteration2"]
    tolerance = 1.0e-12
    summary = {
        "condition": condition,
        "relative_prior_strength": float(kappa),
        "n_core_cells": int(len(merged)),
        "n_cells_improved": int((merged["iou_delta"] > tolerance).sum()),
        "n_cells_damaged": int((merged["iou_delta"] < -tolerance).sum()),
        "n_cells_unchanged": int(
            (merged["iou_delta"].abs() <= tolerance).sum()
        ),
        "mean_cell_iou_delta": float(merged["iou_delta"].mean()),
        "minimum_cell_iou_delta": float(merged["iou_delta"].min()),
        "maximum_cell_iou_delta": float(merged["iou_delta"].max()),
    }
    return summary, merged


def _write_report(
    *,
    output: Path,
    joint: pd.DataFrame,
    evolution: pd.DataFrame,
    transitions: pd.DataFrame,
    cell_summary: pd.DataFrame,
    external: pd.DataFrame,
) -> None:
    robust = joint.iloc[0]
    current = evolution.loc[evolution["condition"] == "current_capture"]
    capture = evolution.loc[evolution["condition"] == "capture_thinned"]
    current_cell = cell_summary.loc[
        cell_summary["condition"] == "current_capture"
    ].iloc[0]
    capture_cell = cell_summary.loc[
        cell_summary["condition"] == "capture_thinned"
    ].iloc[0]
    current_transition = transitions.loc[
        transitions["condition"] == "current_capture"
    ].iloc[0]
    capture_transition = transitions.loc[
        transitions["condition"] == "capture_thinned"
    ].iloc[0]
    bin2cell_current = external.loc[
        (external["condition"] == "current_high_depth")
        & (external["method"] == "Bin2Cell"),
        "mean_core_cell_iou",
    ].iloc[0]
    bin2cell_capture = external.loc[
        (external["condition"] == "capture_thinned")
        & (external["method"] == "Bin2Cell"),
        "mean_core_cell_iou",
    ].iloc[0]
    lines = [
        "# Exact leave-one-bin-out EM: validation result",
        "",
        "This is an EM-only experiment on the same four validation patches and "
        "43 core cells. Margin cells compete for ownership. Ground truth is used "
        "only after inference. PPO and RL are not involved.",
        "",
        "## Result",
        "",
        (
            "The best converged cross-capture setting is "
            f"kappa={float(robust['relative_prior_strength']):g}. Its "
            f"current-capture IoU is "
            f"{float(robust['mean_core_cell_iou_current']):.6f}, its "
            f"capture-thinned IoU is "
            f"{float(robust['mean_core_cell_iou_capture']):.6f}, and the mean "
            f"is {float(robust['mean_iou_across_capture']):.6f}."
        ),
        "",
        (
            "At kappa=2, iteration 2 briefly reaches "
            f"{float(current.loc[current['stage'] == 'iteration_2', 'mean_core_cell_iou'].iloc[0]):.6f} "
            "on current capture and "
            f"{float(capture.loc[capture['stage'] == 'iteration_2', 'mean_core_cell_iou'].iloc[0]):.6f} "
            "after capture thinning. Convergence then lowers these values to "
            f"{float(current.loc[current['stage'] == 'converged', 'mean_core_cell_iou'].iloc[0]):.6f} "
            "and "
            f"{float(capture.loc[capture['stage'] == 'converged', 'mean_core_cell_iou'].iloc[0]):.6f}."
        ),
        "",
        "## Diagnosis",
        "",
        "Exact leave-one-out removes a bin's own previous contribution before "
        "scoring that bin. It does not stop neighboring ambiguous bins from "
        "shaping one another's profiles. The iteration trace shows that this "
        "remaining cross-bin feedback is enough to reverse the early gain.",
        "",
        (
            "Capture-thinned mean normalized entropy falls from "
            f"{float(capture.loc[capture['stage'] == 'iteration_2', 'mean_normalized_entropy'].iloc[0]):.6f} "
            "at iteration 2 to "
            f"{float(capture.loc[capture['stage'] == 'converged', 'mean_normalized_entropy'].iloc[0]):.6f} "
            "at convergence while IoU also falls. The solver is becoming more "
            "confident and less accurate, so the final entropy is not ready for "
            "calibration."
        ),
        "",
        (
            "From iteration 2 to convergence at kappa=2, current capture has "
            f"{int(current_cell['n_cells_damaged'])} damaged and "
            f"{int(current_cell['n_cells_improved'])} improved core cells. "
            "Capture-thinned has "
            f"{int(capture_cell['n_cells_damaged'])} damaged and "
            f"{int(capture_cell['n_cells_improved'])} improved core cells."
        ),
        "",
        (
            "Among bins whose dominant ground-truth owner is one of the 43 core "
            "cells, the same transition damages/corrects "
            f"{int(current_transition['n_damaged'])}/"
            f"{int(current_transition['n_corrected'])} assignments in current "
            "capture and "
            f"{int(capture_transition['n_damaged'])}/"
            f"{int(capture_transition['n_corrected'])} after capture thinning."
        ),
        "",
        (
            "Bin2Cell remains at IoU "
            f"{float(bin2cell_current):.6f} for current capture and "
            f"{float(bin2cell_capture):.6f} for capture-thinned data. The "
            "non-converged iteration-2 LOO result is promising, but selecting "
            "that iteration using ground truth would be leakage."
        ),
        "",
        "## Decision",
        "",
        "Converged exact leave-one-out is not the final formula. The next test "
        "should use deterministic fold-level cross-fitting: score every bin in "
        "a fold from profiles built without all non-nuclear bins in that fold. "
        "Nuclear anchors remain in every training profile. This blocks both "
        "self-feedback and within-fold mutual reinforcement.",
        "",
        "Do not calibrate entropy or train PPO from iteration 2. First define an "
        "inference-time stopping rule that does not use ground truth, or show "
        "that cross-fitting remains stable to convergence on held-out patches.",
        "",
        "## Files",
        "",
        "- `method_comparison.csv`: frozen, full M-step, LOO iteration 2, and converged LOO.",
        "- `joint_converged_ranking.csv`: converged LOO ranking across capture conditions.",
        "- `kappa2_evolution.csv`: iteration 1, iteration 2, and convergence.",
        "- `core_truth_owner_transitions.csv`: corrected and damaged core-truth bins.",
        "- `per_cell_iou_evolution.csv`: per-cell iteration-2-to-convergence changes.",
        "- `summary.json`: source runs and selected result.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    conditions = {
        "current_capture": {
            "loo": _table(args.loo_current_run),
            "loo_iteration2": _table(args.loo_current_iteration2_run),
            "iteration1": _table(args.iteration1_current_run),
            "damping": _table(args.profile_damping_current_run),
        },
        "capture_thinned": {
            "loo": _table(args.loo_capture_run),
            "loo_iteration2": _table(args.loo_capture_iteration2_run),
            "iteration1": _table(args.iteration1_capture_run),
            "damping": _table(args.profile_damping_capture_run),
        },
    }
    comparison_rows: list[dict[str, Any]] = []
    for condition, frames in conditions.items():
        for kappa in KAPPAS:
            for method, row in (
                (
                    "frozen_nuclear_profile",
                    _select(
                        frames["damping"],
                        kappa=kappa,
                        profile_update_damping=0.0,
                    ),
                ),
                (
                    "full_profile_m_step",
                    _select(
                        frames["damping"],
                        kappa=kappa,
                        profile_update_damping=1.0,
                    ),
                ),
                (
                    "leave_one_out_iteration2",
                    _select(frames["loo_iteration2"], kappa=kappa),
                ),
                (
                    "leave_one_out_converged",
                    _select(frames["loo"], kappa=kappa),
                ),
            ):
                comparison_rows.append(
                    _metric_row(
                        condition=condition,
                        method=method,
                        kappa=kappa,
                        row=row,
                    )
                )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output / "method_comparison.csv", index=False)

    current_loo = conditions["current_capture"]["loo"]
    capture_loo = conditions["capture_thinned"]["loo"]
    joint = current_loo[
        ["relative_prior_strength", *METRICS]
    ].merge(
        capture_loo[["relative_prior_strength", *METRICS]],
        on="relative_prior_strength",
        suffixes=("_current", "_capture"),
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
    joint.insert(0, "joint_rank", np.arange(1, len(joint) + 1))
    joint.to_csv(output / "joint_converged_ranking.csv", index=False)

    selected_kappa = float(joint.iloc[0]["relative_prior_strength"])
    diagnostic_kappa = 2.0
    evolution_rows: list[dict[str, Any]] = []
    for condition, frames in conditions.items():
        for stage, frame in (
            ("iteration_1", frames["iteration1"]),
            ("iteration_2", frames["loo_iteration2"]),
            ("converged", frames["loo"]),
        ):
            row = _select(frame, kappa=diagnostic_kappa)
            evolution_rows.append(
                {
                    "condition": condition,
                    "stage": stage,
                    "relative_prior_strength": diagnostic_kappa,
                    **{name: row.get(name, np.nan) for name in METRICS},
                }
            )
    evolution = pd.DataFrame(evolution_rows)
    evolution.to_csv(output / "kappa2_evolution.csv", index=False)

    source_eval = Path(args.source_eval_dir).expanduser().resolve()
    core_ids = set(
        pd.read_csv(source_eval / "per_episode.csv", dtype=str)["cell_id"].astype(str)
    )
    gt = pd.read_csv(
        Path(args.gt_cell_bins_path).expanduser().resolve(),
        dtype={"barcode": str, "cell_id": str},
    )
    gt_core = gt.loc[gt["cell_id"].astype(str).isin(core_ids)].copy()
    if bool(gt_core["barcode"].duplicated().any()):
        raise ValueError("dominant-owner core ground truth must have unique barcodes")

    transition_rows: list[dict[str, Any]] = []
    cell_summary_rows: list[dict[str, Any]] = []
    cell_detail_rows: list[pd.DataFrame] = []
    for condition, frames in conditions.items():
        iteration2_row = _select(
            frames["loo_iteration2"],
            kappa=diagnostic_kappa,
        )
        converged_row = _select(frames["loo"], kappa=diagnostic_kappa)
        iteration2_run = (
            args.loo_current_iteration2_run
            if condition == "current_capture"
            else args.loo_capture_iteration2_run
        )
        converged_run = (
            args.loo_current_run
            if condition == "current_capture"
            else args.loo_capture_run
        )
        transition_rows.append(
            _core_truth_transitions(
                condition=condition,
                iteration2_run=iteration2_run,
                converged_run=converged_run,
                iteration2_row=iteration2_row,
                converged_row=converged_row,
                gt_core=gt_core,
            )
        )
        cell_summary, cell_detail = _per_cell_evolution(
            condition=condition,
            iteration2_run=iteration2_run,
            converged_run=converged_run,
            kappa=diagnostic_kappa,
        )
        cell_summary_rows.append(cell_summary)
        cell_detail_rows.append(cell_detail)
    transitions = pd.DataFrame(transition_rows)
    transitions.to_csv(output / "core_truth_owner_transitions.csv", index=False)
    cell_summary = pd.DataFrame(cell_summary_rows)
    cell_summary.to_csv(output / "per_cell_evolution_summary.csv", index=False)
    pd.concat(cell_detail_rows, ignore_index=True).to_csv(
        output / "per_cell_iou_evolution.csv",
        index=False,
    )

    external = pd.read_csv(
        Path(args.external_comparison_csv).expanduser().resolve()
    )
    external.to_csv(output / "external_method_reference.csv", index=False)
    summary = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "EM-only; four fixed validation patches; 43 core cells",
        "selected_converged_kappa": selected_kappa,
        "diagnostic_evolution_kappa": diagnostic_kappa,
        "selected_converged_result": joint.iloc[0].to_dict(),
        "source_runs": {
            name: str(Path(value).expanduser().resolve())
            for name, value in vars(args).items()
            if name.endswith("_run")
        },
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    _write_report(
        output=output / "README.md",
        joint=joint,
        evolution=evolution,
        transitions=transitions,
        cell_summary=cell_summary,
        external=external,
    )
    print(joint.to_string(index=False))
    print(evolution.to_string(index=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
