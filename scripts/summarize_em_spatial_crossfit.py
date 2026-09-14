#!/usr/bin/env python
"""Summarize spatial-block cross-fitted physical-cell EM experiments."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
    parser.add_argument("--crossfit-current-run", required=True)
    parser.add_argument("--crossfit-capture-run", required=True)
    parser.add_argument("--crossfit-current-iteration2-run", required=True)
    parser.add_argument("--crossfit-capture-iteration2-run", required=True)
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
    block_size_um: float | None = None,
    profile_update_damping: float | None = None,
) -> pd.Series:
    mask = np.isclose(
        frame["relative_prior_strength"].astype(float),
        float(kappa),
    )
    if block_size_um is not None:
        mask &= np.isclose(
            frame["spatial_crossfit_block_size_um"].astype(float),
            float(block_size_um),
        )
    if profile_update_damping is not None:
        mask &= np.isclose(
            frame["profile_update_damping"].astype(float),
            float(profile_update_damping),
        )
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise ValueError(
            "expected one row for "
            f"kappa={kappa:g}, block={block_size_um}, "
            f"profile damping={profile_update_damping}; found {len(rows)}"
        )
    return rows.iloc[0]


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


def _transition(
    *,
    condition: str,
    comparison: str,
    before_run: str | Path,
    after_run: str | Path,
    before_row: pd.Series,
    after_row: pd.Series,
    gt_core: pd.DataFrame,
) -> dict[str, Any]:
    before_owner = _owner_map(_assignment_path(before_run, before_row))
    after_owner = _owner_map(_assignment_path(after_run, after_row))
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
        before = before_owner.get(barcode, "__noncore__")
        after = after_owner.get(barcode, "__noncore__")
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
        "comparison": comparison,
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
    before_run: str | Path,
    after_run: str | Path,
    kappa: float,
    block_size_um: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    before = _table(before_run, "per_cell.csv")
    after = _table(after_run, "per_cell.csv")
    before_mask = np.isclose(
        before["relative_prior_strength"].astype(float),
        kappa,
    ) & np.isclose(
        before["spatial_crossfit_block_size_um"].astype(float),
        block_size_um,
    )
    after_mask = np.isclose(
        after["relative_prior_strength"].astype(float),
        kappa,
    ) & np.isclose(
        after["spatial_crossfit_block_size_um"].astype(float),
        block_size_um,
    )
    first = before.loc[before_mask, ["cell_id", "iou"]].rename(
        columns={"iou": "iou_iteration2"}
    )
    final = after.loc[after_mask, ["cell_id", "iou"]].rename(
        columns={"iou": "iou_converged"}
    )
    merged = first.merge(final, on="cell_id", validate="one_to_one")
    merged.insert(0, "condition", condition)
    merged["iou_delta"] = merged["iou_converged"] - merged["iou_iteration2"]
    tolerance = 1.0e-12
    summary = {
        "condition": condition,
        "n_core_cells": int(len(merged)),
        "n_cells_improved": int((merged["iou_delta"] > tolerance).sum()),
        "n_cells_damaged": int((merged["iou_delta"] < -tolerance).sum()),
        "n_cells_unchanged": int((merged["iou_delta"].abs() <= tolerance).sum()),
        "mean_cell_iou_delta": float(merged["iou_delta"].mean()),
        "minimum_cell_iou_delta": float(merged["iou_delta"].min()),
        "maximum_cell_iou_delta": float(merged["iou_delta"].max()),
    }
    return summary, merged


def _metric_row(
    *,
    condition: str,
    method: str,
    row: pd.Series,
) -> dict[str, Any]:
    return {
        "condition": condition,
        "method": method,
        **{name: row.get(name, np.nan) for name in METRICS},
    }


def _write_report(
    *,
    path: Path,
    selected: pd.Series,
    comparison: pd.DataFrame,
    evolution: pd.DataFrame,
    transitions: pd.DataFrame,
    per_cell_summary: pd.DataFrame,
) -> None:
    kappa = float(selected["relative_prior_strength"])
    block = float(selected["spatial_crossfit_block_size_um"])

    def metric(condition: str, method: str, name: str) -> float:
        return float(
            comparison.loc[
                (comparison["condition"] == condition)
                & (comparison["method"] == method),
                name,
            ].iloc[0]
        )

    def stage(condition: str, stage_name: str, name: str) -> float:
        return float(
            evolution.loc[
                (evolution["condition"] == condition)
                & (evolution["stage"] == stage_name),
                name,
            ].iloc[0]
        )

    drift = transitions.loc[transitions["comparison"] == "iteration2_to_converged"]
    current_drift = drift.loc[drift["condition"] == "current_capture"].iloc[0]
    capture_drift = drift.loc[drift["condition"] == "capture_thinned"].iloc[0]
    current_cells = per_cell_summary.loc[
        per_cell_summary["condition"] == "current_capture"
    ].iloc[0]
    capture_cells = per_cell_summary.loc[
        per_cell_summary["condition"] == "capture_thinned"
    ].iloc[0]
    cross_current = metric(
        "current_capture", "spatial_block_crossfit_converged", "mean_core_cell_iou"
    )
    cross_capture = metric(
        "capture_thinned", "spatial_block_crossfit_converged", "mean_core_cell_iou"
    )
    frozen_current = metric(
        "current_capture", "frozen_nuclear_profile", "mean_core_cell_iou"
    )
    frozen_capture = metric(
        "capture_thinned", "frozen_nuclear_profile", "mean_core_cell_iou"
    )
    loo_current = metric(
        "current_capture", "leave_one_out_converged", "mean_core_cell_iou"
    )
    loo_capture = metric(
        "capture_thinned", "leave_one_out_converged", "mean_core_cell_iou"
    )
    lines = [
        "# Spatial-block cross-fitted EM: validation result",
        "",
        "This is an EM-only experiment on the same four validation patches and "
        "43 core cells. Margin cells still compete for ownership. Ground truth "
        "is read only after inference; PPO and RL are not involved.",
        "",
        "Each absolute-coordinate square block is one held-out fold. When a "
        "non-nuclear bin in fold f is scored for cell c, the profile numerator "
        "excludes every non-nuclear r[j,c] * x[j] contribution in fold f. "
        "Locked nuclear counts remain in every profile.",
        "",
        "## Result",
        "",
        (
            f"The best converged mean across capture conditions is kappa={kappa:g}, "
            f"block={block:g} um: current-capture IoU {cross_current:.6f}, "
            f"capture-thinned IoU {cross_capture:.6f}, mean "
            f"{float(selected['mean_iou_across_capture']):.6f}."
        ),
        "",
        (
            "This improves converged exact LOO by "
            f"{cross_current - loo_current:+.6f} on current capture and "
            f"{cross_capture - loo_capture:+.6f} after thinning. It still trails "
            f"the frozen nuclear-profile baseline by {cross_current - frozen_current:+.6f} "
            f"and {cross_capture - frozen_capture:+.6f}, respectively."
        ),
        "",
        (
            "At kappa=2 and block=16 um, iteration 2 reaches IoU "
            f"{stage('current_capture', 'iteration_2', 'mean_core_cell_iou'):.6f} "
            "on current capture and "
            f"{stage('capture_thinned', 'iteration_2', 'mean_core_cell_iou'):.6f} "
            "after thinning. Convergence lowers them to "
            f"{cross_current:.6f} and {cross_capture:.6f}."
        ),
        "",
        "## Convergence diagnosis",
        "",
        (
            "Mean normalized entropy also falls during the same drift: "
            f"{stage('current_capture', 'iteration_2', 'mean_normalized_entropy'):.6f} "
            f"to {stage('current_capture', 'converged', 'mean_normalized_entropy'):.6f} "
            "on current capture, and "
            f"{stage('capture_thinned', 'iteration_2', 'mean_normalized_entropy'):.6f} "
            f"to {stage('capture_thinned', 'converged', 'mean_normalized_entropy'):.6f} "
            "after thinning. The solver becomes more confident while IoU drops."
        ),
        "",
        (
            "From iteration 2 to convergence, current capture has "
            f"{int(current_drift['n_damaged'])} damaged and "
            f"{int(current_drift['n_corrected'])} corrected core-truth bins; "
            "capture-thinned has "
            f"{int(capture_drift['n_damaged'])} damaged and "
            f"{int(capture_drift['n_corrected'])} corrected bins."
        ),
        "",
        (
            "At the cell level, current capture has "
            f"{int(current_cells['n_cells_damaged'])} damaged and "
            f"{int(current_cells['n_cells_improved'])} improved core cells; "
            "capture-thinned has "
            f"{int(capture_cells['n_cells_damaged'])} damaged and "
            f"{int(capture_cells['n_cells_improved'])} improved cells."
        ),
        "",
        "At kappa=2, changing the block from 4 to 32 um has only a small "
        "effect. The large capture-dependent change across kappa remains. This "
        "points to profile reliability/shrinkage, not block radius, as the main "
        "unresolved sensitivity.",
        "",
        "## Decision",
        "",
        "Spatial-block exclusion reduces the exact-LOO failure, but it does not "
        "produce a stable gain over the frozen nuclear-profile baseline. The "
        "iteration-2 values cannot be selected with ground truth. Keep kappa=2, "
        "profile_update_damping=0 as the current cross-capture baseline. Do not "
        "train PPO or recalibrate entropy from this result.",
        "",
        "The next EM-only test should target the remaining cross-fold feedback "
        "or replace the fixed nuclear-depth reliability curve with an "
        "uncertainty estimate learned without pseudo ground truth.",
        "",
        "## Files",
        "",
        "- `joint_converged_ranking.csv`: all kappa and block-size combinations ranked across captures.",
        "- `method_comparison.csv`: frozen, full M-step, exact LOO, spatial cross-fit, and Bin2Cell references.",
        "- `selected_evolution.csv`: iteration 1, iteration 2, and convergence for kappa=2/block=16 um.",
        "- `core_truth_owner_transitions.csv`: corrected and damaged assignments for two transitions.",
        "- `per_cell_iou_evolution.csv`: cell-level iteration-2-to-convergence changes.",
        "- `summary.json`: selected setting and source runs.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    conditions = {
        "current_capture": {
            "crossfit": _table(args.crossfit_current_run),
            "crossfit_iteration2": _table(args.crossfit_current_iteration2_run),
            "loo": _table(args.loo_current_run),
            "loo_iteration2": _table(args.loo_current_iteration2_run),
            "iteration1": _table(args.iteration1_current_run),
            "damping": _table(args.profile_damping_current_run),
        },
        "capture_thinned": {
            "crossfit": _table(args.crossfit_capture_run),
            "crossfit_iteration2": _table(args.crossfit_capture_iteration2_run),
            "loo": _table(args.loo_capture_run),
            "loo_iteration2": _table(args.loo_capture_iteration2_run),
            "iteration1": _table(args.iteration1_capture_run),
            "damping": _table(args.profile_damping_capture_run),
        },
    }
    current = conditions["current_capture"]["crossfit"]
    capture = conditions["capture_thinned"]["crossfit"]
    join_keys = ["relative_prior_strength", "spatial_crossfit_block_size_um"]
    joint = current[[*join_keys, *METRICS]].merge(
        capture[[*join_keys, *METRICS]],
        on=join_keys,
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
    selected = joint.iloc[0]
    selected_kappa = float(selected["relative_prior_strength"])
    selected_block = float(selected["spatial_crossfit_block_size_um"])

    comparison_rows: list[dict[str, Any]] = []
    selected_rows: dict[str, dict[str, pd.Series]] = {}
    for condition, frames in conditions.items():
        selected_rows[condition] = {
            "frozen": _select(
                frames["damping"],
                kappa=selected_kappa,
                profile_update_damping=0.0,
            ),
            "full": _select(
                frames["damping"],
                kappa=selected_kappa,
                profile_update_damping=1.0,
            ),
            "loo_iteration2": _select(
                frames["loo_iteration2"],
                kappa=selected_kappa,
            ),
            "loo": _select(frames["loo"], kappa=selected_kappa),
            "crossfit_iteration2": _select(
                frames["crossfit_iteration2"],
                kappa=selected_kappa,
                block_size_um=selected_block,
            ),
            "crossfit": _select(
                frames["crossfit"],
                kappa=selected_kappa,
                block_size_um=selected_block,
            ),
        }
        for method, row in (
            ("frozen_nuclear_profile", selected_rows[condition]["frozen"]),
            ("full_profile_m_step", selected_rows[condition]["full"]),
            ("leave_one_out_iteration2", selected_rows[condition]["loo_iteration2"]),
            ("leave_one_out_converged", selected_rows[condition]["loo"]),
            (
                "spatial_block_crossfit_iteration2",
                selected_rows[condition]["crossfit_iteration2"],
            ),
            (
                "spatial_block_crossfit_converged",
                selected_rows[condition]["crossfit"],
            ),
        ):
            comparison_rows.append(
                _metric_row(condition=condition, method=method, row=row)
            )
    external = pd.read_csv(
        Path(args.external_comparison_csv).expanduser().resolve()
    )
    for condition, external_condition in (
        ("current_capture", "current_high_depth"),
        ("capture_thinned", "capture_thinned"),
    ):
        row = external.loc[
            (external["condition"] == external_condition)
            & (external["method"] == "Bin2Cell")
        ].iloc[0]
        comparison_rows.append(
            {
                "condition": condition,
                "method": "Bin2Cell",
                "mean_core_cell_iou": float(row["mean_core_cell_iou"]),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output / "method_comparison.csv", index=False)

    evolution_rows: list[dict[str, Any]] = []
    for condition, frames in conditions.items():
        for stage_name, row in (
            (
                "iteration_1",
                _select(frames["iteration1"], kappa=selected_kappa),
            ),
            ("iteration_2", selected_rows[condition]["crossfit_iteration2"]),
            ("converged", selected_rows[condition]["crossfit"]),
        ):
            evolution_rows.append(
                {
                    "condition": condition,
                    "stage": stage_name,
                    "relative_prior_strength": selected_kappa,
                    "spatial_crossfit_block_size_um": selected_block,
                    **{name: row.get(name, np.nan) for name in METRICS},
                }
            )
    evolution = pd.DataFrame(evolution_rows)
    evolution.to_csv(output / "selected_evolution.csv", index=False)

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
    for condition in conditions:
        runs = (
            (
                args.crossfit_current_iteration2_run,
                args.crossfit_current_run,
                args.profile_damping_current_run,
            )
            if condition == "current_capture"
            else (
                args.crossfit_capture_iteration2_run,
                args.crossfit_capture_run,
                args.profile_damping_capture_run,
            )
        )
        iteration2_run, converged_run, frozen_run = runs
        transition_rows.append(
            _transition(
                condition=condition,
                comparison="iteration2_to_converged",
                before_run=iteration2_run,
                after_run=converged_run,
                before_row=selected_rows[condition]["crossfit_iteration2"],
                after_row=selected_rows[condition]["crossfit"],
                gt_core=gt_core,
            )
        )
        transition_rows.append(
            _transition(
                condition=condition,
                comparison="frozen_to_crossfit_converged",
                before_run=frozen_run,
                after_run=converged_run,
                before_row=selected_rows[condition]["frozen"],
                after_row=selected_rows[condition]["crossfit"],
                gt_core=gt_core,
            )
        )
        cell_summary, cell_detail = _per_cell_evolution(
            condition=condition,
            before_run=iteration2_run,
            after_run=converged_run,
            kappa=selected_kappa,
            block_size_um=selected_block,
        )
        cell_summary_rows.append(cell_summary)
        cell_detail_rows.append(cell_detail)
    transitions = pd.DataFrame(transition_rows)
    transitions.to_csv(output / "core_truth_owner_transitions.csv", index=False)
    per_cell_summary = pd.DataFrame(cell_summary_rows)
    per_cell_summary.to_csv(output / "per_cell_evolution_summary.csv", index=False)
    pd.concat(cell_detail_rows, ignore_index=True).to_csv(
        output / "per_cell_iou_evolution.csv",
        index=False,
    )
    external.to_csv(output / "external_method_reference.csv", index=False)

    summary = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "EM-only; four fixed validation patches; 43 core cells",
        "selected_converged_kappa": selected_kappa,
        "selected_spatial_crossfit_block_size_um": selected_block,
        "selected_converged_result": selected.to_dict(),
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
        path=output / "README.md",
        selected=selected,
        comparison=comparison,
        evolution=evolution,
        transitions=transitions,
        per_cell_summary=per_cell_summary,
    )
    print(joint.head(10).to_string(index=False))
    print(evolution.to_string(index=False))
    print(transitions.to_string(index=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
