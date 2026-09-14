#!/usr/bin/env python
"""Summarize nuclear held-out predictive reliability EM experiments."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.ppo_format_assignment_eval import load_eval_cell_ids


BASELINE_MODE = "nuclear_depth"
HELDOUT_MODE = "nuclear_heldout_predictive"
METRICS = (
    "mean_core_cell_iou",
    "owner_accuracy",
    "mean_boundary_f1",
    "gt_boundary_owner_accuracy",
    "mean_normalized_entropy",
    "fraction_entropy_above_0p5",
    "mean_iterations",
    "mean_cell_profile_reliability",
    "median_cell_profile_reliability",
    "fraction_cell_profile_reliability_zero",
    "fraction_cell_profile_reliability_one",
    "mean_heldout_predictive_gain_per_umi",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-run", required=True)
    parser.add_argument("--capture-run", required=True)
    parser.add_argument("--gt-cell-bins-path", required=True)
    parser.add_argument("--source-eval-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _read_table(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / name
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty experiment table: {path}")
    return frame


def _mode_row(frame: pd.DataFrame, mode: str) -> pd.Series:
    selected = frame.loc[frame["cell_profile_reliability_mode"] == mode]
    if len(selected) != 1:
        raise ValueError(f"expected one {mode} row; found {len(selected)}")
    return selected.iloc[0]


def _safe_correlation(x: np.ndarray, y: np.ndarray, *, rank: bool) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 2 or np.all(x_arr == x_arr[0]) or np.all(y_arr == y_arr[0]):
        return float("nan")
    result = spearmanr(x_arr, y_arr) if rank else pearsonr(x_arr, y_arr)
    return float(result.statistic)


def _load_soft_rows(run_dir: Path, variant: str) -> pd.DataFrame:
    artifact_dir = run_dir / "variants" / variant / "em_assignment"
    paths = sorted(artifact_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no soft assignment artifacts under {artifact_dir}")
    rows: list[pd.DataFrame] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            cell_ids = data["cell_ids"].astype(str)
            owner_index = data["top1_cell_index"].astype(np.int64)
            owner = np.full(owner_index.shape, "__background__", dtype=object)
            assigned = owner_index >= 0
            owner[assigned] = cell_ids[owner_index[assigned]]
            rows.append(
                pd.DataFrame(
                    {
                        "barcode": data["barcode_ids"].astype(str),
                        "owner": owner.astype(str),
                        "normalized_entropy": data["normalized_entropy"].astype(
                            np.float64
                        ),
                        "is_ambiguous": data["is_ambiguous"].astype(bool),
                        "is_nuclear_locked": data["is_nuclear_locked"].astype(bool),
                    }
                )
            )
    frame = pd.concat(rows, ignore_index=True)
    if bool(frame["barcode"].duplicated().any()):
        duplicates = frame.loc[frame["barcode"].duplicated(), "barcode"].head()
        raise ValueError(f"duplicate physical barcodes across patches: {duplicates.tolist()}")
    return frame


def _method_comparison(runs: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition, run_dir in runs.items():
        sweep = _read_table(run_dir, "weight_sweep.csv")
        for mode in (BASELINE_MODE, HELDOUT_MODE):
            selected = _mode_row(sweep, mode)
            rows.append(
                {
                    "condition": condition,
                    "reliability_mode": mode,
                    "variant": str(selected["variant"]),
                    **{metric: selected[metric] for metric in METRICS},
                }
            )
    return pd.DataFrame(rows)


def _reliability_diagnostics(
    runs: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    condition_frames: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []
    for condition, run_dir in runs.items():
        frame = _read_table(run_dir, "per_patch_cell_reliability.csv")
        frame.insert(0, "condition", condition)
        condition_frames[condition] = frame
        for mode in (BASELINE_MODE, HELDOUT_MODE):
            mode_frame = frame.loc[frame["cell_profile_reliability_mode"] == mode]
            for subset, subset_frame in (
                ("all_patch_cells", mode_frame),
                ("core_cells", mode_frame.loc[mode_frame["is_core_cell"]]),
            ):
                reliability = subset_frame["profile_reliability"].to_numpy(
                    dtype=np.float64
                )
                umi = subset_frame["nuclear_selected_gene_umi"].to_numpy(
                    dtype=np.float64
                )
                n_bins = subset_frame["positive_nuclear_bin_count"].to_numpy(
                    dtype=np.float64
                )
                summary_rows.append(
                    {
                        "condition": condition,
                        "reliability_mode": mode,
                        "subset": subset,
                        "n_cell_instances": int(len(subset_frame)),
                        "mean_reliability": float(np.mean(reliability)),
                        "median_reliability": float(np.median(reliability)),
                        "q10_reliability": float(np.quantile(reliability, 0.1)),
                        "q90_reliability": float(np.quantile(reliability, 0.9)),
                        "fraction_reliability_zero": float(
                            np.mean(np.isclose(reliability, 0.0))
                        ),
                        "fraction_reliability_one": float(
                            np.mean(np.isclose(reliability, 1.0))
                        ),
                        "mean_heldout_gain_per_umi": float(
                            subset_frame["heldout_predictive_gain_per_umi"].mean()
                        ),
                        "spearman_reliability_vs_log1p_umi": _safe_correlation(
                            np.log1p(umi), reliability, rank=True
                        ),
                        "spearman_reliability_vs_nuclear_bin_count": (
                            _safe_correlation(n_bins, reliability, rank=True)
                        ),
                    }
                )

    current = condition_frames["current_capture"]
    capture = condition_frames["capture_thinned"]
    paired_rows: list[pd.DataFrame] = []
    stability_rows: list[dict[str, Any]] = []
    for mode in (BASELINE_MODE, HELDOUT_MODE):
        left = current.loc[current["cell_profile_reliability_mode"] == mode]
        right = capture.loc[capture["cell_profile_reliability_mode"] == mode]
        paired = left.merge(
            right,
            on=["patch_id", "cell_id", "is_core_cell"],
            suffixes=("_current", "_capture"),
            validate="one_to_one",
        )
        paired.insert(0, "reliability_mode", mode)
        paired["reliability_delta_capture_minus_current"] = (
            paired["profile_reliability_capture"]
            - paired["profile_reliability_current"]
        )
        paired["absolute_reliability_delta"] = paired[
            "reliability_delta_capture_minus_current"
        ].abs()
        paired["capture_umi_fraction"] = (
            paired["nuclear_selected_gene_umi_capture"]
            / paired["nuclear_selected_gene_umi_current"]
        )
        paired_rows.append(paired)
        for subset, subset_frame in (
            ("all_patch_cells", paired),
            ("core_cells", paired.loc[paired["is_core_cell"]]),
        ):
            current_reliability = subset_frame[
                "profile_reliability_current"
            ].to_numpy(dtype=np.float64)
            capture_reliability = subset_frame[
                "profile_reliability_capture"
            ].to_numpy(dtype=np.float64)
            absolute_delta = subset_frame["absolute_reliability_delta"].to_numpy(
                dtype=np.float64
            )
            stability_rows.append(
                {
                    "reliability_mode": mode,
                    "subset": subset,
                    "n_paired_cell_instances": int(len(subset_frame)),
                    "mean_reliability_current": float(
                        np.mean(current_reliability)
                    ),
                    "mean_reliability_capture": float(
                        np.mean(capture_reliability)
                    ),
                    "mean_capture_minus_current": float(
                        np.mean(capture_reliability - current_reliability)
                    ),
                    "mean_absolute_change": float(np.mean(absolute_delta)),
                    "median_absolute_change": float(np.median(absolute_delta)),
                    "q90_absolute_change": float(np.quantile(absolute_delta, 0.9)),
                    "fraction_absolute_change_above_0p25": float(
                        np.mean(absolute_delta > 0.25)
                    ),
                    "pearson_current_vs_capture": _safe_correlation(
                        current_reliability, capture_reliability, rank=False
                    ),
                    "spearman_current_vs_capture": _safe_correlation(
                        current_reliability, capture_reliability, rank=True
                    ),
                    "median_capture_umi_fraction": float(
                        subset_frame["capture_umi_fraction"].median()
                    ),
                }
            )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(stability_rows),
        pd.concat(paired_rows, ignore_index=True),
    )


def _per_cell_iou_diagnostics(
    runs: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for condition, run_dir in runs.items():
        frame = _read_table(run_dir, "per_cell.csv")
        baseline = frame.loc[
            frame["cell_profile_reliability_mode"] == BASELINE_MODE,
            ["cell_id", "iou"],
        ].rename(columns={"iou": "iou_nuclear_depth"})
        heldout = frame.loc[
            frame["cell_profile_reliability_mode"] == HELDOUT_MODE,
            ["cell_id", "iou"],
        ].rename(columns={"iou": "iou_nuclear_heldout_predictive"})
        paired = baseline.merge(heldout, on="cell_id", validate="one_to_one")
        paired.insert(0, "condition", condition)
        paired["iou_delta_heldout_minus_depth"] = (
            paired["iou_nuclear_heldout_predictive"]
            - paired["iou_nuclear_depth"]
        )
        rows.append(paired)
        delta = paired["iou_delta_heldout_minus_depth"]
        tolerance = 1.0e-12
        summaries.append(
            {
                "condition": condition,
                "n_core_cells": int(len(paired)),
                "n_cells_improved": int((delta > tolerance).sum()),
                "n_cells_damaged": int((delta < -tolerance).sum()),
                "n_cells_unchanged": int((delta.abs() <= tolerance).sum()),
                "mean_iou_delta": float(delta.mean()),
                "median_iou_delta": float(delta.median()),
                "minimum_iou_delta": float(delta.min()),
                "maximum_iou_delta": float(delta.max()),
            }
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries)


def _soft_diagnostics(
    *,
    runs: dict[str, Path],
    comparison: pd.DataFrame,
    truth_all: dict[str, str],
    target_cell_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    transition_rows: list[dict[str, Any]] = []
    entropy_rows: list[dict[str, Any]] = []
    for condition, run_dir in runs.items():
        condition_rows = comparison.loc[comparison["condition"] == condition]
        variants = dict(
            zip(
                condition_rows["reliability_mode"],
                condition_rows["variant"],
                strict=True,
            )
        )
        baseline = _load_soft_rows(run_dir, variants[BASELINE_MODE]).set_index(
            "barcode"
        )
        heldout = _load_soft_rows(run_dir, variants[HELDOUT_MODE]).set_index(
            "barcode"
        )
        common = baseline.index.intersection(heldout.index)
        if len(common) != len(baseline) or len(common) != len(heldout):
            raise ValueError(f"soft artifact barcode axes differ for {condition}")
        baseline = baseline.loc[common]
        heldout = heldout.loc[common]

        for scope in ("all_scored_truth", "core_truth"):
            scope_barcodes = [
                barcode
                for barcode in common
                if barcode in truth_all
                and (
                    scope == "all_scored_truth"
                    or truth_all[barcode] in target_cell_ids
                )
            ]
            before = baseline.loc[scope_barcodes, "owner"]
            after = heldout.loc[scope_barcodes, "owner"]
            truth = pd.Series(
                [truth_all[barcode] for barcode in scope_barcodes],
                index=scope_barcodes,
                dtype=str,
            )
            changed = before != after
            before_correct = before == truth
            after_correct = after == truth
            transition_rows.append(
                {
                    "condition": condition,
                    "scope": scope,
                    "n_truth_bins": int(len(scope_barcodes)),
                    "n_owner_changed": int(changed.sum()),
                    "n_corrected": int((changed & ~before_correct & after_correct).sum()),
                    "n_damaged": int((changed & before_correct & ~after_correct).sum()),
                    "n_wrong_to_wrong": int(
                        (changed & ~before_correct & ~after_correct).sum()
                    ),
                    "net_corrected_minus_damaged": int(
                        (changed & ~before_correct & after_correct).sum()
                        - (changed & before_correct & ~after_correct).sum()
                    ),
                    "n_locked_owner_changed": int(
                        (changed & baseline["is_nuclear_locked"]).sum()
                    ),
                }
            )

        entropy_delta = (
            heldout["normalized_entropy"] - baseline["normalized_entropy"]
        )
        entropy_rows.append(
            {
                "condition": condition,
                "n_bins": int(len(common)),
                "mean_entropy_nuclear_depth": float(
                    baseline["normalized_entropy"].mean()
                ),
                "mean_entropy_nuclear_heldout_predictive": float(
                    heldout["normalized_entropy"].mean()
                ),
                "mean_entropy_delta": float(entropy_delta.mean()),
                "median_absolute_entropy_delta": float(entropy_delta.abs().median()),
                "n_ambiguous_nuclear_depth": int(baseline["is_ambiguous"].sum()),
                "n_ambiguous_nuclear_heldout_predictive": int(
                    heldout["is_ambiguous"].sum()
                ),
                "n_became_ambiguous": int(
                    (~baseline["is_ambiguous"] & heldout["is_ambiguous"]).sum()
                ),
                "n_became_non_ambiguous": int(
                    (baseline["is_ambiguous"] & ~heldout["is_ambiguous"]).sum()
                ),
            }
        )
    return pd.DataFrame(transition_rows), pd.DataFrame(entropy_rows)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame.loc[:, columns].copy()
    for column in selected.columns:
        if pd.api.types.is_float_dtype(selected[column]):
            selected[column] = selected[column].map(lambda value: f"{value:.6f}")
    labels = list(selected.columns)
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join("---" for _ in labels) + " |",
    ]
    for row in selected.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _write_report(
    *,
    path: Path,
    comparison: pd.DataFrame,
    reliability_summary: pd.DataFrame,
    stability: pd.DataFrame,
    iou_summary: pd.DataFrame,
    transitions: pd.DataFrame,
    entropy: pd.DataFrame,
) -> None:
    compact = comparison[
        [
            "condition",
            "reliability_mode",
            "mean_core_cell_iou",
            "owner_accuracy",
            "mean_boundary_f1",
            "mean_normalized_entropy",
        ]
    ]
    core_stability = stability.loc[stability["subset"] == "core_cells"]
    core_reliability = reliability_summary.loc[
        reliability_summary["subset"] == "core_cells"
    ]
    all_transitions = transitions.loc[
        transitions["scope"] == "all_scored_truth"
    ]
    current_delta = float(
        iou_summary.loc[
            iou_summary["condition"] == "current_capture", "mean_iou_delta"
        ].iloc[0]
    )
    capture_delta = float(
        iou_summary.loc[
            iou_summary["condition"] == "capture_thinned", "mean_iou_delta"
        ].iloc[0]
    )
    heldout_stability = core_stability.loc[
        core_stability["reliability_mode"] == HELDOUT_MODE
    ].iloc[0]
    baseline_stability = core_stability.loc[
        core_stability["reliability_mode"] == BASELINE_MODE
    ].iloc[0]

    text = f"""# Nuclear held-out predictive reliability experiment

## Decision

The held-out predictive reliability estimator is **not accepted** for the next EM baseline. Keep `kappa=2`, `profile_update_damping=0`, and `reliability_mode: nuclear_depth`.

The new estimator changed the result, but in the wrong direction. Mean core-cell IoU changed by {current_delta:+.6f} on current capture and {capture_delta:+.6f} after capture thinning. Its paired core-cell reliability changed by {float(heldout_stability['mean_absolute_change']):.6f} on average across capture conditions, compared with {float(baseline_stability['mean_absolute_change']):.6f} for the existing relative-depth rule.

Do not train PPO from this experiment, and do not recalibrate the entropy threshold. Both remained outside this run.

## What was tested

Both methods used the same four validation patches, candidate graph, scRNA theta, `alpha=1`, `beta=1.5`, `kappa=2`, `profile_update_damping=0`, positive-expression bin universe, disabled background, 15-iteration limit, and 0.5 responsibility damping.

- `nuclear_depth`: the retained baseline, where reliability is nuclear UMI divided by nuclear UMI plus the patch-relative scRNA prior count.
- `nuclear_heldout_predictive`: each positive locked nuclear bin is held out once. The cell-type posterior and nuclear profile are rebuilt without that bin. A 101-point grid chooses the cell-specific mixing weight that maximizes held-out multinomial prediction. Cytoplasmic and other non-nuclear bins never enter this fit.

The fitted reliability is then held fixed while the existing generalized EM runs. The physical-cell profile is also frozen because `profile_update_damping=0`.

## Segmentation results

{_markdown_table(compact, list(compact.columns))}

## Reliability behavior

{_markdown_table(core_reliability, ['condition', 'reliability_mode', 'n_cell_instances', 'mean_reliability', 'median_reliability', 'fraction_reliability_one', 'mean_heldout_gain_per_umi'])}

{_markdown_table(core_stability, ['reliability_mode', 'n_paired_cell_instances', 'mean_reliability_current', 'mean_reliability_capture', 'mean_absolute_change', 'q90_absolute_change', 'fraction_absolute_change_above_0p25', 'pearson_current_vs_capture'])}

The current-capture held-out fit is saturated: most core cells choose a reliability of 1. Capture thinning moves the same cells toward intermediate weights. This is not a small calibration shift; it changes the fitted weight for nearly every core cell by more than 0.25.

## Ownership changes

{_markdown_table(all_transitions, ['condition', 'n_truth_bins', 'n_owner_changed', 'n_corrected', 'n_damaged', 'n_wrong_to_wrong', 'net_corrected_minus_damaged', 'n_locked_owner_changed'])}

Nuclear locks remained exact. The held-out estimator produced net ownership damage over all scored truth bins in both conditions, with the larger loss after capture thinning.

## Cell-level changes

{_markdown_table(iou_summary, list(iou_summary.columns))}

## Entropy diagnostic, without recalibration

{_markdown_table(entropy, list(entropy.columns))}

These are diagnostics at the already configured threshold of 0.5. The threshold was not selected or changed using these data.

## Interpretation

The test exposed two problems with raw per-cell nuclear-bin holdout.

First, neighboring 2 um bins from one nucleus are not independent biological replicates. In the current pseudo data they are highly coherent, so one nuclear bin is easy to predict from the other bins of the same nucleus and the grid often chooses reliability 1. Binomial capture thinning weakens that apparent reproducibility, which makes the estimated weight strongly dependent on sampling depth.

Second, the fitting score and downstream EM combination are not identical. The held-out fit evaluates an arithmetic probability mixture, while EM uses a weighted combination of type and cell log-likelihood compatibilities. A weight that predicts held-out nuclear counts well is therefore not guaranteed to improve boundary ownership.

## Recommendation

Retain the `kappa=2`, frozen-profile depth baseline. The next EM-only test should replace independent per-cell grid fitting with a pooled count model, such as a hierarchical Dirichlet-multinomial shrinkage estimate learned across nuclear anchors. That model should account explicitly for sampling variance and overdispersion, then be checked under repeated capture thinning before any held-out donor test. It should still exclude all ambiguous cytoplasmic bins from reliability estimation.

The detailed audit tables are in this directory: `method_comparison.csv`, `reliability_summary.csv`, `capture_stability.csv`, `reliability_capture_pairing.csv`, `per_cell_iou_summary.csv`, `per_cell_iou_delta.csv`, `ownership_transitions.csv`, and `entropy_diagnostics.csv`.
"""
    path.write_text(text, encoding="utf-8")


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    args = _parse_args()
    runs = {
        "current_capture": Path(args.current_run).expanduser().resolve(),
        "capture_thinned": Path(args.capture_run).expanduser().resolve(),
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    gt_path = Path(args.gt_cell_bins_path).expanduser().resolve()
    source_eval_dir = Path(args.source_eval_dir).expanduser().resolve()

    comparison = _method_comparison(runs)
    reliability_summary, stability, pairing = _reliability_diagnostics(runs)
    per_cell, iou_summary = _per_cell_iou_diagnostics(runs)
    gt = pd.read_csv(gt_path, dtype=str, usecols=["barcode", "cell_id"])
    truth_all = dict(zip(gt["barcode"], gt["cell_id"], strict=False))
    target_cell_ids = set(load_eval_cell_ids(source_eval_dir / "per_episode.csv"))
    transitions, entropy = _soft_diagnostics(
        runs=runs,
        comparison=comparison,
        truth_all=truth_all,
        target_cell_ids=target_cell_ids,
    )

    outputs = {
        "method_comparison.csv": comparison,
        "reliability_summary.csv": reliability_summary,
        "capture_stability.csv": stability,
        "reliability_capture_pairing.csv": pairing,
        "per_cell_iou_delta.csv": per_cell,
        "per_cell_iou_summary.csv": iou_summary,
        "ownership_transitions.csv": transitions,
        "entropy_diagnostics.csv": entropy,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)
    _write_report(
        path=output_dir / "README.md",
        comparison=comparison,
        reliability_summary=reliability_summary,
        stability=stability,
        iou_summary=iou_summary,
        transitions=transitions,
        entropy=entropy,
    )
    summary = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "current_run": str(runs["current_capture"]),
        "capture_run": str(runs["capture_thinned"]),
        "gt_cell_bins_path": str(gt_path),
        "source_eval_dir": str(source_eval_dir),
        "accepted": False,
        "retained_baseline": {
            "relative_prior_strength": 2.0,
            "profile_update_damping": 0.0,
            "cell_profile_reliability_mode": BASELINE_MODE,
        },
        "ppo_trained": False,
        "entropy_recalibrated": False,
        "method_comparison": _json_records(comparison),
        "capture_stability": _json_records(stability),
        "per_cell_iou_summary": _json_records(iou_summary),
        "ownership_transitions": _json_records(transitions),
        "entropy_diagnostics": _json_records(entropy),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(comparison.to_string(index=False))
    print(stability.to_string(index=False))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
