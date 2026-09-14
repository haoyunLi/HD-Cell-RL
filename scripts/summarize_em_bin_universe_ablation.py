#!/usr/bin/env python
"""Summarize EM bin-universe/background ablations with matched denominators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--gt-cell-bins-path", required=True)
    return parser.parse_args()


def _latest_child(parent: Path, pattern: str) -> Path:
    matches = sorted(path for path in parent.glob(pattern) if path.is_dir())
    if not matches:
        raise FileNotFoundError(f"no directory matching {pattern!r} under {parent}")
    return matches[-1]


def _variant_paths(run_root: Path) -> list[tuple[str, Path, Path]]:
    initialization_root = run_root / "initializations"
    evaluation_root = run_root / "evaluations"
    out: list[tuple[str, Path, Path]] = []
    for variant_dir in sorted(path for path in initialization_root.iterdir() if path.is_dir()):
        label = variant_dir.name
        initialization = _latest_child(variant_dir, "em_only_*")
        evaluation = _latest_child(evaluation_root, f"em_bin_universe_{label}_*")
        out.append((label, initialization, evaluation))
    if not out:
        raise ValueError(f"no EM variants found under {initialization_root}")
    return out


def _artifact_counts(initialization: Path) -> dict[str, Any]:
    n_bins = 0
    n_locked = 0
    n_unassigned = 0
    zero_conf = 0
    zero_conf_unassigned = 0
    positive_conf_unassigned = 0
    background_sum = 0.0
    for path in sorted((initialization / "em_assignment").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            confidence = np.asarray(data["expression_confidence"], dtype=np.float64)
            locked = np.asarray(data["is_nuclear_locked"], dtype=bool)
            is_unassigned = (
                np.asarray(data["is_unassigned"], dtype=bool)
                if "is_unassigned" in data.files
                else np.zeros(confidence.shape, dtype=bool)
            )
            background = (
                np.asarray(data["background_probability"], dtype=np.float64)
                if "background_probability" in data.files
                else np.zeros(confidence.shape, dtype=np.float64)
            )
            zero = (confidence == 0.0) & (~locked)
            positive = (confidence > 0.0) & (~locked)
            n_bins += int(confidence.size)
            n_locked += int(np.sum(locked))
            n_unassigned += int(np.sum(is_unassigned))
            zero_conf += int(np.sum(zero))
            zero_conf_unassigned += int(np.sum(zero & is_unassigned))
            positive_conf_unassigned += int(np.sum(positive & is_unassigned))
            background_sum += float(np.sum(background))
    if n_bins <= 0:
        raise ValueError(f"no EM artifacts found under {initialization}")
    return {
        "n_included_bins": n_bins,
        "n_nuclear_locked": n_locked,
        "n_unassigned": n_unassigned,
        "n_zero_conf_non_nuclear_included": zero_conf,
        "n_zero_conf_non_nuclear_unassigned": zero_conf_unassigned,
        "n_positive_conf_non_nuclear_unassigned": positive_conf_unassigned,
        "mean_background_probability": background_sum / n_bins,
    }


def _prediction_owner_map(
    *,
    assignments: pd.DataFrame,
    per_cell: pd.DataFrame,
) -> dict[str, str]:
    matched = {
        str(row.cell_id): str(row.matched_gt_cell_id)
        for row in per_cell.itertuples(index=False)
        if pd.notna(row.matched_gt_cell_id)
    }
    owner: dict[str, str] = {}
    for row in assignments.itertuples(index=False):
        gt_cell_id = matched.get(str(row.cell_id))
        if gt_cell_id is None:
            continue
        barcode = str(row.barcode)
        previous = owner.get(barcode)
        if previous is not None and previous != gt_cell_id:
            raise ValueError(
                f"prediction assigns barcode {barcode!r} to multiple evaluated cells"
            )
        owner[barcode] = gt_cell_id
    return owner


def _boundary_metrics(
    *,
    assignments: pd.DataFrame,
    per_cell: pd.DataFrame,
    gt: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    pred_owner = _prediction_owner_map(assignments=assignments, per_cell=per_cell)
    boundary_gt = gt.loc[gt["is_boundary"].astype(int) == 1, ["cell_id", "barcode"]].copy()
    correct = np.asarray(
        [pred_owner.get(str(row.barcode)) == str(row.cell_id) for row in boundary_gt.itertuples(index=False)],
        dtype=bool,
    )
    per_cell_rows: list[dict[str, Any]] = []
    assignments_by_cell = {
        str(cell_id): group.copy()
        for cell_id, group in assignments.groupby(assignments["cell_id"].astype(str), sort=False)
    }
    for row in per_cell.itertuples(index=False):
        episode_cell = str(row.cell_id)
        if pd.isna(row.matched_gt_cell_id):
            continue
        gt_cell = str(row.matched_gt_cell_id)
        pred = assignments_by_cell.get(episode_cell)
        pred_coords = set()
        if pred is not None:
            pred_coords = set(
                zip(
                    pred["array_row"].astype(int).tolist(),
                    pred["array_col"].astype(int).tolist(),
                    strict=True,
                )
            )
        pred_boundary = _boundary_from_coords(pred_coords)
        gt_cell_rows = gt.loc[gt["cell_id"].astype(str) == gt_cell]
        gt_boundary = set(
            zip(
                gt_cell_rows.loc[gt_cell_rows["is_boundary"].astype(int) == 1, "array_row"].astype(int).tolist(),
                gt_cell_rows.loc[gt_cell_rows["is_boundary"].astype(int) == 1, "array_col"].astype(int).tolist(),
                strict=True,
            )
        )
        overlap = len(pred_boundary & gt_boundary)
        precision = overlap / len(pred_boundary) if pred_boundary else 0.0
        recall = overlap / len(gt_boundary) if gt_boundary else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        area_bias = (
            (float(row.pred_n_bins) - float(row.gt_n_bins)) / float(row.gt_n_bins)
            if float(row.gt_n_bins) > 0.0
            else np.nan
        )
        per_cell_rows.append(
            {
                "cell_id": episode_cell,
                "matched_gt_cell_id": gt_cell,
                "pred_n_bins": int(row.pred_n_bins),
                "gt_n_bins": int(row.gt_n_bins),
                "area_bias_fraction": area_bias,
                "pred_iou": float(row.pred_iou),
                "pred_precision": float(row.pred_precision),
                "pred_recall": float(row.pred_recall),
                "boundary_precision": precision,
                "boundary_recall": recall,
                "boundary_f1": f1,
            }
        )
    per_cell_out = pd.DataFrame(per_cell_rows)
    return (
        {
            "gt_boundary_owner_accuracy": float(np.mean(correct)) if correct.size else np.nan,
            "n_gt_boundary_bins": int(correct.size),
            "mean_boundary_f1": float(per_cell_out["boundary_f1"].mean()),
            "mean_area_bias_fraction": float(per_cell_out["area_bias_fraction"].mean()),
            "mean_absolute_area_bias_fraction": float(
                per_cell_out["area_bias_fraction"].abs().mean()
            ),
        },
        per_cell_out,
    )


def _boundary_from_coords(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    if not coords:
        return set()
    offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    return {
        coord
        for coord in coords
        if any((coord[0] + dr, coord[1] + dc) not in coords for dr, dc in offsets)
    }


def _load_gt(path: Path, matched_ids: set[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    columns = ["cell_id", "barcode", "array_row", "array_col", "is_boundary"]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=1_000_000):
        chunk["cell_id"] = chunk["cell_id"].astype(str)
        keep = chunk["cell_id"].isin(matched_ids)
        if bool(keep.any()):
            pieces.append(chunk.loc[keep].copy())
    if not pieces:
        raise ValueError("no requested cells found in dominant-owner GT")
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    args = _parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    variants = _variant_paths(run_root)
    first_per_cell = pd.read_csv(variants[0][2] / "per_episode.csv")
    matched_ids = set(first_per_cell["matched_gt_cell_id"].dropna().astype(str))
    gt = _load_gt(Path(args.gt_cell_bins_path).expanduser().resolve(), matched_ids)

    raw: list[tuple[str, Path, Path, dict[str, Any], dict[str, Any], pd.DataFrame]] = []
    full_bin_count = 0
    for label, initialization, evaluation in variants:
        with (evaluation / "summary.json").open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        counts = _artifact_counts(initialization)
        per_cell = pd.read_csv(evaluation / "per_episode.csv")
        raw.append((label, initialization, evaluation, summary, counts, per_cell))
        full_bin_count = max(full_bin_count, int(counts["n_included_bins"]))

    comparison_rows: list[dict[str, Any]] = []
    per_cell_frames: list[pd.DataFrame] = []
    for label, initialization, evaluation, summary, counts, per_cell in raw:
        assignments = pd.read_csv(initialization / "source" / "assignments.csv")
        boundary, per_cell_out = _boundary_metrics(
            assignments=assignments,
            per_cell=per_cell,
            gt=gt,
        )
        per_cell_out.insert(0, "variant", label)
        per_cell_frames.append(per_cell_out)
        hard_assigned = int(counts["n_included_bins"]) - int(counts["n_unassigned"])
        comparison_rows.append(
            {
                "variant": label,
                "n_full_universe_bins": full_bin_count,
                **counts,
                "input_coverage": int(counts["n_included_bins"]) / full_bin_count,
                "hard_assignment_coverage": hard_assigned / full_bin_count,
                "unassigned_fraction_of_included": int(counts["n_unassigned"])
                / int(counts["n_included_bins"]),
                "mean_core_cell_iou": float(summary["mean_pred_iou"]),
                "mean_core_cell_precision": float(summary["mean_pred_precision"]),
                "mean_core_cell_recall": float(summary["mean_pred_recall"]),
                "mean_fractional_iou": float(summary["mean_pred_fractional_iou"]),
                "mean_gene_spearman_r": float(summary["mean_gene_spearman_r"]),
                **boundary,
                "initialization_dir": str(initialization),
                "evaluation_dir": str(evaluation),
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values("variant").reset_index(drop=True)
    per_cell_all = pd.concat(per_cell_frames, ignore_index=True)
    comparison_path = run_root / "method_comparison.csv"
    per_cell_path = run_root / "per_cell_diagnostics.csv"
    summary_path = run_root / "summary.json"
    comparison.to_csv(comparison_path, index=False)
    per_cell_all.to_csv(per_cell_path, index=False)
    payload = {
        "run_root": str(run_root),
        "gt_cell_bins_path": str(Path(args.gt_cell_bins_path).expanduser().resolve()),
        "metric_scope": {
            "cell_metrics": "43 matched core cells across four patches",
            "em_universe": "patch-specific unique bins inside outer bounds plus locked margin nuclear seeds",
            "boundary_owner_accuracy": "micro accuracy on dominant-owner GT bins marked is_boundary=1",
            "boundary_f1": "macro exact-bin F1 between predicted 8-neighbor boundary and GT is_boundary bins",
            "area_bias": "(predicted bin count - dominant-owner GT bin count) / GT bin count",
        },
        "variants": comparison.to_dict(orient="records"),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(comparison.to_string(index=False))
    print(f"Wrote {comparison_path}")
    print(f"Wrote {per_cell_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
