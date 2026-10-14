#!/usr/bin/env python
"""Diagnose which global-objective components drive patch REPLACE actions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.ppo_eval_metrics import normalize_cell_id


OBJECTIVE_COMPONENTS = (
    "expression_confidence",
    "expression_compatibility",
    "distance_penalty",
    "overlap_penalty",
    "neighbor_support",
    "shape_prior",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-debug-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--global-delta-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--configured-max-steps", type=int, default=0)
    parser.add_argument(
        "--entropy-thresholds",
        default="0.5,0.9,0.95,0.97,0.98,0.99",
        help="Comma-separated normalized-entropy thresholds.",
    )
    parser.add_argument("--residual-tolerance", type=float, default=1.0e-5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    patch_debug_dir = Path(args.patch_debug_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    thresholds = tuple(
        float(item)
        for item in str(args.entropy_thresholds).split(",")
        if item.strip()
    )
    payloads = _load_patch_payloads(patch_debug_dir)
    action_rows = _collect_replace_rows(payloads)
    entropy_rows = _collect_entropy_rows(payloads, action_rows)
    actions = pd.DataFrame(action_rows)
    entropy = pd.DataFrame(entropy_rows)
    if actions.empty:
        raise RuntimeError("no applied REPLACE actions were found")
    if "reward_residual" not in actions:
        raise RuntimeError(
            "patch traces do not contain objective attribution; rerun checkpoint evaluation"
        )
    max_residual = float(actions["reward_residual"].abs().max())
    if max_residual > float(args.residual_tolerance):
        raise RuntimeError(
            "objective component sum does not reconstruct reward: "
            f"max residual {max_residual:.6g} > {args.residual_tolerance:.6g}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    actions.to_csv(output_dir / "replace_reward_attribution.csv", index=False)
    entropy.to_csv(output_dir / "em_entropy_bins.csv", index=False)
    summary_by_outcome = _component_summary_by_outcome(actions)
    summary_by_outcome.to_csv(
        output_dir / "component_summary_by_outcome.csv",
        index=False,
    )
    drop_one = _drop_one_counterfactual(
        actions,
        epsilon=float(args.global_delta_epsilon),
    )
    drop_one.to_csv(output_dir / "drop_one_counterfactual.csv", index=False)
    drop_one_summary = _drop_one_summary(drop_one)
    drop_one_summary.to_csv(
        output_dir / "drop_one_counterfactual_summary.csv",
        index=False,
    )
    scale_summary = _component_scale_summary(
        actions,
        epsilon=float(args.global_delta_epsilon),
    )
    scale_summary.to_csv(output_dir / "component_scale_summary.csv", index=False)
    entropy_calibration = _entropy_calibration(entropy)
    entropy_calibration.to_csv(
        output_dir / "entropy_calibration.csv",
        index=False,
    )
    threshold_summary = _entropy_threshold_summary(
        entropy,
        thresholds=thresholds,
    )
    threshold_summary.to_csv(
        output_dir / "entropy_threshold_summary.csv",
        index=False,
    )
    summary = _build_summary(
        payloads=payloads,
        actions=actions,
        max_residual=max_residual,
        configured_max_steps=int(args.configured_max_steps),
        drop_one_summary=drop_one_summary,
        scale_summary=scale_summary,
        threshold_summary=threshold_summary,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        _markdown_report(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Reward attribution report: {output_dir}")


def _load_patch_payloads(patch_debug_dir: Path) -> list[dict[str, Any]]:
    manifest_path = patch_debug_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payloads = []
    for item in manifest.get("patches", []):
        path = patch_debug_dir / str(item["file"])
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    if not payloads:
        raise RuntimeError("patch debug manifest contains no patches")
    return payloads


def _owner_mapping(payload: dict[str, Any]) -> tuple[dict[str, str], set[str]]:
    owner_to_gt: dict[str, str] = {}
    for cell in payload.get("cells", []):
        owner = normalize_cell_id(cell.get("cell_id"))
        gt_owner = normalize_cell_id(cell.get("matched_gt_cell_id"))
        if owner is not None and gt_owner is not None:
            owner_to_gt[owner] = gt_owner
    gt_ids = {
        value
        for row in payload.get("bins", [])
        if (value := normalize_cell_id(row.get("gt_owner_cell_id"))) is not None
    }
    return owner_to_gt, gt_ids


def _map_owner_to_gt(
    owner: Any,
    *,
    owner_to_gt: dict[str, str],
    gt_ids: set[str],
) -> str | None:
    normalized = normalize_cell_id(owner)
    if normalized is None:
        return None
    if normalized in owner_to_gt:
        return owner_to_gt[normalized]
    return normalized if normalized in gt_ids else None


def _outcome(
    *,
    gt_owner: str | None,
    old_gt_owner: str | None,
    new_gt_owner: str | None,
) -> str:
    if gt_owner is None:
        return "no_gt"
    before = old_gt_owner == gt_owner
    after = new_gt_owner == gt_owner
    if before and not after:
        return "damaged"
    if not before and after:
        return "corrected"
    if before and after:
        return "unchanged_correct"
    if old_gt_owner is None or new_gt_owner is None:
        return "unresolved_owner_mapping"
    return "wrong_to_wrong"


def _collect_replace_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        patch_id = str(payload["patch_id"])
        bins = {str(row["barcode"]): row for row in payload.get("bins", [])}
        owner_to_gt, gt_ids = _owner_mapping(payload)
        for step in payload.get("trajectory", {}).get("steps", []):
            actions = [
                row
                for row in step.get("actions", [])
                if str(row.get("type")) == "replace"
                and bool(row.get("applied", False))
            ]
            if not actions:
                continue
            if len(actions) != 1:
                raise RuntimeError(
                    "per-action attribution requires one applied action per step; "
                    f"patch {patch_id} step {step.get('step_index')} has {len(actions)}"
                )
            action = actions[0]
            barcode = str(action["barcode"])
            bin_row = bins.get(barcode, {})
            gt_owner = normalize_cell_id(bin_row.get("gt_owner_cell_id"))
            old_owner = normalize_cell_id(action.get("old_cell_id"))
            new_owner = normalize_cell_id(action.get("cell_id"))
            old_gt_owner = _map_owner_to_gt(
                old_owner,
                owner_to_gt=owner_to_gt,
                gt_ids=gt_ids,
            )
            new_gt_owner = _map_owner_to_gt(
                new_owner,
                owner_to_gt=owner_to_gt,
                gt_ids=gt_ids,
            )
            components = dict(step.get("objective_delta_components") or {})
            row = {
                "patch_id": patch_id,
                "step_index": int(step["step_index"]),
                "barcode": barcode,
                "old_cell_id": old_owner,
                "new_cell_id": new_owner,
                "gt_owner_cell_id": gt_owner,
                "old_matched_gt_cell_id": old_gt_owner,
                "new_matched_gt_cell_id": new_gt_owner,
                "outcome": _outcome(
                    gt_owner=gt_owner,
                    old_gt_owner=old_gt_owner,
                    new_gt_owner=new_gt_owner,
                ),
                "reward": float(step["reward"]),
                "em_normalized_entropy": _finite(
                    bin_row.get("em_normalized_entropy")
                ),
                "em_top1_probability": _finite(
                    bin_row.get("em_top1_probability")
                ),
                "em_margin": _finite(bin_row.get("em_margin")),
                "inside_core": bool(bin_row.get("inside_core", False)),
                "available_action_count": int(
                    step.get("available_action_count") or 0
                ),
                "chosen_action_reward": _finite(
                    step.get("chosen_action_reward")
                ),
                "best_available_action_reward": _finite(
                    step.get("best_available_action_reward")
                ),
                "chosen_action_reward_rank": int(
                    step.get("chosen_action_reward_rank") or 0
                ),
            }
            if (
                row["available_action_count"] > 1
                and row["chosen_action_reward_rank"] > 0
            ):
                row["chosen_action_rank_fraction"] = float(
                    (row["chosen_action_reward_rank"] - 1)
                    / (row["available_action_count"] - 1)
                )
            else:
                row["chosen_action_rank_fraction"] = 0.0
            if (
                row["chosen_action_reward"] is not None
                and row["best_available_action_reward"] is not None
                and row["best_available_action_reward"] > 0.0
            ):
                row["chosen_reward_fraction_of_best"] = float(
                    row["chosen_action_reward"]
                    / row["best_available_action_reward"]
                )
            else:
                row["chosen_reward_fraction_of_best"] = np.nan
            for key in OBJECTIVE_COMPONENTS:
                row[key] = float(components.get(key, 0.0))
            positive = {
                key: row[key]
                for key in OBJECTIVE_COMPONENTS
                if row[key] > 0.0
            }
            row["dominant_positive_component"] = max(positive, key=positive.get) if positive else None
            row["penalty_rounding_residual"] = float(
                components.get("penalty_rounding_residual", 0.0)
            )
            row["numeric_residual"] = float(
                components.get("numeric_residual", 0.0)
            )
            row["component_sum"] = float(
                components.get("component_sum", np.nan)
            )
            row["reward_residual"] = float(
                components.get("reward_residual", np.nan)
            )
            rows.append(row)
    return rows


def _collect_entropy_rows(
    payloads: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    modified = {(row["patch_id"], row["barcode"]) for row in action_rows}
    out: list[dict[str, Any]] = []
    for payload in payloads:
        patch_id = str(payload["patch_id"])
        owner_to_gt, gt_ids = _owner_mapping(payload)
        initial = {
            str(row["barcode"]): normalize_cell_id(row.get("cell_id"))
            for row in payload.get("trajectory", {}).get("initial_owners", [])
        }
        for row in payload.get("bins", []):
            entropy = _finite(row.get("em_normalized_entropy"))
            if entropy is None:
                continue
            barcode = str(row["barcode"])
            gt_owner = normalize_cell_id(row.get("gt_owner_cell_id"))
            initial_owner = initial.get(barcode)
            initial_gt_owner = _map_owner_to_gt(
                initial_owner,
                owner_to_gt=owner_to_gt,
                gt_ids=gt_ids,
            )
            out.append(
                {
                    "patch_id": patch_id,
                    "barcode": barcode,
                    "normalized_entropy": entropy,
                    "is_nuclear_locked": bool(
                        row.get("em_is_nuclear_locked", False)
                    ),
                    "is_ambiguous": bool(row.get("em_is_ambiguous", False)),
                    "initial_owner_cell_id": initial_owner,
                    "gt_owner_cell_id": gt_owner,
                    "initial_matched_gt_cell_id": initial_gt_owner,
                    "em_correct": (
                        np.nan
                        if gt_owner is None or initial_gt_owner is None
                        else bool(initial_gt_owner == gt_owner)
                    ),
                    "rl_modified": (patch_id, barcode) in modified,
                    "inside_core": bool(row.get("inside_core", False)),
                }
            )
    return out


def _component_summary_by_outcome(actions: pd.DataFrame) -> pd.DataFrame:
    columns = list(OBJECTIVE_COMPONENTS) + ["reward"]
    return (
        actions.groupby("outcome", dropna=False)[columns]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )


def _drop_one_counterfactual(
    actions: pd.DataFrame,
    *,
    epsilon: float,
) -> pd.DataFrame:
    rows = []
    for action in actions.to_dict(orient="records"):
        for component in OBJECTIVE_COMPONENTS:
            counterfactual = float(action["reward"]) - float(action[component])
            rows.append(
                {
                    "patch_id": action["patch_id"],
                    "step_index": action["step_index"],
                    "barcode": action["barcode"],
                    "outcome": action["outcome"],
                    "removed_component": component,
                    "original_reward": float(action["reward"]),
                    "removed_delta": float(action[component]),
                    "counterfactual_reward": counterfactual,
                    "would_remain_exposed": bool(
                        counterfactual > float(epsilon)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _drop_one_summary(drop_one: pd.DataFrame) -> pd.DataFrame:
    return (
        drop_one.groupby(["removed_component", "outcome"], dropna=False)
        .agg(
            n_actions=("barcode", "size"),
            n_rejected=(
                "would_remain_exposed",
                lambda values: int((~values).sum()),
            ),
            mean_counterfactual_reward=("counterfactual_reward", "mean"),
        )
        .reset_index()
    )


def _component_scale_summary(
    actions: pd.DataFrame,
    *,
    epsilon: float,
) -> pd.DataFrame:
    rows = []
    for component in OBJECTIVE_COMPONENTS:
        for scale in (0.0, 0.25, 0.5, 0.75, 1.0):
            counterfactual = (
                actions["reward"]
                + (float(scale) - 1.0) * actions[component]
            )
            for outcome, indices in actions.groupby("outcome").groups.items():
                values = counterfactual.loc[indices]
                rows.append(
                    {
                        "scaled_component": component,
                        "scale": float(scale),
                        "outcome": str(outcome),
                        "n_actions": int(len(values)),
                        "n_would_remain_exposed": int(
                            (values > float(epsilon)).sum()
                        ),
                        "mean_counterfactual_reward": float(values.mean()),
                    }
                )
    return pd.DataFrame(rows)


def _entropy_calibration(entropy: pd.DataFrame) -> pd.DataFrame:
    scored = entropy.loc[entropy["em_correct"].notna()].copy()
    edges = np.linspace(0.0, 1.0, 6)
    rows = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (scored["normalized_entropy"] >= lower) & (
            scored["normalized_entropy"] <= upper
            if index == len(edges) - 2
            else scored["normalized_entropy"] < upper
        )
        subset = scored.loc[mask]
        rows.append(
            {
                "entropy_min": float(lower),
                "entropy_max": float(upper),
                "n_bins": int(len(subset)),
                "em_accuracy": (
                    np.nan
                    if subset.empty
                    else float(subset["em_correct"].astype(float).mean())
                ),
                "rl_modified_fraction": (
                    np.nan
                    if subset.empty
                    else float(subset["rl_modified"].mean())
                ),
            }
        )
    return pd.DataFrame(rows)


def _entropy_threshold_summary(
    entropy: pd.DataFrame,
    *,
    thresholds: tuple[float, ...],
) -> pd.DataFrame:
    scored = entropy.loc[
        entropy["em_correct"].notna() & (~entropy["is_nuclear_locked"])
    ].copy()
    total_errors = int((~scored["em_correct"].astype(bool)).sum())
    rows = []
    for threshold in thresholds:
        eligible = scored.loc[
            scored["normalized_entropy"] > float(threshold)
        ]
        errors = int((~eligible["em_correct"].astype(bool)).sum())
        rows.append(
            {
                "entropy_threshold": float(threshold),
                "n_scored_non_nuclear_bins": int(len(scored)),
                "n_eligible_bins": int(len(eligible)),
                "eligible_fraction": (
                    0.0 if scored.empty else float(len(eligible) / len(scored))
                ),
                "eligible_em_error_rate": (
                    np.nan if eligible.empty else float(errors / len(eligible))
                ),
                "em_error_coverage": (
                    np.nan if total_errors == 0 else float(errors / total_errors)
                ),
                "n_rl_modified": int(eligible["rl_modified"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _build_summary(
    *,
    payloads: list[dict[str, Any]],
    actions: pd.DataFrame,
    max_residual: float,
    configured_max_steps: int,
    drop_one_summary: pd.DataFrame,
    scale_summary: pd.DataFrame,
    threshold_summary: pd.DataFrame,
) -> dict[str, Any]:
    outcome_counts = Counter(str(value) for value in actions["outcome"])
    component_totals = {
        key: float(actions[key].sum())
        for key in OBJECTIVE_COMPONENTS
    }
    n_steps = [int(payload.get("n_steps") or 0) for payload in payloads]
    dominant = (
        actions["dominant_positive_component"]
        .fillna("none")
        .value_counts()
        .sort_index()
        .to_dict()
    )
    dominant_by_outcome = pd.crosstab(
        actions["outcome"],
        actions["dominant_positive_component"].fillna("none"),
    )
    valid_rank = actions.loc[actions["chosen_action_reward_rank"] > 0]
    rank_by_outcome = (
        valid_rank.groupby("outcome")["chosen_action_reward_rank"]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
    )
    return {
        "n_patches": int(len(payloads)),
        "n_replace_actions": int(len(actions)),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "total_reward": float(actions["reward"].sum()),
        "component_delta_totals": component_totals,
        "dominant_positive_component_counts": {
            str(key): int(value) for key, value in dominant.items()
        },
        "dominant_positive_component_by_outcome": dominant_by_outcome.to_dict(
            orient="index"
        ),
        "action_choice_diagnostics": {
            "n_ranked_actions": int(len(valid_rank)),
            "fraction_rank_1": (
                None
                if valid_rank.empty
                else float((valid_rank["chosen_action_reward_rank"] == 1).mean())
            ),
            "mean_rank": (
                None
                if valid_rank.empty
                else float(valid_rank["chosen_action_reward_rank"].mean())
            ),
            "mean_rank_fraction": (
                None
                if valid_rank.empty
                else float(valid_rank["chosen_action_rank_fraction"].mean())
            ),
            "mean_chosen_reward_fraction_of_best": (
                None
                if valid_rank.empty
                else float(valid_rank["chosen_reward_fraction_of_best"].mean())
            ),
            "rank_by_outcome": rank_by_outcome.to_dict(orient="records"),
        },
        "max_absolute_reward_residual": float(max_residual),
        "reward_reconstruction_passed": True,
        "configured_max_steps": int(configured_max_steps),
        "patch_step_counts": n_steps,
        "all_patches_hit_configured_max_steps": bool(
            configured_max_steps > 0
            and n_steps
            and all(value == configured_max_steps for value in n_steps)
        ),
        "active_stop_action_available": False,
        "active_stop_action_note": (
            "TorchSingleAgentPatchEnv masks action 0 while positive global-delta "
            "ADD/REPLACE actions remain."
        ),
        "competition_margin_in_global_objective": False,
        "competition_margin_note": (
            "The completed overfit configuration set competition_margin_enabled=false; "
            "global objective contains the nucleus-overlap penalty instead."
        ),
        "drop_one_summary": drop_one_summary.replace(
            {np.nan: None}
        ).to_dict(orient="records"),
        "component_scale_summary": scale_summary.replace(
            {np.nan: None}
        ).to_dict(orient="records"),
        "entropy_threshold_summary": threshold_summary.replace(
            {np.nan: None}
        ).to_dict(orient="records"),
    }


def _markdown_report(summary: dict[str, Any]) -> str:
    choice = summary["action_choice_diagnostics"]
    dominant = summary["dominant_positive_component_counts"]
    shape_zero = {
        row["outcome"]: row
        for row in summary["component_scale_summary"]
        if row["scaled_component"] == "shape_prior" and row["scale"] == 0.0
    }
    lines = [
        "# Patch RL reward attribution",
        "",
        f"- Patches: {summary['n_patches']}",
        f"- Applied REPLACE actions: {summary['n_replace_actions']}",
        f"- Outcomes: {json.dumps(summary['outcome_counts'], sort_keys=True)}",
        f"- Total reward: {summary['total_reward']:.8f}",
        (
            "- Maximum reward reconstruction residual: "
            f"{summary['max_absolute_reward_residual']:.3e}"
        ),
        (
            "- All patches hit configured max steps: "
            f"{summary['all_patches_hit_configured_max_steps']}"
        ),
        "- STOP while positive actions remain: unavailable (masked).",
        "",
        "## Total component deltas",
        "",
    ]
    for key, value in summary["component_delta_totals"].items():
        lines.append(f"- {key}: {value:+.8f}")
    lines.extend(
        [
            "",
            "## Main findings",
            "",
            (
                "- Dominant positive component counts: "
                f"{json.dumps(dominant, sort_keys=True)}"
            ),
            (
                "- Selected action was rank 1 for "
                f"{100.0 * float(choice['fraction_rank_1']):.2f}% of steps; "
                f"mean rank={float(choice['mean_rank']):.2f}, "
                f"mean normalized rank={float(choice['mean_rank_fraction']):.3f}."
            ),
            (
                "- Mean selected/best immediate reward ratio: "
                f"{float(choice['mean_chosen_reward_fraction_of_best']):.3f}."
            ),
            (
                "- With shape weight set to zero on the same logged states, "
                f"{int(shape_zero['damaged']['n_would_remain_exposed'])}/"
                f"{int(shape_zero['damaged']['n_actions'])} damaged and "
                f"{int(shape_zero['no_gt']['n_would_remain_exposed'])}/"
                f"{int(shape_zero['no_gt']['n_actions'])} no-GT actions remain exposed."
            ),
            "- Expression compatibility contribution is zero because w5=0.",
            "",
            "## Files",
            "",
            "- replace_reward_attribution.csv: one row per applied REPLACE.",
            "- component_summary_by_outcome.csv: attribution by GT outcome.",
            "- drop_one_counterfactual.csv: offline component ablations.",
            "- component_scale_summary.csv: offline weight-scale sweep.",
            "- entropy_calibration.csv: EM accuracy by normalized entropy.",
            "- entropy_threshold_summary.csv: threshold trade-off.",
            "",
        ]
    )
    return "\n".join(lines)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


if __name__ == "__main__":
    main()
