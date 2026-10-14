#!/usr/bin/env python
"""Export a non-overwriting patch-training evolution bundle from steps.jsonl."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


_OUTPUT_NAMES = (
    "training_evolution.csv",
    "training_evolution.svg",
    "training_evolution.pdf",
    "training_evolution.png",
    "report.md",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an evolution bundle already present in this run.",
    )
    return parser.parse_args()


def _load_updates(steps_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with steps_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") != "update_complete":
                continue
            payload = dict(record["payload"])
            rows.append(
                {
                    "update_index": int(payload["update_index"]),
                    "average_raw_patch_reward": float(payload["average_raw_patch_reward"]),
                    "average_batch_reward": float(payload["average_batch_reward"]),
                    "moving_average_reward": payload.get("moving_average_reward"),
                    "average_patch_steps": float(payload["average_patch_steps"]),
                    "entropy": float(payload["entropy"]),
                    "approx_kl": float(payload["approx_kl"]),
                    "n_episodes": int(payload["n_episodes"]),
                    "n_transitions": int(payload["n_transitions"]),
                    "checkpoint_saved": bool(payload["checkpoint_saved"]),
                    "no_improve_count": int(payload["no_improve_count"]),
                }
            )
    if not rows:
        raise RuntimeError(f"no update_complete events in {steps_path}")
    rows.sort(key=lambda row: row["update_index"])
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    update = np.asarray([row["update_index"] for row in rows], dtype=np.int64)
    raw = np.asarray([row["average_raw_patch_reward"] for row in rows], dtype=np.float64)
    batch = np.asarray([row["average_batch_reward"] for row in rows], dtype=np.float64)
    moving = np.asarray(
        [
            np.nan if row["moving_average_reward"] is None else row["moving_average_reward"]
            for row in rows
        ],
        dtype=np.float64,
    )
    steps = np.asarray([row["average_patch_steps"] for row in rows], dtype=np.float64)
    entropy = np.asarray([row["entropy"] for row in rows], dtype=np.float64)
    abs_kl = np.maximum(
        np.abs(np.asarray([row["approx_kl"] for row in rows], dtype=np.float64)),
        1.0e-8,
    )
    checkpoint_updates = [
        row["update_index"] for row in rows if row["checkpoint_saved"]
    ]

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)

    axes[0].plot(update, raw, color="#2563EB", marker="o", markersize=2.5, linewidth=1.2)
    axes[0].axhline(0.0, color="#6B7280", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("PPO update")
    axes[0].set_ylabel("Mean raw patch reward")
    axes[0].set_title("a  Exact objective", loc="left", fontweight="bold")

    axes[1].plot(update, batch, color="#7C3AED", linewidth=1.2, label="Batch reward")
    valid = np.isfinite(moving)
    if np.any(valid):
        axes[1].plot(
            update[valid],
            moving[valid],
            color="#111827",
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label="Moving average",
        )
    for checkpoint_update in checkpoint_updates:
        axes[1].axvline(
            checkpoint_update,
            color="#D97706",
            linewidth=0.9,
            linestyle=":",
            label="Best checkpoint" if checkpoint_update == checkpoint_updates[0] else None,
        )
    axes[1].set_xlabel("PPO update")
    axes[1].set_ylabel("GRPO training reward")
    axes[1].set_title("b  Training score", loc="left", fontweight="bold")
    axes[1].ticklabel_format(style="plain", axis="y", useOffset=False)
    axes[1].legend(fontsize=6)

    axes[2].plot(update, entropy, color="#0F766E", linewidth=1.2, label="Policy entropy")
    axes[2].set_xlabel("PPO update")
    axes[2].set_ylabel("Policy entropy", color="#0F766E")
    axes[2].tick_params(axis="y", colors="#0F766E")
    kl_axis = axes[2].twinx()
    kl_axis.spines["top"].set_visible(False)
    kl_axis.plot(update, abs_kl, color="#C2410C", linewidth=1.0, label="|Approx. KL|")
    kl_axis.set_yscale("log")
    kl_axis.set_ylabel("|Approx. KL|", color="#C2410C")
    kl_axis.tick_params(axis="y", colors="#C2410C")
    axes[2].set_title("c  Policy stability", loc="left", fontweight="bold")

    stem = output_dir / "training_evolution"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_report(*, rows: list[dict[str, Any]], run_dir: Path, path: Path) -> None:
    checkpoint_updates = [
        row["update_index"] for row in rows if row["checkpoint_saved"]
    ]
    best_text = ", ".join(str(value) for value in checkpoint_updates) or "none"
    path.write_text(
        "\n".join(
            [
                "# Patch PPO training evolution",
                "",
                f"- Run: `{run_dir}`",
                f"- Updates: `{len(rows)}`",
                f"- Best-checkpoint update(s): `{best_text}`",
                f"- Raw patch reward range: `{min(row['average_raw_patch_reward'] for row in rows):.6g}` to `{max(row['average_raw_patch_reward'] for row in rows):.6g}`",
                f"- Mean patch steps range: `{min(row['average_patch_steps'] for row in rows):.4g}` to `{max(row['average_patch_steps'] for row in rows):.4g}`",
                "",
                "The raw patch reward is the mean exact environment objective delta. The GRPO batch reward is a different, group-relative training score and is shown on a separate panel and scale.",
                "",
                "This is one seed (`7`) on four fixed training patches, repeated as four-member groups (16 trajectories per update). Lines show update trajectories, not confidence intervals.",
                "",
                "Source data: `training_evolution.csv`; original events: `../logs/steps.jsonl`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = (
        run_dir / "evolution"
        if args.output_dir is None
        else Path(args.output_dir).expanduser().resolve()
    )
    existing = [output_dir / name for name in _OUTPUT_NAMES if (output_dir / name).exists()]
    if existing and not args.force:
        raise FileExistsError(
            "refusing to overwrite existing evolution outputs: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_updates(run_dir / "logs" / "steps.jsonl")
    _write_csv(rows, output_dir / "training_evolution.csv")
    _plot(rows, output_dir)
    _write_report(rows=rows, run_dir=run_dir, path=output_dir / "report.md")
    print(f"Training evolution: {output_dir}")


if __name__ == "__main__":
    main()
