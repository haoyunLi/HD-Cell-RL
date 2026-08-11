"""Static patch-level assignment and overlap plots."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False
    plt = None  # type: ignore[assignment]
    PatchCollection = None  # type: ignore[assignment]
    Rectangle = None  # type: ignore[assignment]


_OVERLAP_COLORS = {
    "correct_owner": "#2A9D8F",
    "wrong_owner": "#E9A23B",
    "unmatched_owner": "#8B5CF6",
    "pred_only": "#D95F59",
    "gt_only": "#4C78A8",
}


def save_patch_overview_plot(*, payload: dict[str, Any], output_path: Path) -> str | None:
    """Render assignment ownership and GT overlap for one patch."""
    if not HAS_MATPLOTLIB:
        return None

    bins = list(payload.get("bins", []))
    if not bins:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bin_size_um = float(payload.get("bin_size_um", 2.0))
    owner_ids = sorted(
        {
            str(row["predicted_owner_cell_id"])
            for row in bins
            if row.get("predicted_owner_cell_id") is not None
        }
    )
    owner_colors = _owner_color_map(owner_ids)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.2), constrained_layout=True)
    _draw_assignment_panel(
        ax=axes[0],
        bins=bins,
        owner_colors=owner_colors,
        bin_size_um=bin_size_um,
    )
    _draw_overlap_panel(
        ax=axes[1],
        bins=bins,
        bin_size_um=bin_size_um,
    )

    metrics = dict(payload.get("metrics", {}))
    patch_score = _format_metric(payload.get("patch_score"))
    total_reward = _format_metric(payload.get("total_reward"))
    foreground_iou = _format_metric(metrics.get("foreground_iou"))
    owner_accuracy = _format_metric(metrics.get("owner_accuracy"))
    fig.suptitle(
        (
            f"{payload.get('patch_id', 'patch')}  "
            f"score={patch_score}  reward={total_reward}  "
            f"foreground IoU={foreground_iou}  owner accuracy={owner_accuracy}"
        ),
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output_path)


def _draw_assignment_panel(
    *,
    ax: Any,
    bins: list[dict[str, Any]],
    owner_colors: dict[str, str],
    bin_size_um: float,
) -> None:
    by_owner: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bins:
        owner = row.get("predicted_owner_cell_id")
        if owner is not None:
            by_owner[str(owner)].append(row)

    for owner, rows in by_owner.items():
        _add_bin_rectangles(
            ax=ax,
            rows=rows,
            bin_size_um=bin_size_um,
            facecolor=owner_colors[owner],
            edgecolor="none",
            alpha=0.82,
            zorder=2,
        )
        _draw_grid_outline(
            ax=ax,
            rows=rows,
            bin_size_um=bin_size_um,
            color=owner_colors[owner],
            linewidth=1.05,
            linestyle="-",
            alpha=1.0,
            zorder=4,
        )

    gt_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bins:
        gt_owner = row.get("gt_owner_cell_id")
        if gt_owner is not None:
            gt_by_cell[str(gt_owner)].append(row)
    for rows in gt_by_cell.values():
        _draw_grid_outline(
            ax=ax,
            rows=rows,
            bin_size_um=bin_size_um,
            color="#FFFFFF",
            linewidth=2.8,
            linestyle="-",
            alpha=1.0,
            zorder=5,
        )
        _draw_grid_outline(
            ax=ax,
            rows=rows,
            bin_size_um=bin_size_um,
            color="#111827",
            linewidth=1.25,
            linestyle="-",
            alpha=1.0,
            zorder=6,
        )

    ax.set_title("Predicted owner assignment")
    ax.text(
        0.01,
        0.01,
        "filled = predicted owner, black + white halo = matched GT",
        transform=ax.transAxes,
        fontsize=8,
        color="#4B5563",
        va="bottom",
    )
    _finish_patch_axis(ax=ax, bins=bins, bin_size_um=bin_size_um)


def _draw_overlap_panel(*, ax: Any, bins: list[dict[str, Any]], bin_size_um: float) -> None:
    legend_handles: list[Any] = []
    for category, color in _OVERLAP_COLORS.items():
        rows = [row for row in bins if str(row.get("overlap_category")) == category]
        if not rows:
            continue
        facecolor = "none" if category == "gt_only" else color
        _add_bin_rectangles(
            ax=ax,
            rows=rows,
            bin_size_um=bin_size_um,
            facecolor=facecolor,
            edgecolor=color,
            alpha=0.86,
            linewidth=0.75 if category == "gt_only" else 0.15,
            zorder=3,
        )
        legend_handles.append(
            Rectangle(
                (0.0, 0.0),
                1.0,
                1.0,
                facecolor=facecolor,
                edgecolor=color,
                linewidth=0.9,
                label=_category_label(category),
            )
        )

    ax.set_title("GT overlap and owner correctness")
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=False,
            fontsize=8,
        )
    _finish_patch_axis(ax=ax, bins=bins, bin_size_um=bin_size_um)


def _add_bin_rectangles(
    *,
    ax: Any,
    rows: list[dict[str, Any]],
    bin_size_um: float,
    facecolor: str,
    edgecolor: str,
    alpha: float,
    zorder: int,
    linewidth: float = 0.0,
    label: str | None = None,
) -> None:
    half = 0.5 * float(bin_size_um)
    rectangles = [
        Rectangle(
            (float(row["x_um"]) - half, float(row["y_um"]) - half),
            float(bin_size_um),
            float(bin_size_um),
        )
        for row in rows
    ]
    if not rectangles:
        return
    collection = PatchCollection(
        rectangles,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
        label=label,
    )
    ax.add_collection(collection)


def _draw_grid_outline(
    *,
    ax: Any,
    rows: list[dict[str, Any]],
    bin_size_um: float,
    color: str,
    linewidth: float,
    linestyle: Any,
    alpha: float,
    zorder: int,
) -> None:
    occupied = {
        (int(row["array_row"]), int(row["array_col"])): (float(row["x_um"]), float(row["y_um"]))
        for row in rows
    }
    half = 0.5 * float(bin_size_um)
    for (array_row, array_col), (x_um, y_um) in occupied.items():
        if (array_row, array_col - 1) not in occupied:
            ax.plot(
                [x_um - half, x_um - half],
                [y_um - half, y_um + half],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                zorder=zorder,
            )
        if (array_row, array_col + 1) not in occupied:
            ax.plot(
                [x_um + half, x_um + half],
                [y_um - half, y_um + half],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                zorder=zorder,
            )
        if (array_row - 1, array_col) not in occupied:
            ax.plot(
                [x_um - half, x_um + half],
                [y_um - half, y_um - half],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                zorder=zorder,
            )
        if (array_row + 1, array_col) not in occupied:
            ax.plot(
                [x_um - half, x_um + half],
                [y_um + half, y_um + half],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                zorder=zorder,
            )


def _finish_patch_axis(*, ax: Any, bins: list[dict[str, Any]], bin_size_um: float) -> None:
    x_values = np.asarray([float(row["x_um"]) for row in bins], dtype=np.float64)
    y_values = np.asarray([float(row["y_um"]) for row in bins], dtype=np.float64)
    pad = max(float(bin_size_um), 1.0)
    ax.set_xlim(float(np.min(x_values)) - pad, float(np.max(x_values)) + pad)
    ax.set_ylim(float(np.min(y_values)) - pad, float(np.max(y_values)) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _owner_color_map(owner_ids: list[str]) -> dict[str, str]:
    palette = [
        "#2563EB",
        "#DC2626",
        "#059669",
        "#D97706",
        "#7C3AED",
        "#0891B2",
        "#DB2777",
        "#4D7C0F",
        "#9333EA",
        "#C2410C",
        "#0F766E",
        "#4338CA",
        "#BE123C",
        "#3F6212",
        "#0369A1",
        "#A21CAF",
        "#B45309",
        "#047857",
        "#1D4ED8",
        "#B91C1C",
    ]
    return {owner: palette[idx % len(palette)] for idx, owner in enumerate(owner_ids)}


def _category_label(category: str) -> str:
    return {
        "correct_owner": "correct owner",
        "wrong_owner": "wrong owner",
        "unmatched_owner": "owner not GT-matched",
        "pred_only": "prediction only",
        "gt_only": "GT only",
    }.get(category, category)


def _format_metric(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.3f}"
