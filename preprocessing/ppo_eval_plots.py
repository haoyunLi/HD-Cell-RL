from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False
    plt = None  # type: ignore[assignment]


_PALETTE = {
    "ink": "#1F2933",
    "muted": "#667085",
    "grid": "#E6E8EC",
    "candidate": "#D6DAE0",
    "gt": "#3B82C4",
    "tp": "#2A9D8F",
    "fp": "#D95F59",
    "fn": "#4C78A8",
    "accent": "#E9A23B",
}


def _configure_matplotlib_style() -> None:
    if not HAS_MATPLOTLIB:
        return
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 240,
        }
    )


def _slug(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:80] if text else "unknown"


def _choose_overlay_indices(df: pd.DataFrame, n_pick: int, selection: str, seed: int) -> list[int]:
    if n_pick <= 0 or len(df) == 0:
        return []
    n_pick = min(int(n_pick), len(df))
    if selection == "first":
        return list(range(n_pick))
    if selection == "top_reward" and "total_reward" in df.columns:
        scores = pd.to_numeric(df["total_reward"], errors="coerce").fillna(-np.inf).to_numpy(dtype=np.float64)
        order = np.argsort(-scores)
        return [int(i) for i in order[:n_pick]]
    if selection == "best_iou" and "pred_iou" in df.columns:
        scores = pd.to_numeric(df["pred_iou"], errors="coerce").fillna(-np.inf).to_numpy(dtype=np.float64)
        order = np.argsort(-scores)
        return [int(i) for i in order[:n_pick]]
    if selection == "worst_iou" and "pred_iou" in df.columns:
        scores = pd.to_numeric(df["pred_iou"], errors="coerce").fillna(np.inf).to_numpy(dtype=np.float64)
        order = np.argsort(scores)
        return [int(i) for i in order[:n_pick]]
    rng = np.random.default_rng(int(seed))
    choices = rng.choice(len(df), size=n_pick, replace=False)
    return [int(i) for i in np.asarray(choices, dtype=np.int64)]


def _estimate_grid_step(coords: np.ndarray) -> float:
    vals = np.unique(np.asarray(coords, dtype=np.float64))
    if vals.size <= 1:
        return 2.0
    diffs = np.diff(np.sort(vals))
    diffs = diffs[np.isfinite(diffs) & (diffs > 1.0e-6)]
    if diffs.size == 0:
        return 2.0
    return float(np.median(diffs))


def _build_gt_contour_grid(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    pts = np.asarray(xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        return None

    step_x = _estimate_grid_step(pts[:, 0])
    step_y = _estimate_grid_step(pts[:, 1])
    x0 = float(np.min(pts[:, 0]))
    y0 = float(np.min(pts[:, 1]))
    ix = np.rint((pts[:, 0] - x0) / step_x).astype(np.int64)
    iy = np.rint((pts[:, 1] - y0) / step_y).astype(np.int64)
    if ix.size == 0 or iy.size == 0:
        return None

    min_ix = int(ix.min())
    max_ix = int(ix.max())
    min_iy = int(iy.min())
    max_iy = int(iy.max())
    width = max_ix - min_ix + 1
    height = max_iy - min_iy + 1
    if width <= 0 or height <= 0:
        return None

    grid = np.zeros((height + 2, width + 2), dtype=np.uint8)
    gx = (ix - min_ix + 1).astype(np.int64, copy=False)
    gy = (iy - min_iy + 1).astype(np.int64, copy=False)
    grid[gy, gx] = 1
    x_coords = x0 + ((np.arange(width + 2, dtype=np.float64) + min_ix - 1.0) * step_x)
    y_coords = y0 + ((np.arange(height + 2, dtype=np.float64) + min_iy - 1.0) * step_y)
    return x_coords, y_coords, grid


def _build_gt_outline_segments(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    pts = np.asarray(xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        return None

    step_x = _estimate_grid_step(pts[:, 0])
    step_y = _estimate_grid_step(pts[:, 1])
    x0 = float(np.min(pts[:, 0]))
    y0 = float(np.min(pts[:, 1]))
    ix = np.rint((pts[:, 0] - x0) / step_x).astype(np.int64)
    iy = np.rint((pts[:, 1] - y0) / step_y).astype(np.int64)
    occupied = {(int(x), int(y)) for x, y in zip(ix, iy, strict=False)}
    if not occupied:
        return None

    x_segments: list[float] = []
    y_segments: list[float] = []
    half_x = 0.5 * float(step_x)
    half_y = 0.5 * float(step_y)

    def add_segment(x1: float, y1: float, x2: float, y2: float) -> None:
        x_segments.extend([x1, x2, np.nan])
        y_segments.extend([y1, y2, np.nan])

    for gx, gy in occupied:
        cx = x0 + float(gx) * float(step_x)
        cy = y0 + float(gy) * float(step_y)
        if (gx - 1, gy) not in occupied:
            add_segment(cx - half_x, cy - half_y, cx - half_x, cy + half_y)
        if (gx + 1, gy) not in occupied:
            add_segment(cx + half_x, cy - half_y, cx + half_x, cy + half_y)
        if (gx, gy - 1) not in occupied:
            add_segment(cx - half_x, cy - half_y, cx + half_x, cy - half_y)
        if (gx, gy + 1) not in occupied:
            add_segment(cx - half_x, cy + half_y, cx + half_x, cy + half_y)

    if not x_segments:
        return None
    return np.asarray(x_segments, dtype=np.float64), np.asarray(y_segments, dtype=np.float64)


def _grid_keys_for_xy(xy: np.ndarray, *, step_x: float, step_y: float) -> list[tuple[int, int]]:
    pts = np.asarray(xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return []
    sx = max(float(step_x), 1.0e-8)
    sy = max(float(step_y), 1.0e-8)
    ix = np.rint(pts[:, 0] / sx).astype(np.int64)
    iy = np.rint(pts[:, 1] / sy).astype(np.int64)
    return [(int(x), int(y)) for x, y in zip(ix, iy, strict=False)]


def _finite_float_or_none(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) else None


def _numeric_values(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.zeros((0,), dtype=np.float64)
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)
    return values[np.isfinite(values)]


def _numeric_pair(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    if x_col not in df.columns or y_col not in df.columns:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def _annotate_distribution(ax: Any, values: np.ndarray, *, label: str) -> None:
    if values.size == 0:
        ax.text(0.5, 0.5, "no valid values", ha="center", va="center", transform=ax.transAxes, color=_PALETTE["muted"])
        return
    median = float(np.median(values))
    mean = float(np.mean(values))
    ax.axvline(median, color=_PALETTE["ink"], linewidth=1.3, linestyle="-")
    ax.axvline(mean, color=_PALETTE["accent"], linewidth=1.1, linestyle="--")
    ax.text(
        0.98,
        0.94,
        f"n={values.size}\nmedian={median:.3g}\nmean={mean:.3g}",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
        color=_PALETTE["ink"],
    )
    ax.set_xlabel(label)
    ax.set_ylabel("cells")


def _light_grid(ax: Any) -> None:
    ax.grid(True, axis="y", color=_PALETTE["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


def _has_useful_reward_variation(df: pd.DataFrame) -> bool:
    values = _numeric_values(df, "total_reward")
    if values.size == 0:
        return False
    rounded = np.unique(np.round(values, 6))
    return int(rounded.size) >= 8


def _format_overlay_title(row: dict[str, Any], method_label: str) -> str:
    if "total_reward" in row:
        reward = _finite_float_or_none(row.get("total_reward", np.nan))
        reward_text = "nan" if reward is None else f"{reward:.2f}"
        lines = [
            f"cell={row['cell_id']}  reward={reward_text}  "
            f"assigned={row['n_assigned_bins']}/{row['n_candidate_bins']}",
            f"GT match={row.get('match_method', 'none')}",
        ]
    else:
        lines = [
            f"cell={row['cell_id']}  {method_label.lower()}={row.get('matched_pred_cell_id', 'unmatched')}  "
            f"assigned={row['n_assigned_bins']}/{row['n_candidate_bins']}",
            f"GT match={row.get('match_method', 'none')}",
        ]

    iou = _finite_float_or_none(row.get("pred_iou", np.nan))
    dice = _finite_float_or_none(row.get("pred_dice", np.nan))
    precision = _finite_float_or_none(row.get("pred_precision", np.nan))
    recall = _finite_float_or_none(row.get("pred_recall", np.nan))
    metric_parts: list[str] = []
    if iou is not None:
        metric_parts.append(f"IoU={iou:.3f}")
    if dice is not None:
        metric_parts.append(f"Dice={dice:.3f}")
    if precision is not None:
        metric_parts.append(f"P={precision:.3f}")
    if recall is not None:
        metric_parts.append(f"R={recall:.3f}")
    if metric_parts:
        lines.append("  ".join(metric_parts))

    gene = _finite_float_or_none(row.get("gene_spearman_r", np.nan))
    if gene is not None:
        lines.append(f"Gene Spearman={gene:.3f}")
    return "\n".join(lines)


def _save_evaluation_overview(df: pd.DataFrame, plots_dir: Path, *, method_label: str) -> str | None:
    if "pred_iou" not in df.columns:
        return None

    iou = _numeric_values(df, "pred_iou")
    dice = _numeric_values(df, "pred_dice")
    pred_bins, gt_bins = _numeric_pair(df, "pred_n_bins", "gt_n_bins")
    precision, recall = _numeric_pair(df, "pred_precision", "pred_recall")
    precision_color = None
    if {"pred_precision", "pred_recall", "pred_n_bins", "gt_n_bins"}.issubset(df.columns):
        p = pd.to_numeric(df["pred_precision"], errors="coerce").to_numpy(dtype=np.float64)
        r = pd.to_numeric(df["pred_recall"], errors="coerce").to_numpy(dtype=np.float64)
        pred = pd.to_numeric(df["pred_n_bins"], errors="coerce").to_numpy(dtype=np.float64)
        gt = pd.to_numeric(df["gt_n_bins"], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(p) & np.isfinite(r) & np.isfinite(pred) & np.isfinite(gt)
        if np.any(valid):
            precision = p[valid]
            recall = r[valid]
            precision_color = pred[valid] / np.clip(gt[valid], 1.0, None)

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.4))
    fig.suptitle(f"{method_label} evaluation overview", x=0.02, ha="left", fontsize=12, fontweight="bold")

    ax = axes[0, 0]
    if iou.size:
        ax.hist(iou, bins=np.linspace(0.0, 1.0, 26), color=_PALETTE["gt"], alpha=0.88, edgecolor="white", linewidth=0.5)
    _annotate_distribution(ax, iou, label="IoU")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Spatial overlap")
    _light_grid(ax)

    ax = axes[0, 1]
    if dice.size:
        ax.hist(dice, bins=np.linspace(0.0, 1.0, 26), color=_PALETTE["tp"], alpha=0.88, edgecolor="white", linewidth=0.5)
    _annotate_distribution(ax, dice, label="Dice")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Boundary agreement")
    _light_grid(ax)

    ax = axes[1, 0]
    if pred_bins.size:
        ax.scatter(gt_bins, pred_bins, s=18, color=_PALETTE["ink"], alpha=0.58, linewidths=0.0)
        limit = float(max(np.max(pred_bins), np.max(gt_bins), 1.0))
        ax.plot([0.0, limit], [0.0, limit], color=_PALETTE["muted"], linewidth=1.0, linestyle="--")
        over = float(np.mean(pred_bins > gt_bins))
        ax.text(
            0.04,
            0.95,
            f"overgrowth={over:.1%}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            color=_PALETTE["ink"],
        )
        ax.set_xlim(0.0, limit * 1.05)
        ax.set_ylim(0.0, limit * 1.05)
    ax.set_title("Predicted size vs GT size")
    ax.set_xlabel("GT bins")
    ax.set_ylabel("predicted bins")
    _light_grid(ax)

    ax = axes[1, 1]
    if precision.size:
        if precision_color is None:
            ax.scatter(recall, precision, s=18, color=_PALETTE["ink"], alpha=0.58, linewidths=0.0)
        else:
            sc = ax.scatter(
                recall,
                precision,
                c=np.clip(precision_color, 0.0, 2.5),
                cmap="viridis",
                s=18,
                alpha=0.70,
                linewidths=0.0,
            )
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
            cbar.set_label("pred/GT bins", fontsize=8)
            cbar.ax.tick_params(labelsize=7)
    ax.set_title("Precision-recall tradeoff")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    _light_grid(ax)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    path = plots_dir / "evaluation_overview.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def save_summary_plots(df: pd.DataFrame, run_dir: Path, *, method_label: str) -> list[str]:
    if not HAS_MATPLOTLIB:
        return []

    _configure_matplotlib_style()
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    overview = _save_evaluation_overview(df, plots_dir, method_label=method_label)
    if overview is not None:
        saved.append(overview)

    if "total_reward" in df.columns and _has_useful_reward_variation(df):
        rewards = _numeric_values(df, "total_reward")
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.hist(rewards, bins=32, color=_PALETTE["tp"], alpha=0.9, edgecolor="white", linewidth=0.5)
        _annotate_distribution(ax, rewards, label="total reward")
        ax.set_title("Reward distribution")
        _light_grid(ax)
        fig.tight_layout()
        p = plots_dir / "reward_hist.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    assigned = _numeric_values(df, "n_assigned_bins")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(assigned, bins=32, color=_PALETTE["fp"], alpha=0.86, edgecolor="white", linewidth=0.5)
    _annotate_distribution(ax, assigned, label="assigned bins")
    ax.set_title("Predicted cell size distribution")
    _light_grid(ax)
    fig.tight_layout()
    p = plots_dir / "assigned_bins_hist.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    if "total_reward" in df.columns and _has_useful_reward_variation(df):
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.scatter(
            df["n_assigned_bins"].to_numpy(dtype=np.float64),
            df["total_reward"].to_numpy(dtype=np.float64),
            s=12,
            alpha=0.55,
            color=_PALETTE["ink"],
            linewidths=0.0,
        )
        ax.set_title("Reward vs Assigned Bins")
        ax.set_xlabel("Assigned Bins")
        ax.set_ylabel("Total Reward")
        _light_grid(ax)
        fig.tight_layout()
        p = plots_dir / "reward_vs_assigned_bins.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    return saved


def save_overlay_plots(
    *,
    records: list[Any],
    df: pd.DataFrame,
    run_dir: Path,
    max_cells: int,
    selection: str,
    seed: int,
    method_label: str,
) -> list[str]:
    if not HAS_MATPLOTLIB or max_cells <= 0 or not records:
        return []

    _configure_matplotlib_style()
    overlays_dir = run_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    indices = _choose_overlay_indices(df=df, n_pick=max_cells, selection=selection, seed=seed)
    saved: list[str] = []
    for idx in indices:
        rec = records[idx]
        row = rec.metrics
        xy = np.asarray(rec.candidate_bin_xy_um, dtype=np.float32)
        if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] == 0:
            continue
        assigned = np.asarray(rec.final_membership_mask, dtype=np.uint8) == 1
        if assigned.shape[0] != xy.shape[0]:
            continue

        gt_cell_xy = None if rec.gt_cell_xy_um is None else np.asarray(rec.gt_cell_xy_um, dtype=np.float32)
        in_gt = np.zeros((xy.shape[0],), dtype=bool)
        fn_xy = np.zeros((0, 2), dtype=np.float32)
        if gt_cell_xy is not None and gt_cell_xy.ndim == 2 and gt_cell_xy.shape[1] == 2 and gt_cell_xy.shape[0] > 0:
            combined = np.vstack((xy, gt_cell_xy)).astype(np.float64, copy=False)
            step_x = _estimate_grid_step(combined[:, 0])
            step_y = _estimate_grid_step(combined[:, 1])
            candidate_keys = _grid_keys_for_xy(xy, step_x=step_x, step_y=step_y)
            gt_keys = set(_grid_keys_for_xy(gt_cell_xy, step_x=step_x, step_y=step_y))
            assigned_keys = {candidate_keys[i] for i in np.flatnonzero(assigned).tolist()}
            in_gt = np.asarray([key in gt_keys for key in candidate_keys], dtype=bool)
            gt_key_list = _grid_keys_for_xy(gt_cell_xy, step_x=step_x, step_y=step_y)
            fn_keep = [key not in assigned_keys for key in gt_key_list]
            if any(fn_keep):
                fn_xy = gt_cell_xy[np.asarray(fn_keep, dtype=bool)]

        tp = assigned & in_gt
        fp = assigned & (~in_gt)

        fig, ax = plt.subplots(figsize=(6.7, 6.4))
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=5,
            c=_PALETTE["candidate"],
            alpha=0.20,
            linewidths=0.0,
            zorder=1,
            label="candidate",
        )
        if gt_cell_xy is not None and gt_cell_xy.ndim == 2 and gt_cell_xy.shape[1] == 2 and gt_cell_xy.shape[0] > 0:
            outline = _build_gt_outline_segments(gt_cell_xy)
            if outline is not None:
                x_coords, y_coords = outline
                ax.plot(x_coords, y_coords, color=_PALETTE["gt"], linewidth=1.2, alpha=0.88, zorder=4)
                ax.plot([], [], color=_PALETTE["gt"], linewidth=1.2, alpha=0.88, label="GT outline")
        if fn_xy.shape[0] > 0:
            ax.scatter(
                fn_xy[:, 0],
                fn_xy[:, 1],
                s=24,
                facecolors="none",
                edgecolors=_PALETTE["fn"],
                linewidths=0.9,
                marker="s",
                alpha=0.82,
                zorder=2,
                label="FN: GT only",
            )
        if np.any(tp):
            ax.scatter(
                xy[tp, 0],
                xy[tp, 1],
                s=18,
                c=_PALETTE["tp"],
                alpha=0.95,
                linewidths=0.25,
                edgecolors="white",
                zorder=5,
                label="TP overlap",
            )
        if np.any(fp):
            ax.scatter(
                xy[fp, 0],
                xy[fp, 1],
                s=18,
                c=_PALETTE["fp"],
                alpha=0.95,
                linewidths=0.25,
                edgecolors="white",
                zorder=6,
                label="FP: pred only",
            )

        center = np.asarray(rec.nucleus_center_xy_um, dtype=np.float32)
        if center.shape == (2,):
            ax.scatter(
                [float(center[0])],
                [float(center[1])],
                s=80,
                marker="x",
                c=_PALETTE["ink"],
                linewidths=1.9,
                zorder=7,
                label="nucleus",
            )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_title(_format_overlay_title(row, method_label), fontsize=9.5, pad=8, loc="left")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
        ax.grid(False)
        fig.tight_layout()

        out = overlays_dir / f"overlay_{idx:04d}_{_slug(str(row['cell_id']))}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(out))

    return saved
