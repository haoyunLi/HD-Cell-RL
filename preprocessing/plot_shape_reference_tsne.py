#!/usr/bin/env python
"""Plot t-SNE views of GT shape-reference vectors."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def load_shape_reference_npz(path: str | Path, *, feature_layer: str) -> pd.DataFrame:
    """Load per-cell shape vectors from the shape-reference NPZ."""
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"shape reference NPZ not found: {npz_path}")
    key = "shape_features_zscore" if feature_layer == "zscore" else "shape_features"
    with np.load(npz_path, allow_pickle=False) as data:
        if key not in data:
            raise ValueError(f"NPZ is missing {key!r}: {npz_path}")
        if "cell_ids" not in data or "feature_names" not in data:
            raise ValueError("NPZ must contain cell_ids and feature_names")
        features = np.asarray(data[key], dtype=np.float64)
        cell_ids = data["cell_ids"].astype(str)
        feature_names = data["feature_names"].astype(str).tolist()
        if "cell_types" in data:
            cell_types = data["cell_types"].astype(str)
        else:
            cell_types = np.asarray([""] * features.shape[0], dtype="U")

    if features.ndim != 2:
        raise ValueError(f"{key} must have shape (N, F)")
    if features.shape[0] != len(cell_ids):
        raise ValueError("feature rows and cell_ids length mismatch")
    if features.shape[1] != len(feature_names):
        raise ValueError("feature columns and feature_names length mismatch")

    df = pd.DataFrame(features, columns=[f"feature_{name}" for name in feature_names])
    df.insert(0, "cell_type", pd.Series(cell_types).replace("", "unknown").fillna("unknown").astype(str))
    df.insert(0, "cell_id", cell_ids.astype(str))
    finite = np.isfinite(features).all(axis=1)
    if not bool(np.all(finite)):
        LOGGER.warning("Dropping %d cells with non-finite shape vectors", int((~finite).sum()))
        df = df.loc[finite].reset_index(drop=True)
    return df


def run_shape_reference_tsne(
    *,
    shape_reference_npz: str | Path | None,
    output_dir: str | Path,
    prefix: str,
    coordinates_csv: str | Path | None = None,
    feature_layer: str = "zscore",
    n_clusters: str = "6",
    min_cluster_size: int = 1000,
    perplexity: float = 30.0,
    max_cells: int = 50000,
    random_seed: int = 7,
    n_iter: int = 1000,
    point_size: float = 4.0,
    alpha: float = 0.75,
    stratify_by_cell_type: bool = True,
) -> dict[str, Path]:
    """Compute one t-SNE embedding and write cluster/cell-type plots."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if coordinates_csv is not None:
        sampled = _load_existing_coordinates(coordinates_csv)
        feature_cols = [col for col in sampled.columns if col.startswith("feature_")]
        if not feature_cols:
            raise ValueError("existing coordinates CSV has no feature_* columns")
        x = sampled.loc[:, feature_cols].to_numpy(dtype=np.float64)
        k = int(sampled["cluster"].nunique())
        effective_perplexity = float("nan")
        n_input_cells = int(len(sampled))
        LOGGER.info("Reusing existing t-SNE coordinates from %s", coordinates_csv)
    else:
        if shape_reference_npz is None:
            raise ValueError("--shape-reference-npz is required when --coordinates-csv is not provided")
        df = load_shape_reference_npz(shape_reference_npz, feature_layer=feature_layer)
        if len(df) < 3:
            raise ValueError("need at least 3 cells for t-SNE")
        n_input_cells = int(len(df))
        sampled = _sample_cells(
            df,
            max_cells=max_cells,
            random_seed=random_seed,
            stratify_by_cell_type=stratify_by_cell_type,
        )
        feature_cols = [col for col in sampled.columns if col.startswith("feature_")]
        x = sampled.loc[:, feature_cols].to_numpy(dtype=np.float64)
        k = _resolve_n_clusters(n_clusters, sampled["cell_type"], n_cells=len(sampled))

        LOGGER.info("Running KMeans with k=%d on %d cells", k, len(sampled))
        kmeans = KMeans(n_clusters=k, random_state=int(random_seed), n_init=10)
        cluster_ids = kmeans.fit_predict(x)
        sampled["cluster"] = [f"cluster_{idx:02d}" for idx in cluster_ids]

        effective_perplexity = _effective_perplexity(perplexity, n_cells=len(sampled))
        LOGGER.info("Running t-SNE on %d cells with perplexity=%.3f", len(sampled), effective_perplexity)
        tsne = _build_tsne(perplexity=effective_perplexity, random_seed=random_seed, n_iter=n_iter)
        emb = tsne.fit_transform(x)
        sampled["tsne_1"] = emb[:, 0]
        sampled["tsne_2"] = emb[:, 1]

    sampled["cluster"], cluster_merge_summary = _merge_small_clusters(
        labels=sampled["cluster"],
        features=x,
        min_cluster_size=int(min_cluster_size),
    )
    k = int(sampled["cluster"].nunique())
    sampled["shape_cluster"] = sampled["cluster"]

    coord_path = out_dir / f"{prefix}.tsne_coordinates.csv.gz"
    cluster_plot_path = out_dir / f"{prefix}.tsne_by_cluster.png"
    cell_type_plot_path = out_dir / f"{prefix}.tsne_by_cell_type.png"
    cell_type_heatmap_path = out_dir / f"{prefix}.shape_feature_heatmap_by_cell_type.png"
    cluster_heatmap_path = out_dir / f"{prefix}.shape_feature_heatmap_by_cluster.png"
    cell_type_means_path = out_dir / f"{prefix}.shape_feature_means_by_cell_type.csv"
    cluster_means_path = out_dir / f"{prefix}.shape_feature_means_by_cluster.csv"
    contingency_path = out_dir / f"{prefix}.cluster_cell_type_counts.csv"
    clustered_reference_path = out_dir / f"{prefix}.clustered_reference.csv.gz"
    summary_path = out_dir / f"{prefix}.summary.json"

    sampled.to_csv(coord_path, index=False, compression="gzip")
    clustered_reference = sampled.copy()
    clustered_reference.to_csv(clustered_reference_path, index=False, compression="gzip")
    contingency = pd.crosstab(sampled["cluster"], sampled["cell_type"])
    contingency.to_csv(contingency_path)

    cluster_order = _category_order_by_name(sampled["cluster"])
    cell_type_order = _category_order_by_count(sampled["cell_type"])
    _group_feature_means(sampled, group_col="cell_type", feature_cols=feature_cols, group_order=cell_type_order).to_csv(
        cell_type_means_path
    )
    _group_feature_means(sampled, group_col="cluster", feature_cols=feature_cols, group_order=cluster_order).to_csv(
        cluster_means_path
    )
    _plot_categorical_tsne(
        sampled,
        color_col="cluster",
        title="GT shape t-SNE colored by KMeans cluster",
        output_path=cluster_plot_path,
        point_size=point_size,
        alpha=alpha,
        category_order=cluster_order,
    )
    _plot_categorical_tsne(
        sampled,
        color_col="cell_type",
        title="GT shape t-SNE colored by cell type",
        output_path=cell_type_plot_path,
        point_size=point_size,
        alpha=alpha,
        category_order=cell_type_order,
    )
    _plot_feature_heatmap(
        sampled,
        group_col="cell_type",
        feature_cols=feature_cols,
        group_order=cell_type_order,
        title="Mean GT shape vector by cell type",
        output_path=cell_type_heatmap_path,
    )
    _plot_feature_heatmap(
        sampled,
        group_col="cluster",
        feature_cols=feature_cols,
        group_order=cluster_order,
        title="Mean GT shape vector by KMeans cluster",
        output_path=cluster_heatmap_path,
    )

    summary = {
        "shape_reference_npz": None if shape_reference_npz is None else str(Path(shape_reference_npz).expanduser().resolve()),
        "coordinates_csv": None if coordinates_csv is None else str(Path(coordinates_csv).expanduser().resolve()),
        "reused_coordinates": bool(coordinates_csv is not None),
        "feature_layer": feature_layer,
        "feature_columns": feature_cols,
        "n_input_cells": int(n_input_cells),
        "n_embedded_cells": int(len(sampled)),
        "max_cells": int(max_cells),
        "stratify_by_cell_type": bool(stratify_by_cell_type),
        "n_clusters": int(k),
        "min_cluster_size": int(min_cluster_size),
        "cluster_merge_summary": cluster_merge_summary,
        "perplexity_requested": float(perplexity),
        "perplexity_used": float(effective_perplexity),
        "random_seed": int(random_seed),
        "n_iter": int(n_iter),
        "cell_type_counts": sampled["cell_type"].value_counts().sort_index().astype(int).to_dict(),
        "cluster_counts": sampled["cluster"].value_counts().sort_index().astype(int).to_dict(),
        "clustered_reference_csv": str(clustered_reference_path.resolve()),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    LOGGER.info("Wrote t-SNE coordinates: %s", coord_path)
    LOGGER.info("Wrote cluster plot: %s", cluster_plot_path)
    LOGGER.info("Wrote cell-type plot: %s", cell_type_plot_path)
    LOGGER.info("Wrote cell-type feature heatmap: %s", cell_type_heatmap_path)
    LOGGER.info("Wrote cluster feature heatmap: %s", cluster_heatmap_path)
    LOGGER.info("Wrote cell-type feature means: %s", cell_type_means_path)
    LOGGER.info("Wrote cluster feature means: %s", cluster_means_path)
    LOGGER.info("Wrote clustered reference: %s", clustered_reference_path)
    return {
        "coordinates": coord_path,
        "cluster_plot": cluster_plot_path,
        "cell_type_plot": cell_type_plot_path,
        "cell_type_heatmap": cell_type_heatmap_path,
        "cluster_heatmap": cluster_heatmap_path,
        "cell_type_means": cell_type_means_path,
        "cluster_means": cluster_means_path,
        "contingency": contingency_path,
        "clustered_reference": clustered_reference_path,
        "summary": summary_path,
    }


def _sample_cells(
    df: pd.DataFrame,
    *,
    max_cells: int,
    random_seed: int,
    stratify_by_cell_type: bool,
) -> pd.DataFrame:
    if max_cells <= 0 or len(df) <= max_cells:
        return df.reset_index(drop=True).copy()

    rng = np.random.default_rng(int(random_seed))
    if not stratify_by_cell_type:
        keep = np.sort(rng.choice(len(df), size=int(max_cells), replace=False))
        return df.iloc[keep].reset_index(drop=True).copy()

    selected: list[int] = []
    groups = list(df.groupby("cell_type", sort=False).indices.items())
    for _, idx in groups:
        idx_arr = np.asarray(idx, dtype=np.int64)
        quota = max(1, int(round(max_cells * (len(idx_arr) / len(df)))))
        take = min(quota, len(idx_arr))
        selected.extend(rng.choice(idx_arr, size=take, replace=False).tolist())

    selected_arr = np.asarray(sorted(set(selected)), dtype=np.int64)
    if selected_arr.size > max_cells:
        selected_arr = np.sort(rng.choice(selected_arr, size=int(max_cells), replace=False))
    elif selected_arr.size < max_cells:
        missing = int(max_cells - selected_arr.size)
        selected_set = set(selected_arr.tolist())
        remaining = np.asarray([i for i in range(len(df)) if i not in selected_set], dtype=np.int64)
        if remaining.size > 0:
            extra = rng.choice(remaining, size=min(missing, remaining.size), replace=False)
            selected_arr = np.asarray(sorted(np.concatenate([selected_arr, extra]).tolist()), dtype=np.int64)
    return df.iloc[selected_arr].reset_index(drop=True).copy()


def _load_existing_coordinates(path: str | Path) -> pd.DataFrame:
    coord_path = Path(path)
    if not coord_path.exists():
        raise FileNotFoundError(f"coordinates CSV not found: {coord_path}")
    df = pd.read_csv(coord_path, compression="infer")
    required = {"cell_id", "cell_type", "cluster", "tsne_1", "tsne_2"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"coordinates CSV is missing required columns: {missing}")
    for col in ("cell_type", "cluster"):
        df[col] = df[col].fillna("unknown").astype(str)
    feature_cols = [col for col in df.columns if col.startswith("feature_")]
    if not feature_cols:
        raise ValueError("coordinates CSV must contain at least one feature_* column")
    finite_cols = feature_cols + ["tsne_1", "tsne_2"]
    finite = np.isfinite(df.loc[:, finite_cols].to_numpy(dtype=np.float64)).all(axis=1)
    if not bool(np.all(finite)):
        LOGGER.warning("Dropping %d rows with non-finite t-SNE/features from existing coordinates", int((~finite).sum()))
        df = df.loc[finite].reset_index(drop=True)
    return df


def _resolve_n_clusters(n_clusters: str, cell_types: pd.Series, *, n_cells: int) -> int:
    value = str(n_clusters).strip().lower()
    if value == "auto":
        labels = sorted({str(x) for x in cell_types.dropna().astype(str) if str(x) and str(x) != "unknown"})
        k = len(labels) if labels else int(round(np.sqrt(max(n_cells, 1))))
    else:
        k = int(value)
    return int(max(2, min(k, n_cells)))


def _merge_small_clusters(
    *,
    labels: pd.Series,
    features: np.ndarray,
    min_cluster_size: int,
) -> tuple[pd.Series, dict[str, Any]]:
    """Merge tiny KMeans clusters into the nearest sufficiently large cluster."""
    labels_arr = labels.fillna("unknown").astype(str).to_numpy(copy=True)
    counts = pd.Series(labels_arr).value_counts().sort_index()
    summary: dict[str, Any] = {
        "n_clusters_before": int(len(counts)),
        "n_clusters_after": int(len(counts)),
        "min_cluster_size": int(min_cluster_size),
        "merged_clusters": {},
    }
    if min_cluster_size <= 0 or len(counts) <= 1:
        relabeled = _relabel_clusters(labels_arr)
        summary["n_clusters_after"] = int(pd.Series(relabeled).nunique())
        return pd.Series(relabeled, index=labels.index), summary

    small_labels = counts[counts < int(min_cluster_size)].index.astype(str).tolist()
    large_labels = counts[counts >= int(min_cluster_size)].index.astype(str).tolist()
    if not small_labels or not large_labels:
        if small_labels and not large_labels:
            summary["merge_skipped_reason"] = "no_cluster_meets_min_cluster_size"
        relabeled = _relabel_clusters(labels_arr)
        summary["n_clusters_after"] = int(pd.Series(relabeled).nunique())
        return pd.Series(relabeled, index=labels.index), summary

    feature_arr = np.asarray(features, dtype=np.float64)
    centroids = {
        label: feature_arr[labels_arr == label].mean(axis=0)
        for label in list(large_labels) + list(small_labels)
    }
    for small_label in small_labels:
        small_center = centroids[small_label]
        nearest_large = min(
            large_labels,
            key=lambda large_label: float(np.sum((small_center - centroids[large_label]) ** 2)),
        )
        labels_arr[labels_arr == small_label] = nearest_large
        summary["merged_clusters"][small_label] = {
            "n_cells": int(counts[small_label]),
            "merged_into": str(nearest_large),
        }

    relabeled = _relabel_clusters(labels_arr)
    summary["n_clusters_after"] = int(pd.Series(relabeled).nunique())
    return pd.Series(relabeled, index=labels.index), summary


def _relabel_clusters(labels: np.ndarray) -> list[str]:
    mapping = {label: f"cluster_{idx:02d}" for idx, label in enumerate(sorted(set(labels.astype(str))))}
    return [mapping[str(label)] for label in labels]


def _effective_perplexity(perplexity: float, *, n_cells: int) -> float:
    if n_cells < 3:
        raise ValueError("need at least 3 cells for t-SNE")
    max_perplexity = max(1.0, (float(n_cells) - 1.0) / 3.0)
    return float(max(1.0, min(float(perplexity), max_perplexity)))


def _build_tsne(*, perplexity: float, random_seed: int, n_iter: int) -> TSNE:
    kwargs: dict[str, Any] = {
        "n_components": 2,
        "perplexity": float(perplexity),
        "init": "pca",
        "learning_rate": 200.0,
        "random_state": int(random_seed),
        "method": "barnes_hut",
    }
    params = inspect.signature(TSNE).parameters
    if "max_iter" in params:
        kwargs["max_iter"] = int(n_iter)
    else:
        kwargs["n_iter"] = int(n_iter)
    return TSNE(**kwargs)


def _plot_categorical_tsne(
    df: pd.DataFrame,
    *,
    color_col: str,
    title: str,
    output_path: Path,
    point_size: float,
    alpha: float,
    category_order: list[str] | None = None,
) -> None:
    labels = df[color_col].fillna("unknown").astype(str)
    categories = list(category_order) if category_order is not None else sorted(labels.unique().tolist())
    categories = [cat for cat in categories if cat in set(labels.unique())]
    cmap = plt.get_cmap("tab20", max(len(categories), 1))
    color_lookup = {cat: cmap(i % cmap.N) for i, cat in enumerate(categories)}

    legend_cols = 1 if len(categories) <= 30 else 2 if len(categories) <= 60 else 3
    fig_width = 9.5 + max(0, legend_cols - 1) * 2.2
    fig, ax = plt.subplots(figsize=(fig_width, 8.0))
    for cat in categories:
        mask = labels == cat
        ax.scatter(
            df.loc[mask, "tsne_1"],
            df.loc[mask, "tsne_2"],
            s=float(point_size),
            alpha=float(alpha),
            linewidths=0,
            color=color_lookup[cat],
            label=cat,
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])
    if len(categories) <= 80:
        font_size = 8 if len(categories) <= 30 else 6.5
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=font_size,
            markerscale=3,
            frameon=False,
            ncol=legend_cols,
            columnspacing=0.9,
            handletextpad=0.3,
        )
    else:
        ax.text(0.01, 0.01, f"{len(categories)} categories; legend omitted", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_heatmap(
    df: pd.DataFrame,
    *,
    group_col: str,
    feature_cols: list[str],
    group_order: list[str],
    title: str,
    output_path: Path,
) -> None:
    means = _group_feature_means(df, group_col=group_col, feature_cols=feature_cols, group_order=group_order)

    values = means.to_numpy(dtype=np.float64)
    plot_values = _column_scale_heatmap_values(values)
    vmax = float(np.nanmax(np.abs(plot_values))) if plot_values.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    n_rows = int(means.shape[0])
    fig_height = min(max(5.0, 0.28 * n_rows + 2.0), 18.0)
    fig_width = max(7.5, 1.2 * len(feature_cols) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(plot_values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(feature_cols)))
    ax.set_xticklabels([col.replace("feature_", "") for col in feature_cols], rotation=35, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(means.index.astype(str), fontsize=7 if n_rows > 30 else 8)
    ax.set_xlabel("shape feature")
    ax.set_ylabel(group_col)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("column-scaled mean feature value")

    labels = df[group_col].fillna("unknown").astype(str)
    counts = labels.value_counts()
    for row_idx, group_name in enumerate(means.index.astype(str)):
        ax.text(
            len(feature_cols) - 0.03,
            row_idx,
            f" n={int(counts.get(group_name, 0))}",
            va="center",
            ha="left",
            fontsize=6.5,
            color="black",
            clip_on=False,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _group_feature_means(
    df: pd.DataFrame,
    *,
    group_col: str,
    feature_cols: list[str],
    group_order: list[str],
) -> pd.DataFrame:
    labels = df[group_col].fillna("unknown").astype(str)
    work = df.loc[:, feature_cols].copy()
    work[group_col] = labels
    means = work.groupby(group_col, sort=False).mean(numeric_only=True)
    order = [label for label in group_order if label in means.index]
    if not order:
        order = means.index.astype(str).tolist()
    return means.loc[order, feature_cols]


def _column_scale_heatmap_values(values: np.ndarray, *, clip: float = 2.5) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    center = np.nanmean(arr, axis=0, keepdims=True)
    scale = np.nanstd(arr, axis=0, keepdims=True)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    scaled = (arr - center) / scale
    return np.clip(scaled, -float(clip), float(clip))


def _category_order_by_count(labels: pd.Series) -> list[str]:
    values = labels.fillna("unknown").astype(str)
    return values.value_counts().sort_values(ascending=False).index.astype(str).tolist()


def _category_order_by_name(labels: pd.Series) -> list[str]:
    values = labels.fillna("unknown").astype(str)
    return sorted(values.unique().tolist())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot one t-SNE embedding of GT shape reference vectors with cluster and cell-type colors")
    parser.add_argument("--shape-reference-npz", default=None)
    parser.add_argument("--coordinates-csv", default=None, help="Existing *.tsne_coordinates.csv.gz to redraw plots without rerunning KMeans/t-SNE")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="gt_shape_reference")
    parser.add_argument("--feature-layer", choices=("zscore", "raw"), default="zscore")
    parser.add_argument("--n-clusters", default="6", help="Integer cluster count, or auto to match available cell-type count")
    parser.add_argument("--min-cluster-size", type=int, default=1000, help="Merge clusters smaller than this into the nearest larger cluster; <=0 disables merging")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max-cells", type=int, default=50000, help="Subsample limit; <=0 embeds all cells")
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--no-stratify-by-cell-type", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    configure_logging(verbose=bool(args.verbose))
    run_shape_reference_tsne(
        shape_reference_npz=args.shape_reference_npz,
        output_dir=args.output_dir,
        prefix=str(args.prefix),
        coordinates_csv=args.coordinates_csv,
        feature_layer=str(args.feature_layer),
        n_clusters=str(args.n_clusters),
        min_cluster_size=int(args.min_cluster_size),
        perplexity=float(args.perplexity),
        max_cells=int(args.max_cells),
        random_seed=int(args.random_seed),
        n_iter=int(args.n_iter),
        point_size=float(args.point_size),
        alpha=float(args.alpha),
        stratify_by_cell_type=not bool(args.no_stratify_by_cell_type),
    )


if __name__ == "__main__":
    main()
