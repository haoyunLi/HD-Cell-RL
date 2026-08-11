#!/usr/bin/env python
"""Fit an unsupervised morphology shape-prior model from clustered reference cells."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hd_cell_rl.shape_prior import (
    cluster_reference_shapes,
    fit_shape_cluster_distributions,
    save_shape_prior_model,
    select_best_cluster_number,
)

LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def build_shape_prior_model_from_table(
    *,
    reference_table_path: str | Path,
    output_model_path: str | Path,
    output_summary_path: str | Path,
    normalization_summary_path: str | Path | None = None,
    cluster_col: str = "shape_cluster",
    n_clusters: int = 6,
    epsilon: float = 1.0e-4,
    feature_cols: list[str] | None = None,
    recluster: bool = False,
    random_seed: int = 7,
    select_k: bool = False,
) -> dict[str, Path]:
    """Build and save a Gaussian morphology prior from a reference feature table."""
    table_path = Path(reference_table_path)
    if not table_path.exists():
        raise FileNotFoundError(f"shape reference table not found: {table_path}")
    df = pd.read_csv(table_path, compression="infer")
    if df.empty:
        raise ValueError("shape reference table is empty")

    k_report: list[dict[str, Any]] | None = None
    if select_k:
        k_df = select_best_cluster_number(df, feature_cols=feature_cols, random_state=int(random_seed))
        k_report = k_df.to_dict(orient="records")
        n_clusters = int(k_df.iloc[0]["n_clusters"])
        LOGGER.info("Selected n_clusters=%d by silhouette score", n_clusters)

    if recluster or cluster_col not in df.columns:
        LOGGER.info("Running KMeans morphology clustering with k=%d", int(n_clusters))
        df = cluster_reference_shapes(
            df,
            n_clusters=int(n_clusters),
            feature_cols=feature_cols,
            random_state=int(random_seed),
            cluster_col=str(cluster_col),
        )
    else:
        LOGGER.info("Using existing morphology cluster column: %s", cluster_col)

    model = fit_shape_cluster_distributions(
        df,
        cluster_col=str(cluster_col),
        feature_cols=feature_cols,
        epsilon=float(epsilon),
    )
    scaler = _load_normalization_scaler(normalization_summary_path, model.feature_names)
    if scaler is not None:
        model = replace(model, scaler_mean=scaler[0], scaler_std=scaler[1])
    model_path = save_shape_prior_model(model, output_model_path)
    summary_path = Path(output_summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "reference_table_path": str(table_path.expanduser().resolve()),
        "model_path": str(model_path.expanduser().resolve()),
        "normalization_summary_path": None
        if normalization_summary_path is None
        else str(Path(normalization_summary_path).expanduser().resolve()),
        "cluster_col": str(cluster_col),
        "feature_names": list(model.feature_names),
        "n_reference_cells": int(model.n_cells.sum()),
        "n_clusters": int(model.n_clusters),
        "epsilon": float(model.epsilon),
        "cluster_counts": dict(zip(model.cluster_labels, model.n_cells.astype(int).tolist(), strict=False)),
        "cluster_priors": dict(zip(model.cluster_labels, model.priors.astype(float).tolist(), strict=False)),
        "zscored_input": bool(model.zscored_input),
        "selected_k_report": k_report,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    LOGGER.info("Wrote shape prior model: %s", model_path)
    LOGGER.info("Wrote shape prior summary: %s", summary_path)
    return {"model": model_path, "summary": summary_path}


def _load_normalization_scaler(
    summary_path: str | Path | None,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray] | None:
    if summary_path in (None, ""):
        return None
    path = Path(summary_path)
    if not path.exists():
        raise FileNotFoundError(f"shape normalization summary not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    norm = payload.get("normalization", {})
    means: list[float] = []
    stds: list[float] = []
    for feature_name in feature_names:
        item = norm.get(str(feature_name))
        if not isinstance(item, dict):
            raise ValueError(f"normalization summary missing feature {feature_name!r}")
        means.append(float(item["mean"]))
        stds.append(float(item.get("std_used", item.get("std", 1.0))))
    return np.asarray(means, dtype=np.float64), np.asarray(stds, dtype=np.float64)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit an unsupervised shape-prior Gaussian mixture model")
    parser.add_argument("--reference-table", required=True, help="Per-cell shape feature table, preferably clustered_reference.csv.gz")
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--normalization-summary", default=None, help="Shape reference summary JSON with raw feature mean/std")
    parser.add_argument("--cluster-col", default="shape_cluster")
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--epsilon", type=float, default=1.0e-4)
    parser.add_argument("--feature-cols", nargs="*", default=None)
    parser.add_argument("--recluster", action="store_true", help="Ignore existing shape_cluster labels and rerun KMeans")
    parser.add_argument("--select-k", action="store_true", help="Select k by silhouette score before clustering")
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    configure_logging(verbose=bool(args.verbose))
    build_shape_prior_model_from_table(
        reference_table_path=args.reference_table,
        output_model_path=args.output_model,
        output_summary_path=args.output_summary,
        normalization_summary_path=args.normalization_summary,
        cluster_col=str(args.cluster_col),
        n_clusters=int(args.n_clusters),
        epsilon=float(args.epsilon),
        feature_cols=None if args.feature_cols is None else [str(x) for x in args.feature_cols],
        recluster=bool(args.recluster),
        random_seed=int(args.random_seed),
        select_k=bool(args.select_k),
    )


if __name__ == "__main__":
    main()
