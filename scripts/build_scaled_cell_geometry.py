#!/usr/bin/env python3
"""Re-run the existing nucleus-to-cell growth with scaled cytoplasmic area.

The source growth algorithm, nuclear labels, tissue mask, random seed, and
distance limit are unchanged. Only the target number of non-nuclear pixels per
cell changes. This is a geometry stress test, not a realism-calibrated mask.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION = ROOT.parent / "Bin2Cell_Validation"


def scaled_target_areas(nuclear_area: np.ndarray, original_target: np.ndarray, scale: float) -> np.ndarray:
    """Keep every nuclear pixel and scale only the target cytoplasmic area."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("cytoplasmic area scale must be finite and positive")
    nuclear = np.asarray(nuclear_area, dtype=np.int64)
    target = np.asarray(original_target, dtype=np.int64)
    if nuclear.shape != target.shape or np.any(target < nuclear):
        raise ValueError("target and nuclear areas are inconsistent")
    return nuclear + np.rint(scale * (target - nuclear)).astype(np.int64)


def load_growth_module(validation_repo: Path):
    directory = validation_repo / "scripts" / "ground_truth"
    sys.path.insert(0, str(directory))
    source = directory / "expand_nuclei_to_cells.py"
    spec = importlib.util.spec_from_file_location("original_nucleus_growth", source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-repo", type=Path, default=DEFAULT_VALIDATION)
    args = parser.parse_args()
    validation = args.validation_repo.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    growth, source = load_growth_module(validation)
    original = growth.integer_target_cell_areas

    def scaled(nuclear_area, sampled_percent, minimum_percent, maximum_percent):
        target = original(nuclear_area, sampled_percent, minimum_percent, maximum_percent)
        return scaled_target_areas(nuclear_area, target, args.scale)

    growth.integer_target_cell_areas = scaled
    # The existing program resolves its inputs relative to Bin2Cell_Validation.
    command_line = sys.argv
    try:
        sys.argv = [str(source)]
        defaults = growth.parse_args()
    finally:
        sys.argv = command_line
    defaults.output_dir = str(output)
    defaults.overwrite = False
    old_cwd = Path.cwd()
    try:
        import os
        os.chdir(validation)
        growth.run(defaults)
    finally:
        os.chdir(old_cwd)
    metadata = output / "cell_metadata.csv.gz"
    import pandas as pd
    cells = pd.read_csv(metadata, usecols=["cell_id", "nucleus_area_px", "target_cell_area_px", "final_cell_area_px"])
    nucleus = cells.nucleus_area_px.to_numpy(np.int64)
    target = cells.target_cell_area_px.to_numpy(np.int64)
    final = cells.final_cell_area_px.to_numpy(np.int64)
    baseline_metadata = validation / "outputs/ground_truth/colorectal_nucleus_based/cells/cell_metadata.csv.gz"
    baseline = pd.read_csv(baseline_metadata, usecols=["cell_id", "nucleus_area_px", "target_cell_area_px", "final_cell_area_px"])
    if not cells.cell_id.equals(baseline.cell_id):
        raise AssertionError("physical cell IDs or ordering changed")
    if not np.array_equal(nucleus, baseline.nucleus_area_px.to_numpy(np.int64)):
        raise AssertionError("nuclear areas changed")
    expected = scaled_target_areas(nucleus, baseline.target_cell_area_px.to_numpy(np.int64), args.scale)
    if not np.array_equal(target, expected):
        raise AssertionError("scaled target areas differ from paired baseline")
    report = {
        "cytoplasmic_target_area_scale": args.scale,
        "geometry_source": str(source),
        "baseline_metadata": str(baseline_metadata),
        "n_cells": len(cells),
        "nuclear_areas_unchanged": True,
        "target_areas_exactly_scaled": True,
        "target_reached_fraction": float(np.mean(final == target)),
        "median_realized_cytoplasm_ratio_to_baseline": float(np.median(
            (final - nucleus) / np.maximum(1, baseline.final_cell_area_px.to_numpy(np.int64) - nucleus)
        )),
    }
    (output / "area_scale_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
