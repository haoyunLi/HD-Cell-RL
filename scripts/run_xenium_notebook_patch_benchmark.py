#!/usr/bin/env python3
"""Run unchanged assignment methods on the notebook-derived Xenium patch inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "xenium_notebook_crc"


def _run(command: list[str], log: Path, *, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                check=False, text=True)
    if result.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-30:])
        raise RuntimeError(f"Method failed ({result.returncode}): {' '.join(command)}\n{tail}")


def run(data: Path, output: Path, methods: set[str], *, dataset_name: str = DATASET_NAME,
        model_cache: Path | None = None) -> None:
    if not (data / "dataset_manifest.json").exists():
        raise FileNotFoundError(data / "dataset_manifest.json")
    existing = {path.name for path in output.iterdir()} if output.exists() else set()
    if existing - {"data_audit.json"}:
        raise FileExistsError(f"Refusing to overwrite benchmark run: {output}")
    if "data_audit.json" in existing:
        audit = json.loads((output / "data_audit.json").read_text())
        if not audit.get("pass_all") or not all(check.get("pass_all") for check in audit.get("checks", [])):
            raise ValueError("Dataset audit did not pass for every patch")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data / "dataset_manifest.json").read_text())
    source = Path(manifest["source"])
    reference = data / "reference/reference_counts.npz"
    if not reference.exists():
        raise FileNotFoundError(reference)
    bin2cell_python = ROOT / "bin2cell_env/bin/python"
    stcs_python = ROOT / "stcs_env/bin/python"
    em_python = ROOT / "Bin2Cell_Validation/bin/python"
    if not all(path.exists() for path in (bin2cell_python, stcs_python, em_python)):
        raise FileNotFoundError("One of the existing benchmark environments is missing")
    comparison = []
    for entry in manifest["patches"]:
        patch = data / entry["patch_id"]
        patch_output = output / entry["patch_id"]
        seeds = patch / "nuclei/nuclear_bins.csv.gz"
        scale = json.loads((patch / "spatial/scalefactors_json.json").read_text())
        mpp = float(scale["microns_per_pixel"])
        if not 0 < mpp or abs(float(scale["spot_diameter_fullres"]) * mpp - 2.0) > 0.01:
            raise ValueError(f"Patch has inconsistent 2 um geometry: {patch}")
        if "nucleus_only" in methods:
            folder = patch_output / "nucleus_only"
            folder.mkdir(parents=True)
            seed_frame = pd.read_csv(seeds, dtype={"barcode": str, "cell_id": str})
            seed_frame[["barcode", "cell_id", "is_nuclear", "array_row", "array_col"]].to_csv(
                folder / "assignments.csv", index=False)
        if "distance_only" in methods or "em" in methods:
            _run([str(em_python), "scripts/run_lung_pilot_em.py", "--patch-dir", str(patch),
                  "--nuclei-dir", str(patch / "nuclei"), "--reference-npz", str(reference),
                  "--output-dir", str(patch_output / "em_controls")],
                 patch_output / "em_controls/run.log")
        if "bin2cell" in methods:
            # The existing Bin2Cell environment lacks pyarrow, which the shared
            # evaluation import needs. Append only STCS's site-packages as a
            # fallback; Bin2Cell and all its installed method packages win.
            wrapper = ("import sys,runpy;sys.path.append('stcs_env/lib/python3.12/site-packages');"
                       "sys.argv[0]='preprocessing/run_bin2cell_analysis.py';"
                       "runpy.run_path(sys.argv[0],run_name='__main__')")
            model_cache = model_cache or data.parent / "model_cache"
            if not (model_cache / "models/StarDist2D/2D_versatile_fluo/2D_versatile_fluo.zip").exists():
                raise FileNotFoundError("Cached official StarDist 2D_versatile_fluo model is missing")
            environment = dict(os.environ, MPLCONFIGDIR=str(Path("/tmp/hdcellrl_mpl_bin2cell")),
                               KERAS_HOME=str(model_cache.resolve()))
            _run([str(bin2cell_python), "-c", wrapper, "--pseudo_hd_dir", str(patch),
                  "--image_path", str(patch / "he_patch_fullres.png"),
                  "--output_dir", str(patch_output / "bin2cell"), "--dataset_name", dataset_name,
                  "--mpp", str(mpp), "--external_nuclear_bins_path", str(seeds),
                  "--patch_local_grid",
                  "--max_bin_distance", "5", "--prob_thresh_gex", "0.05",
                  "--nms_thresh_gex", "0.5", "--gaussian_sigma", "5"],
                 patch_output / "bin2cell/run.log", env=environment)
        if "stcs" in methods:
            environment = dict(os.environ, MPLCONFIGDIR=str(Path("/tmp/hdcellrl_mpl_stcs")))
            _run([str(stcs_python), "preprocessing/run_stcs_analysis.py", "--pseudo_hd_dir", str(patch),
                  "--output_dir", str(patch_output / "stcs"), "--dataset_name", dataset_name,
                  "--stcs_repo_dir", "external/STCS", "--external_nuclear_bins_path", str(seeds),
                  "--search_radius", "5", "--lambda_spatial", "0.5", "--pseudobulk_mode", "mean"],
                 patch_output / "stcs/run.log", env=environment)
        paths = {
            "nucleus_only": patch_output / "nucleus_only/assignments.csv",
            "distance_only": patch_output / "em_controls/distance_only/assignments.csv",
            "em": patch_output / "em_controls/em/assignments.csv",
            "bin2cell": patch_output / f"bin2cell/{dataset_name}_bin2cell_assignments.csv",
            "stcs": patch_output / f"stcs/{dataset_name}_stcs_assignments.csv",
        }
        for method in ("nucleus_only", "bin2cell", "stcs", "distance_only", "em"):
            if method not in methods:
                continue
            evaluation = patch_output / "evaluation" / method
            _run([str(stcs_python), "scripts/evaluate_xenium_notebook_patch.py",
                  "--patch-dir", str(patch), "--assignments", str(paths[method]),
                  "--method", method, "--source", str(source), "--output-dir", str(evaluation)],
                 evaluation / "run.log")
            comparison.append(json.loads((evaluation / "summary.json").read_text()))
        pd.DataFrame(comparison).to_csv(output / "method_comparison_partial.csv", index=False)
        print(f"Completed {entry['patch_id']}", flush=True)
    table = pd.DataFrame(comparison)
    table.to_csv(output / "method_comparison.csv", index=False)
    protocol = {"dataset": str(data.resolve()), "source": str(source.resolve()),
                "reference": str(reference.resolve()), "methods": sorted(methods),
                "data_rule": "published notebook: qv>=20, whole-cell transcripts, ceil(x/2), ceil(y/2)",
                "barcode_adapter": "10x s_002um_ROW_COL-1 for unchanged STCS parser; barcode_map.parquet retains notebook ids",
                "bin2cell_patch_adapter": "patch-local grid coordinates and exterior zero padding to a single StarDist block; model, thresholds and assignment logic unchanged",
                "nuclear_source": "Source Xenium nucleus_boundaries.parquet, common to all methods",
                "mask_truth": "Source Xenium cell_boundaries.parquet, evaluation only",
                "dataset_name": dataset_name,
                "source_xenium_native_mpp": manifest.get("source_xenium_microns_per_pixel"),
                "synthetic_raster_mpp": manifest.get("synthetic_raster_microns_per_pixel"),
                "method_logic_changed": False, "rl_run": False}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=("nucleus_only", "bin2cell", "stcs", "distance_only", "em"),
                        default=["nucleus_only", "bin2cell", "stcs", "distance_only", "em"])
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--model-cache", type=Path)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, set(args.methods), dataset_name=args.dataset_name,
        model_cache=args.model_cache)


if __name__ == "__main__":
    main()
