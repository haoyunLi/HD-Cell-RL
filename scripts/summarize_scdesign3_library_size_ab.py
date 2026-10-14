#!/usr/bin/env python3
"""Validate and summarize matched scDesign3 A/B outputs, without changing input data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from scipy.sparse import csc_matrix
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance


def distribution(values):
    x = np.asarray(values, dtype=float)
    if x.size < 2 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Expected at least two finite, nonnegative observations")
    mean = float(x.mean())
    var = float(x.var(ddof=1))
    return dict(n=len(x), mean=mean, variance=var, sd=float(np.sqrt(var)),
                cv=float(np.sqrt(var) / mean) if mean else np.nan,
                **{f"p{q:02d}": float(np.percentile(x, q)) for q in (5, 25, 50, 75, 95, 99)})


def compare(reference, simulated):
    r, s = np.asarray(reference, float), np.asarray(simulated, float)
    rs, ss = distribution(r), distribution(s)
    result = {f"{key}_ratio": ss[key] / rs[key] if rs[key] else np.nan
              for key in ("mean", "variance", "cv", "p05", "p50", "p95", "p99")}
    result.update(ks=float(ks_2samp(r, s).statistic),
                  log1p_wasserstein=float(wasserstein_distance(np.log1p(r), np.log1p(s))))
    return result


def corr(a, b):
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(spearmanr(a, b).statistic)


def validate_cells(frame, n, genes):
    if len(frame) != n or frame.cell_id.duplicated().any():
        raise ValueError("Wrong row count or duplicate cell IDs")
    values = frame[["counts", "detected_genes"]].to_numpy(float)
    if (not np.isfinite(values).all() or (values < 0).any()
            or (values != np.floor(values)).any()
            or (frame.detected_genes > genes).any()
            or (frame.detected_genes > frame.counts).any()):
        raise ValueError("Invalid count / detected-gene summaries")


def md_table(frame):
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        return str(value).replace("|", "/")
    return "\n".join(["| " + " | ".join(frame.columns) + " |",
                       "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
                      ["| " + " | ".join(map(fmt, row)) + " |"
                       for row in frame.itertuples(index=False, name=None)])


def validate_h5(path, cells, modeled_genes):
    with h5py.File(path, "r") as handle:
        m = handle["matrix"]
        shape = tuple(m["shape"][:].reshape(-1))
        values = m["data"][:].reshape(-1)
        indices = m["indices"][:].reshape(-1)
        pointers = m["indptr"][:].reshape(-1)
        barcodes = m["barcodes"].asstr()[:].reshape(-1)
        types = m["cell_types"].asstr()[:].reshape(-1)
        genes = m["features/name"].asstr()[:].reshape(-1)
    matrix = csc_matrix((values, indices, pointers), shape=shape)
    matrix.check_format(full_check=True)
    if not matrix.has_canonical_format or not (values > 0).all():
        raise ValueError("H5 sparse matrix has duplicate/unsorted genes or stored nonpositive counts")
    if (not np.array_equal(barcodes, cells.cell_id)
            or not np.array_equal(types, cells.cell_type)
            or not np.array_equal(np.asarray(matrix.sum(axis=0)).ravel(), cells.counts)
            or not np.array_equal(np.diff(pointers), cells.detected_genes)
            or not set(genes[np.unique(indices)]).issubset(set(modeled_genes))):
        raise ValueError("H5 matrix differs from cell summaries or modeled-gene axis")
    return dict(path=path.name, shape=list(map(int, shape)), nnz=int(matrix.nnz),
                counts_and_detected_genes_match=True, unique_gene_indices=True)


def summarize(root: Path, reference_metadata: Path | None = None):
    protocol = json.loads((root / "protocol.json").read_text())
    completion = json.loads((root / "completion.json").read_text())
    if not completion.get("completed") or not completion.get("offset_prediction_check"):
        raise ValueError("Experiment is incomplete or offset prediction was not checked")
    seeds = protocol["simulation_seeds"]
    seeds = seeds if isinstance(seeds, list) else [seeds]
    methods = ["A_current", "B_library_offset"]
    h5_checks = []
    modeled_genes = pd.read_csv(root / "subdiagnostic" / "modeled_genes.csv").gene
    ref = pd.read_csv(root / "reference_per_cell.csv")
    validate_cells(ref, protocol["n_reference_cells"], protocol["n_modeled_genes"])
    types = sorted(ref.cell_type.unique())
    frames = [("reference", 0, ref)]
    for seed in seeds:
        for method in methods:
            frame = pd.read_csv(root / f"{method}_{seed}_per_cell.csv")
            validate_cells(frame, protocol["n_synthetic_cells_per_method_seed"], protocol["n_modeled_genes"])
            if not (frame.method.eq(method).all() and frame.seed.eq(seed).all()):
                raise ValueError("Method/seed provenance mismatch")
            if protocol["phase"] != "smoke" and not frame.cell_type.value_counts().sort_index().equals(
                    ref.cell_type.value_counts().sort_index()):
                raise ValueError("Cell-type composition differs from the reference")
            if method == "B_library_offset":
                draws = pd.read_csv(root / "subdiagnostic" / f"library_draws_{seed}.csv")
                expected = draws.merge(ref[["cell_id", "cell_type", "counts"]],
                                       left_on="source_cell_id", right_on="cell_id", suffixes=("", "_source"),
                                       validate="many_to_one")
                if len(expected) != len(draws) or not (expected.cell_type == expected.cell_type_source).all():
                    raise ValueError("Library draws crossed types or contain unknown reference cells")
                if not np.array_equal(expected.library, expected.counts):
                    raise ValueError("Drawn library size differs from the source cell")
                if not (np.array_equal(frame.cell_id, draws.cell_id)
                        and np.array_equal(frame.requested_library, draws.library)):
                    raise ValueError("Requested library sizes are misaligned")
            if seed == seeds[0]:
                h5_checks.append(validate_h5(root / f"{method}_{seed}.h5", frame, modeled_genes))
            frames.append((method, seed, frame))
    (root / "subdiagnostic" / "h5_validation.json").write_text(json.dumps(h5_checks, indent=2) + "\n")

    rows, distances, coupling = [], [], []
    for method, seed, frame in frames:
        for group in ["ALL"] + types:
            subset = frame if group == "ALL" else frame[frame.cell_type == group]
            r = ref if group == "ALL" else ref[ref.cell_type == group]
            for metric in ("counts", "detected_genes"):
                rows.append(dict(method=method, seed=seed, cell_type=group, metric=metric,
                                 **distribution(subset[metric])))
                if method != "reference":
                    distances.append(dict(method=method, seed=seed, cell_type=group, metric=metric,
                                          **compare(r[metric], subset[metric])))
            coupling.append(dict(method=method, seed=seed, cell_type=group,
                                 counts_genes_spearman=corr(subset.counts, subset.detected_genes),
                                 requested_observed_spearman=(corr(subset.requested_library, subset.counts)
                                                            if method == "B_library_offset" else np.nan)))
    stats, distances = pd.DataFrame(rows), pd.DataFrame(distances)
    stats.to_csv(root / "distribution_summary.csv", index=False)
    distances.to_csv(root / "distribution_comparison.csv", index=False)
    pd.DataFrame(coupling).to_csv(root / "subdiagnostic" / "counts_genes_coupling.csv", index=False)

    # Gene-level guards: a larger total-count CV alone is not a successful simulation.
    gm_ref = pd.read_csv(root / "subdiagnostic" / "reference_gene_moments.csv.gz")
    gene_rows, dominant_rows = [], []
    for seed in seeds:
        for method in methods:
            gm = pd.read_csv(root / "subdiagnostic" / f"{method}_{seed}_gene_moments.csv.gz")
            merged = gm_ref.merge(gm, on=["gene", "cell_type"], suffixes=("_ref", "_sim"), validate="one_to_one")
            if len(merged) != len(gm_ref) or len(gm) != len(gm_ref):
                raise ValueError("Gene/type axes differ")
            for cell_type, z in merged.groupby("cell_type", sort=True):
                row = dict(method=method, seed=seed, cell_type=cell_type, n_genes=len(z))
                for moment in ("mean", "variance"):
                    a, b = z[f"{moment}_ref"], z[f"{moment}_sim"]
                    row[f"{moment}_spearman"] = corr(a, b)
                    row[f"{moment}_log1p_mae"] = float(np.abs(np.log1p(a) - np.log1p(b)).mean())
                row["zero_fraction_mae"] = float(np.abs(z.zero_fraction_ref - z.zero_fraction_sim).mean())
                gene_rows.append(row)
                dominant_rows.append(z.nlargest(20, "mean_ref"))
    gene_metrics = pd.DataFrame(gene_rows)
    gene_metrics.to_csv(root / "gene_fidelity.csv", index=False)
    pd.concat(dominant_rows, ignore_index=True).to_csv(root / "subdiagnostic" / "dominant_gene_diagnostics.csv", index=False)
    macro = distances[distances.cell_type != "ALL"].copy()
    macro["abs_log_cv_ratio"] = np.abs(np.log(macro.cv_ratio))
    macro["abs_log_mean_ratio"] = np.abs(np.log(macro.mean_ratio))
    macro = macro.groupby(["method", "seed", "metric"])[
        ["abs_log_cv_ratio", "abs_log_mean_ratio", "ks", "log1p_wasserstein"]].mean().reset_index()
    macro.to_csv(root / "macro_comparison.csv", index=False)
    stability = macro.groupby(["method", "metric"])["abs_log_cv_ratio"].agg(["mean", "min", "max"]).reset_index()
    stability.to_csv(root / "subdiagnostic" / "seed_stability.csv", index=False)

    # Summaries compare seeds, not independent donors/biological replicates.
    overall = stats[stats.cell_type == "ALL"].groupby(["method", "metric"])[
        ["mean", "variance", "cv", "p05", "p50", "p95"]].mean().reset_index()
    macro_mean = macro.groupby(["method", "metric"])[
        ["abs_log_cv_ratio", "abs_log_mean_ratio", "ks", "log1p_wasserstein"]].mean().reset_index()
    cv_by_type = stats[stats.cell_type != "ALL"].groupby(["cell_type", "metric", "method"]).cv.mean().unstack().reset_index()
    gene_mean = gene_metrics.groupby("method")[
        ["mean_spearman", "variance_spearman", "mean_log1p_mae", "variance_log1p_mae", "zero_fraction_mae"]].mean().reset_index()
    coupling_frame = pd.DataFrame(coupling)
    coupling_mean = coupling_frame[coupling_frame.cell_type != "ALL"].groupby("method")[
        ["counts_genes_spearman", "requested_observed_spearman"]].mean().reset_index()
    improvement = macro.groupby(["method", "metric"]).abs_log_cv_ratio.mean().unstack(0)
    conclusions = []
    for metric in ("counts", "detected_genes"):
        a, b = improvement.loc[metric, ["A_current", "B_library_offset"]]
        conclusions.append(f"- {metric}: cell-type-macro absolute log CV-ratio error {a:.4f} → {b:.4f} "
                           f"({'lower is better; improved' if b < a else 'not improved'}).")
        type_cvs = cv_by_type[cv_by_type.metric == metric]
        better = (np.abs(np.log(type_cvs.B_library_offset / type_cvs.reference)) <
                  np.abs(np.log(type_cvs.A_current / type_cvs.reference)))
        conclusions.append(f"- {metric}: CV fidelity improves in {int(better.sum())}/{len(type_cvs)} types, "
                           "using each method's mean CV across seeds.")
    conclusions.append("- Trade-off: B improves distribution width, but not every statistic. Inspect mean-count errors, "
                       "gene-level fidelity, and the Cycling B outlier before adopting it.")
    full = pd.read_csv(root / "subdiagnostic" / "reference_full_panel.csv")
    full_comparison = full[["cell_id", "counts", "detected_genes"]].merge(
        ref[["cell_id", "counts", "detected_genes"]], on="cell_id", suffixes=("_full", "_modeled"), validate="one_to_one")
    lost_counts = (full_comparison.counts_full - full_comparison.counts_modeled).mean()
    lost_genes = (full_comparison.detected_genes_full - full_comparison.detected_genes_modeled).mean()
    annotation_note = ""
    if reference_metadata is not None:
        source_meta = pd.read_csv(reference_metadata, sep="\t", usecols=[
            "cell", "sct_cell_type", "minor_cell_type", "source_id"])
        matched = ref.merge(source_meta, left_on="cell_id", right_on="cell", validate="one_to_one")
        if len(matched) != len(ref) or not matched.cell_type.eq(matched.sct_cell_type).all():
            raise ValueError("Frozen reference IDs/type labels do not match the original metadata")
        crosswalk = matched.groupby(["cell_type", "minor_cell_type"], dropna=False).size().rename("n").reset_index()
        crosswalk.to_csv(root / "subdiagnostic" / "reference_annotation_crosswalk.csv", index=False)
        matched.groupby(["cell_type", "source_id"]).size().rename("n").reset_index().to_csv(
            root / "subdiagnostic" / "reference_donor_composition.csv", index=False)
        annotation_note = f"""
## Post-hoc reference annotation diagnostic

Frozen generator labels exactly match the original metadata's sct_cell_type column. A different column, minor_cell_type, splits some of these groups and also gives discordant lineage labels. This is an annotation discrepancy to audit, not proof that either column is ground truth.

The Cycling B group illustrates the problem:

{md_table(crosswalk[crosswalk.cell_type == 'Cycling B'])}

The same group's counts–genes Spearman correlation by method (mean across seeds) is:

{md_table(coupling_frame[coupling_frame.cell_type == 'Cycling B'].groupby('method')[['counts_genes_spearman']].mean().reset_index())}

This is an exploratory explanation for residual failure, not a pre-specified model-selection endpoint. Dominant-gene diagnostics are in subdiagnostic/dominant_gene_diagnostics.csv. Labels were not changed and no extra model was fitted using this observation.
"""
    report = f"""# scDesign3 library-size A/B experiment

## Scope and outcome

Matched development experiment, not held-out validation. EM/RL, capture simulation, and existing datasets were not changed.
{chr(10).join(conclusions)}

## Controlled protocol

- Reference: {protocol['n_reference_cells']:,} frozen training cells, {protocol['n_types']} sct_cell_type labels, {protocol['n_modeled_genes']:,} modeled genes.
- A: `{protocol['mu_A']}` using the frozen fitted model; B: `{protocol['mu_B']}` with refitted marginal distributions and residual copula.
- Both: NB, sigma_formula=1, corr_formula=1, unchanged copula feature selection and sparsification, matched type composition and simulation seeds: {seeds}.
- Library sizes are resampled within training cell type. They condition expected counts, not an exact total-count constraint.
- B offset expansion was checked against direct model predictions at 0.5× and 2× reference type-median depth.
- {protocol['n_synthetic_cells_per_method_seed']:,} cells per method per seed. Seeds measure Monte Carlo variation, not independent biological replication.
- Primary comparisons use the same modeled-gene axis. Filtering removes a mean {lost_counts:.2f} counts and {lost_genes:.2f} detected genes per reference cell; unmodeled genes remain zero in generated full-axis H5.

## Overall distributions

Synthetic entries are arithmetic means over simulation seeds. Variance is sample variance (ddof=1); CV is SD / mean.
These reference values are for the selected training population, not the earlier all-38,233-cell summary.

{md_table(overall)}

## Within-type fidelity (each type equally weighted)

Lower is better for all four error measures. Absolute log ratio is zero at an exact match; it treats over/under-dispersion symmetrically. KS and Wasserstein are effect sizes, not significance tests.

{md_table(macro_mean)}

CV-error stability across seeds (not a biological confidence interval):

{md_table(stability)}

### CV by cell type

{md_table(cv_by_type)}

## Gene-level guardrails

Correlations: higher is better. Errors: lower is better. Values average equally across cell types and seeds; detailed per-type results are in gene_fidelity.csv.

{md_table(gene_mean)}

Within-type counts–genes coupling (equal type/seed weights; requested depth applies only to B):

{md_table(coupling_mean)}

{annotation_note}

## Interpretation and limits

- Matching total counts alone does not establish realistic joint expression. Read counts, detected genes, counts–genes coupling, gene means/variances, and zero fractions together.
- This is in-sample distributional validation. B reuses the empirical training depth distribution by design; no independent donor or HD section was evaluated.
- NB dispersion and the residual copula can still add total-count variation beyond the sampled depth. B is not guaranteed to match the observed variance exactly.
- sigma_formula=1 means a per-gene dispersion constant across cells, not variance equal to one. This A/B test does not separately test type-dependent dispersion.
- No fixed overall CV target was imposed. Tissue/type composition and gene panels differ between colorectal scRNA, skin Xenium, and Visium HD.
- This stage simulates scRNA-like cell expression before spatial allocation/capture. Passing it does not establish realistic Visium HD 2/8/16 µm distributions.
- Do not promote B to the production generator based only on this experiment. First examine type-level failures and validate on independent donors, then evaluate the capture stage separately.

## Reproducibility

- Protocol and hashes: [protocol.json](protocol.json).
- Cell summaries and quantiles: [distribution_summary.csv](distribution_summary.csv).
- Per-seed distribution comparisons: [distribution_comparison.csv](distribution_comparison.csv).
- Equal-type-weighted results: [macro_comparison.csv](macro_comparison.csv).
- Gene-level results: [gene_fidelity.csv](gene_fidelity.csv).
- Detailed provenance, library draws, reference IDs, and gene moments: subdiagnostic/.
- Independent H5 round-trip checks: [subdiagnostic/h5_validation.json](subdiagnostic/h5_validation.json).
- scDesign3 library-size specification follows its [official vignette](https://songdongyuan1994.github.io/scDesign3/docs/articles/scDesign3-librarySize-vignette.html).

Run with a **new, nonexistent** output directory; the runner refuses overwriting:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 Rscript scripts/run_scdesign3_library_size_ab.R \\
  {protocol['source_generator']} \\
  {protocol['reference_model_dir']} \\
  {protocol['raw_h5']} \\
  runs/benchmarks/NEW_UNIQUE_AB_RUN 24 0
Bin2Cell_Validation/bin/python scripts/summarize_scdesign3_library_size_ab.py runs/benchmarks/NEW_UNIQUE_AB_RUN{(' --reference-metadata ' + str(reference_metadata.resolve())) if reference_metadata is not None else ''}
```
"""
    (root / "report.md").write_text(report)
    input_files = [root / "protocol.json", root / "completion.json", root / "reference_per_cell.csv"]
    input_files += [root / f"{m}_{s}_per_cell.csv" for m in methods for s in seeds]
    input_files += sorted((root / "subdiagnostic").glob("*_gene_moments.csv.gz"))
    input_files += sorted((root / "subdiagnostic").glob("library_draws_*.csv"))
    receipt = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files}
    if reference_metadata is not None:
        receipt[str(reference_metadata.resolve())] = hashlib.sha256(reference_metadata.read_bytes()).hexdigest()
    (root / "subdiagnostic" / "summary_input_hashes.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--reference-metadata", type=Path)
    args = parser.parse_args()
    summarize(args.run_dir.resolve(), args.reference_metadata)
