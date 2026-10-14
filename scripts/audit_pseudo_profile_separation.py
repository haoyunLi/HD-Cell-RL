#!/usr/bin/env python3
"""Matched-depth within/between-type profile similarity; no ownership inference."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_pseudo_realism_experiment import load_h5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--benchmark-dir', type=Path, required=True)
    parser.add_argument('--validation-repo', type=Path, required=True)
    args = parser.parse_args()
    out, bench, v = args.experiment_dir, args.benchmark_dir, args.validation_repo
    genes = pd.read_csv(out / 'genes.csv').gene.tolist()
    sy, sb, sg = load_h5(bench / 'simulation/scdesign3_generator_disjoint/scdesign3_synthetic_counts.h5')
    assignment = pd.read_csv(bench / 'assignment/ground_truth_cell_to_synthetic_scdesign3.csv')
    smap = {b: i for i, b in enumerate(sb)}
    sy = sy[[smap[b] for b in assignment.sc_cell_barcode]]
    raw, rb, rg = load_h5(v / 'data/single_cell/colorectal/CRC_GSE166555_raw_tumor.h5')
    rm = pd.read_csv(bench / 'simulation/scdesign3_generator_disjoint/model_reference_cell_metadata.tsv', sep='\t')
    rmap = {b: i for i, b in enumerate(rb)}
    raw = raw[[rmap[b] for b in rm.Cell]]
    def align(x, names):
        mapping = {g: i for i, g in enumerate(names)}
        i, j = zip(*[(mapping[g], j) for j, g in enumerate(genes) if g in mapping])
        return (x @ sparse.csr_matrix((np.ones(len(i)), (i, j)), shape=(len(names), len(genes)))).tocsr()
    sy, raw = align(sy, sg), align(raw, rg)
    st, rt = assignment.cell_type.to_numpy(), rm.cell_type.to_numpy()
    types = sorted(t for t, n in assignment.cell_type.value_counts().items() if n >= 10)
    rows = []
    # Equal source sample size per type; multinomial projection removes library-size differences.
    # This is a conditional resampling diagnostic, not recovery of latent true RNA probabilities.
    for seed in (20260907, 20260908, 20260909):
        rng = np.random.default_rng(seed)
        for source, x, labels in [('scdesign3', sy, st), ('raw_generator_scRNA', raw, rt)]:
            indices = np.concatenate([rng.choice(np.flatnonzero(labels == t), min(30, (st == t).sum()), replace=False) for t in types])
            z = np.zeros((len(indices), len(genes)), dtype=np.float32)
            for i, index in enumerate(indices):
                s, e = x.indptr[index:index+2]
                values = x.data[s:e]
                z[i, x.indices[s:e]] = rng.multinomial(1000, values / values.sum())
            z = np.sqrt(z / 1000)
            sim = z @ z.T
            tt = labels[indices]
            for typ in types:
                within = (tt[:, None] == typ) & (tt[None, :] == typ) & np.triu(np.ones(sim.shape, dtype=bool), 1)
                between = (tt[:, None] == typ) & (tt[None, :] != typ)
                rows.append(dict(source=source, seed=seed, cell_type=typ, projected_umis=1000,
                                 n_cells=int((tt == typ).sum()), within_sqrt_cosine=float(sim[within].mean()),
                                 between_sqrt_cosine=float(sim[between].mean()),
                                 type_similarity_gap=float(sim[within].mean()-sim[between].mean())))
    result = pd.DataFrame(rows)
    result.to_csv(out / 'comparison/profile_separation_matched_depth.csv', index=False)
    print(result.groupby(['source', 'cell_type'])[['within_sqrt_cosine', 'between_sqrt_cosine', 'type_similarity_gap']].mean().to_string())


if __name__ == '__main__':
    main()
