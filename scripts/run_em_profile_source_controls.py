#!/usr/bin/env python3
"""Equal-nuclear-depth profile interventions, strictly diagnostic (uses GT profiles)."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hd_cell_rl import em_assignment as em
from hd_cell_rl.experiment_protocol import write_experiment_manifest
from scripts.diagnose_em_library_size import run_generator
from scripts import run_source_separated_method_comparison as comparison

SEEDS = (2026091601, 2026091602, 2026091603, 2026091604, 2026091605)
SAMPLED_ARMS = tuple(f'sampled_nuclear_seed_{seed}' for seed in SEEDS)
EXPECTED_ARM = 'expected_nuclear_shrunk'
ARMS = ('distance_only','frozen_em',*SAMPLED_ARMS,EXPECTED_ARM,'oracle_profile_diagnostic')


def matched_profile_counts(oracle, nuclear_totals, cell_ids, *, seed=None):
    """N*oracle or multinomial(N, oracle), with stable physical-cell RNG streams."""
    oracle = np.asarray(oracle, dtype=float)
    totals = np.asarray(nuclear_totals, dtype=float)
    if oracle.ndim != 2 or oracle.shape[0] != len(cell_ids) or totals.shape != (len(cell_ids),):
        raise ValueError('unaligned profile/count/cell axes')
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError('duplicate physical cells')
    if not np.isfinite(oracle).all() or np.any(oracle < 0) or not np.isfinite(totals).all() or np.any(totals < 0):
        raise ValueError('invalid profile or nuclear count')
    np.testing.assert_allclose(oracle.sum(1), 1., atol=1e-10)
    if not np.allclose(totals, np.rint(totals), rtol=0., atol=1e-8):
        raise ValueError('observed nuclear UMI budgets must be integers')
    if seed is None:
        counts = totals[:,None]*oracle
    else:
        counts = np.empty_like(oracle)
        for i, (cell_id,total) in enumerate(zip(cell_ids,totals,strict=True)):
            entropy = np.frombuffer(hashlib.sha256(str(cell_id).encode()).digest(),dtype='<u4').astype(np.uint32).tolist()
            rng = np.random.default_rng(np.random.SeedSequence([int(seed),*entropy]))
            counts[i] = rng.multinomial(int(round(total)),oracle[i]/oracle[i].sum())
    np.testing.assert_allclose(counts.sum(1),totals,atol=1e-8,rtol=0.)
    return counts


def shrink_counts(counts, q, theta, prior_count, epsilon):
    """Mirror the existing nuclear initializer's exact scRNA normalization."""
    normalized_theta = em._normalize_reference_theta(theta,n_types=q.shape[1],n_genes=counts.shape[1],epsilon=epsilon)
    return em._normalize_cell_gene_counts(counts+prior_count*(q@normalized_theta),epsilon=epsilon)


def make_control(*, arm, seed, records):
    def control(oracle, initialization, original_values):
        _, reliability, totals, prior = original_values
        inp = initialization['em_input']
        q = initialization['cell_type_posterior']
        np.testing.assert_allclose(q, em.initialize_q_from_nuclear_seeds(inp),atol=0,rtol=0)
        counts = matched_profile_counts(oracle,totals,inp.cell_ids,seed=seed)
        profile = shrink_counts(counts,q,initialization['reference_theta'],prior,initialization['epsilon'])
        for i,cell in enumerate(inp.cell_ids):
            records.append(dict(patch_id=inp.patch_id,cell_id=cell,arm=arm,seed=seed,
                observed_nuclear_umi=float(totals[i]),intervention_count_total=float(counts[i].sum()),
                intervention_detected_genes=int(np.count_nonzero(counts[i])),profile_reliability=float(reliability[i]),
                reference_prior_count=float(prior),profile_tv_to_oracle=float(.5*np.abs(profile[i]-oracle[i]).sum()),
                q_initial_sha256=hashlib.sha256(np.asarray(q[i],dtype=np.float64).tobytes()).hexdigest(),
                profile_sha256=hashlib.sha256(np.asarray(profile[i],dtype=np.float64).tobytes()).hexdigest()))
        return profile
    return control


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--profile-dir',type=Path,required=True)
    p.add_argument('--prior-diagnostic-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--evaluation-workers',type=int,default=4)
    p.add_argument('--max-iterations',type=int,default=50)
    a=p.parse_args()
    for key in ('source_dir','profile_dir','prior_diagnostic_dir'):
        setattr(a,key,getattr(a,key).resolve(strict=True))
    a.output_dir=a.output_dir.resolve()
    a.output_dir.mkdir(parents=True,exist_ok=False)
    (a.output_dir/'subdiagnostic').mkdir()
    write_experiment_manifest(a.output_dir,config={**vars(a),'seeds':SEEDS,'arms':ARMS},inputs={
        'source_comparison':a.source_dir/'method_comparison.csv',
        'prior_diagnosis':a.prior_diagnostic_dir/'method_comparison.csv',
        'reference':comparison.REFERENCE,
        'A_profiles':a.profile_dir/'A_current_20260916.h5',
        'B_profiles':a.profile_dir/'B_library_offset_20260916.h5',
        'A_mapping':a.source_dir/'A_current/profile_assignment.csv',
        'B_mapping':a.source_dir/'B_library_offset/profile_assignment.csv'},
        protocol={'scope':'modeled-only four existing development patches; 43 core cells; GT-profile diagnostic only',
            'unchanged':'observed bin counts, LL, Conf, q initialization, lambda, shrinkage prior strength, geometry, EM weights',
            'q_updates':'identical algorithm, trajectories may diverge as a downstream effect of changed profile',
            'sampled':'multinomial selected-gene counts; same observed nuclear UMI total for each physical cell; physical-ID keyed seed',
            'expected':'observed nuclear UMI total times generating profile; same scRNA shrinkage',
            'oracle':'unshrunk generating profile; preserved lambda',
            'convergence':'common diagnostic cap=50, original stopping criteria; no GT-based iteration selection; require prior converged baselines to reproduce',
            'no_training':True,'no_weight_selection':True})
    for generator in ('A_current','B_library_offset'):
        records=[]
        controls={arm:make_control(arm=arm,seed=seed,records=records) for arm,seed in zip(SAMPLED_ARMS,SEEDS,strict=True)}
        controls[EXPECTED_ARM]=make_control(arm=EXPECTED_ARM,seed=None,records=records)
        run_generator(a,generator,profile_controls=controls,evaluation_workers=a.evaluation_workers,max_iterations=a.max_iterations)
        frame=pd.DataFrame(records)
        if (frame.groupby(['patch_id','cell_id']).q_initial_sha256.nunique()!=1).any():
            raise AssertionError('q initialization differs across interventions')
        frame.to_csv(a.output_dir/generator/'subdiagnostic/profile_interventions.csv',index=False)
        checks=[]
        for arm in ('distance_only','frozen_em','oracle_profile_diagnostic'):
            new_path=a.output_dir/generator/'methods'/arm/'assignments.csv'
            old_path=a.prior_diagnostic_dir/generator/'methods'/arm/'assignments.csv'
            new=pd.read_csv(new_path,dtype={'cell_id':str})
            old=pd.read_csv(old_path,dtype={'cell_id':str})
            if set(zip(new.barcode,new.cell_id)) != set(zip(old.barcode,old.cell_id)):
                raise AssertionError(f'{generator} {arm} does not reproduce prior diagnostic')
            checks.append(dict(arm=arm,exact_hard_ownership_reproduction=True))
        comparison.dump(a.output_dir/generator/'subdiagnostic/prior_reproduction.json',checks)


if __name__=='__main__':
    main()
