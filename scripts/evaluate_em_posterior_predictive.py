#!/usr/bin/env python3
"""Post-inference evaluation only; this phase may read generating GT profiles."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_source_separated_method_comparison as protocol
from scripts.run_em_posterior_predictive import ARMS,GENERATORS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--profile-dir',type=Path,required=True)
    args=p.parse_args()
    out=args.run_dir.resolve(strict=True);source=args.source_dir.resolve(strict=True)
    profiles=args.profile_dir.resolve(strict=True)
    if not (out/'inference_complete.json').is_file():raise RuntimeError('inference must finish first')
    arms=tuple(json.loads((out/'inference_complete.json').read_text()).get('arms',ARMS))
    if arms not in (ARMS,('distance_only','frozen_em','reference_only')):raise ValueError('unsupported matched run arms')
    for generator in GENERATORS:
        directory=out/generator
        seal=json.loads((directory/'inference_complete.json').read_text())
        for assignment in seal['assignments']:
            if hashlib.sha256(Path(assignment['path']).read_bytes()).hexdigest()!=assignment['sha256']:
                raise AssertionError('sealed inference output changed')
        context,=(source/generator/'benchmark/modeled_only/methods/nucleus_only').glob('*/evaluation_context')
        protocol.PROFILE_ASSIGNMENT=source/generator/'profile_assignment.csv'
        protocol.PROFILE_H5=profiles/f'{generator}_20260916.h5'
        tasks=[]
        for arm in arms:
            tasks.append((arm,['--assignments-csv',directory/'methods'/arm/'assignments.csv',
                '--method-name',arm,'--method-label',arm,'--external-nuclear-bins-path',
                source/generator/'benchmark/common_confident_nuclear_bins.csv.gz',
                *protocol.evaluation_args(context,directory,arm)]))
        def evaluate(task):
            arm,arguments=task
            protocol.command(directory,arm+'_evaluation','scripts/evaluate_assignment_table.py',arguments)
        with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(evaluate,tasks))


if __name__=='__main__':main()
