#!/usr/bin/env python3
"""Matched reference-only profile ablation; reuse the established inference runner."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_em_posterior_predictive import main

if __name__=='__main__':main(alternative='reference_only')
