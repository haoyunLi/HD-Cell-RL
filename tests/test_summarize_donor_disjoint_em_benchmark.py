from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.summarize_donor_disjoint_em_benchmark import (
    _bootstrap_mean_ci,
    _discover_runs,
)


class DonorDisjointEMBenchmarkSummaryTests(unittest.TestCase):
    def test_discovers_each_canonical_method(self) -> None:
        methods = (
            "hd_cell_rl_nucleus_only",
            "hd_cell_rl_distance_only",
            "hd_cell_rl_frozen_em",
            "bin2cell",
            "stcs",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, method in enumerate(methods):
                run = root / f"run_{index}"
                run.mkdir()
                (run / "summary.json").write_text(
                    json.dumps({"method": method}), encoding="utf-8"
                )
            discovered = _discover_runs(root)

        self.assertEqual(
            set(discovered),
            {"nucleus_only", "distance_only", "frozen_em", "bin2cell", "stcs"},
        )

    def test_bootstrap_interval_is_deterministic_and_contains_constant(self) -> None:
        low, high = _bootstrap_mean_ci(
            np.full(8, 0.25),
            replicates=100,
            rng=np.random.default_rng(7),
        )
        self.assertAlmostEqual(low, 0.25)
        self.assertAlmostEqual(high, 0.25)


if __name__ == "__main__":
    unittest.main()
