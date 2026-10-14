from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.summarize_main_method_benchmark import (
    audit_unique_owner,
    method_from_evaluation_name,
)


class MainMethodBenchmarkSummaryTests(unittest.TestCase):
    def test_method_mapping_includes_own_and_external_methods(self):
        expected = {
            "hd_cell_rl_nucleus_only_v2_43cells_run": "nucleus_only",
            "hd_cell_rl_distance_only_v2_43cells_run": "distance_only",
            "hd_cell_rl_em_only_v2_43cells_run": "em_only",
            "hd_cell_rl_em_plus_frozen_rl_v2_43cells_run": "em_plus_frozen_rl",
            "bin2cell_v2_43cells_run": "bin2cell",
            "stcs_v2_43cells_run": "stcs",
            "smurf_v2_43cells_run": "smurf",
        }
        self.assertEqual(
            {name: method_from_evaluation_name(name) for name in expected},
            expected,
        )

    def test_unique_owner_audit_detects_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assignments.csv"
            pd.DataFrame(
                {
                    "barcode": ["b1", "b2", "b2", "b3", "b3"],
                    "cell_id": [1, 1, 2, 3, 3],
                }
            ).to_csv(path, index=False)
            audit = audit_unique_owner(path)
        self.assertEqual(audit["n_unique_barcodes"], 3)
        self.assertEqual(audit["n_barcodes_with_multiple_cell_owners"], 1)
        self.assertEqual(audit["max_cell_owners_per_barcode"], 2)


if __name__ == "__main__":
    unittest.main()
