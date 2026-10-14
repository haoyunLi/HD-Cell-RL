from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_pseudo_geometry_bias import iou, parse_barcode_grid
from scripts.audit_pseudo_profile_leakage import safe_spearman, standardized_ranks
from scripts.summarize_real_background_stress import (
    audit_unique_owner,
    method_from_evaluation_name,
)


class PseudoStressAuditTests(unittest.TestCase):
    def test_method_name_resolution_prefers_specific_own_methods(self) -> None:
        self.assertEqual(
            method_from_evaluation_name(
                "real_background_20pct_hd_cell_rl_em_plus_frozen_rl_20260827"
            ),
            "em_plus_frozen_rl",
        )
        self.assertEqual(
            method_from_evaluation_name("real_background_20pct_bin2cell_20260827"),
            "bin2cell",
        )

    def test_unique_owner_audit_detects_multi_owner_physical_barcode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "assignments.csv"
            pd.DataFrame(
                {
                    "barcode": ["b1", "b1", "b2"],
                    "cell_id": [1, 2, 3],
                }
            ).to_csv(path, index=False)
            result = audit_unique_owner(path)
        self.assertEqual(result["n_barcodes_with_multiple_cell_owners"], 1)
        self.assertEqual(result["max_cell_owners_per_barcode"], 2)

    def test_standardized_ranks_reproduce_spearman_matrix(self) -> None:
        values = np.asarray([[0.0, 2.0, 1.0], [3.0, 1.0, 2.0]])
        ranks = standardized_ranks(values)
        matrix = ranks @ ranks.T
        self.assertTrue(np.allclose(np.diag(matrix), 1.0))
        self.assertAlmostEqual(
            float(matrix[0, 1]),
            safe_spearman(values[0], values[1]),
        )

    def test_barcode_grid_and_iou(self) -> None:
        coords = parse_barcode_grid(
            pd.Series(["s_002um_00003_00007-1", "s_002um_00004_00009-1"])
        )
        np.testing.assert_array_equal(coords, np.asarray([[7, 3], [9, 4]]))
        self.assertAlmostEqual(iou({"a", "b"}, {"b", "c"}), 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
