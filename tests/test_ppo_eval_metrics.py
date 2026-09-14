from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from preprocessing.ppo_eval_metrics import (
    compute_fractional_spatial_overlap_metrics,
    load_gt_bin_weights_for_cells,
)


class PPOEvalMetricsTests(unittest.TestCase):
    def test_fractional_overlap_uses_soft_gt_mass(self) -> None:
        metrics = compute_fractional_spatial_overlap_metrics(
            {"b1", "b2"},
            {"b1": 1.0, "b2": 0.25, "b3": 0.75},
        )
        self.assertAlmostEqual(metrics["intersection_mass"], 1.25)
        self.assertAlmostEqual(metrics["union_mass"], 2.75)
        self.assertAlmostEqual(metrics["iou"], 1.25 / 2.75)
        self.assertAlmostEqual(metrics["precision"], 1.25 / 2.0)
        self.assertAlmostEqual(metrics["recall"], 1.25 / 2.0)
        self.assertAlmostEqual(metrics["dice"], 2.5 / 4.0)

    def test_fractional_loader_prefers_coverage_and_filters_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gt.csv"
            pd.DataFrame(
                [
                    {
                        "cell_id": 1,
                        "barcode": "b1",
                        "cell_coverage_fraction": 0.25,
                        "weight": 0.9,
                    },
                    {
                        "cell_id": 1,
                        "barcode": "b2",
                        "cell_coverage_fraction": 1.0,
                        "weight": 1.0,
                    },
                    {
                        "cell_id": 2,
                        "barcode": "b3",
                        "cell_coverage_fraction": 0.5,
                        "weight": 0.5,
                    },
                ]
            ).to_csv(path, index=False)

            result = load_gt_bin_weights_for_cells(
                csv_path=path,
                matched_cell_ids={"1"},
            )

        self.assertEqual(set(result), {"1"})
        self.assertEqual(result["1"], {"b1": 0.25, "b2": 1.0})

    def test_fractional_loader_rejects_duplicate_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gt.csv"
            pd.DataFrame(
                [
                    {"cell_id": "A", "barcode": "b1", "weight": 0.5},
                    {"cell_id": "A", "barcode": "b1", "weight": 0.5},
                ]
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_gt_bin_weights_for_cells(
                    csv_path=path,
                    matched_cell_ids={"A"},
                )


if __name__ == "__main__":
    unittest.main()
