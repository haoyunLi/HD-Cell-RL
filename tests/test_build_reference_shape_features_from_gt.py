from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from preprocessing.build_reference_shape_features_from_gt import (
    build_shape_reference_features,
    write_shape_reference_outputs,
)


class BuildReferenceShapeFeaturesFromGtTests(unittest.TestCase):
    def test_builds_grid_shape_features_and_preserves_cell_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            bins_path = root / "gt_cell_bins.csv.gz"
            assignments_path = root / "ground_truth_cell_assignments.csv"

            bins = pd.DataFrame(
                {
                    "cell_id": [1, 1, 1, 1, 2, 2, 2],
                    "barcode": ["a", "b", "c", "d", "e", "f", "g"],
                    "array_col": [0, 1, 0, 1, 0, 1, 2],
                    "array_row": [0, 0, 1, 1, 5, 5, 5],
                }
            )
            bins.to_csv(bins_path, index=False, compression="gzip")
            pd.DataFrame(
                {
                    "cell_id": [1, 2],
                    "sc_cell_barcode": ["sc1", "sc2"],
                    "cell_type": ["A", "B"],
                }
            ).to_csv(assignments_path, index=False)

            per_cell, cell_type_summary, summary = build_shape_reference_features(
                bins_path,
                gt_cell_assignments_csv=assignments_path,
                bin_size_um=2.0,
            )

            self.assertEqual(per_cell["cell_id"].tolist(), ["1", "2"])
            self.assertEqual(per_cell["cell_type"].tolist(), ["A", "B"])
            self.assertEqual(per_cell["area"].tolist(), [4, 3])
            self.assertEqual(per_cell["perimeter"].tolist(), [8, 8])
            self.assertEqual(per_cell["perimeter_um"].tolist(), [16.0, 16.0])
            self.assertAlmostEqual(float(per_cell.loc[0, "log_area"]), float(np.log(5.0)))
            self.assertAlmostEqual(float(per_cell.loc[0, "compactness"]), float(np.pi / 4.0))
            self.assertAlmostEqual(float(per_cell.loc[0, "solidity"]), 1.0)
            self.assertAlmostEqual(float(per_cell.loc[0, "anisotropy"]), 1.0)
            self.assertGreater(float(per_cell.loc[1, "anisotropy"]), 1.0)
            self.assertTrue(np.isfinite(per_cell[["log_area_z", "compactness_z", "solidity_z", "anisotropy_z"]].to_numpy()).all())
            self.assertEqual(cell_type_summary["cell_type"].tolist(), ["A", "B"])
            self.assertEqual(summary["n_cells"], 2)

            outputs = write_shape_reference_outputs(
                per_cell_df=per_cell,
                cell_type_summary_df=cell_type_summary,
                summary=summary,
                per_cell_output_path=root / "shape.per_cell.csv.gz",
                cell_type_summary_output_path=root / "shape.cell_type_summary.csv",
                npz_output_path=root / "shape.npz",
                summary_output_path=root / "shape.summary.json",
            )
            for path in outputs.values():
                self.assertTrue(path.exists())

            with np.load(outputs["npz"]) as data:
                self.assertEqual(data["shape_features"].shape, (2, 4))
                self.assertEqual(data["shape_features_zscore"].shape, (2, 4))
                self.assertEqual(data["feature_names"].astype(str).tolist(), ["log_area", "compactness", "solidity", "anisotropy"])
                self.assertEqual(data["cell_types"].astype(str).tolist(), ["A", "B"])
                self.assertEqual(data["cell_type_labels"].astype(str).tolist(), ["A", "B"])
                self.assertEqual(data["cell_type_feature_means"].shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
