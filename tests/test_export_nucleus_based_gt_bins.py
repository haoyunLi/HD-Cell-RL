from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from preprocessing.export_nucleus_based_gt_bins import export_gt_bins


class ExportNucleusBasedGtBinsTests(unittest.TestCase):
    def test_preserves_multiple_cell_owners_for_one_barcode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assignments_path = root / "assignments.parquet"
            barcodes_path = root / "barcodes.tsv.gz"
            cells_path = root / "cells.csv"
            cell_output = root / "cell_bins.csv.gz"
            nuclear_output = root / "nuclear_bins.csv.gz"
            summary_output = root / "summary.json"

            pd.DataFrame(
                {
                    "barcode_index": [0, 0, 1],
                    "array_row": [0, 0, 0],
                    "array_col": [0, 0, 1],
                    "pxl_row_in_fullres": [10.0, 10.0, 10.0],
                    "pxl_col_in_fullres": [20.0, 20.0, 22.0],
                    "cell_id": [11, 22, 22],
                    "cell_coverage_fraction": [6 / 9, 2 / 9, 1.0],
                    "nuclear_fraction": [2 / 9, 0.0, 0.0],
                    "cytoplasm_fraction": [3 / 9, 1 / 9, 1.0],
                    "membrane_fraction": [1 / 9, 1 / 9, 0.0],
                }
            ).to_parquet(assignments_path, index=False)
            pd.Series(["b0", "b1"]).to_csv(
                barcodes_path,
                sep="\t",
                header=False,
                index=False,
                compression="gzip",
            )
            pd.DataFrame(
                {"cell_id": [11, 22], "cell_type": ["A", "B"]}
            ).to_csv(cells_path, index=False)

            summary = export_gt_bins(
                bin_assignments_path=assignments_path,
                barcodes_path=barcodes_path,
                cell_assignments_path=cells_path,
                cell_output_path=cell_output,
                nuclear_output_path=nuclear_output,
                summary_path=summary_output,
                microns_per_pixel=0.5,
            )

            observed = pd.read_csv(cell_output)
            shared = observed.loc[observed["barcode"] == "b0"]
            self.assertEqual(set(shared["cell_id"]), {11, 22})
            self.assertAlmostEqual(float(shared["weight"].sum()), 8 / 9)
            self.assertEqual(summary["n_cell_bin_rows"], 3)
            self.assertEqual(summary["n_unique_cell_barcodes"], 2)
            self.assertEqual(summary["n_multi_cell_barcodes"], 1)
            self.assertEqual(summary["max_cells_per_barcode"], 2)

    def test_rejects_duplicate_barcode_cell_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assignments_path = root / "assignments.parquet"
            barcodes_path = root / "barcodes.tsv.gz"
            cells_path = root / "cells.csv"
            repeated = {
                "barcode_index": [0, 0],
                "array_row": [0, 0],
                "array_col": [0, 0],
                "pxl_row_in_fullres": [10.0, 10.0],
                "pxl_col_in_fullres": [20.0, 20.0],
                "cell_id": [11, 11],
                "cell_coverage_fraction": [0.5, 0.5],
                "nuclear_fraction": [0.5, 0.5],
                "cytoplasm_fraction": [0.0, 0.0],
                "membrane_fraction": [0.0, 0.0],
            }
            pd.DataFrame(repeated).to_parquet(assignments_path, index=False)
            pd.Series(["b0"]).to_csv(
                barcodes_path,
                sep="\t",
                header=False,
                index=False,
                compression="gzip",
            )
            pd.DataFrame({"cell_id": [11], "cell_type": ["A"]}).to_csv(
                cells_path, index=False
            )

            with self.assertRaisesRegex(ValueError, "duplicate .*barcode_index, cell_id"):
                export_gt_bins(
                    bin_assignments_path=assignments_path,
                    barcodes_path=barcodes_path,
                    cell_assignments_path=cells_path,
                    cell_output_path=root / "cell.csv.gz",
                    nuclear_output_path=root / "nuclear.csv.gz",
                    summary_path=root / "summary.json",
                    microns_per_pixel=0.5,
                )


if __name__ == "__main__":
    unittest.main()
