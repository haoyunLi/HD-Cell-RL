from __future__ import annotations

import argparse
import gzip
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io, sparse

from scripts.build_patch_capture_stress_dataset import run


class BuildPatchCaptureStressDatasetTest(unittest.TestCase):
    def test_builds_aligned_high_and_thinned_patch_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            matrix_dir = source / "filtered_feature_bc_matrix"
            spatial_dir = source / "spatial"
            matrix_dir.mkdir(parents=True)
            spatial_dir.mkdir(parents=True)

            matrix = sparse.coo_matrix(
                np.asarray(
                    [
                        [2, 4, 0, 5],
                        [0, 3, 8, 1],
                        [7, 0, 6, 0],
                    ],
                    dtype=np.int32,
                )
            )
            plain_matrix = matrix_dir / "matrix.mtx"
            io.mmwrite(plain_matrix, matrix, field="integer")
            with plain_matrix.open("rb") as handle, gzip.open(
                matrix_dir / "matrix.mtx.gz", "wb"
            ) as output:
                shutil.copyfileobj(handle, output)
            plain_matrix.unlink()
            with gzip.open(matrix_dir / "features.tsv.gz", "wt") as handle:
                for index in range(3):
                    handle.write(f"g{index}\tg{index}\tGene Expression\n")
            with gzip.open(matrix_dir / "barcodes.tsv.gz", "wt") as handle:
                for index in range(4):
                    handle.write(f"b{index}\n")

            positions = pd.DataFrame(
                {
                    "barcode": ["b0", "b1", "b2", "b3", "outside"],
                    "in_tissue": [1, 1, 1, 1, 0],
                    "array_row": [0, 0, 1, 1, 2],
                    "array_col": [0, 1, 0, 1, 2],
                }
            )
            positions.to_parquet(spatial_dir / "tissue_positions.parquet", index=False)
            (spatial_dir / "scalefactors_json.json").write_text("{}\n")

            metadata = pd.DataFrame(
                {
                    "matrix_col_index": np.arange(4),
                    "barcode": ["b0", "b1", "b2", "b3"],
                    "x_um": [0.0, 10.0, 20.0, 30.0],
                    "y_um": [0.0, 10.0, 20.0, 30.0],
                }
            )
            metadata_path = root / "metadata.parquet"
            metadata.to_parquet(metadata_path, index=False)
            patches = pd.DataFrame(
                {
                    "context_x_min": [5.0],
                    "context_x_max": [25.0],
                    "context_y_min": [5.0],
                    "context_y_max": [25.0],
                }
            )
            patch_path = root / "patches.csv"
            patches.to_csv(patch_path, index=False)
            output_root = root / "output"

            run(
                argparse.Namespace(
                    input_pseudo_hd_dir=str(source),
                    bin_metadata=str(metadata_path),
                    patch_index=str(patch_path),
                    output_root=str(output_root),
                    capture_probability=0.5,
                    seed=17,
                )
            )

            high = io.mmread(
                output_root / "high_depth_local/filtered_feature_bc_matrix/matrix.mtx.gz"
            ).toarray()
            thinned = io.mmread(
                output_root / "capture_thinned/filtered_feature_bc_matrix/matrix.mtx.gz"
            ).toarray()
            np.testing.assert_array_equal(high[:, [0, 3]], 0)
            np.testing.assert_array_equal(high[:, 1:3], matrix.toarray()[:, 1:3])
            self.assertTrue(np.all(thinned <= high))
            self.assertTrue(np.all(thinned[:, [0, 3]] == 0))

            masked = pd.read_parquet(
                output_root / "capture_thinned/spatial/tissue_positions.parquet"
            )
            self.assertEqual(masked["in_tissue"].tolist(), [0, 1, 1, 0, 0])
            self.assertTrue(
                (output_root / "high_depth_local/filtered_feature_bc_matrix/barcodes.tsv.gz").is_symlink()
            )
            self.assertTrue((output_root / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
