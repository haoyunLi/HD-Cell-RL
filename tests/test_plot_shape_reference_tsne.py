from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from preprocessing.plot_shape_reference_tsne import _merge_small_clusters, run_shape_reference_tsne


class PlotShapeReferenceTsneTests(unittest.TestCase):
    def test_writes_same_embedding_with_cluster_and_cell_type_colors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            npz_path = root / "shape_reference.npz"
            output_dir = root / "tsne"
            shape_features = np.asarray(
                [
                    [1.0, 0.7, 1.0, 1.0],
                    [1.1, 0.6, 0.9, 1.2],
                    [0.9, 0.8, 1.0, 1.1],
                    [3.0, 0.3, 0.7, 9.0],
                    [3.1, 0.2, 0.6, 8.0],
                    [2.9, 0.4, 0.8, 7.0],
                ],
                dtype=np.float64,
            )
            z = (shape_features - shape_features.mean(axis=0)) / shape_features.std(axis=0)
            np.savez_compressed(
                npz_path,
                cell_ids=np.asarray([f"c{i}" for i in range(shape_features.shape[0])], dtype="U"),
                cell_types=np.asarray(["A", "A", "A", "B", "B", "B"], dtype="U"),
                feature_names=np.asarray(["log_area", "compactness", "solidity", "anisotropy"], dtype="U"),
                shape_features=shape_features,
                shape_features_zscore=z,
            )

            outputs = run_shape_reference_tsne(
                shape_reference_npz=npz_path,
                output_dir=output_dir,
                prefix="test_shape",
                n_clusters="2",
                perplexity=2.0,
                max_cells=0,
                random_seed=1,
                n_iter=250,
                point_size=8.0,
            )
            for path in outputs.values():
                self.assertTrue(path.exists())
            clustered_reference = pd.read_csv(outputs["clustered_reference"])
            self.assertIn("shape_cluster", clustered_reference.columns)
            self.assertEqual(set(clustered_reference["shape_cluster"]), set(clustered_reference["cluster"]))

            redraw_dir = root / "redraw"
            redraw_outputs = run_shape_reference_tsne(
                shape_reference_npz=None,
                coordinates_csv=outputs["coordinates"],
                output_dir=redraw_dir,
                prefix="test_shape_redraw",
            )
            self.assertTrue(redraw_outputs["cell_type_heatmap"].exists())
            self.assertTrue(redraw_outputs["cluster_heatmap"].exists())
            self.assertTrue(redraw_outputs["cell_type_means"].exists())
            self.assertTrue(redraw_outputs["cluster_means"].exists())
            self.assertTrue(redraw_outputs["clustered_reference"].exists())

    def test_merges_small_clusters_into_nearest_large_cluster(self) -> None:
        labels = pd.Series(["cluster_00"] * 4 + ["cluster_01"] * 4 + ["cluster_02"])
        features = np.asarray(
            [[0.0, 0.0]] * 4
            + [[10.0, 10.0]] * 4
            + [[10.2, 10.1]],
            dtype=np.float64,
        )

        merged, summary = _merge_small_clusters(
            labels=labels,
            features=features,
            min_cluster_size=3,
        )

        self.assertEqual(summary["n_clusters_before"], 3)
        self.assertEqual(summary["n_clusters_after"], 2)
        self.assertEqual(summary["merged_clusters"]["cluster_02"]["n_cells"], 1)
        self.assertEqual(merged.nunique(), 2)
        self.assertEqual(sorted(merged.value_counts().tolist()), [4, 5])


if __name__ == "__main__":
    unittest.main()
