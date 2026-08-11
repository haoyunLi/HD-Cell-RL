from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from hd_cell_rl.episode_artifacts import _MatrixOnDemandExpressionLoader


class EpisodeArtifactsTests(unittest.TestCase):
    def test_matrix_ll_count_cache_preserves_results_and_invalidates_by_log_theta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            matrix_path = root / "matrix.h5"
            reference_path = root / "reference.npz"
            np.savez_compressed(reference_path, genes=np.asarray(["g0", "g1"], dtype="U"))

            with h5py.File(matrix_path, "w") as h5:
                g = h5.create_group("matrix")
                g.create_dataset("shape", data=np.asarray([3, 3], dtype=np.int64))
                g.create_dataset("data", data=np.asarray([2.0, 1.0, 3.0, 5.0], dtype=np.float64))
                g.create_dataset("indices", data=np.asarray([0, 1, 1, 2], dtype=np.int64))
                g.create_dataset("indptr", data=np.asarray([0, 2, 3, 4], dtype=np.int64))
                fg = g.create_group("features")
                fg.create_dataset("name", data=np.asarray([b"g0", b"g1", b"g2"]))

            log_theta = np.log(np.asarray([[0.8, 0.2], [0.25, 0.75]], dtype=np.float64))
            loader = _MatrixOnDemandExpressionLoader(
                matrix_path=matrix_path,
                reference_npz_path=reference_path,
                reference_genes_key="genes",
                cache_size=10,
            )
            try:
                ll, counts = loader.compute_ll_and_bin_counts_for_columns(
                    col_index=np.asarray([0, 1, 2, 0], dtype=np.int64),
                    log_theta=log_theta,
                )
                ll_cached, counts_cached = loader.compute_ll_and_bin_counts_for_columns(
                    col_index=np.asarray([1, 0, 2], dtype=np.int64),
                    log_theta=log_theta,
                )

                np.testing.assert_allclose(ll_cached[0], ll[1])
                np.testing.assert_allclose(ll_cached[1], ll[0])
                np.testing.assert_allclose(ll_cached[2], ll[2])
                np.testing.assert_allclose(counts_cached, np.asarray([3.0, 3.0, 0.0], dtype=np.float64))
                self.assertEqual(len(loader._ll_cache), 3)

                other_log_theta = np.log(np.asarray([[0.5, 0.5], [0.9, 0.1]], dtype=np.float64))
                other_ll, _ = loader.compute_ll_and_bin_counts_for_columns(
                    col_index=np.asarray([0], dtype=np.int64),
                    log_theta=other_log_theta,
                )
                self.assertFalse(np.allclose(other_ll[0], ll[0]))
                self.assertEqual(len(loader._ll_cache), 1)
            finally:
                loader.close()


if __name__ == "__main__":
    unittest.main()
