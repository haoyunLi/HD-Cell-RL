"""Small, data-independent checks for the paired HD/Xenium benchmark adapters."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from shapely.geometry import box
import tifffile

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from evaluate_lung_paired_patch import (
    _nuclear_majority_votes, alpha_shape_um, evaluate_method,
    select_reciprocal_matches,
)
from qc_lung_paired_alignment import (
    fit_hd_grid_to_he_px, project_points, transform_xenium_um_to_he_px,
)


class PairedBenchmarkAdapterTests(unittest.TestCase):
    def test_xenium_um_to_he_pixel_uses_both_scales_and_inverse_alignment(self):
        he_to_dapi = np.array([[0.02, 1.25, -100], [-1.25, 0.02, 800], [0, 0, 1]])
        he = np.array([[123.5, 456.25], [200.0, 300.0]])
        dapi_px = np.column_stack((he, np.ones(len(he)))) @ he_to_dapi.T
        geometry = {"xenium_microns_per_pixel": 0.2125,
                    "he_to_xenium_pixels": he_to_dapi}
        result = transform_xenium_um_to_he_px(dapi_px[:, :2] * 0.2125, geometry)
        np.testing.assert_allclose(result, he, atol=1e-10)

    def test_projective_hd_grid_is_not_assumed_affine(self):
        x, y = np.meshgrid(np.arange(0, 100, 10), np.arange(0, 100, 10))
        grid = np.column_stack((x.ravel(), y.ravel()))
        transform = np.array([[7.3, -0.1, 500], [0.05, 7.3, 700],
                              [1e-5, -2e-5, 1]])
        pixels = project_points(grid, transform)
        positions = pd.DataFrame({"array_col": grid[:, 0], "array_row": grid[:, 1],
                                  "pxl_col_in_fullres": pixels[:, 0],
                                  "pxl_row_in_fullres": pixels[:, 1]})
        fitted = fit_hd_grid_to_he_px(positions)
        np.testing.assert_allclose(project_points(grid, fitted), pixels, atol=1e-3)

    def test_nuclear_vote_is_pixel_majority(self):
        labels = np.array([[1, 1, 1, 0], [1, 1, 2, 2],
                           [0, 0, 2, 2]], dtype=np.int32)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels.tif"
            tifffile.imwrite(path, labels)
            # Cell 1 has three votes for X and two for Y; cell 2 votes for Y.
            votes, diagnostic = _nuclear_majority_votes(
                path, ["X", "Y"], [box(-0.5, -0.5, 2.5, 0.5),
                                   box(-0.5, 0.5, 3.5, 2.5)], (0, 0), 1.0,
            )
        self.assertEqual(votes, {"1": "X", "2": "Y"})
        self.assertEqual(diagnostic["n_stardist_nuclear_pixels"], 9)

    def test_alpha_shape_and_duplicate_barcode_guard(self):
        polygon = alpha_shape_um(np.array([[0, 0], [2, 0], [2, 2], [0, 2]]))
        self.assertTrue(polygon.is_valid)
        self.assertGreater(polygon.area, 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "duplicated.csv"
            pd.DataFrame({"barcode": ["one", "one"],
                          "cell_id": ["A", "B"]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate ownership"):
                evaluate_method(
                    name="bad", assignment_path=path, patch=Path(folder),
                    core_cells=set(), xe_core_ids=set(), xe_ids=[],
                    xe_polygons=[], nuclear_votes={}, gene_context={},
                    hd_mpp=0.2737,
                )

    def test_bidirectional_maximum_intersection_and_nuclear_constraint(self):
        overlaps = [("A", "X", 10.0), ("A", "Y", 20.0),
                    ("B", "X", 12.0), ("B", "Y", 5.0)]
        votes = {"A": "Y", "B": "X"}
        chosen = select_reciprocal_matches(overlaps, votes, {"X", "Y"})
        self.assertEqual({(a, b) for a, b, _ in chosen}, {("A", "Y"), ("B", "X")})
        reversed_chosen = select_reciprocal_matches(list(reversed(overlaps)), votes, {"X", "Y"})
        self.assertEqual({(a, b) for a, b, _ in reversed_chosen},
                         {("A", "Y"), ("B", "X")})
        self.assertEqual(select_reciprocal_matches(overlaps, {"A": "X", "B": "Y"},
                                                   {"X", "Y"}), [])


if __name__ == "__main__":
    unittest.main()
