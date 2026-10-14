from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from scripts.build_colorectal_pseudo_v3_dataset import (
    calibrate_capture_probabilities,
    redistribute_background_ambient,
    smoothed_capture_field,
)
from scripts.build_noisy_nuclear_observations import boundary_mask_for_group
from scripts.prepare_colorectal_pseudo_v3_inputs import parse_remaps, select_context_bins
from scripts.pad_patch_fractional_gt import FRACTION_COLUMNS, build_padded_fractional_gt


class PreparePseudoV3InputsTests(unittest.TestCase):
    def test_parse_remaps_and_reject_conflicts(self) -> None:
        self.assertEqual(
            parse_remaps(["Immature Enterocytes=Enterocytes"]),
            {"Immature Enterocytes": "Enterocytes"},
        )
        with self.assertRaises(ValueError):
            parse_remaps(["A=B", "A=C"])

    def test_context_selection_uses_all_patch_bounds(self) -> None:
        metadata = pd.DataFrame(
            {"x_um": [0.0, 5.0, 20.0], "y_um": [0.0, 5.0, 20.0]}
        )
        patches = pd.DataFrame(
            {
                "context_x_min": [-1.0, 19.0],
                "context_x_max": [6.0, 21.0],
                "context_y_min": [-1.0, 19.0],
                "context_y_max": [6.0, 21.0],
            }
        )
        np.testing.assert_array_equal(
            select_context_bins(metadata, patches), np.asarray([True, True, True])
        )


class PseudoV3CountBuilderTests(unittest.TestCase):
    def test_patch_fractional_gt_padding_preserves_multi_owner_pairs(self) -> None:
        full = pd.DataFrame(
            {
                "barcode_index": [0, 1, 1, 2],
                "array_row": [0, 0, 0, 0],
                "array_col": [0, 1, 1, 2],
                "cell_id": [0, 10, 11, 12],
                "cell_coverage_fraction": [0.0, 0.4, 0.5, 1.0],
                "nuclear_fraction": [0.0, 0.1, 0.0, 0.0],
                "cytoplasm_fraction": [0.0, 0.3, 0.5, 1.0],
                "membrane_fraction": [0.0, 0.0, 0.0, 0.0],
            }
        )
        padded, summary = build_padded_fractional_gt(full, {10, 11})
        self.assertEqual(set(padded["barcode_index"]), {0, 1, 2})
        selected = padded.loc[padded["barcode_index"].eq(1)]
        self.assertEqual(set(selected["cell_id"]), {10, 11})
        zero_rows = padded.loc[padded["cell_id"].eq(0)]
        self.assertEqual(set(zero_rows["barcode_index"]), {0, 2})
        self.assertTrue((zero_rows[list(FRACTION_COLUMNS)] == 0).all().all())
        self.assertEqual(summary["n_multi_owner_positive_barcodes"], 1)

    def test_capture_calibration_is_exact_and_spatially_heterogeneous(self) -> None:
        source = np.asarray([100.0, 100.0, 100.0])
        field = np.asarray([1.0, 2.0, 4.0])
        probabilities = calibrate_capture_probabilities(source, field, 120)
        self.assertAlmostEqual(float(np.dot(source, probabilities)), 120.0, places=6)
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))
        self.assertGreater(probabilities[2], probabilities[1])
        self.assertGreater(probabilities[1], probabilities[0])

    def test_capture_field_is_finite_and_positive(self) -> None:
        coordinates = np.asarray([[0.0, 0.0], [2.0, 0.0], [100.0, 100.0]])
        totals = np.asarray([0.0, 10.0, 30.0])
        field = smoothed_capture_field(
            coordinates, totals, sigma_um=4.0, n_neighbors=3
        )
        self.assertTrue(np.isfinite(field).all())
        self.assertTrue((field > 0).all())

    def test_ambient_redistribution_conserves_gene_counts_and_radius(self) -> None:
        real = sparse.coo_matrix(
            (
                np.asarray([2, 3], dtype=np.int32),
                (np.asarray([0, 1]), np.asarray([0, 0])),
            ),
            shape=(2, 3),
        )
        selected_columns = np.asarray([0, 1, 2], dtype=np.int64)
        coordinates = np.asarray([[0.0, 0.0], [2.0, 0.0], [100.0, 0.0]])
        ambient = redistribute_background_ambient(
            real_matrix=real,
            selected_columns=selected_columns,
            background_columns={0},
            coordinates=coordinates,
            sigma_um=2.0,
            radius_um=4.0,
            rng=np.random.default_rng(7),
        )
        self.assertEqual(int(ambient.sum()), 5)
        np.testing.assert_array_equal(
            np.asarray(ambient.sum(axis=1)).ravel(), np.asarray([2, 3])
        )
        self.assertFalse(np.any(ambient.col == 2))


class NuclearNoiseTests(unittest.TestCase):
    def test_boundary_detection_distinguishes_center_bin(self) -> None:
        group = pd.DataFrame(
            {
                "array_row": [1, 0, 2, 1, 1],
                "array_col": [1, 1, 1, 0, 2],
            }
        )
        boundary = boundary_mask_for_group(group)
        np.testing.assert_array_equal(
            boundary, np.asarray([False, True, True, True, True])
        )


if __name__ == "__main__":
    unittest.main()
