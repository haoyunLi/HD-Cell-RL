from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from scripts.audit_patch_replace_rewards import (
    _assign_splits,
    _binary_auc,
    _classify_outcome,
    _posterior_certainty,
    _select_spatially_separated_patches,
)


class PatchReplaceRewardAuditTests(unittest.TestCase):
    def test_posterior_certainty_is_bounded(self) -> None:
        posterior = torch.tensor(
            [[0.5, 0.5], [0.99, 0.01]],
            dtype=torch.float64,
        )
        certainty = _posterior_certainty(posterior).numpy()
        self.assertAlmostEqual(float(certainty[0]), 0.0, places=7)
        self.assertGreater(float(certainty[1]), 0.9)
        self.assertTrue(np.all((certainty >= 0.0) & (certainty <= 1.0)))

    def test_auc_and_outcome_labels(self) -> None:
        self.assertAlmostEqual(_binary_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0)
        self.assertEqual(_classify_outcome("a", "b", "a"), "corrected")
        self.assertEqual(_classify_outcome("a", "a", "b"), "damaged")
        self.assertEqual(_classify_outcome(None, "a", "b"), "no_gt")
        self.assertEqual(_classify_outcome(np.nan, "a", "b"), "no_gt")
        self.assertEqual(_classify_outcome("c", "a", "b"), "wrong_to_wrong")

    def test_spatial_selection_and_splits(self) -> None:
        patches = pd.DataFrame(
            {
                "patch_id": ["a", "b", "c", "d"],
                "center_x_um": [0.0, 10.0, 100.0, 200.0],
                "center_y_um": [0.0, 0.0, 0.0, 0.0],
                "n_core_cells": [5, 5, 5, 5],
            }
        )
        selected = _select_spatially_separated_patches(
            patches,
            n_patches=3,
            min_core_cells=5,
            min_center_distance_um=50.0,
            seed=4,
        )
        coords = selected[["center_x_um", "center_y_um"]].to_numpy()
        for i in range(len(coords)):
            for j in range(i):
                self.assertGreaterEqual(float(np.linalg.norm(coords[i] - coords[j])), 50.0)
        split = _assign_splits(selected, n_calibration=1, n_validation=1)
        self.assertEqual(split["audit_split"].tolist(), ["calibration", "validation", "test"])


if __name__ == "__main__":
    unittest.main()
