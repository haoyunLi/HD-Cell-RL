from __future__ import annotations

import unittest

import pandas as pd

from scripts import analyze_patch_reward_attribution as attribution


class RewardAttributionTests(unittest.TestCase):
    def _payload(self):
        return {
            "patch_id": "patch_1",
            "n_steps": 1,
            "cells": [
                {"cell_id": "A", "matched_gt_cell_id": "gt_A"},
                {"cell_id": "B", "matched_gt_cell_id": "gt_B"},
            ],
            "bins": [
                {
                    "barcode": "b1",
                    "gt_owner_cell_id": "gt_A",
                    "em_normalized_entropy": 0.8,
                    "em_is_nuclear_locked": False,
                    "em_is_ambiguous": True,
                    "inside_core": True,
                },
                {
                    "barcode": "b2",
                    "gt_owner_cell_id": None,
                    "em_normalized_entropy": 0.6,
                    "em_is_nuclear_locked": False,
                    "em_is_ambiguous": True,
                    "inside_core": True,
                },
            ],
            "trajectory": {
                "initial_owners": [
                    {"barcode": "b1", "cell_id": "A"},
                    {"barcode": "b2", "cell_id": "A"},
                ],
                "steps": [
                    {
                        "step_index": 1,
                        "reward": 0.12,
                        "actions": [
                            {
                                "type": "replace",
                                "cell_id": "B",
                                "old_cell_id": "A",
                                "barcode": "b1",
                                "applied": True,
                            }
                        ],
                        "objective_delta_components": {
                            "expression_confidence": 0.10,
                            "expression_compatibility": 0.0,
                            "distance_penalty": -0.01,
                            "overlap_penalty": 0.0,
                            "neighbor_support": 0.03,
                            "shape_prior": 0.0,
                            "component_sum": 0.12,
                            "reward_residual": 0.0,
                        },
                    }
                ],
            },
        }

    def test_replace_outcome_and_components(self):
        rows = attribution._collect_replace_rows([self._payload()])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["outcome"], "damaged")
        self.assertAlmostEqual(rows[0]["component_sum"], 0.12)
        self.assertAlmostEqual(rows[0]["reward_residual"], 0.0)
        self.assertEqual(rows[0]["dominant_positive_component"], "expression_confidence")
        self.assertEqual(rows[0]["chosen_action_rank_fraction"], 0.0)
        self.assertTrue(pd.isna(rows[0]["chosen_reward_fraction_of_best"]))

    def test_entropy_calibration_uses_initial_em_owner(self):
        payload = self._payload()
        action_rows = attribution._collect_replace_rows([payload])
        rows = attribution._collect_entropy_rows([payload], action_rows)
        by_barcode = {row["barcode"]: row for row in rows}
        self.assertTrue(bool(by_barcode["b1"]["em_correct"]))
        self.assertTrue(bool(by_barcode["b1"]["rl_modified"]))
        self.assertFalse(bool(by_barcode["b2"]["rl_modified"]))

        summary = attribution._entropy_threshold_summary(
            pd.DataFrame(rows),
            thresholds=(0.5, 0.9),
        )
        self.assertEqual(int(summary.iloc[0]["n_eligible_bins"]), 1)
        self.assertEqual(int(summary.iloc[1]["n_eligible_bins"]), 0)

    def test_drop_one_counterfactual(self):
        actions = pd.DataFrame(
            attribution._collect_replace_rows([self._payload()])
        )
        rows = attribution._drop_one_counterfactual(
            actions,
            epsilon=1.0e-6,
        )
        expression = rows.loc[
            rows["removed_component"] == "expression_confidence"
        ].iloc[0]
        self.assertAlmostEqual(
            float(expression["counterfactual_reward"]),
            0.02,
        )
        self.assertTrue(bool(expression["would_remain_exposed"]))
        neighbor = rows.loc[
            rows["removed_component"] == "neighbor_support"
        ].iloc[0]
        self.assertAlmostEqual(
            float(neighbor["counterfactual_reward"]),
            0.09,
        )

    def test_component_scale_counterfactual(self):
        actions = pd.DataFrame(
            attribution._collect_replace_rows([self._payload()])
        )
        summary = attribution._component_scale_summary(
            actions,
            epsilon=1.0e-6,
        )
        expression_zero = summary.loc[
            (summary["scaled_component"] == "expression_confidence")
            & (summary["scale"] == 0.0)
        ].iloc[0]
        self.assertEqual(
            int(expression_zero["n_would_remain_exposed"]),
            1,
        )
        self.assertAlmostEqual(
            float(expression_zero["mean_counterfactual_reward"]),
            0.02,
        )


if __name__ == "__main__":
    unittest.main()
