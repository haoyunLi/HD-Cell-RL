from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from scripts.run_patch_reward_formula_rollout import (
    _calibrated_shape_weight,
    _evolution_table,
    _parse_formula_names,
    _symmetric_compatibility_raw,
    _summarize,
)


class PatchRewardFormulaRolloutTests(unittest.TestCase):
    def test_formula_names_and_calibrated_shape_weight(self) -> None:
        self.assertEqual(
            _parse_formula_names("baseline,w5,symmetric"),
            ("baseline", "w5", "symmetric"),
        )
        self.assertAlmostEqual(
            _calibrated_shape_weight(
                expression_weight=0.15,
                shape_lambda=0.25,
                expression_iqr=0.5,
                shape_iqr=3.0,
            ),
            0.00625,
        )
        with self.assertRaises(ValueError):
            _parse_formula_names("baseline,unknown")

    def test_validation_selection_uses_iou_then_damage(self) -> None:
        patch_df = pd.DataFrame(
            {
                "split": ["validation", "validation"],
                "formula": ["a", "b"],
                "expression_mode": ["x", "x"],
                "patch_id": ["p", "p"],
                "initial_owner_correct": [8, 8],
                "initial_owner_evaluated": [10, 10],
                "final_owner_correct": [8, 9],
                "final_owner_evaluated": [10, 10],
                "n_net_corrected": [2, 3],
                "n_net_damaged": [1, 2],
                "n_net_no_gt": [0, 0],
                "termination_reason": ["no_formula_delta_above_epsilon"] * 2,
            }
        )
        action_df = pd.DataFrame(
            {
                "formula": ["a", "b"],
                "outcome": ["corrected", "corrected"],
                "formula_reward_residual": [0.0, 0.0],
            }
        )
        cell_df = pd.DataFrame(
            {
                "formula": ["a", "a", "b", "b"],
                "stage": ["em_initial", "formula_greedy"] * 2,
                "iou": [0.7, 0.72, 0.7, 0.75],
            }
        )
        summary = _summarize(patch_df, action_df, cell_df)
        self.assertEqual(summary.iloc[0]["formula"], "b")
        self.assertTrue(bool(summary.iloc[0]["selected_on_validation"]))

    def test_evolution_includes_zero_action_patch(self) -> None:
        patch_df = pd.DataFrame(
            {
                "split": ["validation", "validation"],
                "formula": ["a", "a"],
                "patch_id": ["p0", "p1"],
                "initial_macro_cell_iou": [0.5, 0.7],
            }
        )
        action_df = pd.DataFrame(
            {
                "formula": ["a"],
                "patch_id": ["p1"],
                "step_index": [1],
                "macro_iou_after": [0.9],
            }
        )
        evolution = _evolution_table(patch_df, action_df, n_points=3)
        np.testing.assert_allclose(
            evolution["mean_macro_cell_iou"].to_numpy(),
            np.asarray([0.6, 0.65, 0.7]),
        )

    def test_symmetric_compatibility_uses_float32_buffers(self) -> None:
        env = type("FakeEnv", (), {})()
        env._torch = torch
        env._device = torch.device("cpu")
        env._cached_action_cells = torch.tensor([0], dtype=torch.long)
        env._cached_action_bins = torch.tensor([0], dtype=torch.long)
        env._cached_action_old_cells = torch.tensor([1], dtype=torch.long)
        env._cached_action_old_bins = torch.tensor([0], dtype=torch.long)
        env._score_sums = (
            torch.tensor([2.0, 0.0], dtype=torch.float32),
            torch.tensor([0.0, 2.0], dtype=torch.float32),
        )
        env._ll64 = (
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        )
        env._expression_conf64 = (
            torch.tensor([0.5], dtype=torch.float32),
            torch.tensor([0.5], dtype=torch.float32),
        )

        raw = _symmetric_compatibility_raw(env)
        target = torch.softmax(env._score_sums[0], dim=0)[0]
        old = torch.softmax(env._score_sums[1], dim=0)[0]
        expected = 0.5 * (target - old)
        self.assertEqual(raw.dtype, torch.float32)
        self.assertAlmostEqual(float(raw[0]), float(expected), places=6)


if __name__ == "__main__":
    unittest.main()
