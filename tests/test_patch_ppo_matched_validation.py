from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from hd_cell_rl.patch_types import PatchActionEvent, PatchStep, PatchTrajectory
from scripts.run_patch_objective_greedy_ablation import PatchTruth
from scripts.run_patch_ppo_matched_validation import (
    _action_outcome_table,
    _assert_em_alignment,
    _owners_from_masks,
    _paired_patch_table,
    _ppo_trajectory_tables,
)


class PatchPpoMatchedValidationTests(unittest.TestCase):
    def _context(self) -> SimpleNamespace:
        return SimpleNamespace(
            patch_id="p0",
            max_steps=4,
            cells=(
                SimpleNamespace(cell_id="A", candidate_bin_ids=np.asarray(["b0", "b1"])),
                SimpleNamespace(cell_id="B", candidate_bin_ids=np.asarray(["b0", "b1"])),
            ),
        )

    def _truth(self) -> PatchTruth:
        return PatchTruth(
            patch_id="p0",
            core_cell_ids=("A", "B"),
            owner_to_gt={"A": "A", "B": "B"},
            gt_owner_by_barcode={"b0": "A", "b1": "B"},
            gt_barcodes_by_cell={"A": frozenset({"b0"}), "B": frozenset({"b1"})},
            outer_barcodes=frozenset({"b0", "b1"}),
        )

    def test_owner_masks_require_unique_physical_ownership(self) -> None:
        context = self._context()
        owners = _owners_from_masks(
            context=context,
            masks={"A": np.asarray([1, 0]), "B": np.asarray([0, 1])},
        )
        self.assertEqual(owners, {"b0": "A", "b1": "B"})
        with self.assertRaises(RuntimeError):
            _owners_from_masks(
                context=context,
                masks={"A": np.asarray([1, 0]), "B": np.asarray([1, 0])},
            )

    def test_trajectory_records_gt_outcomes_after_policy_action(self) -> None:
        event = PatchActionEvent(
            action_type="replace",
            cell_id="B",
            barcode="b1",
            old_cell_id="A",
        )
        step = PatchStep(
            global_features=np.zeros(1, dtype=np.float32),
            action_features=np.zeros((2, 1), dtype=np.float32),
            action_mask=np.ones(2, dtype=bool),
            action=1,
            reward=0.1,
            done=True,
            old_log_prob=0.0,
            old_value=0.0,
            action_events=(event,),
            chosen_action_reward=0.1,
            best_available_action_reward=0.2,
            chosen_action_reward_rank=2,
        )
        trajectory = PatchTrajectory(
            patch_slot=0,
            patch_id="p0",
            steps=(step,),
            total_reward=0.1,
            patch_score=0.2,
            metrics={},
            initial_masks={"A": np.asarray([1, 1]), "B": np.asarray([0, 0])},
            final_masks={"A": np.asarray([1, 0]), "B": np.asarray([0, 1])},
            initial_patch_score=0.1,
        )
        patch, actions, _cells = _ppo_trajectory_tables(
            context=self._context(),
            truth=self._truth(),
            trajectory=trajectory,
        )
        self.assertEqual(actions[0]["outcome"], "corrected")
        self.assertAlmostEqual(actions[0]["objective_regret"], 0.1)
        self.assertGreater(patch["macro_cell_iou_delta"], 0.0)
        self.assertEqual(patch["n_net_corrected"], 1)

    def test_em_alignment_checks_core_cell_statistics(self) -> None:
        base = pd.DataFrame(
            {
                "patch_id": ["p"],
                "stage": ["em_initial"],
                "cell_id": ["A"],
                "predicted_bins": [3],
                "gt_bins": [4],
                "intersection": [2],
                "union": [5],
                "iou": [0.4],
            }
        )
        _assert_em_alignment(ppo_cells=base, exact_cells=base.copy())
        changed = base.copy()
        changed.loc[0, "predicted_bins"] = 2
        with self.assertRaises(RuntimeError):
            _assert_em_alignment(ppo_cells=changed, exact_cells=base)

        numeric_ids = base.copy()
        numeric_ids["cell_id"] = [1]
        string_ids = numeric_ids.copy()
        string_ids["cell_id"] = ["1"]
        _assert_em_alignment(ppo_cells=string_ids, exact_cells=numeric_ids)


    def test_diagnostic_tables_preserve_method_and_patch_pairs(self) -> None:
        actions = pd.DataFrame(
            {
                "method": [
                    "w5_exact_greedy",
                    "w5_exact_greedy",
                    "frozen_ppo_greedy",
                    "frozen_ppo_greedy",
                ],
                "outcome": ["corrected", "damaged", "corrected", "no_gt"],
                "barcode": ["a", "b", "a", "c"],
                "reward": [0.2, 0.1, 0.15, 0.05],
                "macro_iou_delta": [0.1, -0.2, 0.05, -0.01],
                "chosen_action_reward_rank": [1, 1, 2, 3],
                "objective_regret": [0.0, 0.0, 0.01, 0.02],
            }
        )
        outcomes = _action_outcome_table(actions)
        self.assertEqual(int(outcomes.n_actions.sum()), 4)
        self.assertEqual(set(outcomes.method), {"w5_exact_greedy", "frozen_ppo_greedy"})

        patches = pd.DataFrame(
            {
                "patch_id": ["p", "p", "p"],
                "method": ["em_only", "w5_exact_greedy", "frozen_ppo_greedy"],
                "initial_macro_cell_iou": [0.5, 0.5, 0.5],
                "final_macro_cell_iou": [0.5, 0.6, 0.55],
                "macro_cell_iou_delta": [0.0, 0.1, 0.05],
                "n_replace_actions": [0, 4, 2],
            }
        )
        paired = _paired_patch_table(patches)
        self.assertEqual(paired.loc[0, "patch_id"], "p")
        self.assertAlmostEqual(paired.loc[0, "ppo_minus_exact_iou_delta"], -0.05)
        self.assertFalse(bool(paired.loc[0, "ppo_better_than_exact"]))



if __name__ == "__main__":
    unittest.main()
