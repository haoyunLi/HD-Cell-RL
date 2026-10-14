from __future__ import annotations

import math
import unittest

from scripts.run_patch_objective_greedy_ablation import (
    PatchTruth,
    _cell_iou_rows,
    _owner_accuracy,
    _parse_float_list,
    _transition_outcome,
)


class PatchObjectiveGreedyAblationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.truth = PatchTruth(
            patch_id="patch_test",
            core_cell_ids=("pred_a", "pred_b"),
            owner_to_gt={"pred_a": "gt_a", "pred_b": "gt_b"},
            gt_owner_by_barcode={"b1": "gt_a", "b2": "gt_a", "b3": "gt_b"},
            gt_barcodes_by_cell={
                "gt_a": frozenset(("b1", "b2")),
                "gt_b": frozenset(("b3",)),
            },
            outer_barcodes=frozenset(("b1", "b2", "b3", "outside")),
        )

    def test_cell_iou_and_owner_accuracy(self) -> None:
        owners = {"b1": "pred_a", "b2": "pred_b", "b3": "pred_b"}
        rows = _cell_iou_rows(owners, self.truth, stage="test")
        by_cell = {row["cell_id"]: row for row in rows}
        self.assertAlmostEqual(by_cell["pred_a"]["iou"], 0.5)
        self.assertAlmostEqual(by_cell["pred_b"]["iou"], 0.5)
        correct, evaluated, accuracy = _owner_accuracy(owners, self.truth)
        self.assertEqual((correct, evaluated), (2, 3))
        self.assertTrue(math.isclose(accuracy, 2.0 / 3.0))

    def test_transition_outcomes_are_gt_evaluation_only(self) -> None:
        self.assertEqual(
            _transition_outcome(
                barcode="b2", old_owner="pred_b", new_owner="pred_a", truth=self.truth
            ),
            "corrected",
        )
        self.assertEqual(
            _transition_outcome(
                barcode="b1", old_owner="pred_a", new_owner="pred_b", truth=self.truth
            ),
            "damaged",
        )
        self.assertEqual(
            _transition_outcome(
                barcode="outside", old_owner="pred_a", new_owner="pred_b", truth=self.truth
            ),
            "no_gt",
        )

    def test_float_list_validation(self) -> None:
        self.assertEqual(_parse_float_list("0, 0.25,1", name="scale"), (0.0, 0.25, 1.0))
        with self.assertRaises(ValueError):
            _parse_float_list("", name="scale")
        with self.assertRaises(ValueError):
            _parse_float_list("-1", name="scale")


if __name__ == "__main__":
    unittest.main()
