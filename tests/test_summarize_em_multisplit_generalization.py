from __future__ import annotations

import json
import unittest

import pandas as pd

from scripts.summarize_em_multisplit_generalization import (
    audit_patch_groups,
    standardize_cell_metrics,
)


def _patch(patch_id: str, x: float, cell_id: str) -> dict[str, object]:
    return {
        "patch_id": patch_id,
        "patch_cell_ids": json.dumps([cell_id]),
        "context_x_min": x,
        "context_x_max": x + 10.0,
        "context_y_min": 0.0,
        "context_y_max": 10.0,
    }


class SummarizeEMMultisplitGeneralizationTests(unittest.TestCase):
    def test_standardizes_evaluator_and_direct_metrics(self) -> None:
        evaluated = standardize_cell_metrics(
            pd.DataFrame(
                {
                    "cell_id": [1, 2],
                    "pred_iou": [0.5, 0.75],
                    "pred_precision": [0.6, 0.8],
                    "pred_recall": [0.7, 0.9],
                }
            ),
            source="evaluated",
        )
        direct = standardize_cell_metrics(
            pd.DataFrame(
                {
                    "cell_id": [1, 2],
                    "iou": [0.5, 0.75],
                    "precision": [0.6, 0.8],
                    "recall": [0.7, 0.9],
                }
            ),
            source="direct",
        )
        pd.testing.assert_series_equal(evaluated["iou"], direct["iou"])
        self.assertEqual(evaluated["cell_id"].tolist(), ["1", "2"])

    def test_patch_group_audit_detects_each_overlap_type(self) -> None:
        groups = {
            "a": (pd.DataFrame([_patch("p1", 0.0, "c1")]), {"b1", "b2"}),
            "b": (pd.DataFrame([_patch("p2", 30.0, "c2")]), {"b3"}),
            "c": (pd.DataFrame([_patch("p1", 5.0, "c1")]), {"b2"}),
        }
        audit = audit_patch_groups(groups)
        ab = audit.loc[
            (audit["left_split"] == "a") & (audit["right_split"] == "b")
        ].iloc[0]
        ac = audit.loc[
            (audit["left_split"] == "a") & (audit["right_split"] == "c")
        ].iloc[0]
        self.assertEqual(int(ab["patch_id_overlap"]), 0)
        self.assertFalse(bool(ab["context_rectangle_overlap"]))
        self.assertEqual(int(ac["patch_id_overlap"]), 1)
        self.assertEqual(int(ac["context_cell_id_overlap"]), 1)
        self.assertEqual(int(ac["em_barcode_overlap"]), 1)
        self.assertTrue(bool(ac["context_rectangle_overlap"]))


if __name__ == "__main__":
    unittest.main()
