from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from preprocessing.patch_debug_export import export_patch_debug_bundle


class PatchDebugExportTest(unittest.TestCase):
    def test_exports_patch_owner_and_overlap_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assignments_path = root / "assignments.csv"
            patch_index_path = root / "patches.csv"
            per_episode_path = root / "per_episode.csv"
            gt_path = root / "gt.csv"
            output_dir = root / "patch_debug"

            pd.DataFrame(
                [
                    _assignment("p1", "A", "b00", 0, 0),
                    _assignment("p1", "A", "b01", 0, 1),
                    _assignment("p1", "A", "b20", 2, 0),
                    _assignment("p1", "B", "b10", 1, 0),
                    _assignment("p1", "B", "b11", 1, 1),
                ]
            ).to_csv(assignments_path, index=False)
            pd.DataFrame(
                [
                    {
                        "patch_id": "p1",
                        "outer_x_min": -1.0,
                        "outer_x_max": 5.0,
                        "outer_y_min": -1.0,
                        "outer_y_max": 5.0,
                        "core_x_min": 0.0,
                        "core_x_max": 4.0,
                        "core_y_min": 0.0,
                        "core_y_max": 4.0,
                        "core_cell_ids": json.dumps(["A", "B"]),
                        "margin_cell_ids": json.dumps([]),
                    }
                ]
            ).to_csv(patch_index_path, index=False)
            pd.DataFrame(
                [
                    {
                        "cell_id": "A",
                        "matched_pred_cell_id": "A",
                        "matched_gt_cell_id": "G1",
                        "pred_nuclear_overlap_bins": 3,
                        "pred_iou": 0.5,
                        "pred_dice": 2.0 / 3.0,
                    },
                    {
                        "cell_id": "B",
                        "matched_pred_cell_id": "B",
                        "matched_gt_cell_id": "G2",
                        "pred_nuclear_overlap_bins": 2,
                        "pred_iou": 0.25,
                        "pred_dice": 0.4,
                    },
                ]
            ).to_csv(per_episode_path, index=False)
            pd.DataFrame(
                [
                    _gt("G1", "b00", 0, 0),
                    _gt("G1", "b01", 0, 1),
                    _gt("G2", "b10", 1, 0),
                    _gt("G2", "b12", 1, 2),
                    _gt("G2", "b20", 2, 0),
                ]
            ).to_csv(gt_path, index=False)

            manifest_path = export_patch_debug_bundle(
                patch_assignments_csv=assignments_path,
                patches_index_path=patch_index_path,
                per_episode_csv=per_episode_path,
                gt_cell_bins_path=gt_path,
                output_dir=output_dir,
                patch_ids=["p1"],
                trajectory_metadata={
                    "p1": {
                        "patch_score": 1.25,
                        "total_reward": 7.5,
                        "n_steps": 5,
                        "metrics": {"n_patch_steps": 5.0},
                        "trajectory": {
                            "available": True,
                            "initial_patch_score": 0.25,
                            "initial_raw_patch_score": 0.5,
                            "initial_owned_target_count": 2,
                            "target_count": 5,
                            "initial_owners": [
                                {"barcode": "b00", "cell_id": "A"},
                                {"barcode": "b10", "cell_id": "B"},
                            ],
                            "final_owners": [
                                {"barcode": "b00", "cell_id": "A"},
                                {"barcode": "b01", "cell_id": "A"},
                                {"barcode": "b10", "cell_id": "B"},
                                {"barcode": "b11", "cell_id": "B"},
                                {"barcode": "b22", "cell_id": "B"},
                            ],
                            "steps": [
                                {
                                    "step_index": 1,
                                    "reward": 1.0,
                                    "cumulative_reward": 1.0,
                                    "patch_score_after": 1.25,
                                    "raw_patch_score_after": 2.5,
                                    "owned_target_count_after": 4,
                                    "target_count": 5,
                                    "phase": "prefill",
                                    "outcome": "applied",
                                    "done": False,
                                    "n_local_actions": 2,
                                    "n_noop_actions": 0,
                                    "actions": [
                                        {
                                            "type": "add",
                                            "cell_id": "A",
                                            "old_cell_id": None,
                                            "barcode": "b01",
                                            "applied": True,
                                        }
                                    ],
                                }
                            ],
                            "bin_geometry": {
                                "b22": {
                                    "array_row": 2,
                                    "array_col": 2,
                                    "x_um": 4.0,
                                    "y_um": 4.0,
                                }
                            },
                        },
                    }
                },
                nucleus_centers={"p1": {"A": [0.0, 0.0], "B": [2.0, 0.0]}},
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["n_patches"], 1)
            patch_path = output_dir / manifest["patches"][0]["file"]
            payload = json.loads(patch_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["counts"]["predicted_bins"], 5)
            self.assertEqual(payload["counts"]["gt_bins"], 5)
            self.assertEqual(payload["counts"]["trace_only_bins"], 1)
            self.assertTrue(payload["trajectory"]["available"])
            self.assertEqual(payload["trajectory"]["capture_status"], "exact")
            self.assertEqual(payload["trajectory"]["steps"][0]["actions"][0]["barcode"], "b01")
            self.assertAlmostEqual(payload["metrics"]["foreground_iou"], 4.0 / 6.0)
            self.assertAlmostEqual(payload["metrics"]["owner_accuracy"], 3.0 / 4.0)
            self.assertAlmostEqual(payload["metrics"]["owner_micro_iou"], 3.0 / 7.0)
            self.assertEqual(payload["metrics"]["correct_owner_bins"], 3)
            self.assertEqual(payload["metrics"]["wrong_owner_bins"], 1)
            self.assertEqual(payload["metrics"]["pred_only_bins"], 1)
            self.assertEqual(payload["metrics"]["gt_only_bins"], 1)

            categories = {row["barcode"]: row["overlap_category"] for row in payload["bins"]}
            self.assertEqual(categories["b20"], "wrong_owner")
            self.assertEqual(categories["b11"], "pred_only")
            self.assertEqual(categories["b12"], "gt_only")
            self.assertEqual(categories["b22"], "unscored")
            self.assertTrue(next(row for row in payload["bins"] if row["barcode"] == "b22")["trace_only"])

            cells = {row["cell_id"]: row for row in payload["cells"]}
            self.assertAlmostEqual(cells["A"]["patch_iou"], 2.0 / 3.0)
            self.assertAlmostEqual(cells["B"]["patch_iou"], 1.0 / 4.0)
            self.assertEqual(cells["A"]["nucleus_center_xy_um"], [0.0, 0.0])


def _assignment(patch_id: str, cell_id: str, barcode: str, array_row: int, array_col: int) -> dict[str, object]:
    return {
        "patch_id": patch_id,
        "cell_id": cell_id,
        "barcode": barcode,
        "array_row": array_row,
        "array_col": array_col,
        "x_um": float(array_col * 2),
        "y_um": float(array_row * 2),
    }


def _gt(cell_id: str, barcode: str, array_row: int, array_col: int) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "cell_type": f"type-{cell_id}",
        "barcode": barcode,
        "array_row": array_row,
        "array_col": array_col,
        "x_um": float(array_col * 2),
        "y_um": float(array_row * 2),
        "is_nuclear": array_col == 0,
        "weight": 1.0,
    }


if __name__ == "__main__":
    unittest.main()
