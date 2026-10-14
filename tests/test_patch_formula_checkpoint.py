from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.patch_formula_checkpoint import (
    load_formula_checkpoint,
    remove_formula_checkpoint,
    save_formula_checkpoint,
)


class PatchFormulaCheckpointTests(unittest.TestCase):
    def test_round_trip_and_cleanup(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as raw_parent:
            directory = Path(raw_parent) / "resume"
            formula = {"name": "w5", "w5": 0.15, "shape_weight": 0.006}
            save_formula_checkpoint(
                directory=directory,
                split="validation",
                formula=formula,
                patch_ids=["p0"],
                patch_rows=[{"formula": "w5", "patch_id": "p0", "n_actions": 1}],
                action_rows=[{"formula": "w5", "patch_id": "p0", "step_index": 1}],
                cell_rows=[{"formula": "w5", "patch_id": "p0", "iou": 0.75}],
            )
            restored = load_formula_checkpoint(
                directory=directory,
                split="validation",
                formula=formula,
                patch_ids=["p0"],
            )
            self.assertIsNotNone(restored)
            assert restored is not None
            patch_frame, action_frame, cell_frame = restored
            self.assertEqual(len(patch_frame), 1)
            self.assertEqual(len(action_frame), 1)
            self.assertEqual(len(cell_frame), 1)
            remove_formula_checkpoint(directory)
            self.assertFalse(directory.exists())

    def test_rejects_non_tmp_directory(self) -> None:
        with self.assertRaises(ValueError):
            load_formula_checkpoint(
                directory=Path("/projects/not-allowed"),
                split="validation",
                formula={"name": "w5"},
                patch_ids=["p0"],
            )


if __name__ == "__main__":
    unittest.main()
