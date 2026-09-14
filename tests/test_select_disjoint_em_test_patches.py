from __future__ import annotations

import json
import unittest

import pandas as pd

from scripts.select_disjoint_em_test_patches import select_disjoint_test_patches


def _row(
    patch_id: str,
    *,
    x: float,
    cells: list[str],
    core: list[str],
    source: str = "grid0",
) -> dict[str, object]:
    return {
        "patch_id": patch_id,
        "source": source,
        "context_x_min": x,
        "context_x_max": x + 10.0,
        "context_y_min": 0.0,
        "context_y_max": 10.0,
        "candidate_max_distance_um": 20.0,
        "n_patch_cells": len(cells),
        "n_core_cells": len(core),
        "patch_cell_ids": json.dumps(cells),
        "core_cell_ids": json.dumps(core),
    }


class SelectDisjointEMTestPatchesTests(unittest.TestCase):
    def test_matches_metadata_and_rejects_spatial_or_cell_overlap(self) -> None:
        reference = pd.DataFrame(
            [_row("reference", x=0.0, cells=["a", "b"], core=["a"])]
        )
        full = pd.DataFrame(
            [
                _row("reference", x=0.0, cells=["a", "b"], core=["a"]),
                _row("same_cell", x=30.0, cells=["a", "c"], core=["c"]),
                _row("same_space", x=5.0, cells=["d", "e"], core=["d"]),
                _row("eligible", x=30.0, cells=["f", "g"], core=["f"]),
            ]
        )
        selected, audit = select_disjoint_test_patches(
            full=full,
            reference=reference,
            seed=17,
        )
        self.assertEqual(selected["patch_id"].tolist(), ["eligible"])
        self.assertEqual(audit["reference_patch_id"].tolist(), ["reference"])
        self.assertEqual(audit["n_core_cells"].tolist(), [1])
        self.assertEqual(audit["n_patch_cells"].tolist(), [2])

    def test_excluded_indices_contribute_cells_and_rectangles(self) -> None:
        reference = pd.DataFrame(
            [_row("reference", x=0.0, cells=["a", "b"], core=["a"])]
        )
        excluded = pd.DataFrame(
            [_row("previous_split", x=30.0, cells=["f", "g"], core=["f"])]
        )
        full = pd.DataFrame(
            [
                *reference.to_dict("records"),
                *excluded.to_dict("records"),
                _row("overlap_excluded_cell", x=60.0, cells=["f", "h"], core=["h"]),
                _row("overlap_excluded_space", x=35.0, cells=["i", "j"], core=["i"]),
                _row("eligible", x=60.0, cells=["k", "l"], core=["k"]),
            ]
        )
        selected, _ = select_disjoint_test_patches(
            full=full,
            reference=reference,
            seed=17,
            excluded=[excluded],
        )
        self.assertEqual(selected["patch_id"].tolist(), ["eligible"])


if __name__ == "__main__":
    unittest.main()
