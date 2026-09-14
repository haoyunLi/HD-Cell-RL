from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from scripts.prepare_em_capture_test_context import prepare_context


class PrepareEmCaptureTestContextTests(unittest.TestCase):
    def test_reference_override_is_written_to_episode_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patches = root / "patches.csv"
            pd.DataFrame({"patch_cell_ids": ["[10, 11]"]}).to_csv(
                patches, index=False
            )
            nuclei = root / "nuclei.parquet"
            pd.DataFrame({"cell_id": [10, 11], "x": [0, 1]}).to_parquet(
                nuclei, index=False
            )
            template = root / "template.yaml"
            template.write_text(
                yaml.safe_dump(
                    {
                        "run": {"name": "old", "output_root": "old"},
                        "inputs": {
                            "nuclei_path": "old",
                            "expression": {
                                "matrix_path": "old",
                                "reference_npz_path": "old",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            matrix = root / "matrix"
            matrix.mkdir()
            reference = root / "reference.npz"
            reference.write_bytes(b"fixture")
            output = root / "prepared"
            summary = prepare_context(
                argparse.Namespace(
                    patch_index=str(patches),
                    full_nuclei_path=str(nuclei),
                    episode_config_template=str(template),
                    capture_matrix_path=str(matrix),
                    reference_npz_path=str(reference),
                    episode_output_root=str(root / "episodes"),
                    output_dir=str(output),
                    run_name="new",
                    n_workers=2,
                )
            )
            config = yaml.safe_load(
                (output / "episode_build.capture_thinned.yaml").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                config["inputs"]["expression"]["reference_npz_path"],
                str(reference.resolve()),
            )
            self.assertEqual(summary["reference_npz_path"], str(reference.resolve()))


if __name__ == "__main__":
    unittest.main()
