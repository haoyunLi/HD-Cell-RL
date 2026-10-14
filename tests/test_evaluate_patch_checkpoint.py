from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_patch_checkpoint import _package_patch_eval_source


class EvaluatePatchCheckpointTests(unittest.TestCase):
    def test_package_patch_eval_source_rebases_generated_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            out_dir = root / "intermediate" / "patch_eval_001"
            out_dir.mkdir(parents=True)
            for name in (
                "patch_rl_assignments.csv",
                "patch_rl_patch_assignments.csv",
                "patch_rl_merge_candidates.csv",
            ):
                (out_dir / name).write_text("cell_id\n1\n", encoding="utf-8")
            sample_dir = out_dir / "ppo_eval_source_patch_sample"
            sample_dir.mkdir()
            (sample_dir / "per_episode.csv").write_text("cell_id\n1\n", encoding="utf-8")

            eval_run = root / "runs" / "evaluation_001"
            eval_run.mkdir(parents=True)
            old_prefix = str(out_dir.resolve())
            (eval_run / "summary.json").write_text(
                json.dumps({"assignment": f"{old_prefix}/patch_rl_assignments.csv"}),
                encoding="utf-8",
            )
            (eval_run / "config_used.yaml").write_text(
                f"assignment: {old_prefix}/patch_rl_assignments.csv\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(ppo_eval_run_dir=str(sample_dir))
            pipeline = {
                "assignments_csv": str(out_dir / "patch_rl_assignments.csv"),
                "patch_assignments_csv": str(out_dir / "patch_rl_patch_assignments.csv"),
                "merge_candidates_csv": str(out_dir / "patch_rl_merge_candidates.csv"),
            }

            packaged, assignments, patch_assignments, merge = _package_patch_eval_source(
                out_dir=out_dir,
                eval_run=eval_run,
                args=args,
                pipeline=pipeline,
            )

            self.assertFalse(out_dir.exists())
            self.assertEqual(packaged, eval_run / "source" / "patch_eval_001")
            for path in (assignments, patch_assignments, merge):
                self.assertTrue(path.exists())
            self.assertEqual(
                Path(args.ppo_eval_run_dir),
                packaged / "ppo_eval_source_patch_sample",
            )
            for provenance in (eval_run / "summary.json", eval_run / "config_used.yaml"):
                content = provenance.read_text(encoding="utf-8")
                self.assertNotIn(old_prefix, content)
                self.assertIn(str(packaged), content)
            saved_pipeline = json.loads(
                (packaged / "patch_eval_pipeline_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_pipeline["assignments_csv"], str(assignments))


if __name__ == "__main__":
    unittest.main()
