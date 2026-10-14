"""Regression checks for correctness fixes; no training or production data writes."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite
import torch
import yaml

from hd_cell_rl.em_assignment import run_generalized_em
from hd_cell_rl.episode_artifacts import _MatrixOnDemandExpressionLoader
from hd_cell_rl.matrix_io import resolve_matrix_csc_h5_path
from hd_cell_rl.patch_assignment import reconcile_patch_owners
from hd_cell_rl.patch_dataset import PatchDataset
from hd_cell_rl.patch_env_torch import TorchJointPatchEnv
from hd_cell_rl.patch_env_torch import TorchSingleAgentPatchEnv
from hd_cell_rl.patch_types import PatchBounds, PatchContext
from hd_cell_rl.ppo_checkpoint import atomic_save_checkpoint, checkpoint_contract, validate_checkpoint_contract
from hd_cell_rl.ppo_dataset import _load_nuclear_barcode_assignment_lookup, EpisodeDataset
from hd_cell_rl.provenance import file_fingerprint
from preprocessing.ppo_eval_metrics import build_episode_nuclear_barcode_map
from scripts.run_patch_training import _save_patch_checkpoint
from tests.test_em_assignment import _simple_sparse_input, _config, _episode


class EngineeringSafetyTests(unittest.TestCase):
    def test_em_does_not_stop_while_q_is_changing(self):
        data = _simple_sparse_input(ll=np.array([[-.1, -8], [-8, -.1]]),
            confidence=np.ones(2), distances=np.array([[0, 9], [9, 0]]))
        result = run_generalized_em(data, _config(damping=1.0))
        self.assertGreater(len(result.iterations), 1)
        self.assertGreater(result.responsibility[0], .9)
        self.assertTrue(result.converged)
        self.assertLessEqual(result.iterations[-1].max_q_change, .001)

    def test_nuclear_lock_without_em(self):
        bars = ["s_002um_00000_00000-1", "s_002um_00000_00001-1"]
        cells = tuple(_episode(cid, center, bars, [(0, 0), (2, 0)],
            ll=[[-.1, -8], [-8, -.1]], seed=seed, counts=[10, 10])
            for cid, center, seed in [("A", (0, 0), [1, 0]), ("B", (2, 0), [0, 1])])
        context = PatchContext(patch_id="test", cells=cells, core_cell_ids=("A", "B"),
            margin_cell_ids=(), outer_bounds=PatchBounds(-1, 3, -1, 1), core_bounds=PatchBounds(-1, 3, -1, 1),
            max_steps=4, agent_mode="multi_cell_joint_global_delta", after_fill_actions="replace_only")
        env = TorchJointPatchEnv(context=context, device=torch.device("cpu"))
        env.reset()
        self.assertFalse(bool(env._replace_allowed_by_barcode.any().item()))
        env._ctx = replace(context, em_assignment=object())
        env._em_initial_owner_by_barcode[0] = 1
        with self.assertRaisesRegex(ValueError, "confident nuclear seed"):
            env.reset()

    def test_inference_and_evaluation_share_confident_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "bins.parquet"
            pd.DataFrame({"barcode": ["confident", "ambiguous", "outside"],
                "dominant_cell_id": [1, 1, 1], "has_nuclear_annotation": [True, True, False],
                "ambiguous_nuclear_assignment": [False, True, False]}).to_parquet(source)
            self.assertEqual(_load_nuclear_barcode_assignment_lookup(source), {"confident": "1"})
            self.assertEqual(build_episode_nuclear_barcode_map(source, {"1"}), {"1": {"confident"}})

    def test_merge_is_unique_deterministic_and_locks_nuclei(self):
        claims = {"A": (1., [{"barcode": "b", "cell_id": "A", "patch_id": "p1"}]),
                  "B": (2., [{"barcode": "b", "cell_id": "B", "patch_id": "p2"}])}
        output, conflicts = reconcile_patch_owners(claims, nuclear_owners={})
        self.assertEqual([r["cell_id"] for r in output], ["B"])
        self.assertEqual(len(conflicts), 2)
        self.assertEqual(output, reconcile_patch_owners(dict(reversed(list(claims.items()))), nuclear_owners={})[0])
        self.assertEqual(reconcile_patch_owners(claims, nuclear_owners={"b": "A"})[0][0]["cell_id"], "A")
        self.assertEqual(reconcile_patch_owners(claims, nuclear_owners={"b": "C"})[0], [])

    def test_seed_cell_ids_are_normalized_identically(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "bins.parquet"
            pd.DataFrame({"barcode": ["numeric", "named"], "dominant_cell_id": ["1.0", "cell_A"],
                "has_nuclear_annotation": [True, True], "ambiguous_nuclear_assignment": [False, False]}).to_parquet(source)
            self.assertEqual(_load_nuclear_barcode_assignment_lookup(source), {"numeric": "1", "named": "cell_A"})
            self.assertEqual(build_episode_nuclear_barcode_map(source, {"1", "cell_A"}),
                             {"1": {"numeric"}, "cell_A": {"named"}})

    def test_merge_rejects_nonfinite_scores(self):
        with self.assertRaises(ValueError):
            reconcile_patch_owners({"A": (np.nan, [])}, nuclear_owners={})

    def test_directory_ignores_stale_sibling_and_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "matrix"
            directory.mkdir()
            with h5py.File(directory.with_suffix(".h5"), "w") as h5:
                h5.attrs["stale"] = True
            for name, text in [("barcodes.tsv.gz", "b0\nb1\n"), ("features.tsv.gz", "g0\tg0\tGene Expression\ng1\tg1\tGene Expression\n")]:
                with gzip.open(directory / name, "wt") as handle:
                    handle.write(text)
            mmwrite(directory / "matrix.mtx", sparse.coo_matrix([[1, 0], [0, 2]]))
            resolved = resolve_matrix_csc_h5_path(directory)
            self.assertNotEqual(resolved, directory.with_suffix(".h5"))
            self.assertEqual(resolved, resolve_matrix_csc_h5_path(directory))
            mmwrite(directory / "matrix.mtx", sparse.coo_matrix([[3, 0], [0, 4]]))
            with h5py.File(resolve_matrix_csc_h5_path(directory), "r") as h5:
                np.testing.assert_array_equal(h5["matrix/data"][:], [3, 4])

    def test_sparse_loader_matches_dense_and_preserves_order(self):
        loader = object.__new__(_MatrixOnDemandExpressionLoader)
        loader._n_cols, loader._expression_dim = 3, 2
        loader._indptr = np.array([0, 2, 3, 3])
        loader._indices_ds = np.array([0, 2, 1])
        loader._data_ds = np.array([5, 7, 3])
        loader._feature_lookup = np.array([1, 0, -1])
        loader._cache_size, loader._cache = 0, OrderedDict()
        cols = np.array([2, 1, 0, 1])
        dense = loader.load_columns(cols)
        with patch.object(loader, "load_columns", side_effect=AssertionError("must not densify")):
            actual = loader.load_sparse_columns(cols)
        self.assertTrue(sparse.isspmatrix_csr(actual))
        np.testing.assert_array_equal(actual.toarray(), dense)
        self.assertEqual(loader.load_sparse_columns(np.array([], dtype=int)).shape, (0, 2))
        with self.assertRaises(ValueError):
            loader.load_sparse_columns(np.array([3]))

    def test_sparse_artifact_load_selects_only_requested_barcodes(self):
        dataset = object.__new__(EpisodeDataset)
        dataset._expression_loader = SimpleNamespace(load_sparse_columns=lambda cols: sparse.csr_matrix(cols[:, None]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.npz"
            np.savez(path, candidate_bin_ids=["b0", "b1"], candidate_bin_xy_um=[[0, 0], [1, 0]],
                nucleus_center_xy_um=[0, 0], candidate_matrix_col_index=[3, 9])
            ids, counts = dataset.load_episode_sparse_expression(cell_id="A", artifact_path=path, wanted_barcodes={"b1"})
            self.assertEqual(ids, ("b1",))
            self.assertEqual(counts[0, 0], 9)

    def test_patch_cache_has_lru_bound(self):
        dataset = object.__new__(PatchDataset)
        dataset._settings = SimpleNamespace(cache_patch_contexts=True, patch_context_cache_size=2)
        dataset._context_cache = OrderedDict()
        for key in ["a", "b", "a", "c"]:
            dataset._cache_context(key, None)
        self.assertEqual(list(dataset._context_cache), ["a", "c"])

    def test_new_checkpoint_contract_rejects_same_shape_different_mode(self):
        contract = checkpoint_contract(training_unit="patch", agent_mode="multi_cell_joint_global_delta")
        validate_checkpoint_contract({"feature_contract": contract}, expected=contract)
        with self.assertRaises(ValueError):
            validate_checkpoint_contract({"feature_contract": contract}, expected=checkpoint_contract())
        with self.assertWarns(RuntimeWarning):
            validate_checkpoint_contract({}, expected=contract)
        with self.assertWarns(RuntimeWarning):
            validate_checkpoint_contract({"feature_contract": contract}, expected=checkpoint_contract(), allow_semantic_transfer=True)

    def test_normalization_aliases_preserve_global_objective_scale(self):
        env = object.__new__(TorchSingleAgentPatchEnv)
        for alias, legacy, expected in [("mean_patch_cells", "mean_core_cells", 4.),
                                        ("sqrt_patch_cells", "sqrt_core_cells", 2.),
                                        ("sum_patch_cells", "sum_core_cells", 1.)]:
            env._ctx = SimpleNamespace(cells=(1, 2, 3, 4), score_normalization=alias)
            self.assertEqual(env._score_denominator(), expected)
            env._ctx.score_normalization = legacy
            self.assertEqual(env._score_denominator(), expected)

    def test_failed_atomic_save_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            atomic_save_checkpoint({"value": 1}, path)
            with patch("hd_cell_rl.ppo_checkpoint.torch.save", side_effect=RuntimeError("failed")):
                with self.assertRaises(RuntimeError):
                    atomic_save_checkpoint({"value": 2}, path)
            self.assertEqual(torch.load(path, weights_only=True)["value"], 1)
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_checkpoint_saves_evaluated_snapshot_not_updated_weights(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters())
        snapshot = {key: value.detach().clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            model.weight.add_(1.)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            _save_patch_checkpoint(path=path, model=model, optimizer=optimizer, update_index=0,
                best_moving_avg_reward=1., base_config=SimpleNamespace(to_serializable_dict=lambda: {}),
                patch_config={}, model_state=snapshot, selection_basis="training_rollout_moving_average_pre_update")
            saved = torch.load(path, weights_only=True)
            torch.testing.assert_close(saved["model_state_dict"]["weight"], snapshot["weight"])
            self.assertFalse(torch.equal(saved["model_state_dict"]["weight"], model.weight))
            self.assertIn("feature_contract", saved)

    def test_fingerprint_detects_same_size_content_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.txt"
            path.write_text("one")
            first = file_fingerprint(path)
            path.write_text("two")
            self.assertNotEqual(first["sha256"], file_fingerprint(path)["sha256"])

    def test_patch_runner_keeps_best_and_final_in_separate_new_run(self):
        from tests.test_ppo_training import PPOTrainingTests
        import scripts.run_patch_training as runner
        cell = _episode("A", (0, 0), ["seed", "free"], [(0, 0), (2, 0)],
                        ll=[[-.1, -2.], [-.2, -1.]], seed=[1, 0])
        context = PatchContext(patch_id="synthetic", cells=(cell,), core_cell_ids=("A",),
            margin_cell_ids=(), outer_bounds=PatchBounds(-1, 3, -1, 1), core_bounds=PatchBounds(-1, 3, -1, 1), max_steps=2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_path = PPOTrainingTests()._write_minimal_config(root, batch_cells=1, group_relative_enabled=False)
            previous = root / "runs" / "previous_final"
            previous.mkdir(parents=True)
            marker = previous / "evolution.txt"
            marker.write_text("keep previous final")
            cfg = {"base_ppo_config": str(base_path),
                   "run": {"name": "engineering_fixture", "device": "cpu", "max_updates": 1, "batch_patches": 1},
                   "patch_training": {"patches_index_path": str(root / "patches.csv"), "rollout_backend": "legacy_cpu"},
                   "base_config_overrides": {"stopping": {"moving_avg_window": 1}}}
            config_path = root / "patch.yaml"
            config_path.write_text(yaml.safe_dump(cfg))
            with patch.object(runner, "_parse_args", return_value=SimpleNamespace(config=str(config_path))), \
                 patch.object(runner, "PatchDataset", return_value=SimpleNamespace(n_patches=1, close=lambda: None)), \
                 patch.object(runner, "collect_patch_contexts", return_value=[context]):
                runner.main()
            runs = [path for path in (root / "runs").iterdir() if path != previous]
            self.assertEqual(len(runs), 1)
            best = torch.load(runs[0] / "checkpoints" / "best_model.pt", weights_only=True)
            final = torch.load(runs[0] / "checkpoints" / "final_model.pt", weights_only=True)
            self.assertEqual(best["update_index"], 0)
            self.assertEqual(final["update_index"], 1)
            self.assertEqual(best["selection_basis"], "training_rollout_moving_average_pre_update")
            self.assertEqual(final["selection_basis"], "final_post_update")
            self.assertEqual(marker.read_text(), "keep previous final")
            metadata = json.loads((runs[0] / "metadata.json").read_text())
            self.assertIn("input_fingerprints", metadata)
            self.assertIn("source_sha256", metadata["provenance"])


if __name__ == "__main__":
    unittest.main()
