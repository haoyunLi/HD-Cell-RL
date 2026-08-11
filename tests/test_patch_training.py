from __future__ import annotations

from dataclasses import replace
import importlib
import importlib.util
import sys
import types
import unittest

import numpy as np

_PATCH_MODULES = (
    "hd_cell_rl.patch_training",
    "hd_cell_rl.ppo_dataset",
    "hd_cell_rl.ppo_buffers",
    "hd_cell_rl.ppo_state",
    "hd_cell_rl.shape_prior",
    "hd_cell_rl.stcs_reward",
)


def _load_patch_modules():
    previous_modules = {name: sys.modules.get(name) for name in (*_PATCH_MODULES, "torch")}
    present_modules = {name: name in sys.modules for name in (*_PATCH_MODULES, "torch")}
    if not present_modules["torch"] and importlib.util.find_spec("torch") is None:
        fake_torch = types.ModuleType("torch")
        fake_torch.__patch_training_test_fake__ = True
        sys.modules["torch"] = fake_torch
    patch_training = importlib.import_module("hd_cell_rl.patch_training")
    ppo_state = importlib.import_module("hd_cell_rl.ppo_state")
    shape_prior = importlib.import_module("hd_cell_rl.shape_prior")
    return patch_training, ppo_state, shape_prior, previous_modules, present_modules


def _cleanup_patch_modules(previous_modules: dict[str, object | None], present_modules: dict[str, bool]) -> None:
    for name in (*_PATCH_MODULES, "torch"):
        if name == "torch" and not present_modules[name] and previous_modules[name] is None:
            current_torch = sys.modules.get("torch")
            if current_torch is not None and not getattr(current_torch, "__patch_training_test_fake__", False):
                continue
        if present_modules[name]:
            sys.modules[name] = previous_modules[name]
        else:
            sys.modules.pop(name, None)


def _ctx(episode_context_cls, cell_id: str, bin_ids: list[str], xy: list[tuple[float, float]], seed: list[int]):
    n = len(bin_ids)
    neighbor_index = np.full((n, 8), -1, dtype=np.int32)
    for i in range(n - 1):
        neighbor_index[i, 0] = i + 1
        neighbor_index[i + 1, 0] = i
    return episode_context_cls(
        cell_id=cell_id,
        candidate_bin_ids=tuple(bin_ids),
        initial_membership_mask=np.asarray(seed, dtype=np.uint8),
        candidate_bin_xy_um=np.asarray(xy, dtype=np.float32),
        nucleus_center_xy_um=np.asarray(xy[0], dtype=np.float32),
        ll=np.zeros((n, 2), dtype=np.float32),
        p_dis=np.zeros(n, dtype=np.float32),
        p_overlap=np.zeros(n, dtype=np.float32),
        ll_mean_z=np.zeros(n, dtype=np.float32),
        ll_max_z=np.zeros(n, dtype=np.float32),
        base_penalty=np.zeros(n, dtype=np.float32),
        expression_confidence=np.ones(n, dtype=np.float32),
        bin_count_totals=np.ones(n, dtype=np.float32),
        neighbor_index=neighbor_index,
        max_steps=10,
        log_prior=0.0,
        r_max_um=20.0,
        w1=0.0,
        w2=0.0,
        w3=0.0,
        w4=1.0,
        w5=0.0,
        stop_lambda=0.0,
        stop_stat="max",
        stop_top_k=1,
        expression_confidence_pseudocount=5.0,
        normalize_expression_zscore=True,
        zscore_delta=1.0e-8,
    )


def _obs_array(value, dtype):
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


class PatchTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        (
            self.patch_training,
            self.ppo_state,
            self.shape_prior,
            self._previous_modules,
            self._present_modules,
        ) = _load_patch_modules()

    def tearDown(self) -> None:
        _cleanup_patch_modules(self._previous_modules, self._present_modules)

    def test_patch_env_uses_unique_owner_for_shared_seed(self) -> None:
        a = _ctx(self.ppo_state.EpisodeContext, "a", ["shared", "a2"], [(0, 0), (2, 0)], [1, 0])
        b = _ctx(self.ppo_state.EpisodeContext, "b", ["shared", "b2"], [(10, 0), (8, 0)], [1, 0])
        patch = self.patch_training.PatchContext(
            patch_id="p0",
            cells=(a, b),
            core_cell_ids=("a",),
            margin_cell_ids=("b",),
            outer_bounds=self.patch_training.PatchBounds(-5, 15, -5, 5),
            core_bounds=self.patch_training.PatchBounds(-5, 5, -5, 5),
            max_steps=5,
        )
        env = self.patch_training.MultiCellPatchEnv(patch)
        obs, _ = env.reset()
        self.assertEqual(obs["action_features"].shape[1], self.patch_training.ACTION_FEATURE_DIM)
        masks = env.final_masks()
        self.assertEqual(int(masks["a"][0] + masks["b"][0]), 1)

    def test_patch_env_blocks_owned_barcode_actions(self) -> None:
        a = _ctx(self.ppo_state.EpisodeContext, "a", ["a_seed", "shared"], [(0, 0), (2, 0)], [1, 0])
        b = _ctx(self.ppo_state.EpisodeContext, "b", ["b_seed", "shared"], [(4, 0), (2, 0)], [1, 0])
        patch = self.patch_training.PatchContext(
            patch_id="p1",
            cells=(a, b),
            core_cell_ids=("a", "b"),
            margin_cell_ids=(),
            outer_bounds=self.patch_training.PatchBounds(-5, 10, -5, 5),
            core_bounds=self.patch_training.PatchBounds(-5, 10, -5, 5),
            max_steps=5,
        )
        env = self.patch_training.MultiCellPatchEnv(patch)
        obs, _ = env.reset()
        self.assertGreaterEqual(obs["action_features"].shape[0], 2)
        env.step(1)
        next_obs, *_ = env.step(0)
        del next_obs
        owners = [cell for cell, mask in env.final_masks().items() if mask.sum() > 1]
        self.assertEqual(len(owners), 1)

    def test_batched_patch_rollout_collects_multiple_envs(self) -> None:
        import torch
        if not hasattr(torch, "nn") or not hasattr(torch, "distributions"):
            self.skipTest("torch is required for batched rollout")

        class StopModel(torch.nn.Module):
            def forward(self, global_features, action_features, action_mask):
                logits = torch.full(action_mask.shape, -10.0, device=action_mask.device)
                logits[:, 0] = 10.0
                return torch.distributions.Categorical(logits=logits), torch.zeros(
                    global_features.shape[0], device=global_features.device
                )

        ctx = _ctx(self.ppo_state.EpisodeContext, "a", ["a_seed", "a2"], [(0, 0), (2, 0)], [1, 0])
        patch = self.patch_training.PatchContext(
            patch_id="p_batched",
            cells=(ctx,),
            core_cell_ids=("a",),
            margin_cell_ids=(),
            outer_bounds=self.patch_training.PatchBounds(-5, 5, -5, 5),
            core_bounds=self.patch_training.PatchBounds(-5, 5, -5, 5),
            max_steps=5,
        )

        for backend in ("legacy_cpu", "cached_cpu"):
            trajectories, timing = self.patch_training.collect_patch_trajectories_batched(
                contexts=[patch, patch],
                model=StopModel(),
                device=torch.device("cpu"),
                rng=np.random.default_rng(7),
                policy_mode="greedy",
                rollout_backend=backend,
            )

            self.assertEqual(len(trajectories), 2)
            self.assertTrue(all(len(traj.steps) == 1 for traj in trajectories))
            self.assertEqual(float(timing["rollout_n_model_calls"]), 1.0)

    def test_cached_patch_env_matches_legacy_cpu(self) -> None:
        a = _ctx(self.ppo_state.EpisodeContext, "a", ["a_seed", "a2", "a3"], [(0, 0), (2, 0), (4, 0)], [1, 0, 0])
        neighbor_index = np.full((3, 8), -1, dtype=np.int32)
        neighbor_index[0, 0] = 1
        neighbor_index[1, 0:2] = [0, 2]
        neighbor_index[2, 0] = 1
        a = replace(a, neighbor_index=neighbor_index)
        patch = self.patch_training.PatchContext(
            patch_id="p_cached",
            cells=(a,),
            core_cell_ids=("a",),
            margin_cell_ids=(),
            outer_bounds=self.patch_training.PatchBounds(-5, 6, -5, 5),
            core_bounds=self.patch_training.PatchBounds(-5, 6, -5, 5),
            max_steps=5,
        )
        legacy = self.patch_training.MultiCellPatchEnv(patch)
        cached = self.patch_training.CachedMultiCellPatchEnv(patch)
        legacy_obs, _ = legacy.reset()
        cached_obs, _ = cached.reset()
        np.testing.assert_allclose(cached_obs["global_features"], legacy_obs["global_features"], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(cached_obs["action_features"], legacy_obs["action_features"], rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(cached_obs["action_mask"], legacy_obs["action_mask"])

        legacy_next, legacy_reward, *_ = legacy.step(1)
        cached_next, cached_reward, *_ = cached.step(1)
        self.assertEqual(float(cached_reward), float(legacy_reward))
        np.testing.assert_allclose(cached_next["global_features"], legacy_next["global_features"], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(cached_next["action_features"], legacy_next["action_features"], rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(cached.final_masks()["a"], legacy.final_masks()["a"])

    def test_forced_fill_masks_stop_until_expression_target_is_filled(self) -> None:
        ctx = _ctx(
            self.ppo_state.EpisodeContext,
            "a",
            ["seed", "a2", "a3"],
            [(0, 0), (2, 0), (4, 0)],
            [1, 0, 0],
        )
        neighbor_index = np.full((3, 8), -1, dtype=np.int32)
        neighbor_index[0, 0] = 1
        neighbor_index[1, 0:2] = [0, 2]
        neighbor_index[2, 0] = 1
        ctx = replace(ctx, neighbor_index=neighbor_index)
        patch = self.patch_training.PatchContext(
            patch_id="p_force_fill",
            cells=(ctx,),
            core_cell_ids=("a",),
            margin_cell_ids=(),
            outer_bounds=self.patch_training.PatchBounds(-5, 6, -5, 5),
            core_bounds=self.patch_training.PatchBounds(-5, 6, -5, 5),
            max_steps=5,
            score_normalization="mean_expression_bins",
            force_fill_expression_bins=True,
            stop_action_mode="mask_until_filled",
            force_fill_target_barcodes=("seed", "a2", "a3"),
        )

        env_cases = [
            (self.patch_training.MultiCellPatchEnv, {}),
            (self.patch_training.CachedMultiCellPatchEnv, {}),
        ]
        import torch
        if hasattr(torch, "device"):
            env_cases.append((self.patch_training.TorchPatchEnv, {"device": torch.device("cpu")}))

        for env_cls, kwargs in env_cases:
            with self.subTest(env=env_cls.__name__):
                env = env_cls(patch, **kwargs)
                obs, info = env.reset()
                self.assertEqual(int(info["n_force_fill_owned_expression_bins"]), 1)
                self.assertFalse(bool(_obs_array(obs["action_mask"], bool)[0]))
                with self.assertRaises(ValueError):
                    env.step(0)

                obs, _, terminated, truncated, info = env.step(1)
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                self.assertEqual(int(info["n_force_fill_owned_expression_bins"]), 2)
                self.assertFalse(bool(_obs_array(obs["action_mask"], bool)[0]))
                self.assertAlmostEqual(
                    float(info["patch_score"]),
                    float(info["raw_total_core_reward"]) / 3.0,
                    places=6,
                )

                _, _, terminated, truncated, info = env.step(1)
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertEqual(int(info["n_force_fill_owned_expression_bins"]), 3)
                self.assertAlmostEqual(
                    float(info["patch_score"]),
                    float(info["raw_total_core_reward"]) / 3.0,
                    places=6,
                )

    def test_single_agent_global_delta_add_fills_target_and_auto_stops(self) -> None:
        import torch
        if not hasattr(torch, "as_tensor"):
            self.skipTest("torch is required for single-agent patch env")

        ctx = _ctx(self.ppo_state.EpisodeContext, "a", ["seed", "a2"], [(0, 0), (2, 0)], [1, 0])
        patch = self.patch_training.PatchContext(
            patch_id="p_single_add",
            cells=(ctx,),
            core_cell_ids=("a",),
            margin_cell_ids=(),
            outer_bounds=self.patch_training.PatchBounds(-5, 5, -5, 5),
            core_bounds=self.patch_training.PatchBounds(-5, 5, -5, 5),
            max_steps=5,
            score_normalization="mean_expression_bins",
            force_fill_expression_bins=True,
            force_fill_target_barcodes=("seed", "a2"),
            agent_mode="single_cell_global_delta",
            after_fill_actions="replace_only",
            stop_action_mode="auto_no_improve_after_filled",
        )
        env = self.patch_training.TorchSingleAgentPatchEnv(patch, device=torch.device("cpu"))
        obs, _ = env.reset()
        mask = _obs_array(obs["action_mask"], bool)
        self.assertFalse(bool(mask[0]))
        self.assertEqual(int(mask.sum()), 1)

        _next_obs, reward, terminated, truncated, info = env.step(1)
        self.assertGreater(float(reward), 0.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(int(info["n_force_fill_owned_expression_bins"]), 2)
        self.assertEqual(int(env.final_masks()["a"].sum()), 2)

    def test_single_agent_global_delta_replace_updates_owner_and_auto_stops(self) -> None:
        import torch
        if not hasattr(torch, "as_tensor"):
            self.skipTest("torch is required for single-agent patch env")

        a = _ctx(self.ppo_state.EpisodeContext, "a", ["a_seed", "shared"], [(0, 0), (2, 0)], [1, 0])
        b = _ctx(self.ppo_state.EpisodeContext, "b", ["b_seed", "shared"], [(10, 0), (2, 0)], [1, 1])
        a = replace(a, w4=0.0, base_penalty=np.asarray([0.0, 0.0], dtype=np.float32))
        b = replace(b, w4=0.0, base_penalty=np.asarray([0.0, 10.0], dtype=np.float32))
        patch = self.patch_training.PatchContext(
            patch_id="p_single_replace",
            cells=(a, b),
            core_cell_ids=("a", "b"),
            margin_cell_ids=(),
            outer_bounds=self.patch_training.PatchBounds(-5, 12, -5, 5),
            core_bounds=self.patch_training.PatchBounds(-5, 12, -5, 5),
            max_steps=5,
            score_normalization="mean_expression_bins",
            force_fill_expression_bins=True,
            force_fill_target_barcodes=("a_seed", "b_seed", "shared"),
            agent_mode="single_cell_global_delta",
            after_fill_actions="replace_only",
            stop_action_mode="auto_no_improve_after_filled",
        )
        env = self.patch_training.TorchSingleAgentPatchEnv(patch, device=torch.device("cpu"))
        obs, _ = env.reset()
        features = _obs_array(obs["action_features"], np.float32)
        mask = _obs_array(obs["action_mask"], bool)
        self.assertFalse(bool(mask[0]))
        self.assertEqual(int(mask.sum()), 1)
        self.assertEqual(float(features[1, self.patch_training.A_FEATURE_5]), 1.0)
        self.assertGreater(float(features[1, self.patch_training.A_COMPETITION_MARGIN]), 0.0)

        _next_obs, reward, terminated, truncated, _info = env.step(1)
        self.assertGreater(float(reward), 0.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        masks = env.final_masks()
        self.assertEqual(int(masks["a"][1]), 1)
        self.assertEqual(int(masks["b"][1]), 0)

    def test_multi_cell_global_delta_prefill_exposes_target_add_and_replace(self) -> None:
        import torch
        if not hasattr(torch, "as_tensor"):
            self.skipTest("torch is required for multi-cell global-delta patch env")

        a = _ctx(
            self.ppo_state.EpisodeContext,
            "a",
            ["a_seed", "shared"],
            [(0, 0), (2, 0)],
            [1, 0],
        )
        b = _ctx(
            self.ppo_state.EpisodeContext,
            "b",
            ["b_seed", "shared"],
            [(10, 0), (2, 0)],
            [1, 1],
        )
        c = _ctx(
            self.ppo_state.EpisodeContext,
            "c",
            ["c_seed", "c2"],
            [(20, 0), (22, 0)],
            [1, 0],
        )
        a = replace(a, w4=0.0, base_penalty=np.asarray([0.0, 0.0], dtype=np.float32))
        b = replace(b, w4=0.0, base_penalty=np.asarray([0.0, 10.0], dtype=np.float32))
        c = replace(c, w4=0.0, base_penalty=np.asarray([0.0, 0.0], dtype=np.float32))
        patch = self.patch_training.PatchContext(
            patch_id="p_multi_prefill",
            cells=(a, b, c),
            core_cell_ids=("a", "b", "c"),
            margin_cell_ids=(),
            outer_bounds=self.patch_training.PatchBounds(-5, 24, -5, 5),
            core_bounds=self.patch_training.PatchBounds(-5, 24, -5, 5),
            max_steps=5,
            score_normalization="mean_expression_bins",
            force_fill_expression_bins=True,
            force_fill_target_barcodes=("a_seed", "b_seed", "shared", "c_seed", "c2"),
            agent_mode="multi_cell_global_delta",
            after_fill_actions="replace_only",
            stop_action_mode="auto_no_improve_after_filled",
        )
        env = self.patch_training.TorchSingleAgentPatchEnv(patch, device=torch.device("cpu"))
        obs, _ = env.reset()
        features = _obs_array(obs["action_features"], np.float32)
        mask = _obs_array(obs["action_mask"], bool)
        self.assertFalse(bool(mask[0]))
        self.assertGreaterEqual(int(mask.sum()), 2)
        self.assertIn(0.0, set(features[1:, self.patch_training.A_FEATURE_5].tolist()))
        self.assertIn(1.0, set(features[1:, self.patch_training.A_FEATURE_5].tolist()))

    def test_multi_cell_global_delta_postfill_only_exposes_positive_replace(self) -> None:
        import torch
        if not hasattr(torch, "as_tensor"):
            self.skipTest("torch is required for multi-cell global-delta patch env")

        a = _ctx(self.ppo_state.EpisodeContext, "a", ["a_seed", "shared"], [(0, 0), (2, 0)], [1, 0])
        b = _ctx(self.ppo_state.EpisodeContext, "b", ["b_seed", "shared"], [(10, 0), (2, 0)], [1, 1])
        a = replace(a, w4=0.0, base_penalty=np.asarray([0.0, 0.0], dtype=np.float32))
        b = replace(b, w4=0.0, base_penalty=np.asarray([0.0, 10.0], dtype=np.float32))
        patch = self.patch_training.PatchContext(
            patch_id="p_multi_postfill",
            cells=(a, b),
            core_cell_ids=("a", "b"),
            margin_cell_ids=(),
            outer_bounds=self.patch_training.PatchBounds(-5, 12, -5, 5),
            core_bounds=self.patch_training.PatchBounds(-5, 12, -5, 5),
            max_steps=5,
            score_normalization="mean_expression_bins",
            force_fill_expression_bins=True,
            force_fill_target_barcodes=("a_seed", "b_seed", "shared"),
            agent_mode="multi_cell_global_delta",
            after_fill_actions="replace_only",
            stop_action_mode="auto_no_improve_after_filled",
        )
        env = self.patch_training.TorchSingleAgentPatchEnv(patch, device=torch.device("cpu"))
        obs, _ = env.reset()
        features = _obs_array(obs["action_features"], np.float32)
        self.assertTrue(np.all(features[1:, self.patch_training.A_FEATURE_5] == 1.0))
        _next_obs, reward, terminated, truncated, _info = env.step(1)
        self.assertGreater(float(reward), 0.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        masks = env.final_masks()
        self.assertEqual(int(masks["a"][1]), 1)
        self.assertEqual(int(masks["b"][1]), 0)

    def test_joint_global_delta_prefill_adds_multiple_targets_in_one_macro_step(self) -> None:
        import torch
        if not hasattr(torch, "as_tensor"):
            self.skipTest("torch is required for joint patch env")

        a = _ctx(self.ppo_state.EpisodeContext, "a", ["a_seed", "a2"], [(0, 0), (2, 0)], [1, 0])
        b = _ctx(self.ppo_state.EpisodeContext, "b", ["b_seed", "b2"], [(10, 0), (12, 0)], [1, 0])
        patch = self.patch_training.PatchContext(
            patch_id="p_joint_prefill",
            cells=(a, b),
            core_cell_ids=("a", "b"),
            margin_cell_ids=(),
            outer_bounds=self.patch_training.PatchBounds(-5, 15, -5, 5),
            core_bounds=self.patch_training.PatchBounds(-5, 15, -5, 5),
            max_steps=5,
            score_normalization="mean_expression_bins",
            force_fill_expression_bins=True,
            force_fill_target_barcodes=("a_seed", "a2", "b_seed", "b2"),
            agent_mode="multi_cell_joint_global_delta",
            after_fill_actions="replace_only",
        )
        env = self.patch_training.TorchJointPatchEnv(patch, device=torch.device("cpu"))
        _obs, _info = env.reset()
        observations = env.joint_observations()
        self.assertEqual(len(observations), 2)
        selected = []
        for obs in observations:
            self.assertGreaterEqual(int(obs["action_features"].shape[0]), 2)
            self.assertFalse(bool(obs["action_mask"][0].item()))
            selected.append(
                {
                    "cell_idx": int(obs["joint_action_cells"][1].item()),
                    "bin_idx": int(obs["joint_action_bins"][1].item()),
                    "barcode_index": int(obs["joint_action_barcodes"][1].item()),
                    "is_replace": bool(obs["joint_action_is_replace"][1].item()),
                    "old_cell_idx": int(obs["joint_action_old_cells"][1].item()),
                    "old_bin_idx": int(obs["joint_action_old_bins"][1].item()),
                }
            )

        _next_obs, _reward, terminated, truncated, info = env.step_joint(selected)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(int(info["step_index"]), 1)
        self.assertEqual(int(info["n_force_fill_owned_expression_bins"]), 4)
        self.assertTrue(bool(info["last_step_applied"]))
        self.assertEqual(str(info["last_step_phase"]), "prefill")
        self.assertEqual(str(info["last_step_outcome"]), "applied")
        full_score = float(env._global_raw_objective().detach().cpu().item() / env._score_denominator())
        self.assertAlmostEqual(float(env.patch_score()), full_score, places=6)
        masks = env.final_masks()
        self.assertEqual(int(masks["a"].sum()), 2)
        self.assertEqual(int(masks["b"].sum()), 2)

    def test_joint_rollout_can_capture_compact_owner_trace(self) -> None:
        import torch
        if not hasattr(torch, "nn") or not hasattr(torch, "distributions"):
            self.skipTest("torch is required for joint patch rollout")

        class FirstActionModel(torch.nn.Module):
            def forward(self, global_features, action_features, action_mask):
                logits = torch.full(action_mask.shape, -10.0, device=action_mask.device)
                logits[:, 0] = 0.0
                if int(action_mask.shape[1]) > 1:
                    logits[:, 1] = torch.where(
                        action_mask[:, 1],
                        torch.as_tensor(10.0, device=action_mask.device),
                        torch.as_tensor(-10.0, device=action_mask.device),
                    )
                return torch.distributions.Categorical(logits=logits), torch.zeros(
                    global_features.shape[0], device=global_features.device
                )

        a = _ctx(self.ppo_state.EpisodeContext, "a", ["a_seed", "a2"], [(0, 0), (2, 0)], [1, 0])
        b = _ctx(self.ppo_state.EpisodeContext, "b", ["b_seed", "b2"], [(10, 0), (12, 0)], [1, 0])
        patch = self.patch_training.PatchContext(
            patch_id="p_joint_trace",
            cells=(a, b),
            core_cell_ids=("a", "b"),
            margin_cell_ids=(),
            outer_bounds=self.patch_training.PatchBounds(-5, 15, -5, 5),
            core_bounds=self.patch_training.PatchBounds(-5, 15, -5, 5),
            max_steps=5,
            score_normalization="mean_expression_bins",
            force_fill_expression_bins=True,
            force_fill_target_barcodes=("a_seed", "a2", "b_seed", "b2"),
            agent_mode="multi_cell_joint_global_delta",
            after_fill_actions="replace_only",
        )

        trajectories, _timing = self.patch_training.collect_patch_trajectories_batched(
            contexts=[patch],
            model=FirstActionModel(),
            device=torch.device("cpu"),
            rng=np.random.default_rng(7),
            policy_mode="greedy",
            rollout_backend="torch_gpu",
            capture_trace=True,
        )

        trajectory = trajectories[0]
        self.assertIsNotNone(trajectory.initial_masks)
        self.assertEqual(int(trajectory.initial_masks["a"].sum()), 1)
        self.assertEqual(int(trajectory.initial_masks["b"].sum()), 1)
        self.assertEqual(len(trajectory.steps[0].action_events), 2)
        self.assertEqual(
            {event.barcode for event in trajectory.steps[0].action_events},
            {"a2", "b2"},
        )
        self.assertTrue(all(event.applied for event in trajectory.steps[0].action_events))
        self.assertEqual(trajectory.steps[0].phase, "prefill")
        self.assertEqual(trajectory.steps[-1].outcome, "stop")
        self.assertEqual(trajectory.steps[-1].n_noop_actions, 2)

    def test_joint_global_delta_postfill_exposes_replace_without_add(self) -> None:
        import torch
        if not hasattr(torch, "as_tensor"):
            self.skipTest("torch is required for joint patch env")

        a = _ctx(self.ppo_state.EpisodeContext, "a", ["a_seed", "shared"], [(0, 0), (2, 0)], [1, 0])
        b = _ctx(self.ppo_state.EpisodeContext, "b", ["b_seed", "shared"], [(10, 0), (2, 0)], [1, 1])
        patch = self.patch_training.PatchContext(
            patch_id="p_joint_postfill",
            cells=(a, b),
            core_cell_ids=("a", "b"),
            margin_cell_ids=(),
            outer_bounds=self.patch_training.PatchBounds(-5, 12, -5, 5),
            core_bounds=self.patch_training.PatchBounds(-5, 12, -5, 5),
            max_steps=5,
            score_normalization="mean_expression_bins",
            force_fill_expression_bins=True,
            force_fill_target_barcodes=("a_seed", "b_seed", "shared"),
            agent_mode="multi_cell_joint_global_delta",
            after_fill_actions="replace_only",
        )
        env = self.patch_training.TorchJointPatchEnv(patch, device=torch.device("cpu"))
        _obs, _info = env.reset()
        observations = env.joint_observations()
        replace_flags = []
        for obs in observations:
            if int(obs["action_features"].shape[0]) > 1:
                replace_flags.extend(_obs_array(obs["joint_action_is_replace"][1:], bool).tolist())
        self.assertTrue(replace_flags)
        self.assertTrue(all(bool(flag) for flag in replace_flags))

    def test_stcs_reward_scores_prefer_similar_close_candidates(self) -> None:
        from hd_cell_rl.stcs_reward import StcsCellPayload, StcsRewardConfig, compute_stcs_reward_scores

        payload = StcsCellPayload(
            cell_id="a",
            candidate_bin_ids=("seed", "near_same", "far_diff"),
            candidate_expression=np.asarray(
                [
                    [10.0, 1.0, 0.0, 0.0],
                    [9.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 8.0, 2.0],
                ],
                dtype=np.float32,
            ),
            candidate_bin_xy_um=np.asarray([(0, 0), (2, 0), (20, 0)], dtype=np.float32),
            nucleus_center_xy_um=np.asarray((0, 0), dtype=np.float32),
            initial_membership_mask=np.asarray([1, 0, 0], dtype=np.uint8),
        )
        scores = compute_stcs_reward_scores(
            [payload],
            StcsRewardConfig(n_top_genes=4, n_pcs=2, lambda_spatial=0.5),
        )

        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0].shape, (3,))
        self.assertTrue(np.isfinite(scores[0]).all())
        self.assertGreater(float(scores[0][1]), float(scores[0][2]))

    def test_joint_global_delta_stcs_replace_uses_fixed_score_delta(self) -> None:
        import torch
        if not hasattr(torch, "as_tensor"):
            self.skipTest("torch is required for joint patch env")

        a = replace(
            _ctx(self.ppo_state.EpisodeContext, "a", ["a_seed", "shared"], [(0, 0), (2, 0)], [1, 0]),
            stcs_reward_scores=np.asarray([0.0, 4.0], dtype=np.float32),
        )
        b = replace(
            _ctx(self.ppo_state.EpisodeContext, "b", ["b_seed", "shared"], [(10, 0), (2, 0)], [1, 1]),
            stcs_reward_scores=np.asarray([0.0, -2.0], dtype=np.float32),
        )
        patch = self.patch_training.PatchContext(
            patch_id="p_joint_stcs_replace",
            cells=(a, b),
            core_cell_ids=("a", "b"),
            margin_cell_ids=(),
            outer_bounds=self.patch_training.PatchBounds(-5, 12, -5, 5),
            core_bounds=self.patch_training.PatchBounds(-5, 12, -5, 5),
            max_steps=5,
            score_normalization="mean_expression_bins",
            reward_backend="stcs",
            force_fill_expression_bins=True,
            force_fill_target_barcodes=("a_seed", "b_seed", "shared"),
            agent_mode="multi_cell_joint_global_delta",
            after_fill_actions="replace_only",
        )
        env = self.patch_training.TorchJointPatchEnv(patch, device=torch.device("cpu"))
        _obs, _info = env.reset()
        observations = env.joint_observations()
        selected = []
        for obs in observations:
            flags = _obs_array(obs["joint_action_is_replace"], bool)
            cells = _obs_array(obs["joint_action_cells"], int)
            if int(obs["action_features"].shape[0]) > 1 and bool(flags[1]) and int(cells[1]) == 0:
                selected.append(
                    {
                        "cell_idx": int(obs["joint_action_cells"][1].item()),
                        "bin_idx": int(obs["joint_action_bins"][1].item()),
                        "barcode_index": int(obs["joint_action_barcodes"][1].item()),
                        "is_replace": bool(obs["joint_action_is_replace"][1].item()),
                        "old_cell_idx": int(obs["joint_action_old_cells"][1].item()),
                        "old_bin_idx": int(obs["joint_action_old_bins"][1].item()),
                    }
                )
                break

        self.assertEqual(len(selected), 1)
        _next_obs, reward, terminated, truncated, info = env.step_joint(selected)
        self.assertAlmostEqual(float(reward), 6.0 / 3.0, places=6)
        self.assertFalse(truncated)
        self.assertFalse(terminated)
        self.assertAlmostEqual(float(info["raw_total_core_reward"]), 4.0, places=6)
        masks = env.final_masks()
        self.assertEqual(int(masks["a"][1]), 1)
        self.assertEqual(int(masks["b"][1]), 0)

    def test_joint_rollout_cache_groups_local_actions(self) -> None:
        step_a = self.patch_training.PatchStep(
            global_features=np.zeros((self.patch_training.GLOBAL_FEATURE_DIM,), dtype=np.float32),
            action_features=np.zeros((2, self.patch_training.ACTION_FEATURE_DIM), dtype=np.float32),
            action_mask=np.asarray([True, True], dtype=bool),
            action=1,
            reward=3.0,
            done=True,
            old_log_prob=-0.2,
            old_value=0.5,
            joint_group_id=0,
            joint_old_log_prob=-0.5,
            joint_old_value=0.6,
        )
        step_b = self.patch_training.PatchStep(
            global_features=np.zeros((self.patch_training.GLOBAL_FEATURE_DIM,), dtype=np.float32),
            action_features=np.zeros((2, self.patch_training.ACTION_FEATURE_DIM), dtype=np.float32),
            action_mask=np.asarray([True, True], dtype=bool),
            action=0,
            reward=3.0,
            done=True,
            old_log_prob=-0.3,
            old_value=0.7,
            joint_group_id=0,
            joint_old_log_prob=-0.5,
            joint_old_value=0.6,
        )
        traj = self.patch_training.PatchTrajectory(
            patch_slot=0,
            patch_id="p_joint_cache",
            steps=(step_a, step_b),
            total_reward=3.0,
            patch_score=3.0,
            metrics={},
            final_masks={},
        )
        cache = self.patch_training.build_patch_rollout_cache(
            trajectories=[traj],
            gamma=1.0,
            gae_lambda=1.0,
            normalize_advantages=False,
            training_mode="ppo",
            group_size=1,
            norm_epsilon=1.0e-6,
        )
        self.assertEqual(cache.n_transitions, 1)
        self.assertEqual(cache.transitions[0].joint_group_id, 0)
        self.assertEqual(_obs_array(cache.transitions[0].action, int).shape[0], 2)

    def test_joint_macro_transition_runs_ppo_update(self) -> None:
        import torch
        from hd_cell_rl.ppo_model import ActorCritic

        if not hasattr(torch, "as_tensor"):
            self.skipTest("torch is required for joint patch PPO update")

        step = self.patch_training.PatchStep(
            global_features=np.zeros((2, self.patch_training.GLOBAL_FEATURE_DIM), dtype=np.float32),
            action_features=np.zeros((2, 2, self.patch_training.ACTION_FEATURE_DIM), dtype=np.float32),
            action_mask=np.asarray([[True, True], [True, True]], dtype=bool),
            action=np.asarray([1, 0], dtype=np.int64),
            reward=1.0,
            done=True,
            old_log_prob=-1.0,
            old_value=0.0,
            joint_group_id=0,
            joint_old_log_prob=-1.0,
            joint_old_value=0.0,
        )
        traj = self.patch_training.PatchTrajectory(
            patch_slot=0,
            patch_id="p_joint_macro_update",
            steps=(step,),
            total_reward=1.0,
            patch_score=1.0,
            metrics={},
            final_masks={},
        )
        cache = self.patch_training.build_patch_rollout_cache(
            trajectories=[traj],
            gamma=1.0,
            gae_lambda=1.0,
            normalize_advantages=False,
            training_mode="ppo",
            group_size=1,
            norm_epsilon=1.0e-6,
        )
        model = ActorCritic(
            global_dim=self.patch_training.GLOBAL_FEATURE_DIM,
            action_dim=self.patch_training.ACTION_FEATURE_DIM,
            hidden_dim=8,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        metrics = self.patch_training.patch_ppo_update(
            model=model,
            optimizer=optimizer,
            cache=cache,
            eps_clip=0.2,
            ppo_epochs=1,
            minibatch_size=1,
            vf_coef=0.0,
            ent_coef=0.0,
            max_grad_norm=1.0,
            target_kl=None,
            include_value_loss=False,
            device=torch.device("cpu"),
            rng=np.random.default_rng(7),
        )
        self.assertTrue(np.isfinite(metrics["policy_loss"]))

    def test_patch_dataset_caches_loaded_patch_contexts(self) -> None:
        class FakeCellDataset:
            def __init__(self, episode_context_cls) -> None:
                self.episode_context_cls = episode_context_cls
                self.calls = 0

            def load_episode_context(self, **kwargs):
                self.calls += 1
                return _ctx(self.episode_context_cls, "a", ["seed", "a2"], [(0, 0), (2, 0)], [1, 0])

        dataset = object.__new__(self.patch_training.PatchDataset)
        dataset._base_config = types.SimpleNamespace(max_steps_per_episode=5)
        dataset._settings = types.SimpleNamespace(
            margin_cells_compete=True,
            max_steps_per_patch=5,
            score_normalization="sqrt_core_cells",
            cache_patch_contexts=True,
        )
        fake_cells = FakeCellDataset(self.ppo_state.EpisodeContext)
        dataset._cell_dataset = fake_cells
        dataset._artifact_by_cell = {"a": "artifact"}
        dataset._context_cache = {}
        row = types.SimpleNamespace(
            patch_id="p_cached_load",
            patch_cell_ids='["a"]',
            core_cell_ids='["a"]',
            margin_cell_ids="[]",
            outer_x_min=-5.0,
            outer_x_max=5.0,
            outer_y_min=-5.0,
            outer_y_max=5.0,
            core_x_min=-5.0,
            core_x_max=5.0,
            core_y_min=-5.0,
            core_y_max=5.0,
        )

        first = dataset.load_patch_context(row)
        second = dataset.load_patch_context(row)
        self.assertIs(first, second)
        self.assertEqual(fake_cells.calls, 1)

    def test_patch_dataset_attaches_stcs_reward_scores_when_requested(self) -> None:
        class FakeCellDataset:
            def __init__(self, episode_context_cls) -> None:
                self.episode_context_cls = episode_context_cls

            def load_episode_context(self, **kwargs):
                return _ctx(
                    self.episode_context_cls,
                    "a",
                    ["seed", "near_same", "far_diff"],
                    [(0, 0), (2, 0), (20, 0)],
                    [1, 0, 0],
                )

            def load_episode_candidate_expression(self, **kwargs):
                return (
                    ("seed", "near_same", "far_diff"),
                    np.asarray(
                        [
                            [10.0, 1.0, 0.0, 0.0],
                            [9.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 8.0, 2.0],
                        ],
                        dtype=np.float32,
                    ),
                )

        dataset = object.__new__(self.patch_training.PatchDataset)
        dataset._base_config = types.SimpleNamespace(max_steps_per_episode=5)
        dataset._settings = types.SimpleNamespace(
            margin_cells_compete=True,
            max_steps_per_patch=5,
            score_normalization="mean_core_cells",
            cache_patch_contexts=False,
            reward_backend="stcs",
            stcs_reward_config={"n_top_genes": 4, "n_pcs": 2, "lambda_spatial": 0.5},
        )
        dataset._cell_dataset = FakeCellDataset(self.ppo_state.EpisodeContext)
        dataset._artifact_by_cell = {"a": "artifact"}
        dataset._context_cache = {}
        row = types.SimpleNamespace(
            patch_id="p_stcs_load",
            patch_cell_ids='["a"]',
            core_cell_ids='["a"]',
            margin_cell_ids="[]",
            outer_x_min=-5.0,
            outer_x_max=25.0,
            outer_y_min=-5.0,
            outer_y_max=5.0,
            core_x_min=-5.0,
            core_x_max=25.0,
            core_y_min=-5.0,
            core_y_max=5.0,
        )

        patch = dataset.load_patch_context(row)
        self.assertIsNotNone(patch)
        self.assertEqual(patch.reward_backend, "stcs")
        scores = patch.cells[0].stcs_reward_scores
        self.assertIsNotNone(scores)
        self.assertEqual(scores.shape, (3,))
        self.assertTrue(np.isfinite(scores).all())

    def test_patch_competition_margin_rewards_best_legal_nearby_cell(self) -> None:
        a = replace(
            _ctx(self.ppo_state.EpisodeContext, "a", ["a_seed", "shared"], [(0, 0), (2, 0)], [1, 0]),
            base_penalty=np.asarray([0.0, 1.0], dtype=np.float32),
            w1=0.0,
            w4=0.0,
            w5=0.0,
            stop_lambda=0.0,
            competition_margin_weight=0.25,
            competition_margin_radius_um=20.0,
        )
        b = replace(
            _ctx(self.ppo_state.EpisodeContext, "b", ["b_seed", "shared"], [(4, 0), (2, 0)], [1, 0]),
            base_penalty=np.asarray([0.0, 0.0], dtype=np.float32),
            w1=0.0,
            w4=0.0,
            w5=0.0,
            stop_lambda=0.0,
            competition_margin_weight=0.25,
            competition_margin_radius_um=20.0,
        )
        patch = self.patch_training.PatchContext(
            patch_id="p_compete",
            cells=(a, b),
            core_cell_ids=("a", "b"),
            margin_cell_ids=(),
            outer_bounds=self.patch_training.PatchBounds(-5, 6, -5, 5),
            core_bounds=self.patch_training.PatchBounds(-5, 6, -5, 5),
            max_steps=5,
        )
        env = self.patch_training.MultiCellPatchEnv(patch)
        obs, _ = env.reset()

        self.assertEqual(env._cached_action_map[0][:2], (0, 1))
        self.assertEqual(env._cached_action_map[1][:2], (1, 1))
        self.assertAlmostEqual(env._cached_action_map[0][2], -1.25, places=6)
        self.assertAlmostEqual(env._cached_action_map[1][2], 0.25, places=6)
        margin_col = self.patch_training.A_COMPETITION_MARGIN
        self.assertAlmostEqual(float(obs["action_features"][1, margin_col]), -0.2, places=6)
        self.assertAlmostEqual(float(obs["action_features"][2, margin_col]), 0.2, places=6)

    def test_patch_competition_margin_can_be_disabled_by_patch_context(self) -> None:
        a = replace(
            _ctx(self.ppo_state.EpisodeContext, "a", ["a_seed", "shared"], [(0, 0), (2, 0)], [1, 0]),
            base_penalty=np.asarray([0.0, 1.0], dtype=np.float32),
            w1=0.0,
            w4=0.0,
            w5=0.0,
            stop_lambda=0.0,
            competition_margin_weight=0.25,
            competition_margin_radius_um=20.0,
        )
        b = replace(
            _ctx(self.ppo_state.EpisodeContext, "b", ["b_seed", "shared"], [(4, 0), (2, 0)], [1, 0]),
            base_penalty=np.asarray([0.0, 0.0], dtype=np.float32),
            w1=0.0,
            w4=0.0,
            w5=0.0,
            stop_lambda=0.0,
            competition_margin_weight=0.25,
            competition_margin_radius_um=20.0,
        )
        patch = self.patch_training.PatchContext(
            patch_id="p_no_compete",
            cells=(a, b),
            core_cell_ids=("a", "b"),
            margin_cell_ids=(),
            outer_bounds=self.patch_training.PatchBounds(-5, 6, -5, 5),
            core_bounds=self.patch_training.PatchBounds(-5, 6, -5, 5),
            max_steps=5,
            competition_margin_enabled=False,
        )
        envs = [
            self.patch_training.MultiCellPatchEnv(patch),
            self.patch_training.CachedMultiCellPatchEnv(patch),
        ]
        import torch
        if hasattr(torch, "as_tensor"):
            envs.append(self.patch_training.TorchPatchEnv(patch, device=torch.device("cpu")))

        margin_col = self.patch_training.A_COMPETITION_MARGIN
        for env in envs:
            with self.subTest(env=env.__class__.__name__):
                obs, _ = env.reset()
                if getattr(env, "_cached_action_map", None):
                    rewards = [float(item[2]) for item in env._cached_action_map]
                else:
                    rewards = _obs_array(env._cached_action_rewards, np.float32).tolist()
                self.assertAlmostEqual(float(rewards[0]), -1.0, places=6)
                self.assertAlmostEqual(float(rewards[1]), 0.0, places=6)
                action_features = _obs_array(obs["action_features"], np.float32)
                self.assertAlmostEqual(float(action_features[1, margin_col]), 0.0, places=6)
                self.assertAlmostEqual(float(action_features[2, margin_col]), 0.0, places=6)

    def test_patch_competition_margin_ignores_out_of_radius_competitor(self) -> None:
        a = replace(
            _ctx(self.ppo_state.EpisodeContext, "a", ["a_seed", "shared"], [(0, 0), (2, 0)], [1, 0]),
            base_penalty=np.asarray([0.0, 1.0], dtype=np.float32),
            w1=0.0,
            w4=0.0,
            w5=0.0,
            stop_lambda=0.0,
            competition_margin_weight=0.25,
            competition_margin_radius_um=1.0,
        )
        b = replace(
            _ctx(self.ppo_state.EpisodeContext, "b", ["b_seed", "shared"], [(4, 0), (2, 0)], [1, 0]),
            base_penalty=np.asarray([0.0, 0.0], dtype=np.float32),
            w1=0.0,
            w4=0.0,
            w5=0.0,
            stop_lambda=0.0,
            competition_margin_weight=0.25,
            competition_margin_radius_um=1.0,
        )
        patch = self.patch_training.PatchContext(
            patch_id="p_compete_radius",
            cells=(a, b),
            core_cell_ids=("a", "b"),
            margin_cell_ids=(),
            outer_bounds=self.patch_training.PatchBounds(-5, 6, -5, 5),
            core_bounds=self.patch_training.PatchBounds(-5, 6, -5, 5),
            max_steps=5,
        )
        env = self.patch_training.MultiCellPatchEnv(patch)
        obs, _ = env.reset()

        self.assertAlmostEqual(env._cached_action_map[0][2], -1.0, places=6)
        self.assertAlmostEqual(env._cached_action_map[1][2], 0.0, places=6)
        margin_col = self.patch_training.A_COMPETITION_MARGIN
        self.assertAlmostEqual(float(obs["action_features"][1, margin_col]), 0.0, places=6)
        self.assertAlmostEqual(float(obs["action_features"][2, margin_col]), 0.0, places=6)

    def test_patch_competition_margin_can_leave_stop_reward_unadjusted(self) -> None:
        a = replace(
            _ctx(self.ppo_state.EpisodeContext, "a", ["a_seed", "shared"], [(0, 0), (2, 0)], [1, 0]),
            base_penalty=np.asarray([0.0, 1.0], dtype=np.float32),
            w1=0.0,
            w4=0.0,
            w5=0.0,
            stop_lambda=1.0,
            competition_margin_weight=0.25,
            competition_margin_radius_um=20.0,
            competition_margin_affects_stop=False,
        )
        b = replace(
            _ctx(self.ppo_state.EpisodeContext, "b", ["b_seed", "shared"], [(4, 0), (2, 0)], [1, 0]),
            base_penalty=np.asarray([0.0, 0.0], dtype=np.float32),
            w1=0.0,
            w4=0.0,
            w5=0.0,
            stop_lambda=1.0,
            competition_margin_weight=0.0,
            competition_margin_radius_um=20.0,
            competition_margin_affects_stop=False,
        )
        patch = self.patch_training.PatchContext(
            patch_id="p_compete_stop",
            cells=(a, b),
            core_cell_ids=("a", "b"),
            margin_cell_ids=(),
            outer_bounds=self.patch_training.PatchBounds(-5, 6, -5, 5),
            core_bounds=self.patch_training.PatchBounds(-5, 6, -5, 5),
            max_steps=5,
        )
        envs = [
            self.patch_training.MultiCellPatchEnv(patch),
            self.patch_training.CachedMultiCellPatchEnv(patch),
        ]
        import torch
        if hasattr(torch, "as_tensor"):
            envs.append(self.patch_training.TorchPatchEnv(patch, device=torch.device("cpu")))

        for env in envs:
            env.reset()
            if getattr(env, "_cached_action_map", None):
                rewards = [float(item[2]) for item in env._cached_action_map]
            else:
                rewards = _obs_array(env._cached_action_rewards, np.float32).tolist()
            self.assertAlmostEqual(float(rewards[0]), -1.25, places=6)
            self.assertAlmostEqual(float(rewards[1]), 0.0, places=6)
            self.assertAlmostEqual(float(env._stop_reward_value), 0.5, places=6)

    def test_patch_competition_margin_uses_nonfrontier_nearby_competitor(self) -> None:
        a = replace(
            _ctx(self.ppo_state.EpisodeContext, "a", ["a_seed", "shared"], [(0, 0), (2, 0)], [1, 0]),
            base_penalty=np.asarray([0.0, 1.0], dtype=np.float32),
            w1=0.0,
            w4=0.0,
            w5=0.0,
            stop_lambda=0.0,
            competition_margin_weight=0.25,
            competition_margin_radius_um=20.0,
        )
        b = replace(
            _ctx(self.ppo_state.EpisodeContext, "b", ["b_seed", "shared"], [(4, 0), (2, 0)], [1, 0]),
            base_penalty=np.asarray([0.0, 0.0], dtype=np.float32),
            neighbor_index=np.full((2, 8), -1, dtype=np.int32),
            w1=0.0,
            w4=0.0,
            w5=0.0,
            stop_lambda=0.0,
            competition_margin_weight=0.25,
            competition_margin_radius_um=20.0,
        )
        patch = self.patch_training.PatchContext(
            patch_id="p_compete_legal",
            cells=(a, b),
            core_cell_ids=("a", "b"),
            margin_cell_ids=(),
            outer_bounds=self.patch_training.PatchBounds(-5, 6, -5, 5),
            core_bounds=self.patch_training.PatchBounds(-5, 6, -5, 5),
            max_steps=5,
        )
        envs = [
            self.patch_training.MultiCellPatchEnv(patch),
            self.patch_training.CachedMultiCellPatchEnv(patch),
        ]
        import torch
        if hasattr(torch, "as_tensor"):
            envs.append(self.patch_training.TorchPatchEnv(patch, device=torch.device("cpu")))

        for env in envs:
            obs, _ = env.reset()
            if getattr(env, "_cached_action_map", None):
                rewards = [float(item[2]) for item in env._cached_action_map]
            else:
                rewards = _obs_array(env._cached_action_rewards, np.float32).tolist()
            self.assertEqual(len(rewards), 1)
            self.assertAlmostEqual(float(rewards[0]), -1.25, places=6)
            action_features = _obs_array(obs["action_features"], np.float32)
            self.assertAlmostEqual(
                float(action_features[1, self.patch_training.A_COMPETITION_MARGIN]),
                -0.2,
                places=6,
            )

    def test_patch_competition_margin_clips_patch_normalized_score(self) -> None:
        a = replace(
            _ctx(self.ppo_state.EpisodeContext, "a", ["a_seed", "shared"], [(0, 0), (2, 0)], [1, 0]),
            base_penalty=np.asarray([0.0, 0.0], dtype=np.float32),
            w1=10.0,
            w4=0.0,
            w5=0.0,
            stop_lambda=0.0,
            competition_margin_weight=1.0,
            competition_margin_radius_um=20.0,
            competition_margin_clip=0.5,
        )
        b = replace(
            _ctx(self.ppo_state.EpisodeContext, "b", ["b_seed", "shared"], [(4, 0), (2, 0)], [1, 0]),
            base_penalty=np.asarray([0.0, 0.0], dtype=np.float32),
            w1=10.0,
            w4=0.0,
            w5=0.0,
            stop_lambda=0.0,
            competition_margin_weight=1.0,
            competition_margin_radius_um=20.0,
            competition_margin_clip=0.5,
        )
        adjusted, margins, _ = self.patch_training._compute_competition_adjusted_rewards_np(
            cells=(a, b),
            competition_candidates=(((), ((1, 1),)), ((), ((0, 1),))),
            add_map=[(0, 1, 0.0), (1, 1, 0.0)],
            competition_expr_by_cell={
                0: np.asarray([0.0, 100.0], dtype=np.float32),
                1: np.asarray([0.0, -100.0], dtype=np.float32),
            },
            membership_masks_by_cell={
                0: np.asarray([1, 0], dtype=np.uint8),
                1: np.asarray([1, 0], dtype=np.uint8),
            },
        )

        np.testing.assert_allclose(adjusted, np.asarray([0.5, -0.5], dtype=np.float32), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(margins, np.asarray([0.5, -0.5], dtype=np.float32), rtol=1e-6, atol=1e-6)

    def test_patch_competition_margin_backends_match(self) -> None:
        import torch
        if not hasattr(torch, "as_tensor"):
            self.skipTest("torch is required for torch patch backend")

        a = replace(
            _ctx(self.ppo_state.EpisodeContext, "a", ["a_seed", "shared"], [(0, 0), (2, 0)], [1, 0]),
            base_penalty=np.asarray([0.0, 1.0], dtype=np.float32),
            w1=0.0,
            w4=0.0,
            w5=0.0,
            stop_lambda=0.0,
            competition_margin_weight=0.25,
            competition_margin_radius_um=20.0,
        )
        b = replace(
            _ctx(self.ppo_state.EpisodeContext, "b", ["b_seed", "shared"], [(4, 0), (2, 0)], [1, 0]),
            base_penalty=np.asarray([0.0, 0.0], dtype=np.float32),
            w1=0.0,
            w4=0.0,
            w5=0.0,
            stop_lambda=0.0,
            competition_margin_weight=0.25,
            competition_margin_radius_um=20.0,
        )
        patch = self.patch_training.PatchContext(
            patch_id="p_compete_backends",
            cells=(a, b),
            core_cell_ids=("a", "b"),
            margin_cell_ids=(),
            outer_bounds=self.patch_training.PatchBounds(-5, 6, -5, 5),
            core_bounds=self.patch_training.PatchBounds(-5, 6, -5, 5),
            max_steps=5,
        )
        legacy = self.patch_training.MultiCellPatchEnv(patch)
        cached = self.patch_training.CachedMultiCellPatchEnv(patch)
        torch_env = self.patch_training.TorchPatchEnv(patch, device=torch.device("cpu"))
        legacy_obs, _ = legacy.reset()
        cached_obs, _ = cached.reset()
        torch_obs, _ = torch_env.reset()

        np.testing.assert_allclose(cached_obs["action_features"], legacy_obs["action_features"], rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(
            _obs_array(torch_obs["action_features"], np.float32),
            legacy_obs["action_features"],
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        _, legacy_reward, *_ = legacy.step(1)
        _, cached_reward, *_ = cached.step(1)
        _, torch_reward, *_ = torch_env.step(1)
        self.assertAlmostEqual(float(cached_reward), float(legacy_reward), places=6)
        self.assertAlmostEqual(float(torch_reward), float(legacy_reward), places=6)

    def test_patch_competition_margin_with_shape_backends_match(self) -> None:
        import torch
        if not hasattr(torch, "as_tensor"):
            self.skipTest("torch is required for torch patch backend")

        shape_model = self.shape_prior.ShapePriorModel(
            feature_names=self.shape_prior.SHAPE_FEATURE_NAMES,
            cluster_labels=("cluster_00",),
            n_cells=np.asarray([10], dtype=np.int64),
            means=np.zeros((1, 4), dtype=np.float64),
            covariances=np.eye(4, dtype=np.float64).reshape(1, 4, 4),
            inv_covariances=np.eye(4, dtype=np.float64).reshape(1, 4, 4),
            log_determinants=np.zeros((1,), dtype=np.float64),
            priors=np.ones((1,), dtype=np.float64),
            epsilon=1.0e-4,
            scaler_mean=np.zeros((4,), dtype=np.float64),
            scaler_std=np.ones((4,), dtype=np.float64),
            zscored_input=False,
        )
        neighbor_a = np.full((4, 8), -1, dtype=np.int32)
        neighbor_a[0, 0] = 1
        neighbor_a[1, 0] = 2
        neighbor_a[2, 0:2] = [1, 3]
        neighbor_a[3, 0] = 2
        neighbor_b = np.full((4, 8), -1, dtype=np.int32)
        neighbor_b[0, 0:2] = [1, 2]
        neighbor_b[1, 0] = 0
        neighbor_b[2, 0:2] = [0, 3]
        neighbor_b[3, 0] = 2
        a = replace(
            _ctx(
                self.ppo_state.EpisodeContext,
                "a",
                ["a0", "a1", "shared", "a3"],
                [(0, 0), (2, 0), (4, 0), (6, 0)],
                [1, 1, 0, 0],
            ),
            w1=0.0,
            w4=0.0,
            w5=0.0,
            stop_lambda=0.0,
            neighbor_index=neighbor_a,
            competition_margin_weight=0.25,
            competition_margin_radius_um=20.0,
            shape_prior_model=shape_model,
            shape_prior_weight=0.4,
            shape_prior_mode="mixture",
            shape_prior_normalize_over_frontier=True,
            shape_prior_clip=5.0,
        )
        b = replace(
            _ctx(
                self.ppo_state.EpisodeContext,
                "b",
                ["b0", "b1", "shared", "b3"],
                [(4, 2), (4, 4), (4, 0), (4, -2)],
                [1, 1, 0, 0],
            ),
            w1=0.0,
            w4=0.0,
            w5=0.0,
            stop_lambda=0.0,
            neighbor_index=neighbor_b,
            competition_margin_weight=0.25,
            competition_margin_radius_um=20.0,
            shape_prior_model=shape_model,
            shape_prior_weight=0.4,
            shape_prior_mode="mixture",
            shape_prior_normalize_over_frontier=True,
            shape_prior_clip=5.0,
        )
        patch = self.patch_training.PatchContext(
            patch_id="p_compete_shape",
            cells=(a, b),
            core_cell_ids=("a", "b"),
            margin_cell_ids=(),
            outer_bounds=self.patch_training.PatchBounds(-5, 7, -3, 5),
            core_bounds=self.patch_training.PatchBounds(-5, 7, -3, 5),
            max_steps=5,
        )
        legacy = self.patch_training.MultiCellPatchEnv(patch)
        cached = self.patch_training.CachedMultiCellPatchEnv(patch)
        torch_env = self.patch_training.TorchPatchEnv(patch, device=torch.device("cpu"))
        legacy_obs, _ = legacy.reset()
        cached_obs, _ = cached.reset()
        torch_obs, _ = torch_env.reset()

        np.testing.assert_allclose(cached_obs["action_features"], legacy_obs["action_features"], rtol=1.0e-4, atol=1.0e-4)
        np.testing.assert_allclose(
            _obs_array(torch_obs["action_features"], np.float32),
            legacy_obs["action_features"],
            rtol=1.0e-4,
            atol=1.0e-4,
        )
        _, legacy_reward, *_ = legacy.step(1)
        _, cached_reward, *_ = cached.step(1)
        _, torch_reward, *_ = torch_env.step(1)
        self.assertAlmostEqual(float(cached_reward), float(legacy_reward), places=4)
        self.assertAlmostEqual(float(torch_reward), float(legacy_reward), places=4)

    def test_torch_patch_env_matches_legacy_cpu_without_shape_prior(self) -> None:
        import torch
        if not hasattr(torch, "as_tensor"):
            self.skipTest("torch is required for torch patch backend")

        ctx = _ctx(
            self.ppo_state.EpisodeContext,
            "a",
            ["seed", "a1", "a2", "a3"],
            [(0, 0), (2, 0), (0, 2), (2, 2)],
            [1, 0, 0, 0],
        )
        neighbor_index = np.full((4, 8), -1, dtype=np.int32)
        neighbor_index[0, 0:2] = [1, 2]
        neighbor_index[1, 0:2] = [0, 3]
        neighbor_index[2, 0:2] = [0, 3]
        neighbor_index[3, 0:2] = [1, 2]
        ctx = replace(
            ctx,
            ll=np.asarray(
                [
                    [0.2, -0.1, 0.0],
                    [0.8, -0.3, 0.1],
                    [-0.2, 0.7, 0.2],
                    [0.1, 0.0, 0.6],
                ],
                dtype=np.float32,
            ),
            ll_mean_z=np.asarray([0.1, 1.2, -0.5, 0.4], dtype=np.float32),
            ll_max_z=np.asarray([0.2, 1.1, 0.8, 0.7], dtype=np.float32),
            base_penalty=np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float32),
            neighbor_index=neighbor_index,
            w1=0.7,
            w4=0.25,
            w5=0.15,
            stop_lambda=0.2,
            stop_top_k=2,
            shape_prior_weight=0.0,
        )
        patch = self.patch_training.PatchContext(
            patch_id="p_torch",
            cells=(ctx,),
            core_cell_ids=("a",),
            margin_cell_ids=(),
            outer_bounds=self.patch_training.PatchBounds(-5, 5, -5, 5),
            core_bounds=self.patch_training.PatchBounds(-5, 5, -5, 5),
            max_steps=5,
        )

        legacy = self.patch_training.MultiCellPatchEnv(patch)
        torch_env = self.patch_training.TorchPatchEnv(patch, device=torch.device("cpu"))
        legacy_obs, _ = legacy.reset()
        torch_obs, _ = torch_env.reset()

        np.testing.assert_allclose(
            _obs_array(torch_obs["global_features"], np.float32),
            legacy_obs["global_features"],
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            _obs_array(torch_obs["action_features"], np.float32),
            legacy_obs["action_features"],
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        np.testing.assert_array_equal(_obs_array(torch_obs["action_mask"], bool), legacy_obs["action_mask"])

        legacy_next, legacy_reward, *_ = legacy.step(1)
        torch_next, torch_reward, *_ = torch_env.step(1)
        self.assertAlmostEqual(float(torch_reward), float(legacy_reward), places=6)
        np.testing.assert_allclose(
            _obs_array(torch_next["global_features"], np.float32),
            legacy_next["global_features"],
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            _obs_array(torch_next["action_features"], np.float32),
            legacy_next["action_features"],
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        np.testing.assert_array_equal(torch_env.final_masks()["a"], legacy.final_masks()["a"])

    def test_torch_patch_env_dirty_cache_matches_legacy_for_shared_barcode(self) -> None:
        import torch
        if not hasattr(torch, "as_tensor"):
            self.skipTest("torch is required for torch patch backend")

        a = _ctx(self.ppo_state.EpisodeContext, "a", ["a_seed", "shared", "a2"], [(0, 0), (2, 0), (4, 0)], [1, 0, 0])
        b = _ctx(self.ppo_state.EpisodeContext, "b", ["b_seed", "shared", "b2"], [(4, 0), (2, 0), (0, 0)], [1, 0, 0])
        neighbor_index = np.full((3, 8), -1, dtype=np.int32)
        neighbor_index[0, 0:2] = [1, 2]
        neighbor_index[1, 0] = 0
        neighbor_index[2, 0] = 0
        a = replace(a, neighbor_index=neighbor_index)
        b = replace(b, neighbor_index=neighbor_index)
        patch = self.patch_training.PatchContext(
            patch_id="p_torch_shared",
            cells=(a, b),
            core_cell_ids=("a", "b"),
            margin_cell_ids=(),
            outer_bounds=self.patch_training.PatchBounds(-5, 6, -5, 5),
            core_bounds=self.patch_training.PatchBounds(-5, 6, -5, 5),
            max_steps=5,
        )

        legacy = self.patch_training.MultiCellPatchEnv(patch)
        torch_env = self.patch_training.TorchPatchEnv(patch, device=torch.device("cpu"))
        legacy.reset()
        torch_env.reset()
        legacy_next, legacy_reward, *_ = legacy.step(1)
        torch_next, torch_reward, *_ = torch_env.step(1)

        self.assertAlmostEqual(float(torch_reward), float(legacy_reward), places=6)
        np.testing.assert_allclose(
            _obs_array(torch_next["global_features"], np.float32),
            legacy_next["global_features"],
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            _obs_array(torch_next["action_features"], np.float32),
            legacy_next["action_features"],
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        np.testing.assert_array_equal(_obs_array(torch_next["action_mask"], bool), legacy_next["action_mask"])
        masks = torch_env.final_masks()
        self.assertEqual(int(masks["a"][1] + masks["b"][1]), 1)

    def test_torch_patch_env_matches_legacy_cpu_with_shape_prior(self) -> None:
        import torch
        if not hasattr(torch, "as_tensor"):
            self.skipTest("torch is required for torch patch backend")

        shape_model = self.shape_prior.ShapePriorModel(
            feature_names=self.shape_prior.SHAPE_FEATURE_NAMES,
            cluster_labels=("cluster_00",),
            n_cells=np.asarray([10], dtype=np.int64),
            means=np.zeros((1, 4), dtype=np.float64),
            covariances=np.eye(4, dtype=np.float64).reshape(1, 4, 4),
            inv_covariances=np.eye(4, dtype=np.float64).reshape(1, 4, 4),
            log_determinants=np.zeros((1,), dtype=np.float64),
            priors=np.ones((1,), dtype=np.float64),
            epsilon=1.0e-4,
            scaler_mean=np.zeros((4,), dtype=np.float64),
            scaler_std=np.ones((4,), dtype=np.float64),
            zscored_input=False,
        )
        ctx = _ctx(
            self.ppo_state.EpisodeContext,
            "a",
            ["seed", "a1", "a2", "a3", "a4"],
            [(0, 0), (2, 0), (0, 2), (2, 2), (4, 2)],
            [1, 0, 0, 0, 0],
        )
        neighbor_index = np.full((5, 8), -1, dtype=np.int32)
        neighbor_index[0, 0:2] = [1, 2]
        neighbor_index[1, 0:3] = [0, 2, 3]
        neighbor_index[2, 0:3] = [0, 1, 3]
        neighbor_index[3, 0:4] = [0, 1, 2, 4]
        neighbor_index[4, 0] = 3
        ctx = replace(
            ctx,
            neighbor_index=neighbor_index,
            w1=0.5,
            w4=0.05,
            w5=0.1,
            stop_lambda=0.2,
            stop_top_k=2,
            shape_prior_model=shape_model,
            shape_prior_weight=0.65,
            shape_prior_mode="mixture",
            shape_prior_normalize_over_frontier=True,
            shape_prior_clip=5.0,
        )
        patch = self.patch_training.PatchContext(
            patch_id="p_shape",
            cells=(ctx,),
            core_cell_ids=("a",),
            margin_cell_ids=(),
            outer_bounds=self.patch_training.PatchBounds(-5, 5, -5, 5),
            core_bounds=self.patch_training.PatchBounds(-5, 5, -5, 5),
            max_steps=5,
        )
        legacy = self.patch_training.MultiCellPatchEnv(patch)
        torch_env = self.patch_training.TorchPatchEnv(patch, device=torch.device("cpu"))
        legacy_obs, _ = legacy.reset()
        torch_obs, _ = torch_env.reset()
        np.testing.assert_allclose(
            _obs_array(torch_obs["global_features"], np.float32),
            legacy_obs["global_features"],
            rtol=1.0e-4,
            atol=1.0e-4,
        )
        np.testing.assert_allclose(
            _obs_array(torch_obs["action_features"], np.float32),
            legacy_obs["action_features"],
            rtol=1.0e-4,
            atol=1.0e-4,
        )
        np.testing.assert_array_equal(_obs_array(torch_obs["action_mask"], bool), legacy_obs["action_mask"])
        legacy_next, legacy_reward, *_ = legacy.step(1)
        torch_next, torch_reward, *_ = torch_env.step(1)
        self.assertAlmostEqual(float(torch_reward), float(legacy_reward), places=4)
        np.testing.assert_allclose(
            _obs_array(torch_next["global_features"], np.float32),
            legacy_next["global_features"],
            rtol=1.0e-4,
            atol=1.0e-4,
        )
        np.testing.assert_allclose(
            _obs_array(torch_next["action_features"], np.float32),
            legacy_next["action_features"],
            rtol=1.0e-4,
            atol=1.0e-4,
        )
        np.testing.assert_array_equal(_obs_array(torch_next["action_mask"], bool), legacy_next["action_mask"])
        np.testing.assert_array_equal(torch_env.final_masks()["a"], legacy.final_masks()["a"])

    def test_torch_shape_prior_inside_hull_shortcut_matches_legacy_cpu(self) -> None:
        import torch
        if not hasattr(torch, "as_tensor"):
            self.skipTest("torch is required for torch patch backend")

        shape_model = self.shape_prior.ShapePriorModel(
            feature_names=self.shape_prior.SHAPE_FEATURE_NAMES,
            cluster_labels=("cluster_00",),
            n_cells=np.asarray([10], dtype=np.int64),
            means=np.zeros((1, 4), dtype=np.float64),
            covariances=np.eye(4, dtype=np.float64).reshape(1, 4, 4),
            inv_covariances=np.eye(4, dtype=np.float64).reshape(1, 4, 4),
            log_determinants=np.zeros((1,), dtype=np.float64),
            priors=np.ones((1,), dtype=np.float64),
            epsilon=1.0e-4,
            scaler_mean=np.zeros((4,), dtype=np.float64),
            scaler_std=np.ones((4,), dtype=np.float64),
            zscored_input=False,
        )
        xy = [(0, 0), (2, 0), (4, 0), (0, 2), (4, 2), (0, 4), (2, 4), (4, 4), (2, 2), (6, 2)]
        ctx = _ctx(
            self.ppo_state.EpisodeContext,
            "a",
            ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "inside", "outside"],
            xy,
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        )
        neighbor_index = np.full((10, 8), -1, dtype=np.int32)
        neighbor_index[8, 0:4] = [1, 3, 4, 6]
        neighbor_index[9, 0] = 4
        ctx = replace(
            ctx,
            neighbor_index=neighbor_index,
            w1=0.5,
            w4=0.05,
            w5=0.1,
            stop_lambda=0.2,
            stop_top_k=2,
            shape_prior_model=shape_model,
            shape_prior_weight=0.65,
            shape_prior_mode="mixture",
            shape_prior_normalize_over_frontier=True,
            shape_prior_clip=5.0,
        )
        patch = self.patch_training.PatchContext(
            patch_id="p_shape_inside",
            cells=(ctx,),
            core_cell_ids=("a",),
            margin_cell_ids=(),
            outer_bounds=self.patch_training.PatchBounds(-5, 8, -5, 8),
            core_bounds=self.patch_training.PatchBounds(-5, 8, -5, 8),
            max_steps=5,
        )
        legacy = self.patch_training.MultiCellPatchEnv(patch)
        torch_env = self.patch_training.TorchPatchEnv(patch, device=torch.device("cpu"))
        legacy_obs, _ = legacy.reset()
        torch_obs, _ = torch_env.reset()
        inside = self.patch_training._torch_grid_cells_inside_hull(
            torch_env._shape_grid_coords[0][8:9],
            torch_env._shape_states[0].hull_equations,
            dtype=torch.float64,
            epsilon=1.0e-8,
        )
        self.assertTrue(bool(inside[0].detach().cpu().item()))
        np.testing.assert_allclose(
            _obs_array(torch_obs["global_features"], np.float32),
            legacy_obs["global_features"],
            rtol=1.0e-4,
            atol=1.0e-4,
        )
        np.testing.assert_allclose(
            _obs_array(torch_obs["action_features"], np.float32),
            legacy_obs["action_features"],
            rtol=1.0e-4,
            atol=1.0e-4,
        )
        np.testing.assert_array_equal(_obs_array(torch_obs["action_mask"], bool), legacy_obs["action_mask"])


if __name__ == "__main__":
    unittest.main()
