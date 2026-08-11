from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hd_cell_rl.ppo_state import EpisodeContext, _compute_state_feature_bundle
from hd_cell_rl.shape_prior import (
    cluster_reference_shapes,
    compute_delta_shape_rewards_for_candidates,
    compute_delta_shape_reward,
    compute_shape_reward,
    fit_shape_cluster_distributions,
    gaussian_log_likelihood,
    load_shape_prior_model,
    resolve_shape_feature_matrix,
    save_shape_prior_model,
    score_shape_against_clusters,
    select_best_cluster_number,
    shape_features_from_grid_coords,
)
from hd_cell_rl.reward import build_eight_neighbor_index


class ShapePriorTests(unittest.TestCase):
    def test_resolves_existing_project_feature_columns(self) -> None:
        df = pd.DataFrame(
            {
                "feature_log_area": [0.0, 1.0],
                "feature_compactness": [0.0, -1.0],
                "feature_solidity": [0.5, -0.5],
                "feature_anisotropy": [0.1, -0.1],
            }
        )
        matrix = resolve_shape_feature_matrix(df)
        self.assertEqual(matrix.feature_names, ("log_area", "compactness", "solidity", "anisotropy"))
        self.assertTrue(matrix.zscored_input)
        self.assertEqual(matrix.values.shape, (2, 4))

    def test_raw_features_are_zscored_when_needed(self) -> None:
        df = pd.DataFrame(
            {
                "log_area": [1.0, 2.0, 3.0],
                "compactness": [0.2, 0.3, 0.4],
                "solidity": [0.8, 0.9, 1.0],
                "anisotropy": [1.0, 2.0, 3.0],
            }
        )
        matrix = resolve_shape_feature_matrix(df)
        self.assertFalse(matrix.zscored_input)
        self.assertTrue(np.allclose(matrix.values.mean(axis=0), 0.0, atol=1e-7))
        explicit = resolve_shape_feature_matrix(df, feature_cols=["log_area", "compactness", "solidity", "anisotropy"])
        self.assertFalse(explicit.zscored_input)
        self.assertTrue(np.allclose(explicit.values.mean(axis=0), 0.0, atol=1e-7))

    def test_clustering_and_shape_model_scoring(self) -> None:
        df = pd.DataFrame(
            {
                "feature_log_area": [-1.1, -0.9, -1.0, 0.9, 1.0, 1.1],
                "feature_compactness": [1.0, 1.1, 0.9, -1.0, -0.9, -1.1],
                "feature_solidity": [0.9, 1.0, 1.1, -0.8, -0.9, -1.0],
                "feature_anisotropy": [-0.5, -0.4, -0.6, 0.5, 0.4, 0.6],
            }
        )
        clustered = cluster_reference_shapes(df, n_clusters=2, random_state=1)
        self.assertIn("shape_cluster", clustered.columns)
        self.assertEqual(clustered["shape_cluster"].nunique(), 2)

        k_scores = select_best_cluster_number(df, k_range=range(2, 4), random_state=1, sample_size=0)
        self.assertIn("silhouette_score", k_scores.columns)

        model = fit_shape_cluster_distributions(clustered, epsilon=1e-4)
        self.assertEqual(model.n_clusters, 2)
        z = df.iloc[0].to_numpy(dtype=np.float64)
        score = score_shape_against_clusters(z, model)
        self.assertIn(score["best_cluster"], model.cluster_labels)
        self.assertTrue(np.isfinite(score["mixture_log_likelihood"]))
        self.assertTrue(np.isfinite(compute_shape_reward(z, model, mode="mixture")))
        self.assertTrue(np.isfinite(compute_delta_shape_reward(z, z + 0.1, model)))
        self.assertTrue(np.isfinite(gaussian_log_likelihood(z, model.means[0], model.covariances[0])))

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "shape_prior.npz"
            save_shape_prior_model(model, path)
            loaded = load_shape_prior_model(path)
            self.assertEqual(loaded.cluster_labels, model.cluster_labels)
            self.assertTrue(np.allclose(loaded.means, model.means))

    def test_grid_shape_features_use_exact_hull_convention(self) -> None:
        coords = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64)
        features = shape_features_from_grid_coords(coords)
        self.assertAlmostEqual(float(features[0]), float(np.log(5.0)))
        self.assertAlmostEqual(float(features[1]), float(np.pi / 4.0))
        self.assertAlmostEqual(float(features[2]), 1.0)
        self.assertAlmostEqual(float(features[3]), 1.0)

    def test_candidate_shape_delta_matches_full_recompute(self) -> None:
        ref = pd.DataFrame(
            {
                "feature_log_area": [-1.0, -0.8, 0.8, 1.0],
                "feature_compactness": [0.8, 1.0, -0.8, -1.0],
                "feature_solidity": [0.5, 0.6, -0.5, -0.6],
                "feature_anisotropy": [-0.2, -0.1, 0.1, 0.2],
                "shape_cluster": ["cluster_00", "cluster_00", "cluster_01", "cluster_01"],
            }
        )
        model = fit_shape_cluster_distributions(ref, epsilon=1e-3)
        grid = np.asarray(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
                [2, 0],
                [2, 1],
                [1, 2],
                [3, 0],
            ],
            dtype=np.int64,
        )
        xy = (grid * 2.0).astype(np.float32)
        mask = np.asarray([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.uint8)
        candidates = np.asarray([3, 4, 5, 6, 7], dtype=np.int64)

        fast, current_raw, current_reward = compute_delta_shape_rewards_for_candidates(
            membership_mask=mask,
            candidate_indices=candidates,
            candidate_bin_xy_um=xy,
            shape_model=model,
            mode="mixture",
            bin_size_um=2.0,
        )
        current_z = model.transform_raw_features(current_raw)
        self.assertAlmostEqual(float(compute_shape_reward(current_z, model)), float(current_reward), places=6)

        current_coords = grid[mask.astype(bool)]
        expected = np.zeros((grid.shape[0],), dtype=np.float32)
        for idx in candidates.tolist():
            after_coords = np.unique(np.vstack((current_coords, grid[idx].reshape(1, 2))), axis=0)
            after_raw = shape_features_from_grid_coords(after_coords)
            after_z = model.transform_raw_features(after_raw)
            expected[idx] = np.float32(compute_shape_reward(after_z, model) - current_reward)

        np.testing.assert_allclose(fast[candidates], expected[candidates], rtol=1e-5, atol=1e-5)

    def test_ppo_add_reward_includes_shape_weight(self) -> None:
        ref = pd.DataFrame(
            {
                "feature_log_area": [-1.0, -0.8, 0.8, 1.0],
                "feature_compactness": [0.8, 1.0, -0.8, -1.0],
                "feature_solidity": [0.5, 0.6, -0.5, -0.6],
                "feature_anisotropy": [-0.2, -0.1, 0.1, 0.2],
                "shape_cluster": ["cluster_00", "cluster_00", "cluster_01", "cluster_01"],
            }
        )
        model = fit_shape_cluster_distributions(ref, epsilon=1e-3)
        xy = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [4.0, 0.0]], dtype=np.float32)
        bin_ids = tuple(f"s_002um_0000{i}_00000-1" for i in range(xy.shape[0]))
        neighbor_index = build_eight_neighbor_index(bin_ids, xy).astype(np.int32)
        base_kwargs = dict(
            cell_id="cell1",
            candidate_bin_ids=bin_ids,
            initial_membership_mask=np.asarray([1, 0, 0, 0], dtype=np.uint8),
            candidate_bin_xy_um=xy,
            nucleus_center_xy_um=np.asarray([0.0, 0.0], dtype=np.float32),
            ll=np.zeros((4, 2), dtype=np.float32),
            p_dis=np.zeros((4,), dtype=np.float32),
            p_overlap=np.zeros((4,), dtype=np.float32),
            ll_mean_z=np.zeros((4,), dtype=np.float32),
            ll_max_z=np.zeros((4,), dtype=np.float32),
            base_penalty=np.zeros((4,), dtype=np.float32),
            expression_confidence=np.ones((4,), dtype=np.float32),
            bin_count_totals=np.ones((4,), dtype=np.float32),
            neighbor_index=neighbor_index,
            max_steps=10,
            log_prior=-np.log(2.0),
            r_max_um=20.0,
            w1=0.0,
            w2=1.0,
            w3=1.0,
            w4=0.0,
            w5=0.0,
            stop_lambda=1.0,
            stop_stat="topk_mean",
            stop_top_k=2,
            expression_confidence_pseudocount=5.0,
            normalize_expression_zscore=False,
            zscore_delta=1e-8,
            shape_prior_model=model,
            shape_prior_mode="mixture",
            shape_prior_reward_mode="delta",
            shape_prior_normalize_over_frontier=False,
            shape_prior_clip=None,
            shape_prior_bin_size_um=2.0,
        )
        ctx_no_shape = EpisodeContext(**base_kwargs, shape_prior_weight=0.0)
        ctx_shape = EpisodeContext(**base_kwargs, shape_prior_weight=1.0)
        mask = np.asarray([1, 0, 0, 0], dtype=np.uint8)
        no_shape = _compute_state_feature_bundle(ctx=ctx_no_shape, membership_mask=mask, step_index=0)
        with_shape = _compute_state_feature_bundle(ctx=ctx_shape, membership_mask=mask, step_index=0)
        diff = with_shape.add_rewards - no_shape.add_rewards
        self.assertTrue(np.any(np.abs(with_shape.shape_raw[with_shape.frontier_mask]) > 0.0))
        self.assertTrue(np.allclose(diff, with_shape.shape_term, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
