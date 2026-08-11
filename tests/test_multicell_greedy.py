from __future__ import annotations

import unittest

import numpy as np

from hd_cell_rl.multicell_greedy import (
    GreedyCellState,
    assignment_rows_for_target,
    run_multicell_greedy_patch,
)
from hd_cell_rl.ppo_state import EpisodeContext


def _ctx(cell_id: str, bin_ids: list[str], seed: list[int]) -> EpisodeContext:
    n_bins = len(bin_ids)
    neighbor_index = np.full((n_bins, 8), -1, dtype=np.int32)
    for idx in range(n_bins - 1):
        neighbor_index[idx, 0] = idx + 1
        neighbor_index[idx + 1, 0] = idx
    return EpisodeContext(
        cell_id=cell_id,
        candidate_bin_ids=tuple(bin_ids),
        initial_membership_mask=np.asarray(seed, dtype=np.uint8),
        candidate_bin_xy_um=np.stack([np.arange(n_bins, dtype=np.float32) * 2.0, np.zeros(n_bins, dtype=np.float32)], axis=1),
        nucleus_center_xy_um=np.asarray([0.0, 0.0], dtype=np.float32),
        ll=np.zeros((n_bins, 2), dtype=np.float32),
        p_dis=np.zeros(n_bins, dtype=np.float32),
        p_overlap=np.zeros(n_bins, dtype=np.float32),
        ll_mean_z=np.zeros(n_bins, dtype=np.float32),
        ll_max_z=np.zeros(n_bins, dtype=np.float32),
        base_penalty=np.zeros(n_bins, dtype=np.float32),
        expression_confidence=np.ones(n_bins, dtype=np.float32),
        bin_count_totals=np.ones(n_bins, dtype=np.float32),
        neighbor_index=neighbor_index,
        max_steps=10,
        log_prior=0.0,
        r_max_um=20.0,
        w1=1.0,
        w2=0.0,
        w3=0.0,
        w4=0.0,
        w5=0.0,
        stop_lambda=0.0,
        stop_stat="max",
        stop_top_k=1,
        expression_confidence_pseudocount=5.0,
        normalize_expression_zscore=True,
        zscore_delta=1.0e-8,
    )


class MultiCellGreedyTests(unittest.TestCase):
    def test_shared_frontier_bin_goes_to_highest_scoring_cell(self) -> None:
        a = GreedyCellState.from_context(_ctx("a", ["a_seed", "shared"], [1, 0]))
        b = GreedyCellState.from_context(_ctx("b", ["b_seed", "shared"], [1, 0]))

        def score_fn(state: GreedyCellState, legal: np.ndarray, step_index: int) -> np.ndarray:
            del legal, step_index
            if state.cell_id == "a":
                return np.asarray([-np.inf, 0.2], dtype=np.float32)
            return np.asarray([-np.inf, 0.9], dtype=np.float32)

        result = run_multicell_greedy_patch(
            target_cell_id="a",
            cell_states=[a, b],
            max_steps=1,
            score_fn=score_fn,
        )

        self.assertEqual(result.owner_by_barcode["shared"], "b")
        self.assertEqual(int(a.membership_mask.sum()), 1)
        self.assertEqual(int(b.membership_mask.sum()), 2)
        self.assertEqual(result.n_contested_candidate_barcodes, 1)

    def test_owned_seed_bin_cannot_be_stolen(self) -> None:
        a = GreedyCellState.from_context(_ctx("a", ["shared", "a_next"], [1, 0]))
        b = GreedyCellState.from_context(_ctx("b", ["b_seed", "shared"], [1, 0]))

        def score_fn(state: GreedyCellState, legal: np.ndarray, step_index: int) -> np.ndarray:
            del state, legal, step_index
            return np.asarray([-np.inf, 10.0], dtype=np.float32)

        result = run_multicell_greedy_patch(
            target_cell_id="a",
            cell_states=[a, b],
            max_steps=1,
            score_fn=score_fn,
        )

        self.assertEqual(result.owner_by_barcode["shared"], "a")
        self.assertEqual(int(b.membership_mask.sum()), 1)
        self.assertGreater(result.n_blocked_frontier_actions, 0)

    def test_assignment_rows_keep_target_nuclear_flags(self) -> None:
        state = GreedyCellState.from_context(
            _ctx("a", ["s_002um_00001_00002-1", "s_002um_00001_00003-1"], [1, 1])
        )
        result = run_multicell_greedy_patch(target_cell_id="a", cell_states=[state], max_steps=0)
        rows = assignment_rows_for_target(result=result, target_state=state)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["array_row"], 1)
        self.assertEqual(rows[0]["array_col"], 2)
        self.assertTrue(rows[0]["is_nuclear"])


if __name__ == "__main__":
    unittest.main()
