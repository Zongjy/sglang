"""Unit tests for DFLASH D-Cut dense-skip, graph-bucket fill, and scoring."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.speculative.dflash_dcut import (
    DFlashDcutPlanner,
    dcut_cost_curve_is_flat,
    fill_dcut_keep_count_to_graph_bucket,
    get_dflash_dcut_keep_count,
    score_dcut_candidates,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


# Measured PP=2 Qwen3.5-27B-FP8 verify costs (ms) from the 20260903_144839 sweep.
_C8_BS4_COSTS = (25.8505, 26.4847, 27.0244, 27.3242)
_C16_BS8_COSTS = (29.59, 30.33, 31.25, 32.07)
_C32_BS16_COSTS = (36.76, 38.47, 40.86, 51.14)
_C128_BS64_COSTS = (80.88, 107.61, 136.64, 166.34)


class TestDcutCostCurveIsFlat(CustomTestCase):
    def test_small_batch_plateau_is_flat(self):
        self.assertTrue(dcut_cost_curve_is_flat(_C8_BS4_COSTS))
        self.assertTrue(dcut_cost_curve_is_flat(_C16_BS8_COSTS))

    def test_compute_cliff_is_not_flat(self):
        self.assertFalse(dcut_cost_curve_is_flat(_C32_BS16_COSTS))
        self.assertFalse(dcut_cost_curve_is_flat(_C128_BS64_COSTS))

    def test_empty_costs_raise(self):
        with self.assertRaises(ValueError):
            dcut_cost_curve_is_flat(())


class TestFillDcutKeepCountToGraphBucket(CustomTestCase):
    def test_already_at_bucket_is_noop(self):
        keep = get_dflash_dcut_keep_count(bs=4, block_size=16, ratio=0.5)
        self.assertEqual(keep, 28)
        self.assertEqual(
            fill_dcut_keep_count_to_graph_bucket(
                bs=4, keep_count=keep, block_size=16, graph_num_tokens=32
            ),
            28,
        )

    def test_fills_leftover_slots_in_the_paid_bucket(self):
        # 4 + 28 = 32 tokens, but the graph rounded up to 48: take the free 16.
        self.assertEqual(
            fill_dcut_keep_count_to_graph_bucket(
                bs=4, keep_count=28, block_size=16, graph_num_tokens=48
            ),
            44,
        )

    def test_does_not_exceed_full_block(self):
        self.assertEqual(
            fill_dcut_keep_count_to_graph_bucket(
                bs=4, keep_count=28, block_size=16, graph_num_tokens=128
            ),
            60,
        )

    def test_non_capture_batch_stays_compact_in_lower_bucket(self):
        keep = get_dflash_dcut_keep_count(bs=11, block_size=16, ratio=0.75)
        filled = fill_dcut_keep_count_to_graph_bucket(
            bs=11, keep_count=keep, block_size=16, graph_num_tokens=160
        )
        self.assertEqual(11 + filled, 160)
        self.assertLess(160, 11 * 16)


class TestScoreDcutCandidates(CustomTestCase):
    def test_flat_curve_argmax_is_full_width(self):
        # Diminishing expected drafts: extra tokens barely add accept.
        expected = torch.tensor([7.1, 10.9, 12.2, 12.6])
        costs = torch.tensor(_C8_BS4_COSTS)
        scores = score_dcut_candidates(expected=expected, costs=costs)
        self.assertEqual(int(torch.argmax(scores).item()), 3)

    def test_steep_curve_prefers_a_compact_ratio(self):
        expected = torch.tensor([115.2, 172.8, 185.6, 193.9])
        costs = torch.tensor(_C128_BS64_COSTS)
        scores = score_dcut_candidates(expected=expected, costs=costs)
        self.assertEqual(int(torch.argmax(scores).item()), 1)

    def test_compact_overhead_breaks_near_ties_toward_dense(self):
        expected = torch.tensor([12.0, 12.2, 12.4, 12.5])
        costs = torch.tensor(_C8_BS4_COSTS)
        scores = score_dcut_candidates(
            expected=expected,
            costs=costs,
            min_relative_save=0.0,
            compact_overhead_ms=3.0,
        )
        self.assertEqual(int(torch.argmax(scores).item()), 3)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            score_dcut_candidates(
                expected=torch.ones(3), costs=torch.ones(4)
            )


class TestDcutRelayPlan(CustomTestCase):
    @staticmethod
    def _planner(expected_graph_num_tokens: int) -> DFlashDcutPlanner:
        # Exercise relay validation without constructing distributed/GPU state.
        planner = object.__new__(DFlashDcutPlanner)
        planner.value = "auto"
        planner.block_size = 16
        planner.gamma = 15
        planner.device = torch.device("cpu")
        planner._graph_num_tokens = lambda total: expected_graph_num_tokens
        return planner

    def test_relay_accepts_token_bucket_above_live_full_width(self):
        planner = self._planner(expected_graph_num_tokens=192)
        verify_lens = torch.full((11,), 16, dtype=torch.int32)
        layout = object()
        with patch.object(
            RaggedVerifyLayout,
            "from_verify_lens_device",
            return_value=layout,
        ):
            plan = planner.plan_from_relay(
                verify_lens=verify_lens,
                keep_count=165,
                graph_num_tokens=192,
                candidate_index=3,
            )

        self.assertIs(plan.layout, layout)
        self.assertFalse(plan.is_compact)

    def test_relay_rejects_stale_token_bucket(self):
        planner = self._planner(expected_graph_num_tokens=192)
        verify_lens = torch.full((11,), 16, dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "expected=192"):
            planner.plan_from_relay(
                verify_lens=verify_lens,
                keep_count=165,
                graph_num_tokens=176,
                candidate_index=3,
            )

    def test_missing_profile_does_not_force_full_width(self):
        planner = object.__new__(DFlashDcutPlanner)
        planner.block_size = 16
        planner._dense_by_bs = {}
        planner._profile_costs_for_bs = lambda bs: None

        self.assertFalse(planner._should_use_dense(11))


if __name__ == "__main__":
    unittest.main()
