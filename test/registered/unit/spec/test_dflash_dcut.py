"""Unit tests for DFLASH D-Cut dense-skip, graph-bucket fill, and scoring."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.dflash_dcut import (
    DFlashDcutEpilogue,
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


class TestAcceptedPrefixMaterialization(CustomTestCase):
    def test_graph_epilogue_only_scatter_top1(self):
        epilogue = DFlashDcutEpilogue(
            max_bs=2, block_size=4, device=torch.device("cpu")
        )
        epilogue.begin_step(torch.tensor([2, 4], dtype=torch.int32))
        logits = torch.randn(6, 8)

        with patch(
            "sglang.srt.speculative.dflash_dcut.scatter_compact_to_strided_into"
        ) as scatter:
            epilogue(compact_logits=logits, bs=2)

        self.assertEqual(scatter.call_count, 1)
        self.assertEqual(tuple(scatter.call_args.kwargs["compact"].shape), (6, 1))

    def test_pack_preserves_row_major_prefix_order(self):
        from sglang.srt.speculative.dflash_worker_v2 import (
            _pack_accepted_compact_rows,
        )

        bs, block_size, hidden_size = 3, 4, 2
        verify_lens = torch.tensor([2, 4, 3], dtype=torch.int32)
        compact_hidden = torch.arange(
            verify_lens.sum().item() * hidden_size, dtype=torch.float32
        ).view(-1, hidden_size)
        positions = torch.arange(100, 100 + bs * block_size, dtype=torch.int64)
        cache_loc_2d = torch.arange(200, 200 + bs * block_size, dtype=torch.int64).view(
            bs, block_size
        )
        commit_lens = torch.tensor([1, 3, 2], dtype=torch.int32)

        packed_hidden, packed_locs, packed_positions = _pack_accepted_compact_rows(
            compact_hidden=compact_hidden,
            positions=positions,
            cache_loc_2d=cache_loc_2d,
            verify_lens=verify_lens,
            commit_lens=commit_lens,
            block_size=block_size,
        )

        expected_compact_rows = torch.tensor([0, 2, 3, 4, 6, 7], dtype=torch.int64)
        expected_dense_rows = torch.tensor([0, 4, 5, 6, 8, 9], dtype=torch.int64)
        torch.testing.assert_close(
            packed_hidden,
            compact_hidden.index_select(0, expected_compact_rows),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            packed_locs,
            cache_loc_2d.reshape(-1).index_select(0, expected_dense_rows),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            packed_positions,
            positions.index_select(0, expected_dense_rows),
            rtol=0,
            atol=0,
        )

    def test_worker_projects_only_packed_rows(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        class RecordingDraft:
            def __init__(self):
                self.projected = None

            def project_target_hidden(self, hidden):
                self.projected = hidden.clone()
                return hidden

        class RecordingMaterializer:
            def __init__(self):
                self.hidden = None
                self.positions = None

            def materialize(self, *, ctx_hidden, positions, write_layer_kv):
                del write_layer_kv
                self.hidden = ctx_hidden.clone()
                self.positions = positions.clone()

        draft = RecordingDraft()
        materializer = RecordingMaterializer()
        worker = object.__new__(DFlashWorkerV2)
        worker.model_runner = SimpleNamespace(device=torch.device("cpu"))
        worker.draft_model = draft
        worker.draft_model_runner = SimpleNamespace(token_to_kv_pool=None)
        worker.block_size = 4
        worker._block_pos_offsets = torch.arange(4, dtype=torch.int64)
        worker._use_fused_kv_materialize = True
        worker._fused_kv_helper = materializer

        bs, block_size, hidden_size = 2, 4, 3
        verify_lens = torch.tensor([2, 3], dtype=torch.int32)
        compact_hidden = torch.arange(
            verify_lens.sum().item() * hidden_size, dtype=torch.float32
        ).view(-1, hidden_size)
        positions = torch.arange(bs * block_size, dtype=torch.int64)
        cache_loc_2d = torch.arange(50, 50 + bs * block_size, dtype=torch.int64).view(
            bs, block_size
        )
        commit_lens = torch.tensor([1, 2], dtype=torch.int32)

        DFlashWorkerV2._append_compact_target_hidden_to_draft_kv_by_loc(
            worker,
            compact_hidden=compact_hidden,
            positions=positions,
            cache_loc_2d=cache_loc_2d,
            verify_lens=verify_lens,
            commit_lens=commit_lens,
        )

        expected_compact_rows = torch.tensor([0, 2, 3], dtype=torch.int64)
        self.assertEqual(tuple(draft.projected.shape), (3, hidden_size))
        torch.testing.assert_close(
            draft.projected,
            compact_hidden.index_select(0, expected_compact_rows),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            materializer.positions,
            positions.index_select(0, torch.tensor([0, 4, 5])),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
