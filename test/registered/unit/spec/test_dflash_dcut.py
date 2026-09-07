"""Unit tests for DFLASH D-Cut selection, PP cycle cost, and materialization."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.dflash_dcut import (
    DFlashDcutEpilogue,
    DFlashDcutPlanner,
    get_dflash_dcut_keep_count,
    pp_pipeline_cycle_cost,
    pp_pipeline_flowshop_makespan,
    score_dcut_candidates,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestPipelineCycleCost(CustomTestCase):
    def test_forward_pipeline_includes_fill_and_drain(self):
        stage_costs = torch.tensor([[10.0, 15.0], [20.0, 25.0]])
        torch.testing.assert_close(
            pp_pipeline_cycle_cost(stage_costs, microbatch_count=2),
            torch.tensor([50.0, 65.0]),
        )

    def test_flowshop_makespan_accounts_for_job_order_and_transfer(self):
        self.assertEqual(
            pp_pipeline_flowshop_makespan(
                jobs=((10.0, 20.0), (15.0, 10.0)),
            ),
            40.0,
        )
        self.assertEqual(
            pp_pipeline_flowshop_makespan(
                jobs=((10.0, 20.0), (15.0, 10.0)),
                transfer_costs=(5.0,),
            ),
            45.0,
        )


class TestScoreDcutCandidates(CustomTestCase):
    def test_flat_curve_argmax_is_full_width(self):
        # Diminishing expected drafts: extra tokens barely add accept.
        expected = torch.tensor([7.1, 10.9, 12.2, 12.6])
        costs = torch.tensor([25.8505, 26.4847, 27.0244, 27.3242])
        scores = score_dcut_candidates(expected=expected, costs=costs)
        self.assertEqual(int(torch.argmax(scores).item()), 3)

    def test_steep_curve_prefers_a_compact_ratio(self):
        expected = torch.tensor([115.2, 172.8, 185.6, 193.9])
        costs = torch.tensor([80.88, 107.61, 136.64, 166.34])
        scores = score_dcut_candidates(expected=expected, costs=costs)
        self.assertEqual(int(torch.argmax(scores).item()), 1)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            score_dcut_candidates(
                expected=torch.ones(3), costs=torch.ones(4)
            )


class TestDcutProfileResolution(CustomTestCase):
    @staticmethod
    def _planner() -> DFlashDcutPlanner:
        planner = object.__new__(DFlashDcutPlanner)
        planner.value = "auto"
        planner.block_size = 4
        planner.gamma = 3
        planner._offline_keep_counts = {}
        planner._graph_num_tokens = lambda total: total
        return planner

    def test_sparse_graph_table_reuses_nearest_candidate(self):
        planner = self._planner()
        planner._costs_by_bs = {2: [10.0, 20.0, 30.0, 40.0]}

        costs = planner._profile_costs_for_bs(5)

        self.assertEqual(costs, [10.0, 20.0, 30.0, 40.0])

    def test_sparse_pp_stage_table_reuses_nearest_candidate(self):
        planner = self._planner()
        planner._stage_costs_by_bs = {
            2: ((1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0))
        }

        stage_costs = planner._profile_stage_costs_for_bs(5)

        self.assertEqual(
            stage_costs,
            ((1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)),
        )

    def test_runtime_cost_ema_only_raises_profiled_cost(self):
        planner = self._planner()
        planner._runtime_stage_cost_ema = {}

        planner.observe_runtime_stage_cost(
            bs=4, candidate_index=1, cost_ms=10.0
        )
        planner.observe_runtime_stage_cost(
            bs=4, candidate_index=1, cost_ms=20.0, smoothing=0.5
        )

        self.assertEqual(planner._runtime_cost_for_candidate(4, 1), 15.0)


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

    def test_compact_overhead_profile_accepts_string_device(self):
        planner = object.__new__(DFlashDcutPlanner)
        planner.device = "cpu"
        planner.gamma = 15

        self.assertEqual(
            planner._profile_compact_overhead_ms(bs=2, keep_count=1), 0.0
        )


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
