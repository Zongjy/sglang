"""Unit tests for DFLASH D-Cut selection, PP cycle cost, and materialization."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.dflash_dcut import (
    DFlashDcutEpilogue,
    DFlashDcutPlanner,
    _DcutPipelineSlot,
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


class TestDcutPinFullGate(CustomTestCase):
    """Pin-full gate: batches whose profiled savings cannot pay for the
    selector are pinned to full-width verify without running selection."""

    @staticmethod
    def _planner(costs_by_bs, threshold_ms=5.0) -> DFlashDcutPlanner:
        from sglang.srt.environ import envs

        planner = object.__new__(DFlashDcutPlanner)
        planner.value = "auto"
        planner.block_size = 16
        planner.gamma = 15
        planner.device = torch.device("cpu")
        planner.tp_rank = 0
        planner.pp_group = SimpleNamespace(rank_in_group=0, world_size=1)
        planner._costs_by_bs = costs_by_bs
        planner._pin_full_max_bs = 0
        with envs.SGLANG_DFLASH_DCUT_PIN_FULL_MIN_SAVINGS_MS.override(
            threshold_ms
        ):
            planner._compute_pin_full_max_bs()
        return planner

    # Spreads mirror the Qwen3.5-9B PP2 startup table: small batches save
    # less than the selector overhead, large batches save far more.
    _COSTS = {
        4: [13.0, 14.5, 15.7, 16.4],  # spread 3.4
        8: [15.0, 16.8, 17.8, 18.5],  # spread 3.5
        16: [18.9, 20.5, 22.7, 24.2],  # spread 5.3
        32: [25.0, 28.8, 37.7, 43.2],  # spread 18.2
    }

    def test_pins_exactly_the_below_threshold_prefix(self):
        planner = self._planner(self._COSTS, threshold_ms=5.0)
        self.assertEqual(planner._pin_full_max_bs, 8)
        self.assertTrue(planner.is_full_pinned(1))
        self.assertTrue(planner.is_full_pinned(8))
        self.assertFalse(planner.is_full_pinned(9))
        self.assertFalse(planner.is_full_pinned(32))

    def test_threshold_covers_next_bucket(self):
        planner = self._planner(self._COSTS, threshold_ms=5.5)
        self.assertEqual(planner._pin_full_max_bs, 16)

    def test_prefix_stops_at_first_unpinned_bucket(self):
        # A later cheap bucket must not extend the prefix past an expensive one.
        costs = {4: [10.0, 20.0, 30.0, 40.0], 8: list(self._COSTS[8])}
        planner = self._planner(costs, threshold_ms=5.0)
        self.assertEqual(planner._pin_full_max_bs, 0)

    def test_incomplete_bucket_does_not_break_prefix(self):
        costs = {
            4: [13.0, 14.5, 15.7, 16.4],
            6: [None, None, None, None],
            8: [15.0, 16.8, 17.8, 18.5],
            16: [18.9, 20.5, 22.7, 24.2],
        }
        planner = self._planner(costs, threshold_ms=5.0)
        self.assertEqual(planner._pin_full_max_bs, 8)

    def test_non_positive_threshold_disables_pinning(self):
        planner = self._planner(self._COSTS, threshold_ms=0.0)
        self.assertEqual(planner._pin_full_max_bs, 0)
        self.assertFalse(planner.is_full_pinned(1))

    def test_fixed_ratio_mode_never_pins(self):
        planner = self._planner(self._COSTS, threshold_ms=100.0)
        planner.value = 0.5
        from sglang.srt.environ import envs

        with envs.SGLANG_DFLASH_DCUT_PIN_FULL_MIN_SAVINGS_MS.override(100.0):
            planner._compute_pin_full_max_bs()
        self.assertEqual(planner._pin_full_max_bs, 0)
        self.assertFalse(planner.is_full_pinned(4))

    def test_pinned_full_plan_publishes_full_budget(self):
        planner = self._planner(self._COSTS, threshold_ms=5.0)
        planner._graph_num_tokens = lambda total: total
        layout = object()
        with patch.object(
            RaggedVerifyLayout,
            "from_verify_lens_device",
            return_value=layout,
        ):
            plan = planner.pinned_full_plan(bs=4)
        self.assertIs(plan.layout, layout)
        self.assertEqual(plan.keep_count, 4 * 15)
        self.assertFalse(plan.is_compact)
        # The DP tier alignment reads the last candidate as the keep budget;
        # the pinned path must publish the full-width candidate.
        self.assertEqual(plan.candidate_index, 3)
        self.assertEqual(planner.last_candidate_index, 3)
        self.assertEqual(planner.current_keep_budget(4), 4 * 15)


class TestHostSidePipelineSelection(CustomTestCase):
    """The PP per-microbatch path resolves the candidate on the host."""

    @staticmethod
    def _planner() -> DFlashDcutPlanner:
        planner = object.__new__(DFlashDcutPlanner)
        planner.value = "auto"
        planner.block_size = 4
        planner.gamma = 3
        planner.device = torch.device("cpu")
        planner.tp_rank = 0
        planner.pp_group = SimpleNamespace(world_size=2)
        planner.pp_microbatch_count = 2
        planner._auto_index_device = torch.zeros((), dtype=torch.int64)
        planner._warned_missing_profile_bs = set()
        planner._graph_num_tokens = lambda total: total
        planner._offline_keep_counts = {}
        planner._costs_by_bs = {4: [40.0, 54.0, 70.0, 84.0]}
        planner._stage_costs_by_bs = {4: ((18.0, 25.0, 32.0, 39.0), (14.0, 19.0, 25.0, 30.0))}
        planner._stage_fold_costs_by_bs = {4: ((1.0,) * 4, (1.5,) * 4)}
        planner._stage_overhead_costs_by_bs = {}
        planner._stage_draft_costs_by_bs = {4: ((0.0,) * 4, (5.0,) * 4)}
        planner._overhead_costs_by_bs = {}
        planner._cost_tensors_by_bs = {}
        planner._runtime_stage_cost_ema = {}
        planner._pp_pipeline_slots = {}
        planner._pp_pipeline_slot_seen = {}
        planner._pp_pipeline_selection_step = 0
        planner._flowshop_makespan_cache = {}
        planner._full_hold_bs = None
        planner._full_hold_remaining = 0
        planner._pipeline_transfer_costs = (0.0,)
        planner._pipeline_transfer_models = ()
        planner._select_group = SimpleNamespace(world_size=1)
        return planner

    def _reference_selection(self, planner, confidence, bs, other_slots):
        """Straightforward reimplementation of the expected/makespan argmax."""
        survival = torch.cumprod(confidence.to(torch.float32), dim=1).flatten()
        prefix = torch.cumsum(torch.sort(survival, descending=True).values, dim=0)
        keep_counts = planner.candidate_keep_counts(bs)
        expected = [
            (float(prefix[k - 1]) + float(bs)) if k > 0 else float(bs)
            for k in keep_counts
        ]
        stage_rows = planner._stage_cost_rows_host(
            bs=bs,
            stage_costs=planner._stage_costs_by_bs[bs],
            stage_fold_costs=planner._stage_fold_costs_by_bs[bs],
            stage_overhead_costs=None,
            stage_draft_costs=planner._stage_draft_costs_by_bs[bs],
            overhead_costs=None,
        )
        scores = []
        for candidate_index in range(4):
            current = tuple(row[candidate_index] for row in stage_rows)
            jobs, transfers, total_expected = [], [], 0.0
            for slot_id in (0, 1):
                slot = other_slots.get(slot_id)
                if slot_id == 0 or slot is None:  # 0 is the deciding slot here
                    jobs.append(current)
                    transfers.append((0.0,))
                    total_expected += expected[candidate_index]
                else:
                    jobs.append(slot.stage_costs)
                    transfers.append(slot.transfer_costs)
                    total_expected += slot.expected_tokens
            makespan = pp_pipeline_flowshop_makespan(
                jobs,
                transfer_costs=planner._pipeline_transfer_costs,
                job_transfer_costs=transfers,
            )
            scores.append(total_expected / max(makespan, 1e-6))
        return max(range(4), key=lambda i: scores[i])

    def test_host_selection_matches_reference_without_device_sync(self):
        planner = self._planner()
        confidence = torch.tensor(
            [[0.9, 0.8, 0.3], [0.4, 0.2, 0.1], [0.95, 0.9, 0.85], [0.1, 0.1, 0.1]]
        )
        selected = planner._write_auto_index_local(
            confidence=confidence, bs=4, pipeline_mb_id=0
        )

        reference = self._reference_selection(planner, confidence, 4, {})
        self.assertEqual(selected, reference)
        # Host-resolved: the device index buffer is untouched for a trivial
        # select group (no fill_, no broadcast, no .item()).
        self.assertEqual(int(planner._auto_index_device.item()), 0)

    def test_host_selection_uses_other_slot_state(self):
        planner = self._planner()
        confidence = torch.full((4, 3), 0.7)
        other = {
            1: _DcutPipelineSlot(
                stage_costs=(100.0, 100.0),
                expected_tokens=9.0,
                transfer_costs=(0.0,),
            )
        }
        planner._pp_pipeline_slots[1] = other[1]
        planner._pp_pipeline_slot_seen[1] = 0

        selected = planner._write_auto_index_local(
            confidence=confidence, bs=4, pipeline_mb_id=0
        )

        reference = self._reference_selection(planner, confidence, 4, other)
        self.assertEqual(selected, reference)

    def test_runtime_ema_raises_candidate_cost_on_host_path(self):
        planner = self._planner()
        # Make candidate 0 look artificially expensive: it must not win.
        planner._runtime_stage_cost_ema = {(4, 0): 10_000.0}
        confidence = torch.full((4, 3), 0.05)

        selected = planner._write_auto_index_local(
            confidence=confidence, bs=4, pipeline_mb_id=0
        )

        self.assertNotEqual(selected, 0)

    def test_makespan_cache_reuse_and_invalidation(self):
        planner = self._planner()
        expected = [8.0, 10.0, 11.0, 12.0]
        candidate_stage_costs = tuple(
            tuple(row[i] for row in ((18.0, 25.0, 32.0, 39.0), (14.0, 19.0, 25.0, 30.0)))
            for i in range(4)
        )
        candidate_transfer_costs = ((0.0,),) * 4
        kwargs = dict(
            expected=expected,
            candidate_stage_costs=candidate_stage_costs,
            candidate_transfer_costs=candidate_transfer_costs,
            bs=4,
            pipeline_mb_id=0,
        )

        first = planner._select_pipeline_candidate(**kwargs)
        self.assertEqual(len(planner._flowshop_makespan_cache), 1)
        # Same inputs again: the deciding slot's own previous record must not
        # invalidate the cache.
        second = planner._select_pipeline_candidate(**kwargs)
        self.assertEqual(second, first)
        self.assertEqual(len(planner._flowshop_makespan_cache), 1)
        # A change in expected scores alone still hits the makespan cache.
        third = planner._select_pipeline_candidate(
            **{**kwargs, "expected": [8.0, 20.0, 40.0, 60.0]}
        )
        self.assertEqual(len(planner._flowshop_makespan_cache), 1)
        self.assertNotEqual(third, first)
        # A decision on the sibling slot adds it to later keys.
        planner._select_pipeline_candidate(**{**kwargs, "pipeline_mb_id": 1})
        planner._select_pipeline_candidate(**kwargs)
        self.assertGreaterEqual(len(planner._flowshop_makespan_cache), 3)

    def test_select_auto_candidate_skips_broadcast_for_trivial_group(self):
        planner = self._planner()

        def _forbidden_broadcast(*args, **kwargs):
            raise AssertionError("broadcast must not run for a trivial group")

        planner._select_group = SimpleNamespace(
            world_size=1, rank_in_group=0, broadcast=_forbidden_broadcast
        )
        confidence = torch.tensor(
            [[0.9, 0.8, 0.3], [0.4, 0.2, 0.1], [0.95, 0.9, 0.85], [0.1, 0.1, 0.1]]
        )

        selected = planner._select_auto_candidate(
            confidence=confidence, bs=4, pipeline_mb_id=0
        )

        self.assertEqual(
            selected, self._reference_selection(planner, confidence, 4, {})
        )

    def test_select_auto_candidate_broadcasts_host_choice_for_groups(self):
        from unittest.mock import MagicMock

        planner = self._planner()
        group = SimpleNamespace(world_size=2, rank_in_group=0, broadcast=MagicMock())
        planner._select_group = group
        confidence = torch.full((4, 3), 0.7)

        selected = planner._select_auto_candidate(
            confidence=confidence, bs=4, pipeline_mb_id=0
        )

        group.broadcast.assert_called_once()
        self.assertEqual(int(planner._auto_index_device.item()), selected)
        self.assertEqual(
            selected, self._reference_selection(planner, confidence, 4, {})
        )


class TestDcutCostTableLoad(CustomTestCase):
    """Loading an offline-profiled D-Cut cost table (auto mode)."""

    @staticmethod
    def _planner() -> DFlashDcutPlanner:
        planner = object.__new__(DFlashDcutPlanner)
        planner.value = "auto"
        planner.block_size = 16
        planner.gamma = 15
        planner.device = torch.device("cpu")
        planner.tp_rank = 0
        planner.tp_group = SimpleNamespace(world_size=1, cpu_group=None)
        planner.pp_group = SimpleNamespace(world_size=2, rank_in_group=0, cpu_group=None)
        # Skip the collective capture-grid init; the loader only needs it
        # resolved, and unit tests have no process groups.
        planner._common_capture_num_tokens = (16, 32)
        planner._costs_by_bs = {}
        planner._stage_costs_by_bs = {}
        planner._offline_keep_counts = {}
        planner._offline_profiled = False
        planner._pin_full_max_bs = 0
        planner._warned_missing_profile_bs = set()
        return planner

    @staticmethod
    def _load(planner: DFlashDcutPlanner, path: str, **kwargs) -> None:
        def _fake_gather(gathered, obj, group=None):
            for index in range(len(gathered)):
                gathered[index] = obj

        with patch("torch.distributed.all_gather_object", _fake_gather):
            planner.load_dcut_cost_table(path, **kwargs)

    # Column order follows the file's "ratios" array, deliberately shuffled.
    _TABLE = {
        "version": 1,
        "block_size": 16,
        "pp_size": 2,
        "pp_layer_partition": [20, 12],
        "ratios": [1.0, 0.25, 0.5, 0.75],
        "buckets": {
            "4": {
                "stage_costs_ms": [[16.4, 13.0, 14.5, 15.7], [17.4, 14.0, 15.5, 16.7]]
            },
            "8": {
                "stage_costs_ms": [[18.5, 15.0, 16.8, 17.8], [19.5, 16.0, 17.8, 18.8]]
            },
        },
    }

    def _write_table(self, payload) -> str:
        import tempfile

        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        import json

        json.dump(payload, handle)
        handle.close()
        return handle.name

    def test_load_reorders_columns_and_fills_tables(self):
        planner = self._planner()
        path = self._write_table(self._TABLE)

        self._load(planner, path, expected_partition=(20, 12))

        # Columns are reordered into the _AUTO_RATIOS order (0.25/0.5/0.75/1.0).
        self.assertEqual(
            planner._stage_costs_by_bs[4],
            ((13.0, 14.5, 15.7, 16.4), (14.0, 15.5, 16.7, 17.4)),
        )
        # Step costs default to the per-candidate bottleneck stage.
        self.assertEqual(planner._costs_by_bs[4], [14.0, 15.5, 16.7, 17.4])
        self.assertEqual(planner._offline_keep_counts[4], (12, 28, 44, 60))
        self.assertTrue(planner._offline_profiled)
        # The pin-full gate is computed from the loaded table: both buckets
        # have spreads below the default 5 ms threshold.
        self.assertEqual(planner._pin_full_max_bs, 8)
        self.assertTrue(planner.is_full_pinned(8))
        self.assertFalse(planner.is_full_pinned(9))

    def test_explicit_step_costs_are_respected(self):
        import copy

        planner = self._planner()
        payload = copy.deepcopy(self._TABLE)
        payload["buckets"]["4"]["step_costs_ms"] = [100.0, 50.0, 60.0, 70.0]
        self._load(planner, self._write_table(payload))

        self.assertEqual(planner._costs_by_bs[4], [50.0, 60.0, 70.0, 100.0])

    def test_load_is_idempotent(self):
        planner = self._planner()
        self._load(planner, self._write_table(self._TABLE))
        planner.load_dcut_cost_table(self._write_table({"version": 2}))

        self.assertEqual(len(planner._costs_by_bs), 2)

    def test_fixed_ratio_mode_skips_loading(self):
        planner = self._planner()
        planner.value = 0.5
        self._load(planner, self._write_table(self._TABLE))

        self.assertTrue(planner._offline_profiled)
        self.assertEqual(planner._costs_by_bs, {})

    def test_validation_errors(self):
        cases = [
            {"version": 2},
            {"block_size": 8},
            {"pp_size": 1},
            {"ratios": [0.25, 0.5, 1.0]},
            {"buckets": {}},
            {"buckets": {"4": {"stage_costs_ms": [[1.0, 2.0, 3.0, 4.0]]}}},
            {
                "buckets": {
                    "4": {
                        "stage_costs_ms": [
                            [16.4, 13.0, 14.5, -15.7],
                            [17.4, 14.0, 15.5, 16.7],
                        ]
                    }
                }
            },
        ]
        for patch in cases:
            import copy

            planner = self._planner()
            payload = copy.deepcopy(self._TABLE)
            payload.update(patch)
            with self.assertRaisesRegex(RuntimeError, "invalid D-Cut cost table"):
                self._load(
                    planner, self._write_table(payload), expected_partition=(20, 12)
                )

    def test_partition_mismatch_is_rejected(self):
        planner = self._planner()
        with self.assertRaisesRegex(RuntimeError, "pp_layer_partition mismatch"):
            self._load(
                planner, self._write_table(self._TABLE), expected_partition=(22, 10)
            )


if __name__ == "__main__":
    unittest.main()
