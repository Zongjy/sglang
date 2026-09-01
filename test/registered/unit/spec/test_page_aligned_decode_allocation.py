import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocation_sizing import page_aligned_decode_alloc_lens
from sglang.srt.speculative.dflash_info_v2 import DFlashPPVerifyInputRaw
from sglang.srt.speculative.dspark_components.dspark_verify import (
    DSparkPPVerifyInputRaw,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _req(*, committed: int, allocated: int):
    return SimpleNamespace(
        kv_committed_len=committed,
        kv=SimpleNamespace(kv_allocated_len=allocated),
    )


class TestPageAlignedDecodeAllocation(CustomTestCase):
    def test_rounds_up_and_never_shrinks_existing_watermark(self):
        reqs = [
            _req(committed=65, allocated=64),
            _req(committed=100, allocated=192),
        ]

        cur, nxt, needed = page_aligned_decode_alloc_lens(
            reqs, reserve=16, page_size=64
        )

        self.assertEqual(cur, [64, 192])
        self.assertEqual(nxt, [128, 192])
        self.assertEqual(needed, 64)

    def test_shared_pp_prepare_keeps_reserved_relay_contract(self):
        reqs = [
            _req(committed=65, allocated=64),
            _req(committed=100, allocated=192),
        ]
        batch = SimpleNamespace(
            device=torch.device("cpu"),
            batch_size=lambda: len(reqs),
            reqs=reqs,
            token_to_kv_pool_allocator=SimpleNamespace(page_size=64),
            tree_cache=object(),
            req_to_token_pool=object(),
            req_pool_indices=torch.arange(len(reqs), dtype=torch.int64),
            seq_lens_cpu=None,
            seq_lens_sum=0,
        )
        pp_inputs = [
            DFlashPPVerifyInputRaw(
                bonus_tokens=torch.tensor([1, 2]),
                draft_tokens=torch.ones((2, 7), dtype=torch.int64),
                accept_lens=torch.ones(2, dtype=torch.int64),
            ),
            DSparkPPVerifyInputRaw(
                bonus_tokens=torch.tensor([1, 2]),
                draft_tokens=torch.ones((2, 7), dtype=torch.int64),
                new_seq_lens=[65, 100],
                accept_lens=torch.ones(2, dtype=torch.int64),
            ),
        ]

        for pp_input in pp_inputs:
            with self.subTest(type=type(pp_input).__name__), patch(
                "sglang.srt.speculative.dflash_info_v2.get_spec",
                return_value=SimpleNamespace(speculative_num_draft_tokens=8),
            ), patch(
                "sglang.srt.speculative.dflash_info_v2._get_overlap_plan_stream",
                return_value=(None, contextlib.nullcontext()),
            ), patch(
                "sglang.srt.speculative.dflash_info_v2.alloc_for_spec_decode"
            ) as alloc:
                pp_input.prepare_for_decode(batch)

            torch.testing.assert_close(
                pp_input.reserved_seq_lens_cpu,
                torch.tensor([128, 192], dtype=torch.int32),
            )
            self.assertEqual(pp_input.reserved_seq_lens_sum, 320)
            self.assertEqual(alloc.call_args.kwargs["num_needed_tokens"], 64)


if __name__ == "__main__":
    unittest.main()
