import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.spec_tp_sync import (
    SpecTpSync,
    SpecTpSyncSite,
    parse_spec_tp_sync,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSpecTpSync(unittest.TestCase):
    def test_parse_presets_and_site_overrides(self):
        all_sites = frozenset(SpecTpSyncSite)
        self.assertEqual(
            parse_spec_tp_sync("all,-dspark-plan"),
            all_sites - {SpecTpSyncSite.DSPARK_PLAN},
        )
        self.assertEqual(
            parse_spec_tp_sync("off,dspark-target,2"),
            {SpecTpSyncSite.DSPARK_TARGET, SpecTpSyncSite.DSPARK_DRAFT_GREEDY},
        )
        with self.assertRaisesRegex(ValueError, "unknown token"):
            parse_spec_tp_sync("not-a-site")

    def test_sync_only_broadcasts_enabled_sites(self):
        group = SimpleNamespace(
            world_size=2, rank_in_group=0, broadcast=MagicMock()
        )
        values = torch.tensor([1, 2])
        with envs.SGLANG_SPEC_TP_SYNC.override("rng"):
            sync = SpecTpSync(group)
            self.assertIs(sync.sync(SpecTpSyncSite.DSPARK_TARGET, values), values)
            sync.sync(SpecTpSyncSite.DSPARK_DRAFT_GREEDY, values)

        group.broadcast.assert_called_once_with(values, src=0)

    def test_memory_probe_uses_group_min_only_when_enabled(self):
        sync_group = SimpleNamespace(
            world_size=1,
            rank_in_group=0,
            broadcast=MagicMock(),
        )
        memory_group = SimpleNamespace(
            world_size=2,
            cpu_group=object(),
        )
        with patch(
            "sglang.srt.speculative.spec_tp_sync.get_available_gpu_memory",
            return_value=3.5,
        ) as get_memory:
            with envs.SGLANG_SPEC_TP_SYNC.override("init"):
                sync = SpecTpSync(sync_group)
                self.assertEqual(
                    sync.available_memory_gb(
                        SpecTpSyncSite.DSPARK_MEM,
                        "cuda",
                        0,
                        group=memory_group,
                    ),
                    3.5,
                )
            get_memory.assert_called_once_with(
                "cuda", 0, distributed=True, cpu_group=memory_group.cpu_group
            )

            get_memory.reset_mock()
            with envs.SGLANG_SPEC_TP_SYNC.override("off"):
                sync = SpecTpSync(sync_group)
                sync.available_memory_gb(
                    SpecTpSyncSite.DSPARK_MEM,
                    "cuda",
                    0,
                    group=memory_group,
                )
            get_memory.assert_called_once_with(
                "cuda", 0, distributed=False, cpu_group=None
            )


if __name__ == "__main__":
    unittest.main()
