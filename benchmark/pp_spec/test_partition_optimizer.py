import unittest

from benchmark.pp_spec.partition_optimizer import (
    choose_dynamic_dcut_ratio,
    optimize_joint,
    optimize_partition_across_buckets,
    optimize_partition_across_ratios,
)
from benchmark.pp_spec.stage_model import StageCostModel


class TestJointPartitionOptimizer(unittest.TestCase):
    @staticmethod
    def _model() -> StageCostModel:
        return StageCostModel.from_bucket_profiles(
            {8: {
                "layer_cost_ms": 1.0,
                "gdn_cost_ms": 1.0,
                "full_cost_ms": 1.0,
                "fixed_ms": [0.0, 4.0],
            }},
            num_layers=12,
            pp_size=2,
            baseline_partition=(6, 6),
        )

    def test_joint_search_can_move_boundary_for_cut(self):
        result = optimize_joint(
            self._model(),
            {1.0: 10.0, 0.5: 5.0},
            target_bs=8,
            min_layers=1,
            k_best=20,
        )
        self.assertEqual(result.selected.partition, (10, 2))
        self.assertEqual(result.selected.dcut_ratio, 0.5)
        self.assertLess(result.cycle_time_ms, result.baseline.cycle_time_ms)

    def test_stage_profile_preserves_asymmetry(self):
        result = optimize_joint(
            self._model(),
            {1.0: [10.0, 20.0], 0.5: [8.0, 8.0]},
            target_bs=8,
            k_best=10,
        )
        self.assertEqual(result.selected.dcut_ratio, 0.5)

    def test_dynamic_ratio_reacts_to_stage_imbalance(self):
        ratio = choose_dynamic_dcut_ratio(
            [10.0, 20.0],
            {1.0: [10.0, 20.0], 0.5: [8.0, 8.0]},
        )
        self.assertEqual(ratio, 0.5)

    def test_all_boundaries_mode_enumerates_nonuniform_compositions(self):
        model = StageCostModel.from_bucket_profiles(
            {8: {
                "layer_cost_ms": 1.0,
                "gdn_cost_ms": 1.0,
                "full_cost_ms": 1.0,
                "fixed_ms": [0.0, 0.0, 8.0],
            }},
            num_layers=9,
            pp_size=3,
            baseline_partition=(3, 3, 3),
        )
        result = optimize_joint(
            model,
            {1.0: 1.0},
            target_bs=8,
            all_boundaries=True,
            k_best=100,
        )
        self.assertTrue(any(item.partition == (1, 1, 7) for item in result.candidates))

    def test_robust_search_scores_each_partition_over_all_ratios(self):
        result = optimize_partition_across_ratios(
            self._model(),
            {1.0: 10.0, 0.5: 5.0},
            target_bs=8,
            k_best=20,
        )
        self.assertEqual(result.selected.partition, (8, 4))
        self.assertEqual(set(result.selected.cycle_by_ratio), {0.5, 1.0})
        self.assertEqual(result.to_dict()["runtime_dcut"], "auto")

    def test_multi_bucket_search_uses_one_static_partition(self):
        model = StageCostModel.from_bucket_profiles(
            {
                4: {
                    "layer_cost_ms": 1.0,
                    "gdn_cost_ms": 1.0,
                    "full_cost_ms": 1.0,
                    "fixed_ms": [0.0, 4.0],
                },
                16: {
                    "layer_cost_ms": 1.0,
                    "gdn_cost_ms": 1.0,
                    "full_cost_ms": 1.0,
                    "fixed_ms": [0.0, 7.0],
                },
            },
            num_layers=12,
            pp_size=2,
            baseline_partition=(6, 6),
        )
        result = optimize_partition_across_buckets(
            model,
            {
                4: {1.0: 1.0, 0.5: 0.5},
                16: {1.0: 1.0, 0.5: 0.5},
            },
            all_boundaries=True,
            k_best=10,
        )
        self.assertEqual(result.target_buckets, (4, 16))
        self.assertEqual(result.to_dict()["runtime_dcut"], "auto")
        self.assertEqual(len(result.selected.cycle_by_bucket), 2)


if __name__ == "__main__":
    unittest.main()
