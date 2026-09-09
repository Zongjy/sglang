import unittest

from benchmark.pp_spec.partition_optimizer import (
    OptimizerError,
    choose_dynamic_dcut_ratio,
    optimize_joint,
    optimize_partition_across_buckets,
    optimize_partition_across_ratios,
    optimize_per_cell_partitions,
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


class TestPerCellPartitionOptimizer(unittest.TestCase):
    @staticmethod
    def _model() -> StageCostModel:
        return StageCostModel.from_bucket_profiles(
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

    # Per-stage costs measured at the baseline (6,6): deep cuts help stage 0
    # (pure layer work) far more than stage 1 (fixed draft/head floor).
    _DCUT = {
        4: {1.0: [6.0, 10.0], 0.25: [2.0, 6.0]},
        16: {1.0: [6.0, 13.0], 0.25: [2.0, 8.0]},
    }

    def test_per_cell_optima_and_majority_selection(self):
        result = optimize_per_cell_partitions(
            self._model(),
            self._DCUT,
            min_layers=1,
            all_boundaries=True,
        )
        self.assertEqual(len(result.cells), 4)
        by_cell = {(cell.bucket, cell.ratio): cell for cell in result.cells}
        # Full width: balance against the fixed stage-1 floor.
        self.assertEqual(by_cell[(4, 1.0)].partition, (8, 4))
        self.assertEqual(by_cell[(16, 1.0)].partition, (9, 3))
        # Deep cut: stage 0's layer work shrinks 3x, so layers move there.
        self.assertEqual(by_cell[(4, 0.25)].partition, (11, 1))
        self.assertEqual(by_cell[(16, 0.25)].partition, (11, 1))
        # Most cell wins, no weighting.
        self.assertEqual(result.selected, (11, 1))
        self.assertEqual(result.cell_wins[0], ((11, 1), 2))

    def test_runtime_cost_table_schema_and_values(self):
        result = optimize_per_cell_partitions(
            self._model(),
            self._DCUT,
            min_layers=1,
            all_boundaries=True,
        )
        table = result.to_runtime_cost_table(block_size=16)
        self.assertEqual(table["version"], 1)
        self.assertEqual(table["pp_size"], 2)
        self.assertEqual(table["pp_layer_partition"], [11, 1])
        self.assertEqual(table["ratios"], [0.25, 1.0])
        # Rows are per stage, columns follow the sorted ratio order.
        rows = table["buckets"]["4"]["stage_costs_ms"]
        self.assertEqual(len(rows), 2)
        # (11,1): stage0 = 11 layers; stage1 = 1 layer + fixed 4.
        self.assertAlmostEqual(rows[0][0], 11.0 / 3.0, places=4)  # ratio 0.25
        self.assertAlmostEqual(rows[0][1], 11.0, places=4)  # ratio 1.0
        self.assertAlmostEqual(rows[1][0], 4.0 + 0.6, places=4)
        self.assertAlmostEqual(rows[1][1], 5.0, places=4)

    def test_per_cell_requires_matching_buckets(self):
        with self.assertRaises(OptimizerError):
            optimize_per_cell_partitions(
                self._model(),
                {4: {1.0: 1.0}},
                min_layers=1,
            )


if __name__ == "__main__":
    unittest.main()
