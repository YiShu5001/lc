from __future__ import annotations

import unittest
from pathlib import Path

from lc.planning.configs import PlanningExperimentConfig
from lc.planning.experiments import run_planning_comparison


class TestPlanningCompareOutputs(unittest.TestCase):
    def test_compare_outputs_include_new_experiments_and_figures(self) -> None:
        result = run_planning_comparison(PlanningExperimentConfig(episodes=4, eval_episodes=1, seed=19))
        expected_methods = {
            "task_decomposed",
            "single_stream_mlp",
            "without_curriculum",
            "without_pyramid_per",
            "uniform_replay",
            "td_only_priority",
            "high_old_mix",
            "contribution_only_priority",
            "rare_only_priority",
            "balanced_sample_ratio",
            "low_old_mix",
            "high_rare_ratio",
        }
        self.assertTrue(expected_methods.issubset(result["results"].keys()))

        training = result["training"]["task_decomposed"]
        self.assertIn("replay_stats", training)
        self.assertIn("trainer_overrides", training)
        self.assertIn("train_overrides", training)
        self.assertIn("stage_transition_summary", training)
        self.assertIn("last_sample_source_fractions", training["replay_stats"])

        figure_names = {Path(path).name for path in result["figures"]}
        self.assertIn("reward_components_curve.svg", figure_names)
        self.assertIn("replay_distribution.svg", figure_names)
        self.assertIn("stage_retention_curve.svg", figure_names)
        self.assertIn("old_pool_mix_curve.svg", figure_names)
        self.assertIn("priority_mode_comparison.svg", figure_names)
        self.assertIn("sample_ratio_comparison.svg", figure_names)
        self.assertIn("reward_breakdown_bar.svg", figure_names)


if __name__ == "__main__":
    unittest.main()
