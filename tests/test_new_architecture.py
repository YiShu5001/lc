from __future__ import annotations

from pathlib import Path
import unittest

from lc.control.configs import ControlExperimentConfig
from lc.control.experiments import run_control_comparison, run_control_generalization
from lc.integration.pipeline import run_bridge_experiment
from lc.planning.configs import PlanningExperimentConfig
from lc.planning.experiments import run_planning_comparison


class TestNewArchitecture(unittest.TestCase):
    def test_control_comparison_writes_outputs(self) -> None:
        result = run_control_comparison(
            ControlExperimentConfig(train_episodes=2, compare_episodes=2, episodes=2, seed_runs=1)
        )
        self.assertIn("mddpg_ladrc", result["results"])
        self.assertIn("ddpg_ladrc", result["results"])
        self.assertTrue((Path(result["output_dir"]) / "summary.json").exists())
        self.assertTrue((Path(result["output_dir"]) / "scenario.json").exists())
        self.assertTrue((Path(result["output_dir"]) / "control_objective.json").exists())
        self.assertTrue((Path(result["output_dir"]) / "ladrc_tuning_snapshots.csv").exists())
        self.assertTrue((Path(result["output_dir"]) / "seed_metrics.csv").exists())

    def test_control_generalization_writes_outputs(self) -> None:
        result = run_control_generalization(
            ControlExperimentConfig(
                difficulty_levels=("easy", "medium"),
                train_episodes=1,
                compare_episodes=1,
                episodes=1,
                seed_runs=1,
            )
        )
        self.assertIn("easy", result["results"])
        self.assertIn("medium", result["results"])
        self.assertTrue((Path(result["output_dir"]) / "summary.json").exists())
        self.assertTrue((Path(result["output_dir"]) / "metrics.csv").exists())

    def test_planning_comparison_writes_outputs(self) -> None:
        result = run_planning_comparison(PlanningExperimentConfig(episodes=12, eval_episodes=2))
        self.assertIn("task_decomposed", result["results"])
        self.assertIn("single_stream_mlp", result["results"])
        self.assertIn("without_curriculum", result["results"])
        self.assertIn("without_pyramid_per", result["results"])
        self.assertIn("uniform_replay", result["results"])
        self.assertTrue((Path(result["output_dir"]) / "summary.json").exists())

    def test_bridge_demo_writes_outputs(self) -> None:
        result = run_bridge_experiment()
        self.assertTrue((Path(result["output_dir"]) / "summary.json").exists())
        self.assertIn("mapped_control_reference", result["summary"])


if __name__ == "__main__":
    unittest.main()
