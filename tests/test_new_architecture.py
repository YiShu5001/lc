from __future__ import annotations

from pathlib import Path
import unittest

from lc.control.configs import ArtifactConfig, PyBulletControlExperimentConfig
from lc.control.experiments import run_control_comparison, run_control_generalization


class TestNewArchitecture(unittest.TestCase):
    def test_control_comparison_writes_outputs(self) -> None:
        result = run_control_comparison(
            PyBulletControlExperimentConfig(
                duration_sec=1.5,
                train_episodes=1,
                eval_episodes=1,
                artifact=ArtifactConfig(output_root="outputs/control_pybullet_test_arch"),
            )
        )
        self.assertIn("training", result)
        self.assertIn("evaluation", result)
        self.assertTrue((Path(result["training"]["x"]["output_dir"]) / "summary.json").exists())
        self.assertTrue((Path(result["evaluation"]["x"]["output_dir"]) / "summary.json").exists())

    def test_control_generalization_writes_outputs(self) -> None:
        result = run_control_generalization(
            PyBulletControlExperimentConfig(
                duration_sec=1.2,
                eval_episodes=1,
                artifact=ArtifactConfig(output_root="outputs/control_pybullet_test_generalization"),
            )
        )
        self.assertIn("easy", result["results"])
        self.assertIn("medium", result["results"])
        self.assertTrue((Path(result["output_dir"]) / "summary.json").exists())
        self.assertTrue((Path(result["output_dir"]) / "metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()
