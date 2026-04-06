from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lc.control.configs import ArtifactConfig, PyBulletControlExperimentConfig
from lc.control.controllers import create_controller_bundle
from lc.control.experiments import (
    run_pybullet_axis_training,
    run_pybullet_controller_benchmark,
    run_pybullet_ladrc_axis_tuning,
)


class TestChapter3PyBulletProtocol(unittest.TestCase):
    def test_controller_variants_can_be_instantiated(self) -> None:
        for name in (
            "pid_pos_att",
            "ladrc_pos_pid_att",
            "ladrc_pos_att",
            "ladrc_x_pos_pid_att",
            "ladrc_y_pos_pid_att",
            "ladrc_z_pos_pid_att",
        ):
            bundle = create_controller_bundle(name)
            self.assertEqual(bundle.name, name)

    def test_benchmark_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = PyBulletControlExperimentConfig(
                duration_sec=1.5,
                train_episodes=2,
                eval_episodes=1,
                artifact=ArtifactConfig(output_root=tmp),
            )
            result = run_pybullet_controller_benchmark(config, axis="x", controller="all")
            run_dir = Path(result["x"]["output_dir"])
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "metrics.csv").exists())
            self.assertTrue(any(run_dir.glob("**/timeseries.csv")))
            self.assertTrue(any(run_dir.glob("**/*.png")))
            self.assertTrue(any(run_dir.glob("**/legacy_logger/*.csv")))

    def test_training_writes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = PyBulletControlExperimentConfig(
                duration_sec=1.5,
                train_episodes=2,
                artifact=ArtifactConfig(output_root=tmp),
            )
            result = run_pybullet_axis_training(config, axis="x")
            run_dir = Path(result["x"]["output_dir"])
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue(any(run_dir.glob("checkpoints/*.pt")))
            self.assertTrue(any(run_dir.glob("figures/*.png")))

    def test_ladrc_axis_tuning_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = PyBulletControlExperimentConfig(
                duration_sec=1.2,
                eval_episodes=1,
                artifact=ArtifactConfig(output_root=tmp),
            )
            result = run_pybullet_ladrc_axis_tuning(config, axis="x")
            run_dir = Path(result.output_dir)
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "coarse_search.csv").exists())
            self.assertTrue((run_dir / "fine_search.csv").exists())
            self.assertTrue((run_dir / "recommended_params.json").exists())
            self.assertTrue((run_dir / "rl_bounds.json").exists())
            self.assertTrue(any(run_dir.glob("figures/*.png")))


if __name__ == "__main__":
    unittest.main()
