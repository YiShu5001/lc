from __future__ import annotations

import unittest
from pathlib import Path

from lc.control.configs import PyBulletControlExperimentConfig
from lc.control.controllers import create_controller_bundle
from lc.control.experiments import run_pybullet_ladrc_axis_tuning


class TestChapter3PyBulletLADRCTuning(unittest.TestCase):
    def test_single_axis_variants_only_replace_one_position_axis(self) -> None:
        cases = {
            "ladrc_x_pos_pid_att": "x",
            "ladrc_y_pos_pid_att": "y",
            "ladrc_z_pos_pid_att": "z",
        }
        for name, axis in cases.items():
            bundle = create_controller_bundle(name)
            self.assertTrue(bundle.uses_ladrc_on_axis(axis))
            for other in {"x", "y", "z"} - {axis}:
                self.assertFalse(bundle.uses_ladrc_on_axis(other))
            self.assertFalse(bundle.use_ladrc_attitude)

    def test_axis_tuning_outputs_rl_bounds(self) -> None:
        result = run_pybullet_ladrc_axis_tuning(
            PyBulletControlExperimentConfig(duration_sec=1.0, eval_episodes=1),
            axis="y",
        )
        self.assertIn("b0", result.recommended_params)
        self.assertIn("wc", result.recommended_params)
        self.assertIn("k", result.recommended_params)
        self.assertIn("b0_min", result.rl_bounds)
        self.assertIn("b0_max", result.rl_bounds)
        self.assertIn("wc_min", result.rl_bounds)
        self.assertIn("k_max", result.rl_bounds)

    def test_axis_tuning_writes_sequential_stage_artifacts(self) -> None:
        result = run_pybullet_ladrc_axis_tuning(
            PyBulletControlExperimentConfig(duration_sec=1.0, eval_episodes=1),
            axis="x",
        )
        run_dir = Path(result.output_dir)
        self.assertTrue((run_dir / "b0_stage.csv").exists())
        self.assertTrue((run_dir / "wc_stage.csv").exists())
        self.assertTrue((run_dir / "k_stage.csv").exists())
        self.assertTrue((run_dir / "local_refine.csv").exists())
        self.assertTrue((run_dir / "recommended_params.json").exists())
        self.assertTrue((run_dir / "rl_bounds.json").exists())
        self.assertTrue((run_dir / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
