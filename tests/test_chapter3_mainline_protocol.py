from __future__ import annotations

import unittest
from pathlib import Path

from lc.control.configs import ArtifactConfig, PyBulletControlExperimentConfig
from lc.control.controllers import LADRCController
from lc.control.experiments import run_control_comparison


class TestChapter3MainlineProtocol(unittest.TestCase):
    def test_ladrc_parameterization_uses_wc_k_relation(self) -> None:
        controller = LADRCController(b0=1.2, omega_c=5.0, k=4.0)
        self.assertAlmostEqual(controller.omega_o, 20.0)
        controller.omega_o = 15.0
        self.assertAlmostEqual(controller.k, 3.0)

    def test_control_comparison_runs_in_pybullet(self) -> None:
        result = run_control_comparison(
            PyBulletControlExperimentConfig(
                duration_sec=1.5,
                train_episodes=1,
                eval_episodes=1,
                artifact=ArtifactConfig(output_root="outputs/control_pybullet_test_mainline"),
            )
        )
        self.assertIn("training", result)
        self.assertIn("evaluation", result)
        self.assertIn("x", result["training"])
        self.assertIn("x", result["evaluation"])
        self.assertTrue((Path(result["training"]["x"]["output_dir"]) / "summary.json").exists())
        self.assertTrue((Path(result["evaluation"]["x"]["output_dir"]) / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
