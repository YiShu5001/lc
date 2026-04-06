from __future__ import annotations

import unittest

import numpy as np

from lc.control.configs import ControlExperimentConfig
from lc.control.controllers import LADRCController
from lc.control.envs import ControlTrackingEnv
from lc.control.experiments import run_control_comparison
from lc.envs.scenarios import build_control_scenario


class TestChapter3MainlineProtocol(unittest.TestCase):
    def test_ladrc_parameterization_uses_wc_k_relation(self) -> None:
        controller = LADRCController(b0=1.2, omega_c=5.0, k=4.0)
        self.assertAlmostEqual(controller.omega_o, 20.0)
        controller.omega_o = 15.0
        self.assertAlmostEqual(controller.k, 3.0)

    def test_tracking_env_uses_axiswise_recursive_reference(self) -> None:
        env = ControlTrackingEnv(
            scenario=build_control_scenario("medium"),
            axis="y",
            seed=5,
            episode_length=40,
        )
        obs = env.reset()
        self.assertEqual(obs.shape[0], 8)
        self.assertTrue(np.allclose(env.reference_bundle.positions[:, 0], env.reference_bundle.positions[0, 0]))
        self.assertFalse(np.allclose(env.reference_bundle.positions[:, 1], env.reference_bundle.positions[0, 1]))
        self.assertTrue(np.allclose(env.reference_bundle.positions[:, 2], env.reference_bundle.positions[0, 2]))

    def test_control_comparison_reports_axis_results(self) -> None:
        result = run_control_comparison(
            ControlExperimentConfig(
                difficulty="easy",
                axes=("x", "y"),
                train_episodes=1,
                compare_episodes=1,
                episodes=1,
                seed_runs=1,
            )
        )
        self.assertIn("axis_results", result)
        self.assertIn("x", result["axis_results"])
        self.assertIn("y", result["axis_results"])


if __name__ == "__main__":
    unittest.main()
