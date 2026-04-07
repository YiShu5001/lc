from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from lc.control.RLcontrolRefLine import build_default_xy_task_config, build_refline_episode
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

    def test_tracking_env_supports_refline_six_phase_mode(self) -> None:
        bundle = build_refline_episode(build_default_xy_task_config("x"), seed=13)
        env = ControlTrackingEnv(
            scenario=build_control_scenario("medium"),
            axis="x",
            seed=13,
            episode_length=len(bundle.time),
            reference_profile_mode="rl_refline_six_phase",
            external_episode_bundle=bundle,
        )
        obs = env.reset()
        self.assertEqual(obs.shape[0], 4)
        self.assertEqual(env.reference_schedule, "rl_refline_six_phase")
        self.assertEqual(len(env.reference_summary()), 6)

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

    def test_control_comparison_supports_x_axis_refline_shared_value_sweep(self) -> None:
        result = run_control_comparison(
            ControlExperimentConfig(
                difficulty="easy",
                axes=("x",),
                reference_profile_mode="rl_refline_six_phase",
                mddpg_shared_values=(1, 3),
                train_episodes=1,
                compare_episodes=1,
                episodes=1,
                seed_runs=1,
            )
        )
        self.assertEqual(sorted(result["axis_results"].keys()), ["x"])
        self.assertIn(result["best_mddpg_value"], {1, 3})
        self.assertEqual(len(result["mddpg_shared_value_sweep"]), 2)
        output_dir = Path(result["output_dir"])
        self.assertTrue((output_dir / "summary.json").exists())
        self.assertTrue((output_dir / "mddpg_shared_value_sweep.csv").exists())


if __name__ == "__main__":
    unittest.main()
