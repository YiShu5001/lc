from __future__ import annotations

import unittest

import numpy as np

from lc.control.RLcontrolRefLine import (
    PhaseKind,
    adapt_episode_to_tracking_inputs,
    build_default_xy_task_config,
    build_refline_episode,
    sample_phase_plan,
)
from lc.control.envs import ControlTrackingEnv
from lc.envs.scenarios import build_control_scenario


class TestChapter3RLControlRefLine(unittest.TestCase):
    def test_default_xy_plan_has_fixed_six_phase_order(self) -> None:
        config = build_default_xy_task_config("x")
        plan = sample_phase_plan(config, seed=7)
        self.assertEqual(
            [phase.kind for phase in plan.phases],
            [
                PhaseKind.HOLD_START,
                PhaseKind.FORWARD_CONSTANT_VELOCITY,
                PhaseKind.DISTURBANCE_HOLD,
                PhaseKind.REVERSE_CONSTANT_VELOCITY,
                PhaseKind.DISTURBANCE_RECOVERY,
                PhaseKind.HOLD_END,
            ],
        )
        self.assertEqual(plan.total_steps, int(round(config.total_duration_sec * config.control_frequency_hz)))

    def test_episode_values_match_expected_signs(self) -> None:
        config = build_default_xy_task_config("y")
        episode = build_refline_episode(config, seed=11)
        phase_table = episode.phase_table
        forward = phase_table[1]
        reverse = phase_table[3]
        recovery = phase_table[4]
        self.assertGreater(float(forward["reference_velocity"]), 0.0)
        self.assertLess(float(reverse["reference_velocity"]), 0.0)
        self.assertAlmostEqual(float(recovery["disturbance_end"]), 0.0, places=6)
        self.assertEqual(len(episode.time), len(episode.reference_position))
        self.assertEqual(len(episode.reference_position), len(episode.disturbance))

    def test_tracking_env_consumes_external_episode_bundle(self) -> None:
        config = build_default_xy_task_config("x")
        episode = build_refline_episode(config, seed=5)
        adapted = adapt_episode_to_tracking_inputs(episode)
        env = ControlTrackingEnv(
            scenario=build_control_scenario("medium"),
            axis="x",
            seed=5,
            episode_length=len(episode.time),
            reference_profile_mode="rl_refline_six_phase",
            external_episode_bundle=episode,
        )
        obs = env.reset()
        self.assertEqual(obs.shape[0], 4)
        self.assertEqual(env.reference_schedule, "rl_refline_six_phase")
        self.assertTrue(np.allclose(adapted["reference_velocity"], np.asarray(env.external_reference_velocity, dtype=np.float32)))
        self.assertTrue(np.allclose(adapted["disturbance"], np.asarray(env.external_disturbance, dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
