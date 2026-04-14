from __future__ import annotations

import unittest

from lc.control.configs import ControlExperimentConfig
from lc.control.envs import ControlTrackingEnv
from lc.control.policies import ControlLADRLAgent
from lc.control.trainers.control_trainer import ControlTrainer
from lc.entrypoints.run_x_axis_rl_refline_parallel_suite import build_jobs
from lc.envs.scenarios import build_control_scenario


class TestXAxisRlReflineDecayConfig(unittest.TestCase):
    def test_linear_exploration_noise_schedule(self) -> None:
        trainer = ControlTrainer(
            env=ControlTrackingEnv(
                scenario=build_control_scenario("medium"),
                axis="x",
                seed=7,
                episode_length=100,
                reference_profile_mode="rl_refline_six_phase",
            ),
            exploration_noise_schedule="linear",
            exploration_noise_start=0.1,
            exploration_noise_end=0.02,
        )
        self.assertAlmostEqual(trainer._episode_exploration_noise(0, 500), 0.1, places=6)
        self.assertAlmostEqual(trainer._episode_exploration_noise(499, 500), 0.02, places=6)
        self.assertGreater(trainer._episode_exploration_noise(100, 500), trainer._episode_exploration_noise(300, 500))

    def test_config_can_store_output_subdir(self) -> None:
        cfg = ControlExperimentConfig(output_subdir="custom-output", snapshot_interval=10)
        self.assertEqual(cfg.output_subdir, "custom-output")
        self.assertEqual(cfg.snapshot_interval, 10)

    def test_policy_receives_model_hyperparameters(self) -> None:
        agent = ControlLADRLAgent(
            obs_dim=4,
            stack_size=7,
            actor_lr=3e-4,
            critic_lr=3e-4,
            hidden_dim=768,
            dropout_p=0.25,
            tau=0.02,
            soft_update_interval=10,
            exploration_noise_schedule="linear",
            exploration_noise_start=0.1,
            exploration_noise_end=0.04,
        )
        self.assertEqual(agent.policy.config.hidden_dim, 768)
        self.assertAlmostEqual(agent.policy.config.dropout_p, 0.25, places=6)
        self.assertAlmostEqual(agent.policy.config.tau, 0.02, places=6)
        self.assertEqual(agent.policy.config.soft_update_interval, 10)

    def test_parallel_suite_builds_three_distinct_jobs(self) -> None:
        jobs = build_jobs()
        self.assertEqual(len(jobs), 3)
        self.assertEqual(len({job[0] for job in jobs}), 3)
        self.assertEqual(len({job[2] for job in jobs}), 3)


if __name__ == "__main__":
    unittest.main()
