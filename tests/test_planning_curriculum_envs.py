from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from lc.envs.scenarios import PLANNING_CURRICULUM_ENVS, build_planning_scenario
from lc.planning.curriculum import MIN_EFFECTIVE_SEQUENCE, CurriculumScheduler
from lc.planning.envs import PlanningSwarmEnv
from lc.planning.experiments.manual_block_training import ReviewMixPromotionController, RollingSuccessPromotionController
from lc.planning.rewards import compute_planning_reward


class TestPlanningCurriculumEnvs(unittest.TestCase):
    def test_guidance_envs_have_expected_workspace_and_target_modes(self) -> None:
        g1 = build_planning_scenario(curriculum_env="guidance_G1")
        g2 = build_planning_scenario(curriculum_env="guidance_G2")
        self.assertEqual(g1.num_obstacles, 0)
        self.assertEqual(g1.workspace_size_x, 1.5)
        self.assertEqual(g1.workspace_size_y, 1.5)
        self.assertAlmostEqual(g1.target_min_spawn_distance, 0.25, places=6)
        self.assertAlmostEqual(g1.target_max_spawn_distance, 0.5, places=6)
        self.assertFalse(g1.target_is_dynamic)
        self.assertTrue(g2.target_is_dynamic)
        self.assertAlmostEqual(g2.target_max_drift, 0.30, places=6)

    def test_avoidance_and_cooperation_envs_follow_new_counts(self) -> None:
        a1 = build_planning_scenario(curriculum_env="avoidance_A1")
        a1_variant = build_planning_scenario(curriculum_env="avoidance_A1_gmix_softtimeout")
        a1_free = build_planning_scenario(curriculum_env="avoidance_A1_gmix_softtimeout_free_motion")
        a1_soft = build_planning_scenario(curriculum_env="avoidance_A1_gmix_softtimeout_soft_penalty")
        a4 = build_planning_scenario(curriculum_env="avoidance_A4")
        c1 = build_planning_scenario(curriculum_env="cooperation_C1")
        c4 = build_planning_scenario(curriculum_env="cooperation_C4")
        self.assertAlmostEqual(a1.target_hold_radius, 0.10, places=6)
        self.assertAlmostEqual(a1.timeout_seconds, 60.0, places=6)
        self.assertEqual(a1_variant.curriculum_env, "avoidance_A1_gmix_softtimeout")
        self.assertAlmostEqual(a1_variant.timeout_seconds, 60.0, places=6)
        self.assertEqual(a1_variant.workspace_size_x, 5.0)
        self.assertEqual(a1_variant.workspace_size_y, 5.0)
        self.assertEqual(a1_variant.spawn_region, (-0.32, 0.32, -0.32, 0.32))
        self.assertEqual(a1_variant.target_region_bounds, (1.60, 2.35, 1.20, 2.35))
        self.assertEqual(a1_free.workspace_size_x, 5.0)
        self.assertEqual(a1_free.curriculum_env, "avoidance_A1_gmix_softtimeout_free_motion")
        self.assertEqual(a1_soft.workspace_size_y, 5.0)
        self.assertEqual(a1_soft.curriculum_env, "avoidance_A1_gmix_softtimeout_soft_penalty")
        self.assertEqual(a4.workspace_size_x, 2.5)
        self.assertEqual(a4.obstacle_count_range, (8, 10))
        self.assertTrue(a4.path_feasibility_check)
        self.assertTrue(a4.target_is_dynamic)
        self.assertEqual(c1.num_uavs, 2)
        self.assertEqual(c1.num_obstacles, 0)
        self.assertFalse(c1.target_is_dynamic)
        self.assertEqual(c1.success_mode, "cooperative_approach_target")
        self.assertEqual(c1.spawn_region, (-1.35, -0.45, -1.35, -0.45))
        self.assertEqual(c1.target_region_mode, "upper_right_square")
        self.assertEqual(c4.num_uavs, 3)
        self.assertGreaterEqual(c4.obstacle_count_range[0], 2)
        self.assertTrue(c4.target_is_dynamic)
        self.assertEqual(c4.success_mode, "encircle_target")

    def test_natural_a1_to_a4_curriculum_progresses_spatial_before_dynamic_complexity(self) -> None:
        a1 = build_planning_scenario(curriculum_env="avoidance_A1_static_single")
        a2 = build_planning_scenario(curriculum_env="avoidance_A2_static_multi")
        a3 = build_planning_scenario(curriculum_env="avoidance_A3_dynamic_few")
        a4 = build_planning_scenario(curriculum_env="avoidance_A4_dynamic_multi_target")
        self.assertEqual(a1.num_obstacles, 1)
        self.assertFalse(a1.obstacle_is_dynamic)
        self.assertFalse(a1.target_is_dynamic)
        self.assertEqual(a1.obstacle_count_range, (1, 1))
        self.assertGreaterEqual(a2.num_obstacles, 2)
        self.assertFalse(a2.obstacle_is_dynamic)
        self.assertFalse(a2.target_is_dynamic)
        self.assertGreaterEqual(a2.obstacle_count_range[0], 2)
        self.assertTrue(a3.obstacle_is_dynamic)
        self.assertTrue(a3.target_is_dynamic)
        self.assertLessEqual(a3.obstacle_count_range[1], 2)
        self.assertTrue(a4.obstacle_is_dynamic)
        self.assertTrue(a4.target_is_dynamic)
        self.assertGreater(a4.workspace_size_x, a3.workspace_size_x)
        self.assertGreaterEqual(a4.obstacle_count_range[0], 3)

    def test_cooperation_curriculum_progresses_pair_target_work_to_ring_with_obstacles(self) -> None:
        c1 = build_planning_scenario(curriculum_env="cooperation_C1")
        c2 = build_planning_scenario(curriculum_env="cooperation_C2")
        c3 = build_planning_scenario(curriculum_env="cooperation_C3")
        c4 = build_planning_scenario(curriculum_env="cooperation_C4")
        self.assertEqual(c1.num_uavs, 2)
        self.assertEqual(c1.num_obstacles, 0)
        self.assertFalse(c1.target_is_dynamic)
        self.assertEqual(c1.success_mode, "cooperative_approach_target")
        self.assertEqual(c2.num_uavs, 2)
        self.assertEqual(c2.num_obstacles, 0)
        self.assertTrue(c2.target_is_dynamic)
        self.assertEqual(c2.success_mode, "cooperative_approach_target")
        self.assertEqual(c3.num_uavs, 3)
        self.assertEqual(c3.num_obstacles, 0)
        self.assertFalse(c3.target_is_dynamic)
        self.assertEqual(c3.success_mode, "encircle_target")
        self.assertEqual(c4.num_uavs, 3)
        self.assertGreaterEqual(c4.obstacle_count_range[0], 2)
        self.assertTrue(c4.target_is_dynamic)
        self.assertEqual(c4.success_mode, "encircle_target")

    def test_cooperative_approach_success_requires_pair_slots_near_target(self) -> None:
        scenario = build_planning_scenario(curriculum_env="cooperation_C1")
        env = PlanningSwarmEnv(scenario=scenario)
        env.reset()
        target = np.array([0.5, 0.5], dtype=np.float32)
        env._target_position = target
        env._position = target + np.array([scenario.encircle_radius, 0.0], dtype=np.float32)
        env._velocity = np.zeros(2, dtype=np.float32)
        env._neighbor_positions_state = np.array(
            [target + np.array([-scenario.encircle_radius, 0.0], dtype=np.float32)],
            dtype=np.float32,
        )
        env._neighbor_velocities = np.zeros_like(env._neighbor_positions_state)
        done = False
        info = {}
        for _ in range(max(1, int(round(scenario.target_hold_seconds / env.step_dt)))):
            _, _, done, info = env.step(np.zeros(2, dtype=np.float32))
        self.assertTrue(done)
        self.assertEqual(info["success"], 1.0)
        self.assertEqual(info["success_mode"], "cooperative_approach_target")

    def test_dynamic_curriculum_envs_assign_nonzero_obstacle_velocities(self) -> None:
        for env_name in ("avoidance_A3_dynamic_few", "avoidance_A4_dynamic_multi_target"):
            scenario = build_planning_scenario(curriculum_env=env_name)
            env = PlanningSwarmEnv(scenario=scenario)
            env.reset()
            active = int(len(env._obstacle_positions))
            self.assertGreater(active, 0)
            velocity_norms = np.linalg.norm(env._obstacle_velocities[:active], axis=1)
            self.assertGreater(float(velocity_norms.max()), 0.0)

    def test_rolling_promotion_uses_50_episode_80_percent_success_gate(self) -> None:
        class FakeEnv:
            def __init__(self) -> None:
                self.scenario = build_planning_scenario(curriculum_env="avoidance_A1_static_single")

            def set_scenario(self, scenario) -> None:  # type: ignore[no-untyped-def]
                self.scenario = scenario

        class FakeTrainer:
            def __init__(self) -> None:
                self.env = FakeEnv()

        trainer = FakeTrainer()
        controller = RollingSuccessPromotionController(
            env_sequence=["avoidance_A1_static_single", "avoidance_A2_static_multi"],
            window_size=50,
            min_successes=40,
            max_duration_seconds=1e9,
            max_episodes_per_env=800,
        )
        for episode in range(1, 51):
            controller.observe(
                {
                    "trainer": trainer,
                    "episode_count": episode,
                    "episode_metrics": {
                        "curriculum_env": "avoidance_A1_static_single",
                        "episode_steps": 20,
                        "episode_duration_seconds": 25.0,
                        "episode_success": 1.0 if episode <= 40 else 0.0,
                    },
                    "training_counters": {},
                }
            )
        self.assertEqual(trainer.env.scenario.curriculum_env, "avoidance_A2_static_multi")
        self.assertTrue(controller.last_state["promotion_triggered"])
        self.assertEqual(controller.last_state["promotion_reason"], "success_window")

    def test_rolling_promotion_forces_next_stage_at_episode_cap(self) -> None:
        class FakeEnv:
            def __init__(self) -> None:
                self.scenario = build_planning_scenario(curriculum_env="avoidance_A1_static_single")

            def set_scenario(self, scenario) -> None:  # type: ignore[no-untyped-def]
                self.scenario = scenario

        class FakeTrainer:
            def __init__(self) -> None:
                self.env = FakeEnv()

        trainer = FakeTrainer()
        controller = RollingSuccessPromotionController(
            env_sequence=["avoidance_A1_static_single", "avoidance_A2_static_multi"],
            window_size=50,
            min_successes=40,
            max_duration_seconds=1e9,
            max_episodes_per_env=3,
        )
        for episode in range(1, 4):
            controller.observe(
                {
                    "trainer": trainer,
                    "episode_count": episode,
                    "episode_metrics": {
                        "curriculum_env": "avoidance_A1_static_single",
                        "episode_steps": 20,
                        "episode_duration_seconds": 25.0,
                        "episode_success": 0.0,
                    },
                    "training_counters": {},
                }
            )
        self.assertEqual(trainer.env.scenario.curriculum_env, "avoidance_A2_static_multi")
        self.assertEqual(controller.last_state["promotion_reason"], "max_episodes")

    def test_review_mix_promotion_samples_previous_stage_and_rolls_back_on_review_failures(self) -> None:
        class FakeEnv:
            def __init__(self) -> None:
                self.scenario = build_planning_scenario(curriculum_env="avoidance_A2_static_multi")

            def set_scenario(self, scenario) -> None:  # type: ignore[no-untyped-def]
                self.scenario = scenario

        class FakeTrainer:
            def __init__(self) -> None:
                self.env = FakeEnv()

        trainer = FakeTrainer()
        controller = ReviewMixPromotionController(
            env_sequence=["avoidance_A1_static_single", "avoidance_A2_static_multi"],
            window_size=50,
            min_successes=40,
            current_sample_count=9,
            previous_sample_count=1,
            rollback_previous_failures=3,
        )
        controller.current_index = 1
        controller.current_env = "avoidance_A2_static_multi"
        previous_samples = 0
        first_warning_seen = False
        for episode in range(1, 101):
            controller.on_episode_start({"trainer": trainer, "episode_count": episode})
            sampled_env = trainer.env.scenario.curriculum_env
            if sampled_env == "avoidance_A1_static_single":
                previous_samples += 1
            previous_failure = sampled_env == "avoidance_A1_static_single" and previous_samples in {1, 2, 3, 6, 7, 8}
            controller.observe(
                {
                    "trainer": trainer,
                    "episode_count": episode,
                    "episode_metrics": {
                        "curriculum_env": sampled_env,
                        "episode_steps": 20,
                        "episode_success": 0.0 if previous_failure else 1.0,
                    },
                    "training_counters": {},
                }
            )
            if controller.last_state.get("degradation_warning") and not first_warning_seen:
                first_warning_seen = True
                self.assertEqual(controller.current_env, "avoidance_A2_static_multi")
            if controller.last_state.get("rollback_triggered"):
                break
        self.assertTrue(first_warning_seen)
        self.assertGreaterEqual(previous_samples, 8)
        self.assertEqual(controller.current_env, "avoidance_A1_static_single")
        self.assertEqual(controller.last_state["promotion_reason"], "previous_review_failure")

    def test_review_mix_promotion_requires_success_above_60_and_collision_below_30(self) -> None:
        class FakeEnv:
            def __init__(self) -> None:
                self.scenario = build_planning_scenario(curriculum_env="avoidance_A1_static_single")

            def set_scenario(self, scenario) -> None:  # type: ignore[no-untyped-def]
                self.scenario = scenario

        class FakeTrainer:
            def __init__(self) -> None:
                self.env = FakeEnv()

        trainer = FakeTrainer()
        controller = ReviewMixPromotionController(
            env_sequence=["avoidance_A1_static_single", "avoidance_A2_static_multi"],
            window_size=50,
            min_successes=31,
            max_collision_rate=0.30,
        )
        for episode in range(1, 51):
            controller.observe(
                {
                    "trainer": trainer,
                    "episode_count": episode,
                    "episode_metrics": {
                        "curriculum_env": "avoidance_A1_static_single",
                        "episode_steps": 20,
                        "episode_success": 1.0 if episode <= 35 else 0.0,
                        "episode_collision": 1.0 if episode <= 15 else 0.0,
                    },
                    "training_counters": {},
                }
            )
        self.assertEqual(controller.current_env, "avoidance_A1_static_single")
        self.assertFalse(controller.last_state["promotion_triggered"])
        self.assertAlmostEqual(controller.last_state["rolling_collision_rate"], 0.30)

        controller = ReviewMixPromotionController(
            env_sequence=["avoidance_A1_static_single", "avoidance_A2_static_multi"],
            window_size=50,
            min_successes=31,
            max_collision_rate=0.30,
        )
        for episode in range(1, 51):
            controller.observe(
                {
                    "trainer": trainer,
                    "episode_count": episode,
                    "episode_metrics": {
                        "curriculum_env": "avoidance_A1_static_single",
                        "episode_steps": 20,
                        "episode_success": 1.0 if episode <= 31 else 0.0,
                        "episode_collision": 1.0 if episode <= 14 else 0.0,
                    },
                    "training_counters": {},
                }
            )
        self.assertEqual(controller.current_env, "avoidance_A2_static_multi")
        self.assertTrue(controller.last_state["promotion_triggered"])

    def test_obstacle_observation_uses_relative_xy_and_nearest_four(self) -> None:
        scenario = build_planning_scenario(curriculum_env="avoidance_A4")
        env = PlanningSwarmEnv(scenario=scenario)
        env.reset()
        env._position[:] = np.array([0.0, 0.0], dtype=np.float32)
        env._obstacle_positions = np.array(
            [
                [0.1, 0.0],
                [0.2, 0.0],
                [0.3, 0.0],
                [0.4, 0.0],
                [0.5, 0.0],
                [0.6, 0.0],
            ],
            dtype=np.float32,
        )
        env._obstacle_radii = np.full(6, 0.08, dtype=np.float32)
        env._obstacle_velocities = np.array(
            [
                [0.01, 0.0],
                [0.02, 0.0],
                [0.03, 0.0],
                [0.04, 0.0],
                [0.05, 0.0],
                [0.06, 0.0],
            ],
            dtype=np.float32,
        )
        obs = env._make_observation()
        self.assertEqual(obs["obstacles"].shape[1], 5)
        self.assertEqual(int(obs["obstacle_mask"].sum()), 4)
        self.assertTrue((obs["obstacles"][:4, 0] == np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)).all())
        self.assertTrue((obs["obstacles"][:4, 2] == np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)).all())
        self.assertTrue((obs["obstacles"][:4, 4] == np.array([0.08, 0.08, 0.08, 0.08], dtype=np.float32)).all())

    def test_neighbor_observation_uses_nearest_two_with_relative_velocity_and_radius(self) -> None:
        scenario = replace(build_planning_scenario(curriculum_env="cooperation_C1"), max_neighbors=2, num_uavs=3)
        env = PlanningSwarmEnv(scenario=scenario)
        env.reset()
        env._position[:] = np.array([0.0, 0.0], dtype=np.float32)
        env._velocity[:] = np.array([0.1, -0.2], dtype=np.float32)
        env._neighbor_positions_state = np.array(
            [
                [0.6, 0.0],
                [0.2, 0.0],
                [0.4, 0.0],
            ],
            dtype=np.float32,
        )
        env._neighbor_velocities = np.array(
            [
                [0.6, 0.1],
                [0.4, 0.2],
                [0.5, 0.3],
            ],
            dtype=np.float32,
        )
        obs = env._make_observation()
        self.assertEqual(obs["neighbors"].shape, (2, 5))
        self.assertEqual(int(obs["neighbor_mask"].sum()), 2)
        self.assertTrue(np.allclose(obs["neighbors"][0], np.array([0.2, 0.0, 0.3, 0.4, 0.1], dtype=np.float32)))
        self.assertTrue(np.allclose(obs["neighbors"][1], np.array([0.4, 0.0, 0.4, 0.5, 0.1], dtype=np.float32)))

    def test_env_limits_velocity_change_by_max_acceleration(self) -> None:
        scenario = build_planning_scenario(curriculum_env="guidance_G1")
        env = PlanningSwarmEnv(scenario=scenario, delta_v_max=2.4)
        env.reset()
        _, _, _, info = env.step([0.8, -0.8])
        expected_delta = 2.4 / 24.0
        expected_velocity = np.array([0.8, -0.8], dtype=np.float32)
        expected_velocity = expected_velocity / np.linalg.norm(expected_velocity) * expected_delta
        self.assertTrue(np.allclose(env._velocity, expected_velocity, atol=1e-6))
        self.assertAlmostEqual(float(np.linalg.norm(env._velocity)), expected_delta, places=6)
        self.assertAlmostEqual(float(env._position[0]), float(expected_velocity[0] / 24.0), places=6)
        self.assertAlmostEqual(float(env._position[1]), float(expected_velocity[1] / 24.0), places=6)
        self.assertAlmostEqual(float(info["velocity_delta_norm"]), expected_delta, places=6)
        self.assertEqual(float(info["acceleration_clipped"]), 1.0)
        self.assertEqual(env.physics_hz, 240)
        self.assertEqual(env.control_hz, 48)
        self.assertEqual(env.action_hz, 24)

    def test_env_does_not_flip_velocity_instantly_on_opposite_actions(self) -> None:
        scenario = build_planning_scenario(curriculum_env="guidance_G1")
        env = PlanningSwarmEnv(scenario=scenario, delta_v_max=2.4)
        env.reset()
        env.step([0.8, 0.0])
        forward_velocity = env._velocity.copy()
        _, _, _, info = env.step([-0.8, 0.0])
        self.assertGreater(float(forward_velocity[0]), 0.0)
        self.assertGreater(float(env._velocity[0]), -0.01)
        self.assertLess(float(env._velocity[0]), float(forward_velocity[0]))
        self.assertAlmostEqual(float(info["velocity_delta_norm"]), 2.4 / 24.0, places=6)
        self.assertEqual(float(info["acceleration_clipped"]), 1.0)

    def test_a1_single_step_acceleration_limit_matches_action_rate(self) -> None:
        scenario = build_planning_scenario(curriculum_env="avoidance_A1")
        env = PlanningSwarmEnv(scenario=scenario, delta_v_max=2.4, action_hz=48, control_hz=48)
        env.reset()
        obs, _, _, info = env.step([0.8, 0.0])
        self.assertAlmostEqual(float(info["velocity_delta_norm"]), 2.4 / 48.0, places=6)
        self.assertAlmostEqual(float(env._velocity[0]), 2.4 / 48.0, places=6)
        self.assertAlmostEqual(float(obs["self_state"][2]), (2.4 / 48.0) / env.action_limit, places=6)
        self.assertAlmostEqual(float(info["executed_velocity"][0]), 2.4 / 48.0, places=6)
        self.assertAlmostEqual(float(info["commanded_action"][0]), 0.8, places=6)

    def test_a1_gmix_softtimeout_single_step_acceleration_limit_matches_updated_rate(self) -> None:
        scenario = build_planning_scenario(curriculum_env="avoidance_A1_gmix_softtimeout")
        env = PlanningSwarmEnv(scenario=scenario, delta_v_max=9.6, action_hz=48, control_hz=48)
        env.reset()
        obs, _, _, info = env.step([0.8, 0.0])
        self.assertAlmostEqual(float(info["velocity_delta_norm"]), 9.6 / 48.0, places=6)
        self.assertAlmostEqual(float(env._velocity[0]), 9.6 / 48.0, places=6)
        self.assertAlmostEqual(float(obs["self_state"][2]), (9.6 / 48.0) / env.action_limit, places=6)
        self.assertAlmostEqual(float(info["executed_velocity"][0]), 9.6 / 48.0, places=6)

    def test_empty_guidance_env_supports_square_boundary_tracking(self) -> None:
        scenario = build_planning_scenario(curriculum_env="guidance_G1")
        env = PlanningSwarmEnv(scenario=scenario, delta_v_max=1e6)
        env.reset()
        xmin, xmax, ymin, ymax = env._workspace_bounds()
        env._position = np.array([xmin + 0.2, ymin + 0.2], dtype=np.float32)
        env._velocity = np.zeros(2, dtype=np.float32)
        env._previous_action = np.zeros(2, dtype=np.float32)
        env._target_position = np.array([0.0, 0.0], dtype=np.float32)
        env.trajectory = [env._position.tolist()]
        env.target_trajectory = [env._target_position.tolist()]
        side_steps = 48
        commands = (
            np.array([0.5, 0.0], dtype=np.float32),
            np.array([0.0, 0.5], dtype=np.float32),
            np.array([-0.5, 0.0], dtype=np.float32),
            np.array([0.0, -0.5], dtype=np.float32),
        )
        for action in commands:
            for _ in range(side_steps):
                _, _, done, info = env.step(action)
                self.assertFalse(done)
                self.assertEqual(info["out_of_bounds"], 0.0)
        self.assertAlmostEqual(float(env._position[0]), xmin + 0.2, places=5)
        self.assertAlmostEqual(float(env._position[1]), ymin + 0.2, places=5)

    def test_hold_success_triggers_immediately_on_target_entry(self) -> None:
        scenario = build_planning_scenario(curriculum_env="guidance_G1")
        env = PlanningSwarmEnv(scenario=scenario)
        env.reset()
        env._target_position = np.array([0.01, 0.0], dtype=np.float32)
        _, _, done, info = env.step([0.0, 0.0])
        self.assertTrue(done)
        self.assertEqual(info["success"], 1.0)

    def test_cooperation_spawn_and_target_regions_follow_constraints(self) -> None:
        scenario = build_planning_scenario(curriculum_env="cooperation_C1")
        env = PlanningSwarmEnv(scenario=scenario)
        env.reset()
        positions = np.vstack([env._position, env._neighbor_positions_state])
        left, right, bottom, top = scenario.spawn_region
        for point in positions:
            self.assertGreaterEqual(float(point[0]), left)
            self.assertLessEqual(float(point[0]), right)
            self.assertGreaterEqual(float(point[1]), bottom)
            self.assertLessEqual(float(point[1]), top)
        min_distance = min(
            float(np.linalg.norm(positions[i] - positions[j]))
            for i in range(len(positions))
            for j in range(i + 1, len(positions))
        )
        self.assertGreaterEqual(min_distance, scenario.spawn_min_separation)
        t_left, t_right, t_bottom, t_top = scenario.target_region_bounds
        self.assertGreaterEqual(float(env._target_position[0]), t_left)
        self.assertGreaterEqual(float(env._target_position[1]), t_bottom)
        self.assertLessEqual(float(env._target_position[0]), t_right)
        self.assertLessEqual(float(env._target_position[1]), t_top)

    def test_scheduler_uses_ema_thresholds_and_stops_at_a3(self) -> None:
        scheduler = CurriculumScheduler(sequence=MIN_EFFECTIVE_SEQUENCE, ema_alpha=1.0, promotion_grace_episodes=2)
        self.assertEqual(scheduler.curriculum_env, "guidance_G1")
        profile = {
            "reward": 1.0,
            "success_rate": 0.95,
            "collision_rate": 0.0,
            "timeout_rate": 0.0,
            "hold_completion": 1.0,
            "encircle_completion": 1.0,
        }
        while scheduler.curriculum_env != "avoidance_A3":
            scheduler.update(profile)
        self.assertEqual(scheduler.curriculum_env, "avoidance_A3")
        self.assertEqual(scheduler.current_stage, 1)

    def test_scheduler_blocks_rollback_during_grace_period_then_allows_it(self) -> None:
        scheduler = CurriculumScheduler(sequence=MIN_EFFECTIVE_SEQUENCE, ema_alpha=1.0, promotion_grace_episodes=2)
        good = {
            "reward": 1.0,
            "success_rate": 0.95,
            "collision_rate": 0.0,
            "timeout_rate": 0.0,
            "hold_completion": 1.0,
            "encircle_completion": 1.0,
        }
        bad = {
            "reward": -1.0,
            "success_rate": 0.0,
            "collision_rate": 1.0,
            "timeout_rate": 1.0,
            "hold_completion": 0.0,
            "encircle_completion": 0.0,
        }
        scheduler.update(good)
        scheduler.update(good)
        self.assertEqual(scheduler.curriculum_env, "avoidance_A1")
        self.assertGreater(scheduler.grace_remaining, 0)
        scheduler.update(bad)
        self.assertEqual(scheduler.curriculum_env, "avoidance_A1")
        scheduler.update(bad)
        self.assertEqual(scheduler.curriculum_env, "guidance_G2")

    def test_scene_snapshot_contains_plotting_fields(self) -> None:
        scenario = build_planning_scenario(curriculum_env="cooperation_C4")
        env = PlanningSwarmEnv(scenario=scenario)
        env.reset()
        snapshot = env.get_scene_snapshot()
        self.assertIn("workspace_bounds", snapshot)
        self.assertIn("spawn_region", snapshot)
        self.assertIn("target_region_bounds", snapshot)
        self.assertIn("obstacle_radii_initial", snapshot)
        self.assertIn("uav_positions_initial", snapshot)
        self.assertIn("target_trajectory_hint", snapshot)
        self.assertGreaterEqual(len(snapshot["target_trajectory_hint"]), 1)

    def test_reset_and_step_keep_structured_observation_contract(self) -> None:
        scenario = build_planning_scenario(curriculum_env="avoidance_A1")
        env = PlanningSwarmEnv(scenario=scenario)
        obs = env.reset()
        self.assertSetEqual(
            set(obs.keys()),
            {"self_state", "obstacles", "neighbors", "obstacle_mask", "neighbor_mask"},
        )
        self.assertEqual(tuple(obs["self_state"].shape), (6,))
        self.assertEqual(obs["obstacles"].shape, (scenario.max_obstacles, 5))
        self.assertEqual(obs["neighbors"].shape, (scenario.max_neighbors, 5))
        next_obs, reward, done, info = env.step(np.array([0.05, -0.02], dtype=np.float32))
        self.assertSetEqual(set(next_obs.keys()), set(obs.keys()))
        self.assertIsInstance(float(reward), float)
        self.assertIsInstance(done, bool)
        for key in ("stage_name", "curriculum_env", "success", "failure_reason", "occupancy_error", "formation_error", "risk"):
            self.assertIn(key, info)
        for key in ("target_distance", "obstacle_center_distance", "obstacle_clearance", "neighbor_center_distance", "neighbor_clearance", "physics_hz", "control_hz", "action_hz"):
            self.assertIn(key, info)
        self.assertIn("boundary_distance", info)

    def test_episode_trace_matches_rollout_length(self) -> None:
        scenario = build_planning_scenario(curriculum_env="avoidance_A1")
        env = PlanningSwarmEnv(scenario=scenario)
        env.reset()
        rollout_steps = 6
        for _ in range(rollout_steps):
            _, _, done, _ = env.step(np.array([0.02, 0.01], dtype=np.float32))
            if done:
                break
        trace = env.get_episode_trace()
        self.assertEqual(len(trace["trajectory"]), env.step_count + 1)
        self.assertEqual(len(trace["commanded_action_history"]), env.step_count)
        self.assertEqual(len(trace["executed_velocity_history"]), env.step_count)
        self.assertEqual(len(trace["velocity_delta_history"]), env.step_count)
        self.assertEqual(len(trace["acceleration_clipped_history"]), env.step_count)
        self.assertEqual(len(trace["target_trajectory"]), env.step_count + 1)
        self.assertGreaterEqual(len(trace["trajectory"]), 2)
        self.assertEqual(trace["curriculum_env"], "avoidance_A1")

    def test_avoidance_a1_mixes_blocking_and_nonblocking_layouts(self) -> None:
        scenario = build_planning_scenario(curriculum_env="avoidance_A1")
        env = PlanningSwarmEnv(scenario=scenario)
        layout_counts = {"path_center_blocking": 0, "path_center_offset_nonblocking": 0}
        for _ in range(120):
            env.reset()
            self.assertGreaterEqual(len(env._obstacle_positions), 1)
            layout_counts[env._episode_obstacle_layout] = layout_counts.get(env._episode_obstacle_layout, 0) + 1
        blocking_ratio = layout_counts["path_center_blocking"] / 120.0
        nonblocking_ratio = layout_counts["path_center_offset_nonblocking"] / 120.0
        self.assertGreater(blocking_ratio, 0.25)
        self.assertLess(blocking_ratio, 0.55)
        self.assertGreater(nonblocking_ratio, 0.45)

    def test_soft_timeout_variant_does_not_terminate_on_timeout(self) -> None:
        scenario = build_planning_scenario(curriculum_env="avoidance_A1_gmix_softtimeout")
        env = PlanningSwarmEnv(scenario=scenario, action_hz=48, control_hz=48)
        env.reset()
        env.step_count = env.horizon - 1
        _, _, done, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        self.assertFalse(done)
        self.assertEqual(info["timeout"], 1.0)
        self.assertEqual(info["soft_timeout_active"], 1.0)
        self.assertEqual(info["episode_step_cap_hit"], 0.0)
        self.assertEqual(env.failure_reason, "timeout")

    def test_guidance_reward_uses_distance_bleed_and_large_success_bonus(self) -> None:
        breakdown = compute_planning_reward(
            stage_name="guidance",
            occupancy_error=0.2,
            previous_occupancy_error=0.2,
            formation_error=0.0,
            angle_error=0.0,
            target_distance=0.2,
            obstacle_distance=10.0,
            neighbor_distance=10.0,
            boundary_distance=10.0,
            obstacle_margin=9.8,
            neighbor_margin=9.8,
            collision=False,
            out_of_bounds=False,
            action=np.zeros(2, dtype=np.float32),
            success=True,
        )
        self.assertAlmostEqual(breakdown.target_reward, -0.02, places=6)
        self.assertAlmostEqual(breakdown.success_bonus, 20.0, places=6)
        self.assertAlmostEqual(breakdown.total_reward, 20.0, places=6)

    def test_guidance_collision_returns_hard_failure_reward(self) -> None:
        breakdown = compute_planning_reward(
            stage_name="guidance",
            occupancy_error=1.5,
            previous_occupancy_error=1.5,
            formation_error=0.0,
            angle_error=0.0,
            target_distance=1.5,
            obstacle_distance=10.0,
            neighbor_distance=10.0,
            boundary_distance=10.0,
            obstacle_margin=9.8,
            neighbor_margin=9.8,
            collision=True,
            out_of_bounds=True,
            action=np.array([0.5, 0.0], dtype=np.float32),
            success=False,
        )
        self.assertEqual(breakdown.total_reward, -10.0)
        self.assertEqual(breakdown.success_bonus, 0.0)

    def test_avoidance_reward_applies_linear_near_field_penalty(self) -> None:
        breakdown = compute_planning_reward(
            stage_name="avoidance",
            occupancy_error=0.8,
            previous_occupancy_error=0.8,
            formation_error=0.0,
            angle_error=0.0,
            target_distance=0.8,
            obstacle_distance=0.12,
            neighbor_distance=0.18,
            boundary_distance=10.0,
            obstacle_clearance=0.12,
            neighbor_clearance=0.18,
            obstacle_margin=0.0,
            neighbor_margin=0.0,
            collision=False,
            out_of_bounds=False,
            action=np.zeros(2, dtype=np.float32),
            success=False,
        )
        self.assertAlmostEqual(breakdown.target_reward, -0.08, places=6)
        self.assertAlmostEqual(breakdown.avoidance_reward, -1.0, places=6)
        self.assertAlmostEqual(breakdown.total_reward, -1.08, places=6)

    def test_clearance_uses_body_radii(self) -> None:
        scenario = build_planning_scenario(curriculum_env="avoidance_A1")
        env = PlanningSwarmEnv(scenario=scenario)
        env.reset()
        env._position = np.array([0.0, 0.0], dtype=np.float32)
        env._obstacle_positions = np.array([[0.30, 0.0]], dtype=np.float32)
        env._obstacle_radii = np.array([0.08], dtype=np.float32)
        obstacle_center_distance, obstacle_radius, obstacle_clearance = env._nearest_obstacle()
        self.assertAlmostEqual(obstacle_center_distance, 0.30, places=6)
        self.assertAlmostEqual(obstacle_radius, 0.18, places=6)
        self.assertAlmostEqual(obstacle_clearance, 0.12, places=6)
        env._neighbor_positions_state = np.array([[0.26, 0.0]], dtype=np.float32)
        neighbor_center_distance, neighbor_clearance = env._nearest_neighbor_distance()
        self.assertAlmostEqual(neighbor_center_distance, 0.26, places=6)
        self.assertAlmostEqual(neighbor_clearance, 0.06, places=6)

    def test_boundary_repulsion_features_enter_self_state(self) -> None:
        scenario = build_planning_scenario(curriculum_env="guidance_G1")
        env = PlanningSwarmEnv(scenario=scenario)
        env.reset()
        env._position = np.array([0.6, -0.4], dtype=np.float32)
        obs = env._make_observation()
        self.assertEqual(obs["self_state"].shape[0], 6)
        wall_repulsion_x, wall_repulsion_y = obs["self_state"][4:]
        self.assertAlmostEqual(float(wall_repulsion_x), -0.5, places=6)
        self.assertAlmostEqual(float(wall_repulsion_y), 0.0, places=6)

    def test_guidance_samples_targets_within_new_distance_band(self) -> None:
        scenario = build_planning_scenario(curriculum_env="guidance_G1")
        env = PlanningSwarmEnv(scenario=scenario)
        for _ in range(20):
            env.reset()
            target_distance = float(np.linalg.norm(env._target_position - env._position))
            self.assertGreaterEqual(target_distance, 0.25 - 1e-6)
            self.assertLessEqual(target_distance, 0.5 + 1e-6)

    def test_reward_purity_timeout_is_worse_than_early_crash(self) -> None:
        timeout_reward = compute_planning_reward(
            stage_name="avoidance",
            curriculum_env="avoidance_A1",
            occupancy_error=0.6,
            previous_occupancy_error=0.55,
            formation_error=0.0,
            angle_error=0.0,
            target_distance=0.6,
            obstacle_distance=0.20,
            neighbor_distance=10.0,
            obstacle_clearance=0.20,
            neighbor_clearance=10.0,
            previous_obstacle_clearance=0.18,
            previous_neighbor_clearance=10.0,
            obstacle_margin=0.0,
            neighbor_margin=0.0,
            collision=False,
            out_of_bounds=False,
            timeout=True,
            action=np.zeros(2, dtype=np.float32),
            success=False,
        ).total_reward
        crash_reward = compute_planning_reward(
            stage_name="avoidance",
            curriculum_env="avoidance_A1",
            occupancy_error=0.6,
            previous_occupancy_error=0.55,
            formation_error=0.0,
            angle_error=0.0,
            target_distance=0.6,
            obstacle_distance=0.06,
            neighbor_distance=10.0,
            obstacle_clearance=0.06,
            neighbor_clearance=10.0,
            previous_obstacle_clearance=0.09,
            previous_neighbor_clearance=10.0,
            obstacle_margin=0.0,
            neighbor_margin=0.0,
            collision=True,
            out_of_bounds=False,
            timeout=False,
            action=np.zeros(2, dtype=np.float32),
            success=False,
        ).total_reward
        success_reward = compute_planning_reward(
            stage_name="avoidance",
            curriculum_env="avoidance_A1",
            occupancy_error=0.05,
            previous_occupancy_error=0.10,
            formation_error=0.0,
            angle_error=0.0,
            target_distance=0.05,
            obstacle_distance=0.30,
            neighbor_distance=10.0,
            obstacle_clearance=0.30,
            neighbor_clearance=10.0,
            previous_obstacle_clearance=0.22,
            previous_neighbor_clearance=10.0,
            obstacle_margin=0.0,
            neighbor_margin=0.0,
            collision=False,
            out_of_bounds=False,
            timeout=False,
            action=np.array([0.08, 0.02], dtype=np.float32),
            success=True,
        ).total_reward
        self.assertEqual(timeout_reward, -6.0)
        self.assertEqual(crash_reward, -15.0)
        self.assertEqual(success_reward, 50.0)
        self.assertGreater(timeout_reward, crash_reward)

    def test_avoidance_collision_returns_hard_failure_reward(self) -> None:
        breakdown = compute_planning_reward(
            stage_name="avoidance",
            occupancy_error=0.8,
            previous_occupancy_error=0.8,
            formation_error=0.0,
            angle_error=0.0,
            target_distance=0.8,
            obstacle_distance=0.1,
            neighbor_distance=10.0,
            obstacle_margin=0.0,
            neighbor_margin=9.8,
            collision=True,
            out_of_bounds=False,
            action=np.array([0.5, 0.0], dtype=np.float32),
            success=False,
        )
        self.assertEqual(breakdown.total_reward, -10.0)

    def test_avoidance_a1_uses_segmented_penalties_and_positive_improvement_terms(self) -> None:
        mild = compute_planning_reward(
            stage_name="avoidance",
            curriculum_env="avoidance_A1",
            occupancy_error=0.45,
            previous_occupancy_error=0.55,
            formation_error=0.0,
            angle_error=0.0,
            target_distance=0.45,
            obstacle_distance=0.24,
            neighbor_distance=10.0,
            obstacle_clearance=0.06,
            neighbor_clearance=10.0,
            previous_obstacle_clearance=0.03,
            previous_neighbor_clearance=10.0,
            obstacle_margin=0.0,
            neighbor_margin=0.0,
            collision=False,
            out_of_bounds=False,
            timeout=False,
            action=np.array([0.1, 0.02], dtype=np.float32),
            success=False,
        )
        critical = compute_planning_reward(
            stage_name="avoidance",
            curriculum_env="avoidance_A1",
            occupancy_error=0.45,
            previous_occupancy_error=0.55,
            formation_error=0.0,
            angle_error=0.0,
            target_distance=0.45,
            obstacle_distance=0.20,
            neighbor_distance=10.0,
            obstacle_clearance=0.01,
            neighbor_clearance=10.0,
            previous_obstacle_clearance=0.03,
            previous_neighbor_clearance=10.0,
            obstacle_margin=0.0,
            neighbor_margin=0.0,
            collision=False,
            out_of_bounds=False,
            timeout=False,
            action=np.array([0.78, 0.02], dtype=np.float32),
            success=False,
        )
        self.assertGreater(mild.risk_drop_reward, 0.0)
        self.assertGreater(mild.clearance_gain_reward, 0.0)
        self.assertGreater(mild.detour_trend_reward, 0.0)
        self.assertEqual(mild.near_collision_penalty, 0.0)
        self.assertEqual(mild.severe_near_collision_penalty, 0.0)
        self.assertLess(critical.near_collision_penalty, 0.0)
        self.assertLess(critical.severe_near_collision_penalty, 0.0)
        self.assertLess(critical.critical_collision_margin_penalty, 0.0)
        self.assertGreater(critical.action_saturation_penalty, mild.action_saturation_penalty)


if __name__ == "__main__":
    unittest.main()
