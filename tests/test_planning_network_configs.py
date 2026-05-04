from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import torch

from lc.envs.scenarios import build_planning_scenario
from lc.planning.configs import build_planning_network_config
from lc.planning.critics import build_local_risk_critic, build_mlp_critic
from lc.planning.envs import PlanningSwarmEnv
from lc.planning.models import build_mlp_policy, build_transformer_actor, flatten_structured_observation


class TestPlanningNetworkConfigs(unittest.TestCase):
    def test_env_observation_contains_masks(self) -> None:
        scenario = build_planning_scenario(curriculum_env="cooperation_C1")
        env = PlanningSwarmEnv(scenario=scenario)
        obs = env.reset()
        self.assertIn("obstacle_mask", obs)
        self.assertIn("neighbor_mask", obs)
        self.assertEqual(obs["self_state"].shape[0], 6)
        self.assertEqual(obs["obstacles"].shape[1], 5)

    def test_obstacle_mask_limits_visible_tokens_to_four(self) -> None:
        scenario = replace(build_planning_scenario(curriculum_env="avoidance_A4"), max_obstacles=4)
        env = PlanningSwarmEnv(scenario=scenario)
        env.reset()
        env._position[:] = np.zeros(2, dtype=np.float32)
        env._obstacle_positions = np.array(
            [[0.1 * (index + 1), 0.0] for index in range(6)],
            dtype=np.float32,
        )
        env._obstacle_velocities = np.zeros((6, 2), dtype=np.float32)
        env._obstacle_radii = np.full(6, 0.08, dtype=np.float32)

        obs = env._make_observation()

        self.assertEqual(int(obs["obstacle_mask"].sum()), 4)
        self.assertTrue(np.allclose(obs["obstacles"][0], np.array([0.1, 0.0, 0.0, 0.0, 0.08], dtype=np.float32)))

    def test_mlp_and_transformer_forward_shapes(self) -> None:
        scenario = build_planning_scenario(curriculum_env="avoidance_A2")
        mlp_cfg = build_planning_network_config("mlp_large")
        env = PlanningSwarmEnv(
            scenario=replace(scenario, max_obstacles=mlp_cfg.max_obstacles, max_neighbors=mlp_cfg.max_neighbors),
            self_dim=mlp_cfg.self_dim,
            obstacle_dim=mlp_cfg.obstacle_dim,
            neighbor_dim=mlp_cfg.neighbor_dim,
        )
        obs = env.reset()
        mlp_actor = build_mlp_policy(mlp_cfg)
        mlp_critic = build_mlp_critic(mlp_cfg)
        flat = torch.tensor(flatten_structured_observation(obs, use_mask_features=mlp_cfg.use_mask_features), dtype=torch.float32).unsqueeze(0)
        mlp_action = mlp_actor(flat)
        mlp_q = mlp_critic(flat, mlp_action)
        self.assertEqual(tuple(mlp_action.shape), (1, 2))
        self.assertEqual(tuple(mlp_q.shape), (1, 1))

        transformer_cfg = build_planning_network_config("transformer_large")
        transformer_actor = build_transformer_actor(transformer_cfg)
        local_risk_critic = build_local_risk_critic(transformer_cfg)
        structured = {key: torch.tensor(value, dtype=torch.float32).unsqueeze(0) for key, value in obs.items()}
        avoid_action, final_action = transformer_actor(structured)
        q_value = local_risk_critic(structured, final_action)
        self.assertEqual(tuple(avoid_action.shape), (1, 2))
        self.assertEqual(tuple(final_action.shape), (1, 2))
        self.assertEqual(tuple(q_value.shape), (1, 1))
        self.assertEqual(transformer_actor.config.self_dim, 6)
        self.assertEqual(local_risk_critic.config.self_dim, 6)

    def test_transformer_stage_modes_bypass_collab_until_cooperation(self) -> None:
        scenario = build_planning_scenario(curriculum_env="cooperation_C1")
        cfg = build_planning_network_config("transformer_large")
        env = PlanningSwarmEnv(
            scenario=replace(scenario, max_obstacles=cfg.max_obstacles, max_neighbors=cfg.max_neighbors),
            self_dim=cfg.self_dim,
            obstacle_dim=cfg.obstacle_dim,
            neighbor_dim=cfg.neighbor_dim,
        )
        obs = env.reset()
        actor = build_transformer_actor(cfg)
        critic = build_local_risk_critic(cfg)
        structured = {key: torch.tensor(value, dtype=torch.float32).unsqueeze(0) for key, value in obs.items()}

        avoid_action, guidance_action = actor(structured, stage_mode="guidance")
        cooperation_avoid, cooperation_action = actor(structured, stage_mode="cooperation")

        self.assertTrue(torch.allclose(avoid_action, guidance_action))
        self.assertEqual(tuple(cooperation_avoid.shape), (1, 2))
        self.assertEqual(tuple(critic(structured, avoid_action, stage_mode="guidance").shape), (1, 1))
        self.assertEqual(tuple(critic(structured, cooperation_action, stage_mode="cooperation").shape), (1, 1))

    def test_guidance_stage_forces_obstacle_and_neighbor_masks_off(self) -> None:
        cfg = build_planning_network_config("transformer_large")
        actor = build_transformer_actor(cfg)
        critic = build_local_risk_critic(cfg)
        base_observation = {
            "self_state": torch.zeros((1, cfg.self_dim), dtype=torch.float32),
            "obstacles": torch.zeros((1, cfg.max_obstacles, cfg.obstacle_dim), dtype=torch.float32),
            "neighbors": torch.zeros((1, cfg.max_neighbors, cfg.neighbor_dim), dtype=torch.float32),
            "obstacle_mask": torch.zeros((1, cfg.max_obstacles), dtype=torch.float32),
            "neighbor_mask": torch.zeros((1, cfg.max_neighbors), dtype=torch.float32),
        }
        noisy_guidance = {key: value.clone() for key, value in base_observation.items()}
        noisy_guidance["obstacles"].normal_(mean=0.0, std=2.0)
        noisy_guidance["neighbors"].normal_(mean=0.0, std=2.0)
        noisy_guidance["obstacle_mask"].fill_(1.0)
        noisy_guidance["neighbor_mask"].fill_(1.0)

        _, action_clean = actor(base_observation, stage_mode="guidance")
        _, action_noisy = actor(noisy_guidance, stage_mode="guidance")
        q_clean = critic(base_observation, action_clean, stage_mode="guidance")
        q_noisy = critic(noisy_guidance, action_clean, stage_mode="guidance")

        self.assertTrue(torch.allclose(action_clean, action_noisy, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(q_clean, q_noisy, atol=1e-5, rtol=1e-5))

    def test_masked_tokens_do_not_affect_transformer_outputs(self) -> None:
        cfg = build_planning_network_config("transformer_large")
        actor = build_transformer_actor(cfg)
        critic = build_local_risk_critic(cfg)
        observation = {
            "self_state": torch.zeros((1, cfg.self_dim), dtype=torch.float32),
            "obstacles": torch.zeros((1, cfg.max_obstacles, cfg.obstacle_dim), dtype=torch.float32),
            "neighbors": torch.zeros((1, cfg.max_neighbors, cfg.neighbor_dim), dtype=torch.float32),
            "obstacle_mask": torch.zeros((1, cfg.max_obstacles), dtype=torch.float32),
            "neighbor_mask": torch.zeros((1, cfg.max_neighbors), dtype=torch.float32),
        }
        masked_observation = {key: value.clone() for key, value in observation.items()}
        masked_observation["obstacles"].fill_(123.0)
        masked_observation["neighbors"].fill_(-77.0)

        _, action_a = actor(observation, stage_mode="cooperation")
        _, action_b = actor(masked_observation, stage_mode="cooperation")
        q_a = critic(observation, action_a, stage_mode="cooperation")
        q_b = critic(masked_observation, action_a, stage_mode="cooperation")

        self.assertTrue(torch.allclose(action_a, action_b, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(q_a, q_b, atol=1e-5, rtol=1e-5))

    def test_transformer_actor_uses_small_output_init_and_layernorm(self) -> None:
        cfg = build_planning_network_config("transformer_large")
        actor = build_transformer_actor(cfg)
        self.assertIsInstance(actor.avoidance_backbone.self_norm, torch.nn.LayerNorm)
        self.assertIsInstance(actor.avoidance_backbone.action_head.norm, torch.nn.LayerNorm)
        final_linear = actor.avoidance_backbone.action_head.output
        self.assertLessEqual(float(final_linear.weight.max().item()), 1.1e-4)
        self.assertGreaterEqual(float(final_linear.weight.min().item()), -1.1e-4)
        self.assertTrue(torch.allclose(final_linear.bias, torch.zeros_like(final_linear.bias)))

    def test_transformer_guidance_init_actions_start_small(self) -> None:
        cfg = build_planning_network_config("transformer_large")
        actor = build_transformer_actor(cfg)
        observation = {
            "self_state": torch.tensor([[0.35 / 1.5, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            "obstacles": torch.zeros((1, cfg.max_obstacles, cfg.obstacle_dim), dtype=torch.float32),
            "neighbors": torch.zeros((1, cfg.max_neighbors, cfg.neighbor_dim), dtype=torch.float32),
            "obstacle_mask": torch.zeros((1, cfg.max_obstacles), dtype=torch.float32),
            "neighbor_mask": torch.zeros((1, cfg.max_neighbors), dtype=torch.float32),
        }
        _, action = actor(observation, stage_mode="guidance")
        self.assertLess(float(action.abs().max().item()), 0.02)

    def test_guidance_self_only_path_matches_safe_feature_to_self_token(self) -> None:
        cfg = build_planning_network_config("transformer_large")
        actor = build_transformer_actor(cfg)
        observation = {
            "self_state": torch.tensor([[0.3, -0.2, 0.1, -0.1, 0.2, -0.3]], dtype=torch.float32),
            "obstacles": torch.randn((1, cfg.max_obstacles, cfg.obstacle_dim), dtype=torch.float32),
            "neighbors": torch.randn((1, cfg.max_neighbors, cfg.neighbor_dim), dtype=torch.float32),
            "obstacle_mask": torch.ones((1, cfg.max_obstacles), dtype=torch.float32),
            "neighbor_mask": torch.ones((1, cfg.max_neighbors), dtype=torch.float32),
        }
        actor(observation, stage_mode="guidance")
        features = actor.last_feature_snapshot
        self.assertTrue(torch.allclose(features["safe_feature"], features["self_token"], atol=1e-6, rtol=1e-6))

    def test_self_token_changes_with_target_direction(self) -> None:
        cfg = build_planning_network_config("transformer_large")
        actor = build_transformer_actor(cfg)
        base = {
            "self_state": torch.tensor([[0.3, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            "obstacles": torch.zeros((1, cfg.max_obstacles, cfg.obstacle_dim), dtype=torch.float32),
            "neighbors": torch.zeros((1, cfg.max_neighbors, cfg.neighbor_dim), dtype=torch.float32),
            "obstacle_mask": torch.zeros((1, cfg.max_obstacles), dtype=torch.float32),
            "neighbor_mask": torch.zeros((1, cfg.max_neighbors), dtype=torch.float32),
        }
        actor(base, stage_mode="guidance")
        token_a = actor.last_feature_snapshot["self_token"].clone()
        flipped = {key: value.clone() for key, value in base.items()}
        flipped["self_state"][0, 0] = -0.3
        actor(flipped, stage_mode="guidance")
        token_b = actor.last_feature_snapshot["self_token"].clone()
        self.assertGreater(float(torch.norm(token_a - token_b).item()), 1e-3)

    def test_boundary_repulsion_features_encode_wall_direction(self) -> None:
        scenario = build_planning_scenario(curriculum_env="guidance_G1")
        env = PlanningSwarmEnv(scenario=scenario)
        env.reset()
        env._position = np.array([0.6, 0.0], dtype=np.float32)
        obs_right = env._make_observation()
        env._position = np.array([-0.6, 0.0], dtype=np.float32)
        obs_left = env._make_observation()
        env._position = np.array([0.0, 0.6], dtype=np.float32)
        obs_top = env._make_observation()
        env._position = np.array([0.0, -0.6], dtype=np.float32)
        obs_bottom = env._make_observation()
        self.assertLess(float(obs_right["self_state"][4]), 0.0)
        self.assertGreater(float(obs_left["self_state"][4]), 0.0)
        self.assertLess(float(obs_top["self_state"][5]), 0.0)
        self.assertGreater(float(obs_bottom["self_state"][5]), 0.0)

    def test_obstacle_generation_respects_target_clearance_radius(self) -> None:
        scenario = replace(
            build_planning_scenario(curriculum_env="avoidance_A1"),
            num_obstacles=1,
            obstacle_count_range=(1, 1),
            obstacle_size_small_range=(1, 1),
            obstacle_size_medium_range=(0, 0),
            obstacle_size_large_range=(0, 0),
            obstacle_layout_modes=("wide_random",),
            obstacle_safe_buffer=0.16,
            target_clearance_radius=0.75,
        )
        env = PlanningSwarmEnv(scenario=scenario)
        for _ in range(20):
            env.reset()
            self.assertEqual(len(env._obstacle_positions), 1)
            center_distance = float(np.linalg.norm(env._obstacle_positions[0] - env._target_position))
            obstacle_clearance = float(center_distance - env._obstacle_radii[0])
            self.assertGreater(obstacle_clearance, env.scenario.target_clearance_radius)

    def test_path_center_layout_places_small_obstacle_between_spawn_and_target(self) -> None:
        scenario = replace(
            build_planning_scenario(curriculum_env="avoidance_A1"),
            num_obstacles=1,
            obstacle_count_range=(1, 1),
            obstacle_size_small_range=(1, 1),
            obstacle_size_medium_range=(0, 0),
            obstacle_size_large_range=(0, 0),
            obstacle_layout_modes=("path_center",),
            target_clearance_radius=0.5,
        )
        env = PlanningSwarmEnv(scenario=scenario, small_obstacle_radius=0.05)
        env.reset()
        env._position = np.zeros(2, dtype=np.float32)
        env._target_position = np.array([1.3, 0.0], dtype=np.float32)
        radii = np.array([0.05], dtype=np.float32)
        obstacle = env._sample_obstacle_positions("path_center", radii)[0]
        start = env._spawn_reference_point()
        target = env._target_position
        direction = target - start
        norm = float(np.linalg.norm(direction))
        self.assertGreater(norm, 1e-6)
        unit = direction / norm
        alpha = float(np.dot(obstacle - start, unit) / norm)
        lateral = float(abs(np.cross(np.append(unit, 0.0), np.append(obstacle - start, 0.0))[2]))
        clearance = float(np.linalg.norm(obstacle - target) - radii[0])

        self.assertGreater(alpha, 0.3)
        self.assertLess(alpha, 0.7)
        self.assertLess(lateral, 0.08)
        self.assertGreater(clearance, 0.5)

    def test_path_center_offset_layout_stays_between_spawn_and_target_with_lateral_jitter(self) -> None:
        scenario = replace(
            build_planning_scenario(curriculum_env="avoidance_A1"),
            num_obstacles=1,
            obstacle_count_range=(1, 1),
            obstacle_size_small_range=(1, 1),
            obstacle_size_medium_range=(0, 0),
            obstacle_size_large_range=(0, 0),
            obstacle_layout_modes=("path_center_offset",),
            target_clearance_radius=0.3,
        )
        env = PlanningSwarmEnv(scenario=scenario, small_obstacle_radius=0.06)
        env.reset()
        env._position = np.zeros(2, dtype=np.float32)
        env._target_position = np.array([1.3, 0.0], dtype=np.float32)
        radii = np.array([0.06], dtype=np.float32)
        obstacle = env._sample_obstacle_positions("path_center_offset", radii)[0]
        start = env._spawn_reference_point()
        target = env._target_position
        direction = target - start
        norm = float(np.linalg.norm(direction))
        unit = direction / norm
        alpha = float(np.dot(obstacle - start, unit) / norm)
        lateral = float(abs(np.cross(np.append(unit, 0.0), np.append(obstacle - start, 0.0))[2]))
        clearance = float(np.linalg.norm(obstacle - target) - radii[0])

        self.assertGreater(alpha, 0.42)
        self.assertLess(alpha, 0.58)
        self.assertGreater(lateral, 0.0)
        self.assertLess(lateral, 0.14)
        self.assertGreater(clearance, 0.30)

    def test_local_risk_critic_respects_q_head_dim(self) -> None:
        cfg = build_planning_network_config("transformer_large")
        cfg = replace(cfg, local_risk_critic=replace(cfg.local_risk_critic, q_head_dim=512))
        critic = build_local_risk_critic(cfg)
        self.assertEqual(critic.config.q_head_dim, 512)
        self.assertEqual(critic.q_head[0].in_features, critic.config.embed_dim * 3)
        self.assertEqual(critic.q_head[0].out_features, 512)

    def test_network_versions_expose_expected_modes(self) -> None:
        no_mask = build_planning_network_config("transformer_large_no_mask")
        self.assertEqual(no_mask.mask_mode, "zero_only")
        self.assertFalse(no_mask.mlp.use_mask_features)

        no_collab = build_planning_network_config("transformer_large_no_collab")
        self.assertTrue(no_collab.transformer.disable_collab_residual)

        guidance_attn = build_planning_network_config("transformer_large_guidance_attn")
        self.assertFalse(guidance_attn.transformer.guidance_self_only)

        mlp_critic = build_planning_network_config("transformer_large_mlp_critic")
        self.assertEqual(mlp_critic.critic_type, "mlp")


if __name__ == "__main__":
    unittest.main()
