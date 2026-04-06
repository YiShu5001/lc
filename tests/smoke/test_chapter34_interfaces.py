import unittest
from unittest.mock import patch

import torch

from src.lc.control.chapter3_interfaces import Chapter3ExperimentConfig, build_reference_trajectory, run_chapter3_control_experiment
from src.lc.env.evolution_scenarios import build_course_progression, get_scenario_by_stage
from src.lc.planning.chapter4_interfaces import Chapter4ExperimentConfig, build_chapter4_experiment_bundle
from src.lc.planning.models import MultiUAVModel, MultiUAVModelConfig
from src.lc.system.chapter34_bridge import Chapter34BridgeConfig, run_chapter34_pipeline


class Chapter34InterfaceSmokeTests(unittest.TestCase):
    def test_chapter3_control_interface(self):
        config = Chapter3ExperimentConfig(channel_count=6)
        refs = build_reference_trajectory(10, channel_count=6)
        meas = refs * 0.95
        result = run_chapter3_control_experiment(refs, meas, config)
        self.assertEqual(result.outputs.shape, refs.shape)
        self.assertEqual(result.final_errors.shape[0], 6)

    def test_chapter4_bundle_interface(self):
        config = Chapter4ExperimentConfig()
        bundle = build_chapter4_experiment_bundle(config)
        self.assertIsInstance(bundle.actor, MultiUAVModel)
        self.assertIsNotNone(bundle.pyramid_per)
        self.assertIsNotNone(bundle.multi_level_buffer)

    def test_chapter4_model_supports_dict_and_flat_inputs(self):
        cfg = MultiUAVModelConfig(
            self_dim=4,
            obstacle_dim=3,
            neighbor_dim=2,
            embed_dim=16,
            num_heads=4,
            ff_dim=32,
            action_dim=2,
            max_obstacles=2,
            max_neighbors=3,
        )
        model = MultiUAVModel(cfg)
        model.eval()

        self_state = torch.randn(2, cfg.self_dim)
        obstacles = torch.randn(2, cfg.max_obstacles, cfg.obstacle_dim)
        neighbors = torch.randn(2, cfg.max_neighbors, cfg.neighbor_dim)
        dict_obs = {
            "self_state": self_state,
            "obstacles": obstacles,
            "neighbors": neighbors,
        }
        flat_obs = torch.cat(
            [
                self_state,
                obstacles.reshape(2, -1),
                neighbors.reshape(2, -1),
            ],
            dim=1,
        )

        avoid_dict, final_dict = model(dict_obs)
        avoid_flat, final_flat = model(flat_obs)
        self.assertEqual(tuple(avoid_dict.shape), (2, cfg.action_dim))
        self.assertEqual(tuple(final_dict.shape), (2, cfg.action_dim))
        self.assertTrue(torch.allclose(avoid_dict, avoid_flat, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(final_dict, final_flat, atol=1e-5, rtol=1e-5))

    def test_chapter4_stage2_consumes_stage1_feature(self):
        cfg = MultiUAVModelConfig(
            self_dim=4,
            obstacle_dim=3,
            neighbor_dim=2,
            embed_dim=16,
            num_heads=4,
            ff_dim=32,
            action_dim=2,
        )
        model = MultiUAVModel(cfg)
        model.eval()

        obs = {
            "self_state": torch.randn(2, cfg.self_dim),
            "obstacles": torch.randn(2, 2, cfg.obstacle_dim),
            "neighbors": torch.randn(2, 3, cfg.neighbor_dim),
        }
        with torch.no_grad():
            _, _, safe_feature = model.policy_stages(obs)

        with patch.object(model.collaborative_branch, "forward", wraps=model.collaborative_branch.forward) as wrapped:
            model(obs)

        self.assertEqual(wrapped.call_count, 1)
        _, passed_safe_feature = wrapped.call_args.args[:2]
        self.assertTrue(torch.allclose(passed_safe_feature, safe_feature, atol=1e-5, rtol=1e-5))

    def test_course_progression(self):
        stages = build_course_progression()
        self.assertGreaterEqual(len(stages), 5)
        self.assertEqual(get_scenario_by_stage(0).num_drones, 1)
        self.assertGreaterEqual(get_scenario_by_stage(4).num_drones, 5)

    def test_chapter34_pipeline(self):
        result = run_chapter34_pipeline(Chapter34BridgeConfig(scenario_stage=2, rollout_steps=8))
        self.assertEqual(result.control_result.outputs.shape[0], 8)
        self.assertIsInstance(result.planning_actions, torch.Tensor)
        self.assertGreaterEqual(result.scenario.num_obstacles, 0)


if __name__ == "__main__":
    unittest.main()
