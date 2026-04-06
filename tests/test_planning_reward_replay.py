from __future__ import annotations

import unittest

from lc.planning.memory import GuidanceReplayMemory, StagePyramidReplayMemory
from lc.planning.rewards import compute_planning_reward


class TestPlanningRewardReplay(unittest.TestCase):
    def test_guidance_reward_emphasizes_reach(self) -> None:
        reward = compute_planning_reward(
            stage_name="guidance",
            occupancy_error=0.2,
            previous_occupancy_error=0.5,
            formation_error=0.8,
            angle_error=0.7,
            obstacle_margin=0.0,
            neighbor_margin=0.0,
            collision=False,
            success=True,
        )
        self.assertGreater(reward.total_reward, 0.0)
        self.assertGreater(reward.target_reward, reward.collaboration_reward * 0.2)

    def test_avoidance_filtered_layer_keeps_unlearned_samples(self) -> None:
        replay = StagePyramidReplayMemory(stage_name="avoidance", capacity=64, seed=11)
        for _ in range(8):
            replay.push(
                {"reward": 0.1},
                td_error=1.0,
                previous_td_error=1.0,
                contribution=0.5,
                rare_event_score=0.0,
                success=False,
            )
        replay.push(
            {"reward": 0.2},
            td_error=0.05,
            previous_td_error=1.0,
            contribution=0.8,
            rare_event_score=0.0,
            success=False,
        )
        replay.push(
            {"reward": 0.3},
            td_error=0.7,
            previous_td_error=1.0,
            contribution=0.8,
            rare_event_score=0.0,
            success=False,
        )
        stats = replay.stats()
        self.assertEqual(stats["bucket_sizes"]["filtered_layer"], 1)
        self.assertEqual(stats["bucket_sizes"]["td_layer"], 10)

    def test_stage_replay_can_mix_old_pool_samples(self) -> None:
        guidance = GuidanceReplayMemory(capacity=32, old_pool_capacity=8, seed=3)
        for index in range(10):
            guidance.push(
                {
                    "reward": float(index),
                    "success": float(index >= 7),
                    "occupancy_error": 1.0 / (index + 1),
                },
                priority=float(index + 1),
            )
        guidance.refresh_old_pool()
        replay = StagePyramidReplayMemory(stage_name="cooperation", capacity=32, seed=5)
        for _ in range(20):
            replay.push(
                {"reward": 0.3},
                td_error=0.9,
                previous_td_error=1.0,
                contribution=0.7,
                rare_event_score=0.6,
                success=True,
            )
        batch = replay.sample_entries(10, old_pool=guidance.old_pool, old_fraction=0.2)
        old_count = sum(1 for row in batch if row["source"] == "cooperation_old_pool")
        current_count = sum(1 for row in batch if row["source"] == "cooperation_current")
        self.assertEqual(len(batch), 10)
        self.assertEqual(old_count, 2)
        self.assertEqual(current_count, 8)

    def test_priority_modes_and_ratios_are_reported(self) -> None:
        replay = StagePyramidReplayMemory(
            stage_name="cooperation",
            capacity=32,
            sample_ratio=(1, 1, 1),
            secondary_priority_mode="contribution_only",
            rare_priority_mode="rare_only",
            seed=17,
        )
        for _ in range(12):
            replay.push(
                {"reward": 0.5},
                td_error=0.8,
                previous_td_error=1.0,
                contribution=0.4,
                rare_event_score=0.9,
                success=True,
            )
        replay.sample_entries(6)
        stats = replay.stats()
        self.assertEqual(stats["sample_ratio"], [1, 1, 1])
        self.assertEqual(stats["secondary_priority_mode"], "contribution_only")
        self.assertEqual(stats["rare_priority_mode"], "rare_only")
        self.assertIn("priority_means", stats)
        self.assertIn("sampling_fractions", stats)
        self.assertGreaterEqual(stats["priority_means"]["rare_layer"], 0.0)


if __name__ == "__main__":
    unittest.main()
