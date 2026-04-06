from __future__ import annotations

import unittest

from lc.envs.scenarios import PLANNING_CURRICULUM_ENVS, build_planning_scenario
from lc.planning.curriculum import CurriculumScheduler
from lc.planning.envs import PlanningSwarmEnv


class TestPlanningCurriculumEnvs(unittest.TestCase):
    def test_guidance_envs_have_no_obstacles(self) -> None:
        for env_name in ("guidance_G1", "guidance_G2"):
            scenario = build_planning_scenario(curriculum_env=env_name)
            self.assertEqual(scenario.num_obstacles, 0)
            self.assertFalse(scenario.obstacle_is_dynamic)
            self.assertEqual(scenario.num_uavs, 1)

    def test_course_env_counts_follow_plan(self) -> None:
        expected = {
            "avoidance_A1": (1, 2, False, False),
            "avoidance_A4": (1, 6, True, True),
            "cooperation_C1": (3, 3, False, False),
            "cooperation_C3": (5, 6, True, True),
        }
        for env_name, (uavs, obstacles, target_dynamic, obstacle_dynamic) in expected.items():
            scenario = build_planning_scenario(curriculum_env=env_name)
            self.assertEqual(scenario.num_uavs, uavs)
            self.assertEqual(scenario.num_obstacles, obstacles)
            self.assertEqual(scenario.target_is_dynamic, target_dynamic)
            self.assertEqual(scenario.obstacle_is_dynamic, obstacle_dynamic)

    def test_scheduler_progresses_across_nine_envs(self) -> None:
        scheduler = CurriculumScheduler(
            window_size=2,
            decision_window=1,
            stable_windows_required=1,
            rollback_windows_required=1,
        )
        self.assertEqual(scheduler.curriculum_env, "guidance_G1")
        for _ in range(len(PLANNING_CURRICULUM_ENVS) - 1):
            scheduler.update({"reward": 1.0, "success_rate": 1.0})
            scheduler.update({"reward": 1.0, "success_rate": 1.0})
        self.assertEqual(scheduler.curriculum_env, "cooperation_C3")
        self.assertEqual(scheduler.current_stage, 2)

    def test_env_info_reports_curriculum_env(self) -> None:
        scenario = build_planning_scenario(curriculum_env="avoidance_A4")
        env = PlanningSwarmEnv(scenario=scenario)
        env.reset()
        _, _, _, info = env.step([0.2, 0.1])
        self.assertEqual(info["curriculum_env"], "avoidance_A4")


if __name__ == "__main__":
    unittest.main()
