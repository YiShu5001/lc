from __future__ import annotations

from lc.envs.scenarios.configs import ControlScenarioConfig, PlanningScenarioConfig


def _base_values(difficulty: str) -> dict[str, object]:
    table = {
        "easy": dict(num_uavs=1, num_obstacles=2, obstacle_layout="sparse", dynamic_obstacles=False, target_motion="static", world_scale=1.0, density=0.2),
        "medium": dict(num_uavs=3, num_obstacles=5, obstacle_layout="corridor", dynamic_obstacles=False, target_motion="linear", world_scale=1.5, density=0.35),
        "hard": dict(num_uavs=5, num_obstacles=8, obstacle_layout="dense", dynamic_obstacles=True, target_motion="curve", world_scale=2.0, density=0.55),
        "extreme": dict(num_uavs=8, num_obstacles=12, obstacle_layout="mixed", dynamic_obstacles=True, target_motion="evasive", world_scale=2.5, density=0.75),
    }
    if difficulty not in table:
        raise ValueError(f"Unsupported difficulty: {difficulty}")
    return table[difficulty]


PLANNING_CURRICULUM_ENVS: dict[str, dict[str, object]] = {
    "guidance_G1": dict(
        difficulty="easy",
        stage_index=0,
        stage_name="guidance",
        num_uavs=1,
        num_obstacles=0,
        obstacle_layout="none",
        dynamic_obstacles=False,
        target_motion="static",
        target_is_dynamic=False,
        obstacle_is_dynamic=False,
        target_distance_band="random",
        target_speed_scale=0.0,
        obstacle_speed_scale=0.0,
        world_scale=1.0,
        density=0.05,
    ),
    "guidance_G2": dict(
        difficulty="medium",
        stage_index=0,
        stage_name="guidance",
        num_uavs=1,
        num_obstacles=0,
        obstacle_layout="none",
        dynamic_obstacles=False,
        target_motion="linear",
        target_is_dynamic=True,
        obstacle_is_dynamic=False,
        target_distance_band="random",
        target_speed_scale=0.35,
        obstacle_speed_scale=0.0,
        world_scale=1.3,
        density=0.1,
    ),
    "avoidance_A1": dict(
        difficulty="easy",
        stage_index=1,
        stage_name="avoidance",
        num_uavs=1,
        num_obstacles=2,
        obstacle_layout="sparse",
        dynamic_obstacles=False,
        target_motion="static",
        target_is_dynamic=False,
        obstacle_is_dynamic=False,
        target_distance_band="random",
        target_speed_scale=0.0,
        obstacle_speed_scale=0.0,
        world_scale=1.2,
        density=0.25,
    ),
    "avoidance_A2": dict(
        difficulty="medium",
        stage_index=1,
        stage_name="avoidance",
        num_uavs=1,
        num_obstacles=5,
        obstacle_layout="corridor",
        dynamic_obstacles=False,
        target_motion="static",
        target_is_dynamic=False,
        obstacle_is_dynamic=False,
        target_distance_band="random",
        target_speed_scale=0.0,
        obstacle_speed_scale=0.0,
        world_scale=1.5,
        density=0.4,
    ),
    "avoidance_A3": dict(
        difficulty="hard",
        stage_index=1,
        stage_name="avoidance",
        num_uavs=1,
        num_obstacles=5,
        obstacle_layout="corridor",
        dynamic_obstacles=False,
        target_motion="curve",
        target_is_dynamic=True,
        obstacle_is_dynamic=False,
        target_distance_band="random",
        target_speed_scale=0.45,
        obstacle_speed_scale=0.0,
        world_scale=1.8,
        density=0.48,
    ),
    "avoidance_A4": dict(
        difficulty="extreme",
        stage_index=1,
        stage_name="avoidance",
        num_uavs=1,
        num_obstacles=6,
        obstacle_layout="dense_corridor",
        dynamic_obstacles=True,
        target_motion="curve",
        target_is_dynamic=True,
        obstacle_is_dynamic=True,
        target_distance_band="random",
        target_speed_scale=0.5,
        obstacle_speed_scale=0.25,
        world_scale=2.0,
        density=0.58,
    ),
    "cooperation_C1": dict(
        difficulty="medium",
        stage_index=2,
        stage_name="cooperation",
        num_uavs=3,
        num_obstacles=3,
        obstacle_layout="sparse_ring",
        dynamic_obstacles=False,
        target_motion="static",
        target_is_dynamic=False,
        obstacle_is_dynamic=False,
        target_distance_band="medium",
        target_speed_scale=0.0,
        obstacle_speed_scale=0.0,
        world_scale=1.7,
        density=0.35,
    ),
    "cooperation_C2": dict(
        difficulty="hard",
        stage_index=2,
        stage_name="cooperation",
        num_uavs=4,
        num_obstacles=5,
        obstacle_layout="clustered_corridor",
        dynamic_obstacles=False,
        target_motion="curve",
        target_is_dynamic=True,
        obstacle_is_dynamic=False,
        target_distance_band="medium",
        target_speed_scale=0.45,
        obstacle_speed_scale=0.0,
        world_scale=2.0,
        density=0.52,
    ),
    "cooperation_C3": dict(
        difficulty="extreme",
        stage_index=2,
        stage_name="cooperation",
        num_uavs=5,
        num_obstacles=6,
        obstacle_layout="mixed",
        dynamic_obstacles=True,
        target_motion="maneuver",
        target_is_dynamic=True,
        obstacle_is_dynamic=True,
        target_distance_band="far",
        target_speed_scale=0.6,
        obstacle_speed_scale=0.25,
        world_scale=2.4,
        density=0.68,
    ),
}


_LEGACY_PLANNING_BY_STAGE = {
    0: {"easy": "guidance_G1", "medium": "guidance_G2", "hard": "guidance_G2", "extreme": "guidance_G2"},
    1: {"easy": "avoidance_A1", "medium": "avoidance_A2", "hard": "avoidance_A3", "extreme": "avoidance_A4"},
    2: {"easy": "cooperation_C1", "medium": "cooperation_C2", "hard": "cooperation_C3", "extreme": "cooperation_C3"},
}


def build_control_scenario(difficulty: str) -> ControlScenarioConfig:
    base = _base_values(difficulty)
    disturbance = {"easy": 0.05, "medium": 0.1, "hard": 0.2, "extreme": 0.35}[difficulty]
    return ControlScenarioConfig(
        difficulty=difficulty,
        disturbance_level=disturbance,
        control_frequency_hz=100,
        rl_frequency_hz=10,
        **base,
    )


def build_planning_scenario(
    difficulty: str | None = None,
    stage_index: int = 0,
    curriculum_env: str | None = None,
) -> PlanningScenarioConfig:
    env_name = curriculum_env
    if env_name is None:
        difficulty_name = difficulty or "medium"
        env_name = _LEGACY_PLANNING_BY_STAGE[min(max(stage_index, 0), 2)][difficulty_name]
    if env_name not in PLANNING_CURRICULUM_ENVS:
        raise ValueError(f"Unsupported planning curriculum env: {env_name}")
    spec = PLANNING_CURRICULUM_ENVS[env_name]
    return PlanningScenarioConfig(
        curriculum_env=env_name,
        max_neighbors=max(int(spec["num_uavs"]) - 1, 1),
        max_obstacles=max(int(spec["num_obstacles"]), 1),
        **spec,
    )
