from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioConfig:
    difficulty: str
    num_uavs: int
    num_obstacles: int
    obstacle_layout: str
    dynamic_obstacles: bool
    target_motion: str
    world_scale: float
    density: float


@dataclass(frozen=True)
class ControlScenarioConfig(ScenarioConfig):
    disturbance_level: float
    control_frequency_hz: int
    rl_frequency_hz: int


@dataclass(frozen=True)
class PlanningScenarioConfig(ScenarioConfig):
    stage_index: int
    stage_name: str
    curriculum_env: str
    target_is_dynamic: bool
    obstacle_is_dynamic: bool
    target_distance_band: str
    target_speed_scale: float
    obstacle_speed_scale: float
    max_neighbors: int
    max_obstacles: int
