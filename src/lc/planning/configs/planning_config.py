from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanningModelConfig:
    self_dim: int = 4
    obstacle_dim: int = 3
    neighbor_dim: int = 2
    hidden_dim: int = 32
    action_dim: int = 2
    max_obstacles: int = 12
    max_neighbors: int = 7


@dataclass(frozen=True)
class PlanningExperimentConfig:
    difficulty: str = "medium"
    stage_index: int = 1
    curriculum_env: str | None = None
    episodes: int = 240
    eval_episodes: int = 5
    seed: int = 7
