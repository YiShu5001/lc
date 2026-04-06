from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlExperimentConfig:
    primary_method: str = "ddpg_ladrc"
    difficulty: str = "medium"
    difficulty_levels: tuple[str, ...] = ("easy", "medium", "hard", "extreme")
    axes: tuple[str, ...] = ("x", "y", "z")
    episode_length: int = 100
    episodes: int = 6
    seed: int = 7
    seed_runs: int = 3
    compare_episodes: int = 6
    train_episodes: int = 12
    warmup_steps: int = 32
    batch_size: int = 16
    updates_per_step: int = 1
    enhanced_stack_size: int = 4
    enhanced_n_step: int = 4
    enhanced_action_hold_steps: int = 4
    export_reference_preview: bool = True
    reference_profile_mode: str = "piecewise_constant_velocity"
