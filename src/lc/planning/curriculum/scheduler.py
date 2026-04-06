from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np


CURRICULUM_SEQUENCE: tuple[str, ...] = (
    "guidance_G1",
    "guidance_G2",
    "avoidance_A1",
    "avoidance_A2",
    "avoidance_A3",
    "avoidance_A4",
    "cooperation_C1",
    "cooperation_C2",
    "cooperation_C3",
)


def curriculum_stage_name(curriculum_env: str) -> str:
    return curriculum_env.split("_", 1)[0]


def curriculum_stage_index(curriculum_env: str) -> int:
    return {"guidance": 0, "avoidance": 1, "cooperation": 2}[curriculum_stage_name(curriculum_env)]


@dataclass
class CurriculumScheduler:
    sequence: tuple[str, ...] = CURRICULUM_SEQUENCE
    current_step: int = 0
    window_size: int = 6
    decision_window: int = 10
    stable_windows_required: int = 5
    rollback_windows_required: int = 5
    reward_thresholds: tuple[float, ...] = (0.2, 0.22, 0.16, 0.14, 0.12, 0.1, 0.1, 0.08, 0.07)
    success_thresholds: tuple[float, ...] = (0.0, 0.0, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16)
    std_thresholds: tuple[float, ...] = (0.42, 0.45, 0.48, 0.52, 0.56, 0.6, 0.64, 0.68, 0.72)
    rollback_reward_drop: float = 0.14
    rollback_success_drop: float = 0.2
    reward_history: deque[float] = field(default_factory=lambda: deque(maxlen=6))
    success_history: deque[float] = field(default_factory=lambda: deque(maxlen=6))
    stable_window_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    bad_window_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    stage_history: list[dict[str, float | int | str]] = field(default_factory=list)
    stage_metric_sums: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    stage_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    env_metric_sums: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    env_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _stable_windows: int = 0
    _rollback_windows: int = 0
    _last_window_reward_mean: float | None = None
    _last_window_success_mean: float | None = None

    def __post_init__(self) -> None:
        self.reward_history = deque(self.reward_history, maxlen=self.window_size)
        self.success_history = deque(self.success_history, maxlen=self.window_size)
        self.stable_window_history = deque(self.stable_window_history, maxlen=self.decision_window)
        self.bad_window_history = deque(self.bad_window_history, maxlen=self.decision_window)

    @property
    def curriculum_env(self) -> str:
        return self.sequence[self.current_step]

    @property
    def stage_name(self) -> str:
        return curriculum_stage_name(self.curriculum_env)

    @property
    def current_stage(self) -> int:
        return curriculum_stage_index(self.curriculum_env)

    def update(self, metrics: dict[str, float]) -> int:
        stage_name = self.stage_name
        env_name = self.curriculum_env
        self.stage_counts[stage_name] += 1
        self.env_counts[env_name] += 1
        stage_sums = self.stage_metric_sums.setdefault(stage_name, {})
        env_sums = self.env_metric_sums.setdefault(env_name, {})
        for key, value in metrics.items():
            stage_sums[key] = stage_sums.get(key, 0.0) + float(value)
            env_sums[key] = env_sums.get(key, 0.0) + float(value)

        reward = float(metrics.get("reward", 0.0))
        success_rate = float(metrics.get("success_rate", 0.0))
        self.reward_history.append(reward)
        self.success_history.append(success_rate)

        previous_step = self.current_step
        reward_mean = float(np.mean(self.reward_history)) if self.reward_history else reward
        reward_std = float(np.std(self.reward_history)) if len(self.reward_history) > 1 else 0.0
        success_mean = float(np.mean(self.success_history)) if self.success_history else success_rate
        promoted = False
        rolled_back = False
        if len(self.reward_history) >= self.window_size:
            threshold_index = min(self.current_step, len(self.reward_thresholds) - 1)
            stable = (
                reward_mean >= self.reward_thresholds[threshold_index]
                and success_mean >= self.success_thresholds[threshold_index]
                and reward_std <= self.std_thresholds[threshold_index]
            )
            bad_window = False
            if stable:
                self.stable_window_history.append(1.0)
                self.bad_window_history.append(0.0)
            else:
                previous_window_mean = self._last_window_reward_mean if self._last_window_reward_mean is not None else reward_mean
                previous_success_mean = self._last_window_success_mean if self._last_window_success_mean is not None else success_mean
                reward_drop = reward_mean < previous_window_mean - self.rollback_reward_drop
                success_drop = success_mean < max(0.0, previous_success_mean - self.rollback_success_drop)
                bad_window = reward_drop and success_drop
                self.stable_window_history.append(0.0)
                self.bad_window_history.append(1.0 if bad_window else 0.0)
            self._last_window_reward_mean = reward_mean
            self._last_window_success_mean = success_mean
            stable_count = int(sum(self.stable_window_history))
            bad_count = int(sum(self.bad_window_history))
            if stable_count >= self.stable_windows_required and self.current_step < len(self.sequence) - 1:
                self.current_step += 1
                promoted = True
                self._reset_window_state()
            elif bad_count >= self.rollback_windows_required and self.current_step > 0:
                self.current_step -= 1
                rolled_back = True
                self._reset_window_state()

        self.stage_history.append(
            {
                "previous_env": self.sequence[previous_step],
                "previous_stage_name": curriculum_stage_name(self.sequence[previous_step]),
                "previous_stage_index": curriculum_stage_index(self.sequence[previous_step]),
                "previous_stage": curriculum_stage_index(self.sequence[previous_step]),
                "new_env": self.curriculum_env,
                "new_stage_name": self.stage_name,
                "new_stage": self.current_stage,
                "new_stage_index": self.current_stage,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "success_mean": success_mean,
                "promoted": float(promoted),
                "rolled_back": float(rolled_back),
            }
        )
        return self.current_stage

    def get_stage_averages(self) -> dict[str, dict[str, float]]:
        averages: dict[str, dict[str, float]] = {}
        for stage_name, sums in self.stage_metric_sums.items():
            count = max(1, self.stage_counts.get(stage_name, 0))
            averages[stage_name] = {key: float(value / count) for key, value in sums.items()}
        return averages

    def get_env_averages(self) -> dict[str, dict[str, float]]:
        averages: dict[str, dict[str, float]] = {}
        for env_name, sums in self.env_metric_sums.items():
            count = max(1, self.env_counts.get(env_name, 0))
            averages[env_name] = {key: float(value / count) for key, value in sums.items()}
        return averages

    def _reset_window_state(self) -> None:
        self.reward_history.clear()
        self.success_history.clear()
        self.stable_window_history.clear()
        self.bad_window_history.clear()
        self._last_window_reward_mean = None
        self._last_window_success_mean = None
