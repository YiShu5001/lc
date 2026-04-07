from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lc.common.types import ActionSpec, ObservationSpec, TaskInfo
from lc.control.RLcontrolRefLine import (
    AxisRLRefLineTaskConfig,
    RefLineEpisodeBundle,
    adapt_episode_to_tracking_inputs,
    build_default_xy_task_config,
    build_refline_episode,
)
from lc.control.configs.pybullet_control_config import AxisTrainingConfig, PyBulletControlExperimentConfig
from lc.control.reference_generators import ReferenceBundle, build_xyz_reference_trajectory
from lc.envs.base import BaseTaskEnv
from lc.envs.scenarios import ControlScenarioConfig


@dataclass
class ControlTrackingEnv(BaseTaskEnv):
    """Chapter-3 lightweight environment with axis-wise recursive position references."""

    scenario: ControlScenarioConfig
    axis: str = "x"
    seed: int = 7
    episode_length: int = 100
    reference_profile_mode: str = "piecewise_constant_velocity"
    include_velocity_reward: bool = True
    state: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    t: int = 0
    disturbance: float = 0.0
    disturbance_proxy: float = 0.0
    reference_bundle: ReferenceBundle | None = None
    reference_schedule: str = "piecewise_constant_velocity"
    external_episode_bundle: RefLineEpisodeBundle | None = None
    external_task_config: AxisRLRefLineTaskConfig | None = None
    external_reference_position: tuple[float, ...] = field(default_factory=tuple)
    external_reference_velocity: tuple[float, ...] = field(default_factory=tuple)
    external_disturbance: tuple[float, ...] = field(default_factory=tuple)
    external_phase_table: tuple[dict[str, float | str], ...] = field(default_factory=tuple)
    errors: list[float] = field(default_factory=list)
    velocity_errors: list[float] = field(default_factory=list)
    controls: list[float] = field(default_factory=list)
    references: list[float] = field(default_factory=list)
    reference_velocities: list[float] = field(default_factory=list)
    disturbances: list[float] = field(default_factory=list)
    outputs: list[float] = field(default_factory=list)

    @property
    def obs_spec(self) -> ObservationSpec:
        return ObservationSpec(
            shape=(8,),
            description=(
                "pos_error_axis, vel_error_axis, current_pos_axis, current_vel_axis, "
                "reference_pos_axis, reference_vel_axis, disturbance_proxy, normalized_time"
            ),
        )

    @property
    def action_spec(self) -> ActionSpec:
        return ActionSpec(shape=(3,), low=-1.0, high=1.0, description="normalized_b0, normalized_wc, normalized_k")

    @property
    def task_info(self) -> TaskInfo:
        return TaskInfo(name=f"control_tracking_{self.axis}", stage="chapter3", difficulty=self.scenario.difficulty)

    @property
    def axis_index(self) -> int:
        return {"x": 0, "y": 1, "z": 2}[self.axis]

    @property
    def reference(self) -> float:
        return float(self._target_position(self.t))

    @property
    def reference_velocity(self) -> float:
        return float(self._target_velocity(self.t))

    @property
    def state_axis(self) -> float:
        return float(self.state[self.axis_index])

    def reset(
        self,
        axis: str | None = None,
        seed: int | None = None,
        external_episode_bundle: RefLineEpisodeBundle | None = None,
    ) -> np.ndarray:
        """Reset environment and generate one recursive piecewise-velocity reference episode."""
        if axis is not None:
            self.axis = axis
        if seed is not None:
            self.seed = seed
        if external_episode_bundle is not None:
            self.external_episode_bundle = external_episode_bundle
        self.state = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.state[2] = 1.0 if self.axis != "z" else 0.0
        self.t = 0
        self.disturbance = 0.0
        self.disturbance_proxy = 0.0
        self.errors.clear()
        self.velocity_errors.clear()
        self.controls.clear()
        self.references.clear()
        self.reference_velocities.clear()
        self.disturbances.clear()
        self.outputs.clear()
        self.reference_schedule = self.reference_profile_mode
        self.external_reference_position = tuple()
        self.external_reference_velocity = tuple()
        self.external_disturbance = tuple()
        self.external_phase_table = tuple()
        self.reference_bundle = self._build_reference_bundle()
        return self._obs()

    def step(self, control_signal: float) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Advance one control step for the active axis."""
        dt = 1.0 / self.scenario.control_frequency_hz
        axis_index = self.axis_index
        reference = self._target_position(self.t)
        reference_velocity = self._target_velocity(self.t)
        self.disturbance = self._disturbance(self.t)
        damping = 0.18 if axis_index < 2 else 0.24
        accel = control_signal - damping * float(self.velocity[axis_index]) + self.disturbance
        self.velocity[axis_index] += accel * dt
        self.state[axis_index] += float(self.velocity[axis_index]) * dt
        if axis_index != 2:
            self.state[2] = 1.0
        self.t += 1
        error = reference - float(self.state[axis_index])
        velocity_error = reference_velocity - float(self.velocity[axis_index])
        reward = -abs(error)
        if self.include_velocity_reward:
            reward -= 0.15 * abs(velocity_error)
        self.disturbance_proxy += dt * 0.5 * (self.disturbance - self.disturbance_proxy)
        self.errors.append(float(error))
        self.velocity_errors.append(float(velocity_error))
        self.controls.append(float(control_signal))
        self.references.append(float(reference))
        self.reference_velocities.append(float(reference_velocity))
        self.disturbances.append(float(self.disturbance))
        self.outputs.append(float(self.state[axis_index]))
        done = self.t >= self.episode_length
        info = {
            "axis": self.axis,
            "error": float(error),
            "velocity_error": float(velocity_error),
            "disturbance": float(self.disturbance),
            "reference_velocity": float(reference_velocity),
        }
        return self._obs(), reward, done, info

    def reference_summary(self) -> list[dict[str, float]]:
        if self.reference_bundle is None:
            return list(self.external_phase_table)
        summary: list[dict[str, float]] = []
        for index, (segment, speed) in enumerate(
            zip(self.reference_bundle.stage_slices, self.reference_bundle.stage_velocities)
        ):
            summary.append({"stage": float(index), "start": float(segment.start), "stop": float(segment.stop), "velocity": float(speed)})
        return summary

    def _obs(self) -> np.ndarray:
        axis_index = self.axis_index
        target_pos = self._target_position(self.t)
        target_vel = self._target_velocity(self.t)
        error = target_pos - float(self.state[axis_index])
        velocity_error = target_vel - float(self.velocity[axis_index])
        return np.asarray(
            [
                error,
                velocity_error,
                float(self.state[axis_index]),
                float(self.velocity[axis_index]),
                target_pos,
                target_vel,
                float(self.disturbance_proxy),
                self.t / max(self.episode_length, 1),
            ],
            dtype=np.float32,
        )

    def _build_reference_bundle(self) -> ReferenceBundle:
        if self.reference_profile_mode == "rl_refline_six_phase" or self.external_episode_bundle is not None:
            return self._build_external_episode_bundle()
        pybullet_cfg = PyBulletControlExperimentConfig(
            control_freq_hz=self.scenario.control_frequency_hz,
            rl_freq_hz=self.scenario.rl_frequency_hz,
            duration_sec=self.episode_length / max(self.scenario.control_frequency_hz, 1),
            seed=self.seed,
        )
        axis_cfg = self._axis_training_config()
        rng = np.random.default_rng(self.seed)
        return build_xyz_reference_trajectory(axis_cfg, pybullet_cfg, rng=rng)

    def _build_external_episode_bundle(self) -> ReferenceBundle | None:
        bundle = self.external_episode_bundle
        if bundle is None:
            task_config = self.external_task_config or build_default_xy_task_config(self.axis)
            task_config = AxisRLRefLineTaskConfig(
                axis=self.axis,
                total_duration_sec=self.episode_length / max(self.scenario.control_frequency_hz, 1),
                control_frequency_hz=self.scenario.control_frequency_hz,
                rl_frequency_hz=self.scenario.rl_frequency_hz,
                enable_randomization=task_config.enable_randomization,
                disturbance_decay_mode=task_config.disturbance_decay_mode,
                min_phase_duration_sec=task_config.min_phase_duration_sec,
                phase_specs=task_config.phase_specs,
            )
            bundle = build_refline_episode(task_config, seed=self.seed)
            self.external_episode_bundle = bundle
        adapted = adapt_episode_to_tracking_inputs(bundle)
        self.reference_schedule = "rl_refline_six_phase"
        self.external_reference_position = tuple(float(value) for value in adapted["reference_position"])
        self.external_reference_velocity = tuple(float(value) for value in adapted["reference_velocity"])
        self.external_disturbance = tuple(float(value) for value in adapted["disturbance"])
        self.external_phase_table = tuple(bundle.phase_table)
        return None

    def _axis_training_config(self) -> AxisTrainingConfig:
        fixed_height = 1.0 if self.axis != "z" else 0.0
        primary, reverse = self._speed_ranges()
        disturbance_scale = max(self.scenario.disturbance_level, 0.05)
        return AxisTrainingConfig(
            axis=self.axis,
            initial_position=(0.0, 0.0, fixed_height),
            fixed_axes=(0.0, fixed_height),
            primary_speed_range=primary,
            reverse_speed_range=reverse,
            disturbance_scale=disturbance_scale,
        )

    def _speed_ranges(self) -> tuple[tuple[float, float], tuple[float, float]]:
        scale = {
            "easy": 0.5,
            "medium": 0.8,
            "hard": 1.0,
            "extreme": 1.2,
        }.get(self.scenario.difficulty, 0.8)
        axis_scale = 0.7 if self.axis == "z" else 1.0
        upper = 0.7 * scale * axis_scale
        lower = 0.3 * scale * axis_scale
        reverse_upper = -0.2 * scale * axis_scale
        reverse_lower = -0.5 * scale * axis_scale
        return (lower, upper), (reverse_lower, reverse_upper)

    def _target_position(self, step: int) -> float:
        if self.external_reference_position:
            index = min(step, len(self.external_reference_position) - 1)
            return float(self.external_reference_position[index])
        if self.reference_bundle is None:
            return 0.0
        index = min(step, len(self.reference_bundle.positions) - 1)
        return float(self.reference_bundle.positions[index, self.axis_index])

    def _target_velocity(self, step: int) -> float:
        if self.external_reference_velocity:
            index = min(step, len(self.external_reference_velocity) - 1)
            return float(self.external_reference_velocity[index])
        if self.reference_bundle is None:
            return 0.0
        index = min(step, len(self.reference_bundle.velocities) - 1)
        return float(self.reference_bundle.velocities[index, self.axis_index])

    def _disturbance(self, step: int) -> float:
        if self.external_disturbance:
            index = min(step, len(self.external_disturbance) - 1)
            return float(self.external_disturbance[index])
        amplitude = self.scenario.disturbance_level * (1.15 if self.axis == "z" else 1.0)
        window = 0.0
        if step > self.episode_length // 3:
            window += 0.6
        if step > (2 * self.episode_length) // 3:
            window -= 0.35
        periodic = np.sin(0.12 * step) + 0.35 * np.sign(np.sin(0.04 * step))
        return float(amplitude * window * periodic)
