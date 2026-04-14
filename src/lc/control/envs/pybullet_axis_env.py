from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lc.control.configs import PyBulletControlExperimentConfig, get_axis_ladrc_action_bounds
from lc.control.controllers import create_controller_bundle
from lc.control.reference_generators import build_xyz_reference_trajectory
from lc.control.simulators import create_ctrl_aviary, step_controller_loop
from lc.control.simulators.pybullet_runner import _decode_action_to_ladrc_params


@dataclass
class PyBulletAxisLADRLEnv:
    config: PyBulletControlExperimentConfig
    controller_variant: str = "ladrc_pos_pid_att"

    def __post_init__(self) -> None:
        self.axis = "x"
        self.controller = create_controller_bundle(self.controller_variant)
        self.backend = create_ctrl_aviary(self.config)
        self.reference_bundle = build_xyz_reference_trajectory(self.config.axis_config(self.axis), self.config)
        self.state = np.zeros(20, dtype=np.float32)
        self.step_index = 0

    def reset(self, axis: str | None = None, seed: int | None = None) -> np.ndarray:
        self.axis = axis or self.axis
        rng = np.random.default_rng(seed if seed is not None else self.config.seed)
        self.reference_bundle = build_xyz_reference_trajectory(self.config.axis_config(self.axis), self.config, rng=rng)
        self.state = np.zeros(20, dtype=np.float32)
        self.state[0:3] = self.reference_bundle.positions[0]
        self.state[3:7] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.state[16:20] = 4300.0
        self.step_index = 0
        self.controller.reset()
        return self._build_observation(self.state, self.reference_bundle.positions[0], self.reference_bundle.velocities[0])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, float]]:
        axis_index = {"x": 0, "y": 1, "z": 2}[self.axis]
        target_pos = self.reference_bundle.positions[self.step_index]
        target_vel = self.reference_bundle.velocities[self.step_index]
        bounds = get_axis_ladrc_action_bounds(self.axis)
        r, b0, omega_c, k = _decode_action_to_ladrc_params(action, bounds)
        self.controller.parameter_set.axis_config(self.axis).r = float(r)
        self.controller.parameter_set.axis_config(self.axis).b0 = float(b0)
        self.controller.parameter_set.axis_config(self.axis).omega_c = float(omega_c)
        self.controller.parameter_set.axis_config(self.axis).k = float(k)
        if hasattr(self.controller, "_sync_from_parameter_set"):
            self.controller._sync_from_parameter_set()
        next_state, rpm, pos_error, _ = step_controller_loop(
            self.state,
            self.controller,
            target_pos,
            target_vel,
            self.config.control_dt,
        )
        rpm_delta = float(np.mean(np.abs(rpm - self.state[16:20])))
        reward = self._compute_reward(
            pos_error[axis_index],
            target_vel[axis_index] - next_state[10 + axis_index],
            rpm_delta,
            target_vel=float(target_vel[axis_index]),
        )
        self.state = next_state
        self.step_index += 1
        done = self.step_index >= self.config.step_count or self._compute_terminated()
        next_target_pos = self.reference_bundle.positions[min(self.step_index, self.config.step_count - 1)]
        next_target_vel = self.reference_bundle.velocities[min(self.step_index, self.config.step_count - 1)]
        obs = self._build_observation(self.state, next_target_pos, next_target_vel)
        return obs, reward, done, {"axis": float(axis_index), "time_index": float(self.step_index)}

    def _build_observation(self, state: np.ndarray, target_pos: np.ndarray, target_vel: np.ndarray) -> np.ndarray:
        axis_index = {"x": 0, "y": 1, "z": 2}[self.axis]
        residual = target_pos[axis_index] - state[axis_index] - 0.2 * state[10 + axis_index]
        return np.asarray(
            [
                target_pos[axis_index] - state[axis_index],
                target_vel[axis_index] - state[10 + axis_index],
                state[axis_index],
                state[10 + axis_index],
                target_pos[axis_index],
                target_vel[axis_index],
                residual,
                self.step_index / max(self.config.step_count - 1, 1),
            ],
            dtype=np.float32,
        )

    def _compute_reward(self, pos_error: float, vel_error: float, rpm_delta: float, *, target_vel: float) -> float:
        del vel_error, rpm_delta, target_vel
        return float(-abs(pos_error))

    def _compute_terminated(self) -> bool:
        return bool(np.linalg.norm(self.state[0:3]) > 5.0 or self.state[2] < 0.1 or np.max(np.abs(self.state[7:10])) > 1.2)
