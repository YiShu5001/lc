from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import importlib.util
import sys

import numpy as np

from lc.control.configs import PyBulletControlExperimentConfig
from lc.control.controllers import ControllerBundle
from lc.control.reference_generators.piecewise_velocity import ReferenceBundle


@dataclass
class SimulationArtifacts:
    timeseries: list[dict[str, float]]
    logger_rows: list[dict[str, float]]
    rewards: list[float]
    final_state: np.ndarray
    backend: str


def create_ctrl_aviary(config: PyBulletControlExperimentConfig) -> dict[str, Any]:
    return {
        "backend": "gym_env" if _can_use_gym_env_backend() else "fallback",
        "config": config,
        "env": None,
        "axis": None,
    }


def close_ctrl_aviary(env: dict[str, Any]) -> None:
    if env.get("backend") == "gym_env" and env.get("env") is not None:
        env["env"].close()
        env["env"] = None


def step_controller_loop(
    state: np.ndarray,
    controller_bundle: ControllerBundle,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    control_dt: float,
    disturbance: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rpm, pos_error, yaw_error = controller_bundle.compute_control_from_state(
        control_timestep=control_dt,
        state=state,
        target_pos=target_pos,
        target_vel=target_vel,
        target_rpy=np.zeros(3, dtype=np.float32),
        target_rpy_rates=np.zeros(3, dtype=np.float32),
    )
    disturbance = disturbance if disturbance is not None else np.zeros(3, dtype=np.float32)
    thrust_proxy = (float(np.mean(rpm)) - 4300.0) / 420.0
    accel_world = np.asarray(
        [
            1.6 * (target_pos[0] - state[0]) + 0.55 * (target_vel[0] - state[10]),
            1.6 * (target_pos[1] - state[1]) + 0.55 * (target_vel[1] - state[11]),
            thrust_proxy + 1.8 * (target_pos[2] - state[2]) + 0.65 * (target_vel[2] - state[12]),
        ],
        dtype=np.float32,
    )
    accel_world += disturbance
    next_state = state.copy()
    next_state[10:13] = 0.94 * state[10:13] + accel_world * control_dt
    next_state[0:3] = state[0:3] + next_state[10:13] * control_dt
    desired_rpy = np.asarray(
        [
            float(np.clip(-0.16 * accel_world[1], -0.4, 0.4)),
            float(np.clip(0.16 * accel_world[0], -0.4, 0.4)),
            0.0,
        ],
        dtype=np.float32,
    )
    rpy_error = desired_rpy - state[7:10]
    next_state[13:16] = 0.88 * state[13:16] + 2.4 * rpy_error * control_dt
    next_state[7:10] = state[7:10] + next_state[13:16] * control_dt
    quaternion = _euler_to_quaternion(next_state[7:10])
    next_state[3:7] = quaternion
    next_state[16:20] = rpm
    return next_state, rpm, pos_error, yaw_error


def run_training_episode(
    env: dict[str, Any],
    policy: Any,
    controller_bundle: ControllerBundle,
    reference_bundle: ReferenceBundle,
    axis: str,
    action_hold_steps: int,
    config: PyBulletControlExperimentConfig,
) -> SimulationArtifacts:
    if env["backend"] == "gym_env":
        return _run_real_training_episode(env, policy, controller_bundle, reference_bundle, axis, action_hold_steps, config)
    return _run_fallback_training_episode(env, policy, controller_bundle, reference_bundle, axis, action_hold_steps, config)


def run_evaluation_episode(
    env: dict[str, Any],
    controller_bundle: ControllerBundle,
    reference_bundle: ReferenceBundle,
    axis: str,
    config: PyBulletControlExperimentConfig,
) -> SimulationArtifacts:
    if env["backend"] == "gym_env":
        return _run_real_evaluation_episode(env, controller_bundle, reference_bundle, axis, config)
    return _run_fallback_evaluation_episode(env, controller_bundle, reference_bundle, axis, config)


def _run_real_training_episode(
    env: dict[str, Any],
    policy: Any,
    controller_bundle: ControllerBundle,
    reference_bundle: ReferenceBundle,
    axis: str,
    action_hold_steps: int,
    config: PyBulletControlExperimentConfig,
) -> SimulationArtifacts:
    real_env = _ensure_real_env(env, config, reference_bundle)
    obs, _ = real_env.reset(seed=config.seed)
    state = np.asarray(obs[0], dtype=np.float32)
    controller_bundle.reset()
    if hasattr(policy, "reset"):
        policy.reset()
    timeseries: list[dict[str, float]] = []
    logger_rows: list[dict[str, float]] = []
    rewards: list[float] = []
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    hold_counter = 0
    action = np.zeros(3, dtype=np.float32)
    prev_rpm = np.full(4, 4300.0, dtype=np.float32)
    for step in range(config.step_count):
        target_pos = reference_bundle.positions[step]
        target_vel = reference_bundle.velocities[step]
        observation = _build_axis_observation(state, target_pos, target_vel, axis, step, config.step_count)
        if hold_counter <= 0:
            action = policy.select_action(observation, explore=True)
            hold_counter = max(action_hold_steps - 1, 0)
        else:
            hold_counter -= 1
        checkpoint = controller_bundle.snapshot_params()
        _apply_axis_action(controller_bundle, axis, action)
        rpm, _, _ = controller_bundle.compute_control_from_state(
            control_timestep=config.control_dt,
            state=state,
            target_pos=target_pos,
            target_vel=target_vel,
            target_rpy=np.zeros(3, dtype=np.float32),
            target_rpy_rates=np.zeros(3, dtype=np.float32),
        )
        next_obs, _, terminated, truncated, _ = real_env.step(rpm.reshape(1, 4))
        next_state = np.asarray(next_obs[0], dtype=np.float32)
        pos_error = float(target_pos[axis_index] - next_state[axis_index])
        vel_error = float(target_vel[axis_index] - next_state[10 + axis_index])
        rpm_delta = float(np.mean(np.abs(rpm - prev_rpm)))
        reward = _compute_axis_reward(pos_error, vel_error, rpm_delta)
        done = bool(terminated or truncated or step == config.step_count - 1)
        next_observation = _build_axis_observation(next_state, target_pos, target_vel, axis, step + 1, config.step_count)
        policy.store_transition(observation, action, reward, next_observation, done)
        rewards.append(float(reward))
        timeseries.append(
            _timeseries_row(step, config.control_dt, axis, state, target_pos, target_vel, rpm, reward, checkpoint, env["backend"])
        )
        logger_rows.append(_logger_row(step, config.control_dt, state, target_pos, target_vel))
        state = next_state
        prev_rpm = rpm
        if done:
            break
    return SimulationArtifacts(timeseries=timeseries, logger_rows=logger_rows, rewards=rewards, final_state=state, backend=env["backend"])


def _run_real_evaluation_episode(
    env: dict[str, Any],
    controller_bundle: ControllerBundle,
    reference_bundle: ReferenceBundle,
    axis: str,
    config: PyBulletControlExperimentConfig,
) -> SimulationArtifacts:
    real_env = _ensure_real_env(env, config, reference_bundle)
    obs, _ = real_env.reset(seed=config.seed)
    state = np.asarray(obs[0], dtype=np.float32)
    controller_bundle.reset()
    timeseries: list[dict[str, float]] = []
    logger_rows: list[dict[str, float]] = []
    rewards: list[float] = []
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    prev_rpm = np.full(4, 4300.0, dtype=np.float32)
    for step in range(config.step_count):
        target_pos = reference_bundle.positions[step]
        target_vel = reference_bundle.velocities[step]
        checkpoint = controller_bundle.snapshot_params()
        rpm, _, _ = controller_bundle.compute_control_from_state(
            control_timestep=config.control_dt,
            state=state,
            target_pos=target_pos,
            target_vel=target_vel,
            target_rpy=np.zeros(3, dtype=np.float32),
            target_rpy_rates=np.zeros(3, dtype=np.float32),
        )
        next_obs, _, terminated, truncated, _ = real_env.step(rpm.reshape(1, 4))
        next_state = np.asarray(next_obs[0], dtype=np.float32)
        pos_error = float(target_pos[axis_index] - next_state[axis_index])
        vel_error = float(target_vel[axis_index] - next_state[10 + axis_index])
        rpm_delta = float(np.mean(np.abs(rpm - prev_rpm)))
        reward = _compute_axis_reward(pos_error, vel_error, rpm_delta)
        rewards.append(float(reward))
        timeseries.append(
            _timeseries_row(step, config.control_dt, axis, state, target_pos, target_vel, rpm, reward, checkpoint, env["backend"])
        )
        logger_rows.append(_logger_row(step, config.control_dt, state, target_pos, target_vel))
        state = next_state
        prev_rpm = rpm
        if terminated or truncated:
            break
    return SimulationArtifacts(timeseries=timeseries, logger_rows=logger_rows, rewards=rewards, final_state=state, backend=env["backend"])


def _run_fallback_training_episode(
    env: dict[str, Any],
    policy: Any,
    controller_bundle: ControllerBundle,
    reference_bundle: ReferenceBundle,
    axis: str,
    action_hold_steps: int,
    config: PyBulletControlExperimentConfig,
) -> SimulationArtifacts:
    del env
    state = _initial_state(reference_bundle)
    controller_bundle.reset()
    if hasattr(policy, "reset"):
        policy.reset()
    timeseries: list[dict[str, float]] = []
    logger_rows: list[dict[str, float]] = []
    rewards: list[float] = []
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    hold_counter = 0
    action = np.zeros(3, dtype=np.float32)
    prev_rpm = np.full(4, 4300.0, dtype=np.float32)
    for step in range(config.step_count):
        target_pos = reference_bundle.positions[step]
        target_vel = reference_bundle.velocities[step]
        observation = _build_axis_observation(state, target_pos, target_vel, axis, step, config.step_count)
        if hold_counter <= 0:
            action = policy.select_action(observation, explore=True)
            hold_counter = max(action_hold_steps - 1, 0)
        else:
            hold_counter -= 1
        checkpoint = controller_bundle.snapshot_params()
        _apply_axis_action(controller_bundle, axis, action)
        disturbance = _disturbance_vector(axis_index, step, config, config.axis_config(axis))
        next_state, rpm, pos_error, _ = step_controller_loop(
            state,
            controller_bundle,
            target_pos=target_pos,
            target_vel=target_vel,
            control_dt=config.control_dt,
            disturbance=disturbance,
        )
        next_observation = _build_axis_observation(next_state, target_pos, target_vel, axis, step + 1, config.step_count)
        rpm_delta = float(np.mean(np.abs(rpm - prev_rpm)))
        reward = _compute_axis_reward(pos_error[axis_index], target_vel[axis_index] - next_state[10 + axis_index], rpm_delta)
        done = step == config.step_count - 1
        policy.store_transition(observation, action, reward, next_observation, done)
        rewards.append(float(reward))
        timeseries.append(
            _timeseries_row(step, config.control_dt, axis, state, target_pos, target_vel, rpm, reward, checkpoint, "fallback")
        )
        logger_rows.append(_logger_row(step, config.control_dt, state, target_pos, target_vel))
        state = next_state
        prev_rpm = rpm
    return SimulationArtifacts(timeseries=timeseries, logger_rows=logger_rows, rewards=rewards, final_state=state, backend="fallback")


def _run_fallback_evaluation_episode(
    env: dict[str, Any],
    controller_bundle: ControllerBundle,
    reference_bundle: ReferenceBundle,
    axis: str,
    config: PyBulletControlExperimentConfig,
) -> SimulationArtifacts:
    del env
    state = _initial_state(reference_bundle)
    controller_bundle.reset()
    timeseries: list[dict[str, float]] = []
    logger_rows: list[dict[str, float]] = []
    rewards: list[float] = []
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    for step in range(config.step_count):
        target_pos = reference_bundle.positions[step]
        target_vel = reference_bundle.velocities[step]
        checkpoint = controller_bundle.snapshot_params()
        disturbance = _disturbance_vector(axis_index, step, config, config.axis_config(axis))
        next_state, rpm, pos_error, _ = step_controller_loop(
            state,
            controller_bundle,
            target_pos=target_pos,
            target_vel=target_vel,
            control_dt=config.control_dt,
            disturbance=disturbance,
        )
        reward = _compute_axis_reward(pos_error[axis_index], target_vel[axis_index] - next_state[10 + axis_index], 0.0)
        rewards.append(float(reward))
        timeseries.append(
            _timeseries_row(step, config.control_dt, axis, state, target_pos, target_vel, rpm, reward, checkpoint, "fallback")
        )
        logger_rows.append(_logger_row(step, config.control_dt, state, target_pos, target_vel))
        state = next_state
    return SimulationArtifacts(timeseries=timeseries, logger_rows=logger_rows, rewards=rewards, final_state=state, backend="fallback")


def _build_axis_observation(
    state: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    axis: str,
    step: int,
    step_count: int,
) -> np.ndarray:
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
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
            min(step / max(step_count - 1, 1), 1.0),
        ],
        dtype=np.float32,
    )


def _compute_axis_reward(pos_error: float, vel_error: float, rpm_delta: float) -> float:
    return float(-abs(pos_error) - 0.15 * abs(vel_error) - 0.0008 * rpm_delta)


def _initial_state(reference_bundle: ReferenceBundle) -> np.ndarray:
    state = np.zeros(20, dtype=np.float32)
    state[0:3] = reference_bundle.positions[0]
    state[2] = max(state[2], 0.4)
    state[3:7] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    state[16:20] = 4300.0
    return state


def _apply_axis_action(controller_bundle: ControllerBundle, axis: str, action: np.ndarray) -> None:
    axis_config = controller_bundle.parameter_set.axis_config(axis)
    axis_config.b0 = float(np.clip(axis_config.b0 + 0.05 * action[0], 0.2, 4.0))
    axis_config.omega_c = float(np.clip(axis_config.omega_c + 0.12 * action[1], 0.5, 12.0))
    axis_config.k = float(np.clip(axis_config.k + 0.08 * action[2], 2.0, 6.0))
    if hasattr(controller_bundle, "_sync_from_parameter_set"):
        controller_bundle._sync_from_parameter_set()


def _disturbance_vector(
    axis_index: int,
    step: int,
    config: PyBulletControlExperimentConfig,
    axis_config: Any,
) -> np.ndarray:
    disturbance = np.zeros(3, dtype=np.float32)
    if axis_config.include_disturbance:
        disturbance[axis_index] = axis_config.disturbance_scale * np.sin(0.11 * step) * axis_config.disturbance_axis_bias
    return disturbance


def _timeseries_row(
    step: int,
    dt: float,
    axis: str,
    state: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    rpm: np.ndarray,
    reward: float,
    checkpoint: dict[str, float],
    backend: str,
) -> dict[str, float]:
    row = {
        "time": float(step * dt),
        "axis": axis,
        "backend": backend,
        "x": float(state[0]),
        "y": float(state[1]),
        "z": float(state[2]),
        "vx": float(state[10]),
        "vy": float(state[11]),
        "vz": float(state[12]),
        "roll": float(state[7]),
        "pitch": float(state[8]),
        "yaw": float(state[9]),
        "target_x": float(target_pos[0]),
        "target_y": float(target_pos[1]),
        "target_z": float(target_pos[2]),
        "target_vx": float(target_vel[0]),
        "target_vy": float(target_vel[1]),
        "target_vz": float(target_vel[2]),
        "rpm0": float(rpm[0]),
        "rpm1": float(rpm[1]),
        "rpm2": float(rpm[2]),
        "rpm3": float(rpm[3]),
        "reward": float(reward),
    }
    row.update(checkpoint)
    return row


def _logger_row(step: int, dt: float, state: np.ndarray, target_pos: np.ndarray, target_vel: np.ndarray) -> dict[str, float]:
    return {
        "timestamp": float(step * dt),
        "pos_x": float(state[0]),
        "pos_y": float(state[1]),
        "pos_z": float(state[2]),
        "vel_x": float(state[10]),
        "vel_y": float(state[11]),
        "vel_z": float(state[12]),
        "roll": float(state[7]),
        "pitch": float(state[8]),
        "yaw": float(state[9]),
        "target_pos_x": float(target_pos[0]),
        "target_pos_y": float(target_pos[1]),
        "target_pos_z": float(target_pos[2]),
        "target_vel_x": float(target_vel[0]),
        "target_vel_y": float(target_vel[1]),
        "target_vel_z": float(target_vel[2]),
    }


def _euler_to_quaternion(euler_xyz: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in euler_xyz]
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    return np.asarray(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float32,
    )


def _ensure_real_env(env: dict[str, Any], config: PyBulletControlExperimentConfig, reference_bundle: ReferenceBundle) -> Any:
    if env.get("env") is not None:
        return env["env"]
    package_root = Path(__file__).resolve().parents[4] / "Gym_env"
    if str(package_root) not in sys.path:
        sys.path.append(str(package_root))
    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
    from gym_pybullet_drones.utils.enums import DroneModel, Physics

    drone_model = DroneModel(config.drone_model)
    env["env"] = CtrlAviary(
        drone_model=drone_model,
        num_drones=1,
        physics=Physics.PYB,
        neighbourhood_radius=10.0,
        initial_xyzs=np.asarray([reference_bundle.positions[0]], dtype=np.float32),
        pyb_freq=config.simulation_freq_hz,
        ctrl_freq=config.control_freq_hz,
        gui=config.gui,
        record=False,
        obstacles=False,
        user_debug_gui=False,
    )
    env["axis"] = reference_bundle.axis
    return env["env"]


def _can_use_gym_env_backend() -> bool:
    if importlib.util.find_spec("gymnasium") is None or importlib.util.find_spec("pybullet") is None:
        return False
    package_root = Path(__file__).resolve().parents[4] / "Gym_env"
    if str(package_root) not in sys.path:
        sys.path.append(str(package_root))
    return importlib.util.find_spec("gym_pybullet_drones") is not None
