from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import importlib.util
import sys
from collections import deque

import numpy as np

from lc.control.configs import PyBulletControlExperimentConfig, get_axis_ladrc_action_bounds
from lc.control.controllers import ControllerBundle
from lc.control.policies.stacking import stack_state
from lc.control.reference_generators.piecewise_velocity import ReferenceBundle


@dataclass
class SimulationArtifacts:
    timeseries: list[dict[str, float]]
    logger_rows: list[dict[str, float]]
    rewards: list[float]
    final_state: np.ndarray
    backend: str


def _recording_enabled(config: PyBulletControlExperimentConfig) -> bool:
    return bool(getattr(config.artifact, "record_video", False))


def _capture_real_frame(real_env: Any, output_dir: Path, frame_idx: int) -> None:
    if importlib.util.find_spec("pybullet") is None or importlib.util.find_spec("PIL") is None:
        return
    import pybullet as p
    from PIL import Image

    client = getattr(real_env, "CLIENT", None)
    if client is None:
        return
    width = int(getattr(real_env, "VID_WIDTH", 640))
    height = int(getattr(real_env, "VID_HEIGHT", 480))
    view = getattr(real_env, "CAM_VIEW", None)
    proj = getattr(real_env, "CAM_PRO", None)
    if view is None or proj is None:
        view = p.computeViewMatrixFromYawPitchRoll(
            distance=3.0,
            yaw=-30.0,
            pitch=-30.0,
            roll=0.0,
            cameraTargetPosition=[0.0, 0.0, 0.0],
            upAxisIndex=2,
            physicsClientId=client,
        )
        proj = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=width / max(height, 1),
            nearVal=0.1,
            farVal=1000.0,
        )
    _, _, rgb, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client,
    )
    image = np.asarray(rgb, dtype=np.uint8).reshape(height, width, 4)[..., :3]
    Image.fromarray(image).save(output_dir / f"frame_{frame_idx:04d}.png")


def _finalize_recording(output_dir: Path, fps: int) -> str | None:
    if importlib.util.find_spec("PIL") is None:
        return None
    from PIL import Image

    frame_paths = sorted(output_dir.glob("frame_*.png"))
    if not frame_paths:
        return None
    images = [Image.open(path) for path in frame_paths]
    gif_path = output_dir / "episode.gif"
    duration_ms = max(int(round(1000 / max(fps, 1))), 1)
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    for image in images:
        image.close()
    return str(gif_path)


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
    n_step: int,
    config: PyBulletControlExperimentConfig,
) -> SimulationArtifacts:
    del action_hold_steps
    return run_policy_episode(
        env,
        policy,
        controller_bundle,
        reference_bundle,
        axis,
        config,
        explore=True,
        store_transitions=True,
        n_step=n_step,
    )


def run_policy_episode(
    env: dict[str, Any],
    policy: Any,
    controller_bundle: ControllerBundle,
    reference_bundle: ReferenceBundle,
    axis: str,
    config: PyBulletControlExperimentConfig,
    *,
    explore: bool,
    store_transitions: bool,
    n_step: int = 1,
) -> SimulationArtifacts:
    if env["backend"] == "gym_env":
        return _run_real_policy_episode(
            env,
            policy,
            controller_bundle,
            reference_bundle,
            axis,
            config,
            explore=explore,
            store_transitions=store_transitions,
            n_step=n_step,
        )
    return _run_fallback_policy_episode(
        env,
        policy,
        controller_bundle,
        reference_bundle,
        axis,
        config,
        explore=explore,
        store_transitions=store_transitions,
        n_step=n_step,
    )


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
    n_step: int,
    config: PyBulletControlExperimentConfig,
) -> SimulationArtifacts:
    return _run_real_policy_episode(
        env,
        policy,
        controller_bundle,
        reference_bundle,
        axis,
        config,
        explore=True,
        store_transitions=True,
        n_step=n_step,
    )


def _run_real_policy_episode(
    env: dict[str, Any],
    policy: Any,
    controller_bundle: ControllerBundle,
    reference_bundle: ReferenceBundle,
    axis: str,
    config: PyBulletControlExperimentConfig,
    *,
    explore: bool,
    store_transitions: bool,
    n_step: int,
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
    action = np.zeros(policy.config.action_dim, dtype=np.float32)
    rollout: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque()
    history: list[np.ndarray] = []
    prev_rpm = np.full(4, 4300.0, dtype=np.float32)
    prev_param_values: np.ndarray | None = None
    recording_dir = Path(config.artifact.video_output_dir or Path(config.artifact.output_root) / "videos")
    gif_path: str | None = None
    if _recording_enabled(config):
        recording_dir.mkdir(parents=True, exist_ok=True)
        _capture_real_frame(real_env, recording_dir, 0)
    for step in range(config.step_count):
        target_pos = reference_bundle.positions[step]
        target_vel = reference_bundle.velocities[step]
        observation = _build_axis_observation(state, target_pos, target_vel, axis, step, config.step_count)
        stacked_observation = stack_state(history, observation.copy(), policy.config.stack_size)
        action = policy.select_action(stacked_observation, explore=explore)
        applied_params = _apply_axis_action(controller_bundle, axis, action)
        checkpoint = controller_bundle.snapshot_params()
        disturbance = _disturbance_vector(axis_index, step, config, config.axis_config(axis))
        rpm, _, _ = controller_bundle.compute_control_from_state(
            control_timestep=config.control_dt,
            state=state,
            target_pos=target_pos,
            target_vel=target_vel,
            target_rpy=np.zeros(3, dtype=np.float32),
            target_rpy_rates=np.zeros(3, dtype=np.float32),
        )
        _apply_real_disturbance(real_env, disturbance)
        next_obs, _, terminated, truncated, _ = real_env.step(rpm.reshape(1, 4))
        next_state = np.asarray(next_obs[0], dtype=np.float32)
        pos_error = float(target_pos[axis_index] - next_state[axis_index])
        vel_error = float(target_vel[axis_index] - next_state[10 + axis_index])
        rpm_delta = float(np.mean(np.abs(rpm - prev_rpm)))
        param_delta = 0.0 if prev_param_values is None else float(np.mean(np.abs(applied_params - prev_param_values)))
        reward = _compute_axis_reward(
            pos_error,
            vel_error,
            rpm_delta,
            step=step,
            step_count=config.step_count,
            target_vel=float(target_vel[axis_index]),
            param_delta=param_delta,
        )
        done = bool(terminated or truncated or step == config.step_count - 1)
        next_target_pos = reference_bundle.positions[min(step + 1, config.step_count - 1)]
        next_target_vel = reference_bundle.velocities[min(step + 1, config.step_count - 1)]
        next_observation = _build_axis_observation(
            next_state,
            next_target_pos,
            next_target_vel,
            axis,
            step + 1,
            config.step_count,
        )
        next_history = list(history)
        next_stacked_observation = stack_state(next_history, next_observation.copy(), policy.config.stack_size)
        if store_transitions:
            rollout.append((stacked_observation.copy(), action.copy(), reward, next_stacked_observation.copy(), done))
            _flush_n_step_transitions(policy, rollout, n_step, policy.config.gamma, force=done)
        rewards.append(float(reward))
        timeseries.append(
            _timeseries_row(
                step,
                config.control_dt,
                axis,
                state,
                target_pos,
                target_vel,
                rpm,
                reward,
                checkpoint,
                env["backend"],
                disturbance=disturbance,
            )
        )
        logger_rows.append(_logger_row(step, config.control_dt, state, target_pos, target_vel))
        state = next_state
        history = next_history
        prev_rpm = rpm
        prev_param_values = applied_params
        if done:
            break
        if _recording_enabled(config):
            _capture_real_frame(real_env, recording_dir, step + 1)
    if _recording_enabled(config):
        gif_path = _finalize_recording(recording_dir, fps=config.control_freq_hz)
        if gif_path is not None:
            env["recording_gif"] = gif_path
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
    recording_dir = Path(config.artifact.video_output_dir or Path(config.artifact.output_root) / "videos")
    gif_path: str | None = None
    if _recording_enabled(config):
        recording_dir.mkdir(parents=True, exist_ok=True)
        _capture_real_frame(real_env, recording_dir, 0)
    for step in range(config.step_count):
        target_pos = reference_bundle.positions[step]
        target_vel = reference_bundle.velocities[step]
        checkpoint = controller_bundle.snapshot_params()
        disturbance = _disturbance_vector(axis_index, step, config, config.axis_config(axis))
        rpm, _, _ = controller_bundle.compute_control_from_state(
            control_timestep=config.control_dt,
            state=state,
            target_pos=target_pos,
            target_vel=target_vel,
            target_rpy=np.zeros(3, dtype=np.float32),
            target_rpy_rates=np.zeros(3, dtype=np.float32),
        )
        _apply_real_disturbance(real_env, disturbance)
        next_obs, _, terminated, truncated, _ = real_env.step(rpm.reshape(1, 4))
        next_state = np.asarray(next_obs[0], dtype=np.float32)
        pos_error = float(target_pos[axis_index] - next_state[axis_index])
        vel_error = float(target_vel[axis_index] - next_state[10 + axis_index])
        rpm_delta = float(np.mean(np.abs(rpm - prev_rpm)))
        reward = _compute_axis_reward(
            pos_error,
            vel_error,
            rpm_delta,
            step=step,
            step_count=config.step_count,
            target_vel=float(target_vel[axis_index]),
            param_delta=0.0,
        )
        rewards.append(float(reward))
        timeseries.append(
            _timeseries_row(
                step,
                config.control_dt,
                axis,
                state,
                target_pos,
                target_vel,
                rpm,
                reward,
                checkpoint,
                env["backend"],
                disturbance=disturbance,
            )
        )
        logger_rows.append(_logger_row(step, config.control_dt, state, target_pos, target_vel))
        state = next_state
        prev_rpm = rpm
        if terminated or truncated:
            break
        if _recording_enabled(config):
            _capture_real_frame(real_env, recording_dir, step + 1)
    if _recording_enabled(config):
        gif_path = _finalize_recording(recording_dir, fps=config.control_freq_hz)
        if gif_path is not None:
            env["recording_gif"] = gif_path
    return SimulationArtifacts(timeseries=timeseries, logger_rows=logger_rows, rewards=rewards, final_state=state, backend=env["backend"])


def _run_fallback_training_episode(
    env: dict[str, Any],
    policy: Any,
    controller_bundle: ControllerBundle,
    reference_bundle: ReferenceBundle,
    axis: str,
    action_hold_steps: int,
    n_step: int,
    config: PyBulletControlExperimentConfig,
) -> SimulationArtifacts:
    return _run_fallback_policy_episode(
        env,
        policy,
        controller_bundle,
        reference_bundle,
        axis,
        config,
        explore=True,
        store_transitions=True,
        n_step=n_step,
    )


def _run_fallback_policy_episode(
    env: dict[str, Any],
    policy: Any,
    controller_bundle: ControllerBundle,
    reference_bundle: ReferenceBundle,
    axis: str,
    config: PyBulletControlExperimentConfig,
    *,
    explore: bool,
    store_transitions: bool,
    n_step: int,
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
    action = np.zeros(policy.config.action_dim, dtype=np.float32)
    rollout: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque()
    history: list[np.ndarray] = []
    prev_rpm = np.full(4, 4300.0, dtype=np.float32)
    prev_param_values: np.ndarray | None = None
    for step in range(config.step_count):
        target_pos = reference_bundle.positions[step]
        target_vel = reference_bundle.velocities[step]
        observation = _build_axis_observation(state, target_pos, target_vel, axis, step, config.step_count)
        stacked_observation = stack_state(history, observation.copy(), policy.config.stack_size)
        action = policy.select_action(stacked_observation, explore=explore)
        applied_params = _apply_axis_action(controller_bundle, axis, action)
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
        next_target_pos = reference_bundle.positions[min(step + 1, config.step_count - 1)]
        next_target_vel = reference_bundle.velocities[min(step + 1, config.step_count - 1)]
        next_observation = _build_axis_observation(
            next_state,
            next_target_pos,
            next_target_vel,
            axis,
            step + 1,
            config.step_count,
        )
        rpm_delta = float(np.mean(np.abs(rpm - prev_rpm)))
        param_delta = 0.0 if prev_param_values is None else float(np.mean(np.abs(applied_params - prev_param_values)))
        reward = _compute_axis_reward(
            pos_error[axis_index],
            target_vel[axis_index] - next_state[10 + axis_index],
            rpm_delta,
            step=step,
            step_count=config.step_count,
            target_vel=float(target_vel[axis_index]),
            param_delta=param_delta,
        )
        done = step == config.step_count - 1
        next_history = list(history)
        next_stacked_observation = stack_state(next_history, next_observation.copy(), policy.config.stack_size)
        if store_transitions:
            rollout.append((stacked_observation.copy(), action.copy(), reward, next_stacked_observation.copy(), done))
            _flush_n_step_transitions(policy, rollout, n_step, policy.config.gamma, force=done)
        rewards.append(float(reward))
        timeseries.append(
            _timeseries_row(
                step,
                config.control_dt,
                axis,
                state,
                target_pos,
                target_vel,
                rpm,
                reward,
                checkpoint,
                "fallback",
                disturbance=disturbance,
            )
        )
        logger_rows.append(_logger_row(step, config.control_dt, state, target_pos, target_vel))
        state = next_state
        history = next_history
        prev_rpm = rpm
        prev_param_values = applied_params
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
        reward = _compute_axis_reward(
            pos_error[axis_index],
            target_vel[axis_index] - next_state[10 + axis_index],
            0.0,
            step=step,
            step_count=config.step_count,
            target_vel=float(target_vel[axis_index]),
            param_delta=0.0,
        )
        rewards.append(float(reward))
        timeseries.append(
            _timeseries_row(
                step,
                config.control_dt,
                axis,
                state,
                target_pos,
                target_vel,
                rpm,
                reward,
                checkpoint,
                "fallback",
                disturbance=disturbance,
            )
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


def _compute_axis_reward(
    pos_error: float,
    vel_error: float,
    rpm_delta: float,
    *,
    step: int,
    step_count: int,
    target_vel: float,
    param_delta: float,
) -> float:
    del vel_error, rpm_delta, step, step_count, target_vel, param_delta
    position_term = abs(float(pos_error))
    return float(-position_term)


def _initial_state(reference_bundle: ReferenceBundle) -> np.ndarray:
    state = np.zeros(20, dtype=np.float32)
    state[0:3] = reference_bundle.positions[0]
    state[2] = max(state[2], 0.4)
    state[3:7] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    state[16:20] = 4300.0
    return state


def _apply_axis_action(controller_bundle: ControllerBundle, axis: str, action: np.ndarray) -> np.ndarray:
    axis_config = controller_bundle.parameter_set.axis_config(axis)
    bounds = get_axis_ladrc_action_bounds(axis)
    r, b0, omega_c, k = _decode_action_to_ladrc_params(action, bounds)
    axis_config.r = float(r)
    axis_config.b0 = float(b0)
    axis_config.omega_c = float(omega_c)
    axis_config.k = float(k)
    if hasattr(controller_bundle, "_sync_from_parameter_set"):
        controller_bundle._sync_from_parameter_set()
    return np.asarray([axis_config.r, axis_config.b0, axis_config.omega_c, axis_config.k], dtype=np.float32)


def _decode_action_to_ladrc_params(action: np.ndarray, bounds: Any) -> tuple[float, float, float, float]:
    clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    r = np.clip(bounds.train_anchor.r + _map_to_bounds(clipped[0], bounds.delta_r), bounds.r[0], bounds.r[1])
    b0 = np.clip(bounds.train_anchor.b0 + _map_to_bounds(clipped[1], bounds.delta_b0), bounds.b0[0], bounds.b0[1])
    omega_c = np.clip(
        bounds.train_anchor.wc + _map_to_bounds(clipped[2], bounds.delta_wc),
        bounds.wc[0],
        bounds.wc[1],
    )
    k = np.clip(bounds.train_anchor.k + _map_to_bounds(clipped[3], bounds.delta_k), bounds.k[0], bounds.k[1])
    return float(r), float(b0), float(omega_c), float(k)


def _flush_n_step_transitions(
    policy: Any,
    rollout: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]],
    n_step: int,
    gamma: float,
    force: bool = False,
) -> None:
    effective_n = max(int(n_step), 1)
    while rollout and (force or len(rollout) >= effective_n):
        reward = 0.0
        next_state = rollout[0][3]
        done = rollout[0][4]
        for index, (_, _, step_reward, step_next_state, step_done) in enumerate(rollout):
            if index >= effective_n:
                break
            reward += (gamma**index) * float(step_reward)
            next_state = step_next_state
            done = bool(step_done)
            if step_done:
                break
        state, action, _, _, _ = rollout.popleft()
        policy.store_transition(state, action, reward, next_state, done)
        if not force:
            break


def _disturbance_vector(
    axis_index: int,
    step: int,
    config: PyBulletControlExperimentConfig,
    axis_config: Any,
) -> np.ndarray:
    disturbance = np.zeros(3, dtype=np.float32)
    if axis_config.include_disturbance:
        window = getattr(axis_config, "disturbance_step_window", None)
        if window is None or (int(window[0]) <= int(step) < int(window[1])):
            mode = str(getattr(axis_config, "disturbance_mode", "sine")).lower()
            if mode == "random_uniform":
                rng = np.random.default_rng(int(config.seed) * 10007 + int(axis_index) * 1009 + int(step))
                sample = rng.uniform(-1.0, 1.0)
                disturbance[axis_index] = axis_config.disturbance_scale * sample * axis_config.disturbance_axis_bias
            else:
                omega = float(getattr(axis_config, "disturbance_frequency_rad", 0.11))
                disturbance[axis_index] = axis_config.disturbance_scale * np.sin(omega * step) * axis_config.disturbance_axis_bias
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
    disturbance: np.ndarray | None = None,
) -> dict[str, float]:
    disturbance = disturbance if disturbance is not None else np.zeros(3, dtype=np.float32)
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
        "disturbance_x": float(disturbance[0]),
        "disturbance_y": float(disturbance[1]),
        "disturbance_z": float(disturbance[2]),
    }
    row.update(checkpoint)
    return row


def _map_to_bounds(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    scaled = (float(value) + 1.0) * 0.5
    return low + scaled * (high - low)


def _apply_real_disturbance(real_env: Any, disturbance: np.ndarray) -> None:
    if not np.any(np.abs(disturbance) > 0.0):
        return
    if importlib.util.find_spec("pybullet") is None:
        return
    import pybullet as p

    drone_ids = getattr(real_env, "DRONE_IDS", None)
    client = getattr(real_env, "CLIENT", None)
    if drone_ids is None or client is None or len(drone_ids) == 0:
        return
    p.applyExternalForce(
        int(drone_ids[0]),
        -1,
        forceObj=[float(disturbance[0]), float(disturbance[1]), float(disturbance[2])],
        posObj=[0.0, 0.0, 0.0],
        flags=p.WORLD_FRAME,
        physicsClientId=client,
    )


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
    video_output_dir = config.artifact.video_output_dir
    if video_output_dir is None:
        video_output_dir = str(Path(config.artifact.output_root) / "videos")
    Path(video_output_dir).mkdir(parents=True, exist_ok=True)
    env["env"] = CtrlAviary(
        drone_model=drone_model,
        num_drones=1,
        physics=Physics.PYB,
        neighbourhood_radius=10.0,
        initial_xyzs=np.asarray([reference_bundle.positions[0]], dtype=np.float32),
        pyb_freq=config.simulation_freq_hz,
        ctrl_freq=config.control_freq_hz,
        gui=config.gui,
        record=config.artifact.record_video,
        obstacles=False,
        user_debug_gui=False,
        output_folder=video_output_dir,
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
