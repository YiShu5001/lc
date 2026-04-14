from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lc.control.configs.pybullet_control_config import AxisTrainingConfig, PyBulletControlExperimentConfig


@dataclass(frozen=True)
class ReferenceBundle:
    axis: str
    positions: np.ndarray
    velocities: np.ndarray
    stage_slices: tuple[slice, ...]
    stage_velocities: tuple[float, ...]


def build_axis_piecewise_velocity_profile(
    config: AxisTrainingConfig,
    rng: np.random.Generator,
    step_count: int,
    dt: float,
) -> tuple[np.ndarray, tuple[slice, ...], tuple[float, ...]]:
    if config.fixed_stage_lengths and config.fixed_stage_velocities:
        lengths = [max(int(value), 1) for value in config.fixed_stage_lengths]
        if len(lengths) != len(config.fixed_stage_velocities):
            raise ValueError("fixed_stage_lengths and fixed_stage_velocities must have the same length")
        total = int(sum(lengths))
        if total != step_count:
            lengths[-1] += step_count - total
        velocities = np.zeros(step_count, dtype=np.float32)
        stage_slices: list[slice] = []
        cursor = 0
        for length, speed in zip(lengths, config.fixed_stage_velocities):
            stop = min(cursor + int(length), step_count)
            stage_slice = slice(cursor, stop)
            velocities[stage_slice] = float(speed)
            stage_slices.append(stage_slice)
            cursor = stop
        if cursor < step_count:
            velocities[cursor:] = float(config.fixed_stage_velocities[-1])
            stage_slices[-1] = slice(stage_slices[-1].start, step_count)
        return velocities, tuple(stage_slices), tuple(float(speed) for speed in config.fixed_stage_velocities)

    stage_count = max(config.stage_count, 4)
    duration_low, duration_high = config.stage_duration_range
    durations = rng.uniform(duration_low, duration_high, size=stage_count)
    durations = durations / max(np.sum(durations), 1e-6)
    raw_lengths = np.maximum((durations * step_count).astype(int), 1)
    raw_lengths[-1] += step_count - int(np.sum(raw_lengths))
    velocities = np.zeros(step_count, dtype=np.float32)
    stage_slices: list[slice] = []
    stage_values = (
        0.0,
        float(rng.uniform(*config.primary_speed_range)),
        0.0,
        float(rng.uniform(*config.reverse_speed_range)),
    )
    cursor = 0
    for index in range(stage_count):
        segment_length = int(raw_lengths[index])
        stop = min(cursor + segment_length, step_count)
        stage_slice = slice(cursor, stop)
        speed = stage_values[index] if index < len(stage_values) else float(rng.uniform(*config.primary_speed_range))
        velocities[stage_slice] = speed
        stage_slices.append(stage_slice)
        cursor = stop
    if cursor < step_count:
        velocities[cursor:] = velocities[max(cursor - 1, 0)]
        stage_slices[-1] = slice(stage_slices[-1].start, step_count)
    return velocities, tuple(stage_slices), tuple(float(velocities[segment.start]) for segment in stage_slices)


def integrate_velocity_profile(initial_position: float, velocity_profile: np.ndarray, dt: float) -> np.ndarray:
    positions = np.zeros(len(velocity_profile), dtype=np.float32)
    current = float(initial_position)
    for index, velocity in enumerate(velocity_profile):
        positions[index] = current
        current += float(velocity) * dt
    return positions


def build_xyz_reference_trajectory(
    axis_config: AxisTrainingConfig,
    episode_config: PyBulletControlExperimentConfig,
    rng: np.random.Generator | None = None,
) -> ReferenceBundle:
    generator = rng or np.random.default_rng(episode_config.seed)
    step_count = episode_config.step_count
    velocities_axis, stage_slices, stage_velocities = build_axis_piecewise_velocity_profile(
        axis_config,
        generator,
        step_count=step_count,
        dt=episode_config.control_dt,
    )
    positions = np.zeros((step_count, 3), dtype=np.float32)
    velocities = np.zeros((step_count, 3), dtype=np.float32)
    positions[:, 0] = axis_config.initial_position[0]
    positions[:, 1] = axis_config.initial_position[1]
    positions[:, 2] = axis_config.initial_position[2]
    axis_index = {"x": 0, "y": 1, "z": 2}[axis_config.axis]
    positions[:, axis_index] = integrate_velocity_profile(
        axis_config.initial_position[axis_index],
        velocities_axis,
        episode_config.control_dt,
    )
    velocities[:, axis_index] = velocities_axis
    return ReferenceBundle(
        axis=axis_config.axis,
        positions=positions,
        velocities=velocities,
        stage_slices=stage_slices,
        stage_velocities=stage_velocities,
    )


def summarize_reference_segments(bundle: ReferenceBundle) -> list[dict[str, float]]:
    summary: list[dict[str, float]] = []
    for index, (segment, speed) in enumerate(zip(bundle.stage_slices, bundle.stage_velocities)):
        summary.append(
            {
                "stage": float(index),
                "start": float(segment.start),
                "stop": float(segment.stop),
                "velocity": float(speed),
            }
        )
    return summary
