from __future__ import annotations

from typing import Iterable

import numpy as np

from control.Tuning_ladrc.schemas import ManualTargetProfile
from lc.control.reference_generators.piecewise_velocity import ReferenceBundle, integrate_velocity_profile


def build_manual_reference_profile(
    profile: ManualTargetProfile,
    *,
    control_dt: float,
    step_count: int,
) -> ReferenceBundle:
    if profile.mode == "x_hold_disturbance_hold":
        return _build_x_hold_disturbance_hold_reference(profile, control_dt=control_dt, step_count=step_count)
    if profile.mode == "x_hold_sine_disturbance_hold":
        return _build_x_hold_sine_disturbance_hold_reference(profile, control_dt=control_dt, step_count=step_count)
    if profile.mode == "x_small_step_hold":
        return _build_x_small_step_hold_reference(profile, control_dt=control_dt, step_count=step_count)
    if profile.mode == "z_hold":
        return _build_z_hold_reference(profile, control_dt=control_dt, step_count=step_count)
    if profile.mode == "z_small_step":
        return _build_z_small_step_reference(profile, control_dt=control_dt, step_count=step_count)
    if profile.mode == "step":
        return _build_step_reference(profile, control_dt=control_dt, step_count=step_count)
    if profile.mode == "hold_step_hold":
        return _build_hold_step_hold_reference(profile, control_dt=control_dt, step_count=step_count)
    if profile.mode == "hold_step_hold_reverse":
        return _build_hold_step_hold_reverse_reference(profile, control_dt=control_dt, step_count=step_count)
    if profile.mode == "piecewise_velocity":
        return _build_piecewise_velocity_reference(profile, control_dt=control_dt, step_count=step_count)
    raise ValueError(f"Unsupported target profile mode: {profile.mode}")


def _base_arrays(profile: ManualTargetProfile, step_count: int) -> tuple[np.ndarray, np.ndarray, int]:
    positions = np.zeros((step_count, 3), dtype=np.float32)
    velocities = np.zeros((step_count, 3), dtype=np.float32)
    positions[:, 0] = profile.initial_position[0]
    positions[:, 1] = profile.initial_position[1]
    positions[:, 2] = profile.initial_position[2]
    axis_index = {"x": 0, "y": 1, "z": 2}[profile.axis]
    return positions, velocities, axis_index


def _durations_to_slices(durations: Iterable[float], control_dt: float, step_count: int) -> tuple[slice, ...]:
    lengths = [max(int(round(float(duration) / control_dt)), 1) for duration in durations]
    cursor = 0
    slices: list[slice] = []
    for length in lengths:
        stop = min(cursor + length, step_count)
        slices.append(slice(cursor, stop))
        cursor = stop
    if slices:
        slices[-1] = slice(slices[-1].start, step_count)
    return tuple(slices)


def _build_step_reference(profile: ManualTargetProfile, *, control_dt: float, step_count: int) -> ReferenceBundle:
    positions, velocities, axis_index = _base_arrays(profile, step_count)
    step_start = max(step_count // 4, 1)
    positions[step_start:, axis_index] = profile.initial_position[axis_index] + float(profile.step_value)
    stage_slices = (slice(0, step_start), slice(step_start, step_count))
    return ReferenceBundle(
        axis=profile.axis,
        positions=positions,
        velocities=velocities,
        stage_slices=stage_slices,
        stage_velocities=(0.0, 0.0),
    )


def _build_hold_step_hold_reference(profile: ManualTargetProfile, *, control_dt: float, step_count: int) -> ReferenceBundle:
    positions, velocities, axis_index = _base_arrays(profile, step_count)
    stage_slices = _durations_to_slices(profile.segment_durations, control_dt, step_count)
    hold_value = profile.initial_position[axis_index]
    step_value = hold_value + float(profile.step_value)
    if len(stage_slices) >= 1:
        positions[stage_slices[0], axis_index] = hold_value
    if len(stage_slices) >= 2:
        positions[stage_slices[1], axis_index] = step_value
    if len(stage_slices) >= 3:
        positions[stage_slices[2], axis_index] = step_value
    return ReferenceBundle(
        axis=profile.axis,
        positions=positions,
        velocities=velocities,
        stage_slices=stage_slices,
        stage_velocities=tuple(0.0 for _ in stage_slices),
    )


def _build_hold_step_hold_reverse_reference(profile: ManualTargetProfile, *, control_dt: float, step_count: int) -> ReferenceBundle:
    positions, velocities, axis_index = _base_arrays(profile, step_count)
    stage_slices = _durations_to_slices(profile.segment_durations, control_dt, step_count)
    hold_value = profile.initial_position[axis_index]
    forward_value = hold_value + float(profile.step_value)
    reverse_offset = float(profile.reverse_step_value if profile.reverse_step_value is not None else -profile.step_value)
    reverse_value = hold_value + reverse_offset
    values = (hold_value, forward_value, forward_value, reverse_value)
    for segment, value in zip(stage_slices, values):
        positions[segment, axis_index] = value
    return ReferenceBundle(
        axis=profile.axis,
        positions=positions,
        velocities=velocities,
        stage_slices=stage_slices,
        stage_velocities=tuple(0.0 for _ in stage_slices),
    )


def _build_piecewise_velocity_reference(profile: ManualTargetProfile, *, control_dt: float, step_count: int) -> ReferenceBundle:
    positions, velocities, axis_index = _base_arrays(profile, step_count)
    stage_slices = _durations_to_slices(profile.segment_durations, control_dt, step_count)
    speed_plan = (0.0, float(profile.velocity_value), 0.0, -float(profile.velocity_value))
    axis_velocity = np.zeros(step_count, dtype=np.float32)
    for segment, speed in zip(stage_slices, speed_plan):
        axis_velocity[segment] = float(speed)
    positions[:, axis_index] = integrate_velocity_profile(
        profile.initial_position[axis_index],
        axis_velocity,
        control_dt,
    )
    velocities[:, axis_index] = axis_velocity
    return ReferenceBundle(
        axis=profile.axis,
        positions=positions,
        velocities=velocities,
        stage_slices=stage_slices,
        stage_velocities=tuple(float(axis_velocity[segment.start]) for segment in stage_slices),
    )


def _build_z_hold_reference(profile: ManualTargetProfile, *, control_dt: float, step_count: int) -> ReferenceBundle:
    positions, velocities, axis_index = _base_arrays(profile, step_count)
    positions[:, axis_index] = float(profile.hover_reference)
    stage_slices = (slice(0, step_count),)
    return ReferenceBundle(
        axis=profile.axis,
        positions=positions,
        velocities=velocities,
        stage_slices=stage_slices,
        stage_velocities=(0.0,),
    )


def _build_z_small_step_reference(profile: ManualTargetProfile, *, control_dt: float, step_count: int) -> ReferenceBundle:
    local_profile = ManualTargetProfile(
        axis=profile.axis,
        mode="hold_step_hold",
        initial_position=profile.initial_position,
        step_value=profile.step_value,
        reverse_step_value=profile.reverse_step_value,
        fixed_axes=profile.fixed_axes,
        segment_durations=profile.segment_durations,
        velocity_value=profile.velocity_value,
        total_duration=profile.total_duration,
        hover_reference=profile.hover_reference,
    )
    return _build_hold_step_hold_reference(local_profile, control_dt=control_dt, step_count=step_count)


def _build_x_hold_disturbance_hold_reference(
    profile: ManualTargetProfile,
    *,
    control_dt: float,
    step_count: int,
) -> ReferenceBundle:
    positions, velocities, axis_index = _base_arrays(profile, step_count)
    stage_slices = _durations_to_slices(profile.segment_durations, control_dt, step_count)
    hold_value = profile.initial_position[axis_index]
    for segment in stage_slices:
        positions[segment, axis_index] = hold_value
    return ReferenceBundle(
        axis=profile.axis,
        positions=positions,
        velocities=velocities,
        stage_slices=stage_slices,
        stage_velocities=tuple(0.0 for _ in stage_slices),
    )


def _build_x_hold_sine_disturbance_hold_reference(
    profile: ManualTargetProfile,
    *,
    control_dt: float,
    step_count: int,
) -> ReferenceBundle:
    return _build_x_hold_disturbance_hold_reference(profile, control_dt=control_dt, step_count=step_count)


def _build_x_small_step_hold_reference(
    profile: ManualTargetProfile,
    *,
    control_dt: float,
    step_count: int,
) -> ReferenceBundle:
    local_profile = ManualTargetProfile(
        axis=profile.axis,
        mode="hold_step_hold",
        initial_position=profile.initial_position,
        step_value=profile.step_value,
        reverse_step_value=profile.reverse_step_value,
        fixed_axes=profile.fixed_axes,
        segment_durations=profile.segment_durations,
        velocity_value=profile.velocity_value,
        total_duration=profile.total_duration,
        hover_reference=profile.hover_reference,
    )
    return _build_hold_step_hold_reference(local_profile, control_dt=control_dt, step_count=step_count)
