from __future__ import annotations

from typing import Iterable

import numpy as np

from .task_spec import AxisRLRefLineTaskConfig, PhaseKind, PhaseSpec, RefLineEpisodeBundle, SampledPhase, SampledPhasePlan


def sample_phase_durations(
    config: AxisRLRefLineTaskConfig,
    rng: np.random.Generator,
) -> list[int]:
    """Sample phase durations and normalize them to the configured episode length."""

    total_steps = max(int(round(config.total_duration_sec * config.control_frequency_hz)), 1)
    min_steps = max(int(round(config.min_phase_duration_sec * config.control_frequency_hz)), 1)
    raw_steps: list[int] = []
    for phase in config.phase_specs:
        low, high = phase.duration_range_sec
        sampled = float(rng.uniform(low, high)) if config.enable_randomization and phase.randomize_duration else float(low)
        raw_steps.append(max(int(round(sampled * config.control_frequency_hz)), min_steps))
    sum_steps = sum(raw_steps)
    if sum_steps == total_steps:
        return raw_steps
    scaled = [max(int(round(step * total_steps / max(sum_steps, 1))), min_steps) for step in raw_steps]
    delta = total_steps - sum(scaled)
    index = len(scaled) - 1
    while delta != 0:
        candidate = scaled[index % len(scaled)]
        if delta > 0:
            scaled[index % len(scaled)] = candidate + 1
            delta -= 1
        elif candidate > min_steps:
            scaled[index % len(scaled)] = candidate - 1
            delta += 1
        index -= 1
    return scaled


def sample_phase_value(
    value_range: tuple[float, float],
    *,
    enable_randomization: bool,
    randomize_value: bool,
    rng: np.random.Generator,
) -> float:
    low, high = value_range
    if enable_randomization and randomize_value and low != high:
        return float(rng.uniform(low, high))
    return float(low)


def build_sampled_phase_plan(
    config: AxisRLRefLineTaskConfig,
    rng: np.random.Generator,
) -> SampledPhasePlan:
    """Convert a task template into one fixed episode plan."""

    control_dt = 1.0 / max(config.control_frequency_hz, 1)
    rl_dt = 1.0 / max(config.rl_frequency_hz, 1)
    durations = sample_phase_durations(config, rng)
    phases: list[SampledPhase] = []
    cursor = 0
    last_disturbance_end = 0.0
    for phase_spec, duration_steps in zip(config.phase_specs, durations):
        reference_velocity = sample_phase_value(
            phase_spec.reference_velocity_range,
            enable_randomization=config.enable_randomization,
            randomize_value=phase_spec.randomize_reference_velocity,
            rng=rng,
        )
        disturbance_value = sample_phase_value(
            phase_spec.disturbance_range,
            enable_randomization=config.enable_randomization,
            randomize_value=phase_spec.randomize_disturbance,
            rng=rng,
        )
        disturbance_start, disturbance_end = _resolve_disturbance_edges(
            phase_spec,
            disturbance_value,
            previous_end=last_disturbance_end,
        )
        phase = SampledPhase(
            kind=phase_spec.kind,
            start_step=cursor,
            stop_step=cursor + duration_steps,
            start_time_sec=cursor * control_dt,
            stop_time_sec=(cursor + duration_steps) * control_dt,
            duration_steps=duration_steps,
            duration_sec=duration_steps * control_dt,
            reference_velocity=reference_velocity,
            disturbance_start=disturbance_start,
            disturbance_end=disturbance_end,
        )
        phases.append(phase)
        cursor += duration_steps
        last_disturbance_end = disturbance_end
    total_steps = sum(durations)
    return SampledPhasePlan(
        axis=config.axis,
        total_steps=total_steps,
        total_duration_sec=total_steps * control_dt,
        control_dt=control_dt,
        rl_dt=rl_dt,
        phases=tuple(phases),
    )


def build_reference_velocity_profile(plan: SampledPhasePlan) -> np.ndarray:
    velocities = np.zeros(plan.total_steps, dtype=np.float32)
    for phase in plan.phases:
        velocities[phase.start_step : phase.stop_step] = phase.reference_velocity
    return velocities


def integrate_reference_position(plan: SampledPhasePlan, reference_velocity: np.ndarray) -> np.ndarray:
    positions = np.zeros(plan.total_steps, dtype=np.float32)
    current = 0.0
    for index, velocity in enumerate(reference_velocity):
        positions[index] = current
        current += float(velocity) * plan.control_dt
    return positions


def build_disturbance_profile(
    config: AxisRLRefLineTaskConfig,
    plan: SampledPhasePlan,
) -> np.ndarray:
    disturbance = np.zeros(plan.total_steps, dtype=np.float32)
    for phase in plan.phases:
        disturbance[phase.start_step : phase.stop_step] = _phase_disturbance_values(config, phase)
    return disturbance


def build_time_vector(plan: SampledPhasePlan) -> np.ndarray:
    return np.asarray([index * plan.control_dt for index in range(plan.total_steps)], dtype=np.float32)


def build_phase_table(phases: Iterable[SampledPhase]) -> tuple[dict[str, float | str], ...]:
    rows: list[dict[str, float | str]] = []
    for index, phase in enumerate(phases):
        rows.append(
            {
                "phase_index": float(index),
                "phase_name": phase.kind.value,
                "start_step": float(phase.start_step),
                "stop_step": float(phase.stop_step),
                "start_time_sec": float(phase.start_time_sec),
                "stop_time_sec": float(phase.stop_time_sec),
                "duration_steps": float(phase.duration_steps),
                "duration_sec": float(phase.duration_sec),
                "reference_velocity": float(phase.reference_velocity),
                "disturbance_start": float(phase.disturbance_start),
                "disturbance_end": float(phase.disturbance_end),
            }
        )
    return tuple(rows)


def build_episode_bundle(
    config: AxisRLRefLineTaskConfig,
    plan: SampledPhasePlan,
) -> RefLineEpisodeBundle:
    reference_velocity = build_reference_velocity_profile(plan)
    reference_position = integrate_reference_position(plan, reference_velocity)
    disturbance = build_disturbance_profile(config, plan)
    time = build_time_vector(plan)
    return RefLineEpisodeBundle(
        axis=config.axis,
        time=tuple(float(value) for value in time),
        reference_position=tuple(float(value) for value in reference_position),
        reference_velocity=tuple(float(value) for value in reference_velocity),
        disturbance=tuple(float(value) for value in disturbance),
        phase_table=build_phase_table(plan.phases),
    )


def _resolve_disturbance_edges(phase_spec: PhaseSpec, disturbance_value: float, previous_end: float) -> tuple[float, float]:
    if phase_spec.kind is PhaseKind.DISTURBANCE_HOLD:
        return float(disturbance_value), float(disturbance_value)
    if phase_spec.kind is PhaseKind.REVERSE_CONSTANT_VELOCITY:
        return float(previous_end), float(previous_end)
    if phase_spec.kind is PhaseKind.DISTURBANCE_RECOVERY:
        return float(previous_end), 0.0
    return 0.0, 0.0


def _phase_disturbance_values(config: AxisRLRefLineTaskConfig, phase: SampledPhase) -> np.ndarray:
    length = max(phase.duration_steps, 1)
    if phase.kind is PhaseKind.DISTURBANCE_RECOVERY and phase.disturbance_start != phase.disturbance_end:
        if config.disturbance_decay_mode == "exponential":
            values = np.geomspace(max(abs(phase.disturbance_start), 1e-6), 1e-6, num=length)
            signed = np.sign(phase.disturbance_start) if phase.disturbance_start != 0.0 else 1.0
            return np.asarray([signed * value for value in values], dtype=np.float32)
        return np.linspace(phase.disturbance_start, phase.disturbance_end, num=length, dtype=np.float32)
    return np.full(length, phase.disturbance_end, dtype=np.float32)

