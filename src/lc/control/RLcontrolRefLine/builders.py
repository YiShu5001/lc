from __future__ import annotations

import numpy as np

from .profiles import build_episode_bundle, build_sampled_phase_plan
from .task_spec import AxisRLRefLineTaskConfig, PhaseKind, PhaseSpec, RefLineEpisodeBundle, SampledPhasePlan


def build_default_xy_task_config(axis: str) -> AxisRLRefLineTaskConfig:
    """Return the default six-phase task template for x/y axis RL-LADRC tuning."""

    phase_specs = (
        PhaseSpec(PhaseKind.HOLD_START, (0.8, 1.2), (0.0, 0.0), (0.0, 0.0), randomize_reference_velocity=False, randomize_disturbance=False),
        PhaseSpec(PhaseKind.FORWARD_CONSTANT_VELOCITY, (1.2, 2.0), (0.25, 0.55), (0.0, 0.0), randomize_disturbance=False),
        PhaseSpec(PhaseKind.DISTURBANCE_HOLD, (1.0, 1.8), (0.0, 0.0), (0.06, 0.16), randomize_reference_velocity=False),
        PhaseSpec(PhaseKind.REVERSE_CONSTANT_VELOCITY, (1.2, 2.0), (-0.50, -0.20), (0.0, 0.0), randomize_disturbance=False),
        PhaseSpec(PhaseKind.DISTURBANCE_RECOVERY, (0.8, 1.5), (0.0, 0.0), (0.0, 0.0), randomize_reference_velocity=False, randomize_disturbance=False),
        PhaseSpec(PhaseKind.HOLD_END, (0.8, 1.2), (0.0, 0.0), (0.0, 0.0), randomize_reference_velocity=False, randomize_disturbance=False),
    )
    return AxisRLRefLineTaskConfig(axis=axis, phase_specs=phase_specs)


def sample_phase_plan(config: AxisRLRefLineTaskConfig, seed: int | None = None) -> SampledPhasePlan:
    """Sample a concrete episode plan from a task template."""

    rng = np.random.default_rng(seed)
    return build_sampled_phase_plan(config, rng)


def build_refline_episode(config: AxisRLRefLineTaskConfig, seed: int | None = None) -> RefLineEpisodeBundle:
    """Generate one complete six-phase episode bundle."""

    plan = sample_phase_plan(config, seed=seed)
    return build_episode_bundle(config, plan)

