from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PhaseKind(str, Enum):
    """Fixed six-phase task order for chapter-3 RL-LADRC axis training."""

    HOLD_START = "hold_start"
    FORWARD_CONSTANT_VELOCITY = "forward_constant_velocity"
    DISTURBANCE_HOLD = "disturbance_hold"
    REVERSE_CONSTANT_VELOCITY = "reverse_constant_velocity"
    DISTURBANCE_RECOVERY = "disturbance_recovery"
    HOLD_END = "hold_end"


@dataclass(frozen=True)
class PhaseSpec:
    """User-facing phase config.

    Update this object when you want to change one phase's duration, reference speed,
    or disturbance range without touching the profile generation code.
    """

    kind: PhaseKind
    duration_range_sec: tuple[float, float]
    reference_velocity_range: tuple[float, float]
    disturbance_range: tuple[float, float] = (0.0, 0.0)
    randomize_duration: bool = True
    randomize_reference_velocity: bool = True
    randomize_disturbance: bool = True


@dataclass(frozen=True)
class AxisRLRefLineTaskConfig:
    """Single-axis six-phase task template used by RL-LADRC training."""

    axis: str
    total_duration_sec: float = 8.0
    control_frequency_hz: int = 100
    rl_frequency_hz: int = 10
    enable_randomization: bool = True
    disturbance_decay_mode: str = "linear"
    min_phase_duration_sec: float = 0.6
    phase_specs: tuple[PhaseSpec, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SampledPhase:
    """One concrete phase after random sampling for an episode."""

    kind: PhaseKind
    start_step: int
    stop_step: int
    start_time_sec: float
    stop_time_sec: float
    duration_steps: int
    duration_sec: float
    reference_velocity: float
    disturbance_start: float
    disturbance_end: float


@dataclass(frozen=True)
class SampledPhasePlan:
    """Concrete six-phase episode plan with fixed timing and amplitudes."""

    axis: str
    total_steps: int
    total_duration_sec: float
    control_dt: float
    rl_dt: float
    phases: tuple[SampledPhase, ...]


@dataclass(frozen=True)
class RefLineEpisodeBundle:
    """Episode artifact consumed by the chapter-3 environment."""

    axis: str
    time: tuple[float, ...]
    reference_position: tuple[float, ...]
    reference_velocity: tuple[float, ...]
    disturbance: tuple[float, ...]
    phase_table: tuple[dict[str, float | str], ...]

