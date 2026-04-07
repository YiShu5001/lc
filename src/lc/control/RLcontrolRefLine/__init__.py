"""Six-phase single-axis reference-line tasks for chapter-3 RL-LADRC training."""

from .adapters import adapt_episode_to_tracking_inputs
from .builders import build_default_xy_task_config, build_refline_episode, sample_phase_plan
from .task_spec import (
    AxisRLRefLineTaskConfig,
    PhaseKind,
    PhaseSpec,
    RefLineEpisodeBundle,
    SampledPhase,
    SampledPhasePlan,
)

__all__ = [
    "PhaseKind",
    "PhaseSpec",
    "AxisRLRefLineTaskConfig",
    "SampledPhase",
    "SampledPhasePlan",
    "RefLineEpisodeBundle",
    "build_default_xy_task_config",
    "sample_phase_plan",
    "build_refline_episode",
    "adapt_episode_to_tracking_inputs",
]

