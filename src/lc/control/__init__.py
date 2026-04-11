"""Chapter-3 control layer implementation."""

from .experiments.compare import run_control_comparison, run_pid_vs_fixed_ladrc_tracking
from .experiments.pybullet_compare import (
    run_pybullet_axis_training,
    run_pybullet_controller_benchmark,
    run_pybullet_full_experiment,
)

__all__ = [
    "run_control_comparison",
    "run_pid_vs_fixed_ladrc_tracking",
    "run_pybullet_axis_training",
    "run_pybullet_controller_benchmark",
    "run_pybullet_full_experiment",
]
