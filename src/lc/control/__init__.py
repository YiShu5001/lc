"""Chapter-3 control layer implementation."""

from .experiments.compare import run_control_comparison
from .experiments.pybullet_compare import (
    run_pybullet_axis_training,
    run_pybullet_controller_benchmark,
    run_pybullet_full_experiment,
)

__all__ = [
    "run_control_comparison",
    "run_pybullet_axis_training",
    "run_pybullet_controller_benchmark",
    "run_pybullet_full_experiment",
]
