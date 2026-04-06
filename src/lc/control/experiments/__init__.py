"""Control experiments."""

from .compare import run_control_comparison, run_control_generalization
from .pybullet_compare import run_pybullet_axis_training, run_pybullet_controller_benchmark, run_pybullet_full_experiment
from .pybullet_ladrc_tuning import run_pybullet_ladrc_axis_tuning, run_pybullet_ladrc_full_tuning

__all__ = [
    "run_control_comparison",
    "run_control_generalization",
    "run_pybullet_axis_training",
    "run_pybullet_controller_benchmark",
    "run_pybullet_full_experiment",
    "run_pybullet_ladrc_axis_tuning",
    "run_pybullet_ladrc_full_tuning",
]
