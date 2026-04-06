from __future__ import annotations

from lc.control.configs import AxisTuningResult, PyBulletControlExperimentConfig
from lc.control.trainers import PyBulletAxisTrainer


def run_pybullet_ladrc_axis_tuning(
    config: PyBulletControlExperimentConfig | None = None,
    axis: str = "x",
) -> AxisTuningResult:
    cfg = config or PyBulletControlExperimentConfig()
    trainer = PyBulletAxisTrainer(cfg)
    return trainer.tune_single_axis_ladrc(axis)


def run_pybullet_ladrc_full_tuning(
    config: PyBulletControlExperimentConfig | None = None,
) -> dict[str, AxisTuningResult]:
    cfg = config or PyBulletControlExperimentConfig()
    trainer = PyBulletAxisTrainer(cfg)
    return {axis: trainer.tune_single_axis_ladrc(axis) for axis in ("x", "y", "z")}
