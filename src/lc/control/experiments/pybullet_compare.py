from __future__ import annotations

from lc.control.configs import PyBulletControlExperimentConfig
from lc.control.trainers import PyBulletAxisTrainer


def run_pybullet_axis_training(config: PyBulletControlExperimentConfig | None = None, axis: str = "all") -> dict[str, object]:
    cfg = config or PyBulletControlExperimentConfig()
    trainer = PyBulletAxisTrainer(cfg)
    axes = ("x", "y", "z") if axis == "all" else (axis,)
    return {name: trainer.train_axis(name) for name in axes}


def run_pybullet_controller_benchmark(
    config: PyBulletControlExperimentConfig | None = None,
    axis: str = "all",
    controller: str = "all",
) -> dict[str, object]:
    cfg = config or PyBulletControlExperimentConfig()
    trainer = PyBulletAxisTrainer(cfg)
    axes = ("x", "y", "z") if axis == "all" else (axis,)
    variants = [variant.name for variant in cfg.controller_variants] if controller == "all" else [controller]
    return {name: trainer.evaluate_axis(name, controller_variants=variants) for name in axes}


def run_pybullet_full_experiment(config: PyBulletControlExperimentConfig | None = None) -> dict[str, object]:
    cfg = config or PyBulletControlExperimentConfig()
    trainer = PyBulletAxisTrainer(cfg)
    return trainer.run_full_protocol()
