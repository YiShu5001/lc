from __future__ import annotations

import numpy as np

from lc.control.configs import PyBulletControlExperimentConfig
from lc.control.controllers import ControllerBundle
from lc.control.reference_generators.piecewise_velocity import ReferenceBundle
from lc.control.simulators import create_ctrl_aviary, run_evaluation_episode
from lc.envs.metrics import compute_control_metrics


def run_controller_episode(
    config: PyBulletControlExperimentConfig,
    controller: ControllerBundle,
    reference_bundle: ReferenceBundle,
) -> dict[str, object]:
    backend = create_ctrl_aviary(config)
    try:
        artifacts = run_evaluation_episode(backend, controller, reference_bundle, axis=reference_bundle.axis, config=config)
    finally:
        if hasattr(backend.get("env"), "close"):
            backend["env"].close()
            backend["env"] = None
    metrics = compute_episode_metrics(artifacts.timeseries, reference_bundle.axis)
    metrics["backend"] = artifacts.backend
    return {
        "metrics": metrics,
        "timeseries": artifacts.timeseries,
        "legacy_rows": artifacts.logger_rows,
        "backend": artifacts.backend,
    }


def collect_episode_timeseries(result: dict[str, object]) -> list[dict[str, float]]:
    return list(result["timeseries"])


def compute_episode_metrics(timeseries: list[dict[str, float]], axis: str) -> dict[str, float]:
    pos_key = axis
    target_key = f"target_{axis}"
    controls = [float(np.mean([row["rpm0"], row["rpm1"], row["rpm2"], row["rpm3"]])) for row in timeseries]
    errors = [float(row[target_key] - row[pos_key]) for row in timeseries]
    metrics = compute_control_metrics(errors, controls)
    velocity_key = f"v{axis}"
    target_velocity_key = f"target_v{axis}"
    metrics["velocity_rmse"] = float(
        np.sqrt(np.mean([(row[target_velocity_key] - row[velocity_key]) ** 2 for row in timeseries]))
    )
    metrics["reward"] = float(np.mean([row["reward"] for row in timeseries])) if timeseries else 0.0
    return metrics
