from __future__ import annotations

import numpy as np


def compute_control_metrics(errors: list[float], controls: list[float]) -> dict[str, float]:
    if not errors:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "iae": 0.0,
            "overshoot": 0.0,
            "settling_time": 0.0,
            "steady_state_error": 0.0,
            "control_energy": 0.0,
            "disturbance_recovery_time": 0.0,
            "control_variation": 0.0,
        }
    errors_arr = np.asarray(errors, dtype=float)
    controls_arr = np.asarray(controls, dtype=float)
    delta = np.diff(controls_arr) if len(controls_arr) > 1 else np.asarray([0.0])
    steady_window = max(5, len(errors_arr) // 10)
    abs_errors = np.abs(errors_arr)
    overshoot = float(np.max(np.maximum(-errors_arr, 0.0)))
    threshold = 0.05
    settling_index = len(abs_errors) - 1
    for index in range(len(abs_errors)):
        if np.all(abs_errors[index:] <= threshold):
            settling_index = index
            break
    disturbance_recovery = settling_index
    return {
        "mae": float(np.mean(np.abs(errors_arr))),
        "rmse": float(np.sqrt(np.mean(errors_arr**2))),
        "iae": float(np.sum(abs_errors)),
        "overshoot": overshoot,
        "settling_time": float(settling_index),
        "steady_state_error": float(np.mean(abs_errors[-steady_window:])),
        "control_energy": float(np.sum(controls_arr**2)),
        "disturbance_recovery_time": float(disturbance_recovery),
        "control_variation": float(np.mean(np.abs(delta))),
    }


def compute_planning_metrics(
    collisions: int,
    occupancy_errors: list[float],
    formation_errors: list[float],
    success: bool,
) -> dict[str, float]:
    return {
        "success_rate": float(success),
        "collision_rate": 1.0 if collisions > 0 else 0.0,
        "occupancy_error": float(np.mean(occupancy_errors)) if occupancy_errors else 0.0,
        "formation_error": float(np.mean(formation_errors)) if formation_errors else 0.0,
    }
