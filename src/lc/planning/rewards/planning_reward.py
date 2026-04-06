from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PlanningRewardBreakdown:
    total_reward: float
    target_reward: float
    avoidance_reward: float
    collaboration_reward: float
    recovery_reward: float
    smoothness_penalty: float
    consistency_penalty: float
    success_bonus: float

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def compute_planning_reward(
    *,
    stage_name: str,
    occupancy_error: float,
    previous_occupancy_error: float,
    formation_error: float,
    angle_error: float,
    obstacle_margin: float,
    neighbor_margin: float,
    collision: bool,
    action: np.ndarray | None = None,
    previous_action: np.ndarray | None = None,
    safe_action: np.ndarray | None = None,
    success: bool = False,
) -> PlanningRewardBreakdown:
    action_arr = np.asarray(action if action is not None else np.zeros(2, dtype=float), dtype=float)
    previous_action_arr = np.asarray(
        previous_action if previous_action is not None else np.zeros_like(action_arr),
        dtype=float,
    )
    safe_action_arr = np.asarray(safe_action if safe_action is not None else action_arr, dtype=float)
    delta = action_arr - previous_action_arr
    progress = previous_occupancy_error - occupancy_error
    reach_reward = (
        1.4 * max(0.0, progress)
        - 0.45 * max(0.0, -progress)
        + 0.6 * max(0.0, 1.0 - occupancy_error)
    )
    obstacle_term = min(max(obstacle_margin, -0.4), 0.6)
    neighbor_term = min(max(neighbor_margin, -0.3), 0.6)
    avoid_reward = (
        0.9 * obstacle_term
        + 0.7 * neighbor_term
        + 0.3 * max(0.0, progress)
        - (2.5 if collision else 0.0)
    )
    formation_alignment = max(0.0, 1.0 - formation_error)
    angular_uniformity = max(0.0, 1.0 - angle_error)
    collaboration_reward = (
        0.35 * max(0.0, progress)
        + 0.85 * formation_alignment
        + 0.75 * angular_uniformity
    )
    recovery_reward = (
        0.55 * max(0.0, progress)
        + 0.25 * max(0.0, obstacle_margin)
        + 0.2 * max(0.0, neighbor_margin)
        + (0.4 if success else 0.0)
    )
    smoothness_penalty = 0.1 * float(np.linalg.norm(delta, ord=2))
    safe_norm = np.linalg.norm(safe_action_arr)
    action_norm = np.linalg.norm(action_arr)
    if safe_norm <= 1e-6 or action_norm <= 1e-6:
        consistency_penalty = 0.0
    else:
        cosine = float(np.dot(safe_action_arr, action_arr) / (safe_norm * action_norm + 1e-6))
        consistency_penalty = 0.22 * max(0.0, 0.25 - cosine)
    success_bonus = _success_bonus(stage_name) if success else 0.0
    weights = _stage_weights(stage_name)
    total_reward = (
        weights["target"] * reach_reward
        + weights["avoidance"] * avoid_reward
        + weights["collaboration"] * collaboration_reward
        + weights["recovery"] * recovery_reward
        + weights["success"] * success_bonus
        - weights["smoothness"] * smoothness_penalty
        - weights["consistency"] * consistency_penalty
    )
    return PlanningRewardBreakdown(
        total_reward=float(total_reward),
        target_reward=float(reach_reward),
        avoidance_reward=float(avoid_reward),
        collaboration_reward=float(collaboration_reward),
        recovery_reward=float(recovery_reward),
        smoothness_penalty=float(smoothness_penalty),
        consistency_penalty=float(consistency_penalty),
        success_bonus=float(success_bonus),
    )


def _stage_weights(stage_name: str) -> dict[str, float]:
    table = {
        "guidance": {
            "target": 0.72,
            "avoidance": 0.0,
            "collaboration": 0.0,
            "recovery": 0.12,
            "smoothness": 0.08,
            "consistency": 0.04,
            "success": 0.22,
        },
        "avoidance": {
            "target": 0.28,
            "avoidance": 0.47,
            "collaboration": 0.0,
            "recovery": 0.15,
            "smoothness": 0.08,
            "consistency": 0.07,
            "success": 0.22,
        },
        "cooperation": {
            "target": 0.18,
            "avoidance": 0.22,
            "collaboration": 0.36,
            "recovery": 0.16,
            "smoothness": 0.06,
            "consistency": 0.08,
            "success": 0.24,
        },
    }
    return table[stage_name]


def _success_bonus(stage_name: str) -> float:
    return {"guidance": 1.0, "avoidance": 1.5, "cooperation": 2.0}[stage_name]
