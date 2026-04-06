from __future__ import annotations

import numpy as np


def map_planning_to_control_reference(final_action: np.ndarray) -> float:
    action = np.asarray(final_action, dtype=float).reshape(-1)
    if action.size == 0:
        return 0.0
    return float(np.clip(0.5 * action[0] + 0.25 * action[min(1, action.size - 1)], -1.0, 1.0))

