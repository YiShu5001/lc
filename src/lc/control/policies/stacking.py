from __future__ import annotations

import numpy as np


def stack_state(history: list[np.ndarray], new_obs: np.ndarray, stack_size: int) -> np.ndarray:
    history.append(new_obs)
    while len(history) < stack_size:
        history.insert(0, history[0].copy())
    history[:] = history[-stack_size:]
    return np.concatenate(history, axis=0)

