from __future__ import annotations

import numpy as np

from .task_spec import RefLineEpisodeBundle


def adapt_episode_to_tracking_inputs(bundle: RefLineEpisodeBundle) -> dict[str, object]:
    """Convert an episode bundle into tracking-env friendly arrays.

    This adapter keeps the environment interface small: the env only needs axis,
    per-step reference position/velocity, disturbance, and the phase table.
    """

    return {
        "axis": bundle.axis,
        "time": np.asarray(bundle.time, dtype=np.float32),
        "reference_position": np.asarray(bundle.reference_position, dtype=np.float32),
        "reference_velocity": np.asarray(bundle.reference_velocity, dtype=np.float32),
        "disturbance": np.asarray(bundle.disturbance, dtype=np.float32),
        "phase_table": list(bundle.phase_table),
    }

