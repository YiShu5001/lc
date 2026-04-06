from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lc.planning import Chapter4ExperimentConfig, run_chapter4_planning_forward


def main() -> None:
    config = Chapter4ExperimentConfig()
    self_state = torch.zeros((2, config.self_state_dim), dtype=torch.float32)
    obstacle_states = torch.zeros((2, 4, config.obstacle_state_dim), dtype=torch.float32)
    neighbor_states = torch.zeros((2, 3, config.neighbor_state_dim), dtype=torch.float32)
    avoid_action, final_action = run_chapter4_planning_forward(self_state, obstacle_states, neighbor_states, config)
    print("chapter4_avoid_shape", tuple(avoid_action.shape))
    print("chapter4_final_shape", tuple(final_action.shape))


if __name__ == "__main__":
    main()
