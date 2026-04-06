from __future__ import annotations

import torch
from torch import nn


class SingleStreamMLPPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, action_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, flat_observation: torch.Tensor) -> torch.Tensor:
        return self.net(flat_observation)

