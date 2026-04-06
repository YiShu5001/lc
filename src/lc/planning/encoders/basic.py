from __future__ import annotations

import torch
from torch import nn


class SelfEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.net(tensor)


class ObstacleEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        embedded = torch.relu(self.embedding(tensor))
        return embedded.mean(dim=1)


class NeighborEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        embedded = torch.relu(self.embedding(tensor))
        return embedded.mean(dim=1)

