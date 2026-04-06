from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class StructuredCriticConfig:
    self_dim: int = 4
    obstacle_dim: int = 3
    neighbor_dim: int = 2
    action_dim: int = 2
    embed_dim: int = 32
    ff_dim: int = 64
    max_obstacles: int = 12
    max_neighbors: int = 7


class StructuredCritic(nn.Module):
    """State-action attention critic aligned with the Chapter 4 actor input format."""

    def __init__(
        self,
        config: StructuredCriticConfig | None = None,
        *,
        self_embedding: nn.Linear | None = None,
        obstacle_embedding: nn.Linear | None = None,
        neighbor_embedding: nn.Linear | None = None,
    ) -> None:
        super().__init__()
        self.config = config or StructuredCriticConfig()
        self.self_embedding = self_embedding if self_embedding is not None else nn.Linear(self.config.self_dim, self.config.embed_dim)
        self.obstacle_embedding = obstacle_embedding if obstacle_embedding is not None else nn.Linear(self.config.obstacle_dim, self.config.embed_dim)
        self.neighbor_embedding = neighbor_embedding if neighbor_embedding is not None else nn.Linear(self.config.neighbor_dim, self.config.embed_dim)
        self.action_embedding = nn.Linear(self.config.action_dim, self.config.embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.config.embed_dim,
            num_heads=1,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(self.config.embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.config.embed_dim, self.config.ff_dim),
            nn.ReLU(),
            nn.Linear(self.config.ff_dim, self.config.embed_dim),
        )
        self.norm2 = nn.LayerNorm(self.config.embed_dim)
        self.q_head = nn.Sequential(
            nn.Linear(self.config.embed_dim * 2, self.config.ff_dim),
            nn.ReLU(),
            nn.Linear(self.config.ff_dim, 1),
        )

    def forward(self, observation: dict[str, torch.Tensor] | torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        structured_obs = self._ensure_structured_input(observation)
        action_token = self.action_embedding(action).unsqueeze(1)
        self_token = self.self_embedding(structured_obs["self_state"]).unsqueeze(1)
        obstacle_tokens = self.obstacle_embedding(structured_obs["obstacles"])
        neighbor_tokens = self.neighbor_embedding(self._select_topk_neighbors(structured_obs["neighbors"]))
        context = torch.cat([self_token, obstacle_tokens, neighbor_tokens], dim=1)
        attended, _ = self.attention(query=action_token, key=context, value=context, need_weights=False)
        hidden = self.norm1(action_token + attended)
        hidden = self.norm2(hidden + self.feed_forward(hidden))
        q_input = torch.cat([hidden.squeeze(1), self_token.squeeze(1)], dim=-1)
        return self.q_head(q_input)

    def _ensure_structured_input(self, observation: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, torch.Tensor]:
        if isinstance(observation, dict):
            return observation
        if not torch.is_tensor(observation):
            raise TypeError("observation must be a dict or a flat tensor")
        return self._split_flat_observation(observation)

    def _split_flat_observation(self, observation: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = observation.size(0)
        self_end = self.config.self_dim
        obstacle_size = self.config.max_obstacles * self.config.obstacle_dim
        obstacle_end = self_end + obstacle_size
        neighbor_size = self.config.max_neighbors * self.config.neighbor_dim
        neighbor_end = obstacle_end + neighbor_size
        return {
            "self_state": observation[:, :self_end],
            "obstacles": observation[:, self_end:obstacle_end].reshape(
                batch_size, self.config.max_obstacles, self.config.obstacle_dim
            ),
            "neighbors": observation[:, obstacle_end:neighbor_end].reshape(
                batch_size, self.config.max_neighbors, self.config.neighbor_dim
            ),
        }

    def _select_topk_neighbors(self, neighbors: torch.Tensor) -> torch.Tensor:
        if neighbors.size(1) <= self.config.max_neighbors:
            return neighbors
        distances = torch.norm(neighbors, dim=-1)
        indices = torch.topk(distances, k=self.config.max_neighbors, dim=1, largest=False).indices
        gather_index = indices.unsqueeze(-1).expand(-1, -1, neighbors.size(-1))
        return torch.gather(neighbors, dim=1, index=gather_index)
