from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MultiUAVModelConfig:
    self_dim: int = 4
    obstacle_dim: int = 3
    neighbor_dim: int = 2
    embed_dim: int = 32
    num_heads: int = 4
    ff_dim: int = 64
    action_dim: int = 2
    max_obstacles: int = 12
    max_neighbors: int = 7
    dropout: float = 0.0
    action_activation: str = "tanh"

    @property
    def hidden_dim(self) -> int:
        return self.embed_dim


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attended, weights = self.attention(query=query, key=key_value, value=key_value, need_weights=True)
        hidden = self.norm1(query + attended)
        ff_hidden = self.feed_forward(hidden)
        return self.norm2(hidden + ff_hidden), weights


class ActionHead(nn.Module):
    def __init__(self, input_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class AvoidanceBackbone(nn.Module):
    def __init__(self, config: MultiUAVModelConfig) -> None:
        super().__init__()
        self.self_embedding = nn.Linear(config.self_dim, config.embed_dim)
        self.obstacle_embedding = nn.Linear(config.obstacle_dim, config.embed_dim)
        self.block = TransformerBlock(config.embed_dim, config.num_heads, config.ff_dim, config.dropout)
        self.action_head = ActionHead(config.embed_dim * 2, config.action_dim)

    def forward(self, self_state: torch.Tensor, obstacles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self_token = self.self_embedding(self_state).unsqueeze(1)
        obstacle_tokens = self.obstacle_embedding(obstacles)
        tokens = torch.cat([self_token, obstacle_tokens], dim=1)
        encoded, attention = self.block(self_token, tokens)
        safety_feature = encoded.squeeze(1)
        avoid_action = self.action_head(torch.cat([self.self_embedding(self_state), safety_feature], dim=-1))
        return avoid_action, safety_feature, attention


class CollaborativeBackbone(nn.Module):
    def __init__(self, config: MultiUAVModelConfig) -> None:
        super().__init__()
        self.neighbor_embedding = nn.Linear(config.neighbor_dim, config.embed_dim)
        self.safe_projection = nn.Linear(config.embed_dim, config.embed_dim)
        self.block = TransformerBlock(config.embed_dim, config.num_heads, config.ff_dim, config.dropout)
        self.residual_head = ActionHead(config.embed_dim * 2, config.action_dim)
        self.gate_head = nn.Sequential(
            nn.Linear(config.embed_dim * 2 + 1, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        neighbors: torch.Tensor,
        safe_feature: torch.Tensor,
        safe_action: torch.Tensor,
        risk_indicator: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        safe_token = self.safe_projection(safe_feature).unsqueeze(1)
        neighbor_tokens = self.neighbor_embedding(neighbors)
        tokens = torch.cat([safe_token, neighbor_tokens], dim=1)
        encoded, attention = self.block(safe_token, tokens)
        coop_feature = encoded.squeeze(1)
        residual = self.residual_head(torch.cat([safe_feature, coop_feature], dim=-1))
        gate = self.gate_head(torch.cat([safe_feature, coop_feature, risk_indicator], dim=-1))
        final_action = torch.clamp(safe_action + gate * residual, min=-1.0, max=1.0)
        return final_action, residual, gate, attention


class MultiUAVModel(nn.Module):
    """Safety-first collaborative actor with residual correction and dynamic gating."""

    def __init__(self, config: MultiUAVModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or MultiUAVModelConfig()
        self.avoidance_backbone = AvoidanceBackbone(self.config)
        self.collaborative_branch = CollaborativeBackbone(self.config)
        self.last_attention: dict[str, torch.Tensor] = {}
        self.last_gate: torch.Tensor | None = None

    def forward(self, observation: dict[str, torch.Tensor] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        avoid_action, final_action, _ = self.policy_stages(observation)
        return avoid_action, final_action

    def policy_stages(
        self,
        observation: dict[str, torch.Tensor] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        structured_obs = self._ensure_structured_input(observation)
        avoid_action, safe_feature, avoid_attention = self.avoidance_backbone(
            structured_obs["self_state"],
            structured_obs["obstacles"],
        )
        risk_indicator = self._estimate_risk_indicator(structured_obs).unsqueeze(-1)
        final_action, coop_residual, gate, collaborative_attention = self.collaborative_branch(
            structured_obs["neighbors"],
            safe_feature,
            avoid_action,
            risk_indicator,
        )
        self.last_attention = {
            "avoidance": avoid_attention.detach(),
            "collaboration": collaborative_attention.detach(),
            "cooperation_residual": coop_residual.detach(),
        }
        self.last_gate = gate.detach()
        return avoid_action, final_action, safe_feature

    def shared_embeddings(self) -> dict[str, nn.Linear]:
        return {
            "self_embedding": self.avoidance_backbone.self_embedding,
            "obstacle_embedding": self.avoidance_backbone.obstacle_embedding,
            "neighbor_embedding": self.collaborative_branch.neighbor_embedding,
        }

    def _estimate_risk_indicator(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        obstacles = observation["obstacles"]
        neighbors = observation["neighbors"]
        obstacle_distance = torch.norm(obstacles[..., :2], dim=-1)
        obstacle_radius = obstacles[..., 2]
        obstacle_risk = torch.clamp(1.0 / (1.0 + obstacle_distance) + obstacle_radius, 0.0, 2.0).mean(dim=1)
        neighbor_distance = torch.norm(neighbors, dim=-1)
        neighbor_risk = torch.clamp(1.0 / (1.0 + neighbor_distance), 0.0, 1.0).mean(dim=1)
        return torch.clamp(0.7 * obstacle_risk + 0.3 * neighbor_risk, 0.0, 1.0)

    def _ensure_structured_input(
        self,
        observation: dict[str, torch.Tensor] | torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if isinstance(observation, dict):
            return observation
        if not torch.is_tensor(observation):
            raise TypeError("observation must be a dict of tensors or a flat tensor")
        return self._split_flat_observation(observation)

    def _split_flat_observation(self, observation: torch.Tensor) -> dict[str, torch.Tensor]:
        if observation.dim() != 2:
            raise ValueError("flat observation must have shape [batch, features]")
        batch_size = observation.size(0)
        self_end = self.config.self_dim
        obstacle_size = self.config.max_obstacles * self.config.obstacle_dim
        obstacle_end = self_end + obstacle_size
        neighbor_size = self.config.max_neighbors * self.config.neighbor_dim
        neighbor_end = obstacle_end + neighbor_size
        if observation.size(1) != neighbor_end:
            raise ValueError(f"expected flat observation dim {neighbor_end}, got {observation.size(1)}")
        return {
            "self_state": observation[:, :self_end],
            "obstacles": observation[:, self_end:obstacle_end].reshape(
                batch_size, self.config.max_obstacles, self.config.obstacle_dim
            ),
            "neighbors": observation[:, obstacle_end:neighbor_end].reshape(
                batch_size, self.config.max_neighbors, self.config.neighbor_dim
            ),
        }
