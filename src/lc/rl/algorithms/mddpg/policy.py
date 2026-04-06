from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass(frozen=True)
class MDDPGConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 128
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    gamma: float = 0.95
    tau: float = 0.02
    batch_size: int = 32
    buffer_size: int = 50000
    action_hold_steps: int = 5
    stack_size: int = 4
    expl_noise: float = 0.1
    device: str = "cpu"


class _Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class _Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class _ReplayBuffer:
    def __init__(self, capacity: int):
        self.rows: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.rows.append((state.copy(), action.copy(), float(reward), next_state.copy(), float(done)))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.rows, k=min(batch_size, len(self.rows)))
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            np.asarray(reward, dtype=np.float32).reshape(-1, 1),
            np.asarray(next_state, dtype=np.float32),
            np.asarray(done, dtype=np.float32).reshape(-1, 1),
        )

    def __len__(self) -> int:
        return len(self.rows)


class MDDPGPolicy:
    """第三章控制层使用的 DDPG/mDDPG 共用策略骨架。

    基础 DDPG 通过配置关闭状态堆叠、动作保持和 N 步回报；
    增强版 mDDPG 则打开这些机制。
    """

    def __init__(self, config: MDDPGConfig):
        self.config = config
        self.device = torch.device(config.device)
        effective_state_dim = config.state_dim * config.stack_size
        self.actor = _Actor(effective_state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.actor_target = _Actor(effective_state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = _Critic(effective_state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.critic_target = _Critic(effective_state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.replay = _ReplayBuffer(config.buffer_size)
        self._hold_counter = 0
        self._last_action = np.zeros(config.action_dim, dtype=np.float32)
        self._normalizer = np.ones(effective_state_dim, dtype=np.float32)

    def reset(self) -> None:
        self._hold_counter = 0
        self._last_action = np.zeros(self.config.action_dim, dtype=np.float32)

    def select_action(self, stacked_state: np.ndarray, explore: bool = False) -> np.ndarray:
        if self._hold_counter > 0:
            self._hold_counter -= 1
            return self._last_action.copy()
        self._update_normalizer(stacked_state)
        norm_state = stacked_state / self._normalizer
        state_t = torch.as_tensor(norm_state, dtype=torch.float32, device=self.device).view(1, -1)
        action = self.actor(state_t).detach().cpu().numpy().reshape(-1)
        if explore:
            action = np.clip(action + np.random.normal(0.0, self.config.expl_noise, size=action.shape), -1.0, 1.0)
        self._last_action = action.astype(np.float32)
        self._hold_counter = self.config.action_hold_steps - 1
        return self._last_action.copy()

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        self._update_normalizer(state)
        self._update_normalizer(next_state)
        self.replay.push(state / self._normalizer, action, reward, next_state / self._normalizer, done)

    def update(self, updates: int = 1) -> dict[str, float]:
        """执行一次或多次 actor-critic 参数更新。"""
        if len(self.replay) < self.config.batch_size:
            return {"critic_loss": 0.0, "actor_loss": 0.0}
        critic_losses = []
        actor_losses = []
        for _ in range(updates):
            state, action, reward, next_state, done = self.replay.sample(self.config.batch_size)
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
            reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            next_state_t = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
            done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                next_action = self.actor_target(next_state_t)
                target_q = reward_t + (1.0 - done_t) * self.config.gamma * self.critic_target(next_state_t, next_action)
            current_q = self.critic(state_t, action_t)
            critic_loss = nn.functional.mse_loss(current_q, target_q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = -self.critic(state_t, self.actor(state_t)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)
            critic_losses.append(float(critic_loss.item()))
            actor_losses.append(float(actor_loss.item()))
        return {"critic_loss": float(np.mean(critic_losses)), "actor_loss": float(np.mean(actor_losses))}

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.config.tau * source_param.data + (1.0 - self.config.tau) * target_param.data)

    def _update_normalizer(self, state: np.ndarray) -> None:
        self._normalizer = np.maximum(self._normalizer, np.abs(state).astype(np.float32))
