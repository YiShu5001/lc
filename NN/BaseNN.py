# models.py
from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """
    网络结构配置（只放“模型相关”的参数）
    """
    hidden_sizes: Tuple[int, ...] = (64, 64)
    activation: str = "tanh"   # 可扩展：relu/gelu
    # 连续动作常用：
    log_std_init: float = -0.5 # 初始 log_std（连续动作）


class ActionDist(ABC):
    """
    动作分布抽象：
    - 离散动作：Categorical
    - 连续动作：Gaussian (Normal)
    统一提供 sample / log_prob / entropy
    """
    @abstractmethod
    def sample(self) -> torch.Tensor:
        """采样动作"""

    @abstractmethod
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        计算动作对数概率
        Returns: shape = (batch,)
        """

    @abstractmethod
    def entropy(self) -> torch.Tensor:
        """
        计算分布熵
        Returns: shape = (batch,)
        """


@dataclass
class ActOutput:
    """
    模型在采样阶段输出的标准结构（给 Trainer 用）
    """
    actions: torch.Tensor      # shape: (N,) or (N, act_dim)
    logprobs: torch.Tensor     # shape: (N,)
    values: torch.Tensor       # shape: (N,)
    extras: Dict[str, torch.Tensor]  # 可选：比如 mean/std、logits 等


@dataclass
class EvalOutput:
    """
    模型在训练评估阶段输出（给 Algo 用）
    """
    logprobs: torch.Tensor     # shape: (B,)
    entropies: torch.Tensor    # shape: (B,)
    values: torch.Tensor       # shape: (B,)
    extras: Dict[str, torch.Tensor]


class BaseRLModel(nn.Module, ABC):
    """
    强化学习模型基类：规定 Trainer/Algo 需要的核心接口。

    必须实现：
    - forward_dist(obs): 返回动作分布（ActionDist）
    - value(obs): V(s)
    并通过 act/evaluate 提供统一输出。
    """

    @abstractmethod
    def forward_dist(self, obs: torch.Tensor) -> ActionDist:
        """
        输入 obs，输出动作分布对象。
        Args:
            obs: shape = (batch, obs_dim)
        """

    @abstractmethod
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        输出状态价值 V(s)
        Returns:
            shape = (batch,)
        """

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> ActOutput:
        """
        采样接口（给 Trainer 用）
        输入：一批观测 obs
        输出：动作、logprob、value（不建立反向图）
        """
        dist = self.forward_dist(obs)
        actions = dist.sample()
        logprobs = dist.log_prob(actions)
        values = self.value(obs)
        return ActOutput(actions=actions, logprobs=logprobs, values=values, extras={})

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> EvalOutput:
        """
        评估接口（给 Algo 更新用）
        输入：obs + 已执行过的 actions
        输出：新策略下 logprob、entropy、value（建立计算图）
        """
        dist = self.forward_dist(obs)
        logprobs = dist.log_prob(actions)
        entropies = dist.entropy()
        values = self.value(obs)
        return EvalOutput(logprobs=logprobs, entropies=entropies, values=values, extras={})


# 你后续可以实现：
# - DiscreteActorCritic(BaseRLModel)
# - ContinuousActorCritic(BaseRLModel)
# 只要满足 forward_dist/value 两个抽象方法，Trainer/Algo 就能通用。
