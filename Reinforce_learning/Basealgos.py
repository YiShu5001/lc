# algos.py
from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class AlgoConfig:
    """
    算法配置基类（PPO/A2C/SAC 等都可继承/扩展）
    """
    learning_rate: float = 3e-4


@dataclass
class RolloutBatch:
    """
    Trainer -> Algo 的标准数据包（已经 flatten 后）
    约定：
      B = num_envs * num_steps
    """
    obs: torch.Tensor            # (B, obs_dim)
    actions: torch.Tensor        # (B,) or (B, act_dim)
    old_logprobs: torch.Tensor   # (B,)
    advantages: torch.Tensor     # (B,)
    returns: torch.Tensor        # (B,)
    old_values: torch.Tensor     # (B,)
    # 可选扩展：比如 masks、timeouts、rnn_states、等等


class BaseAlgo(ABC):
    """
    强化学习算法基类：规定“如何用一批 rollout 数据更新模型”
    """

    def __init__(self, cfg: AlgoConfig):
        self.cfg = cfg

    @abstractmethod
    def update(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: RolloutBatch
    ) -> Dict[str, float]:
        """
        执行一次（或多次 epoch 的）参数更新
        Returns:
            metrics: 用于日志记录的标量字典
        """
