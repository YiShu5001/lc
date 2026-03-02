"""
连续动作空间的Actor-Critic模型
"""
from __future__ import annotations
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from NN.BaseNN import BaseRLModel, ModelConfig, ActionDist
from NN.action_dists import GaussianActionDist


class ContinuousActorCritic(BaseRLModel):
    """
    连续动作空间的Actor-Critic模型
    
    Actor: 输出动作的均值和标准差（高斯分布）
    Critic: 输出状态价值 V(s)
    """
    
    def __init__(
        self,
        cfg: ModelConfig,
        obs_dim: int,
        action_dim: int,
        action_high: Optional[torch.Tensor] = None,
        action_low: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            cfg: 模型配置
            obs_dim: 观测维度
            action_dim: 动作维度
            action_high: 动作上界（如果为None，假设为1.0）
            action_low: 动作下界（如果为None，假设为-1.0）
        """
        super().__init__()
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 动作边界
        if action_high is None:
            self.action_high = torch.ones(action_dim)
        else:
            self.action_high = torch.tensor(action_high) if not isinstance(action_high, torch.Tensor) else action_high
        
        if action_low is None:
            self.action_low = -torch.ones(action_dim)
        else:
            self.action_low = torch.tensor(action_low) if not isinstance(action_low, torch.Tensor) else action_low
        
        # 激活函数
        activation_fn = self._get_activation(cfg.activation)
        
        # Actor网络（输出均值和log_std）
        actor_layers = []
        prev_size = obs_dim
        for hidden_size in cfg.hidden_sizes:
            actor_layers.append(nn.Linear(prev_size, hidden_size))
            actor_layers.append(activation_fn())
            prev_size = hidden_size
        
        # 均值输出层
        self.actor_mean = nn.Linear(prev_size, action_dim)
        # log_std输出层（可学习参数）
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim) + cfg.log_std_init)
        
        self.actor_body = nn.Sequential(*actor_layers)
        
        # Critic网络（输出V值）
        critic_layers = []
        prev_size = obs_dim
        for hidden_size in cfg.hidden_sizes:
            critic_layers.append(nn.Linear(prev_size, hidden_size))
            critic_layers.append(activation_fn())
            prev_size = hidden_size
        
        self.critic_head = nn.Linear(prev_size, 1)
        self.critic_body = nn.Sequential(*critic_layers)
    
    def _get_activation(self, activation_name: str):
        """获取激活函数"""
        activations = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "elu": nn.ELU,
        }
        return activations.get(activation_name.lower(), nn.Tanh)
    
    def forward_dist(self, obs: torch.Tensor) -> ActionDist:
        """
        前向传播，返回动作分布
        
        Args:
            obs: shape = (batch, obs_dim)
        
        Returns:
            ActionDist: 动作分布对象
        """
        # Actor前向传播
        features = self.actor_body(obs)
        mean = self.actor_mean(features)
        
        # 计算标准差（确保为正）
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std.clamp(-20, 2))  # 限制范围避免数值问题
        
        # 创建高斯分布
        dist = GaussianActionDist(mean, std)
        
        return dist
    
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        计算状态价值 V(s)
        
        Args:
            obs: shape = (batch, obs_dim)
        
        Returns:
            values: shape = (batch,)
        """
        features = self.critic_body(obs)
        values = self.critic_head(features).squeeze(-1)
        return values
    
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """
        确定性动作（用于评估）
        
        Args:
            obs: shape = (batch, obs_dim)
        
        Returns:
            actions: shape = (batch, action_dim)
        """
        dist = self.forward_dist(obs)
        # 使用均值作为确定性动作
        actions = dist.mode()
        # 裁剪到动作空间范围
        actions = torch.clamp(actions, self.action_low.to(actions.device), self.action_high.to(actions.device))
        return actions
