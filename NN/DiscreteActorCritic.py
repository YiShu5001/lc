"""
离散动作空间的Actor-Critic模型
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from NN.BaseNN import BaseRLModel, ModelConfig, ActionDist
from NN.action_dists import CategoricalActionDist


class DiscreteActorCritic(BaseRLModel):
    """
    离散动作空间的Actor-Critic模型
    
    Actor: 输出动作的logits（分类分布）
    Critic: 输出状态价值 V(s)
    """
    
    def __init__(
        self,
        cfg: ModelConfig,
        obs_dim: int,
        action_dim: int,
    ):
        """
        Args:
            cfg: 模型配置
            obs_dim: 观测维度
            action_dim: 动作数量（离散动作空间的动作个数）
        """
        super().__init__()
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 激活函数
        activation_fn = self._get_activation(cfg.activation)
        
        # Actor网络（输出logits）
        actor_layers = []
        prev_size = obs_dim
        for hidden_size in cfg.hidden_sizes:
            actor_layers.append(nn.Linear(prev_size, hidden_size))
            actor_layers.append(activation_fn())
            prev_size = hidden_size
        
        # Logits输出层
        self.actor_head = nn.Linear(prev_size, action_dim)
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
        return activations.get(activation_name.lower(), nn.ReLU)
    
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
        logits = self.actor_head(features)
        
        # 创建分类分布
        dist = CategoricalActionDist(logits)
        
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
            actions: shape = (batch,)，动作索引
        """
        dist = self.forward_dist(obs)
        # 使用众数（概率最大的动作）作为确定性动作
        return dist.mode()
