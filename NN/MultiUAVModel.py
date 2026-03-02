"""
多无人机协调规划神经网络模型
集成避障分支和协作分支，实现完整的Actor-Critic架构
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from .BaseNN import BaseRLModel, ModelConfig, ActionDist
from .obstacle_branch import ObstacleAvoidanceBranch
from .collaborative_branch import CollaborativeBranch


class SigmoidNormal(ActionDist):
    """
    基于Sigmoid的连续动作分布
    
    将动作映射到[0, 1]区间，适用于归一化的连续动作空间
    """
    
    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        """
        Args:
            mean: 动作均值，shape = (batch_size, action_dim)
            log_std: 对数标准差，shape = (batch_size, action_dim) 或 (batch_size,)
        """
        self.mean = mean
        self.log_std = log_std
        self.std = torch.exp(log_std)
        
        # 创建正态分布
        # 使用tanh映射到[-1, 1]，然后映射到[0, 1]
        self.normal_dist = dist.Normal(mean, self.std)
    
    def sample(self) -> torch.Tensor:
        """采样动作"""
        # 从正态分布采样
        sample = self.normal_dist.sample()
        # 使用sigmoid映射到[0, 1]
        return torch.sigmoid(sample)
    
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        计算动作的对数概率
        
        Args:
            actions: 动作，shape = (batch_size, action_dim)
        
        Returns:
            对数概率，shape = (batch_size,)
        """
        # 将[0, 1]映射回实数域（inverse sigmoid）
        # 使用logit函数：logit(x) = log(x / (1-x))
        # 添加小的epsilon避免数值不稳定
        epsilon = 1e-8
        actions_clamped = torch.clamp(actions, epsilon, 1 - epsilon)
        logits = torch.log(actions_clamped / (1 - actions_clamped))
        
        # 计算log_prob，需要考虑sigmoid变换的雅可比行列式
        log_prob_normal = self.normal_dist.log_prob(logits)
        # sigmoid的导数：sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        # log|det(J)| = log(sigmoid(x) * (1 - sigmoid(x)))
        log_det_jacobian = torch.log(actions_clamped * (1 - actions_clamped) + epsilon)
        
        # 总的对数概率
        log_prob = log_prob_normal.sum(dim=-1) + log_det_jacobian.sum(dim=-1)
        
        return log_prob
    
    def entropy(self) -> torch.Tensor:
        """
        计算分布熵
        
        Returns:
            熵，shape = (batch_size,)
        """
        # 近似计算（简化版本）
        return self.normal_dist.entropy().sum(dim=-1)


@dataclass
class MultiUAVModelConfig(ModelConfig):
    """
    多无人机模型配置
    """
    # 输入维度配置
    self_dim: int = 10
    obstacle_dim: int = 4
    neighbor_dim: int = 6
    
    # 网络结构配置
    embed_dim: int = 128
    num_heads: int = 8
    ff_dim: int = 256
    action_dim: int = 3
    
    # 可选配置
    max_obstacles: Optional[int] = None
    max_neighbors: Optional[int] = None
    
    # 正则化
    dropout: float = 0.1
    activation: str = "relu"
    
    # 价值网络配置
    value_hidden_sizes: Tuple[int, ...] = (128, 64)
    
    # 动作分布配置
    log_std_init: float = -0.5


class MultiUAVModel(BaseRLModel):
    """
    多无人机协调规划模型
    
    集成避障分支和协作分支，实现Actor-Critic架构
    """
    
    def __init__(self, cfg: MultiUAVModelConfig):
        """
        Args:
            cfg: 模型配置
        """
        super().__init__()
        self.cfg = cfg
        
        # 避障动作分支
        self.obstacle_branch = ObstacleAvoidanceBranch(
            self_dim=cfg.self_dim,
            obstacle_dim=cfg.obstacle_dim,
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            ff_dim=cfg.ff_dim,
            action_dim=cfg.action_dim,
            max_obstacles=cfg.max_obstacles,
            dropout=cfg.dropout,
            activation=cfg.activation
        )
        
        # 协作动作分支
        self.collaborative_branch = CollaborativeBranch(
            neighbor_dim=cfg.neighbor_dim,
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            ff_dim=cfg.ff_dim,
            action_dim=cfg.action_dim,
            max_neighbors=cfg.max_neighbors,
            dropout=cfg.dropout,
            activation=cfg.activation
        )
        
        # 价值网络（共享特征提取，独立的价值头）
        # 使用避障分支的特征提取部分
        value_input_dim = cfg.embed_dim * 2  # self + obstacle嵌入
        
        value_layers = []
        prev_dim = value_input_dim
        for hidden_dim in cfg.value_hidden_sizes:
            value_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        value_layers.append(nn.Linear(prev_dim, 1))
        
        self.value_net = nn.Sequential(*value_layers)
        
        # 动作分布的对数标准差（可学习参数）
        self.log_std = nn.Parameter(
            torch.ones(cfg.action_dim) * cfg.log_std_init
        )
    
    def _parse_obs(self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        解析观测数据
        
        Args:
            obs: 观测数据
                - 如果是字典：包含'self_state', 'obstacles', 'neighbors'键
                - 如果是张量：需要按照配置的维度拆分
        
        Returns:
            (self_state, obstacles, neighbors)
        """
        if isinstance(obs, dict):
            return obs['self_state'], obs['obstacles'], obs['neighbors']
        else:
            # 如果obs是单个张量，需要按照维度拆分
            # 这里假设obs是concatenated的：[self_state, obstacles_flattened, neighbors_flattened]
            # 这是一个简化的实现，实际使用时建议使用字典格式
            batch_size = obs.shape[0]
            start_idx = 0
            
            self_state = obs[:, start_idx:start_idx + self.cfg.self_dim]
            start_idx += self.cfg.self_dim
            
            if self.cfg.max_obstacles is not None:
                obstacle_size = self.cfg.max_obstacles * self.cfg.obstacle_dim
                obstacles_flat = obs[:, start_idx:start_idx + obstacle_size]
                obstacles = obstacles_flat.view(batch_size, self.cfg.max_obstacles, self.cfg.obstacle_dim)
                start_idx += obstacle_size
            else:
                # 可变数量障碍物需要特殊处理
                raise ValueError("当max_obstacles为None时，必须使用字典格式的obs")
            
            if self.cfg.max_neighbors is not None:
                neighbor_size = self.cfg.max_neighbors * self.cfg.neighbor_dim
                neighbors_flat = obs[:, start_idx:start_idx + neighbor_size]
                neighbors = neighbors_flat.view(batch_size, self.cfg.max_neighbors, self.cfg.neighbor_dim)
            else:
                # 可变数量邻近无人机需要特殊处理
                raise ValueError("当max_neighbors为None时，必须使用字典格式的obs")
            
            return self_state, obstacles, neighbors
    
    def forward_dist(self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> ActionDist:
        """
        前向传播，返回动作分布
        
        Args:
            obs: 观测数据（字典或张量）
        
        Returns:
            动作分布对象
        """
        self_state, obstacles, neighbors = self._parse_obs(obs)
        
        # 避障分支：返回避障动作和Feed Forward输出
        obstacle_action, obstacle_ff_output = self.obstacle_branch(
            self_state, obstacles
        )
        
        # 协作分支：使用避障分支的Feed Forward输出
        final_action = self.collaborative_branch(
            neighbors, obstacle_ff_output
        )
        
        # 创建动作分布
        # 使用final_action作为均值，log_std作为标准差
        log_std = self.log_std.expand_as(final_action)
        
        return SigmoidNormal(final_action, log_std)
    
    def value(self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        计算状态价值 V(s)
        
        Args:
            obs: 观测数据（字典或张量）
        
        Returns:
            状态价值，shape = (batch_size,)
        """
        self_state, obstacles, neighbors = self._parse_obs(obs)
        
        # 使用避障分支的嵌入层提取特征
        from .embeddings import SelfEmbedding, ObstacleEmbedding
        
        # 创建临时嵌入层（或者重用避障分支的嵌入层）
        # 为了简化，我们直接使用避障分支的嵌入层
        self_emb = self.obstacle_branch.self_embedding(self_state)
        obstacle_emb = self.obstacle_branch.obstacle_embedding(obstacles)
        
        # 合并特征
        value_features = torch.cat([self_emb, obstacle_emb], dim=-1)
        
        # 通过价值网络
        values = self.value_net(value_features).squeeze(-1)
        
        return values
