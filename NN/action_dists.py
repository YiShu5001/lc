"""
动作分布实现
支持离散和连续动作空间
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.distributions as dist

from NN.BaseNN import ActionDist


class GaussianActionDist(ActionDist):
    """
    高斯动作分布（用于连续动作空间）
    
    使用重参数化技巧，支持可微分的采样
    """
    
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Args:
            mean: 均值，shape = (batch, action_dim)
            std: 标准差，shape = (batch, action_dim) or (action_dim,) or scalar
        """
        self.mean = mean
        self.std = std
        self.dist = dist.Normal(mean, std)
    
    def sample(self) -> torch.Tensor:
        """采样动作（使用重参数化）"""
        return self.dist.rsample()
    
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        计算动作的对数概率
        
        Args:
            actions: shape = (batch, action_dim)
        
        Returns:
            log_prob: shape = (batch,)
        """
        log_probs = self.dist.log_prob(actions)
        # 对action_dim维度求和
        return log_probs.sum(dim=-1)
    
    def entropy(self) -> torch.Tensor:
        """
        计算分布的熵
        
        Returns:
            entropy: shape = (batch,)
        """
        entropies = self.dist.entropy()
        return entropies.sum(dim=-1)
    
    def mode(self) -> torch.Tensor:
        """返回分布的众数（均值）"""
        return self.mean


class CategoricalActionDist(ActionDist):
    """
    分类动作分布（用于离散动作空间）
    """
    
    def __init__(self, logits: torch.Tensor):
        """
        Args:
            logits: 未归一化的log概率，shape = (batch, num_actions)
        """
        self.logits = logits
        self.dist = dist.Categorical(logits=logits)
    
    def sample(self) -> torch.Tensor:
        """采样动作"""
        return self.dist.sample()
    
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        计算动作的对数概率
        
        Args:
            actions: shape = (batch,)
        
        Returns:
            log_prob: shape = (batch,)
        """
        return self.dist.log_prob(actions)
    
    def entropy(self) -> torch.Tensor:
        """
        计算分布的熵
        
        Returns:
            entropy: shape = (batch,)
        """
        return self.dist.entropy()
    
    def mode(self) -> torch.Tensor:
        """返回分布的众数（概率最大的动作）"""
        return self.dist.probs.argmax(dim=-1)


class TanhNormalActionDist(ActionDist):
    """
    Tanh变换的高斯分布（用于有界连续动作空间）
    
    先采样高斯分布，然后通过tanh映射到[-1, 1]
    常用于SAC等算法
    """
    
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, epsilon: float = 1e-6):
        """
        Args:
            mean: 均值，shape = (batch, action_dim)
            std: 标准差，shape = (batch, action_dim)
            epsilon: 数值稳定性常数
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon
        self.normal_dist = dist.Normal(mean, std)
    
    def sample(self) -> torch.Tensor:
        """采样动作（使用重参数化 + tanh）"""
        normal_sample = self.normal_dist.rsample()
        return torch.tanh(normal_sample)
    
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        计算动作的对数概率（考虑tanh的雅可比行列式）
        
        Args:
            actions: shape = (batch, action_dim)，范围应该在[-1, 1]
        
        Returns:
            log_prob: shape = (batch,)
        """
        # 将actions映射回normal空间
        normal_actions = torch.atanh(torch.clamp(actions, -1 + self.epsilon, 1 - self.epsilon))
        
        # 计算normal分布的对数概率
        log_probs = self.normal_dist.log_prob(normal_actions)
        
        # 减去tanh的雅可比行列式的对数
        # log(1 - tanh^2(x)) = log(1 - action^2)
        log_probs -= torch.log(1 - actions.pow(2) + self.epsilon)
        
        return log_probs.sum(dim=-1)
    
    def entropy(self) -> torch.Tensor:
        """
        计算分布的熵（近似）
        
        Returns:
            entropy: shape = (batch,)
        """
        # 使用normal分布的熵作为近似
        entropies = self.normal_dist.entropy()
        return entropies.sum(dim=-1)
    
    def mode(self) -> torch.Tensor:
        """返回分布的众数（tanh(mean)）"""
        return torch.tanh(self.mean)
