"""
DDPG (Deep Deterministic Policy Gradient) 算法重构版本
适配BaseAlgo接口（需要OffPolicyTrainer支持）
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from Reinforce_learning.Basealgos import BaseAlgo, AlgoConfig


@dataclass
class DDPGConfig(AlgoConfig):
    """
    DDPG算法配置
    """
    learning_rate: float = 3e-4
    actor_lr: float = 1e-4                 # Actor学习率
    critic_lr: float = 1e-3                 # Critic学习率
    tau: float = 0.005                     # 软更新系数
    gamma: float = 0.99                    # 折扣因子
    noise_std: float = 0.1                 # 动作噪声标准差（用于探索）
    device: str = "cpu"


@dataclass
class DDPGBatch:
    """DDPG算法的batch类型（off-policy需要next_obs）"""
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class DDPGAlgo(BaseAlgo):
    """
    DDPG算法实现（重构版本）
    
    核心思想：
    1. Actor-Critic架构，适用于连续动作空间
    2. 使用目标网络稳定训练
    3. 软更新目标网络
    4. 使用噪声探索
    
    注意：DDPG是off-policy算法，需要OffPolicyTrainer支持
    """
    
    def __init__(self, cfg: DDPGConfig):
        """
        Args:
            cfg: DDPG配置
        """
        super().__init__(cfg)
        self.cfg: DDPGConfig = cfg
        self.device = torch.device(cfg.device)
    
    def update(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: DDPGBatch
    ) -> Dict[str, float]:
        """
        执行DDPG更新
        
        Args:
            model: 策略模型（需要支持DDPG的特殊接口）
            optimizer: 优化器（需要分别处理actor和critic）
            batch: 经验批次
        
        Returns:
            metrics: 包含loss、critic_loss、actor_loss等指标
        
        注意：DDPG需要特殊的模型接口：
        - model.actor(obs): 动作
        - model.actor_target(obs): 目标动作
        - model.critic(obs, action): Q值
        - model.critic_target(obs, action): 目标Q值
        """
        states = batch.obs
        actions = batch.actions
        rewards = batch.rewards
        next_states = batch.next_obs
        dones = batch.dones
        
        # 更新Critic
        with torch.no_grad():
            # 目标动作（从目标Actor网络）
            target_actions = model.actor_target(next_states)
            # 目标Q值（从目标Critic网络）
            target_q = model.critic_target(next_states, target_actions)
            # TD目标
            target_q = rewards + (1 - dones) * self.cfg.gamma * target_q
        
        # 当前Q值
        current_q = model.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        # 更新Critic
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        
        # 更新Actor
        new_actions = model.actor(states)
        actor_loss = -model.critic(states, new_actions).mean()
        
        optimizer.zero_grad()
        actor_loss.backward()
        optimizer.step()
        
        # 软更新目标网络
        self._soft_update_target_networks(model)
        
        metrics = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "total_loss": (critic_loss + actor_loss).item(),
        }
        
        return metrics
    
    def _soft_update_target_networks(self, model: nn.Module):
        """软更新目标网络"""
        if hasattr(model, 'update_target_networks'):
            model.update_target_networks(self.cfg.tau)
        else:
            # 手动更新（如果模型有actor_target和critic_target属性）
            for param, target_param in zip(model.actor.parameters(), model.actor_target.parameters()):
                target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)
            
            for param, target_param in zip(model.critic.parameters(), model.critic_target.parameters()):
                target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)
