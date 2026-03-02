"""
TD3 (Twin Delayed DDPG) 算法实现
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
class TD3Config(AlgoConfig):
    """
    TD3算法配置
    """
    learning_rate: float = 3e-4
    tau: float = 0.005                      # 软更新系数
    gamma: float = 0.99                    # 折扣因子
    policy_delay: int = 2                   # 策略更新延迟（每2次critic更新更新1次actor）
    noise_clip: float = 0.5                 # 目标策略平滑噪声裁剪
    target_noise: float = 0.2               # 目标策略平滑噪声标准差
    target_update_interval: int = 1         # 目标网络更新间隔
    device: str = "cpu"


class TD3Algo(BaseAlgo):
    """
    TD3算法实现
    
    核心思想：
    1. 双Critic网络减少过估计
    2. 延迟策略更新（policy_delay）
    3. 目标策略平滑（target policy smoothing）
    
    注意：TD3是off-policy算法，需要OffPolicyTrainer支持
    """
    
    def __init__(self, cfg: TD3Config):
        """
        Args:
            cfg: TD3配置
        """
        super().__init__(cfg)
        self.cfg: TD3Config = cfg
        self.device = torch.device(cfg.device)
        self.update_counter = 0
    
    def update(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: "TD3Batch"  # 自定义batch类型
    ) -> Dict[str, float]:
        """
        执行TD3更新
        
        Args:
            model: 策略模型（需要支持TD3的特殊接口）
            optimizer: 优化器
            batch: 经验批次
        
        Returns:
            metrics: 包含loss、q1_loss、q2_loss、policy_loss等指标
        """
        states = batch.obs
        actions = batch.actions
        rewards = batch.rewards
        next_states = batch.next_obs
        dones = batch.dones
        
        self.update_counter += 1
        
        # TD3需要特殊的模型接口：
        # - model.q1(obs, action): Q1值
        # - model.q2(obs, action): Q2值
        # - model.q1_target(obs, action): 目标Q1值
        # - model.q2_target(obs, action): 目标Q2值
        # - model.actor(obs): 动作
        # - model.actor_target(obs): 目标动作
        
        # 更新Critic
        with torch.no_grad():
            # 目标策略平滑
            next_actions = model.actor_target(next_states)
            noise = torch.randn_like(next_actions) * self.cfg.target_noise
            noise = torch.clamp(noise, -self.cfg.noise_clip, self.cfg.noise_clip)
            next_actions = torch.clamp(next_actions + noise, -1.0, 1.0)  # 假设动作范围[-1, 1]
            
            # 计算目标Q值（取两个Q的最小值）
            target_q1 = model.q1_target(next_states, next_actions)
            target_q2 = model.q2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # TD目标
            target_q = rewards + (1 - dones) * self.cfg.gamma * target_q
        
        # 更新Q网络
        current_q1 = model.q1(states, actions)
        current_q2 = model.q2(states, actions)
        
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss
        
        # 反向传播更新Critic
        optimizer.zero_grad()
        q_loss.backward()
        optimizer.step()
        
        # 延迟更新策略
        policy_loss = None
        if self.update_counter % self.cfg.policy_delay == 0:
            # 更新Actor
            new_actions = model.actor(states)
            q1_new = model.q1(states, new_actions)
            policy_loss = -q1_new.mean()
            
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
            
            # 软更新目标网络
            self._soft_update_target_networks(model)
        
        metrics = {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "q_loss": q_loss.item(),
        }
        
        if policy_loss is not None:
            metrics["policy_loss"] = policy_loss.item()
        
        return metrics
    
    def _soft_update_target_networks(self, model: nn.Module):
        """软更新目标网络"""
        if hasattr(model, 'update_target_networks'):
            model.update_target_networks(self.cfg.tau)
        else:
            # 手动更新
            for param, target_param in zip(model.actor.parameters(), model.actor_target.parameters()):
                target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)
            
            for param, target_param in zip(model.q1.parameters(), model.q1_target.parameters()):
                target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)
            
            for param, target_param in zip(model.q2.parameters(), model.q2_target.parameters()):
                target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)


# TD3需要特殊的batch类型
@dataclass
class TD3Batch:
    """TD3算法的batch类型"""
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor
