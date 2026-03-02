"""
SAC (Soft Actor-Critic) 算法实现
SAC-v2版本，支持自动温度调整
适配BaseAlgo接口（需要OffPolicyTrainer支持）
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from Reinforce_learning.Basealgos import BaseAlgo, AlgoConfig


@dataclass
class SACConfig(AlgoConfig):
    """
    SAC算法配置
    """
    learning_rate: float = 3e-4
    tau: float = 0.005                      # 软更新系数
    alpha: Optional[float] = None           # 温度参数（None表示自动调整）
    target_entropy: Optional[float] = None   # 目标熵（用于自动调整alpha）
    alpha_lr: float = 3e-4                  # alpha的学习率
    gamma: float = 0.99                    # 折扣因子
    target_update_interval: int = 1         # 目标网络更新间隔
    device: str = "cpu"


class SACAlgo(BaseAlgo):
    """
    SAC算法实现
    
    核心思想：
    1. 最大熵强化学习，平衡探索和利用
    2. 双Q网络减少过估计
    3. 自动调整温度参数alpha
    4. 软更新目标网络
    
    注意：SAC是off-policy算法，需要OffPolicyTrainer支持
    """
    
    def __init__(self, cfg: SACConfig):
        """
        Args:
            cfg: SAC配置
        """
        super().__init__(cfg)
        self.cfg: SACConfig = cfg
        self.device = torch.device(cfg.device)
        
        # 自动调整alpha
        self.automatic_entropy_tuning = cfg.alpha is None
        if self.automatic_entropy_tuning:
            if cfg.target_entropy is None:
                # 默认目标熵为-action_dim（启发式）
                # 这里需要从环境获取，暂时设为-1
                self.target_entropy = -1.0
            else:
                self.target_entropy = cfg.target_entropy
            
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        else:
            self.alpha = cfg.alpha
    
    def get_alpha(self) -> float:
        """获取当前alpha值"""
        if self.automatic_entropy_tuning:
            return self.log_alpha.exp().item()
        return self.alpha
    
    def update(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: "SACBatch"  # 自定义batch类型，包含next_obs和dones
    ) -> Dict[str, float]:
        """
        执行SAC更新
        
        Args:
            model: 策略模型（需要支持SAC的特殊接口）
            optimizer: 优化器
            batch: 经验批次（需要包含next_obs）
        
        Returns:
            metrics: 包含loss、policy_loss、q1_loss、q2_loss、alpha等指标
        """
        # SAC需要特殊的模型接口：
        # - model.q1(obs, action): Q1值
        # - model.q2(obs, action): Q2值
        # - model.q1_target(obs, action): 目标Q1值
        # - model.q2_target(obs, action): 目标Q2值
        # - model.actor(obs): 动作分布
        
        states = batch.obs
        actions = batch.actions
        rewards = batch.rewards
        next_states = batch.next_obs
        dones = batch.dones
        
        # 获取当前alpha
        alpha = self.get_alpha()
        
        # 计算目标Q值
        with torch.no_grad():
            # 从目标策略采样下一个动作
            next_dist = model.forward_dist(next_states)
            next_actions = next_dist.sample()
            next_logprobs = next_dist.log_prob(next_actions)
            
            # 计算目标Q值（取两个Q的最小值）
            target_q1 = model.q1_target(next_states, next_actions)
            target_q2 = model.q2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # SAC目标：Q_target = r + gamma * (Q - alpha * log_prob)
            target_q = rewards + (1 - dones) * self.cfg.gamma * (
                target_q - alpha * next_logprobs.unsqueeze(1)
            )
        
        # 更新Q网络
        current_q1 = model.q1(states, actions)
        current_q2 = model.q2(states, actions)
        
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss
        
        # 更新策略网络
        dist = model.forward_dist(states)
        new_actions = dist.sample()
        new_logprobs = dist.log_prob(new_actions)
        
        q1_new = model.q1(states, new_actions)
        q2_new = model.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (alpha * new_logprobs.unsqueeze(1) - q_new).mean()
        
        # 总损失
        total_loss = q_loss + policy_loss
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 更新alpha（如果自动调整）
        alpha_loss = None
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (new_logprobs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # 软更新目标网络
        self._soft_update_target_networks(model)
        
        metrics = {
            "loss": total_loss.item(),
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": self.get_alpha(),
        }
        
        if alpha_loss is not None:
            metrics["alpha_loss"] = alpha_loss.item()
        
        return metrics
    
    def _soft_update_target_networks(self, model: nn.Module):
        """软更新目标网络"""
        # 这里需要模型提供target网络的更新方法
        # 或者直接在这里更新
        if hasattr(model, 'update_target_networks'):
            model.update_target_networks(self.cfg.tau)
        else:
            # 手动更新（如果模型有q1_target和q2_target属性）
            for param, target_param in zip(model.q1.parameters(), model.q1_target.parameters()):
                target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)
            
            for param, target_param in zip(model.q2.parameters(), model.q2_target.parameters()):
                target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)


# SAC需要特殊的batch类型
@dataclass
class SACBatch:
    """SAC算法的batch类型"""
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor
