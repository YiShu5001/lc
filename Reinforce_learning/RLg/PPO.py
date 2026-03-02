"""
PPO (Proximal Policy Optimization) 算法实现
支持PPO-Clip版本，适配BaseAlgo接口
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from Reinforce_learning.Basealgos import BaseAlgo, AlgoConfig, RolloutBatch


@dataclass
class PPOConfig(AlgoConfig):
    """
    PPO算法配置
    """
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2              # PPO clip参数
    value_coef: float = 0.5                # 价值函数损失系数
    entropy_coef: float = 0.01              # 熵正则化系数
    max_grad_norm: float = 0.5             # 梯度裁剪
    num_epochs: int = 4                    # 每次rollout后的更新轮数
    batch_size: int = 64                   # 每轮更新的batch大小
    gamma: float = 0.99                    # 折扣因子（通常由Trainer提供）
    gae_lambda: float = 0.95               # GAE lambda（通常由Trainer提供）


class PPOAlgo(BaseAlgo):
    """
    PPO算法实现（PPO-Clip版本）
    
    核心思想：
    1. 使用重要性采样比率限制策略更新幅度
    2. 通过clip机制防止策略更新过大
    3. 支持多epoch更新以提高样本利用率
    """
    
    def __init__(self, cfg: PPOConfig):
        """
        Args:
            cfg: PPO配置
        """
        super().__init__(cfg)
        self.cfg: PPOConfig = cfg
    
    def update(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: RolloutBatch
    ) -> Dict[str, float]:
        """
        执行PPO更新
        
        Args:
            model: 策略模型（BaseRLModel）
            optimizer: 优化器
            batch: 经验批次
        
        Returns:
            metrics: 包含loss、policy_loss、value_loss、entropy等指标
        """
        # 计算重要性采样比率
        with torch.no_grad():
            old_logprobs = batch.old_logprobs
        
        # 评估当前策略
        eval_output = model.evaluate(batch.obs, batch.actions)
        new_logprobs = eval_output.logprobs
        values = eval_output.values
        entropies = eval_output.entropies
        
        # 计算重要性采样比率
        ratio = torch.exp(new_logprobs - old_logprobs)
        
        # PPO-Clip损失
        advantages = batch.advantages
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 未clip的策略损失
        policy_loss_unclipped = ratio * advantages_normalized
        
        # clip后的策略损失
        policy_loss_clipped = torch.clamp(
            ratio,
            1.0 - self.cfg.clip_epsilon,
            1.0 + self.cfg.clip_epsilon
        ) * advantages_normalized
        
        # 取最小值（保守更新）
        policy_loss = -torch.min(policy_loss_unclipped, policy_loss_clipped).mean()
        
        # 价值函数损失
        value_loss = F.mse_loss(values, batch.returns)
        
        # 熵损失（鼓励探索）
        entropy_loss = -entropies.mean()
        
        # 总损失
        total_loss = (
            policy_loss
            + self.cfg.value_coef * value_loss
            + self.cfg.entropy_coef * entropy_loss
        )
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
        
        optimizer.step()
        
        # 计算指标
        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > self.cfg.clip_epsilon).float().mean()
            approx_kl = (old_logprobs - new_logprobs).mean()
        
        metrics = {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropies.mean().item(),
            "clip_fraction": clip_fraction.item(),
            "approx_kl": approx_kl.item(),
        }
        
        return metrics
