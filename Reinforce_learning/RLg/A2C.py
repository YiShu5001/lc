"""
A2C (Advantage Actor-Critic) 算法实现
适配BaseAlgo接口
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from Reinforce_learning.Basealgos import BaseAlgo, AlgoConfig, RolloutBatch


@dataclass
class A2CConfig(AlgoConfig):
    """
    A2C算法配置
    """
    learning_rate: float = 3e-4
    value_coef: float = 0.5                # 价值函数损失系数
    entropy_coef: float = 0.01             # 熵正则化系数
    max_grad_norm: float = 0.5             # 梯度裁剪
    n_steps: int = 5                       # n-step returns（通常由Trainer处理）


class A2CAlgo(BaseAlgo):
    """
    A2C算法实现
    
    核心思想：
    1. 使用优势函数（Advantage）而非return
    2. 同时更新Actor和Critic
    3. 支持n-step returns
    """
    
    def __init__(self, cfg: A2CConfig):
        """
        Args:
            cfg: A2C配置
        """
        super().__init__(cfg)
        self.cfg: A2CConfig = cfg
    
    def update(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: RolloutBatch
    ) -> Dict[str, float]:
        """
        执行A2C更新
        
        Args:
            model: 策略模型（BaseRLModel）
            optimizer: 优化器
            batch: 经验批次
        
        Returns:
            metrics: 包含loss、policy_loss、value_loss、entropy等指标
        """
        # 评估当前策略
        eval_output = model.evaluate(batch.obs, batch.actions)
        new_logprobs = eval_output.logprobs
        values = eval_output.values
        entropies = eval_output.entropies
        
        # 优势函数（已经由Trainer计算好）
        advantages = batch.advantages
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 策略损失（使用优势函数）
        policy_loss = -(new_logprobs * advantages_normalized).mean()
        
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
        metrics = {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropies.mean().item(),
        }
        
        return metrics
