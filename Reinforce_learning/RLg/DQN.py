"""
DQN系列算法实现
支持Vanilla DQN、Double DQN、Dueling DQN、Prioritized DQN
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
class DQNConfig(AlgoConfig):
    """
    DQN算法配置
    """
    learning_rate: float = 1e-3
    gamma: float = 0.99                    # 折扣因子
    epsilon_start: float = 1.0             # 初始探索率
    epsilon_end: float = 0.01              # 最终探索率
    epsilon_decay_steps: int = 10000       # 探索率衰减步数
    target_update_freq: int = 100          # 目标网络更新频率
    double_dqn: bool = True                # 是否使用Double DQN
    dueling: bool = False                  # 是否使用Dueling DQN
    prioritized: bool = False              # 是否使用优先经验回放
    device: str = "cpu"


@dataclass
class DQNBatch:
    """DQN算法的batch类型（off-policy需要next_obs）"""
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor
    weights: Optional[torch.Tensor] = None  # 优先回放的权重
    indices: Optional[torch.Tensor] = None  # 优先回放的索引


class DQNAlgo(BaseAlgo):
    """
    DQN算法实现
    
    支持变体：
    1. Vanilla DQN: 标准DQN
    2. Double DQN: 使用主网络选择动作，目标网络评估
    3. Dueling DQN: 分离状态价值和优势函数
    4. Prioritized DQN: 优先经验回放（需要外部Buffer支持）
    
    注意：DQN是off-policy算法，需要OffPolicyTrainer支持
    """
    
    def __init__(self, cfg: DQNConfig):
        """
        Args:
            cfg: DQN配置
        """
        super().__init__(cfg)
        self.cfg: DQNConfig = cfg
        self.device = torch.device(cfg.device)
        self.update_counter = 0
    
    def update(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: DQNBatch
    ) -> Dict[str, float]:
        """
        执行DQN更新
        
        Args:
            model: Q网络模型（需要支持DQN的特殊接口）
            optimizer: 优化器
            batch: 经验批次
        
        Returns:
            metrics: 包含loss、q_value等指标
        
        注意：DQN需要特殊的模型接口：
        - model.q_network(obs): Q值（主网络）
        - model.q_target(obs): Q值（目标网络）
        - 如果使用Dueling DQN，需要model.value和model.advantage
        """
        states = batch.obs
        actions = batch.actions.long()  # 离散动作
        rewards = batch.rewards
        next_states = batch.next_obs
        dones = batch.dones
        
        self.update_counter += 1
        
        # 当前Q值
        q_values = model.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            if self.cfg.double_dqn:
                # Double DQN: 使用主网络选择动作，目标网络评估
                next_actions = model.q_network(next_states).argmax(1, keepdim=True)
                next_q_values = model.q_target(next_states)
                next_q_value = next_q_values.gather(1, next_actions).squeeze(1)
            else:
                # Vanilla DQN: 直接使用目标网络的最大Q值
                next_q_values = model.q_target(next_states)
                next_q_value = next_q_values.max(1)[0]
            
            # TD目标
            target_q = rewards + (1 - dones) * self.cfg.gamma * next_q_value
        
        # 计算损失
        if self.cfg.prioritized and batch.weights is not None:
            # 优先经验回放：使用重要性采样权重
            td_error = q_value - target_q
            loss = (batch.weights * (td_error ** 2)).mean()
        else:
            # 标准MSE损失
            loss = F.mse_loss(q_value, target_q)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪（可选）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        # 更新目标网络
        if self.update_counter % self.cfg.target_update_freq == 0:
            self._update_target_network(model)
        
        # 计算TD误差（用于优先回放）
        with torch.no_grad():
            td_errors = torch.abs(q_value - target_q).cpu().numpy()
        
        metrics = {
            "loss": loss.item(),
            "q_value": q_value.mean().item(),
            "target_q": target_q.mean().item(),
        }
        
        return metrics
    
    def _update_target_network(self, model: nn.Module):
        """更新目标网络"""
        if hasattr(model, 'update_target_network'):
            model.update_target_network()
        else:
            # 硬更新：直接复制参数
            model.q_target.load_state_dict(model.q_network.state_dict())


class DuelingDQN(nn.Module):
    """
    Dueling DQN网络结构
    
    将Q值分解为：
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: tuple = (128, 128)):
        super().__init__()
        
        # 共享特征层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
        )
        
        # 价值流（V(s)）
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )
        
        # 优势流（A(s,a)）
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], action_dim),
        )
    
    def forward(self, state):
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
