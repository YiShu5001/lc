"""
TD误差指标
计算时序差分误差，用于识别需要重点学习的经验
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import torch

from .BaseMetric import BaseMetric


class TDMetric(BaseMetric):
    """
    TD误差指标
    
    计算Q值的时序差分误差，误差越大表示该经验越需要学习
    """
    
    def __init__(self, gamma: float = 0.99, device: str = "cpu"):
        """
        Args:
            gamma: 折扣因子
            device: 计算设备
        """
        super().__init__("TDError")
        self.gamma = gamma
        self.device = device
    
    def compute(
        self,
        experience: Dict[str, Any],
        model: Optional[Any] = None,
        **kwargs
    ) -> float:
        """
        计算TD误差
        
        Args:
            experience: 经验字典，需要包含：
                - state: 当前状态
                - action: 执行的动作
                - reward: 奖励
                - next_state: 下一状态
                - done: 是否结束
            model: Q网络模型（需要支持q_network和q_target方法）
        
        Returns:
            TD误差绝对值
        """
        if model is None:
            # 如果没有模型，返回默认值或从experience中获取
            if "td_error" in experience:
                return abs(float(experience["td_error"]))
            return 0.0
        
        # 提取经验数据
        state = experience.get("state")
        action = experience.get("action")
        reward = experience.get("reward", 0.0)
        next_state = experience.get("next_state")
        done = experience.get("done", False)
        
        if state is None or action is None or next_state is None:
            return 0.0
        
        # 转换为tensor
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 计算当前Q值
            if hasattr(model, 'q_network'):
                q_values = model.q_network(state)
            elif hasattr(model, 'critic'):
                # 对于Actor-Critic模型，使用critic
                if isinstance(action, np.ndarray):
                    action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
                else:
                    action_tensor = torch.tensor([[action]], dtype=torch.long).to(self.device)
                q_values = model.critic(state, action_tensor)
            else:
                return 0.0
            
            # 获取当前动作的Q值
            if isinstance(action, (int, np.integer)):
                current_q = q_values[0, action].item()
            else:
                # 连续动作
                current_q = q_values.item() if q_values.dim() == 0 else q_values[0].item()
            
            # 计算目标Q值
            if hasattr(model, 'q_target'):
                next_q_values = model.q_target(next_state)
            elif hasattr(model, 'critic_target'):
                # 使用目标critic
                next_q_values = model.critic_target(next_state, torch.zeros_like(action_tensor))
            else:
                # 如果没有目标网络，使用当前网络
                if hasattr(model, 'q_network'):
                    next_q_values = model.q_network(next_state)
                else:
                    next_q_values = model.critic(next_state, torch.zeros_like(action_tensor))
            
            # 获取最大Q值（离散）或直接使用（连续）
            if next_q_values.dim() > 1:
                target_q = next_q_values.max().item()
            else:
                target_q = next_q_values.item()
            
            # 计算TD目标
            target = reward + (1.0 - float(done)) * self.gamma * target_q
            
            # 计算TD误差
            td_error = abs(current_q - target)
        
        return float(td_error)
    
    def compute_batch(
        self,
        experiences: list,
        model: Optional[Any] = None
    ) -> np.ndarray:
        """
        批量计算TD误差
        
        Args:
            experiences: 经验列表
            model: Q网络模型
        
        Returns:
            TD误差数组
        """
        td_errors = []
        for exp in experiences:
            td_error = self.compute(exp, model)
            td_errors.append(td_error)
        return np.array(td_errors)
