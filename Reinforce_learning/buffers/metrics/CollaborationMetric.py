"""
协作价值/冲突指标
计算多智能体场景下的协作价值和冲突指标
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import torch

from .BaseMetric import BaseMetric


class CollaborationMetric(BaseMetric):
    """
    协作价值/冲突指标
    
    用于多智能体场景，评估经验的协作价值和冲突程度
    """
    
    def __init__(self, num_agents: int = 2, conflict_weight: float = 0.5):
        """
        Args:
            num_agents: 智能体数量
            conflict_weight: 冲突权重（0-1之间）
        """
        super().__init__("Collaboration")
        self.num_agents = num_agents
        self.conflict_weight = conflict_weight
    
    def compute(
        self,
        experience: Dict[str, Any],
        model: Optional[Any] = None,
        **kwargs
    ) -> float:
        """
        计算协作价值分数
        
        Args:
            experience: 经验字典，需要包含：
                - state: 状态（可能包含多智能体信息）
                - reward: 奖励（可能包含协作奖励）
                - info: 额外信息（可能包含协作相关指标）
            model: 模型对象（可选）
        
        Returns:
            协作价值分数（0-1之间，1表示高协作价值）
        """
        # 方法1：基于奖励中的协作成分
        reward = experience.get("reward", 0.0)
        info = experience.get("info", {})
        
        # 如果info中包含协作指标
        if isinstance(info, dict):
            collaboration_reward = info.get("collaboration_reward", 0.0)
            conflict_penalty = info.get("conflict_penalty", 0.0)
            
            # 归一化到[0, 1]
            collaboration_score = (collaboration_reward + 1.0) / 2.0  # 假设奖励范围[-1, 1]
            conflict_score = abs(conflict_penalty)
            
            # 综合分数：协作价值 - 冲突惩罚
            value = collaboration_score * (1 - self.conflict_weight) - conflict_score * self.conflict_weight
            return float(np.clip(value, 0.0, 1.0))
        
        # 方法2：基于状态中的多智能体距离
        state = experience.get("state")
        if state is not None:
            if isinstance(state, torch.Tensor):
                state = state.detach().cpu().numpy()
            
            state = np.array(state).flatten()
            
            # 假设状态包含多个智能体的位置信息
            # 对于多智能体，状态可能是：每个智能体的状态拼接
            if len(state) >= self.num_agents * 3:
                # 提取每个智能体的位置（前3维）
                positions = []
                for i in range(self.num_agents):
                    start_idx = i * (len(state) // self.num_agents)
                    agent_state = state[start_idx:start_idx+3]
                    positions.append(agent_state)
                
                # 计算智能体之间的距离
                distances = []
                for i in range(self.num_agents):
                    for j in range(i+1, self.num_agents):
                        dist = np.linalg.norm(positions[i] - positions[j])
                        distances.append(dist)
                
                # 协作价值：距离适中时最高（太近冲突，太远无协作）
                if len(distances) > 0:
                    avg_distance = np.mean(distances)
                    # 假设最优协作距离为2.0（可配置）
                    optimal_distance = 2.0
                    collaboration_value = 1.0 / (1.0 + abs(avg_distance - optimal_distance))
                    return float(np.clip(collaboration_value, 0.0, 1.0))
        
        # 默认：中等协作价值
        return 0.5


# 修复导入
import torch
