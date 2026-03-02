"""
避障风险指标
计算避障相关的风险指标，用于识别高风险经验
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import torch

from .BaseMetric import BaseMetric


class RiskMetric(BaseMetric):
    """
    避障风险指标
    
    基于状态中的障碍物信息计算风险分数
    风险越高，该经验越重要（需要重点学习避障）
    """
    
    def __init__(self, risk_threshold: float = 1.0, use_distance: bool = True):
        """
        Args:
            risk_threshold: 风险阈值（距离障碍物的最小安全距离）
            use_distance: 是否使用距离计算风险
        """
        super().__init__("ObstacleRisk")
        self.risk_threshold = risk_threshold
        self.use_distance = use_distance
    
    def compute(
        self,
        experience: Dict[str, Any],
        model: Optional[Any] = None,
        **kwargs
    ) -> float:
        """
        计算避障风险
        
        Args:
            experience: 经验字典，需要包含state（包含障碍物信息）
            model: 模型对象（可选）
        
        Returns:
            风险分数（0-1之间，1表示最高风险）
        """
        state = experience.get("state")
        if state is None:
            return 0.0
        
        # 转换为numpy数组
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        
        state = np.array(state).flatten()
        
        # 假设状态包含位置和障碍物信息
        # 对于无人机环境，状态通常包含：位置(3) + 速度(3) + 姿态(3) + 角速度(3) = 12维
        # 如果有障碍物信息，可能在额外维度中
        
        # 方法1：基于位置距离障碍物的风险
        if self.use_distance and len(state) >= 3:
            # 提取位置（前3维通常是x, y, z）
            position = state[:3]
            
            # 计算到原点的距离（假设原点附近有障碍物）
            # 实际应用中，需要从状态中提取障碍物位置
            distance_to_origin = np.linalg.norm(position)
            
            # 风险分数：距离越近，风险越高
            if distance_to_origin < self.risk_threshold:
                risk = 1.0 - (distance_to_origin / self.risk_threshold)
            else:
                risk = 0.0
            
            return float(np.clip(risk, 0.0, 1.0))
        
        # 方法2：基于状态中的障碍物特征
        # 如果状态包含障碍物距离信息（通常在特定维度）
        if len(state) > 12:
            # 假设12维之后是障碍物相关信息
            obstacle_features = state[12:]
            if len(obstacle_features) > 0:
                # 计算最小障碍物距离
                min_obstacle_dist = np.min(obstacle_features) if len(obstacle_features) > 0 else float('inf')
                
                # 风险分数
                if min_obstacle_dist < self.risk_threshold:
                    risk = 1.0 - (min_obstacle_dist / self.risk_threshold)
                else:
                    risk = 0.0
                
                return float(np.clip(risk, 0.0, 1.0))
        
        # 默认：无风险
        return 0.0


# 修复导入
import torch
