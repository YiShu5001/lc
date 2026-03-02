"""
优先级筛选函数
综合多个指标计算经验的优先级，用于采样
"""
from __future__ import annotations
from typing import Dict, Any, List
from dataclasses import dataclass
import numpy as np

from .BaseFilter import BaseFilter


@dataclass
class PriorityFilterConfig:
    """优先级筛选配置"""
    td_error_weight: float = 0.4        # TD误差权重
    risk_weight: float = 0.2            # 避障风险权重
    collaboration_weight: float = 0.2   # 协作价值权重
    novelty_weight: float = 0.2         # 新颖性权重
    alpha: float = 0.6                  # 优先级指数（PER论文中的alpha）
    epsilon: float = 1e-6              # 最小优先级常数


class PriorityFilter(BaseFilter):
    """
    优先级筛选函数
    
    综合多个指标计算经验的优先级
    用于优先经验回放采样
    """
    
    def __init__(self, config: PriorityFilterConfig = None):
        """
        Args:
            config: 筛选配置
        """
        super().__init__("PriorityFilter")
        self.config = config or PriorityFilterConfig()
    
    def compute_priority(self, metrics: Dict[str, float]) -> float:
        """
        计算经验的优先级
        
        Args:
            metrics: 指标字典，包含：
                - "TDError": TD误差
                - "ObstacleRisk": 避障风险
                - "Collaboration": 协作价值
                - "Novelty": 新颖性
        
        Returns:
            优先级值（越大优先级越高）
        """
        # 获取各项指标
        td_error = metrics.get("TDError", 0.0)
        risk = metrics.get("ObstacleRisk", 0.0)
        collaboration = metrics.get("Collaboration", 0.5)
        novelty = metrics.get("Novelty", 0.5)
        
        # 加权求和
        weighted_score = (
            td_error * self.config.td_error_weight +
            risk * self.config.risk_weight +
            collaboration * self.config.collaboration_weight +
            novelty * self.config.novelty_weight
        )
        
        # 应用优先级公式（类似PER）
        priority = (weighted_score + self.config.epsilon) ** self.config.alpha
        
        return float(priority)
    
    def should_transfer(self, experience: Dict[str, Any], metrics: Dict[str, float]) -> bool:
        """
        判断经验是否应该转移（用于采样）
        
        注意：这个方法主要用于接口兼容，实际优先级计算使用compute_priority
        """
        priority = self.compute_priority(metrics)
        # 优先级大于0就应该考虑采样
        return priority > self.config.epsilon
    
    def compute_priorities_batch(self, metrics_list: List[Dict[str, float]]) -> np.ndarray:
        """
        批量计算优先级
        
        Args:
            metrics_list: 指标字典列表
        
        Returns:
            优先级数组
        """
        priorities = []
        for metrics in metrics_list:
            priority = self.compute_priority(metrics)
            priorities.append(priority)
        return np.array(priorities)
