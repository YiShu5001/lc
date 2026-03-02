"""
待学习筛选函数
基于TD误差筛选需要重点学习的经验（基座覆盖池 → 难点聚焦池）
"""
from __future__ import annotations
from typing import Dict, Any, List
from dataclasses import dataclass
import numpy as np

from .BaseFilter import BaseFilter


@dataclass
class LearningFilterConfig:
    """待学习筛选配置"""
    td_error_threshold: float = 0.1    # TD误差阈值
    min_td_error: float = 0.01          # 最小TD误差（避免噪声）
    top_k_ratio: float = 0.1            # 选择top-k比例的经验


class LearningFilter(BaseFilter):
    """
    待学习筛选函数
    
    基于TD误差筛选需要重点学习的经验
    TD误差越大，表示该经验越需要学习
    """
    
    def __init__(self, config: LearningFilterConfig = None):
        """
        Args:
            config: 筛选配置
        """
        super().__init__("LearningFilter")
        self.config = config or LearningFilterConfig()
    
    def should_transfer(self, experience: Dict[str, Any], metrics: Dict[str, float]) -> bool:
        """
        判断经验是否应该转移到难点聚焦池
        
        Args:
            experience: 经验字典
            metrics: 指标字典，需要包含"TDError"指标
        
        Returns:
            True表示应该转移到难点聚焦池
        """
        td_error = metrics.get("TDError", 0.0)
        
        # 筛选条件：TD误差大于阈值
        if td_error >= self.config.td_error_threshold:
            return True
        
        # 或者TD误差大于最小值（避免完全忽略小误差）
        if td_error >= self.config.min_td_error:
            # 可以添加其他条件，如随机采样一定比例
            # 这里简化处理，只使用阈值
            pass
        
        return False
    
    def filter_top_k(self, experiences: List[Dict[str, Any]], td_errors: List[float]) -> List[Dict[str, Any]]:
        """
        筛选top-k高TD误差的经验
        
        Args:
            experiences: 经验列表
            td_errors: TD误差列表
        
        Returns:
            筛选后的经验列表
        """
        if len(experiences) == 0:
            return []
        
        # 按TD误差排序
        sorted_indices = sorted(
            range(len(experiences)),
            key=lambda i: td_errors[i],
            reverse=True
        )
        
        # 选择top-k
        k = max(1, int(len(experiences) * self.config.top_k_ratio))
        selected_indices = sorted_indices[:k]
        
        return [experiences[i] for i in selected_indices]
