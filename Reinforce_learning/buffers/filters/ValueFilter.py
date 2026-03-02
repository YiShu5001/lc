"""
价值筛选函数
基于价值性和稀有性筛选关键事件（难点聚焦池 → 关键事件池）
"""
from __future__ import annotations
from typing import Dict, Any, List
from dataclasses import dataclass

from .BaseFilter import BaseFilter


@dataclass
class ValueFilterConfig:
    """价值筛选配置"""
    value_threshold: float = 0.7        # 价值性阈值
    novelty_threshold: float = 0.5      # 新颖性阈值
    value_weight: float = 0.6            # 价值性权重
    novelty_weight: float = 0.4         # 新颖性权重
    combined_threshold: float = 0.6     # 综合分数阈值
    top_k_ratio: float = 0.05           # 选择top-k比例的经验


class ValueFilter(BaseFilter):
    """
    价值筛选函数
    
    基于价值性和稀有性筛选关键事件
    价值性：基于奖励、协作价值等
    稀有性：基于新颖性指标
    """
    
    def __init__(self, config: ValueFilterConfig = None):
        """
        Args:
            config: 筛选配置
        """
        super().__init__("ValueFilter")
        self.config = config or ValueFilterConfig()
    
    def should_transfer(self, experience: Dict[str, Any], metrics: Dict[str, float]) -> bool:
        """
        判断经验是否应该转移到关键事件池
        
        Args:
            experience: 经验字典
            metrics: 指标字典，需要包含：
                - "Novelty": 新颖性指标
                - "Collaboration": 协作价值（可选）
                - "ObstacleRisk": 避障风险（可选）
        
        Returns:
            True表示应该转移到关键事件池
        """
        # 获取各项指标
        novelty = metrics.get("Novelty", 0.0)
        collaboration = metrics.get("Collaboration", 0.5)
        risk = metrics.get("ObstacleRisk", 0.0)
        
        # 计算价值性分数（综合多个指标）
        # 高协作价值 + 高风险（需要学习避障） = 高价值
        value_score = (
            collaboration * 0.5 +
            risk * 0.5
        )
        
        # 计算综合分数
        combined_score = (
            value_score * self.config.value_weight +
            novelty * self.config.novelty_weight
        )
        
        # 筛选条件：综合分数大于阈值
        if combined_score >= self.config.combined_threshold:
            return True
        
        # 或者价值性和新颖性都较高
        if value_score >= self.config.value_threshold and novelty >= self.config.novelty_threshold:
            return True
        
        return False
    
    def filter_top_k(
        self,
        experiences: List[Dict[str, Any]],
        value_scores: List[float],
        novelty_scores: List[float]
    ) -> List[Dict[str, Any]]:
        """
        筛选top-k高价值经验
        
        Args:
            experiences: 经验列表
            value_scores: 价值性分数列表
            novelty_scores: 新颖性分数列表
        
        Returns:
            筛选后的经验列表
        """
        if len(experiences) == 0:
            return []
        
        # 计算综合分数
        combined_scores = [
            value * self.config.value_weight + novelty * self.config.novelty_weight
            for value, novelty in zip(value_scores, novelty_scores)
        ]
        
        # 按综合分数排序
        sorted_indices = sorted(
            range(len(experiences)),
            key=lambda i: combined_scores[i],
            reverse=True
        )
        
        # 选择top-k
        k = max(1, int(len(experiences) * self.config.top_k_ratio))
        selected_indices = sorted_indices[:k]
        
        return [experiences[i] for i in selected_indices]
