"""
筛选函数基类
定义统一的筛选接口
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseFilter(ABC):
    """
    筛选函数基类
    
    所有筛选函数都需要继承此类，实现filter或should_transfer方法
    """
    
    def __init__(self, name: str = None):
        """
        Args:
            name: 筛选器名称
        """
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def should_transfer(self, experience: Dict[str, Any], metrics: Dict[str, float]) -> bool:
        """
        判断经验是否应该转移到下一层（抽象方法）
        
        Args:
            experience: 经验字典
            metrics: 指标字典，包含各种评价指标的值
        
        Returns:
            True表示应该转移，False表示不转移
        """
        pass
    
    def filter(self, experiences: List[Dict[str, Any]], metrics_dict: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """
        批量筛选经验
        
        Args:
            experiences: 经验列表
            metrics_dict: 指标字典，key为指标名，value为指标值列表
        
        Returns:
            筛选后的经验列表
        """
        filtered = []
        for i, exp in enumerate(experiences):
            # 构建当前经验的指标字典
            metrics = {name: values[i] for name, values in metrics_dict.items()}
            if self.should_transfer(exp, metrics):
                filtered.append(exp)
        return filtered
    
    def get_name(self) -> str:
        """获取筛选器名称"""
        return self.name
