"""
评价指标基类
定义统一的指标计算接口
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseMetric(ABC):
    """
    评价指标基类
    
    所有评价指标都需要继承此类，实现compute方法
    """
    
    def __init__(self, name: str = None):
        """
        Args:
            name: 指标名称
        """
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def compute(
        self,
        experience: Dict[str, Any],
        model: Optional[Any] = None,
        **kwargs
    ) -> float:
        """
        计算指标值（抽象方法）
        
        Args:
            experience: 经验字典，包含state, action, reward, next_state, done等
            model: 模型对象（某些指标需要模型计算，如TD误差）
            **kwargs: 额外参数
        
        Returns:
            指标值（float）
        """
        pass
    
    def update(self, batch_experiences: list, **kwargs) -> None:
        """
        批量更新（用于需要维护状态的指标）
        
        Args:
            batch_experiences: 经验列表
            **kwargs: 额外参数
        """
        pass
    
    def reset(self) -> None:
        """重置指标状态"""
        pass
    
    def get_name(self) -> str:
        """获取指标名称"""
        return self.name
