"""
新颖性/稀有性指标
基于状态访问频率计算经验的新颖性和稀有性
"""
from __future__ import annotations
from typing import Dict, Any, Optional
from collections import defaultdict, Counter
import numpy as np
import torch

from .BaseMetric import BaseMetric


class NoveltyMetric(BaseMetric):
    """
    新颖性/稀有性指标
    
    维护状态访问频率，计算经验的新颖性
    访问频率越低，新颖性越高
    """
    
    def __init__(self, decay_factor: float = 0.99, hash_size: int = 1000):
        """
        Args:
            decay_factor: 访问频率衰减因子（定期衰减，避免频率无限增长）
            hash_size: 状态哈希表大小（用于离散化状态）
        """
        super().__init__("Novelty")
        self.decay_factor = decay_factor
        self.hash_size = hash_size
        self.state_counter = Counter()  # 状态访问计数器
        self.total_visits = 0  # 总访问次数
    
    def _hash_state(self, state: np.ndarray) -> int:
        """
        将状态哈希为整数
        
        Args:
            state: 状态数组
        
        Returns:
            哈希值
        """
        # 简化哈希：将状态离散化
        state_flat = np.array(state).flatten()
        # 使用前几个维度进行哈希
        hash_value = 0
        for i, val in enumerate(state_flat[:min(10, len(state_flat))]):
            hash_value = (hash_value * 31 + int(val * self.hash_size)) % (2**31)
        return hash_value
    
    def compute(
        self,
        experience: Dict[str, Any],
        model: Optional[Any] = None,
        **kwargs
    ) -> float:
        """
        计算新颖性分数
        
        Args:
            experience: 经验字典，需要包含state
            model: 模型对象（可选）
        
        Returns:
            新颖性分数（0-1之间，1表示最高新颖性/稀有性）
        """
        state = experience.get("state")
        if state is None:
            return 0.5  # 默认中等新颖性
        
        # 转换为numpy数组
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        
        state = np.array(state)
        
        # 哈希状态
        state_hash = self._hash_state(state)
        
        # 获取访问频率
        visit_count = self.state_counter.get(state_hash, 0)
        
        # 更新计数器
        self.state_counter[state_hash] = visit_count + 1
        self.total_visits += 1
        
        # 计算新颖性：访问频率越低，新颖性越高
        if self.total_visits > 0:
            frequency = visit_count / max(self.total_visits, 1)
            novelty = 1.0 - frequency  # 频率越低，新颖性越高
        else:
            novelty = 1.0  # 首次访问，最高新颖性
        
        return float(np.clip(novelty, 0.0, 1.0))
    
    def update(self, batch_experiences: list, **kwargs) -> None:
        """
        批量更新状态访问频率
        
        Args:
            batch_experiences: 经验列表
        """
        for exp in batch_experiences:
            self.compute(exp)
    
    def reset(self) -> None:
        """重置计数器"""
        self.state_counter.clear()
        self.total_visits = 0
    
    def decay(self) -> None:
        """
        衰减访问频率（定期调用，避免频率无限增长）
        """
        # 对所有计数应用衰减
        for key in list(self.state_counter.keys()):
            self.state_counter[key] = int(self.state_counter[key] * self.decay_factor)
            if self.state_counter[key] < 1:
                del self.state_counter[key]
        
        self.total_visits = int(self.total_visits * self.decay_factor)


# 修复导入
import torch
