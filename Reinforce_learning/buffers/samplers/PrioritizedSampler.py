"""
优先采样器
基于优先级的采样（使用SumTree）
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np

from Reinforce_learning.RLg.SumTree import SumTree


class PrioritizedSampler:
    """
    优先采样器
    
    使用SumTree实现基于优先级的采样
    """
    
    def __init__(self, capacity: int):
        """
        Args:
            capacity: 容量
        """
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.size = 0
    
    def add(self, priority: float, experience: Dict[str, Any]) -> None:
        """
        添加经验和优先级
        
        Args:
            priority: 优先级
            experience: 经验字典
        """
        self.tree.add(priority, experience)
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """
        基于优先级采样
        
        Args:
            batch_size: 批次大小
        
        Returns:
            (batch, indices, priorities) 元组
        """
        if self.size < batch_size:
            raise ValueError(f"数据不足，需要至少 {batch_size} 条经验")
        
        batch = []
        indices = []
        priorities = []
        
        total_priority = self.tree.total()
        if total_priority == 0:
            # 如果没有优先级，均匀采样
            indices = np.random.choice(self.size, batch_size, replace=False)
            for idx in indices:
                data_idx = idx - self.capacity + 1
                if 0 <= data_idx < len(self.tree.data):
                    batch.append(self.tree.data[data_idx])
                    priorities.append(1.0)
            return batch, np.array(indices), np.array(priorities)
        
        segment = total_priority / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(p)
        
        return batch, np.array(indices), np.array(priorities)
    
    def update_priority(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        更新优先级
        
        Args:
            indices: 索引数组
            priorities: 优先级数组
        """
        for idx, priority in zip(indices, priorities):
            if idx < len(self.tree.tree):
                self.tree.update(idx, priority)
    
    def clear(self) -> None:
        """清空采样器"""
        self.tree = SumTree(self.capacity)
        self.size = 0
