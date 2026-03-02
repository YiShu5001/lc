"""
基座覆盖池
存储所有基础经验，提供广泛覆盖
"""
from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any
from collections import deque
import numpy as np

from Reinforce_learning.buffers.BaseBuffer import BaseBuffer, BufferConfig


class BaseCoveragePool(BaseBuffer):
    """
    基座覆盖池
    
    功能：
    - 存储所有基础经验
    - 容量最大，提供广泛覆盖
    - 支持优先回放（可选）
    """
    
    def __init__(self, config: Optional[BufferConfig] = None, use_priority: bool = False):
        """
        Args:
            config: 缓冲区配置
            use_priority: 是否使用优先回放
        """
        super().__init__(config)
        self.use_priority = use_priority
        
        if use_priority:
            # 使用SumTree实现优先回放
            from Reinforce_learning.RLg.SumTree import SumTree
            self.tree = SumTree(self.capacity)
            self.priorities = np.zeros(self.capacity)
            self.max_priority = 1.0
        else:
            # 使用deque存储
            self.buffer = deque(maxlen=self.capacity)
        
        self.experiences = []  # 存储经验对象（用于筛选）
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        添加经验到基座覆盖池
        
        Args:
            priority: 优先级（如果use_priority=True）
        """
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "index": self.size,  # 记录索引
            **kwargs
        }
        
        if self.use_priority:
            # 使用优先回放
            if priority is None:
                priority = self.max_priority
            else:
                self.max_priority = max(self.max_priority, priority)
            
            self.tree.add(priority, experience)
            self.priorities[self.tree.write - 1] = priority
        else:
            # 使用FIFO
            self.buffer.append(experience)
        
        self.experiences.append(experience)
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: Optional[int] = None) -> Tuple:
        """
        采样经验
        
        Returns:
            (states, actions, rewards, next_states, dones) 或带优先级的元组
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if not self.is_ready(batch_size):
            raise ValueError(f"缓冲区数据不足，需要至少 {batch_size} 条经验")
        
        if self.use_priority:
            # 优先采样
            batch = []
            indices = []
            priorities = []
            segment = self.tree.total() / batch_size
            
            for i in range(batch_size):
                a = segment * i
                b = segment * (i + 1)
                s = np.random.uniform(a, b)
                idx, p, data = self.tree.get(s)
                batch.append(data)
                indices.append(idx)
                priorities.append(p)
            
            # 提取字段
            states = np.array([e["state"] for e in batch])
            actions = np.array([e["action"] for e in batch])
            rewards = np.array([e["reward"] for e in batch])
            next_states = np.array([e["next_state"] for e in batch])
            dones = np.array([e["done"] for e in batch])
            
            return states, actions, rewards, next_states, dones, np.array(indices), np.array(priorities)
        else:
            # 均匀采样
            indices = np.random.choice(self.size, batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]
            
            states = np.array([e["state"] for e in batch])
            actions = np.array([e["action"] for e in batch])
            rewards = np.array([e["reward"] for e in batch])
            next_states = np.array([e["next_state"] for e in batch])
            dones = np.array([e["done"] for e in batch])
            
            return states, actions, rewards, next_states, dones
    
    def update_priority(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        更新优先级
        
        Args:
            indices: 经验索引（SumTree中的索引）
            priorities: 新的优先级
        """
        if not self.use_priority:
            return
        
        for idx, priority in zip(indices, priorities):
            if idx < len(self.tree.tree):
                self.tree.update(idx, priority)
                self.max_priority = max(self.max_priority, priority)
    
    def get_all_experiences(self) -> List[Dict[str, Any]]:
        """
        获取所有经验（用于筛选）
        
        Returns:
            经验列表
        """
        if self.use_priority:
            # 从SumTree提取所有经验
            experiences = []
            for i in range(self.size):
                if i < len(self.tree.data):
                    exp = self.tree.data[i]
                    if exp is not None:
                        experiences.append(exp)
            return experiences
        else:
            return list(self.buffer)
    
    def clear(self) -> None:
        """清空缓冲区"""
        super().clear()
        if self.use_priority:
            from Reinforce_learning.RLg.SumTree import SumTree
            self.tree = SumTree(self.capacity)
            self.priorities = np.zeros(self.capacity)
            self.max_priority = 1.0
        else:
            self.buffer.clear()
        self.experiences.clear()
