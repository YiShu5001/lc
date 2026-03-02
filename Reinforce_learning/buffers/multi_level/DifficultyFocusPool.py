"""
难点聚焦池
存储高TD误差的经验，用于重点学习
"""
from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any
from collections import deque
import numpy as np

from Reinforce_learning.buffers.BaseBuffer import BaseBuffer, BufferConfig
from Reinforce_learning.buffers.filters import LearningFilter


class DifficultyFocusPool(BaseBuffer):
    """
    难点聚焦池
    
    功能：
    - 存储高TD误差的经验
    - 从基座覆盖池筛选转移而来
    - 容量中等，聚焦难点
    """
    
    def __init__(
        self,
        config: Optional[BufferConfig] = None,
        learning_filter: Optional[LearningFilter] = None
    ):
        """
        Args:
            config: 缓冲区配置
            learning_filter: 学习筛选器
        """
        super().__init__(config)
        self.learning_filter = learning_filter or LearningFilter()
        self.buffer = deque(maxlen=self.capacity)
        self.experiences = []
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        添加经验到难点聚焦池
        
        Args:
            td_error: TD误差（用于记录）
        """
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "td_error": td_error,
            "index": self.size,
            **kwargs
        }
        self.buffer.append(experience)
        self.experiences.append(experience)
        self.size = min(self.size + 1, self.capacity)
    
    def add_from_coverage_pool(
        self,
        experiences: List[Dict[str, Any]],
        td_errors: np.ndarray
    ) -> int:
        """
        从基座覆盖池筛选并添加经验
        
        Args:
            experiences: 基座覆盖池的经验列表
            td_errors: TD误差数组
        
        Returns:
            添加的经验数量
        """
        # 使用筛选器筛选
        metrics_dict = {"TDError": td_errors.tolist()}
        filtered = self.learning_filter.filter(experiences, metrics_dict)
        
        # 或者使用top-k筛选
        if len(filtered) == 0:
            filtered = self.learning_filter.filter_top_k(experiences, td_errors.tolist())
        
        # 添加筛选后的经验
        count = 0
        # 创建索引映射
        exp_to_idx = {id(exp): i for i, exp in enumerate(experiences)}
        
        for exp in filtered:
            if self.size < self.capacity:
                # 使用id来查找索引，避免numpy数组比较问题
                exp_id = id(exp)
                idx = exp_to_idx.get(exp_id)
                td_error = td_errors[idx] if idx is not None and idx < len(td_errors) else None
                self.add(
                    state=exp["state"],
                    action=exp["action"],
                    reward=exp["reward"],
                    next_state=exp["next_state"],
                    done=exp["done"],
                    td_error=td_error,
                    **{k: v for k, v in exp.items() if k not in ["state", "action", "reward", "next_state", "done"]}
                )
                count += 1
        
        return count
    
    def sample(self, batch_size: Optional[int] = None) -> Tuple:
        """
        采样经验
        
        Returns:
            (states, actions, rewards, next_states, dones)
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if not self.is_ready(batch_size):
            raise ValueError(f"缓冲区数据不足，需要至少 {batch_size} 条经验")
        
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = np.array([e["state"] for e in batch])
        actions = np.array([e["action"] for e in batch])
        rewards = np.array([e["reward"] for e in batch])
        next_states = np.array([e["next_state"] for e in batch])
        dones = np.array([e["done"] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def get_all_experiences(self) -> List[Dict[str, Any]]:
        """获取所有经验"""
        return list(self.buffer)
    
    def clear(self) -> None:
        """清空缓冲区"""
        super().clear()
        self.buffer.clear()
        self.experiences.clear()
