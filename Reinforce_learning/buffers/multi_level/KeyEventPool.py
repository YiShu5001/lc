"""
关键事件池
存储高价值、高稀有性的经验
"""
from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any
from collections import deque
import numpy as np

from Reinforce_learning.buffers.BaseBuffer import BaseBuffer, BufferConfig
from Reinforce_learning.buffers.filters.ValueFilter import ValueFilter


class KeyEventPool(BaseBuffer):
    """
    关键事件池
    
    功能：
    - 存储高价值、高稀有性的经验
    - 从难点聚焦池筛选转移而来
    - 容量较小，聚焦关键事件
    """
    
    def __init__(
        self,
        config: Optional[BufferConfig] = None,
        value_filter: Optional[ValueFilter] = None
    ):
        """
        Args:
            config: 缓冲区配置
            value_filter: 价值筛选器
        """
        super().__init__(config)
        self.value_filter = value_filter or ValueFilter()
        self.buffer = deque(maxlen=self.capacity)
        self.experiences = []
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        value_score: Optional[float] = None,
        novelty_score: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        添加经验到关键事件池
        
        Args:
            value_score: 价值性分数
            novelty_score: 新颖性分数
        """
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "value_score": value_score,
            "novelty_score": novelty_score,
            "index": self.size,
            **kwargs
        }
        self.buffer.append(experience)
        self.experiences.append(experience)
        self.size = min(self.size + 1, self.capacity)
    
    def add_from_difficulty_pool(
        self,
        experiences: List[Dict[str, Any]],
        value_scores: np.ndarray,
        novelty_scores: np.ndarray
    ) -> int:
        """
        从难点聚焦池筛选并添加经验
        
        Args:
            experiences: 难点聚焦池的经验列表
            value_scores: 价值性分数数组
            novelty_scores: 新颖性分数数组
        
        Returns:
            添加的经验数量
        """
        # 使用筛选器筛选
        metrics_dict = {
            "Collaboration": value_scores.tolist(),  # 简化：使用value_scores作为协作价值
            "Novelty": novelty_scores.tolist()
        }
        filtered = self.value_filter.filter(experiences, metrics_dict)
        
        # 或者使用top-k筛选
        if len(filtered) == 0:
            filtered = self.value_filter.filter_top_k(
                experiences,
                value_scores.tolist(),
                novelty_scores.tolist()
            )
        
        # 添加筛选后的经验
        count = 0
        # 创建索引映射
        exp_to_idx = {id(exp): i for i, exp in enumerate(experiences)}
        
        for exp in filtered:
            if self.size < self.capacity:
                # 使用id来查找索引，避免numpy数组比较问题
                exp_id = id(exp)
                idx = exp_to_idx.get(exp_id, 0)
                self.add(
                    state=exp["state"],
                    action=exp["action"],
                    reward=exp["reward"],
                    next_state=exp["next_state"],
                    done=exp["done"],
                    value_score=value_scores[idx] if idx < len(value_scores) else None,
                    novelty_score=novelty_scores[idx] if idx < len(novelty_scores) else None,
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
