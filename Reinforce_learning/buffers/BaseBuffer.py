"""
经验池基础类
定义经验回放缓冲区的统一接口，支持标准回放、优先回放、多智能体回放等
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
from collections import deque

import numpy as np
import torch


@dataclass
class BufferConfig:
    """
    经验池配置基类
    """
    capacity: int = 10000              # 缓冲区容量
    batch_size: int = 64               # 批次大小


class BaseBuffer(ABC):
    """
    经验池基类
    
    定义了经验回放缓冲区的统一接口，支持：
    - 标准经验回放
    - 优先经验回放
    - 多智能体经验回放
    """
    
    def __init__(self, config: Optional[BufferConfig] = None):
        """
        Args:
            config: 缓冲区配置
        """
        self.config = config or BufferConfig()
        self.capacity = self.config.capacity
        self.batch_size = self.config.batch_size
        self.size = 0
    
    @abstractmethod
    def add(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
        **kwargs
    ) -> None:
        """
        添加经验到缓冲区（抽象方法）
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
            **kwargs: 额外信息（如优先级、权重等）
        """
        pass
    
    @abstractmethod
    def sample(self, batch_size: Optional[int] = None) -> Tuple:
        """
        从缓冲区采样经验（抽象方法）
        
        Args:
            batch_size: 批次大小（如果为None，使用配置的batch_size）
        
        Returns:
            经验元组，格式由子类决定
        """
        pass
    
    def update_priority(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        更新经验的优先级（可选方法，优先回放需要实现）
        
        Args:
            indices: 经验索引
            priorities: 新的优先级
        """
        pass
    
    def __len__(self) -> int:
        """返回缓冲区当前大小"""
        return self.size
    
    def is_ready(self, batch_size: Optional[int] = None) -> bool:
        """
        检查缓冲区是否有足够的数据进行采样
        
        Args:
            batch_size: 批次大小
        
        Returns:
            是否可以进行采样
        """
        if batch_size is None:
            batch_size = self.batch_size
        return self.size >= batch_size
    
    def clear(self) -> None:
        """清空缓冲区"""
        self.size = 0


class ReplayBuffer(BaseBuffer):
    """
    标准经验回放缓冲区
    
    使用FIFO队列存储经验，均匀随机采样
    """
    
    def __init__(self, config: Optional[BufferConfig] = None):
        """
        Args:
            config: 缓冲区配置
        """
        super().__init__(config)
        self.buffer = deque(maxlen=self.capacity)
    
    def add(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
        **kwargs
    ) -> None:
        """
        添加经验到缓冲区
        """
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            **kwargs
        }
        self.buffer.append(experience)
        self.size = len(self.buffer)
    
    def sample(self, batch_size: Optional[int] = None) -> Tuple:
        """
        均匀随机采样经验
        
        Returns:
            (states, actions, rewards, next_states, dones) 元组
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if not self.is_ready(batch_size):
            raise ValueError(f"缓冲区数据不足，需要至少 {batch_size} 条经验")
        
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # 提取各个字段
        states = np.array([e["state"] for e in batch])
        actions = np.array([e["action"] for e in batch])
        rewards = np.array([e["reward"] for e in batch])
        next_states = np.array([e["next_state"] for e in batch])
        dones = np.array([e["done"] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def clear(self) -> None:
        """清空缓冲区"""
        super().clear()
        self.buffer.clear()


class PrioritizedReplayBuffer(BaseBuffer):
    """
    优先经验回放缓冲区
    
    根据TD误差的优先级采样经验，提高学习效率
    """
    
    @dataclass
    class Config(BufferConfig):
        alpha: float = 0.6              # 优先级指数（0=均匀采样，1=完全按优先级）
        beta: float = 0.4                # 重要性采样指数（初始值）
        beta_increment: float = 0.001    # beta的增量
        epsilon: float = 1e-6           # 避免优先级为0的小常数
    
    def __init__(self, config: Optional[Config] = None):
        """
        Args:
            config: 优先回放缓冲区配置
        """
        super().__init__(config)
        self.config = config or self.Config()
        
        # 使用SumTree存储优先级（简化实现，使用数组）
        self.priorities = np.zeros(self.capacity)
        self.buffer = [None] * self.capacity
        self.pos = 0
        self.max_priority = 1.0
        self.beta = self.config.beta
    
    def add(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        添加经验到缓冲区
        
        Args:
            priority: 优先级（如果为None，使用最大优先级）
        """
        if priority is None:
            priority = self.max_priority
        
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            **kwargs
        }
        
        self.buffer[self.pos] = experience
        self.priorities[self.pos] = priority ** self.config.alpha
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: Optional[int] = None) -> Tuple:
        """
        根据优先级采样经验
        
        Returns:
            (states, actions, rewards, next_states, dones, indices, weights) 元组
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if not self.is_ready(batch_size):
            raise ValueError(f"缓冲区数据不足，需要至少 {batch_size} 条经验")
        
        # 计算优先级总和
        priorities = self.priorities[:self.size]
        total_priority = np.sum(priorities)
        
        # 采样
        segment = total_priority / batch_size
        indices = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)
            idx = self._sample_index(value, priorities, total_priority)
            indices.append(idx)
        
        # 提取批次数据
        batch = [self.buffer[i] for i in indices]
        states = np.array([e["state"] for e in batch])
        actions = np.array([e["action"] for e in batch])
        rewards = np.array([e["reward"] for e in batch])
        next_states = np.array([e["next_state"] for e in batch])
        dones = np.array([e["done"] for e in batch])
        
        # 计算重要性采样权重
        priorities_sample = priorities[indices]
        probabilities = priorities_sample / total_priority
        weights = (self.size * probabilities) ** (-self.beta)
        weights = weights / weights.max()  # 归一化
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.config.beta_increment)
        
        return states, actions, rewards, next_states, dones, np.array(indices), weights
    
    def _sample_index(self, value: float, priorities: np.ndarray, total: float) -> int:
        """
        根据值查找对应的索引（简化实现）
        """
        cumulative = np.cumsum(priorities)
        idx = np.searchsorted(cumulative, value)
        return min(idx, len(priorities) - 1)
    
    def update_priority(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        更新经验的优先级
        
        Args:
            indices: 经验索引
            priorities: 新的优先级（TD误差）
        """
        priorities = np.abs(priorities) + self.config.epsilon
        for idx, priority in zip(indices, priorities):
            if idx < self.size:
                self.priorities[idx] = priority ** self.config.alpha
                self.max_priority = max(self.max_priority, priority)
    
    def clear(self) -> None:
        """清空缓冲区"""
        super().clear()
        self.priorities.fill(0)
        self.buffer = [None] * self.capacity
        self.pos = 0
        self.max_priority = 1.0
        self.beta = self.config.beta


class MultiAgentBuffer(BaseBuffer):
    """
    多智能体经验回放缓冲区
    
    存储多个智能体的经验，支持联合采样和独立采样
    """
    
    @dataclass
    class Config(BufferConfig):
        num_agents: int = 2             # 智能体数量
        shared_buffer: bool = False     # 是否使用共享缓冲区
    
    def __init__(self, config: Optional[Config] = None):
        """
        Args:
            config: 多智能体缓冲区配置
        """
        super().__init__(config)
        self.config = config or self.Config()
        
        if self.config.shared_buffer:
            # 共享缓冲区
            self.buffer = deque(maxlen=self.capacity)
        else:
            # 每个智能体独立的缓冲区
            self.buffers = [deque(maxlen=self.capacity) for _ in range(self.config.num_agents)]
    
    def add(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
        agent_id: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        添加经验到缓冲区
        
        Args:
            agent_id: 智能体ID（如果为None，添加到所有缓冲区）
        """
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "agent_id": agent_id,
            **kwargs
        }
        
        if self.config.shared_buffer:
            self.buffer.append(experience)
            self.size = len(self.buffer)
        else:
            if agent_id is None:
                # 添加到所有缓冲区
                for buf in self.buffers:
                    buf.append(experience)
            else:
                self.buffers[agent_id].append(experience)
            self.size = min(len(buf) for buf in self.buffers)
    
    def sample(
        self,
        batch_size: Optional[int] = None,
        agent_id: Optional[int] = None,
        joint: bool = False
    ) -> Tuple:
        """
        采样经验
        
        Args:
            batch_size: 批次大小
            agent_id: 智能体ID（如果为None，从所有智能体采样）
            joint: 是否联合采样（所有智能体的经验一起采样）
        
        Returns:
            经验元组
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if self.config.shared_buffer:
            # 从共享缓冲区采样
            if not self.is_ready(batch_size):
                raise ValueError(f"缓冲区数据不足")
            
            indices = np.random.choice(self.size, batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]
        else:
            if agent_id is not None:
                # 从指定智能体的缓冲区采样
                buf = self.buffers[agent_id]
                if len(buf) < batch_size:
                    raise ValueError(f"智能体 {agent_id} 的缓冲区数据不足")
                indices = np.random.choice(len(buf), batch_size, replace=False)
                batch = [buf[i] for i in indices]
            elif joint:
                # 联合采样：从所有智能体混合采样
                all_experiences = []
                for buf in self.buffers:
                    all_experiences.extend(list(buf))
                if len(all_experiences) < batch_size:
                    raise ValueError(f"联合缓冲区数据不足")
                indices = np.random.choice(len(all_experiences), batch_size, replace=False)
                batch = [all_experiences[i] for i in indices]
            else:
                # 独立采样：每个智能体采样相同数量的经验
                batch = []
                per_agent_size = batch_size // self.config.num_agents
                for buf in self.buffers:
                    if len(buf) < per_agent_size:
                        continue
                    indices = np.random.choice(len(buf), per_agent_size, replace=False)
                    batch.extend([buf[i] for i in indices])
        
        # 提取字段
        states = np.array([e["state"] for e in batch])
        actions = np.array([e["action"] for e in batch])
        rewards = np.array([e["reward"] for e in batch])
        next_states = np.array([e["next_state"] for e in batch])
        dones = np.array([e["done"] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def clear(self) -> None:
        """清空缓冲区"""
        super().clear()
        if self.config.shared_buffer:
            self.buffer.clear()
        else:
            for buf in self.buffers:
                buf.clear()
