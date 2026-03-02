"""
多池采样器
从多个经验池中按权重采样
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from Reinforce_learning.buffers.BaseBuffer import BaseBuffer


class MultiPoolSampler:
    """
    多池采样器
    
    从多个经验池中按权重采样，支持优先采样
    """
    
    def __init__(
        self,
        pools: List[BaseBuffer],
        weights: Optional[List[float]] = None
    ):
        """
        Args:
            pools: 经验池列表
            weights: 各池的采样权重（如果为None，均匀权重）
        """
        self.pools = pools
        if weights is None:
            # 均匀权重
            self.weights = [1.0 / len(pools)] * len(pools)
        else:
            # 归一化权重
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
    
    def sample(
        self,
        batch_size: int,
        use_priority: bool = True
    ) -> Tuple:
        """
        从多个池中采样
        
        Args:
            batch_size: 批次大小
            use_priority: 是否使用优先采样
        
        Returns:
            (states, actions, rewards, next_states, dones) 或带额外信息的元组
        """
        # 计算每个池的采样数量
        pool_sizes = [len(pool) for pool in self.pools]
        total_size = sum(pool_sizes)
        
        if total_size == 0:
            raise ValueError("所有池都为空")
        
        # 按权重分配采样数量
        pool_batch_sizes = []
        remaining = batch_size
        
        for i, (pool, weight) in enumerate(zip(self.pools, self.weights)):
            if i == len(self.pools) - 1:
                # 最后一个池，分配剩余数量
                pool_batch_sizes.append(remaining)
            else:
                size = max(1, int(batch_size * weight))
                size = min(size, pool_sizes[i], remaining)
                pool_batch_sizes.append(size)
                remaining -= size
        
        # 从各池采样
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []
        
        for pool, pool_batch_size in zip(self.pools, pool_batch_sizes):
            if pool_batch_size > 0 and len(pool) >= pool_batch_size:
                result = pool.sample(pool_batch_size)
                if len(result) == 5:
                    states, actions, rewards, next_states, dones = result
                elif len(result) == 7:
                    # 带优先级的采样结果
                    states, actions, rewards, next_states, dones, _, _ = result
                else:
                    continue
                
                all_states.append(states)
                all_actions.append(actions)
                all_rewards.append(rewards)
                all_next_states.append(next_states)
                all_dones.append(dones)
        
        # 合并结果
        if len(all_states) == 0:
            raise ValueError("无法从任何池中采样")
        
        states = np.concatenate(all_states, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        rewards = np.concatenate(all_rewards, axis=0)
        next_states = np.concatenate(all_next_states, axis=0)
        dones = np.concatenate(all_dones, axis=0)
        
        return states, actions, rewards, next_states, dones
    
    def set_weights(self, weights: List[float]) -> None:
        """
        设置采样权重
        
        Args:
            weights: 权重列表
        """
        if len(weights) != len(self.pools):
            raise ValueError(f"权重数量({len(weights)})与池数量({len(self.pools)})不匹配")
        
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]
