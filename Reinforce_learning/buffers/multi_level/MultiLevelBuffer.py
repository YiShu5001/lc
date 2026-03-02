"""
多层经验池主类
管理三层经验池的协调工作，实现经验流转和采样
"""
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from Reinforce_learning.buffers.BaseBuffer import BaseBuffer, BufferConfig
from Reinforce_learning.buffers.MultiLevelBufferConfig import MultiLevelBufferConfig
from Reinforce_learning.buffers.multi_level.BaseCoveragePool import BaseCoveragePool
from Reinforce_learning.buffers.multi_level.DifficultyFocusPool import DifficultyFocusPool
from Reinforce_learning.buffers.multi_level.KeyEventPool import KeyEventPool
from Reinforce_learning.buffers.samplers.MultiPoolSampler import MultiPoolSampler
from Reinforce_learning.buffers.filters.LearningFilter import LearningFilter
from Reinforce_learning.buffers.filters.ValueFilter import ValueFilter
from Reinforce_learning.buffers.filters.PriorityFilter import PriorityFilter
from Reinforce_learning.buffers.metrics.TDMetric import TDMetric
from Reinforce_learning.buffers.metrics.RiskMetric import RiskMetric
from Reinforce_learning.buffers.metrics.CollaborationMetric import CollaborationMetric
from Reinforce_learning.buffers.metrics.NoveltyMetric import NoveltyMetric


class MultiLevelBuffer(BaseBuffer):
    """
    多层经验池主类
    
    管理三层经验池：
    1. 基座覆盖池：存储所有经验
    2. 难点聚焦池：存储高TD误差经验
    3. 关键事件池：存储高价值、高稀有性经验
    
    实现经验流转和优先采样
    """
    
    def __init__(self, config: Optional[MultiLevelBufferConfig] = None):
        """
        Args:
            config: 多层经验池配置
        """
        # 使用基座覆盖池的容量作为总容量
        self.multi_level_config = config or MultiLevelBufferConfig()
        base_config = BufferConfig(
            capacity=self.multi_level_config.coverage_capacity,
            batch_size=self.multi_level_config.batch_size
        )
        super().__init__(base_config)
        
        # 创建三层经验池
        coverage_config = BufferConfig(
            capacity=self.multi_level_config.coverage_capacity,
            batch_size=self.multi_level_config.batch_size
        )
        difficulty_config = BufferConfig(
            capacity=self.multi_level_config.difficulty_capacity,
            batch_size=self.multi_level_config.batch_size
        )
        keyevent_config = BufferConfig(
            capacity=self.multi_level_config.keyevent_capacity,
            batch_size=self.multi_level_config.batch_size
        )
        
        self.coverage_pool = BaseCoveragePool(
            config=coverage_config,
            use_priority=self.multi_level_config.use_priority
        )
        self.difficulty_pool = DifficultyFocusPool(
            config=difficulty_config,
            learning_filter=LearningFilter()
        )
        self.keyevent_pool = KeyEventPool(
            config=keyevent_config,
            value_filter=ValueFilter()
        )
        
        # 创建采样器
        self.sampler = MultiPoolSampler(
            pools=[self.coverage_pool, self.difficulty_pool, self.keyevent_pool],
            weights=self.multi_level_config.pool_weights
        )
        
        # 创建评价指标
        self.td_metric = TDMetric()
        self.risk_metric = RiskMetric()
        self.collaboration_metric = CollaborationMetric()
        self.novelty_metric = NoveltyMetric()
        
        # 创建优先级筛选器
        self.priority_filter = PriorityFilter()
        
        # 训练步数（用于控制筛选频率）
        self.step_count = 0
        self.last_filter_step = 0
        self.config = self.multi_level_config  # 兼容性
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        **kwargs
    ) -> None:
        """
        添加经验到基座覆盖池
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
            **kwargs: 额外信息
        """
        # 计算初始优先级（如果需要）
        priority = None
        if self.multi_level_config.use_priority:
            # 计算基础指标
            experience = {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done
            }
            metrics = self._compute_metrics(experience)
            priority = self.priority_filter.compute_priority(metrics)
        
        # 添加到基座覆盖池
        self.coverage_pool.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            priority=priority,
            **kwargs
        )
        
        self.size = len(self.coverage_pool)
        self.step_count += 1
        
        # 定期执行筛选和转移
        if self.step_count - self.last_filter_step >= self.multi_level_config.filter_freq:
            self.filter_and_transfer()
            self.last_filter_step = self.step_count
    
    def sample(self, batch_size: Optional[int] = None) -> Tuple:
        """
        从所有池按优先级采样
        
        Args:
            batch_size: 批次大小
        
        Returns:
            (states, actions, rewards, next_states, dones) 或带优先级的元组
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        return self.sampler.sample(batch_size, use_priority=self.config.use_priority)
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        更新TD误差（用于优先级计算）
        
        Args:
            indices: 经验索引（如果使用优先回放）
            td_errors: TD误差数组
        """
        if self.multi_level_config.use_priority:
            # 更新基座覆盖池的优先级
            priorities = []
            for td_error in td_errors:
                metrics = {"TDError": float(td_error)}
                priority = self.priority_filter.compute_priority(metrics)
                priorities.append(priority)
            
            self.coverage_pool.update_priority(indices, np.array(priorities))
    
    def filter_and_transfer(self) -> Dict[str, int]:
        """
        执行筛选和转移（基座→难点，难点→关键）
        
        Returns:
            转移统计信息
        """
        stats = {
            "coverage_to_difficulty": 0,
            "difficulty_to_keyevent": 0
        }
        
        # 1. 基座覆盖池 → 难点聚焦池
        coverage_experiences = self.coverage_pool.get_all_experiences()
        if len(coverage_experiences) > 0:
            # 计算TD误差
            td_errors = []
            for exp in coverage_experiences:
                td_error = self.td_metric.compute(exp)
                td_errors.append(td_error)
            td_errors = np.array(td_errors)
            
            # 筛选并转移
            count = self.difficulty_pool.add_from_coverage_pool(
                coverage_experiences,
                td_errors
            )
            stats["coverage_to_difficulty"] = count
        
        # 2. 难点聚焦池 → 关键事件池
        difficulty_experiences = self.difficulty_pool.get_all_experiences()
        if len(difficulty_experiences) > 0:
            # 计算价值性和新颖性
            value_scores = []
            novelty_scores = []
            
            for exp in difficulty_experiences:
                # 计算价值性（综合多个指标）
                risk = self.risk_metric.compute(exp)
                collaboration = self.collaboration_metric.compute(exp)
                value_score = (risk + collaboration) / 2.0
                value_scores.append(value_score)
                
                # 计算新颖性
                novelty = self.novelty_metric.compute(exp)
                novelty_scores.append(novelty)
            
            value_scores = np.array(value_scores)
            novelty_scores = np.array(novelty_scores)
            
            # 筛选并转移
            count = self.keyevent_pool.add_from_difficulty_pool(
                difficulty_experiences,
                value_scores,
                novelty_scores
            )
            stats["difficulty_to_keyevent"] = count
        
        return stats
    
    def _compute_metrics(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """
        计算经验的所有指标
        
        Args:
            experience: 经验字典
        
        Returns:
            指标字典
        """
        metrics = {
            "TDError": self.td_metric.compute(experience),
            "ObstacleRisk": self.risk_metric.compute(experience),
            "Collaboration": self.collaboration_metric.compute(experience),
            "Novelty": self.novelty_metric.compute(experience),
        }
        return metrics
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        获取各池统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "coverage_pool_size": len(self.coverage_pool),
            "difficulty_pool_size": len(self.difficulty_pool),
            "keyevent_pool_size": len(self.keyevent_pool),
            "total_size": len(self.coverage_pool),
            "step_count": self.step_count,
        }
    
    def clear(self) -> None:
        """清空所有池"""
        super().clear()
        self.coverage_pool.clear()
        self.difficulty_pool.clear()
        self.keyevent_pool.clear()
        self.step_count = 0
        self.last_filter_step = 0
    
    def set_pool_weights(self, weights: List[float]) -> None:
        """
        设置各池采样权重
        
        Args:
            weights: 权重列表 [coverage, difficulty, keyevent]
        """
        self.sampler.set_weights(weights)
        self.multi_level_config.pool_weights = weights
