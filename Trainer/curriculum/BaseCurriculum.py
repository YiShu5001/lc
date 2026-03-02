"""
课程学习基础类
定义课程学习的统一接口，支持难度递增、任务序列等课程学习策略
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

import numpy as np


@dataclass
class CurriculumConfig:
    """
    课程学习配置基类
    """
    pass


class BaseCurriculum(ABC):
    """
    课程学习基类
    
    定义了课程学习的统一接口，支持：
    - 难度递增课程
    - 任务序列课程
    - 自适应课程调整
    """
    
    def __init__(self, config: Optional[CurriculumConfig] = None):
        """
        Args:
            config: 课程学习配置
        """
        self.config = config or CurriculumConfig()
        self.current_level = 0
        self.episode_count = 0
        self.step_count = 0
    
    @abstractmethod
    def get_current_task(self) -> Dict[str, Any]:
        """
        获取当前任务配置（抽象方法）
        
        Returns:
            包含当前任务参数的字典（如难度、目标位置等）
        """
        pass
    
    @abstractmethod
    def update(self, performance: Dict[str, Any]) -> bool:
        """
        更新课程进度（抽象方法）
        
        Args:
            performance: 性能指标字典（如成功率、平均奖励等）
        
        Returns:
            是否成功进入下一阶段
        """
        pass
    
    def reset(self, episode: Optional[int] = None) -> None:
        """
        重置课程学习状态
        
        Args:
            episode: 当前episode编号
        """
        if episode is not None:
            self.episode_count = episode
        else:
            self.episode_count += 1
        self.step_count = 0
    
    def update_step(self, step: Optional[int] = None) -> None:
        """
        更新步数
        
        Args:
            step: 当前步数
        """
        if step is not None:
            self.step_count = step
        else:
            self.step_count += 1
    
    def is_complete(self) -> bool:
        """
        判断课程是否完成
        
        Returns:
            是否完成所有课程阶段
        """
        return False
    
    def get_progress(self) -> Dict[str, Any]:
        """
        获取课程学习进度信息
        
        Returns:
            包含进度信息的字典
        """
        return {
            "current_level": self.current_level,
            "episode_count": self.episode_count,
            "step_count": self.step_count
        }


class DifficultyCurriculum(BaseCurriculum):
    """
    难度递增课程学习
    
    根据性能逐步增加任务难度
    """
    
    @dataclass
    class Config(CurriculumConfig):
        difficulty_levels: List[float] = None      # 难度级别列表
        success_threshold: float = 0.8              # 进入下一阶段的成功率阈值
        min_episodes_per_level: int = 100          # 每个级别最少episode数
        performance_metric: str = "success_rate"    # 性能指标名称
    
    def __init__(self, config: Optional[Config] = None):
        """
        Args:
            config: 难度课程配置
        """
        super().__init__(config)
        self.config = config or self.Config()
        
        if self.config.difficulty_levels is None:
            # 默认难度级别：0.0 到 1.0，步长0.1
            self.config.difficulty_levels = [i * 0.1 for i in range(11)]
        
        self.max_level = len(self.config.difficulty_levels) - 1
        self.level_episode_count = 0
        self.level_success_count = 0
        self.level_performance_history = []
    
    def get_current_task(self) -> Dict[str, Any]:
        """
        获取当前难度级别的任务配置
        """
        difficulty = self.config.difficulty_levels[self.current_level]
        return {
            "difficulty": difficulty,
            "level": self.current_level,
            "max_level": self.max_level
        }
    
    def update(self, performance: Dict[str, Any]) -> bool:
        """
        根据性能更新课程进度
        
        Args:
            performance: 性能指标字典
        
        Returns:
            是否成功进入下一级别
        """
        self.level_episode_count += 1
        
        # 获取性能指标
        metric_value = performance.get(self.config.performance_metric, 0.0)
        self.level_performance_history.append(metric_value)
        
        # 判断是否成功（根据具体指标）
        if self.config.performance_metric == "success_rate":
            if performance.get("success", False):
                self.level_success_count += 1
        
        # 检查是否可以进入下一级别
        if self.current_level >= self.max_level:
            return False
        
        if self.level_episode_count < self.config.min_episodes_per_level:
            return False
        
        # 计算成功率
        success_rate = self.level_success_count / self.level_episode_count
        
        if success_rate >= self.config.success_threshold:
            # 进入下一级别
            self.current_level += 1
            self.level_episode_count = 0
            self.level_success_count = 0
            self.level_performance_history = []
            return True
        
        return False
    
    def is_complete(self) -> bool:
        """判断是否完成所有难度级别"""
        return self.current_level >= self.max_level
    
    def get_progress(self) -> Dict[str, Any]:
        """获取课程进度"""
        progress = super().get_progress()
        progress.update({
            "current_difficulty": self.config.difficulty_levels[self.current_level],
            "level_episode_count": self.level_episode_count,
            "level_success_rate": (
                self.level_success_count / self.level_episode_count
                if self.level_episode_count > 0 else 0.0
            ),
            "is_complete": self.is_complete()
        })
        return progress


class TaskCurriculum(BaseCurriculum):
    """
    任务序列课程学习
    
    按照预定义的任务序列逐步学习
    """
    
    @dataclass
    class Config(CurriculumConfig):
        task_sequence: List[Dict[str, Any]] = None  # 任务序列列表
        success_threshold: float = 0.8              # 进入下一任务的阈值
        min_episodes_per_task: int = 100            # 每个任务最少episode数
        performance_metric: str = "success_rate"    # 性能指标名称
    
    def __init__(self, config: Optional[Config] = None):
        """
        Args:
            config: 任务序列课程配置
        """
        super().__init__(config)
        self.config = config or self.Config()
        
        if self.config.task_sequence is None:
            # 默认空序列
            self.config.task_sequence = []
        
        self.max_task = len(self.config.task_sequence) - 1
        self.task_episode_count = 0
        self.task_success_count = 0
        self.task_performance_history = []
    
    def get_current_task(self) -> Dict[str, Any]:
        """
        获取当前任务配置
        """
        if self.current_level < len(self.config.task_sequence):
            task_config = self.config.task_sequence[self.current_level].copy()
            task_config["task_id"] = self.current_level
            task_config["total_tasks"] = len(self.config.task_sequence)
            return task_config
        else:
            return {
                "task_id": self.current_level,
                "total_tasks": len(self.config.task_sequence)
            }
    
    def update(self, performance: Dict[str, Any]) -> bool:
        """
        根据性能更新课程进度
        
        Args:
            performance: 性能指标字典
        
        Returns:
            是否成功进入下一任务
        """
        self.task_episode_count += 1
        
        # 获取性能指标
        metric_value = performance.get(self.config.performance_metric, 0.0)
        self.task_performance_history.append(metric_value)
        
        # 判断是否成功
        if self.config.performance_metric == "success_rate":
            if performance.get("success", False):
                self.task_success_count += 1
        
        # 检查是否可以进入下一任务
        if self.current_level >= self.max_task:
            return False
        
        if self.task_episode_count < self.config.min_episodes_per_task:
            return False
        
        # 计算成功率
        success_rate = self.task_success_count / self.task_episode_count
        
        if success_rate >= self.config.success_threshold:
            # 进入下一任务
            self.current_level += 1
            self.task_episode_count = 0
            self.task_success_count = 0
            self.task_performance_history = []
            return True
        
        return False
    
    def is_complete(self) -> bool:
        """判断是否完成所有任务"""
        return self.current_level > self.max_task
    
    def get_progress(self) -> Dict[str, Any]:
        """获取课程进度"""
        progress = super().get_progress()
        progress.update({
            "current_task": self.get_current_task(),
            "task_episode_count": self.task_episode_count,
            "task_success_rate": (
                self.task_success_count / self.task_episode_count
                if self.task_episode_count > 0 else 0.0
            ),
            "is_complete": self.is_complete()
        })
        return progress
