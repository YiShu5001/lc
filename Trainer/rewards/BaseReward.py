"""
奖励函数基础类
定义奖励函数的统一接口，支持奖励塑形、课程学习等
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RewardConfig:
    """
    奖励函数配置基类
    """
    pass


class BaseRewardFunction(ABC):
    """
    奖励函数基类
    
    定义了奖励函数的统一接口，支持：
    - 基础奖励计算
    - 奖励塑形
    - 课程学习
    - 多目标奖励
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        """
        Args:
            config: 奖励函数配置
        """
        self.config = config or RewardConfig()
        self.episode_count = 0
        self.step_count = 0
    
    @abstractmethod
    def compute(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
        next_obs: Union[np.ndarray, torch.Tensor],
        done: Union[bool, np.ndarray],
        info: Optional[Dict[str, Any]] = None
    ) -> Union[float, np.ndarray]:
        """
        计算奖励（抽象方法）
        
        Args:
            obs: 当前观测
            action: 执行的动作
            next_obs: 下一时刻观测
            done: 是否结束
            info: 额外信息字典
        
        Returns:
            奖励值，float 或 (num_envs,) 的数组
        """
        pass
    
    def reset(self, episode: Optional[int] = None) -> None:
        """
        重置奖励函数状态（每个episode开始时调用）
        
        Args:
            episode: 当前episode编号（如果为None，使用内部计数器）
        """
        if episode is not None:
            self.episode_count = episode
        else:
            self.episode_count += 1
        self.step_count = 0
    
    def update_step(self, step: Optional[int] = None) -> None:
        """
        更新步数计数器
        
        Args:
            step: 当前步数（如果为None，使用内部计数器）
        """
        if step is not None:
            self.step_count = step
        else:
            self.step_count += 1
    
    def get_reward_info(self) -> Dict[str, Any]:
        """
        获取奖励相关信息（用于日志记录）
        
        Returns:
            包含奖励相关信息的字典
        """
        return {
            "episode": self.episode_count,
            "step": self.step_count
        }


class ShapedReward(BaseRewardFunction):
    """
    奖励塑形函数
    
    在基础奖励上添加塑形项，引导智能体学习
    """
    
    @dataclass
    class Config(RewardConfig):
        base_reward_weight: float = 1.0      # 基础奖励权重
        shaping_weight: float = 0.1         # 塑形奖励权重
        use_potential_based: bool = True     # 是否使用基于势能的塑形
    
    def __init__(
        self,
        base_reward_fn: BaseRewardFunction,
        shaping_fn: Optional[BaseRewardFunction] = None,
        config: Optional[Config] = None
    ):
        """
        Args:
            base_reward_fn: 基础奖励函数
            shaping_fn: 塑形奖励函数（可选）
            config: ShapedReward配置
        """
        super().__init__(config)
        self.config = config or self.Config()
        self.base_reward_fn = base_reward_fn
        self.shaping_fn = shaping_fn
        self.prev_potential = None
    
    def compute(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
        next_obs: Union[np.ndarray, torch.Tensor],
        done: Union[bool, np.ndarray],
        info: Optional[Dict[str, Any]] = None
    ) -> Union[float, np.ndarray]:
        """
        计算塑形奖励
        
        如果使用基于势能的塑形：
        reward = base_reward + shaping_weight * (potential(next_state) - potential(state))
        否则：
        reward = base_reward_weight * base_reward + shaping_weight * shaping_reward
        """
        # 计算基础奖励
        base_reward = self.base_reward_fn.compute(obs, action, next_obs, done, info)
        
        if self.shaping_fn is None:
            return base_reward * self.config.base_reward_weight
        
        if self.config.use_potential_based:
            # 基于势能的塑形
            current_potential = self._compute_potential(obs)
            next_potential = self._compute_potential(next_obs)
            
            # 处理batch维度
            if isinstance(current_potential, np.ndarray) and len(current_potential.shape) > 0:
                potential_diff = next_potential - current_potential
            else:
                potential_diff = next_potential - current_potential
            
            shaping_reward = self.config.shaping_weight * potential_diff
            reward = base_reward + shaping_reward
        else:
            # 直接相加
            shaping_reward = self.shaping_fn.compute(obs, action, next_obs, done, info)
            reward = (
                self.config.base_reward_weight * base_reward
                + self.config.shaping_weight * shaping_reward
            )
        
        return reward
    
    def _compute_potential(self, obs: Union[np.ndarray, torch.Tensor]) -> Union[float, np.ndarray]:
        """
        计算势能函数值（需要子类实现或使用shaping_fn）
        
        这里使用shaping_fn的奖励作为势能
        """
        if self.shaping_fn is None:
            return 0.0
        
        # 使用shaping_fn计算势能（假设它返回的是势能值）
        # 这里简化处理，实际应该根据具体问题设计势能函数
        dummy_action = np.zeros_like(obs) if isinstance(obs, np.ndarray) else torch.zeros_like(obs)
        dummy_done = False
        return self.shaping_fn.compute(obs, dummy_action, obs, dummy_done)
    
    def reset(self, episode: Optional[int] = None) -> None:
        """重置状态"""
        super().reset(episode)
        self.base_reward_fn.reset(episode)
        if self.shaping_fn is not None:
            self.shaping_fn.reset(episode)
        self.prev_potential = None


class CurriculumReward(BaseRewardFunction):
    """
    课程学习奖励函数
    
    根据训练进度动态调整奖励函数，实现从简单到复杂的课程学习
    """
    
    @dataclass
    class Config(RewardConfig):
        curriculum_stages: list = None      # 课程阶段列表
        stage_threshold: float = 0.8        # 进入下一阶段的阈值（成功率）
        min_episodes_per_stage: int = 100   # 每个阶段最少episode数
    
    def __init__(
        self,
        reward_functions: list[BaseRewardFunction],
        config: Optional[Config] = None
    ):
        """
        Args:
            reward_functions: 不同阶段的奖励函数列表
            config: CurriculumReward配置
        """
        super().__init__(config)
        self.config = config or self.Config()
        self.reward_functions = reward_functions
        self.current_stage = 0
        self.stage_episode_count = 0
        self.stage_success_count = 0
        
        if self.config.curriculum_stages is None:
            self.config.curriculum_stages = list(range(len(reward_functions)))
    
    def compute(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
        next_obs: Union[np.ndarray, torch.Tensor],
        done: Union[bool, np.ndarray],
        info: Optional[Dict[str, Any]] = None
    ) -> Union[float, np.ndarray]:
        """
        使用当前阶段的奖励函数计算奖励
        """
        current_fn = self.reward_functions[self.current_stage]
        reward = current_fn.compute(obs, action, next_obs, done, info)
        
        # 更新成功率统计（如果done为True）
        if isinstance(done, bool):
            if done:
                self.stage_episode_count += 1
                # 假设info中包含success信息，或者根据reward判断
                if info and info.get("success", False):
                    self.stage_success_count += 1
        elif isinstance(done, np.ndarray):
            # batch处理
            done_count = np.sum(done)
            self.stage_episode_count += done_count
            if info and "success" in info:
                success_count = np.sum(info["success"])
                self.stage_success_count += success_count
        
        return reward
    
    def update_stage(self) -> bool:
        """
        检查是否需要进入下一阶段
        
        Returns:
            是否成功进入下一阶段
        """
        if self.current_stage >= len(self.reward_functions) - 1:
            return False
        
        if self.stage_episode_count < self.config.min_episodes_per_stage:
            return False
        
        success_rate = self.stage_success_count / self.stage_episode_count
        if success_rate >= self.config.stage_threshold:
            self.current_stage += 1
            self.stage_episode_count = 0
            self.stage_success_count = 0
            return True
        
        return False
    
    def reset(self, episode: Optional[int] = None) -> None:
        """重置状态"""
        super().reset(episode)
        for fn in self.reward_functions:
            fn.reset(episode)
    
    def get_reward_info(self) -> Dict[str, Any]:
        """获取奖励信息，包括当前阶段"""
        info = super().get_reward_info()
        info.update({
            "current_stage": self.current_stage,
            "stage_episode_count": self.stage_episode_count,
            "stage_success_rate": (
                self.stage_success_count / self.stage_episode_count
                if self.stage_episode_count > 0 else 0.0
            )
        })
        return info


class MultiObjectiveReward(BaseRewardFunction):
    """
    多目标奖励函数
    
    组合多个奖励项，支持加权求和或Pareto优化
    """
    
    @dataclass
    class Config(RewardConfig):
        weights: list[float] = None         # 各目标权重
        normalize: bool = False             # 是否归一化
        method: str = "weighted_sum"        # 组合方法：weighted_sum, pareto
    
    def __init__(
        self,
        reward_functions: list[BaseRewardFunction],
        config: Optional[Config] = None
    ):
        """
        Args:
            reward_functions: 各目标的奖励函数列表
            config: MultiObjectiveReward配置
        """
        super().__init__(config)
        self.config = config or self.Config()
        self.reward_functions = reward_functions
        
        if self.config.weights is None:
            # 默认均匀权重
            self.config.weights = [1.0 / len(reward_functions)] * len(reward_functions)
        
        if len(self.config.weights) != len(reward_functions):
            raise ValueError("权重数量必须与奖励函数数量相同")
    
    def compute(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
        next_obs: Union[np.ndarray, torch.Tensor],
        done: Union[bool, np.ndarray],
        info: Optional[Dict[str, Any]] = None
    ) -> Union[float, np.ndarray]:
        """
        计算多目标奖励
        """
        rewards = []
        for fn in self.reward_functions:
            reward = fn.compute(obs, action, next_obs, done, info)
            rewards.append(reward)
        
        # 转换为numpy数组便于处理
        rewards = np.array(rewards)
        weights = np.array(self.config.weights)
        
        # 归一化（如果需要）
        if self.config.normalize:
            rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8)
        
        # 组合方法
        if self.config.method == "weighted_sum":
            combined_reward = np.sum(weights * rewards)
        elif self.config.method == "pareto":
            # Pareto优化：返回所有目标的向量（简化实现）
            combined_reward = rewards
        else:
            raise ValueError(f"不支持的组合方法: {self.config.method}")
        
        return combined_reward
    
    def reset(self, episode: Optional[int] = None) -> None:
        """重置状态"""
        super().reset(episode)
        for fn in self.reward_functions:
            fn.reset(episode)
    
    def get_reward_info(self) -> Dict[str, Any]:
        """获取各目标的奖励信息"""
        info = super().get_reward_info()
        info["num_objectives"] = len(self.reward_functions)
        info["weights"] = self.config.weights
        return info
