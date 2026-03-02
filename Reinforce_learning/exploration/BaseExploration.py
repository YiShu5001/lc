"""
探索策略基础类
定义探索策略的统一接口，支持离散和连续动作空间
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Union
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ExplorationConfig:
    """
    探索策略配置基类
    """
    pass


class BaseExploration(ABC):
    """
    探索策略基类
    
    定义了探索策略的统一接口，支持：
    - 离散动作空间：epsilon-greedy, softmax, boltzmann等
    - 连续动作空间：噪声探索（高斯噪声、OU噪声等）
    """
    
    def __init__(self, config: Optional[ExplorationConfig] = None):
        """
        Args:
            config: 探索策略配置
        """
        self.config = config or ExplorationConfig()
        self.step_count = 0
    
    @abstractmethod
    def select_action(
        self,
        action_values: Union[torch.Tensor, np.ndarray],
        deterministic: bool = False
    ) -> Union[int, np.ndarray]:
        """
        选择动作（抽象方法）
        
        Args:
            action_values: 动作值
                - 离散动作：Q值或logits，shape = (action_dim,) 或 (batch, action_dim)
                - 连续动作：确定性动作，shape = (action_dim,) 或 (batch, action_dim)
            deterministic: 是否使用确定性策略（不探索）
        
        Returns:
            - 离散动作：动作索引 int 或 (batch,) 的数组
            - 连续动作：动作数组，shape = (action_dim,) 或 (batch, action_dim)
        """
        pass
    
    def update(self, step: Optional[int] = None) -> None:
        """
        更新探索参数（如epsilon衰减）
        
        Args:
            step: 当前步数（如果为None，使用内部计数器）
        """
        if step is not None:
            self.step_count = step
        else:
            self.step_count += 1
    
    def reset(self) -> None:
        """重置探索策略状态"""
        self.step_count = 0
    
    def get_exploration_rate(self) -> float:
        """
        获取当前探索率（用于日志记录）
        
        Returns:
            当前探索率（如epsilon值）
        """
        return 0.0


class EpsilonGreedy(BaseExploration):
    """
    Epsilon-Greedy探索策略
    
    以epsilon概率随机选择动作，以(1-epsilon)概率选择最优动作
    适用于离散动作空间
    """
    
    @dataclass
    class Config(ExplorationConfig):
        epsilon_start: float = 1.0      # 初始探索率
        epsilon_end: float = 0.01       # 最终探索率
        epsilon_decay_steps: int = 10000  # 衰减步数
        decay_type: str = "linear"      # 衰减类型：linear, exponential
    
    def __init__(self, config: Optional[Config] = None):
        """
        Args:
            config: EpsilonGreedy配置
        """
        super().__init__(config)
        self.config = config or self.Config()
        self.epsilon = self.config.epsilon_start
        
        if self.config.decay_type == "linear":
            self.epsilon_decay = (
                (self.config.epsilon_start - self.config.epsilon_end) 
                / self.config.epsilon_decay_steps
            )
        elif self.config.decay_type == "exponential":
            self.epsilon_decay = (
                (self.config.epsilon_end / self.config.epsilon_start) 
                ** (1.0 / self.config.epsilon_decay_steps)
            )
        else:
            raise ValueError(f"不支持的衰减类型: {self.config.decay_type}")
    
    def select_action(
        self,
        action_values: Union[torch.Tensor, np.ndarray],
        deterministic: bool = False
    ) -> Union[int, np.ndarray]:
        """
        使用epsilon-greedy策略选择动作
        
        Args:
            action_values: Q值或logits，shape = (action_dim,) 或 (batch, action_dim)
            deterministic: 如果为True，总是选择最优动作
        
        Returns:
            动作索引，int 或 (batch,) 的数组
        """
        if deterministic:
            epsilon = 0.0
        else:
            epsilon = self.epsilon
        
        # 转换为numpy数组
        if isinstance(action_values, torch.Tensor):
            action_values = action_values.detach().cpu().numpy()
        
        # 处理batch维度
        is_batch = len(action_values.shape) > 1
        if not is_batch:
            action_values = action_values.reshape(1, -1)
        
        batch_size = action_values.shape[0]
        actions = np.zeros(batch_size, dtype=np.int64)
        
        for i in range(batch_size):
            if np.random.random() < epsilon:
                # 随机探索
                actions[i] = np.random.randint(0, action_values.shape[1])
            else:
                # 利用：选择Q值最大的动作
                actions[i] = np.argmax(action_values[i])
        
        if not is_batch:
            return int(actions[0])
        return actions
    
    def update(self, step: Optional[int] = None) -> None:
        """更新epsilon值"""
        super().update(step)
        
        if self.config.decay_type == "linear":
            self.epsilon = max(
                self.config.epsilon_end,
                self.config.epsilon_start - self.epsilon_decay * self.step_count
            )
        elif self.config.decay_type == "exponential":
            self.epsilon = max(
                self.config.epsilon_end,
                self.config.epsilon_start * (self.epsilon_decay ** self.step_count)
            )
    
    def reset(self) -> None:
        """重置epsilon到初始值"""
        super().reset()
        self.epsilon = self.config.epsilon_start
    
    def get_exploration_rate(self) -> float:
        """获取当前epsilon值"""
        return self.epsilon


class SoftmaxExploration(BaseExploration):
    """
    Softmax探索策略
    
    使用softmax分布根据Q值选择动作，温度参数控制探索程度
    适用于离散动作空间
    """
    
    @dataclass
    class Config(ExplorationConfig):
        temperature_start: float = 1.0      # 初始温度
        temperature_end: float = 0.1       # 最终温度
        temperature_decay_steps: int = 10000  # 衰减步数
        decay_type: str = "linear"         # 衰减类型
    
    def __init__(self, config: Optional[Config] = None):
        """
        Args:
            config: SoftmaxExploration配置
        """
        super().__init__(config)
        self.config = config or self.Config()
        self.temperature = self.config.temperature_start
        
        if self.config.decay_type == "linear":
            self.temp_decay = (
                (self.config.temperature_start - self.config.temperature_end)
                / self.config.temperature_decay_steps
            )
        elif self.config.decay_type == "exponential":
            self.temp_decay = (
                (self.config.temperature_end / self.config.temperature_start)
                ** (1.0 / self.config.temperature_decay_steps)
            )
        else:
            raise ValueError(f"不支持的衰减类型: {self.config.decay_type}")
    
    def select_action(
        self,
        action_values: Union[torch.Tensor, np.ndarray],
        deterministic: bool = False
    ) -> Union[int, np.ndarray]:
        """
        使用softmax策略选择动作
        
        Args:
            action_values: Q值或logits，shape = (action_dim,) 或 (batch, action_dim)
            deterministic: 如果为True，选择概率最大的动作
        
        Returns:
            动作索引，int 或 (batch,) 的数组
        """
        # 转换为numpy数组
        if isinstance(action_values, torch.Tensor):
            action_values = action_values.detach().cpu().numpy()
        
        # 处理batch维度
        is_batch = len(action_values.shape) > 1
        if not is_batch:
            action_values = action_values.reshape(1, -1)
        
        if deterministic:
            # 选择概率最大的动作
            actions = np.argmax(action_values, axis=1)
        else:
            # 计算softmax概率
            exp_values = np.exp(action_values / self.temperature)
            probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            
            # 根据概率采样
            batch_size = action_values.shape[0]
            actions = np.zeros(batch_size, dtype=np.int64)
            for i in range(batch_size):
                actions[i] = np.random.choice(
                    action_values.shape[1],
                    p=probs[i]
                )
        
        if not is_batch:
            return int(actions[0])
        return actions
    
    def update(self, step: Optional[int] = None) -> None:
        """更新温度参数"""
        super().update(step)
        
        if self.config.decay_type == "linear":
            self.temperature = max(
                self.config.temperature_end,
                self.config.temperature_start - self.temp_decay * self.step_count
            )
        elif self.config.decay_type == "exponential":
            self.temperature = max(
                self.config.temperature_end,
                self.config.temperature_start * (self.temp_decay ** self.step_count)
            )
    
    def reset(self) -> None:
        """重置温度到初始值"""
        super().reset()
        self.temperature = self.config.temperature_start
    
    def get_exploration_rate(self) -> float:
        """获取当前温度值"""
        return self.temperature


class BoltzmannExploration(SoftmaxExploration):
    """
    Boltzmann探索策略（Softmax的别名）
    
    与SoftmaxExploration相同，只是名称不同
    """
    pass


class NoiseExploration(BaseExploration):
    """
    噪声探索策略（高斯噪声）
    
    在确定性动作上添加高斯噪声，适用于连续动作空间
    """
    
    @dataclass
    class Config(ExplorationConfig):
        noise_std_start: float = 0.2      # 初始噪声标准差
        noise_std_end: float = 0.05       # 最终噪声标准差
        noise_decay_steps: int = 10000    # 衰减步数
        action_low: Optional[np.ndarray] = None  # 动作下界
        action_high: Optional[np.ndarray] = None  # 动作上界
        decay_type: str = "linear"       # 衰减类型
    
    def __init__(self, config: Optional[Config] = None):
        """
        Args:
            config: NoiseExploration配置
        """
        super().__init__(config)
        self.config = config or self.Config()
        self.noise_std = self.config.noise_std_start
        
        if self.config.decay_type == "linear":
            self.noise_decay = (
                (self.config.noise_std_start - self.config.noise_std_end)
                / self.config.noise_decay_steps
            )
        elif self.config.decay_type == "exponential":
            self.noise_decay = (
                (self.config.noise_std_end / self.config.noise_std_start)
                ** (1.0 / self.config.noise_decay_steps)
            )
        else:
            raise ValueError(f"不支持的衰减类型: {self.config.decay_type}")
    
    def select_action(
        self,
        action_values: Union[torch.Tensor, np.ndarray],
        deterministic: bool = False
    ) -> np.ndarray:
        """
        在确定性动作上添加噪声
        
        Args:
            action_values: 确定性动作，shape = (action_dim,) 或 (batch, action_dim)
            deterministic: 如果为True，不添加噪声
        
        Returns:
            动作数组，shape = (action_dim,) 或 (batch, action_dim)
        """
        # 转换为numpy数组
        if isinstance(action_values, torch.Tensor):
            action_values = action_values.detach().cpu().numpy()
        
        # 处理batch维度
        is_batch = len(action_values.shape) > 1
        if not is_batch:
            action_values = action_values.reshape(1, -1)
        
        if deterministic:
            actions = action_values.copy()
        else:
            # 添加高斯噪声
            noise = np.random.normal(0, self.noise_std, size=action_values.shape)
            actions = action_values + noise
            
            # 裁剪到动作空间范围
            if self.config.action_low is not None:
                actions = np.maximum(actions, self.config.action_low)
            if self.config.action_high is not None:
                actions = np.minimum(actions, self.config.action_high)
        
        if not is_batch:
            return actions[0]
        return actions
    
    def update(self, step: Optional[int] = None) -> None:
        """更新噪声标准差"""
        super().update(step)
        
        if self.config.decay_type == "linear":
            self.noise_std = max(
                self.config.noise_std_end,
                self.config.noise_std_start - self.noise_decay * self.step_count
            )
        elif self.config.decay_type == "exponential":
            self.noise_std = max(
                self.config.noise_std_end,
                self.config.noise_std_start * (self.noise_decay ** self.step_count)
            )
    
    def reset(self) -> None:
        """重置噪声标准差到初始值"""
        super().reset()
        self.noise_std = self.config.noise_std_start
    
    def get_exploration_rate(self) -> float:
        """获取当前噪声标准差"""
        return self.noise_std


class OUNoiseExploration(BaseExploration):
    """
    Ornstein-Uhlenbeck噪声探索策略
    
    使用OU过程生成时间相关的噪声，适用于连续控制任务
    """
    
    @dataclass
    class Config(ExplorationConfig):
        mu: float = 0.0                    # 均值
        theta: float = 0.15                # 回归速度
        sigma_start: float = 0.2           # 初始波动率
        sigma_end: float = 0.05            # 最终波动率
        sigma_decay_steps: int = 10000     # 衰减步数
        action_dim: Optional[int] = None   # 动作维度
        action_low: Optional[np.ndarray] = None  # 动作下界
        action_high: Optional[np.ndarray] = None  # 动作上界
    
    def __init__(self, config: Optional[Config] = None):
        """
        Args:
            config: OUNoiseExploration配置
        """
        super().__init__(config)
        self.config = config or self.Config()
        self.sigma = self.config.sigma_start
        
        if self.config.action_dim is not None:
            self.state = np.ones(self.config.action_dim) * self.config.mu
        else:
            self.state = None
        
        self.sigma_decay = (
            (self.config.sigma_start - self.config.sigma_end)
            / self.config.sigma_decay_steps
        )
    
    def select_action(
        self,
        action_values: Union[torch.Tensor, np.ndarray],
        deterministic: bool = False
    ) -> np.ndarray:
        """
        在确定性动作上添加OU噪声
        
        Args:
            action_values: 确定性动作，shape = (action_dim,) 或 (batch, action_dim)
            deterministic: 如果为True，不添加噪声
        
        Returns:
            动作数组，shape = (action_dim,) 或 (batch, action_dim)
        """
        # 转换为numpy数组
        if isinstance(action_values, torch.Tensor):
            action_values = action_values.detach().cpu().numpy()
        
        # 处理batch维度
        is_batch = len(action_values.shape) > 1
        if not is_batch:
            action_values = action_values.reshape(1, -1)
        
        if deterministic:
            actions = action_values.copy()
        else:
            # 初始化状态（如果尚未初始化）
            if self.state is None:
                self.state = np.ones(action_values.shape[1]) * self.config.mu
            
            # 更新OU过程状态
            self.state = (
                self.state 
                + self.config.theta * (self.config.mu - self.state)
                + self.sigma * np.random.normal(size=self.state.shape)
            )
            
            # 添加噪声到动作
            actions = action_values + self.state.reshape(1, -1)
            
            # 裁剪到动作空间范围
            if self.config.action_low is not None:
                actions = np.maximum(actions, self.config.action_low)
            if self.config.action_high is not None:
                actions = np.minimum(actions, self.config.action_high)
        
        if not is_batch:
            return actions[0]
        return actions
    
    def update(self, step: Optional[int] = None) -> None:
        """更新噪声参数"""
        super().update(step)
        self.sigma = max(
            self.config.sigma_end,
            self.config.sigma_start - self.sigma_decay * self.step_count
        )
    
    def reset(self) -> None:
        """重置OU过程状态和噪声参数"""
        super().reset()
        self.sigma = self.config.sigma_start
        if self.state is not None:
            self.state = np.ones_like(self.state) * self.config.mu
    
    def get_exploration_rate(self) -> float:
        """获取当前噪声标准差"""
        return self.sigma
