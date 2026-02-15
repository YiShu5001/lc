# envs.py
from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Any

import numpy as np


@dataclass
class EnvConfig:
    """
    环境配置（只放“环境相关”的参数）
    """
    env_id: str
    num_envs: int = 1              # 并行环境数（1 = 单环境）
    seed: int = 0
    capture_video: bool = False    # 是否录制视频（由具体实现决定）
    run_name: str = "exp"          # 日志/视频命名
    max_episode_steps: Optional[int] = None  # 可选：强制截断回合长度


class VectorEnvLike(ABC):
    """
    训练代码期望的“向量化环境”最小接口（不绑定 gymnasium，便于替换后端）
    """

    @property
    @abstractmethod
    def num_envs(self) -> int:
        """并行环境数量"""

    @property
    @abstractmethod
    def obs_shape(self) -> Tuple[int, ...]:
        """观测维度形状，例如 (4,)"""

    @property
    @abstractmethod
    def action_shape(self) -> Tuple[int, ...]:
        """
        动作张量形状。
        - 离散动作常用 () 或 (1,)
        - 连续动作常用 (act_dim,)
        """

    @property
    @abstractmethod
    def is_discrete(self) -> bool:
        """动作空间是否离散"""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """
        动作维度/动作数量：
        - 离散：动作个数 n
        - 连续：动作维度 act_dim
        """

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        重置环境
        Returns:
            obs: shape = (num_envs, *obs_shape)
        """

    @abstractmethod
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        与环境交互一步
        Args:
            actions:
              - 离散：shape = (num_envs,)  int
              - 连续：shape = (num_envs, action_dim) float
        Returns:
            next_obs: shape = (num_envs, *obs_shape)
            rewards:  shape = (num_envs,)
            dones:    shape = (num_envs,)  bool (True 表示该环境回合结束)
            infos:    任意信息字典（可包含 episode return/length 等）
        """


class EnvFactory(ABC):
    """
    环境工厂基类：负责根据 EnvConfig 构造 VectorEnvLike。
    你可以写 GymEnvFactory / PettingZooFactory / 自研仿真工厂等。
    """
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg

    @abstractmethod
    def build(self) -> VectorEnvLike:
        """构造并返回一个可用于训练的向量化环境对象"""
