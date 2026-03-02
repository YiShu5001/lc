"""
Gymnasium环境包装器
将gymnasium环境包装为VectorEnvLike接口
"""
from __future__ import annotations
from typing import Tuple, Optional, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium.vector import VectorEnv

from Gym_env.BaseEnv import VectorEnvLike


class GymnasiumVectorEnv(VectorEnvLike):
    """
    Gymnasium环境包装器
    
    将gymnasium环境（单环境或向量化环境）包装为VectorEnvLike接口
    """
    
    def __init__(self, env: gym.Env | VectorEnv):
        """
        Args:
            env: gymnasium环境实例（单环境或向量化环境）
        """
        self.env = env
        
        # 判断是否为向量化环境
        self._is_vectorized = isinstance(env, VectorEnv)
        
        if self._is_vectorized:
            self._num_envs = env.num_envs
        else:
            self._num_envs = 1
        
        # 获取观测和动作空间
        obs_space = env.observation_space
        action_space = env.action_space
        
        # 处理观测空间
        if isinstance(obs_space, gym.spaces.Box):
            self._obs_shape = obs_space.shape
        elif isinstance(obs_space, gym.spaces.Discrete):
            self._obs_shape = (1,)
        else:
            raise ValueError(f"不支持的观测空间类型: {type(obs_space)}")
        
        # 处理动作空间
        if isinstance(action_space, gym.spaces.Discrete):
            self._is_discrete = True
            self._action_dim = action_space.n
            self._action_shape = ()
        elif isinstance(action_space, gym.spaces.Box):
            self._is_discrete = False
            self._action_dim = action_space.shape[0]
            self._action_shape = action_space.shape
        else:
            raise ValueError(f"不支持的动作空间类型: {type(action_space)}")
    
    @property
    def num_envs(self) -> int:
        """并行环境数量"""
        return self._num_envs
    
    @property
    def obs_shape(self) -> Tuple[int, ...]:
        """观测维度形状"""
        return self._obs_shape
    
    @property
    def action_shape(self) -> Tuple[int, ...]:
        """动作张量形状"""
        return self._action_shape
    
    @property
    def is_discrete(self) -> bool:
        """动作空间是否离散"""
        return self._is_discrete
    
    @property
    def action_dim(self) -> int:
        """动作维度/动作数量"""
        return self._action_dim
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        重置环境
        
        Args:
            seed: 随机种子
        
        Returns:
            obs: shape = (num_envs, *obs_shape)
        """
        if seed is not None:
            if self._is_vectorized:
                # 向量化环境：为每个环境设置不同的种子
                seeds = [seed + i for i in range(self._num_envs)]
                obs, info = self.env.reset(seed=seeds)
            else:
                obs, info = self.env.reset(seed=seed)
        else:
            obs, info = self.env.reset()
        
        # 确保返回numpy数组
        if isinstance(obs, np.ndarray):
            # 如果是单环境且obs没有batch维度，添加batch维度
            if not self._is_vectorized and len(obs.shape) == len(self._obs_shape):
                obs = obs[np.newaxis, ...]
            return obs
        else:
            # 转换为numpy数组
            return np.array(obs)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        与环境交互一步
        
        Args:
            actions:
              - 离散：shape = (num_envs,)  int
              - 连续：shape = (num_envs, action_dim) float
        
        Returns:
            next_obs: shape = (num_envs, *obs_shape)
            rewards:  shape = (num_envs,)
            dones:    shape = (num_envs,)  bool
            infos:    信息字典
        """
        # 处理单环境的情况
        if not self._is_vectorized:
            # 移除batch维度
            if len(actions.shape) > len(self._action_shape):
                actions = actions[0]
        
        # 执行step
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        
        # 处理done信号（gymnasium使用terminated和truncated）
        dones = terminated | truncated
        
        # 确保返回numpy数组
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        
        if not isinstance(rewards, np.ndarray):
            rewards = np.array(rewards)
        
        if not isinstance(dones, np.ndarray):
            dones = np.array(dones)
        
        # 如果是单环境且obs没有batch维度，添加batch维度
        if not self._is_vectorized and len(obs.shape) == len(self._obs_shape):
            obs = obs[np.newaxis, ...]
            rewards = rewards[np.newaxis] if rewards.ndim == 0 else rewards
            dones = dones[np.newaxis] if dones.ndim == 0 else dones
        
        # 合并infos（如果是单环境，包装成列表）
        if not self._is_vectorized:
            infos = [infos]
        
        return obs, rewards, dones, infos
    
    def close(self):
        """关闭环境"""
        self.env.close()
