"""
PyBullet Drones环境工厂
创建HoverAviary和MultiHoverAviary环境
"""
from __future__ import annotations
from typing import Optional

import gymnasium as gym
from gymnasium.vector import make_vec_env

from Gym_env.BaseEnv import EnvFactory, VectorEnvLike
from Gym_env.wrappers.GymnasiumWrapper import GymnasiumVectorEnv
from Gym_env.gym_pybullet_drones.envs.HoverAviary import HoverAviary
from Gym_env.gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from Gym_env.gym_pybullet_drones.utils.enums import ObservationType, ActionType


class PyBulletDronesFactory(EnvFactory):
    """
    PyBullet Drones环境工厂
    
    根据配置创建HoverAviary或MultiHoverAviary环境
    """
    
    def __init__(self, cfg):
        """
        Args:
            cfg: 环境配置，需要包含：
                - env_id: "HoverAviary-v0" 或 "MultiHoverAviary-v0"
                - num_envs: 并行环境数
                - num_drones: 无人机数量（MultiHoverAviary需要）
                - obs: ObservationType（默认kin）
                - act: ActionType（默认one_d_pid）
                - seed: 随机种子
                - gui: 是否显示GUI
        """
        super().__init__(cfg)
        self.cfg = cfg
    
    def build(self) -> VectorEnvLike:
        """
        创建环境实例
        
        Returns:
            VectorEnvLike: 包装后的环境
        """
        # 解析配置
        env_id = self.cfg.env_id if hasattr(self.cfg, 'env_id') else "HoverAviary-v0"
        num_envs = getattr(self.cfg, 'num_envs', 1)
        num_drones = getattr(self.cfg, 'num_drones', 1)
        obs_type = getattr(self.cfg, 'obs', ObservationType('kin'))
        act_type = getattr(self.cfg, 'act', ActionType('one_d_pid'))
        seed = getattr(self.cfg, 'seed', 0)
        gui = getattr(self.cfg, 'gui', False)
        
        # 判断是否为多智能体环境
        is_multiagent = "Multi" in env_id or num_drones > 1
        
        if is_multiagent:
            # 创建多智能体环境
            if num_envs > 1:
                # 向量化环境
                env = make_vec_env(
                    MultiHoverAviary,
                    env_kwargs=dict(
                        num_drones=num_drones,
                        obs=obs_type,
                        act=act_type,
                        gui=gui
                    ),
                    n_envs=num_envs,
                    seed=seed
                )
            else:
                # 单环境
                env = MultiHoverAviary(
                    num_drones=num_drones,
                    obs=obs_type,
                    act=act_type,
                    gui=gui
                )
        else:
            # 创建单智能体环境
            if num_envs > 1:
                # 向量化环境
                env = make_vec_env(
                    HoverAviary,
                    env_kwargs=dict(
                        obs=obs_type,
                        act=act_type,
                        gui=gui
                    ),
                    n_envs=num_envs,
                    seed=seed
                )
            else:
                # 单环境
                env = HoverAviary(
                    obs=obs_type,
                    act=act_type,
                    gui=gui
                )
        
        # 包装为VectorEnvLike
        return GymnasiumVectorEnv(env)
