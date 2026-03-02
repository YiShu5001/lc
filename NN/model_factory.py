"""
模型工厂
根据环境类型和配置自动创建合适的模型
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch

from NN.BaseNN import BaseRLModel, ModelConfig
from NN.ContinuousActorCritic import ContinuousActorCritic
from NN.DiscreteActorCritic import DiscreteActorCritic


def create_model(
    cfg: ModelConfig,
    obs_dim: int,
    action_dim: int,
    is_discrete: bool,
    action_high: Optional[torch.Tensor] = None,
    action_low: Optional[torch.Tensor] = None,
) -> BaseRLModel:
    """
    创建模型实例
    
    Args:
        cfg: 模型配置
        obs_dim: 观测维度
        action_dim: 动作维度（连续）或动作数量（离散）
        is_discrete: 是否为离散动作空间
        action_high: 动作上界（仅连续动作需要）
        action_low: 动作下界（仅连续动作需要）
    
    Returns:
        模型实例
    """
    if is_discrete:
        return DiscreteActorCritic(
            cfg=cfg,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
    else:
        return ContinuousActorCritic(
            cfg=cfg,
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_high=action_high,
            action_low=action_low,
        )


def create_model_from_env(
    cfg: ModelConfig,
    obs_shape: Tuple[int, ...],
    action_shape: Tuple[int, ...],
    is_discrete: bool,
    action_space=None,  # gymnasium Space对象（可选）
) -> BaseRLModel:
    """
    从环境信息创建模型
    
    Args:
        cfg: 模型配置
        obs_shape: 观测形状，例如 (4,) 或 (84, 84, 3)
        action_shape: 动作形状
        is_discrete: 是否为离散动作空间
        action_space: gymnasium动作空间对象（可选，用于获取action bounds）
    
    Returns:
        模型实例
    """
    # 计算观测维度（展平）
    if len(obs_shape) == 1:
        obs_dim = obs_shape[0]
    else:
        obs_dim = 1
        for dim in obs_shape:
            obs_dim *= dim
    
    # 计算动作维度
    if is_discrete:
        # 离散：action_shape通常是()或(1,)，action_dim是动作数量
        if len(action_shape) == 0:
            # 需要从action_space获取，这里假设action_dim已提供
            action_dim = action_shape[0] if len(action_shape) > 0 else 2
        else:
            action_dim = action_shape[0]
        
        return DiscreteActorCritic(
            cfg=cfg,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
    else:
        # 连续：action_dim是动作维度
        action_dim = action_shape[0] if len(action_shape) > 0 else 1
        
        # 获取动作边界
        action_high = None
        action_low = None
        if action_space is not None:
            if hasattr(action_space, 'high'):
                action_high = torch.tensor(action_space.high, dtype=torch.float32)
            if hasattr(action_space, 'low'):
                action_low = torch.tensor(action_space.low, dtype=torch.float32)
        
        return ContinuousActorCritic(
            cfg=cfg,
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_high=action_high,
            action_low=action_low,
        )
