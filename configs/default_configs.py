"""
默认配置
"""
from dataclasses import dataclass
from typing import Optional

from Gym_env.BaseEnv import EnvConfig
from NN.BaseNN import ModelConfig
from Reinforce_learning.Basealgos import AlgoConfig
from Trainer.BaseTrainer import TrainConfig


@dataclass
class DefaultConfig:
    """默认配置集合"""
    env: EnvConfig
    model: ModelConfig
    algo: AlgoConfig
    train: TrainConfig


def get_default_config(
    algo_name: str = "ppo",
    env_id: str = "HoverAviary-v0"
) -> DefaultConfig:
    """
    获取默认配置
    
    Args:
        algo_name: 算法名称
        env_id: 环境ID
    
    Returns:
        默认配置
    """
    # 环境配置
    env_cfg = EnvConfig(
        env_id=env_id,
        num_envs=1,
        seed=0
    )
    
    # 模型配置
    model_cfg = ModelConfig(
        hidden_sizes=(64, 64),
        activation="tanh"
    )
    
    # 算法配置
    from Reinforce_learning.algo_factory import get_algo_config_class
    algo_config_class = get_algo_config_class(algo_name)
    algo_cfg = algo_config_class()
    
    # 训练配置
    train_cfg = TrainConfig(
        total_timesteps=1_000_000,
        num_steps=128,
        gamma=0.99,
        gae_lambda=0.95,
        device="cpu"
    )
    
    return DefaultConfig(
        env=env_cfg,
        model=model_cfg,
        algo=algo_cfg,
        train=train_cfg
    )
