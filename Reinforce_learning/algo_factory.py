"""
算法工厂
根据算法名称和配置创建对应的算法实例
"""
from __future__ import annotations
from typing import Optional

from Reinforce_learning.Basealgos import BaseAlgo, AlgoConfig
from Reinforce_learning.RLg.PPO import PPOAlgo, PPOConfig
from Reinforce_learning.RLg.A2C import A2CAlgo, A2CConfig
from Reinforce_learning.RLg.SAC import SACAlgo, SACConfig
from Reinforce_learning.RLg.TD3 import TD3Algo, TD3Config
from Reinforce_learning.RLg.DDPG_refactored import DDPGAlgo, DDPGConfig
from Reinforce_learning.RLg.DQN import DQNAlgo, DQNConfig


def create_algo(
    algo_name: str,
    config: Optional[AlgoConfig] = None
) -> BaseAlgo:
    """
    创建算法实例
    
    Args:
        algo_name: 算法名称（"ppo", "a2c", "sac", "td3", "ddpg", "dqn"）
        config: 算法配置（如果为None，使用默认配置）
    
    Returns:
        算法实例
    
    Raises:
        ValueError: 如果算法名称不支持
    """
    algo_name = algo_name.lower()
    
    if algo_name == "ppo":
        cfg = config if config is not None else PPOConfig()
        if not isinstance(cfg, PPOConfig):
            cfg = PPOConfig(**cfg.__dict__) if hasattr(cfg, '__dict__') else PPOConfig()
        return PPOAlgo(cfg)
    
    elif algo_name == "a2c":
        cfg = config if config is not None else A2CConfig()
        if not isinstance(cfg, A2CConfig):
            cfg = A2CConfig(**cfg.__dict__) if hasattr(cfg, '__dict__') else A2CConfig()
        return A2CAlgo(cfg)
    
    elif algo_name == "sac":
        cfg = config if config is not None else SACConfig()
        if not isinstance(cfg, SACConfig):
            cfg = SACConfig(**cfg.__dict__) if hasattr(cfg, '__dict__') else SACConfig()
        return SACAlgo(cfg)
    
    elif algo_name == "td3":
        cfg = config if config is not None else TD3Config()
        if not isinstance(cfg, TD3Config):
            cfg = TD3Config(**cfg.__dict__) if hasattr(cfg, '__dict__') else TD3Config()
        return TD3Algo(cfg)
    
    elif algo_name == "ddpg":
        cfg = config if config is not None else DDPGConfig()
        if not isinstance(cfg, DDPGConfig):
            cfg = DDPGConfig(**cfg.__dict__) if hasattr(cfg, '__dict__') else DDPGConfig()
        return DDPGAlgo(cfg)
    
    elif algo_name == "dqn":
        cfg = config if config is not None else DQNConfig()
        if not isinstance(cfg, DQNConfig):
            cfg = DQNConfig(**cfg.__dict__) if hasattr(cfg, '__dict__') else DQNConfig()
        return DQNAlgo(cfg)
    
    else:
        raise ValueError(
            f"不支持的算法名称: {algo_name}. "
            f"支持的算法: ppo, a2c, sac, td3, ddpg, dqn"
        )


def get_algo_config_class(algo_name: str):
    """
    获取算法的配置类
    
    Args:
        algo_name: 算法名称
    
    Returns:
        配置类
    """
    algo_name = algo_name.lower()
    
    config_classes = {
        "ppo": PPOConfig,
        "a2c": A2CConfig,
        "sac": SACConfig,
        "td3": TD3Config,
        "ddpg": DDPGConfig,
        "dqn": DQNConfig,
    }
    
    if algo_name not in config_classes:
        raise ValueError(
            f"不支持的算法名称: {algo_name}. "
            f"支持的算法: {list(config_classes.keys())}"
        )
    
    return config_classes[algo_name]


def is_on_policy(algo_name: str) -> bool:
    """
    判断算法是否为on-policy
    
    Args:
        algo_name: 算法名称
    
    Returns:
        True if on-policy, False if off-policy
    """
    on_policy_algos = {"ppo", "a2c"}
    return algo_name.lower() in on_policy_algos


def is_off_policy(algo_name: str) -> bool:
    """
    判断算法是否为off-policy
    
    Args:
        algo_name: 算法名称
    
    Returns:
        True if off-policy, False if on-policy
    """
    return not is_on_policy(algo_name)
