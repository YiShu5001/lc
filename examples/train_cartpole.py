"""
CartPole环境训练示例（简单测试）
演示如何使用框架训练经典RL环境
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import torch
import torch.optim as optim

from Gym_env.wrappers.GymnasiumWrapper import GymnasiumVectorEnv
from NN.BaseNN import ModelConfig
from NN.model_factory import create_model_from_env
from Reinforce_learning.algo_factory import create_algo
from Reinforce_learning.RLg.PPO import PPOConfig
from Trainer.BaseTrainer import Trainer, TrainConfig
from Trainer.callbacks import CallbackList, ModelCheckpointCallback, LoggerCallback


def train_cartpole():
    """使用PPO训练CartPole环境"""
    print("=" * 50)
    print("CartPole - PPO训练示例")
    print("=" * 50)
    
    # 1. 创建环境
    env = gym.make("CartPole-v1")
    envs = GymnasiumVectorEnv(env)
    
    print(f"[INFO] 环境: obs_shape={envs.obs_shape}, action_dim={envs.action_dim}")
    
    # 2. 创建模型
    model_cfg = ModelConfig(
        hidden_sizes=(64, 64),
        activation="relu"
    )
    model = create_model_from_env(
        cfg=model_cfg,
        obs_shape=envs.obs_shape,
        action_shape=envs.action_shape,
        is_discrete=envs.is_discrete,
        action_space=env.action_space
    )
    
    print(f"[INFO] 模型: {type(model).__name__}")
    
    # 3. 创建算法
    algo_cfg = PPOConfig(learning_rate=3e-4)
    algo = create_algo("ppo", algo_cfg)
    
    # 4. 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # 5. 创建训练配置
    train_cfg = TrainConfig(
        total_timesteps=50000,
        num_steps=128,
        device="cpu"
    )
    
    # 6. 创建回调
    callbacks = CallbackList([
        LoggerCallback(log_interval=1000)
    ])
    callbacks.set_model(model)
    
    # 7. 创建训练器并训练
    trainer = Trainer(
        envs=envs,
        model=model,
        algo=algo,
        optimizer=optimizer,
        cfg=train_cfg
    )
    
    print("[INFO] 开始训练...")
    trainer.train()
    print("[INFO] 训练完成")
    
    env.close()


if __name__ == "__main__":
    train_cartpole()
