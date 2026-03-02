"""
PyBullet Drones环境训练示例
演示如何使用框架训练无人机控制任务
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim

from Gym_env.BaseEnv import EnvConfig
from Gym_env.factories.PyBulletDronesFactory import PyBulletDronesFactory
from NN.BaseNN import ModelConfig
from NN.model_factory import create_model_from_env
from Reinforce_learning.algo_factory import create_algo, is_on_policy
from Reinforce_learning.RLg.PPO import PPOConfig
from Trainer.BaseTrainer import Trainer, TrainConfig
from Trainer.OffPolicyTrainer import OffPolicyTrainer
from Trainer.callbacks import CallbackList, ModelCheckpointCallback, LoggerCallback
from Gym_env.gym_pybullet_drones.utils.enums import ObservationType, ActionType


def train_pybullet_ppo():
    """使用PPO训练PyBullet环境"""
    print("=" * 50)
    print("PyBullet Drones - PPO训练示例")
    print("=" * 50)
    
    # 1. 创建环境
    env_cfg = EnvConfig(
        env_id="HoverAviary-v0",
        num_envs=1,
        seed=0
    )
    env_cfg.gui = False  # 训练时不显示GUI
    env_cfg.obs = ObservationType('kin')
    env_cfg.act = ActionType('one_d_pid')
    
    env_factory = PyBulletDronesFactory(env_cfg)
    envs = env_factory.build()
    
    print(f"[INFO] 环境: obs_shape={envs.obs_shape}, action_shape={envs.action_shape}")
    
    # 2. 创建模型
    model_cfg = ModelConfig(
        hidden_sizes=(128, 128),
        activation="tanh"
    )
    model = create_model_from_env(
        cfg=model_cfg,
        obs_shape=envs.obs_shape,
        action_shape=envs.action_shape,
        is_discrete=envs.is_discrete
    )
    
    print(f"[INFO] 模型: {type(model).__name__}")
    
    # 3. 创建算法
    algo_cfg = PPOConfig(
        learning_rate=3e-4,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01
    )
    algo = create_algo("ppo", algo_cfg)
    
    print(f"[INFO] 算法: PPO")
    
    # 4. 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # 5. 创建训练配置
    train_cfg = TrainConfig(
        total_timesteps=100000,  # 示例：10万步
        num_steps=128,
        device="cpu"
    )
    
    # 6. 创建回调
    callbacks = CallbackList([
        ModelCheckpointCallback(
            save_path="./results/pybullet_ppo",
            save_freq=10000
        ),
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


if __name__ == "__main__":
    train_pybullet_ppo()
