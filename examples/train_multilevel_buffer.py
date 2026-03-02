"""
多层经验池使用示例
演示如何使用MultiLevelBuffer进行训练
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
import numpy as np

from Gym_env.BaseEnv import EnvConfig
from Gym_env.factories.PyBulletDronesFactory import PyBulletDronesFactory
from NN.BaseNN import ModelConfig
from NN.model_factory import create_model_from_env
from Reinforce_learning.algo_factory import create_algo
from Reinforce_learning.RLg.DDPG_refactored import DDPGConfig
from Trainer.OffPolicyTrainer import OffPolicyTrainer
from Trainer.BaseTrainer import TrainConfig
from Reinforce_learning.buffers.multi_level.MultiLevelBuffer import MultiLevelBuffer
from Reinforce_learning.buffers.MultiLevelBufferConfig import MultiLevelBufferConfig
from Gym_env.gym_pybullet_drones.utils.enums import ObservationType, ActionType


def train_with_multilevel_buffer():
    """使用多层经验池训练"""
    print("=" * 50)
    print("多层经验池训练示例")
    print("=" * 50)
    
    # 1. 创建环境
    env_cfg = EnvConfig(
        env_id="HoverAviary-v0",
        num_envs=1,
        seed=0
    )
    env_cfg.gui = False
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
    algo_cfg = DDPGConfig(learning_rate=3e-4)
    algo = create_algo("ddpg", algo_cfg)
    
    print(f"[INFO] 算法: DDPG")
    
    # 4. 创建多层经验池
    buffer_config = MultiLevelBufferConfig(
        coverage_capacity=100000,
        difficulty_capacity=10000,
        keyevent_capacity=1000,
        batch_size=64,
        filter_freq=1000,  # 每1000步筛选一次
        pool_weights=[0.5, 0.3, 0.2]  # 基座50%，难点30%，关键20%
    )
    buffer = MultiLevelBuffer(buffer_config)
    
    print(f"[INFO] 多层经验池创建成功")
    print(f"  - 基座覆盖池容量: {buffer_config.coverage_capacity}")
    print(f"  - 难点聚焦池容量: {buffer_config.difficulty_capacity}")
    print(f"  - 关键事件池容量: {buffer_config.keyevent_capacity}")
    
    # 5. 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # 6. 创建训练配置
    train_cfg = TrainConfig(
        total_timesteps=50000,
        num_steps=128,
        device="cpu"
    )
    
    # 7. 创建训练器
    trainer = OffPolicyTrainer(
        envs=envs,
        model=model,
        algo=algo,
        optimizer=optimizer,
        cfg=train_cfg,
        buffer=buffer
    )
    
    print("[INFO] 开始训练...")
    
    # 训练循环（简化版）
    obs_np = envs.reset()
    obs = torch.tensor(obs_np, device="cpu", dtype=torch.float32)
    
    for step in range(train_cfg.total_timesteps):
        # 采样动作
        with torch.no_grad():
            act_out = model.act(obs)
            actions = act_out.actions
        
        # 环境交互
        actions_np = actions.detach().cpu().numpy()
        next_obs_np, rewards_np, dones_np, infos = envs.step(actions_np)
        
        # 存储经验
        for i in range(envs.num_envs):
            buffer.add(
                state=obs_np[i],
                action=actions_np[i],
                reward=float(rewards_np[i]),
                next_state=next_obs_np[i],
                done=bool(dones_np[i])
            )
        
        # 更新模型
        if buffer.is_ready() and step % 4 == 0:
            # 采样批次
            batch = buffer.sample(batch_size=64)
            states, actions, rewards, next_states, dones = batch
            
            # 转换为tensor
            states_t = torch.tensor(states, dtype=torch.float32)
            actions_t = torch.tensor(actions, dtype=torch.float32)
            rewards_t = torch.tensor(rewards, dtype=torch.float32)
            next_states_t = torch.tensor(next_states, dtype=torch.float32)
            dones_t = torch.tensor(dones, dtype=torch.float32)
            
            # 创建batch对象（简化处理）
            from Reinforce_learning.RLg.DDPG_refactored import DDPGBatch
            batch_obj = DDPGBatch(
                obs=states_t,
                actions=actions_t,
                rewards=rewards_t,
                next_obs=next_states_t,
                dones=dones_t
            )
            
            # 更新模型
            metrics = algo.update(model, optimizer, batch_obj)
        
        # 更新观测
        obs = torch.tensor(next_obs_np, device="cpu", dtype=torch.float32)
        
        # 打印统计信息
        if step % 1000 == 0:
            stats = buffer.get_pool_stats()
            print(f"Step {step}: {stats}")
    
    print("[INFO] 训练完成")


if __name__ == "__main__":
    train_with_multilevel_buffer()
