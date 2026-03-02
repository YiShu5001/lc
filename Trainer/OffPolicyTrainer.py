"""
Off-Policy训练器
支持SAC、TD3、DDPG、DQN等off-policy算法
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import time
import numpy as np
import torch

from Gym_env.BaseEnv import VectorEnvLike
from NN.BaseNN import BaseRLModel
from Reinforce_learning.Basealgos import BaseAlgo
from Reinforce_learning.buffers import BaseBuffer, ReplayBuffer
from Reinforce_learning.buffers.multi_level.MultiLevelBuffer import MultiLevelBuffer
from Trainer.BaseTrainer import TrainConfig


class OffPolicyTrainer:
    """
    Off-Policy训练器
    
    特点：
    - 使用Buffer存储经验
    - 支持异步采样和更新
    - 适用于SAC、TD3、DDPG、DQN等算法
    """
    
    def __init__(
        self,
        envs: VectorEnvLike,
        model: BaseRLModel,
        algo: BaseAlgo,
        optimizer: torch.optim.Optimizer,
        cfg: TrainConfig,
        buffer: Optional[BaseBuffer] = None,
        logger: Optional[object] = None,
    ):
        """
        Args:
            envs: 环境
            model: 模型
            algo: 算法
            optimizer: 优化器
            cfg: 训练配置
            buffer: 经验池（如果为None，创建默认ReplayBuffer）
            logger: 日志记录器
        """
        self.envs = envs
        self.model = model
        self.algo = algo
        self.optimizer = optimizer
        self.cfg = cfg
        self.logger = logger
        
        self.device = torch.device(cfg.device)
        self.model.to(self.device)
        
        # 创建经验池
        if buffer is None:
            from Reinforce_learning.buffers import BufferConfig
            buffer_cfg = BufferConfig(capacity=100000, batch_size=256)
            self.buffer = ReplayBuffer(buffer_cfg)
        else:
            self.buffer = buffer
        
        # 检查是否为MultiLevelBuffer
        self.is_multilevel = isinstance(self.buffer, MultiLevelBuffer)
        
        # 训练步数
        self.global_step = 0
        self.update_freq = 4  # 每N步更新一次
    
    def train(self) -> None:
        """
        完整训练入口
        """
        start_time = time.time()
        
        # 重置环境
        obs_np = self.envs.reset()
        obs = torch.tensor(obs_np, device=self.device, dtype=torch.float32)
        
        # 训练循环
        while self.global_step < self.cfg.total_timesteps:
            # 采样动作
            with torch.no_grad():
                act_out = self.model.act(obs)
                actions = act_out.actions
            
            # 环境交互
            actions_np = actions.detach().cpu().numpy()
            next_obs_np, rewards_np, dones_np, infos = self.envs.step(actions_np)
            
            # 存储经验
            for i in range(self.envs.num_envs):
                self.buffer.add(
                    state=obs_np[i],
                    action=actions_np[i],
                    reward=float(rewards_np[i]),
                    next_state=next_obs_np[i],
                    done=bool(dones_np[i])
                )
            
            # 更新模型
            if self.buffer.is_ready() and self.global_step % self.update_freq == 0:
                self._update_model()
            
            # 如果是MultiLevelBuffer，更新优先级
            if self.is_multilevel and self.global_step % self.update_freq == 0:
                # 这里可以添加TD误差更新逻辑
                pass
            
            # 更新观测
            obs = torch.tensor(next_obs_np, device=self.device, dtype=torch.float32)
            self.global_step += self.envs.num_envs
            
            # 处理done信号（重置环境）
            if np.any(dones_np):
                reset_indices = np.where(dones_np)[0]
                reset_obs_np, _ = self.envs.reset()
                obs_np[reset_indices] = reset_obs_np[reset_indices]
                obs = torch.tensor(obs_np, device=self.device, dtype=torch.float32)
            
            # 日志记录
            if self.global_step % (self.cfg.log_interval * 1000) == 0:
                sps = int(self.global_step / (time.time() - start_time))
                self._log_iter(sps, self.global_step)
    
    def _update_model(self):
        """更新模型"""
        # 从buffer采样
        batch = self.buffer.sample()
        states, actions, rewards, next_states, dones = batch
        
        # 转换为tensor
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)
        
        # 创建batch（根据算法类型）
        from Reinforce_learning.RLg.SAC import SACBatch
        from Reinforce_learning.RLg.TD3 import TD3Batch
        from Reinforce_learning.RLg.DDPG_refactored import DDPGBatch
        from Reinforce_learning.RLg.DQN import DQNBatch
        
        # 判断算法类型（简化处理，实际应该从algo获取）
        algo_name = type(self.algo).__name__.lower()
        
        if 'sac' in algo_name:
            batch_obj = SACBatch(
                obs=states,
                actions=actions,
                rewards=rewards,
                next_obs=next_states,
                dones=dones
            )
        elif 'td3' in algo_name or 'ddpg' in algo_name:
            batch_obj = DDPGBatch(
                obs=states,
                actions=actions,
                rewards=rewards,
                next_obs=next_states,
                dones=dones
            )
        elif 'dqn' in algo_name:
            batch_obj = DQNBatch(
                obs=states,
                actions=actions,
                rewards=rewards,
                next_obs=next_states,
                dones=dones
            )
        else:
            raise ValueError(f"不支持的算法类型: {algo_name}")
        
        # 调用算法更新
        metrics = self.algo.update(self.model, self.optimizer, batch_obj)
        
        # 更新buffer优先级（如果使用优先回放）
        if hasattr(self.buffer, 'update_priority') and hasattr(batch_obj, 'indices'):
            # 这里需要计算TD误差，简化处理
            pass
    
    def _log_iter(self, sps: int, global_step: int):
        """记录日志"""
        if self.logger is not None:
            # 使用logger记录
            pass
        else:
            print(f"Step: {global_step}, SPS: {sps}")
