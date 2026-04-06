import os
import time
import numpy as np
import torch

from Reinforce_learning.RLg.TD3 import TD3
from Reinforce_learning.buffers.PyramidPER import PyramidPER
from Gym_env.gym_pybullet_drones.envs.TSA_LADRC_Env import TSA_LADRC_Env
from NN.TaskDecomposedActor import TaskDecomposedActor

class HGCTrainer:
    """
    HGC-RL (Hierarchical Guided Curriculum RL) 总训练器
    
    负责将以下模块串联整合：
    1. 环境层: TSA_LADRC_Env (双时间尺度控制参数整定)
    2. 网络层: TaskDecomposedActor (Transformer 架构，抗遗忘与注意力机制)
    3. 算法层: TD3 (Off-policy 连续控制基线)
    4. 记忆层: PyramidPER (金字塔优先经验回放，解决课程学习灾难遗忘)
    """
    def __init__(self, 
                 env_kwargs=None, 
                 max_episodes=5000, 
                 max_steps_per_ep=500,
                 batch_size=256,
                 start_timesteps=10000,  # 纯随机探索的步数，用于填充初始经验池
                 explore_noise=0.1,
                 model_save_path="./models/hgc_rl_checkpoint"):
        
        self.max_episodes = max_episodes
        self.max_steps_per_ep = max_steps_per_ep
        self.batch_size = batch_size
        self.start_timesteps = start_timesteps
        self.explore_noise = explore_noise
        self.model_save_path = model_save_path
        
        # 1. 实例化环境
        env_kwargs = env_kwargs or {'ctrl_freq': 100, 'rl_freq': 10, 'gui': False}
        self.env = TSA_LADRC_Env(**env_kwargs)
        
        # 提取环境维度信息
        # 注意: 这里的维度是为了演示和对齐，具体维度需要根据你的实际 Wrapper 调整
        # 在 TSA_LADRC_Env 中，原始 obs 是 50 维。
        # 为了对接 TaskDecomposedActor，我们需要将 50 维拆解为 self_state, obs_states, neighbor_states
        # 这里仅作骨架演示，假设全部由 self_state 承接
        self.self_state_dim = 50 
        self.obs_state_dim = 6   # 假定障碍物维度
        self.neighbor_state_dim = 10 # 假定邻居维度
        self.action_dim = self.env.action_space.shape[1] # 对于单机应为 12
        self.max_action = float(self.env.action_space.high[0][0])
        
        # 2. 实例化算法 (使用 TD3)
        self.agent = TD3(state_dim=self.self_state_dim, action_dim=self.action_dim, max_action=self.max_action)
        
        # 🌟 核心注入: 将 TD3 默认的 MLP Actor 替换为你设计的 TaskDecomposedActor
        print("[INFO] Injecting TaskDecomposedActor into TD3...")
        self.agent.actor = TaskDecomposedActor(
            self_state_dim=self.self_state_dim,
            obs_state_dim=self.obs_state_dim,
            neighbor_state_dim=self.neighbor_state_dim,
            action_dim=self.action_dim,
            d_model=64, n_blocks=2
        )
        self.agent.actor_target = TaskDecomposedActor(
            self_state_dim=self.self_state_dim,
            obs_state_dim=self.obs_state_dim,
            neighbor_state_dim=self.neighbor_state_dim,
            action_dim=self.action_dim,
            d_model=64, n_blocks=2
        )
        self.agent.actor_target.load_state_dict(self.agent.actor.state_dict())
        self.agent.actor_optimizer = torch.optim.Adam(self.agent.actor.parameters(), lr=3e-4)
        self.agent.models["actor"] = self.agent.actor # 更新保存引用
        
        # 3. 实例化金字塔经验池
        print("[INFO] Initializing PyramidPER Buffer...")
        self.buffer = PyramidPER(state_dim=self.self_state_dim, action_dim=self.action_dim, max_size=int(1e5), batch_size=self.batch_size)
        
        self.total_timesteps = 0
        self.episode_num = 0

    def _prepare_actor_inputs(self, state):
        """ 将环境的一维 state 拆解包装为 TaskDecomposedActor 期待的 3D 张量结构 """
        # 在实际工程中，你需要根据传感器数据将一维数组重组。
        # 这里作为跑通主干，构造 dummy 的障碍物和邻居向量
        s_t = torch.FloatTensor(state.reshape(1, -1))
        
        # Dummy obstacles (Batch=1, N_obs=0, Dim) -> 为了避免报错，弄个全 False 的掩码或 dummy 数据
        # 假设当前没有障碍物和邻居 (全被 mask 掉)
        obs_t = torch.zeros((1, 1, self.obs_state_dim))
        nbr_t = torch.zeros((1, 1, self.neighbor_state_dim))
        obs_mask = torch.ones((1, 1), dtype=torch.bool)
        nbr_mask = torch.ones((1, 1), dtype=torch.bool)
        
        return s_t, obs_t, nbr_t, obs_mask, nbr_mask

    def train(self):
        print(f"[INFO] Starting HGC-RL Training Loop...")
        start_time = time.time()
        
        obs, _ = self.env.reset()
        state = obs[0] # 对于单智能体，提取第一行
        episode_reward = 0
        episode_timesteps = 0
        
        for t in range(int(self.max_episodes * self.max_steps_per_ep)):
            self.total_timesteps += 1
            episode_timesteps += 1
            
            # 1. 动作选择 (纯随机探索 or Actor预测+噪声)
            if self.total_timesteps < self.start_timesteps:
                action = self.env.action_space.sample()[0] # 纯随机探索
            else:
                s_t, o_t, n_t, o_m, n_m = self._prepare_actor_inputs(state)
                with torch.no_grad():
                    # 调用 TaskDecomposedActor，获取最终输出 final_collab_action
                    _, action_tensor = self.agent.actor(s_t, o_t, n_t, obs_mask=o_m, neighbor_mask=n_m)
                    action = action_tensor.cpu().numpy().flatten()
                
                # 添加探索噪声
                noise = np.random.normal(0, self.max_action * self.explore_noise, size=self.action_dim)
                action = (action + noise).clip(-self.max_action, self.max_action)

            # 2. 环境交互
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_state = next_obs[0]
            done = bool(terminated or truncated)
            
            # 3. 存入金字塔经验池 (存入 L1 层)
            # 注意: 真正的 done(代表失败或成功) 才存入 replay buffer，截断(truncated)不算作 done
            done_bool = float(done) if episode_timesteps < self.max_steps_per_ep else 0.0
            self.buffer.store_transition(state, action, reward, next_state, done_bool)

            state = next_state
            episode_reward += reward

            # 4. 算法更新
            if self.total_timesteps >= self.start_timesteps:
                # 从 PyramidPER 采样
                batch, is_weights = self.buffer.sample_batch()
                if batch is not None:
                    # 将采样数据喂给 TD3 进行一次梯度更新
                    # （注：这里需要稍微适配 TD3 的 update 方法接受 is_weights 和 batch字典，
                    # 考虑到篇幅，这里用伪逻辑示意，实际应用中 TD3 需要计算带权重的 loss）
                    # metrics = self.agent.update_with_is(batch, is_weights)
                    
                    # 为了跑通，这里用临时适配逻辑:
                    dummy_td_errors = np.random.uniform(0.01, 1.0, size=len(batch['indices']))
                    
                    # 5. 更新完网络后，必须更新经验池的 TD Error 触发晋升
                    self.buffer.update_and_promote(batch['indices'], dummy_td_errors)

            if done or episode_timesteps >= self.max_steps_per_ep:
                elapsed = time.time() - start_time
                print(f"[Iter {self.total_timesteps}] Ep {self.episode_num+1} | "
                      f"Reward: {episode_reward:.2f} | Steps: {episode_timesteps} | "
                      f"Buffer: L1={len(self.buffer.L1_indices)}, L2={len(self.buffer.L2_indices)}, L3={len(self.buffer.L3_indices)} | "
                      f"Time: {elapsed:.1f}s")
                
                # 每 100 个 Episode 模拟一次课程切换，提取 Top 经验到 L3
                if (self.episode_num + 1) % 100 == 0:
                    print("[EVENT] Triggering Curriculum Switch...")
                    self.buffer.curriculum_switch(top_k=50)
                    self.agent.save(self.model_save_path)
                
                # 重置环境
                obs, _ = self.env.reset()
                state = obs[0]
                episode_reward = 0
                episode_timesteps = 0
                self.episode_num += 1
                
                if self.episode_num >= self.max_episodes:
                    break
        
        print("[INFO] Training Complete.")
