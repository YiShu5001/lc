# trainer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import time
import numpy as np
import torch

from Gym_env.BaseEnv import VectorEnvLike
from NN.BaseNN import BaseRLModel
from Reinforce_learning.Basealgos import BaseAlgo, RolloutBatch


@dataclass
class TrainConfig:
    """
    训练流程配置（只放训练/采样相关的参数）
    """
    total_timesteps: int = 1_000_000
    num_steps: int = 128              # 每次 rollout 的时间长度 T
    gamma: float = 0.99
    gae_lambda: float = 0.95
    anneal_lr: bool = False           # 是否学习率退火
    log_interval: int = 10            # 多少个 iteration 打印/记录一次
    device: str = "cpu"               # "cuda" or "cpu"


class RolloutStorage:
    """
    on-policy 轨迹缓存：存 (T, N, ...) 结构的数据，并能计算 GAE、flatten 成 batch。
    """
    def __init__(self, num_steps: int, num_envs: int, obs_shape, action_shape, device: torch.device, discrete: bool):
        self.T = num_steps
        self.N = num_envs
        self.device = device
        self.discrete = discrete

        # 观测： (T, N, *obs_shape)
        self.obs = torch.zeros((self.T, self.N) + tuple(obs_shape), device=device, dtype=torch.float32)

        # 动作：
        # 离散： (T, N) long
        # 连续： (T, N, act_dim) float
        if discrete:
            self.actions = torch.zeros((self.T, self.N), device=device, dtype=torch.long)
        else:
            self.actions = torch.zeros((self.T, self.N) + tuple(action_shape), device=device, dtype=torch.float32)

        self.logprobs = torch.zeros((self.T, self.N), device=device, dtype=torch.float32)
        self.rewards = torch.zeros((self.T, self.N), device=device, dtype=torch.float32)
        self.dones = torch.zeros((self.T, self.N), device=device, dtype=torch.float32)
        self.values = torch.zeros((self.T, self.N), device=device, dtype=torch.float32)

        self.advantages = torch.zeros((self.T, self.N), device=device, dtype=torch.float32)
        self.returns = torch.zeros((self.T, self.N), device=device, dtype=torch.float32)

    def store_step(
        self,
        t: int,
        obs: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """
        存储第 t 步数据（t in [0, T-1]）
        obs:    (N, *obs_shape)
        actions:(N,) or (N, act_dim)
        logprobs:(N,)
        rewards:(N,)
        dones:  (N,)  float(0/1)
        values: (N,)
        """
        self.obs[t] = obs
        self.actions[t] = actions
        self.logprobs[t] = logprobs
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values

    @torch.no_grad()
    def compute_gae(self, next_value: torch.Tensor, next_done: torch.Tensor, gamma: float, lam: float) -> None:
        """
        计算优势与回报（GAE-Lambda）
        next_value: (N,)  rollout 结束时刻的 V(s_{T})
        next_done:  (N,)  rollout 结束时刻是否终止（0/1）
        """
        last_gae = torch.zeros_like(next_value, device=self.device)
        for t in reversed(range(self.T)):
            if t == self.T - 1:
                next_nonterminal = 1.0 - next_done
                next_values = next_value
            else:
                next_nonterminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_values * next_nonterminal - self.values[t]
            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def to_batch(self) -> RolloutBatch:
        """
        flatten (T, N, ...) -> (B, ...)
        B = T * N
        """
        B_obs = self.obs.reshape((-1,) + self.obs.shape[2:])
        B_actions = self.actions.reshape((-1,) + (() if self.discrete else self.actions.shape[2:]))
        B_logprobs = self.logprobs.reshape(-1)
        B_adv = self.advantages.reshape(-1)
        B_ret = self.returns.reshape(-1)
        B_values = self.values.reshape(-1)
        return RolloutBatch(
            obs=B_obs,
            actions=B_actions,
            old_logprobs=B_logprobs,
            advantages=B_adv,
            returns=B_ret,
            old_values=B_values,
        )


class Trainer:
    """
    训练器：负责完整训练流程。
    - 采样 rollout
    - 计算 GAE/returns
    - 调用 algo.update
    - 记录日志 / 控制训练进度
    """

    def __init__(
        self,
        envs: VectorEnvLike,
        model: BaseRLModel,
        algo: BaseAlgo,
        optimizer: torch.optim.Optimizer,
        cfg: TrainConfig,
        logger: Optional[object] = None,
    ):
        self.envs = envs
        self.model = model
        self.algo = algo
        self.optimizer = optimizer
        self.cfg = cfg
        self.logger = logger  # 可传入 TensorBoard/WandB/自研 logger

        self.device = torch.device(cfg.device)
        self.model.to(self.device)

        self.batch_size = envs.num_envs * cfg.num_steps
        self.num_iterations = cfg.total_timesteps // self.batch_size

    def train(self) -> None:
        """
        完整训练入口
        """
        start_time = time.time()
        global_step = 0

        # reset env
        obs_np = self.envs.reset()
        obs = torch.tensor(obs_np, device=self.device, dtype=torch.float32)
        done = torch.zeros(self.envs.num_envs, device=self.device, dtype=torch.float32)

        storage = RolloutStorage(
            num_steps=self.cfg.num_steps,
            num_envs=self.envs.num_envs,
            obs_shape=self.envs.obs_shape,
            action_shape=self.envs.action_shape,
            device=self.device,
            discrete=self.envs.is_discrete,
        )

        for it in range(1, self.num_iterations + 1):
            # 采样 rollout（T 步）
            for t in range(self.cfg.num_steps):
                global_step += self.envs.num_envs

                act_out = self.model.act(obs)  # actions/logprobs/values
                actions = act_out.actions

                # env step (转换成 numpy)
                actions_np = actions.detach().cpu().numpy()
                next_obs_np, rewards_np, dones_np, infos = self.envs.step(actions_np)

                rewards = torch.tensor(rewards_np, device=self.device, dtype=torch.float32)
                next_done = torch.tensor(dones_np.astype(np.float32), device=self.device)

                storage.store_step(
                    t=t,
                    obs=obs,
                    actions=actions,
                    logprobs=act_out.logprobs,
                    rewards=rewards,
                    dones=done,              # 存“当前步开始时”的 done
                    values=act_out.values,
                )

                obs = torch.tensor(next_obs_np, device=self.device, dtype=torch.float32)
                done = next_done

                # 可选：在这里解析 infos 记录 episodic return/len
                if self.logger is not None:
                    self._log_episode_if_any(infos, global_step)

            # bootstrap value
            with torch.no_grad():
                next_value = self.model.value(obs)  # (N,)
            storage.compute_gae(next_value=next_value, next_done=done, gamma=self.cfg.gamma, lam=self.cfg.gae_lambda)

            # flatten + update
            batch = storage.to_batch()
            metrics = self.algo.update(self.model, self.optimizer, batch)

            # logging
            if (it % self.cfg.log_interval) == 0:
                sps = int(global_step / (time.time() - start_time))
                self._log_iter(metrics, sps, global_step, it)

    def _log_episode_if_any(self, infos: dict, global_step: int) -> None:
        """
        解析环境 infos 里的 episode 信息（具体格式由环境 wrapper 决定）
        这里只提供“接口位置”，不绑定任何库的具体字段。
        """
        # 例如：如果你环境约定 infos["episode_return"] / infos["episode_length"]
        # 就在这里记录。
        pass

    def _log_iter(self, metrics: Dict[str, float], sps: int, global_step: int, iteration: int) -> None:
        """
        记录每个 iteration 的指标（同样只提供接口位置）
        """
        # 你可以将 metrics 写入 TB/WandB 或 print
        # 例如：print(f"it={iteration}, step={global_step}, sps={sps}, {metrics}")
        pass
