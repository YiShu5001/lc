import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from Reinforce_learning.Basealgos import BaseAlgo

class Actor(nn.Module):
    """ TD3 的策略网络 (与 DDPG 类似) """
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

class Critic(nn.Module):
    """
    TD3 的 双Q 网络 (Twin Critic)
    为了缓解 DDPG 中的 Q 值高估问题，使用两套独立的 Q 网络。
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa)

class TD3(BaseAlgo):
    """
    双延迟深度确定性策略梯度 (Twin Delayed DDPG, TD3)
    
    核心思想: DDPG 的进阶版本，主要解决 Q 值高估(Overestimation)问题。
    
    关键技术 (TD3的三把斧):
    1. 双 Q 网络 (Clipped Double Q-Learning): 训练两个 Critic，在计算目标 Q 值时取两者的较小值。
    2. 目标策略平滑 (Target Policy Smoothing): 计算目标动作时加入裁剪过的噪声，使得相近动作的 Q 值平滑。
    3. 延迟策略更新 (Delayed Policy Updates): Critic 更新频率高于 Actor，等 Critic 准确后再更新 Actor。
    """
    def __init__(
        self, state_dim, action_dim, max_action=1.0,
        lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, tau=0.005,
        policy_noise=0.2, noise_clip=0.5, policy_freq=2
    ):
        super().__init__(state_dim, action_dim, max_action)
        
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0 # 追踪更新迭代次数
        
        # 注册模型以便基类提供保存和加载功能
        self.models = {"actor": self.actor, "critic": self.critic}

    def select_action(self, state, evaluate=False):
        """ 测试时 evaluate=True 无噪声直接输出。通常由外层 Wrapper 决定是否加探索噪声 """
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, replay_buffer, batch_size=256):
        """ TD3 网络更新 """
        self.total_it += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # 1. 目标策略平滑 (Target Policy Smoothing)
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # 2. 双 Q 网络 (Clipped Double Q)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q

        # 3. 更新 Critic
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_val = 0.0

        # 4. 延迟策略更新 (Delayed Policy Updates)
        if self.total_it % self.policy_freq == 0:
            
            # 使用 Q1 来指导 Actor 更新
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            actor_loss_val = actor_loss.item()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss_val
        }
