import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from Reinforce_learning.Basealgos import BaseAlgo

class Actor(nn.Module):
    """
    策略网络 (Actor)
    在确定性策略梯度(DPG)中，输入状态，直接输出确定的动作值 (不同于概率分布)。
    输出经过 tanh 映射到 [-max_action, max_action]。
    """
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
    价值网络 (Critic)
    输入(状态, 动作)对，输出对应的 Q 值 Q(s, a)。
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

class DDPG(BaseAlgo):
    """
    深度确定性策略梯度 (Deep Deterministic Policy Gradient, DDPG)
    
    核心思想: 
    将 DQN 扩展到了**连续动作空间**。
    包含一个 Actor（负责基于状态选择确定的动作）和一个 Critic（负责评估该动作的 Q 值）。
    
    关键技术:
    1. Actor-Critic 架构。
    2. 目标网络 (Target Networks) & 软更新 (Soft Update)。
    3. 在动作上添加探索噪声 (通常为高斯或 OU 噪声)。
    """
    def __init__(self, state_dim, action_dim, max_action=1.0, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005):
        super().__init__(state_dim, action_dim, max_action)
        
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.tau = tau
        
        self.models = {"actor": self.actor, "critic": self.critic}

    def select_action(self, state, evaluate=False):
        """
        根据当前状态选择确定性动作。
        测试时(evaluate=True)直接调用此函数无噪声。
        通常通过外层环境交互循环(如Trainer)在训练时为其添加噪声。
        """
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, replay_buffer, batch_size=256):
        """
        网络更新: DDPG 的核心逻辑
        """
        # 1. 经验回放采样
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # 2. 计算目标 Q 值 (Target Q)
        with torch.no_grad():
            target_Q_next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, target_Q_next_action)
            target_Q = reward + (not_done * self.gamma * target_Q)

        # 3. 更新 Critic: 最小化 Q 网络的均方误差
        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 4. 更新 Actor: 最大化 Critic 对 Actor 当前选出动作的 Q 值评估
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 5. 软更新目标网络: \theta_{target} = \tau * \theta + (1 - \tau) * \theta_{target}
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item()
        }
