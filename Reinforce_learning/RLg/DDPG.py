import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class Actor(nn.Module):
    def __init__(self, state_size, action_size, action_high):
        super(Actor, self).__init__()
        self.action_high = torch.FloatTensor(action_high)
        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
            nn.Tanh()  # 输出范围[-1,1]
        )

    def forward(self, state):
        return self.net(state) * self.action_high  # 缩放到实际动作范围


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_path = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
        )
        self.action_path = nn.Sequential(
            nn.Linear(action_size, 256),
            nn.ReLU(),
        )
        self.combined = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        s = self.state_path(state)
        a = self.action_path(action)
        return self.combined(torch.cat([s, a], dim=1))


class DDPGAgent:
    def __init__(self, state_size, action_size, action_high):
        # 超参数
        self.gamma = 0.99  # 折扣因子
        self.tau = 0.005  # 软更新系数
        self.batch_size = 128  # 训练批大小
        self.memory_size = 100000
        self.noise_std = 0.1  # 动作噪声标准差

        # 初始化网络
        self.actor = Actor(state_size, action_size, action_high)
        self.actor_target = Actor(state_size, action_size, action_high)
        self.critic = Critic(state_size, action_size)
        self.critic_target = Critic(state_size, action_size)

        # 同步目标网络参数
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-3)

        # 经验回放池
        self.memory = deque(maxlen=self.memory_size)

    def get_action(self, state, add_noise=True):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            action = self.actor(state).numpy()
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            return np.clip(action + noise, -self.actor.action_high, self.actor.action_high)
        return action

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if len(self.memory) < self.batch_size:
            return

        # 从经验池采样
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为PyTorch张量
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Critic损失计算
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, target_actions)
            target = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target)

        # Critic更新
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Actor更新
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # 软更新目标网络
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)