import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from Reinforce_learning.Basealgos import BaseAlgo

class ActorCritic(nn.Module):
    """
    A2C 的网络结构：策略网络 Actor (均值, 标差) + 价值网络 Critic
    通常共享基础特征层，或者拆分成独立的 MLP (这里为了简化结构使用独立的网络)。
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorCritic, self).__init__()
        
        # Actor
        self.actor_mu = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.max_action = max_action

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        # 状态价值
        value = self.critic(state)
        
        # 动作分布
        mu = self.actor_mu(state) * self.max_action
        std = self.actor_log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        
        return dist, value

class A2C(BaseAlgo):
    """
    优势行动者-评论家 (Advantage Actor-Critic, A2C)
    
    核心思想:
    PPO 的前身，属于经典同策略 (On-policy) 算法。
    Critic 用于估计状态价值 $V(s)$，从而计算 Advantage ($A(s,a) = Q(s,a) - V(s)$)。
    Actor 根据 Advantage 来指引策略更新：Advantage > 0 说明动作比平均要好，增加其出现概率。
    
    关键技术:
    1. 同步并行 (Synchronous) 更新架构。
    2. 引入 Advantage 减少了策略梯度更新时的方差 (Variance)。
    """
    def __init__(self, state_dim, action_dim, max_action=1.0, lr=1e-3, gamma=0.99, entropy_coef=0.01):
        super().__init__(state_dim, action_dim, max_action)
        
        self.policy = ActorCritic(state_dim, action_dim, max_action)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        # 记录单个回合的数据
        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []
        
        self.models = {"policy": self.policy}

    def select_action(self, state, evaluate=False):
        """ 动作选择，同时缓存 log_prob 和 value 用于一整个回合作更新 """
        state = torch.FloatTensor(state).unsqueeze(0)
        dist, value = self.policy(state)
        
        if evaluate:
            # 评估模式直接返回均值，不记录信息
            return (self.policy.actor_mu(state) * self.max_action).detach().numpy().flatten()
            
        action = dist.sample()
        
        self.saved_log_probs.append(dist.log_prob(action).sum(dim=-1))
        self.saved_values.append(value)
        
        return action.flatten().numpy()

    def store_reward(self, reward):
        """ 记录单步奖励 """
        self.rewards.append(reward)

    def update(self):
        """ 
        A2C 更新过程 (通常在一个 Episode 结束后，或者 N 步后进行) 
        """
        if not self.rewards:
            return {"loss": 0.0}

        # 计算折扣回报 R_t = \sum_{k=0} \gamma^k r_{t+k}
        returns = []
        R = 0
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8) # 归一化

        policy_losses = []
        value_losses = []
        
        # 将缓存的数据堆叠
        saved_values = torch.cat(self.saved_values)
        saved_log_probs = torch.cat(self.saved_log_probs)
        
        # Advantage = Return - Baseline (这里 Baseline 是 Critic 给出的 Value)
        advantages = returns.detach() - saved_values.squeeze()

        for log_prob, advantage in zip(saved_log_probs, advantages):
            # 策略梯度：最大化 \log \pi(a|s) * Advantage
            policy_losses.append(-log_prob * advantage.detach())
            
        # 价值损失：均方误差 MSE(V(s), Return)
        value_loss = F.mse_loss(saved_values.squeeze(), returns)

        # 综合损失 (可扩展加熵正则项)
        loss = torch.stack(policy_losses).sum() + value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 清空当前回合缓存
        del self.saved_log_probs[:]
        del self.saved_values[:]
        del self.rewards[:]

        return {
            "policy_loss": torch.stack(policy_losses).sum().item(),
            "value_loss": value_loss.item()
        }
