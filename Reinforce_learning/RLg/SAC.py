import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import copy

from Reinforce_learning.Basealgos import BaseAlgo

class Actor(nn.Module):
    """
    SAC 随机策略 Actor (Squashed Gaussian Policy)
    输出均值(mu)和对数标准差(log_std)，并且将采样的动作经过 tanh 映射(Squashing)，保证动作在有界空间内。
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state):
        x = self.net(state)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        """ 在重参数化(Reparameterization Trick)下采样动作，并计算对数概率 """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mu, std)
        x_t = normal.rsample() # rsample 允许反向传播通过采样过程
        y_t = torch.tanh(x_t)  # 将动作压扁(squash)到 [-1, 1]
        action = y_t * self.max_action
        
        # 计算采用 tanh 后的修正对数概率 (Enforcing Action Bounds)
        log_prob = normal.log_prob(x_t)
        # log \pi(a|s) = log p(x|s) - \sum \log(1 - \tanh^2(x))
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, torch.tanh(mu) * self.max_action

class Critic(nn.Module):
    """ SAC 的双 Q 网络架构 """
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

class SAC(BaseAlgo):
    """
    柔性Actor-Critic (Soft Actor-Critic, SAC)
    
    核心思想: 
    基于最大熵强化学习 (Maximum Entropy RL) 框架。
    智能体不仅仅要最大化期望回报，还要最大化其策略的熵 (Entropy) $\mathcal{H}(\pi(\cdot|s))$，
    这极大地增强了探索能力并防止收敛到局部最优。
    
    关键技术:
    1. 最大熵目标: $J(\pi) = \sum \mathbb{E}[ r_t + \alpha \mathcal{H}(\pi) ]$
    2. 双 Q 网络 (同 TD3 的 Clipped Double Q-Learning)。
    3. 重参数化技巧 (Reparameterization Trick) 用于随机策略梯度的回传。
    4. 自动温度调节 (Automatic Entropy Adjustment, 可学习的 $\alpha$)。
    """
    def __init__(self, state_dim, action_dim, max_action=1.0, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        super().__init__(state_dim, action_dim, max_action)
        
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        
        # 自动熵调节配置
        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to("cpu")).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device="cpu")
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)
        self.alpha = alpha
        
        # 注册模型
        self.models = {"actor": self.actor, "critic": self.critic}

    def select_action(self, state, evaluate=False):
        """ 交互过程动作选择 """
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _, mean = self.actor.sample(state)
        # 测试时使用确定的均值(mean)，训练时使用采样动作(action)
        return mean.numpy().flatten() if evaluate else action.numpy().flatten()

    def update(self, replay_buffer, batch_size=256):
        """ SAC 网络更新 """
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # -----------------------------
        # 1. 更新 Critic (价值网络)
        # -----------------------------
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
            target_Q = reward + not_done * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -----------------------------
        # 2. 更新 Actor (策略网络)
        # -----------------------------
        pi, log_pi, _ = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # -----------------------------
        # 3. 更新 Alpha (温度参数 - 动态调整熵的权重)
        # -----------------------------
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()

        # -----------------------------
        # 4. 软更新目标网络
        # -----------------------------
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item()
        }
