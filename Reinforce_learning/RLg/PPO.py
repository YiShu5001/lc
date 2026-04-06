import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from Reinforce_learning.Basealgos import BaseAlgo

class ActorCritic(nn.Module):
    """
    PPO 的网络结构：同时包含 Actor (策略网络) 和 Critic (价值网络)
    通常它们不共享底层特征提取层，以保证稳定，这里直接使用分开的 MLPs。
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorCritic, self).__init__()
        
        # Actor: 输出均值 mu
        self.actor_mu = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Tanh() # 映射到 [-1, 1] 然后乘以 max_action
        )
        self.max_action = max_action
        
        # Actor: 输出标准差的对数 (可学习的独立参数)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic: 输出状态价值 V(s)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self):
        raise NotImplementedError
        
    def act(self, state, evaluate=False):
        """ 用于环境交互：采样动作 """
        action_mean = self.actor_mu(state) * self.max_action
        
        if evaluate:
            # 评估模式直接返回均值，不要额外计算 logprob 等
            return action_mean.detach(), None, None

        action_std = self.actor_log_std.exp().expand_as(action_mean)
        
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        """ 用于 PPO 更新：评估给定动作的概率对数(logprob)和状态价值(V) """
        action_mean = self.actor_mu(state) * self.max_action
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        
        dist = Normal(action_mean, action_std)
        
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class PPO(BaseAlgo):
    """
    近端策略优化 (Proximal Policy Optimization, PPO) (Clip 版本)
    
    核心思想: 
    PPO 属于同策略(On-policy)的 Actor-Critic 算法。
    它通过引入重要性采样比率 (ratio) 和截断机制 (Clip)，限制了策略更新的步长，
    避免了单次更新改变过大导致训练崩溃。
    
    关键技术:
    1. Clip 损失函数: $L^{CLIP}(\theta) = \hat{E}_t [ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) ]$
    2. 广义优势估计 (Generalized Advantage Estimation, GAE)。(此实现假设外部传入算好的 advantage/returns)
    """
    def __init__(
        self, state_dim, action_dim, max_action=1.0, 
        lr_actor=3e-4, lr_critic=1e-3, 
        gamma=0.99, K_epochs=10, eps_clip=0.2, entropy_coef=0.01
    ):
        super().__init__(state_dim, action_dim, max_action)
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        
        self.policy = ActorCritic(state_dim, action_dim, max_action)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor_mu.parameters(), 'lr': lr_actor},
            {'params': [self.policy.actor_log_std], 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        
        self.models = {"policy": self.policy}

    def select_action(self, state, evaluate=False):
        """ 交互过程中的动作选择 """
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, action_logprob, state_val = self.policy.act(state, evaluate=evaluate)
        
        if evaluate:
            return action.flatten().numpy()
            
        # 训练时通常返回额外的 logprob, value，具体实现需要与外层 Trainer 对接
        # 这里维持统一接口，返回字典供收集
        return {
            "action": action.flatten().numpy(),
            "logprob": action_logprob.item(),
            "value": state_val.item()
        }

    def update(self, rollouts, batch_size=64):
        """
        PPO 网络更新: 
        rollouts 包含了完整的轨迹 (states, actions, logprobs, returns, advantages)
        在实际调用中，这里可以进一步拆分成 mini-batch 更新。
        """
        states = torch.FloatTensor(rollouts['states'])
        actions = torch.FloatTensor(rollouts['actions'])
        old_logprobs = torch.FloatTensor(rollouts['logprobs'])
        returns = torch.FloatTensor(rollouts['returns'])
        advantages = torch.FloatTensor(rollouts['advantages'])
        
        # Advantage 归一化 (有助于稳定训练)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 进行 K_epochs 次的自我更新迭代
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(logprobs - old_logprobs)

            # --- Actor Loss (Clip) ---
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # --- Critic Loss ---
            critic_loss = F.mse_loss(state_values, returns)

            # --- Total Loss ---
            loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * dist_entropy.mean()

            # 反向传播更新网络
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item()
        }
