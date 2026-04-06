import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

from Reinforce_learning.Basealgos import BaseAlgo

class DQN(BaseAlgo):
    """
    深度Q网络 (Deep Q-Network, DQN)
    
    核心思想: 
    使用神经网络近似 Q 目标值函数 Q(s, a)。
    DQN 主要用于**离散动作空间**。
    
    关键技术:
    1. 经验回放 (Experience Replay): 打破数据相关性。
    2. 目标网络 (Target Network): 稳定训练，减少自举产生的发散。
    """
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005):
        # 注意: DQN 用于离散空间，通常不涉及 max_action 缩放
        super().__init__(state_dim, action_dim, max_action=1.0)
        self.gamma = gamma
        self.tau = tau
        
        # Q 网络: 输入状态，输出每个动作的 Q 值
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # 目标 Q 网络: 延迟更新，提供稳定的 Q 目标
        self.target_q_net = copy.deepcopy(self.q_net)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        
        self.models = {"q_net": self.q_net, "target_q_net": self.target_q_net}

    def select_action(self, state, evaluate=False, epsilon=0.1):
        """
        动作选择: $\epsilon$-贪心策略
        - evaluate=True 时完全贪心探索 (epsilon = 0)
        """
        if evaluate:
            epsilon = 0.0
            
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state)
            return q_values.argmax().item()

    def update(self, replay_buffer, batch_size=256):
        """
        网络更新: 最小化 TD Error (时序差分误差)
        """
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # 1. 计算当前 Q 值 Q(s, a)
        current_q = self.q_net(state).gather(1, action.long())

        # 2. 计算目标 Q 值: r + \gamma * max_a Q_target(s', a)
        with torch.no_grad():
            max_next_q = self.target_q_net(next_state).max(1, keepdim=True)[0]
            target_q = reward + not_done * self.gamma * max_next_q

        # 3. 计算均方误差损失 (MSE Loss)
        loss = F.mse_loss(current_q, target_q)

        # 4. 反向传播更新 Q 网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 5. 软更新目标网络参数: \theta_{target} = \tau * \theta + (1 - \tau) * \theta_{target}
        for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return {"loss": loss.item()}
