import sys
import gym
import torch
import pylab
import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from prioritized_memory import Memory

EPISODES = 500  # 训练总回合数


# 深度Q网络定义（三層全連接）
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 24),  # 输入层（状态空间维度）
            nn.ReLU(),  # 激活函数
            nn.Linear(24, 24),  # 隐藏层
            nn.ReLU(),
            nn.Linear(24, action_size)  # 输出层（动作空间维度）
        )

    def forward(self, x):
        return self.fc(x)  # 前向传播


# 基于优先经验回放的DQN智能体
class DQNAgent():
    def __init__(self, state_size, action_size):
        # 环境交互设置
        self.render = False  # 是否可视化环境
        self.load_model = False  # 是否加载预训练模型

        # 状态/动作空间维度
        self.state_size = state_size
        self.action_size = action_size

        # DQN超参数配置
        self.discount_factor = 0.99  # 未来奖励折扣率
        self.learning_rate = 0.001  # 学习率
        self.memory_size = 20000  # 经验回放缓存容量
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.explore_step = 5000  # 探索衰减步数
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step  # ε衰减率
        self.batch_size = 64  # 训练批大小
        self.train_start = 1000  # 开始训练的经验阈值

        # 初始化优先经验回放内存
        self.memory = Memory(self.memory_size)

        # 创建Q网络和目标网络
        self.model = DQN(state_size, action_size)  # 主网络
        self.model.apply(self.weights_init)  # 权重初始化
        self.target_model = DQN(state_size, action_size)  # 目标网络
        self.optimizer = optim.Adam(self.model.parameters(),  # Adam优化器
                                    lr=self.learning_rate)
        self.update_target_model()  # 同步目标网络

        if self.load_model:
            self.model = torch.load('save_model/cartpole_dqn')  # 加载预训练模型

    # Xavier权重初始化（保持方差一致性）
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)  # 线性层使用Xavier初始化

    # 同步目标网络参数
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())  # 复制主网络参数

    # ε-贪婪策略选择动作
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:  # 探索：随机选择动作
            return random.randrange(self.action_size)
        else:  # 利用：选择Q值最大的动作
            state = torch.from_numpy(state)
            state = Variable(state).float().cpu()
            q_value = self.model(state)
            _, action = torch.max(q_value, 1)
            return int(action)

    # 存储经验到优先回放缓存
    def append_sample(self, state, action, reward, next_state, done):
        # 计算TD误差作为优先级
        target = self.model(Variable(torch.FloatTensor(state))).data
        old_val = target[0][action]  # 当前动作的Q值

        # 计算目标Q值（Double DQN风格）
        target_val = self.target_model(Variable(torch.FloatTensor(next_state))).data
        target[0][action] = reward if done else reward + self.discount_factor * torch.max(target_val)

        error = abs(old_val - target[0][action])  # 计算TD误差绝对值
        self.memory.add(error, (state, action, reward, next_state, done))  # 存入经验池

    # 训练主网络
    def train_model(self):
        if self.epsilon > self.epsilon_min:  # 线性衰减探索率
            self.epsilon -= self.epsilon_decay

        # 从优先经验池采样
        mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
        mini_batch = np.array(mini_batch).transpose()

        # 解包批量数据
        states = np.vstack(mini_batch[0])  # 当前状态
        actions = list(mini_batch[1])  # 执行动作
        rewards = list(mini_batch[2])  # 即时奖励
        next_states = np.vstack(mini_batch[3])  # 下一状态
        dones = mini_batch[4]  # 终止标志

        # 转换为PyTorch张量
        states = Variable(torch.Tensor(states).float())
        next_states = Variable(torch.Tensor(next_states).float())
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones.astype(int))

        # 主网络预测Q值
        pred = self.model(states)

        # 使用one-hot编码选择执行动作的Q值
        a = torch.LongTensor(actions).view(-1, 1)
        one_hot_action = torch.FloatTensor(self.batch_size, self.action_size).zero_()
        one_hot_action.scatter_(1, a, 1)
        pred = torch.sum(pred.mul(Variable(one_hot_action)), dim=1)

        # 目标网络计算目标Q值
        next_pred = self.target_model(next_states).data
        target = rewards + (1 - dones) * self.discount_factor * next_pred.max(1)[0]
        target = Variable(target)

        # 计算优先级并更新经验池
        errors = torch.abs(pred - target).data.numpy()
        for i in range(self.batch_size):
            self.memory.update(idxs[i], errors[i])

        # 计算带权重的MSE损失
        self.optimizer.zero_grad()
        loss = (torch.FloatTensor(is_weights) * F.mse_loss(pred, target)).mean()
        loss.backward()
        self.optimizer.step()  # 反向传播更新参数


if __name__ == "__main__":
    # 创建CartPole环境
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 初始化智能体
    agent = DQNAgent(state_size, action_size)
    scores, episodes = [], []

    # 训练循环
    for e in range(EPISODES):
        done = False
        score = 0
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render: env.render()

            # 与环境交互
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape([1, state_size])

            # 调整终止状态奖励
            reward = reward if not done or score == 499 else -10

            # 存储经验并训练
            agent.append_sample(state, action, reward, next_state, done)
            if agent.memory.tree.n_entries >= agent.train_start:
                agent.train_model()

            score += reward
            state = next_state

            if done:
                # 更新目标网络并记录训练进度
                agent.update_target_model()
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/cartpole_dqn.png")
                print(f"episode:{e} score:{score} memory:{agent.memory.tree.n_entries} ε:{agent.epsilon:.4f}")

                # 早停条件（近10轮平均分>490）
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    torch.save(agent.model, "./save_model/cartpole_dqn")
                    sys.exit()