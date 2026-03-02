from collections import deque, defaultdict

import numpy as np
import random
from prioritized_memory import Memory
import dotenv
import os
dotenv.load_dotenv()

# 环境变量
cts = os.getenv('cts')

class ob():   # 定义观测类
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.EX = self.to_dict()
#将类转化为字典                            输出名 self.EX
    def to_dict(self):
        return {
            'state': self.state,
            'action': self.action,
            'reward': self.reward,
            'next_state': self.next_state,
            'done': self.done
        }

class MultiLevelBuffer:
    def __init__(self, capacity:int = 2*cts+1, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-5):
        # 初始时间观测表 deque
        self.OBqueue: deque[ob] = deque(maxlen=capacity,)
        # 优先级队列
        self.memory = Memory(capacity)

        # 超参数
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.capacity = capacity
        self.uncertainty_decay = 0.95
        self.time_decay = 0.99

    # 第一层 保存观测状态记录到deque
    def getOB(self, obs:ob=None):
        self.OBqueue.append(obs)
        return self.OBqueue
    # 第二层 时间压缩状态为 样本
    def compress_state(self,):
        mid_T = len(self.OBqueue)//2
        StateCom = [i for i in self.OBqueue[:mid_T].state]
        next_StateCom = [i for i in self.OBqueue[mid_T+1:].state]
        action = self.OBqueue[mid_T].action
        reward = sum(0.97 ** i * reward_item for i, reward_item in enumerate(self.OBqueue[mid_T].reward))
        done = self.OBqueue[-1].done
        return {StateCom, next_StateCom, action, reward, done}
    # 从状态观测层计算优先级
    def compute_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha
    # 计算TD误差
    def _get_TD(self, Sample):  # 待定 sac 可能会有变动
        q_values = self.q_network.predict(Sample.state)
        next_q_values = self.target_network.predict(Sample.next_state)
        max_next_q = np.max(next_q_values, axis=1)
        target_q = Sample.reward + (1 - Sample.done) * self.gamma * max_next_q
        return np.abs(q_values[np.arange(len(Sample.state)), Sample.action] - target_q)

