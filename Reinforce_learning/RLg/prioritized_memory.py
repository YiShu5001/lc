import random
import numpy as np
from SumTree import SumTree


class Memory:  # 实现优先经验回放（Prioritized Experience Replay）的内存池
    # 超参数设置（参见PER论文）
    e = 0.01  # 最小优先级常数，避免零概率采样
    a = 0.6  # 优先级调节因子（0表示均匀采样，1表示完全优先级）
    beta = 0.4  # 重要性采样修正系数初始值
    beta_increment_per_sampling = 0.001  # beta的递增系数

    def __init__(self, capacity):
        self.tree = SumTree(capacity)  # 初始化SumTree数据结构
        self.capacity = capacity  # 经验池最大容量

    # 计算样本优先级（PER核心公式）
    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    # 添加新经验到内存池
    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)  # 存储样本及其优先级

    # 采样批处理数据（核心采样逻辑）
    def sample(self, n):
        batch = []
        idxs = []  # 存储样本索引
        segment = self.tree.total() / n  # 将优先级总和分成n段
        priorities = []  # 存储采样到的优先级值

        # 动态调整beta值（重要性采样修正系数）
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        # 分层采样过程
        for i in range(n):
            a = segment * i  # 区间左边界
            b = segment * (i + 1)  # 区间右边界
            s = random.uniform(a, b)  # 在区间内随机采样

            # 从SumTree中获取样本
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        # 计算重要性采样权重（IS weights）
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()  # 归一化处理

        return batch, idxs, is_weight  # 返回样本、索引和修正权重

    # 更新样本优先级（通常在训练后调用）
    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)  # 更新SumTree中的优先级