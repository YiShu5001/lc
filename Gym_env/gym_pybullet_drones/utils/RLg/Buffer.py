import numpy as np
import random


class SumTree:
    """Sum Tree数据结构实现"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(left + 1, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])


class PriorityReplayBuffer:
    """优先经验回放池"""

    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.tree = SumTree(capacity)
        self.alpha = alpha  # 控制采样优先程度 (0~1)
        self.beta = beta  # 控制重要性采样权重调整
        self.beta_increment = beta_increment
        self.capacity = capacity
        self.max_priority = 1.0  # 初始优先级

    def _get_priority(self, error):
        return (error + 1e-5) ** self.alpha

    def add(self, experience):
        """添加经验"""
        priority = self.max_priority  # 新经验的最高优先级
        self.tree.add(priority, experience)

    def sample(self, batch_size):
        """采样批次数据"""
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_prob = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_prob, -self.beta)
        is_weights /= is_weights.max()

        return batch, idxs, is_weights

    def update_priorities(self, idxs, errors):
        """更新优先级"""
        for idx, error in zip(idxs, errors):
            priority = self._get_priority(error)
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries