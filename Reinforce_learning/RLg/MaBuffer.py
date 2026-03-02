import numpy as np
import random
from prioritized_memory import Memory


class MultiLevelBuffer:
    def __init__(self, capacity=None, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-5):
        if capacity is None:
            capacity = [1e5, 5e4, 25e3]
        self.capacity = [int(c) for c in capacity]  # 转换为整数容量列表
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        # 创建三级记忆池（主池、次级池、精筛池）
        self.memory = [Memory(cap) for cap in self.capacity]  # 初始化三级存储结构
        # 实现优先经验回放（Prioritized Experience Replay）的内存池
        self.memory = Memory(capacity)

        # 各层的容量
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        # 保持原有参数兼容性
        self.uncertainty_decay = 0.95
        self.time_decay = 0.99

    def add(self, experience, leave:str=None):
        if len(self.memory[0]) < self.capacity[0]:
            # 当前池没有满，仅存入当前池
            self.memory[0].add(experience)
        else:
            # 当前池已满，转移到次级池
            PastS = self.memory[0].add(experience)
            leave +=1
            if leave != len(self.capacity):



        # 计算初始优先级
        priority = (np.max([tree.tree.total() for tree in self.memory]) + self.epsilon) ** self.alpha
        # 存储到主池


    def add(self, experience):
        max_priority = np.max([tree.max_priority() for tree in
                               [self.main_tree, self.uncertainty_tree, self.time_tree]] or [1.0])

        # 初始优先级设置
        main_priority = (max_priority + self.epsilon) ** self.alpha
        uncertainty_priority = 1.0  # 初始不确定性最大
        time_priority = 1.0  # 最新样本时间优先级最高

        self.main_tree.add(main_priority, experience)
        self.uncertainty_tree.add(uncertainty_priority, experience)
        self.time_tree.add(time_priority, experience)

    def sample(self, batch_size):
        # 三层采样比例分配
        main_batch = int(batch_size * 0.5)
        uncertainty_batch = int(batch_size * 0.3)
        time_batch = batch_size - main_batch - uncertainty_batch

        samples = []
        for tree, size in zip([self.main_tree, self.uncertainty_tree, self.time_tree],
                              [main_batch, uncertainty_batch, time_batch]):
            if size > 0:
                segment = tree.total() / size
                samples += [tree.get(random.uniform(segment * i, segment * (i + 1)))
                            for i in range(size)]

        # 计算重要性采样权重
        priorities = np.array([p for (p, _, _, idx) in samples])
        weights = np.power(len(self) * priorities / self.main_tree.total(), -self.beta)
        weights /= weights.max()

        self.beta = np.min([1.0, self.beta + self.beta_increment])

        return [(data, weight, idx) for (_, data, _, idx), weight in zip(samples, weights)], indices

    def update_priorities(self, indices, td_errors, uncertainties=None):
        # 更新主树优先级
        priorities = np.power(np.abs(td_errors) + self.epsilon, self.alpha)
        for idx, priority in zip(indices, priorities):
            self.main_tree.update(idx, priority)

        # 更新不确定性树（如果有）
        if uncertainties is not None:
            uncertainties = np.power(uncertainties * self.uncertainty_decay, self.alpha)
            for idx, unc in zip(indices, uncertainties):
                current = self.uncertainty_tree.get_value(idx) * self.uncertainty_decay
                self.uncertainty_tree.update(idx, max(current, unc))

        # 更新时间树（自动衰减）
        for idx in indices:
            current = self.time_tree.get_value(idx) * self.time_decay
            self.time_tree.update(idx, current)

    def __len__(self):
        return int(self.main_tree.total())