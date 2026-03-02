import numpy

# 实现Sum Tree数据结构，用于优先级经验回放
class SumTree:
    write = 0  # 当前写入位置的指针

    def __init__(self, capacity):
        self.capacity = capacity  # 树的总容量
        # 初始化树结构，长度为2*capacity-1的数组
        # 前capacity-1个节点是非叶子节点，后capacity个是叶子节点
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)  # 存储实际数据
        self.n_entries = 0  # 当前存储的数据量

    # 从叶子节点向上传播  更新值: change
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2  # 计算父节点索引
        self.tree[parent] += change
        if parent != 0:  # 递归更新直到根节点
            self._propagate(parent, change)

    # 根据采样值检索对应的叶子节点
    def _retrieve(self, idx, s):
        left = 2 * idx + 1  # 左子节点索引
        right = left + 1  # 右子节点索引

        if left >= len(self.tree):  # 到达叶子节点
            return idx

        if s <= self.tree[left]:  # 向左子树搜索
            return self._retrieve(left, s)
        else:  # 向右子树搜索，并减去左子树的值
            return self._retrieve(right, s - self.tree[left])

    # 获取当前优先级总和
    def total(self):
        return self.tree[0]  # 根节点存储总和

    # 添加新的数据和优先级
    def add(self, p, data):
        idx = self.write + self.capacity - 1  # 计算叶子节点位置 (0- capacity-1  是父节点， capacity-1 到 2*capacity-2 是叶子节点 )
        self.data[self.write] = data  # 存储数据
        self.update(idx, p)  # 更新树结构

        self.write += 1  # 移动写入指针
        if self.write >= self.capacity:  # 循环覆盖旧数据
            self.write = 0
        if self.n_entries < self.capacity:  # 更新当前数据量
            self.n_entries += 1

    # 更新指定位置的优先级值
    def update(self, idx, p):
        change = p - self.tree[idx]  # 计算优先级变化量
        self.tree[idx] = p  # 更新当前节点
        self._propagate(idx, change)  # 向上传播变化

    # 根据采样值获取数据和对应信息
    def get(self, s):
        idx = self._retrieve(0, s)  # 从根节点开始检索
        dataIdx = idx - self.capacity + 1  # 计算数据索引
        return (idx, self.tree[idx], self.data[dataIdx])