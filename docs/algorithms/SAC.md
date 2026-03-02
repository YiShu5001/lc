# SAC (Soft Actor-Critic) 算法

## 算法原理

SAC是一种off-policy的最大熵强化学习算法，平衡探索和利用。

### 核心思想

SAC的核心思想是**最大熵强化学习**，在最大化累积奖励的同时最大化策略熵。

### 数学公式

SAC的目标函数：

$$\pi^* = \arg\max_\pi \mathbb{E}_\pi[\sum_t r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))]$$

其中：
- $\alpha$ 是温度参数（可自动调整）
- $\mathcal{H}$ 是熵

### 算法特点

1. **最大熵**：鼓励探索，提高鲁棒性
2. **Off-policy**：可以使用历史经验
3. **自动温度调整**：自动调整探索程度
4. **双Q网络**：减少过估计

## 实现细节

### 超参数

- `tau`: 软更新系数，默认0.005
- `alpha`: 温度参数（None表示自动调整）
- `gamma`: 折扣因子，默认0.99
- `target_update_interval`: 目标网络更新间隔

### 使用示例

```python
from Reinforce_learning.RLg.SAC import SACAlgo, SACConfig

algo_cfg = SACConfig(
    learning_rate=3e-4,
    tau=0.005,
    alpha=None  # 自动调整
)
algo = SACAlgo(algo_cfg)
```

## 适用场景

- 连续动作空间
- 需要大量探索的任务
- 对样本效率要求高的场景
