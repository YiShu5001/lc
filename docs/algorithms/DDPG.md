# DDPG (Deep Deterministic Policy Gradient) 算法

## 算法原理

DDPG是一种off-policy的Actor-Critic算法，适用于连续动作空间。

### 核心思想

DDPG将DQN扩展到连续动作空间，使用确定性策略和Q函数。

### 数学公式

Critic损失：

$$L_Q = \mathbb{E}[(Q(s,a) - (r + \gamma Q'(s', \mu'(s'))))^2]$$

Actor损失：

$$L_\mu = -\mathbb{E}[Q(s, \mu(s))]$$

### 算法特点

1. **确定性策略**：输出确定性动作
2. **目标网络**：使用软更新稳定训练
3. **经验回放**：off-policy学习
4. **噪声探索**：在动作上添加噪声

## 实现细节

### 超参数

- `tau`: 软更新系数，默认0.005
- `gamma`: 折扣因子，默认0.99
- `noise_std`: 动作噪声标准差，默认0.1

### 使用示例

```python
from Reinforce_learning.RLg.DDPG_refactored import DDPGAlgo, DDPGConfig

algo_cfg = DDPGConfig(
    learning_rate=3e-4,
    tau=0.005
)
algo = DDPGAlgo(algo_cfg)
```

## 适用场景

- 连续动作空间
- 需要确定性策略的任务
- 样本效率要求高的场景
