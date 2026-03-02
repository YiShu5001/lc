# PPO (Proximal Policy Optimization) 算法

## 算法原理

PPO是一种on-policy的强化学习算法，通过限制策略更新的幅度来稳定训练过程。

### 核心思想

PPO的核心思想是**限制策略更新的幅度**，避免策略更新过大导致性能崩溃。

### 数学公式

PPO-Clip的目标函数：

$$L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是重要性采样比率
- $\hat{A}_t$ 是优势函数估计（GAE）
- $\epsilon$ 是clip参数（通常0.2）

### 算法特点

1. **稳定训练**：通过clip机制防止策略更新过大
2. **样本效率**：支持多epoch更新，提高样本利用率
3. **易于实现**：相比TRPO更简单，性能相近

## 实现细节

### 超参数

- `clip_epsilon`: clip参数，默认0.2
- `value_coef`: 价值函数损失系数，默认0.5
- `entropy_coef`: 熵正则化系数，默认0.01
- `num_epochs`: 每次rollout后的更新轮数，默认4
- `max_grad_norm`: 梯度裁剪，默认0.5

### 使用示例

```python
from Reinforce_learning.RLg.PPO import PPOAlgo, PPOConfig

algo_cfg = PPOConfig(
    learning_rate=3e-4,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01
)
algo = PPOAlgo(algo_cfg)
```

## 适用场景

- 连续和离散动作空间
- 需要稳定训练的场景
- 样本效率要求较高的场景
