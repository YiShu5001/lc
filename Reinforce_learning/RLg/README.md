# HGC-RL Framework: Core RL Algorithms (`Reinforce_learning/RLg/`)

此目录包含了用于与 `TSA_LADRC_Env` (或者任何其他 OpenAI Gym 环境) 进行对接的经典强化学习算法的纯净 PyTorch 实现。每个算法均经过结构解耦和充分的代码注释，目的是为了让你能在后续的主题研究（控制论文及消融实验）中作为基线对比模型。

## 包含的算法概览

所有算法模型主要分为**异策略 (Off-Policy)** 与 **同策略 (On-Policy)** 两种范式：

### 1. 异策略连续控制 (Off-Policy Continuous Control)
此类算法将数据存储在经验池 (Replay Buffer) 中并从中随机采样学习，数据利用效率极高，非常适合无人机控制等计算量昂贵的场景。

*   **DDPG (Deep Deterministic Policy Gradient)** (`DDPG.py`):
    *   **特点**: 将 DQN 拓展到了连续动作空间。Actor 负责输出确定的动作，Critic 负责对其进行 Q 值评估。
    *   **优缺点**: 设计简单，但在复杂任务中极易受到 Q 值高估 (Overestimation) 的影响，导致收敛崩溃。
*   **TD3 (Twin Delayed DDPG)** (`TD3.py`):
    *   **特点**: DDPG 的强力进化版，加入了**三把斧** (双Q网络截断、目标策略平滑、延迟Actor更新)。
    *   **用途**: 目前非概率性连续控制的最佳基线，表现极为稳定，是测试 TSA-RL 参数整定环境的最佳起手算法之一。
*   **SAC (Soft Actor-Critic)** (`SAC.py`):
    *   **特点**: 基于最大熵 (Maximum Entropy) 目标的最前沿算法。不仅追求最大化回报，还力求动作策略具备足够的随机探索性（熵）。具有自动调节温度参数 (Alpha) 的能力。
    *   **用途**: 能够探索出更为鲁棒的动作参数组合，抗扰动能力极强，非常适合无人机在面临下洗流或风扰时的控制参数自适应。

### 2. 同策略控制 (On-Policy Control)
此类算法在生成数据和更新模型时保持一致，即使用当前的策略来采集数据。更新后数据被丢弃（或很少被重复利用）。

*   **A2C (Advantage Actor-Critic)** (`A2C.py`):
    *   **特点**: PPO 的基础。通过 Critic 估计的 Value 作为 Baseline，极大降低了策略梯度的方差。
    *   **用途**: 代码最为轻量级，通常用于对比 On-Policy 和 Off-Policy 基础差异时的对照组。
*   **PPO (Proximal Policy Optimization)** (`PPO.py`):
    *   **特点**: 工业界的标杆 (On-policy王者)。通过引入 Clip 截断机制，使得训练在即使数据更新步长很大时也不会偏离太远，彻底解决了策略崩塌问题。
    *   **用途**: 泛化能力强，对超参数调优极其不敏感。如果你的实验计算资源充足，PPO 是一个无需太多调试的万金油。

### 3. 离散控制 (Discrete Control)
*   **DQN (Deep Q-Network)** (`DQN.py`):
    *   **特点**: 仅适用于离散动作空间，采用 $\epsilon$-贪心 策略。
    *   **注意**:由于我们的 `TSA_LADRC_Env` 需要输出连续的 $\Delta \omega_c$ 和 $\Delta b_0$ (共计12维)，DQN **无法直接应用于当前环境**，除非对动作空间做离散化处理（例如将动作转变为 `[增, 减, 保持]` 的组合）。保留它是为了框架的完整性。

## 如何在你的项目中调用它们？

这些算法被设计为极简形式，方便直接放入统一的 Trainer 循环中。这里有一个伪代码的使用流程示例：

```python
import numpy as np
from Gym_env.gym_pybullet_drones.envs.TSA_LADRC_Env import TSA_LADRC_Env
from Reinforce_learning.RLg.TD3 import TD3
from Reinforce_learning.buffers.BaseBuffer import ReplayBuffer # 假设你已有这个池

# 1. 实例化环境
env = TSA_LADRC_Env(ctrl_freq=100, rl_freq=10)
state_dim = env.observation_space.shape[1] # 对于多智能体获取相应的维度
action_dim = env.action_space.shape[1]
max_action = float(env.action_space.high[0][0])

# 2. 实例化算法 (以 TD3 为例)
agent = TD3(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=1e6)

# 3. 训练循环 (Training Loop)
obs, _ = env.reset()
# (省略了多智能体/batch维度的适配处理，仅展示概念)
state = obs[0] 

for t in range(int(1e6)):
    # 动作选择 (并添加探索噪声)
    action = agent.select_action(state)
    action = (action + np.random.normal(0, 0.1, size=action_dim)).clip(-max_action, max_action)
    
    # 环境交互
    next_obs, reward, terminated, truncated, _ = env.step(action)
    next_state = next_obs[0]
    done = terminated or truncated

    # 存入经验池
    replay_buffer.add(state, action, next_state, reward, done)
    state = next_state
    
    # 算法更新网络
    if replay_buffer.size > batch_size:
        loss_metrics = agent.update(replay_buffer, batch_size)

    if done:
        obs, _ = env.reset()
        state = obs[0]
```

## 下一步 (Phase 2):
你可以基于上面的这些基线模型之一，开发专属于你论文创新点的衍生算法（例如将分层/注意力机制组合到上述某个基线的 Actor 网络中），通过继承基类的思想去打造 `TSA_LADRC` 的专属优化算法！