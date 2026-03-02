# 快速开始指南

## 安装

```bash
pip install torch numpy gymnasium
```

## 基本使用

### 1. 使用命令行训练

```bash
python main.py --algo ppo --env-id HoverAviary-v0 --total-timesteps 1000000
```

### 2. 使用Python脚本训练

```python
from examples.train_pybullet import train_pybullet_ppo
train_pybullet_ppo()
```

### 3. 自定义训练

```python
import torch
import torch.optim as optim
from Gym_env.factories.PyBulletDronesFactory import PyBulletDronesFactory
from NN.model_factory import create_model_from_env
from Reinforce_learning.algo_factory import create_algo
from Trainer.BaseTrainer import Trainer

# 创建环境
env_factory = PyBulletDronesFactory(env_cfg)
envs = env_factory.build()

# 创建模型
model = create_model_from_env(...)

# 创建算法
algo = create_algo("ppo")

# 创建训练器
trainer = Trainer(envs, model, algo, optimizer, cfg)
trainer.train()
```

## 支持的算法

- PPO (on-policy)
- A2C (on-policy)
- SAC (off-policy)
- TD3 (off-policy)
- DDPG (off-policy)
- DQN (off-policy)

## 支持的环境

- PyBullet Drones (HoverAviary, MultiHoverAviary)
- Gymnasium标准环境 (CartPole, MountainCar等)
