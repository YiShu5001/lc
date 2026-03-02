# 自定义环境教程

## 实现VectorEnvLike接口

要实现自定义环境，需要继承`VectorEnvLike`并实现以下方法：

```python
from Gym_env.BaseEnv import VectorEnvLike
import numpy as np

class MyCustomEnv(VectorEnvLike):
    @property
    def num_envs(self) -> int:
        return 1
    
    @property
    def obs_shape(self) -> Tuple[int, ...]:
        return (4,)
    
    @property
    def action_shape(self) -> Tuple[int, ...]:
        return ()
    
    @property
    def is_discrete(self) -> bool:
        return True
    
    @property
    def action_dim(self) -> int:
        return 2
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        # 返回 shape = (num_envs, *obs_shape)
        pass
    
    def step(self, actions: np.ndarray) -> Tuple:
        # 返回 (obs, rewards, dones, infos)
        pass
```

## 使用Gymnasium环境

如果已有gymnasium环境，可以使用`GymnasiumVectorEnv`包装：

```python
import gymnasium as gym
from Gym_env.wrappers.GymnasiumWrapper import GymnasiumVectorEnv

env = gym.make("CartPole-v1")
wrapped_env = GymnasiumVectorEnv(env)
```
