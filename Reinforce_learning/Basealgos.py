import os
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAlgo(ABC):
    """
    强化学习算法基类 (统一外部接口规范)
    所有派生的 RL 算法 (DQN, DDPG, TD3, SAC, PPO, A2C 等) 都应继承该类，并实现其核心方法。
    这保证了外部调用者 (如 Trainer) 可以以完全一致的方式进行交互。
    """
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # 存放模型实例的字典，供统一的 save/load 接口使用
        # 比如: self.models = {"actor": self.actor, "critic": self.critic}
        self.models = {}

    @abstractmethod
    def select_action(self, state: Any, evaluate: bool = False) -> Any:
        """
        根据当前状态选择动作。
        
        :param state: 当前状态
        :param evaluate: 是否为评估模式 (默认为 False，即训练模式，通常带有探索噪声或基于概率采样。
                         如果为 True，通常输出确定性的最优动作，或者关闭探索噪声)
        :return: 选择的动作 (通常为 numpy.ndarray，形状视具体动作空间而定)
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Dict[str, float]:
        """
        执行一次网络参数的更新。
        
        针对 Off-policy 算法 (如 TD3/SAC)，通常传入 replay_buffer 和 batch_size。
        针对 On-policy 算法 (如 PPO/A2C)，通常传入收集到的 rollouts 轨迹。
        
        :return: 包含损失、熵等训练指标的字典 (如 {"actor_loss": 0.5, "critic_loss": 0.2})，用于日志记录。
        """
        pass

    def save(self, filename: str):
        """
        统一的模型保存接口。将所有注册在 self.models 中的网络保存到指定文件。
        :param filename: 保存的文件前缀路径 (例如 "./models/checkpoint")
        """
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        for name, model in self.models.items():
            if model is not None:
                torch.save(model.state_dict(), f"{filename}_{name}.pth")
        print(f"[INFO] Models saved successfully with prefix: {filename}")

    def load(self, filename: str, device: str = "cpu"):
        """
        统一的模型加载接口。从指定文件加载所有注册在 self.models 中的网络。
        :param filename: 加载的文件前缀路径
        :param device: 加载时映射的设备 (默认 CPU)
        """
        for name, model in self.models.items():
            if model is not None:
                file_path = f"{filename}_{name}.pth"
                if os.path.exists(file_path):
                    model.load_state_dict(torch.load(file_path, map_location=device))
                    print(f"[INFO] Loaded model: {file_path}")
                else:
                    print(f"[WARNING] Cannot find model file: {file_path}")
