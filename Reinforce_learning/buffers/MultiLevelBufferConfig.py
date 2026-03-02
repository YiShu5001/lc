"""
多层经验池配置
"""
from dataclasses import dataclass
from typing import Optional

from Reinforce_learning.buffers.BaseBuffer import BufferConfig


@dataclass
class MultiLevelBufferConfig:
    """
    多层经验池配置
    """
    # 各池容量
    coverage_capacity: int = 1_000_000      # 基座覆盖池容量
    difficulty_capacity: int = 100_000      # 难点聚焦池容量
    keyevent_capacity: int = 10_000         # 关键事件池容量
    
    # 采样配置
    batch_size: int = 64                    # 批次大小
    pool_weights: list = None               # 各池采样权重 [coverage, difficulty, keyevent]
    
    # 筛选配置
    filter_freq: int = 1000                 # 筛选频率（每N步执行一次）
    td_error_threshold: float = 0.1         # TD误差阈值
    value_threshold: float = 0.7             # 价值性阈值
    
    # 指标配置
    use_priority: bool = True               # 是否使用优先回放
    alpha: float = 0.6                      # 优先级指数
    beta: float = 0.4                       # 重要性采样指数
    beta_increment: float = 0.001           # beta增量
    
    def __post_init__(self):
        """初始化后处理"""
        if self.pool_weights is None:
            # 默认权重：基座50%，难点30%，关键20%
            self.pool_weights = [0.5, 0.3, 0.2]
