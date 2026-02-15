"""
神经网络模块
包含基础RL模型和多无人机协调规划模型
"""
from .BaseNN import (
    BaseRLModel,
    ModelConfig,
    ActionDist,
    ActOutput,
    EvalOutput
)

from .components import (
    MultiHeadAttention,
    TransformerEncoderBlock
)

from .embeddings import (
    SelfEmbedding,
    ObstacleEmbedding,
    NeighborEmbedding
)

from .obstacle_branch import (
    ObstacleAvoidanceBranch
)

from .collaborative_branch import (
    CollaborativeBranch
)

from .MultiUAVModel import (
    MultiUAVModel,
    MultiUAVModelConfig,
    SigmoidNormal
)

__all__ = [
    # 基础类
    "BaseRLModel",
    "ModelConfig",
    "ActionDist",
    "ActOutput",
    "EvalOutput",
    # 核心组件
    "MultiHeadAttention",
    "TransformerEncoderBlock",
    # 嵌入层
    "SelfEmbedding",
    "ObstacleEmbedding",
    "NeighborEmbedding",
    # 分支模块
    "ObstacleAvoidanceBranch",
    "CollaborativeBranch",
    # 主模型
    "MultiUAVModel",
    "MultiUAVModelConfig",
    "SigmoidNormal",
]
