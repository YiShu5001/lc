"""Backward-compatible bridge for legacy NN base abstractions."""

from src.lc.rl.models.base import (
    ActOutput,
    ActionDist,
    BaseRLModel,
    EvalOutput,
    ModelConfig,
)

__all__ = [
    "ModelConfig",
    "ActionDist",
    "ActOutput",
    "EvalOutput",
    "BaseRLModel",
]
