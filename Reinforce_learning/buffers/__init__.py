"""Legacy buffer package exports with lazy compatibility loading."""

from __future__ import annotations

from .BaseBuffer import BaseBuffer, MultiAgentBuffer, PrioritizedReplayBuffer, ReplayBuffer

__all__ = [
    "BaseBuffer",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "MultiAgentBuffer",
    "MultiLevelBuffer",
    "MultiLevelBufferConfig",
    "BufferFactory",
]


def __getattr__(name: str):
    if name == "MultiLevelBuffer":
        from .multi_level.MultiLevelBuffer import MultiLevelBuffer

        return MultiLevelBuffer
    if name == "MultiLevelBufferConfig":
        from .MultiLevelBufferConfig import MultiLevelBufferConfig

        return MultiLevelBufferConfig
    if name == "BufferFactory":
        from .buffer_factory import BufferFactory

        return BufferFactory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
