"""
经验池工厂
根据配置创建不同类型的经验池
"""
from __future__ import annotations
from typing import Optional, Union

from Reinforce_learning.buffers.BaseBuffer import BaseBuffer, BufferConfig, ReplayBuffer, PrioritizedReplayBuffer
from Reinforce_learning.buffers.MultiLevelBufferConfig import MultiLevelBufferConfig
from Reinforce_learning.buffers.multi_level.MultiLevelBuffer import MultiLevelBuffer


class BufferFactory:
    """
    经验池工厂
    
    根据配置创建不同类型的经验池
    """
    
    @staticmethod
    def create_buffer(
        buffer_type: str = "replay",
        config: Optional[Union[BufferConfig, MultiLevelBufferConfig]] = None
    ) -> BaseBuffer:
        """
        创建经验池
        
        Args:
            buffer_type: 经验池类型
                - "replay": 标准经验回放
                - "prioritized": 优先经验回放
                - "multilevel": 多层经验池
            config: 配置对象
        
        Returns:
            经验池实例
        """
        if buffer_type == "replay":
            cfg = config if isinstance(config, BufferConfig) else BufferConfig()
            return ReplayBuffer(cfg)
        
        elif buffer_type == "prioritized":
            cfg = config if isinstance(config, BufferConfig) else BufferConfig()
            return PrioritizedReplayBuffer(cfg)
        
        elif buffer_type == "multilevel":
            cfg = config if isinstance(config, MultiLevelBufferConfig) else MultiLevelBufferConfig()
            return MultiLevelBuffer(cfg)
        
        else:
            raise ValueError(f"Unknown buffer type: {buffer_type}")
