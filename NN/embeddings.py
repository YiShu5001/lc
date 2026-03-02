"""
嵌入层模块
包含自身状态、障碍物和邻近无人机的嵌入层
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


class SelfEmbedding(nn.Module):
    """
    自身状态嵌入层
    
    将自身状态信息映射到统一的嵌入空间
    """
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: 自身状态输入维度
            embed_dim: 嵌入维度
            dropout: Dropout概率
        """
        super().__init__()
        
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, self_state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            self_state: 自身状态，shape = (batch_size, input_dim)
        
        Returns:
            嵌入向量，shape = (batch_size, embed_dim)
        """
        return self.embedding(self_state)


class ObstacleEmbedding(nn.Module):
    """
    障碍物集合嵌入层
    
    将障碍物信息（可能是多个障碍物）映射到统一的嵌入空间
    支持处理可变数量的障碍物
    """
    
    def __init__(
        self,
        obstacle_dim: int,
        embed_dim: int,
        max_obstacles: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Args:
            obstacle_dim: 单个障碍物的特征维度
            embed_dim: 嵌入维度
            max_obstacles: 最大障碍物数量（如果为None，则支持任意数量）
            dropout: Dropout概率
        """
        super().__init__()
        
        self.obstacle_dim = obstacle_dim
        self.embed_dim = embed_dim
        self.max_obstacles = max_obstacles
        
        # 单个障碍物的嵌入层
        self.obstacle_embedding = nn.Sequential(
            nn.Linear(obstacle_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 如果障碍物数量固定，可以使用聚合层
        if max_obstacles is not None:
            self.aggregation = nn.Sequential(
                nn.Linear(max_obstacles * embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            # 对于可变数量，使用平均池化或最大池化
            self.aggregation = None
        
    def forward(self, obstacles: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            obstacles: 障碍物集合
                - 如果max_obstacles不为None: shape = (batch_size, max_obstacles, obstacle_dim)
                - 如果max_obstacles为None: shape = (batch_size, num_obstacles, obstacle_dim)
        
        Returns:
            嵌入向量，shape = (batch_size, embed_dim)
        """
        batch_size = obstacles.shape[0]
        
        # 对每个障碍物进行嵌入
        # obstacles shape: (batch_size, num_obstacles, obstacle_dim)
        embedded = self.obstacle_embedding(obstacles)
        # embedded shape: (batch_size, num_obstacles, embed_dim)
        
        if self.aggregation is not None:
            # 固定数量障碍物：展平后通过全连接层
            embedded_flat = embedded.view(batch_size, -1)
            return self.aggregation(embedded_flat)
        else:
            # 可变数量障碍物：使用平均池化
            return embedded.mean(dim=1)


class NeighborEmbedding(nn.Module):
    """
    邻近无人机信息嵌入层
    
    将邻近无人机信息（可能是多个邻近无人机）映射到统一的嵌入空间
    支持处理可变数量的邻近无人机
    """
    
    def __init__(
        self,
        neighbor_dim: int,
        embed_dim: int,
        max_neighbors: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Args:
            neighbor_dim: 单个邻近无人机的特征维度
            embed_dim: 嵌入维度
            max_neighbors: 最大邻近无人机数量（如果为None，则支持任意数量）
            dropout: Dropout概率
        """
        super().__init__()
        
        self.neighbor_dim = neighbor_dim
        self.embed_dim = embed_dim
        self.max_neighbors = max_neighbors
        
        # 单个邻近无人机的嵌入层
        self.neighbor_embedding = nn.Sequential(
            nn.Linear(neighbor_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 如果邻近无人机数量固定，可以使用聚合层
        if max_neighbors is not None:
            self.aggregation = nn.Sequential(
                nn.Linear(max_neighbors * embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            # 对于可变数量，使用平均池化
            self.aggregation = None
        
    def forward(self, neighbors: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            neighbors: 邻近无人机集合
                - 如果max_neighbors不为None: shape = (batch_size, max_neighbors, neighbor_dim)
                - 如果max_neighbors为None: shape = (batch_size, num_neighbors, neighbor_dim)
        
        Returns:
            嵌入向量，shape = (batch_size, embed_dim)
        """
        batch_size = neighbors.shape[0]
        
        # 对每个邻近无人机进行嵌入
        # neighbors shape: (batch_size, num_neighbors, neighbor_dim)
        embedded = self.neighbor_embedding(neighbors)
        # embedded shape: (batch_size, num_neighbors, embed_dim)
        
        if self.aggregation is not None:
            # 固定数量邻近无人机：展平后通过全连接层
            embedded_flat = embedded.view(batch_size, -1)
            return self.aggregation(embedded_flat)
        else:
            # 可变数量邻近无人机：使用平均池化
            return embedded.mean(dim=1)
