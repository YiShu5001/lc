"""
避障动作分支模块
处理自身状态和障碍物信息，生成避障动作
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from .components import TransformerEncoderBlock
from .embeddings import SelfEmbedding, ObstacleEmbedding


class ObstacleAvoidanceBranch(nn.Module):
    """
    避障动作分支
    
    接收Self Embedding和Obstacle Embedding，
    通过Transformer Encoder Block处理，
    输出避障动作
    """
    
    def __init__(
        self,
        self_dim: int,
        obstacle_dim: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        action_dim: int,
        max_obstacles: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        Args:
            self_dim: 自身状态维度
            obstacle_dim: 单个障碍物特征维度
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            ff_dim: 前馈网络隐藏层维度
            action_dim: 动作维度
            max_obstacles: 最大障碍物数量
            dropout: Dropout概率
            activation: 激活函数类型
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # 嵌入层
        self.self_embedding = SelfEmbedding(self_dim, embed_dim, dropout)
        self.obstacle_embedding = ObstacleEmbedding(
            obstacle_dim, embed_dim, max_obstacles, dropout
        )
        
        # Transformer编码器块
        # 注意：我们需要访问Feed Forward的输出，所以不能直接使用TransformerEncoderBlock
        # 而是需要分别实现各个组件
        from .components import MultiHeadAttention
        
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Feed Forward网络
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 动作输出层
        # 根据架构图，需要将Self和Obstacle的嵌入合并后输入编码器
        # 这里我们将它们concatenate后通过一个线性层映射到embed_dim
        self.input_proj = nn.Linear(embed_dim * 2, embed_dim)
        
        # 输出层：Linear + sigmoid
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, action_dim),
            nn.Sigmoid()
        )
    
    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
    def forward(
        self,
        self_state: torch.Tensor,
        obstacles: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            self_state: 自身状态，shape = (batch_size, self_dim)
            obstacles: 障碍物集合
                - 如果max_obstacles不为None: shape = (batch_size, max_obstacles, obstacle_dim)
                - 如果max_obstacles为None: shape = (batch_size, num_obstacles, obstacle_dim)
            mask: 可选的注意力掩码
        
        Returns:
            避障动作，shape = (batch_size, action_dim)
        """
        batch_size = self_state.shape[0]
        
        # 嵌入
        self_emb = self.self_embedding(self_state)  # (batch_size, embed_dim)
        obstacle_emb = self.obstacle_embedding(obstacles)  # (batch_size, embed_dim)
        
        # 合并嵌入（concatenate）
        combined = torch.cat([self_emb, obstacle_emb], dim=-1)  # (batch_size, embed_dim * 2)
        x = self.input_proj(combined)  # (batch_size, embed_dim)
        
        # 为了输入Transformer编码器，需要添加序列维度
        # 将x扩展为 (batch_size, 1, embed_dim)
        x = x.unsqueeze(1)
        
        # Multi-Head Attention + Add & Norm
        attn_output = self.self_attn(x, x, x, mask)  # (batch_size, 1, embed_dim)
        x = self.norm1(x + attn_output)  # (batch_size, 1, embed_dim)
        
        # Feed Forward（需要保存这个输出用于协作分支融合）
        ff_output = self.ff(x)  # (batch_size, 1, embed_dim)
        
        # Add & Norm
        x = self.norm2(x + ff_output)  # (batch_size, 1, embed_dim)
        
        # 移除序列维度
        x = x.squeeze(1)  # (batch_size, embed_dim)
        ff_output = ff_output.squeeze(1)  # (batch_size, embed_dim)
        
        # 输出避障动作（基于最终的x）
        action = self.action_head(x)  # (batch_size, action_dim)
        
        # 返回动作和Feed Forward输出（用于协作分支融合）
        return action, ff_output
