"""
协作动作分支模块
处理邻近无人机信息，与避障分支融合，生成最终动作
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from .components import TransformerEncoderBlock
from .embeddings import NeighborEmbedding


class CollaborativeBranch(nn.Module):
    """
    协作动作分支
    
    接收Neighbor Embedding，
    通过Transformer Encoder Block处理，
    与避障分支的Feed Forward输出融合（加法），
    输出最终动作
    """
    
    def __init__(
        self,
        neighbor_dim: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        action_dim: int,
        max_neighbors: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        Args:
            neighbor_dim: 单个邻近无人机特征维度
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            ff_dim: 前馈网络隐藏层维度
            action_dim: 动作维度
            max_neighbors: 最大邻近无人机数量
            dropout: Dropout概率
            activation: 激活函数类型
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # 嵌入层
        self.neighbor_embedding = NeighborEmbedding(
            neighbor_dim, embed_dim, max_neighbors, dropout
        )
        
        # Transformer编码器块
        # 注意：我们需要访问第二个Add & Norm的输出，所以需要分别实现
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
        neighbor_state: torch.Tensor,
        obstacle_ff_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            neighbor_state: 邻近无人机集合
                - 如果max_neighbors不为None: shape = (batch_size, max_neighbors, neighbor_dim)
                - 如果max_neighbors为None: shape = (batch_size, num_neighbors, neighbor_dim)
            obstacle_ff_output: 避障分支的Feed Forward输出，shape = (batch_size, embed_dim)
            mask: 可选的注意力掩码
        
        Returns:
            最终动作，shape = (batch_size, action_dim)
        """
        # 嵌入
        neighbor_emb = self.neighbor_embedding(neighbor_state)  # (batch_size, embed_dim)
        
        # 为了输入Transformer编码器，需要添加序列维度
        # 将neighbor_emb扩展为 (batch_size, 1, embed_dim)
        x = neighbor_emb.unsqueeze(1)
        
        # Multi-Head Attention + Add & Norm
        attn_output = self.self_attn(x, x, x, mask)  # (batch_size, 1, embed_dim)
        x = self.norm1(x + attn_output)  # (batch_size, 1, embed_dim)
        
        # Feed Forward
        ff_output = self.ff(x)  # (batch_size, 1, embed_dim)
        
        # Add & Norm
        x = self.norm2(x + ff_output)  # (batch_size, 1, embed_dim)
        
        # 移除序列维度
        x = x.squeeze(1)  # (batch_size, embed_dim)
        
        # 与避障分支的Feed Forward输出融合（加法）
        # 根据架构图，避障分支的Feed Forward输出与协作分支的第二个Add & Norm输出相加
        # 但这里我们需要确保维度匹配
        # obstacle_ff_output shape: (batch_size, embed_dim)
        # x shape: (batch_size, embed_dim)
        fused = x + obstacle_ff_output  # (batch_size, embed_dim)
        
        # 输出最终动作
        action = self.action_head(fused)  # (batch_size, action_dim)
        
        return action
