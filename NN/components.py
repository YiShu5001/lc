"""
核心神经网络组件模块
包含Multi-Head Attention和Transformer Encoder Block
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    实现Scaled Dot-Product Attention的多头版本：
    - Query/Key/Value线性投影层
    - 多个注意力头并行计算
    - Concatenation和输出投影
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Args:
            embed_dim: 嵌入维度（必须是num_heads的倍数）
            num_heads: 注意力头数
            dropout: Dropout概率
            bias: 是否使用偏置
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query: Query张量，shape = (batch_size, seq_len, embed_dim)
            key: Key张量，shape = (batch_size, seq_len, embed_dim)
            value: Value张量，shape = (batch_size, seq_len, embed_dim)
            mask: 可选的注意力掩码，shape = (batch_size, seq_len, seq_len)
        
        Returns:
            输出张量，shape = (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = query.shape
        
        # 线性投影并重塑为多头形式
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # scores shape: (batch_size, num_heads, seq_len, seq_len)
        
        if mask is not None:
            # 应用掩码（mask中True的位置会被mask掉）
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到Value
        attn_output = torch.matmul(attn_weights, V)
        # attn_output shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate所有头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        # attn_output shape: (batch_size, seq_len, embed_dim)
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        return output


class TransformerEncoderBlock(nn.Module):
    """
    Transformer编码器块
    
    包含：
    - Multi-Head Attention
    - Add & Norm（Layer Normalization + Residual Connection）
    - Feed Forward网络
    - Add & Norm
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            ff_dim: 前馈网络隐藏层维度
            dropout: Dropout概率
            activation: 激活函数类型（"relu"或"gelu"）
        """
        super().__init__()
        
        # Multi-Head Attention
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Feed Forward网络
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
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
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，shape = (batch_size, seq_len, embed_dim)
            mask: 可选的注意力掩码
        
        Returns:
            输出张量，shape = (batch_size, seq_len, embed_dim)
        """
        # Multi-Head Attention + Add & Norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed Forward + Add & Norm
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        return x
