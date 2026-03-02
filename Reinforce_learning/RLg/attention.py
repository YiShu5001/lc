import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块

    参数:
        embed_dim: 嵌入维度
        num_heads: 注意力头的数量
        dropout: dropout概率 (default: 0.1)
    """
##### t *  12*(n1 + n2)  ;  一个头12 个维度
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "嵌入维度必须是注意力头数量的整数倍"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 每个注意力头的维度

        # 线性变换层：用于生成Q、K、V
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        # 输出线性层
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 缩放因子，用于缩放点积注意力
        self.scale = 1 / math.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None):
        """
        前向传播

        参数:
            query: 查询张量 [batch_size, seq_len, embed_dim]
            key: 键张量 [batch_size, seq_len, embed_dim]
            value: 值张量 [batch_size, seq_len, embed_dim]
            mask: 掩码张量 [batch_size, seq_len, seq_len] (default: None)

        返回:
            注意力输出 [batch_size, seq_len, embed_dim]
            注意力权重 [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)

        # ==========================
        # 1. 线性变换并分头处理
        # ==========================
        # 线性变换 [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, embed_dim]
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # 重塑张量形状以分头处理 [batch_size, seq_len, num_heads, head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # ==========================
        # 2. 计算缩放点积注意力
        # ==========================
        # 计算Q和K的点积 [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 应用掩码（如果提供）
        if mask is not None:
            # 确保掩码形状正确
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 计算注意力权重 [batch_size, num_heads, seq_len, seq_len]
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # ==========================
        # 3. 应用注意力权重到值
        # ==========================
        # [batch_size, num_heads, seq_len, head_dim]
        output = torch.matmul(attn_weights, V)

        # ==========================
        # 4. 合并注意力头并输出
        # ==========================
        # 重塑张量形状 [batch_size, seq_len, embed_dim]
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim
        )

        # 通过输出线性层
        output = self.out_linear(output)

        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络模块

    参数:
        embed_dim: 嵌入维度
        ff_dim: 前馈网络内部维度 (通常比embed_dim大)
        dropout: dropout概率 (default: 0.1)
    """

    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 使用GELU激活函数

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层

    参数:
        embed_dim: 嵌入维度
        num_heads: 注意力头的数量
        ff_dim: 前馈网络内部维度
        dropout: dropout概率 (default: 0.1)
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        # 自注意力层
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        # 前馈网络层
        self.feed_forward = PositionwiseFeedForward(embed_dim, ff_dim, dropout)

        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向传播

        参数:
            x: 输入张量 [batch_size, seq_len, embed_dim]
            mask: 注意力掩码 (default: None)

        返回:
            输出张量 [batch_size, seq_len, embed_dim]
        """
        # ==========================
        # 1. 自注意力子层
        # ==========================
        # 残差连接
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # ==========================
        # 2. 前馈网络子层
        # ==========================
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x, attn_weights


class TransformerEncoder(nn.Module):
    """
    多层Transformer编码器堆叠

    参数:
        num_layers: 编码器层数量
        embed_dim: 嵌入维度
        num_heads: 注意力头的数量
        ff_dim: 前馈网络内部维度
        dropout: dropout概率 (default: 0.1)
    """

    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        """
        前向传播

        参数:
            x: 输入张量 [batch_size, seq_len, embed_dim]
            mask: 注意力掩码 (default: None)

        返回:
            输出张量 [batch_size, seq_len, embed_dim]
            所有层的注意力权重列表
        """
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            all_attn_weights.append(attn_weights)

        return x, all_attn_weights


# =============================================
# 示例：使用多层Transformer编码器
# =============================================
if __name__ == "__main__":
    # 设置参数
    batch_size = 4
    seq_len = 16
    embed_dim = 512
    num_heads = 8
    ff_dim = 2048
    num_layers = 6

    # 创建多层Transformer编码器
    transformer = TransformerEncoder(
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim
    )

    # 创建随机输入数据
    x = torch.randn(batch_size, seq_len, embed_dim)

    # 创建随机掩码 (可选)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).repeat(batch_size, 1, 1)

    # 前向传播
    output, all_attn_weights = transformer(x, mask)

    # 打印结果
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
    print("注意力权重数量:", len(all_attn_weights))
    print("第一层注意力权重形状:", all_attn_weights[0].shape)

    # 验证输出形状与输入相同
    assert output.shape == (batch_size, seq_len, embed_dim)
    print("\n验证通过：输出形状与输入一致")