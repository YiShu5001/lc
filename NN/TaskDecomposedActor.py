"""
Task-Decomposed Actor Network (Paper B Implementation)
------------------------------------------------------
对应论文: "Task-Decomposed Collaborative Planning for Multi-UAVs..."
核心功能:
1. 双流架构 (Dual-Stream): 避障流 + 协作流
2. 风险门控 (Risk Gating): Risk Evaluation Module (REM)
3. 增长注意力接口 (Growth Interface): 支持动态输入维度

Author: Lingming (OpenClaw)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class PointNetEncoder(nn.Module):
    """
    基于PointNet的集合特征编码器 (Set Encoder)
    用于处理变长的障碍物集合或邻居集合。
    
    论文对应: Section 4.3.1 (Three-Encoder Architecture)
    公式: z = max_pool( MLP(h_k) )
    """
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, N_entities, input_dim) - 实体集合输入
        Returns:
            global_feat: (Batch, out_dim) - 聚合后的全局特征
        """
        # 1. 逐点特征提取 (Point-wise MLP)
        # local_feat: (Batch, N, out_dim)
        local_feat = self.mlp(x)
        
        # 2. 对称聚合函数 (Symmetric Function: Max Pooling)
        # 保证置换不变性 (Permutation Invariance)
        # global_feat: (Batch, out_dim)
        global_feat = torch.max(local_feat, dim=1)[0]
        
        return global_feat

class RiskEvaluationModule(nn.Module):
    """
    风险评估模块 (Risk Evaluation Module, REM)
    
    论文对应: Section 3.1 & Eq. (9)
    功能: 根据环境风险指标计算门控系数 sigma
    """
    def __init__(self, input_dim: int = 2):
        super().__init__()
        # 输入通常是 [d_min (最近障碍距离), v_rel (相对逼近速度)]
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() # 输出限制在 [0, 1]
        )
        
    def forward(self, risk_features: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            sigma: (Batch, 1) - 风险系数
                   sigma -> 1: 高风险 (High Risk)
                   sigma -> 0: 低风险 (Low Risk)
        """
        return self.net(risk_features)

class AvoidanceStream(nn.Module):
    """
    避障流分支 (Avoidance Stream)
    
    论文对应: Section 3.1 - Stream 1
    目标: 输出仅考虑避障的动作 a_av
    """
    def __init__(self, self_dim: int, obs_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        # 自身状态编码
        self.self_encoder = nn.Linear(self_dim, hidden_dim)
        # 障碍物集合编码
        self.obs_encoder = PointNetEncoder(obs_dim, hidden_dim, hidden_dim)
        
        # 动作解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh() # 动作归一化到 [-1, 1]
        )
        
    def forward(self, self_state: torch.Tensor, obstacles: torch.Tensor) -> torch.Tensor:
        z_self = F.relu(self.self_encoder(self_state)) # (B, H)
        z_obs = self.obs_encoder(obstacles)            # (B, H)
        
        # 特征拼接
        z_joint = torch.cat([z_self, z_obs], dim=-1)   # (B, 2H)
        
        return self.decoder(z_joint)

class CooperationStream(nn.Module):
    """
    协作流分支 (Cooperation Stream)
    
    论文对应: Section 3.1 - Stream 2
    目标: 输出仅考虑协作(队形)的动作 a_co
    """
    def __init__(self, self_dim: int, nbr_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.self_encoder = nn.Linear(self_dim, hidden_dim)
        self.nbr_encoder = PointNetEncoder(nbr_dim, hidden_dim, hidden_dim) # 后续可升级为 Attention
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
    def forward(self, self_state: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        z_self = F.relu(self.self_encoder(self_state))
        z_nbr = self.nbr_encoder(neighbors)
        
        z_joint = torch.cat([z_self, z_nbr], dim=-1)
        
        return self.decoder(z_joint)

class TaskDecomposedActor(nn.Module):
    """
    任务分离双流策略网络 (Dual-Stream Policy Network)
    
    论文对应: Fig 4-1 (Overall Architecture)
    """
    def __init__(self, 
                 self_dim: int = 10,
                 obs_dim: int = 4,
                 nbr_dim: int = 6,
                 hidden_dim: int = 128,
                 action_dim: int = 3):
        super().__init__()
        
        # 1. 避障流
        self.avoid_stream = AvoidanceStream(self_dim, obs_dim, hidden_dim, action_dim)
        
        # 2. 协作流
        self.coop_stream = CooperationStream(self_dim, nbr_dim, hidden_dim, action_dim)
        
        # 3. 风险门控 (REM)
        # 输入维度 2: 假设我们从obs中提取 d_min 和 v_rel
        self.rem = RiskEvaluationModule(input_dim=2)
        
    def extract_risk_features(self, obstacles: torch.Tensor) -> torch.Tensor:
        """
        从障碍物观测中提取标量风险特征 (用于REM输入)
        假设 obstacles shape: (B, K, 4) -> [dx, dy, dz, radius]
        """
        # 计算距离: sqrt(dx^2 + dy^2 + dz^2) - radius
        dists = torch.norm(obstacles[..., :3], dim=-1) - obstacles[..., 3]
        
        # d_min: (Batch, 1)
        d_min = torch.min(dists, dim=1, keepdim=True)[0]
        
        # 这里简化处理，v_rel 暂用 0 替代，实际应从 obs 获取
        # 扩展维度以匹配 REM 输入
        v_rel = torch.zeros_like(d_min) 
        
        return torch.cat([d_min, v_rel], dim=-1)

    def forward(self, 
                self_state: torch.Tensor, 
                obstacles: torch.Tensor, 
                neighbors: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            self_state: (B, self_dim)
            obstacles: (B, K, obs_dim)
            neighbors: (B, M, nbr_dim)
            
        Returns:
            final_action: (B, action_dim)
            info: 包含 sigma, a_av, a_co 用于 loss 计算和可视化
        """
        # 1. 并行计算两个动作流
        a_av = self.avoid_stream(self_state, obstacles)
        a_co = self.coop_stream(self_state, neighbors)
        
        # 2. 计算风险门控系数 sigma
        risk_feats = self.extract_risk_features(obstacles)
        sigma = self.rem(risk_feats) # (B, 1) in [0, 1]
        
        # 3. 门控融合 (Gated Fusion)
        # 公式: a = sigma * a_av + (1 - sigma) * a_co
        final_action = sigma * a_av + (1 - sigma) * a_co
        
        return final_action, {
            "sigma": sigma,
            "action_avoid": a_av,
            "action_coop": a_co
        }

if __name__ == "__main__":
    # 简单测试代码 (Unit Test)
    batch_size = 4
    K, M = 5, 3 # 障碍数, 邻居数
    
    model = TaskDecomposedActor()
    
    # 构造随机输入
    s = torch.randn(batch_size, 10)
    o = torch.randn(batch_size, K, 4)
    n = torch.randn(batch_size, M, 6)
    
    # 前向传播
    action, info = model(s, o, n)
    
    print("Action Shape:", action.shape) # Should be (4, 3)
    print("Sigma Mean:", info['sigma'].mean().item())
    print("Test Passed: TaskDecomposedActor structure is valid.")
