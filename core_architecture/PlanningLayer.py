"""
[Paper B Core] 规划层核心架构 - 伪代码布局
对应论文章节: 4.2 网络结构, 4.3 增长注意力, 4.4 门控融合

Designed by: Lingming (OpenClaw)
"""

class RiskEvaluationModule:
    """风险评估模块 (REM) - 论文公式 (9)"""
    def forward(self, obstacles):
        # 1. 提取关键风险指标
        d_min = min(obstacles.distance)
        v_rel = obstacles.relative_velocity
        
        # 2. 计算风险系数 sigma (Sigmoid激活)
        # sigma -> 1: 高风险 (强制避障)
        # sigma -> 0: 低风险 (允许协作)
        sigma = sigmoid(self.weights * [d_min, v_rel] + bias)
        return sigma

class DualStreamActor:
    """双流任务分离网络 - 论文图 4-1"""
    def __init__(self):
        self.avoid_net = PointNetEncoder()  # 避障流
        self.coop_net = AttentionEncoder()  # 协作流
        self.risk_module = RiskEvaluationModule() # 门控

    def forward(self, observation):
        # 1. 解包观测
        self_state = observation.self
        obs_data = observation.obstacles
        nbr_data = observation.neighbors

        # 2. 并行计算双流动作
        # 避障动作 a_av: 专注于远离障碍
        a_av = self.avoid_net(self_state, obs_data)
        
        # 协作动作 a_co: 专注于队形保持
        a_co = self.coop_net(self_state, nbr_data)

        # 3. 计算门控系数
        sigma = self.risk_module(obs_data)

        # 4. 门控融合 (Gated Fusion)
        final_action = sigma * a_av + (1 - sigma) * a_co
        
        return final_action, sigma  # 返回sigma用于可视化分析

class GrowthManager:
    """增长注意力管理器 - 论文章节 4.3.3"""
    def __init__(self, network):
        self.network = network
        self.stage = 1 # 当前课程阶段

    def check_and_grow(self, success_rate):
        """检查是否满足扩容条件"""
        if self.stage == 1 and success_rate > 0.85:
            self.grow_to_stage_2()
        elif self.stage == 2 and success_rate > 0.90:
            self.grow_to_stage_3()

    def grow_to_stage_2(self):
        """权重继承与扩容逻辑 - 论文公式 (12)"""
        print("Growing from Local (K=3) to Expanded (K=5)...")
        
        # 1. 获取旧权重 W_old
        W_old = self.network.attention.weights
        
        # 2. 创建新权重 W_new (更大的维度)
        # 3. 执行权重继承 (Weight Inheritance)
        # W_new = [ W_old   0 ]
        #         [   0   epsilon ]
        W_new = zero_pad_and_copy(W_old)
        
        # 4. 替换网络参数
        self.network.attention.weights = W_new
        self.stage = 2
