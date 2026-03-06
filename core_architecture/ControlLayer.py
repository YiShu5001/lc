"""
[Paper A Core] 控制层核心架构 - 伪代码布局
对应论文章节: 3.2 自适应控制, 3.3 跨时间增强

Designed by: Lingming (OpenClaw)
"""

class AdaptiveLADRC:
    """RL自适应LADRC控制器"""
    def __init__(self):
        self.ladrc = StandardLADRC(omega_c=10, b0=1.0) # 底层控制器
        self.rl_agent = DDPGAgent() # 上层参数调度器

    def step(self, ref_cmd, current_state):
        """双闭环控制步进"""
        # 1. RL 决策 (低频 10Hz)
        if is_rl_step():
            # 状态包含: 跟踪误差 + ESO扰动估计
            s_rl = [error, d_error, self.ladrc.eso.z3]
            
            # 动作: 参数调节系数 (e.g., bandwidth_scale)
            # action = [1.2, 0.9] -> 带宽x1.2, 增益x0.9
            action = self.rl_agent.select_action(s_rl)
            
            # 更新LADRC参数
            self.ladrc.set_params(action)

        # 2. LADRC 控制 (高频 100Hz)
        # 使用最新的参数计算电机推力
        u_motor = self.ladrc.compute_control(ref_cmd, current_state)
        
        return u_motor

class CTIE_Manager:
    """跨时间信息增强管理器 (Cross-Time Information Enhancement)"""
    def __init__(self, buffer):
        self.history_stack = deque(maxlen=5) # 历史状态堆叠
        self.n_step_cache = [] # N步回报缓存
        self.action_hold_counter = 0

    def process_interaction(self, s, a, r, s_next, done):
        """处理每一步交互数据"""
        
        # 1. 历史堆叠 (Historical Stacking)
        s_stacked = self.history_stack + s
        
        # 2. N步回报计算 (N-step Bootstrapping)
        self.n_step_cache.append((s_stacked, a, r))
        if len(self.n_step_cache) == N:
            R_n = calculate_discounted_return(self.n_step_cache)
            # 存入 Experience Buffer
            buffer.add(s_stacked, a, R_n, s_next_N)

    def should_update_action(self):
        """动作保持逻辑 (Action Hold)"""
        if self.action_hold_counter > 0:
            self.action_hold_counter -= 1
            return False # 保持上一步动作
        else:
            self.action_hold_counter = H # 重置计数器
            return True # 允许RL输出新动作
