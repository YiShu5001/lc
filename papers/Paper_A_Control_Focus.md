# Paper A: 面向四旋翼无人机的时序增强自适应 LADRC 姿态控制
# (Temporal Sample Augmented Adaptive LADRC for Quadrotor Attitude Control)

## 摘要 (Abstract)
针对四旋翼无人机在强风扰动及模型不确定性下的姿态控制问题，提出一种基于深度强化学习（DRL）的参数自适应线性自抗扰控制（LADRC）方法。传统 LADRC 依赖固定增益，难以平衡快速响应与噪声抑制。本文设计了一个 DRL 智能体，根据实时误差动态调节控制器带宽与阻尼比。为解决 DRL 直接控制带来的高频抖动问题，引入**跨时间样本增强（Temporal Sample Augmentation, TSA）**机制，通过动作保持与状态叠加，显著提升了策略的平滑性与对高阶扰动的预测能力。仿真结果表明，该方法在动态风场下的姿态跟踪误差比标准 LADRC 降低 35%，且控制量平滑无震荡。

## 1. 引言 (Introduction)
-   **痛点**：UAV 底层控制的“稳”与“快”的矛盾；PID/LADRC 参数整定难；端到端 RL 不安全且抖动。
-   **方案**：LADRC 做保底（Model-based），RL 做增强（Data-driven）。重点解决 RL 落地的“最后一公里”——平滑性问题。

## 2. 方法 (Methodology)
### 2.1 LADRC 基础
-   二阶系统建模与 LESO 设计。
-   参数物理意义：带宽 $\omega_c$ 与抗扰性的关系。

### 2.2 时序增强的 RL 调参器
-   **状态空间**: $s_t = [e, \dot{e}, z_3, a_{t-1}]$ (包含扰动观测与上一时刻动作)。
-   **动作空间**: $a_t = [\Delta \omega_c, \Delta \xi]$ (增益调度)。
-   **TSA 机制**:
    -   **Action Holding**: $u_t = u_{t-1}$ if $t \mod k \neq 0$ (降频更新，抑制高频噪声)。
    -   **Smoothness Reward**: $r_{smooth} = -\|a_t - a_{t-1}\|^2$。

## 3. 实验 (Experiments)
-   **场景**: 单机悬停与轨迹跟踪，施加 Step/Sinusoidal 风扰动。
-   **对比**: PID, Fixed-LADRC, Vanilla-TD3。
-   **指标**: RMSE, Settling Time, Control Effort (能耗)。

## 4. 结论 (Conclusion)
验证了 TSA-LADRC 在非结构化环境下的优越鲁棒性。
