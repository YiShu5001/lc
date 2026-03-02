# Paper B: 基于分层课程学习的无人机集群长时序协同围捕
# (Long-Horizon Cooperative Pursuit of UAV Swarms via Hierarchical Curriculum Learning)

## 摘要 (Abstract)
大规模无人机集群在执行长时序协同围捕任务时，面临状态空间爆炸与灾难性遗忘的双重挑战。本文提出一种**分层引导课程强化学习（HGC-RL）**框架。首先，将复杂任务分解为“单机技能”与“多机协同”两个课程阶段，通过技能蒸馏加速策略成型。其次，设计**金字塔分级经验池（Pyramid Hierarchical Memory, PHM）**，基于 TD-Error 和任务完成度对样本进行分层存储与混合采样，有效解决了稀疏奖励下的样本效率问题。实验表明，该方法在 5 机围捕任务中的成功率达到 92%，训练效率较 MADDPG 提升 2.5 倍。

## 1. 引言 (Introduction)
-   **痛点**：多智能体协作（MARL）难收敛；长任务（先搜后捕）导致遗忘；稀疏奖励导致冷启动难。
-   **方案**：Curriculum（由简入繁）+ Memory（温故知新）。

## 2. 方法 (Methodology)
### 2.1 任务建模
-   Dec-POMDP 模型，定义全局奖励 $R_{group}$ 和个体奖励 $R_{indiv}$。

### 2.2 课程学习框架
-   **Phase 1**: Single Agent Navigation (避障、飞行动力学)。
-   **Phase 2**: Multi-Agent Formation (编队保持)。
-   **Phase 3**: Dynamic Pursuit (动态博弈)。
-   **晋级判据**: 成功率 $SR > \tau$。

### 2.3 金字塔经验池 (PHM)
-   **L0 (Raw)**: 海量短期记忆。
-   **L1 (Surprise)**: 高 TD-Error 样本（困难场景）。
-   **L2 (Success)**: 成功轨迹（高价值回放）。
-   **动态采样**: $\lambda_t$ 退火策略，从探索转向利用。

## 3. 实验 (Experiments)
-   **场景**: 多机围捕动态逃逸目标 (Pursuit-Evasion)。
-   **对比**: MADDPG, QMIX, MAPPO (w/o Curriculum)。
-   **指标**: Capture Rate, Collision Rate, Training Steps。

## 4. 结论 (Conclusion)
验证了 HGC-RL 框架解决 MARL 长时序、稀疏奖励问题的有效性。
