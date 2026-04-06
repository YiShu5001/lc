# 第四章执行概要与交接计划

## 1. 文件定位

本文件用于把第四章当前状态、目标口径、缺失模块、执行顺序和验收标准整理清楚，方便直接交给另一个 AI 或工程协作者继续实现。

目标不是写论文，而是把第四章代码主链做成：

- 可训练
- 可对比
- 可绘图
- 可扩展到系统桥接

## 2. 第四章当前真实状态

### 已经具备的部分

当前 `src/lc/planning/` 已经有一套“可运行骨架”：

- 环境：
  - `envs/swarm.py`
- 输入编码器：
  - `encoders/basic.py`
- 主模型：
  - `models/multi_uav_model.py`
- 基线模型：
  - `models/mlp_baseline.py`
- Critic 骨架：
  - `critics/structured_critic.py`
- 课程学习骨架：
  - `curriculum/scheduler.py`
- 金字塔经验池骨架：
  - `memory/pyramid.py`
- 训练器骨架：
  - `trainers/planning_trainer.py`
- 对比实验入口：
  - `experiments/compare.py`
- 绘图骨架：
  - `plotting/plots.py`

### 当前还不够的地方

虽然第四章已经能“跑出一个骨架结果”，但本质上仍然不是完整训练系统，主要问题是：

1. `PlanningTrainer` 现在只有前向评估，没有真正的 actor-critic 训练更新。
2. `StructuredCritic` 已创建但没有进入训练闭环。
3. `CurriculumScheduler` 现在只是一个阈值计数器，没有和环境采样、阶段切换、经验池联动。
4. `PyramidReplayMemory` 只是三层桶，没有优先级重评分、阶段采样比和旧样本保留逻辑。
5. 第四章奖励函数还过于简化，没有完全拆成“目标占位 / 避障 / 协同 / 平滑 / 恢复”等子项。
6. 对比实验还不够完整，目前只有：
   - `task_decomposed`
   - `single_stream_mlp`
7. 图表还不够论文化，目前只有几张简单柱状图，没有：
   - 收敛曲线
   - 课程切换曲线
   - 复杂度泛化图
   - 消融图
   - 轨迹图
   - 注意力热力图

结论：

- 第四章当前是“结构对了，但训练和实验还没做实”。
- 下一个 AI 不应该重写架构，而应该在现有 `src/lc/planning` 上把训练闭环和实验系统补实。

## 3. 第四章目标口径

### 主方法固定口径

第四章主方法固定为：

- `MultiUAVModel`

方法语义固定为：

- 第一阶段：避障流
- 第二阶段：协同流
- 输出：
  - `avoid_action`
  - `final_action`

### 输入接口固定口径

规划层输入固定为三类结构化观测：

- `self_state`
- `obstacles`
- `neighbors`

不要改成只有扁平向量的主实现口径。扁平输入只可作为实验兼容输入，不应取代主接口。

### 第四章实验目标固定口径

第四章不是单纯“导航”，而是：

- 面向移动目标
- 多无人机协同
- 带障碍物约束
- 多复杂度环境
- 可课程学习
- 可持续学习/防遗忘

## 4. 建议执行顺序

### 阶段 A：补齐真正训练闭环

目标：

- 让 `MultiUAVModel + StructuredCritic` 进入真实 actor-critic 训练闭环。

要做的事：

- 在 `PlanningTrainer` 中加入：
  - 经验采集
  - replay sampling
  - critic loss
  - actor loss
  - target update 或等价稳定机制
- 让 `StructuredCritic` 实际消费：
  - 结构化观测扁平化结果
  - 动作
- 增加训练历史记录：
  - `episode`
  - `reward`
  - `success_rate`
  - `collision_rate`
  - `formation_error`
  - `actor_loss`
  - `critic_loss`

验收标准：

- 第四章不再只是前向评估，而是能真实训练若干 episode 并导出训练日志。

### 阶段 B：做实课程学习

目标：

- 让 `CurriculumScheduler` 真正参与环境难度切换。

要做的事：

- 为课程阶段定义固定场景族：
  - `easy`
  - `medium`
  - `hard`
  - `extreme`
- 将课程切换依据固定为：
  - 成功率
  - 碰撞率
  - 形成恢复率或等价稳定指标
- 实现：
  - 升级
  - 降级/退避
  - 阶段保持
- 将课程阶段写入实验输出：
  - 当前阶段
  - 切换历史
  - 每阶段平均指标

验收标准：

- 第四章训练输出中可以明确看到课程阶段变化，而不是只有固定场景。

### 阶段 C：做实 Pyramid-PER / 多层经验池

目标：

- 让 `PyramidReplayMemory` 不再只是装饰，而是成为第四章防遗忘和跨阶段训练的重要组件。

要做的事：

- 为经验池定义至少三层语义：
  - `high`：关键失败/高风险/高价值样本
  - `medium`：普通阶段样本
  - `low`：旧阶段保留样本或基础覆盖样本
- 引入样本打分逻辑，至少考虑：
  - 风险/碰撞
  - 协同误差
  - 稀有事件
  - TD 误差或近似优先级
- 让课程阶段切换时保留旧阶段代表样本
- 增加分层采样比例配置

验收标准：

- 第四章可以输出“不同层经验池样本数量和采样占比”的统计结果。

### 阶段 D：补齐奖励系统

目标：

- 把第四章奖励从单一混合奖励，拆成论文可解释的多子项。

建议固定为这些子项：

- `target_reward`
- `avoidance_reward`
- `collaboration_reward`
- `recovery_reward`
- `smoothness_penalty`
- `consistency_penalty`

要做的事：

- 在 `planning/rewards/planning_reward.py` 中将奖励拆分为子函数或子项结构
- 在训练日志中记录奖励分项
- 在实验汇总中输出平均子奖励

验收标准：

- 第四章结果分析时可以明确说明“性能提升来自哪些子目标”。

### 阶段 E：补齐对比实验与消融实验

目标：

- 让第四章具备论文级实验对比能力。

至少保留这些实验组：

- 主方法：`task_decomposed`
- 基线：`single_stream_mlp`
- 有/无课程学习
- 有/无 Pyramid-PER
- 不同 memory scoring 方式
- 不同阶段采样比例

至少保留这些复杂度维度：

- 无人机数量变化
- 障碍物数量变化
- 障碍物动态性变化
- 目标动态性变化
- 课程阶段变化

验收标准：

- 第四章实验输出不再只是单场景、双方法对比，而是形成完整对比矩阵。

### 阶段 F：补齐论文插图级输出

目标：

- 让第四章输出可以直接转成论文图表。

需要补的图：

- 收敛曲线图
- 成功率 / 碰撞率曲线
- 编队误差 / 占位误差图
- 课程阶段切换图
- 消融图
- 复杂度泛化图
- 轨迹可视化图
- 注意力热力图

建议输出目录：

- `outputs/planning/<difficulty>/stage_<index>/`
- `outputs/planning/compare/`
- `outputs/planning/ablations/`

验收标准：

- 另一个 AI 完成后，第四章至少能输出 6 类以上可读图表。

## 5. 另一个 AI 不应改变的内容

以下内容建议固定，不要重构：

- 主模型仍以 `MultiUAVModel` 为核心
- 结构化三输入：
  - `self_state`
  - `obstacles`
  - `neighbors`
- 双阶段语义：
  - 先避障
  - 后协同
- 当前 `src/lc/planning/` 目录结构保留
- 第四章仍走 `src` 主链，不回退到旧 `NN/` 和 `Trainer/`

## 6. 交接给另一个 AI 时可直接附带的任务说明

可以直接把下面这段发给另一个 AI：

> 请在现有 `src/lc/planning/` 架构上继续实现第四章，不要重写架构。当前已有环境、模型、Critic、课程学习、经验池、绘图和实验入口骨架，但还缺真正训练闭环。请优先完成：  
> 1. `MultiUAVModel + StructuredCritic` 的 actor-critic 训练闭环  
> 2. `CurriculumScheduler` 与场景切换联动  
> 3. `PyramidReplayMemory` 的优先级、分层采样与旧样本保留  
> 4. 奖励函数拆分与奖励日志  
> 5. 第四章对比实验、消融实验和论文图表输出  
> 不要改掉 `self_state / obstacles / neighbors` 三输入结构，也不要把 `MultiUAVModel` 降级成普通 MLP 主实现。

## 7. 建议新增文件

如果另一个 AI 继续推进，建议优先新增或增强这些文件：

- `docs/chapter4_readme.md`
- `docs/chapter4_experiment_matrix.md`
- `src/lc/planning/trainers/planning_trainer.py`
- `src/lc/planning/experiments/compare.py`
- `src/lc/planning/plotting/plots.py`

## 8. 最终验收口径

第四章完成的最低标准不是“有模型文件”，而是下面这些全部成立：

- 能训练
- 能切课程阶段
- 能用多层经验池
- 能跑对比实验
- 能跑消融实验
- 能输出论文图表
- 能与系统桥接层继续集成
