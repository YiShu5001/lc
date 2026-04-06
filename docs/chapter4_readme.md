# 第四章代码 README

## 1. 文档目的

这份 README 只面向第四章实验代码。

目标是说明：

- 第四章代码现在放在哪里
- 主要算法链路如何组织
- 训练时各模块怎么连接
- 当前已经实现到什么程度
- 继续开发时应该优先看哪些文件

本 README 不讨论论文写作，只讨论代码实现。

## 2. 第四章代码范围

第四章主代码位于：

- `src/lc/planning/envs/`
- `src/lc/planning/models/`
- `src/lc/planning/critics/`
- `src/lc/planning/rewards/`
- `src/lc/planning/memory/`
- `src/lc/planning/curriculum/`
- `src/lc/planning/trainers/`
- `src/lc/planning/experiments/`
- `src/lc/planning/plotting/`

核心入口文件：

- `src/lc/planning/envs/swarm.py`
- `src/lc/planning/models/multi_uav_model.py`
- `src/lc/planning/critics/structured_critic.py`
- `src/lc/planning/rewards/planning_reward.py`
- `src/lc/planning/memory/pyramid.py`
- `src/lc/planning/curriculum/scheduler.py`
- `src/lc/planning/trainers/planning_trainer.py`
- `src/lc/planning/experiments/compare.py`

## 3. 当前方法口径

### 3.1 观测输入

第四章当前固定使用三路结构化输入：

- `self_state = (target_dx, target_dy, self_vx, self_vy)`
- `obstacles = n * (x, y, r)`
- `neighbors = n * (x, y)`

当前默认是二维平面规划，不考虑 `z` 轴。

### 3.2 动作输出

Actor 输出二维高层速度意图：

- `vx`
- `vy`

动作范围固定为 `[-1, 1]`，最终由控制层接口继续映射。

### 3.3 课程阶段

当前固定为三阶段：

- `guidance`
- `avoidance`
- `cooperation`

含义分别是：

- `guidance`：学习到达目标相关槽位，不启用金字塔回放
- `avoidance`：在到达基础上学习避障，启用独立三层金字塔
- `cooperation`：在前两阶段基础上学习协作恢复，启用另一套独立三层金字塔

### 3.4 训练算法

当前训练主链已经按 `TD3` 风格实现，包含：

- 双 Critic
- target smoothing
- delayed actor update
- target network soft update

## 4. 模块结构

### 4.1 环境

文件：

- `src/lc/planning/envs/swarm.py`

当前职责：

- 生成三路观测 `self_state / obstacles / neighbors`
- 按阶段设置目标、障碍、邻机近似结构
- 计算 `risk / occupancy_error / formation_error / angle_error`
- 调用奖励模块输出总奖励和分项奖励
- 生成 rare event 分数

当前环境是实验导向抽象环境，不是高保真多机物理仿真。

### 4.2 Actor

文件：

- `src/lc/planning/models/multi_uav_model.py`

当前结构：

- 第一段：`AvoidanceBackbone`
  - `self_embedding + obstacle_embedding`
  - `MultiHeadAttention`
  - `safe_action`
- 第二段：`CollaborativeBackbone`
  - `neighbor_embedding`
  - 协作残差 `coop_residual`
  - 动态门控 `gate`
  - 最终动作 `final_action = safe_action + gate * residual`

当前还能导出：

- `avoid_action`
- `final_action`
- `safe_feature`
- 注意力权重和 gate 值

### 4.3 Critic

文件：

- `src/lc/planning/critics/structured_critic.py`

当前结构：

- 共享 Actor 输入 embedding
- 单头注意力
- 输入当前结构化观测和当前动作
- 使用最近 `top-k` 邻机 token

这是一个“简单版状态-动作注意力 Critic”，重点是稳定评估，不负责生成动作。

### 4.4 奖励函数

文件：

- `src/lc/planning/rewards/planning_reward.py`

当前奖励项：

- `target_reward`
- `avoidance_reward`
- `collaboration_reward`
- `recovery_reward`
- `smoothness_penalty`
- `consistency_penalty`
- `success_bonus`

当前是阶段感知动态调权：

- `guidance` 偏到达
- `avoidance` 偏安全
- `cooperation` 偏合作恢复

### 4.5 回放系统

文件：

- `src/lc/planning/memory/pyramid.py`

当前采用三段式后端：

- `guidance`
  - 普通 replay/PER
  - 维护 `guidance_old_pool`
- `avoidance`
  - 独立三层金字塔
  - `td_layer / filtered_layer / rare_layer`
- `cooperation`
  - 独立三层金字塔
  - `td_layer / contribution_layer / rare_layer`

阶段旧样本回采规则：

- `avoidance`：`9:1 = 当前避障 : guidance_old_pool`
- `cooperation`：`8:2 = 当前合作 : avoidance_old_pool`

### 4.6 课程调度

文件：

- `src/lc/planning/curriculum/scheduler.py`

当前逻辑：

- 滑动窗口统计
- 同时考虑：
  - 平均奖励
  - 成功率
  - 奖励波动
- 达标后晋级
- 均值和成功率持续下降时回退

### 4.7 训练器

文件：

- `src/lc/planning/trainers/planning_trainer.py`

当前职责：

- TD3 主训练循环
- 按课程阶段切换 replay backend
- 写入 transition 到对应回放池
- 从 guidance/avoidance/cooperation 不同回放后端采样
- 更新 twin critics 和 actor
- 汇总奖励、损失、课程历史、回放统计和注意力代理结果

### 4.8 实验与绘图

文件：

- `src/lc/planning/experiments/compare.py`
- `src/lc/planning/plotting/plots.py`

当前支持的对比组：

- `task_decomposed`
- `single_stream_mlp`
- `without_curriculum`
- `without_pyramid_per`
- `uniform_replay`

当前可输出：

- `summary.json`
- `metrics.csv`
- `training_history.csv`
- `ablation_comparison.svg`
- `convergence_curve.svg`
- `success_collision_curve.svg`
- `formation_occupancy_curve.svg`
- `curriculum_schedule.svg`
- `complexity_generalization.svg`
- `trajectory.svg`
- `attention_heatmap.svg`

默认输出目录：

- `outputs/planning/<difficulty>/stage_<index>/`

## 5. 当前训练数据流

第四章当前主链可以概括为：

1. 环境 `PlanningSwarmEnv` 产生三路结构化观测
2. Actor 先得到 `safe_action`
3. Actor 再通过协作残差和门控得到 `final_action`
4. 环境执行 `final_action`
5. 奖励器返回阶段化奖励分解
6. trainer 计算 `contribution / td proxy / rare_event_score`
7. transition 被写入当前阶段对应 replay backend
8. trainer 从当前 replay backend 采样，并按阶段决定是否混入旧样本
9. twin critics 更新
10. 按 TD3 延迟策略更新 actor
11. scheduler 根据最近窗口统计决定是否晋级或回退

## 6. 快速运行

### 6.1 跑第四章对比实验

```powershell
$env:PYTHONPATH='C:\context_mine\mine_code\GIT_Projects\lc\src'
@'
from lc.planning.experiments import run_planning_comparison
result = run_planning_comparison()
print(result["output_dir"])
'@ | python -
```

### 6.2 跑相关测试

```powershell
$env:PYTHONPATH='C:\context_mine\mine_code\GIT_Projects\lc\src'
python -m pytest tests\test_planning_reward_replay.py -q
python -m pytest tests\test_new_architecture.py -q
```

## 7. 当前实现状态

已经完成的部分：

- 结构化三路输入
- 两阶段注意力 Actor
- 单头注意力 Critic
- TD3 训练主链
- 三阶段课程调度
- guidance 基础 replay
- avoidance / cooperation 双金字塔回放
- 分阶段奖励函数
- 五组对比实验
- 结果导出和基础图表

仍然属于“实验代码版”的部分：

- 环境还是抽象近似环境
- 多机协同还是近似建模，不是真正多智能体联合仿真
- 若干 priority 公式仍是工程启发式
- 与控制层的接口映射还是轻量版

## 8. 推荐阅读顺序

如果后续继续补第四章，建议按这个顺序读代码：

1. `src/lc/planning/envs/swarm.py`
2. `src/lc/planning/models/multi_uav_model.py`
3. `src/lc/planning/critics/structured_critic.py`
4. `src/lc/planning/rewards/planning_reward.py`
5. `src/lc/planning/memory/pyramid.py`
6. `src/lc/planning/curriculum/scheduler.py`
7. `src/lc/planning/trainers/planning_trainer.py`
8. `src/lc/planning/experiments/compare.py`

## 9. 最重要的现实提醒

当前第四章代码已经不是“只有架子”，但也还不是论文最终形态。

最关键的现实点有三个：

- 训练闭环已经打通，但环境保真度还不够高
- 双金字塔已经实现，但层内 priority 仍有继续精化空间
- Actor/Critic 结构已基本定型，后续更值得优先补环境真实性、实验完整性和接口闭环
