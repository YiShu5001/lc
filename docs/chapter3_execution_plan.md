# 第三章执行概要与交接计划

## 1. 文件定位

本文档用于把第三章当前状态、目标口径、缺口、执行顺序和验收标准整理清楚，方便直接交给另一个 AI 或协作者继续推进。

目标不是重写架构，而是在当前 `src/lc/control/` 主链上继续补实：

- 可训练
- 可对比
- 可绘图
- 可复现实验

## 2. 第三章当前真实状态

### 已经具备的部分

当前 `src/lc/control/` 已经有一套可运行主链：

- 控制器
  - `controllers/pid.py`
  - `controllers/ladrc.py`
  - `controllers/adaptive_ladrc.py`
- 环境
  - `envs/pybullet_axis_env.py`
  - `envs/pybullet_eval_env.py`
- 控制策略
  - `policies/mddpg_control.py`
  - `policies/stacking.py`
- 训练器
  - `trainers/pybullet_axis_trainer.py`
- 对比实验入口
  - `experiments/compare.py`
- 绘图模块
  - `plotting/plots.py`

### 当前还不够的地方

虽然第三章已经具备完整骨架和基本训练闭环，但仍然有几项属于“可继续补实”而不是“彻底完成”：

1. 当前被控对象仍以单轴位置跟踪为主，尚未对齐更真实的无人机外环对象。
2. `DDPG-LADRC` 与 `mDDPG-LADRC` 已能训练和导出结果，但训练规模、随机种子重复和收敛验证还不充分。
3. 指标和图表已具备，但还可以继续整理成更论文化的统一插图风格。
4. 当前奖励只按位置误差定义，虽然符合当前口径，但没有扩展到更复杂控制目标。
5. 目前更多是研究型可运行链路，距离更严格的工程复现实验还有加强空间。

结论：

- 第三章现在已经不是空骨架。
- 第三章已经可以作为可运行章节主链。
- 下一位 AI 不应该推翻现有实现，而应继续在当前主链上做实证增强与细节补强。

## 3. 第三章目标口径

### 主方法固定口径

第三章当前主方法固定为：

- `DDPG-LADRC`

增强版对比方法固定为：

- `mDDPG-LADRC`

### 被控对象固定口径

当前默认任务固定为：

- 单轴位置跟踪
- 扰动抑制

### 强化学习动作固定口径

强化学习在线调节 `LADRC` 位置环三个参数：

- `omega_c`
- `omega_o`
- `b0`

强化学习不直接输出控制量。

### 奖励函数固定口径

当前奖励固定为：

```text
reward = -|position_error|
```

不要在后续推进中随意改成多项复杂奖励，除非用户明确要求。

## 4. 建议执行顺序

### 阶段 A：补强训练闭环稳定性

目标：

- 让 `DDPG-LADRC` 和 `mDDPG-LADRC` 的训练过程更稳定、更可复现。

要做的事：

- 检查 replay buffer 刷入、采样和更新逻辑是否与当前设定一致
- 检查 target network 更新与 loss 日志是否齐全
- 增加更多训练历史字段输出
- 增加多随机种子支持或等价重复实验入口

验收标准：

- 第三章可以稳定训练多个 episode
- 日志可用于绘制训练收敛曲线

### 阶段 B：补强 mDDPG 三项增强机制

目标：

- 让 `mDDPG` 的三项增强逻辑更清晰、更可验证。

要做的事：

- 明确过去 `n` 个状态堆叠的缓存与更新方式
- 明确动作保持 `n` 步的执行逻辑
- 明确 `n-step` 回报的构造方式
- 让三项机制都能独立开关，便于消融实验

验收标准：

- 可以单独运行：
  - `no_state_stack`
  - `no_action_hold`
  - `no_n_step`
- 消融结果可单独导出

### 阶段 C：补强实验矩阵

目标：

- 把第三章从单次演示实验推进到完整对比矩阵。

要做的事：

- 保持四组方法：
  - `PID`
  - `LADRC`
  - `DDPG-LADRC`
  - `mDDPG-LADRC`
- 在不同复杂度场景下运行
- 统一输出指标表和 summary 文件

验收标准：

- 第三章至少能形成一套完整对比矩阵，而不是单场景单结果

### 阶段 D：补强图表输出

目标：

- 让第三章图表更适合直接转成论文实验图。

要做的事：

- 统一时域响应图的风格
- 统一指标对比图的标题和标注
- 统一训练曲线导出
- 增加更规范的论文插图版总览图

验收标准：

- 第三章至少能输出：
  - 指标对比图
  - 训练曲线
  - 机制消融图
  - 典型时域响应图

### 阶段 E：补强实验配置与复现文档

目标：

- 让第三章更容易复现实验。

要做的事：

- 增加更明确的配置快照输出
- 补 README 和实验说明
- 记录主要超参数与默认值

验收标准：

- 新协作者可直接根据文档运行第三章实验

## 5. 另一位 AI 不应改变的内容

以下内容建议固定，不要重构：

- 第三章主链仍然放在 `src/lc/control/`
- 默认主方法仍然是 `DDPG-LADRC`
- `mDDPG-LADRC` 仍然作为增强版对比
- 默认任务仍是单轴位置跟踪 + 扰动抑制
- 奖励仍然只按位置误差定义
- 强化学习仍然调 `omega_c / omega_o / b0`

## 6. 交给另一位 AI 时可直接附带的任务说明

可以直接把下面这段发给另一个 AI：

> 请在现有 `src/lc/control/` 架构上继续实现第三章控制层，不要重写架构。当前已经有 `PID / LADRC / DDPG-LADRC / mDDPG-LADRC` 四组方法、单轴跟踪环境、训练器、对比实验入口和绘图模块。请优先继续补强：  
> 1. `DDPG-LADRC` 与 `mDDPG-LADRC` 的训练稳定性与训练日志  
> 2. `mDDPG` 的状态堆叠、动作保持和 `n-step` 回报细节  
> 3. 第三章对比实验矩阵与不同复杂度场景  
> 4. 第三章论文图表输出与统一风格  
> 不要改变第三章的主口径：默认主方法是 `DDPG-LADRC`，强化学习调的是 `omega_c / omega_o / b0`，默认任务是单轴位置跟踪 + 扰动抑制，奖励只按位置误差定义。

## 7. 建议重点关注的文件

建议下一位 AI 优先查看和增强：

- `docs/chapter3_readme.md`
- `docs/chapter3_method_figures.md`
- `src/lc/control/trainers/pybullet_axis_trainer.py`
- `src/lc/control/experiments/compare.py`
- `src/lc/control/plotting/plots.py`
- `src/lc/control/envs/pybullet_axis_env.py`
- `src/lc/control/envs/pybullet_eval_env.py`
- `src/lc/control/policies/mddpg_control.py`

## 8. 最终验收口径

第三章完成的最低标准不是“有代码文件”，而是下面这些全部成立：

- 能训练
- 能跑四方法对比
- 能跑机制消融
- 能输出训练日志
- 能输出论文级图表
- 能通过文档复现实验入口
