# 第三章 README

## 1. 文档定位

本文档用于说明第三章控制层代码在当前仓库中的实现范围、模块结构、强化学习设定、实验输出和后续改进方向。

第三章当前聚焦于：

- 单轴位置跟踪
- 外部扰动抑制
- 位置环 `LADRC` 三参数在线调节
- `PID / LADRC / DDPG-LADRC / mDDPG-LADRC` 四组对比

当前默认主方法为：

- `DDPG-LADRC`

当前增强版对比方法为：

- `mDDPG-LADRC`

## 2. 代码主链位置

第三章主代码位于：

- `src/lc/control/controllers/`
- `src/lc/control/envs/`
- `src/lc/control/policies/`
- `src/lc/control/trainers/`
- `src/lc/control/experiments/`
- `src/lc/control/plotting/`

关键文件对应关系如下：

- `src/lc/control/controllers/pid.py`
  传统 PID 控制器基线。
- `src/lc/control/controllers/ladrc.py`
  基础 `LADRC` 控制器实现。
- `src/lc/control/controllers/adaptive_ladrc.py`
  强化学习调参包装器，负责把策略动作解释为 `omega_c / omega_o / b0` 的增量。
- `src/lc/control/envs/pybullet_axis_env.py`
  单轴 PyBullet 训练环境。
- `src/lc/control/envs/pybullet_eval_env.py`
  单轴 PyBullet 评测与时域轨迹采集入口。
- `src/lc/control/policies/mddpg_control.py`
  第三章控制策略封装，统一 `DDPG-LADRC` 与 `mDDPG-LADRC` 的控制接口。
- `src/lc/control/policies/stacking.py`
  状态堆叠和动作保持相关工具。
- `src/lc/control/trainers/pybullet_axis_trainer.py`
  第三章单轴训练、评估、整定和基准实验主入口。
- `src/lc/control/experiments/compare.py`
  第三章一键对比实验入口。
- `src/lc/control/plotting/plots.py`
  第三章图表生成模块。

## 3. 当前方法口径

### 3.1 被控对象

当前第三章默认被控对象为：

- 单轴位置跟踪对象

当前默认任务为：

- 跟踪参考位置轨迹
- 在存在扰动时维持跟踪精度与恢复能力

### 3.2 强化学习调参目标

强化学习不直接输出控制量，而是在线调节位置环 `LADRC` 三个核心参数：

- `delta_omega_c`
- `delta_omega_o`
- `delta_b0`

因此第三章的方法语义固定为：

- 控制器仍然是 `LADRC`
- 强化学习只负责在线调参
- 最终控制量由 `LADRC` 控制器生成

### 3.3 状态定义

当前控制观测为 5 维：

- 位置误差
- 速度
- 扰动估计输入
- 参考值
- 归一化时间

### 3.4 奖励函数

按照当前固定设定，奖励函数只由位置误差构成：

```text
reward = -|position_error|
```

该定义用于保持第三章的控制目标清晰，不在这一版里混入多目标复合奖励。

## 4. DDPG 与 mDDPG 的区别

### 4.1 DDPG-LADRC

基础 `DDPG-LADRC` 默认采用：

- 单步状态输入
- 每步更新一次动作
- `1-step` 回报

它是第三章当前默认主方法。

### 4.2 mDDPG-LADRC

`mDDPG-LADRC` 在基础 `DDPG-LADRC` 之上增加三项增强机制：

- 状态堆叠：使用过去 `n` 个时刻状态作为策略输入
- 动作保持：同一参数动作保持 `n` 个采样时刻
- `n-step` 自举回报：过去 `n` 步奖励参与目标构造

这三项增强用于提升：

- 短时历史利用能力
- 参数调节平滑性
- 时序收益建模能力

## 5. 当前实验组

第三章当前固定保留四组方法：

- `PID`
- `LADRC`
- `DDPG-LADRC`
- `mDDPG-LADRC`

其中：

- `PID` 是传统控制基线
- `LADRC` 是鲁棒控制基线
- `DDPG-LADRC` 是主方法
- `mDDPG-LADRC` 是增强版对比方法

## 6. 当前指标体系

第三章当前评估指标包括：

- `IAE`
- `RMSE`
- 超调量
- 调节时间
- 稳态误差
- 控制能量
- 扰动恢复时间

这些指标同时用于：

- 方法对比
- 机制消融
- 图表输出

## 7. 当前输出目录

第三章默认输出目录为：

`outputs/control/<difficulty>/`

当前常见输出文件包括：

- `summary.json`
- `metrics.csv`
- `training_ddpg.csv`
- `training_mddpg.csv`
- `ablation_metrics.csv`
- `checkpoints/ddpg_ladrc.pt`
- `checkpoints/mddpg_ladrc.pt`
- `figures/*.svg`

## 8. 当前图表能力

第三章当前已支持输出以下图表：

- 方法指标对比图
  - `mae`
  - `rmse`
  - `iae`
  - `overshoot`
  - `settling_time`
  - `control_energy`
  - `reward`
- 训练曲线
  - `reward`
  - `mae`
  - `actor_loss`
  - `critic_loss`
- 主消融图
  - `DDPG-LADRC vs mDDPG-LADRC`
- 机制消融图
  - `no_state_stack`
  - `no_action_hold`
  - `no_n_step`
- 典型时域响应图
  - `reference + output`
  - `error`
  - `control`
  - `disturbance`

## 9. 第三章方法插图

第三章方法部分已经单独整理了两张论文插图：

- [DDPG-LADRC总体框架图](/C:/context_mine/mine_code/GIT_Projects/lc/docs/figures/chapter3/ddpg_ladrc_framework.svg)
- [mDDPG增强机制图](/C:/context_mine/mine_code/GIT_Projects/lc/docs/figures/chapter3/mddpg_enhancement.svg)

配套文字说明位于：

- [chapter3_method_figures.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter3_method_figures.md)

## 10. 推荐运行方式

推荐入口：

```bash
python -m lc.entrypoints.train_control
```

如果本地运行需要指定 `PYTHONPATH`，Windows PowerShell 示例为：

```powershell
$env:PYTHONPATH='C:\context_mine\mine_code\GIT_Projects\lc\src'
python -m lc.entrypoints.train_control
```

## 11. 当前真实完成度判断

第三章当前已经具备：

- 控制器基线
- RL-LADRC 主链
- `DDPG` 与 `mDDPG` 两条训练设定
- 基本训练日志
- 对比实验
- 机制消融
- 论文插图和实验图输出

但还没有完全做到：

- 与旧无人机仿真控制对象逐项对齐
- 更复杂控制对象扩展到姿态环或三维位置环
- 大规模随机种子重复实验
- 更严格的收敛稳定性验证

## 12. 后续建议

后续继续加强第三章时，建议优先考虑：

- 把单轴对象进一步对齐到更真实的无人机位置控制对象
- 增加多随机种子实验与置信区间统计
- 增加统一论文插图版时域对比图
- 增加 episode 级最优策略保存与筛选
- 在不改变主口径的前提下补更完整的训练可复现实验配置
