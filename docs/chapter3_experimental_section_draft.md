# 第三章 实验设计与结果分析（初稿）

## 3.1 实验目的与总体思路

本章旨在验证基于 `LADRC` 的强化学习在线调参方法在单轴参考轨迹跟踪任务中的有效性、稳定性与可扩展性。围绕这一目标，实验部分按六个层次展开：

1. 主对比实验：比较 `PID`、固定参数 `LADRC`、`DDPG-LADRC` 与不同增强强度的 `mDDPG-LADRC`。
2. 消融实验：重点考察共享增强值 `v` 对性能和训练稳定性的影响。
3. 蒙特卡洛仿真：检验最优配置在多随机种子下的统计稳定性。
4. 泛化能力实验：比较不同难度场景下控制性能的变化规律。
5. 抗干扰性实验：分析在不同扰动强度和扰动形式下的恢复能力。
6. 扩展性实验：将单轴跟踪任务扩展到 `xy` 轴绕圆参考轨迹，验证方法的迁移能力。

当前已完成的结果主要来自 `x` 轴六阶段参考轨迹 (`rl_refline_six_phase`) 的训练与评估结果，以及一组 `v=1..10` 的共享增强值全量扫描结果。其余尚未补跑的实验已整理为正式实验清单，后续可按统一协议继续补齐。

## 3.2 实验平台与统一设置

### 3.2.1 控制任务

主实验任务为 `x` 轴六阶段参考轨迹跟踪，参考轨迹由“静止保持、正向匀速、扰动保持、反向匀速、恢复阶段、末端保持”六个阶段构成。该任务强调：

- 跟踪精度
- 扰动抑制能力
- 恢复速度
- 控制平滑性

控制器主体为 `LADRC`，强化学习部分不直接输出控制量，而是在线调节 `LADRC` 的关键参数。状态观测采用：

- `x`
- `vx`
- `pitch`
- `pitch_rate`

主实验统一使用：

- `difficulty = medium`
- `train_episodes = 500`
- `compare_episodes = 5`
- `seed = 7`
- 最优网络配置：`hidden_dim = 768`，`dropout = 0.25`
- 目标网络更新：`tau = 0.02`，`soft_update_interval = 10`
- 探索策略：线性衰减噪声 `0.10 -> 0.04`

### 3.2.2 评价指标

本章统一采用以下评价指标：

- `RMSE`
- `IAE`
- `Reward`
- `Overshoot`
- `Steady-state error`
- `Control energy`
- `Disturbance recovery time`
- `Velocity RMSE`

其中，`Reward` 反映策略在训练目标下的综合表现，`RMSE` 与 `IAE` 用于表征跟踪误差，`Control energy` 和 `Control variation` 用于衡量控制输入的代价与平滑性，`Disturbance recovery time` 则用于表征扰动下的恢复能力。

## 3.3 主对比实验

### 3.3.1 方法组设置

第三章正式主对比实验至少包含以下五组方法：

1. `PID`
2. `LADRC-fixed`
3. `DDPG-LADRC`
4. `mDDPG-LADRC (v=1)`
5. `mDDPG-LADRC (v=7)`

目前在 `x` 轴六阶段参考轨迹任务下，已完成并可直接复用的主对比结果包括：

- `PID`
- `LADRC-fixed`
- `mDDPG-LADRC (v=1)`
- `mDDPG-LADRC (v=4)`
- `mDDPG-LADRC (v=7)`

其中 `v=4` 为当前扫描得到的最优共享增强值，后文将其作为“当前最优增强配置”参与分析。`DDPG-LADRC` 在同一 `x` 轴 `rl_refline_six_phase` 主场景下的正式结果目前尚缺，后续需补跑后再并入最终主表。

### 3.3.2 当前已获得的主结果

基于现有扫描结果，`x` 轴六阶段主场景下各方法的关键指标如表 3-1 所示。

**表 3-1 现有主场景结果对比**

| 方法 | RMSE | IAE | Reward | Overshoot | Steady-state error | Control energy | Velocity RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PID | 0.5130 | 302.65 | -347.29 | 1.0658 | 0.2217 | 325.34 | 0.4558 |
| LADRC-fixed | 0.4016 | 234.93 | -300.89 | 0.8917 | 0.3618 | 39103.72 | 0.7198 |
| mDDPG-LADRC (v=1) | 0.2930 | 192.06 | -220.12 | 0.5224 | 0.0836 | 1602.53 | 0.2984 |
| mDDPG-LADRC (v=4) | 0.2504 | 170.69 | -192.56 | 0.4217 | 0.1384 | 1620.26 | 0.2483 |
| mDDPG-LADRC (v=7) | 0.3858 | 249.59 | -275.87 | 0.6804 | 0.3312 | 1473.15 | 0.2745 |

从表中可以看出：

- 与 `PID` 和固定参数 `LADRC` 相比，增强型方法在 `RMSE`、`IAE` 和 `Reward` 上均表现出明显优势。
- 当前最佳共享增强值为 `v=4`，其 `RMSE` 为 `0.2504`，`Reward` 为 `-192.56`，显著优于 `v=1` 与 `v=7`。
- `v=7` 并非当前最优值，这说明增强窗口并非越大越好，增强强度与任务结构之间存在最优匹配关系。

对应的现有论文图如下：

- 主方法现有可用柱状图：
  - [fig01_main_methods_available.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig01_main_methods_available.svg)
- 最优 `v=4` 时域响应图：
  - [fig04_best_v4_time_response.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig04_best_v4_time_response.svg)
- `v=1 / 4 / 7` 对比时域响应图：
  - [fig05_time_response_v1_v4_v7.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig05_time_response_v1_v4_v7.svg)

### 3.3.3 关于 DDPG-LADRC 的当前状态说明

当前仓库已有一组历史 `medium` 多轴 (`x/y/z`) 对比结果，其中包含 `DDPG-LADRC` 与 `mDDPG-LADRC` 的完整比较。该组结果表明，增强版方法相较于基础 `DDPG-LADRC` 在多轴历史任务上同样具有明显优势。例如，在历史 `medium` 多轴任务中：

- `DDPG-LADRC`：`RMSE = 0.0219`，`Reward = -3.7041`
- `mDDPG-LADRC`：`RMSE = 0.0044`，`Reward = -1.2046`

这一结果说明增强机制本身具有明确收益，但由于该历史结果对应的是 `piecewise_constant_velocity` 多轴任务，而非本章主场景 `x` 轴 `rl_refline_six_phase`，因此不能直接替代正式主表中的 `DDPG-LADRC` 位置。论文终稿中仍应补跑同口径的 `DDPG-LADRC` 主场景结果。

## 3.4 消融实验

### 3.4.1 消融实验设计原则

本章消融实验不再以“网络大小变化”为核心，而是以“是否做增强处理、增强强度多大”为主线。原因在于第三章方法贡献主要体现在：

- 是否引入增强版 `mDDPG`
- 增强处理的共享值 `v` 如何影响性能
- 探索率衰减是否改善训练稳定性

因此，本章消融重点设置为：

1. `DDPG-LADRC`
2. `mDDPG-LADRC (v=1)`
3. `mDDPG-LADRC (v=7)`
4. `mDDPG-LADRC (best-v=4)`

其中：

- `DDPG` 表示无增强基线
- `mDDPG(v=1)` 表示最小增强
- `mDDPG(v=7)` 表示强增强代表
- `mDDPG(v=4)` 表示当前最优增强配置

### 3.4.2 共享增强值扫描结果

针对 `v=1..10` 的全量扫描结果如图 3-1 所示：

- [fig02_v_sweep_rmse_reward.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig02_v_sweep_rmse_reward.svg)

从扫描结果可以总结出以下规律：

1. 共享增强值对最终性能影响显著。
2. 最优值出现在 `v=4`，而不是直觉上更大的 `v=7` 或 `v=10`。
3. `v=2` 和 `v=3` 的性能明显退化，说明增强处理与任务时间结构存在非线性耦合。
4. 当 `v` 从 `1` 增大到 `4` 时，性能显著提升；继续增大到 `7` 后，`RMSE` 和 `Reward` 均出现回退。

### 3.4.3 `v=1 / 4 / 7` 的奖励收敛分析

图 3-2 给出了 `v=1`、`v=4`、`v=7` 三组代表性共享值的 reward 收敛曲线：

- [fig03_reward_curves_v1_v4_v7.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig03_reward_curves_v1_v4_v7.svg)

对应的统计结果如下：

| 变体 | best reward | worst reward | last100 reward mean | last100 reward std | last100 rmse mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| mDDPG(v=1) | -118.15 | -907.27 | -319.37 | 104.23 | 0.4385 |
| mDDPG(v=4) | -79.56 | -1735.83 | -241.91 | 74.42 | 0.3229 |
| mDDPG(v=7) | -129.06 | -895.83 | -258.47 | 53.86 | 0.3530 |

可以看出：

- `v=4` 的最优 episode 表现最好，说明该共享增强值具有更高的性能上限。
- `v=7` 的后期波动小于 `v=1`，说明更大的共享增强值有助于提升训练稳定性。
- `v=4` 在性能和稳定性之间取得了更好的平衡，因此被选为当前主模型的最优增强值。

### 3.4.4 探索率递减与网络配置的辅助分析

为了进一步分析训练稳定性的来源，本文还比较了三组 `v=7` 条件下的网络与 target update 设置。结果如图 3-3 所示：

- [fig06_v7_tuning_comparison.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig06_v7_tuning_comparison.svg)

已有结果表明：

- 仅降低 `tau` 并不会自动带来更好结果；
- 当更大的网络容量与更保守的 target update 配置结合时，性能会重新提升；
- 探索率递减能明显改善后期 reward 的稳定性，但性能上限仍取决于共享增强值与网络配置的匹配。

这一部分更适合作为支撑性分析，用于说明当前最佳配置的来源，而不是论文主消融结论本身。

## 3.5 蒙特卡洛仿真设计

为了验证当前最优方法并非偶然得到，需要对 `mDDPG-LADRC (v=4)` 进行多随机种子重复训练。蒙特卡洛实验的设计如下：

- 固定主模型配置不变
- 共享增强值固定为 `v=4`
- 随机种子取 `7..16`
- 每个 seed 独立训练并评估一次

统计量包括：

- 平均值
- 标准差
- 95% 置信区间
- 最差结果
- 最优结果
- 训练成功率

绘图形式包括：

- `RMSE` 箱线图
- `Reward` 箱线图
- `mean ± 95% CI` 图

当前该部分尚未补跑，因此论文初稿中可先保留方法设计描述和图位，待结果完成后再补入数值。

## 3.6 泛化能力实验设计

为了验证方法对场景复杂度变化的适应性，本文拟采用难度分层实验：

- `easy`
- `medium`
- `hard`
- `extreme`

比较方法包括：

- `LADRC-fixed`
- `DDPG-LADRC`
- `mDDPG-LADRC (v=4)`

该实验的目标不是做 zero-shot 迁移，而是在统一的最优配置下比较不同难度任务中的性能变化趋势。因此正文应明确表述为“difficulty-adaptive evaluation”，而不是直接宣称“跨场景迁移”。

输出内容包括：

- 不同难度下的指标表
- difficulty 趋势折线图
- 高难度下的典型轨迹图

当前该部分尚未形成正式结果，后续完成后可直接作为本章泛化能力小节。

## 3.7 抗干扰性实验设计

抗干扰实验应单独成节，不建议和难度泛化混在一起。建议将扰动划分为四种类型：

1. 无扰动
2. 训练分布内扰动
3. 强持续扰动
4. 脉冲扰动

比较方法仍然固定为：

- `LADRC-fixed`
- `DDPG-LADRC`
- `mDDPG-LADRC (v=4)`

重点分析指标包括：

- `disturbance_recovery_time`
- `RMSE`
- `Reward`
- `Control energy`

同时至少保留一张“扰动阶段放大时域图”，用于直观展示扰动触发后不同方法的恢复速度和误差峰值变化。

## 3.8 `xy` 绕圆扩展性实验设计

为了进一步验证方法不仅适用于单轴位置跟踪，还具有向双轴耦合任务扩展的潜力，本文拟增加 `xy` 绕圆跟踪任务作为扩展实验。其参考轨迹定义为：

\[
x = R\cos(\omega t), \quad y = R\sin(\omega t)
\]

对比方法包括：

- `LADRC-fixed`
- `DDPG-LADRC`
- `mDDPG-LADRC (v=4)`

评价指标应扩展为：

- 二维轨迹偏差
- 圆半径误差
- 相位滞后
- 二维累计误差面积

输出图至少包括：

- `xy` 平面轨迹图
- 半径误差曲线
- 相位滞后或二维跟踪误差曲线

该组实验不要求像主实验那样做全量蒙特卡洛，但至少应给出一组代表性结果，以证明方法具有扩展性。

## 3.9 当前结果的阶段性结论

基于当前已完成的数据，可以先得到以下阶段性结论：

1. 在 `x` 轴六阶段参考轨迹任务下，增强型 `mDDPG-LADRC` 相对于 `PID` 和固定参数 `LADRC` 具有明显优势。
2. 共享增强值并非越大越好，当前最佳值为 `v=4`。
3. `v=1`、`v=4`、`v=7` 三组对比表明，增强强度不仅影响性能上限，也影响训练稳定性。
4. 探索率递减对 reward 后期稳定性是有效的，但最优结果仍依赖共享增强值和网络配置之间的匹配。
5. 当前最关键的缺口，是在同一主场景下补齐 `DDPG-LADRC` 的正式结果，并完成 `mDDPG(v=4)` 的蒙特卡洛稳定性验证。

## 3.10 本章后续补充内容

为了形成第三章最终定稿，后续应按以下顺序补全实验：

1. 补跑 `DDPG-LADRC` 的 `x` 轴 `rl_refline_six_phase` 正式结果
2. 完成 `mDDPG(v=4)` 的蒙特卡洛仿真
3. 完成 difficulty 泛化
4. 完成抗干扰性实验
5. 完成 `xy` 绕圆扩展性实验

完成上述实验后，本章将形成一套完整的论文实验体系：既有主结果，也有机制消融、统计稳定性验证、跨场景泛化与任务扩展，从而能够较完整地支撑方法的有效性与可推广性论证。

## 3.11 相关图表与数据路径

### 已生成论文图

- [fig00_experiment_status_matrix.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig00_experiment_status_matrix.svg)
- [fig01_main_methods_available.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig01_main_methods_available.svg)
- [fig02_v_sweep_rmse_reward.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig02_v_sweep_rmse_reward.svg)
- [fig03_reward_curves_v1_v4_v7.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig03_reward_curves_v1_v4_v7.svg)
- [fig04_best_v4_time_response.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig04_best_v4_time_response.svg)
- [fig05_time_response_v1_v4_v7.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig05_time_response_v1_v4_v7.svg)
- [fig06_v7_tuning_comparison.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig06_v7_tuning_comparison.svg)

### 已生成数据总表

- [chapter3_master_metrics.csv](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/data/chapter3_master_metrics.csv)
- [chapter3_reward_curve_stats.csv](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/data/chapter3_reward_curve_stats.csv)
- [chapter3_thesis_experiment_manifest.csv](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/data/chapter3_thesis_experiment_manifest.csv)
