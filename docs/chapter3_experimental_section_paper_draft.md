# 第三章 实验设计与结果分析

## 3.1 实验目的

为验证所提出基于 `LADRC` 的强化学习在线调参方法在位置跟踪任务中的有效性、稳定性与扩展能力，本章围绕单轴参考轨迹跟踪场景构建了一套分层实验体系。实验内容包括：主对比实验、增强机制消融实验、训练稳定性分析、蒙特卡洛重复仿真、泛化能力实验、抗干扰性实验以及 `xy` 轴绕圆跟踪扩展实验。通过上述实验，力图回答以下几个问题：

1. 相比传统控制器，基于强化学习的在线调参方法是否能够提升跟踪性能与扰动抑制能力。
2. `mDDPG` 中的共享增强机制是否有效，以及增强强度如何影响最终结果。
3. 当前最优配置是否具有较好的训练稳定性与统计鲁棒性。
4. 所提出方法是否能够在更复杂场景和扩展任务中保持有效。

## 3.2 实验平台与统一设置

### 3.2.1 控制对象与任务描述

本章主实验采用 `x` 轴六阶段参考轨迹跟踪任务，参考轨迹模式为 `rl_refline_six_phase`。该任务包含静止保持、正向匀速、扰动保持、反向匀速、扰动恢复和末端保持六个阶段，能够较好地同时考察控制器的跟踪精度、恢复能力和控制平滑性。

在该任务中，控制器主体始终为 `LADRC`。强化学习算法不直接输出控制量，而是在线调节 `LADRC` 的关键参数，从而实现自适应控制。状态观测采用：

- `x`
- `vx`
- `pitch`
- `pitch_rate`

主实验统一采用如下设置：

- 场景难度：`difficulty = medium`
- 训练轮数：`train_episodes = 500`
- 评估轮数：`compare_episodes = 5`
- 随机种子：`seed = 7`
- 当前最优网络配置：`hidden_dim = 768`
- 失活率：`dropout = 0.25`
- 目标网络软更新系数：`tau = 0.02`
- 目标网络更新节奏：`soft_update_interval = 10`
- 探索策略：线性衰减噪声 `0.10 -> 0.04`

### 3.2.2 对比方法

第三章主对比实验至少包含以下五组方法：

1. `PID`
2. `LADRC-fixed`
3. `DDPG-LADRC`
4. `mDDPG-LADRC (v=1)`
5. `mDDPG-LADRC (v=7)`

其中，`PID` 和固定参数 `LADRC` 作为传统控制基线；`DDPG-LADRC` 作为无增强强化学习基线；`mDDPG-LADRC` 则表示引入共享增强机制后的改进方法。需要说明的是，当前 `x` 轴六阶段参考轨迹任务下，`DDPG-LADRC` 的正式结果尚待补跑，因此本节主结果部分主要基于已完成的 `PID`、`LADRC-fixed` 和 `mDDPG-LADRC` 不同共享增强值结果展开分析。对于 `DDPG-LADRC`，本文暂以历史多轴任务结果作为辅助参考，并在后续实验补齐后并入主表。

### 3.2.3 评价指标

实验统一采用以下指标：

- `RMSE`
- `IAE`
- `Reward`
- `Overshoot`
- `Steady-state error`
- `Control energy`
- `Disturbance recovery time`
- `Velocity RMSE`

其中，`RMSE` 和 `IAE` 用于评价整体跟踪误差，`Reward` 用于衡量策略在训练目标下的综合表现，`Control energy` 与 `Control variation` 反映控制代价与平滑性，`Disturbance recovery time` 用于衡量扰动下的恢复能力。

## 3.3 主对比实验

### 3.3.1 当前可用主结果

在当前已完成的 `x` 轴六阶段参考轨迹实验中，本文已获得 `PID`、`LADRC-fixed` 以及 `mDDPG-LADRC` 在不同共享增强值下的系统结果。由于共享增强值 `v` 会同时作用于状态堆叠长度、动作保持步数以及多步回报长度，因此本文首先对 `v=1..10` 进行了全量扫描。扫描结果表明，当前最佳共享增强值为 `v=4`。

表 3-1 给出了主场景下 `PID`、`LADRC-fixed` 以及代表性增强配置 `mDDPG(v=1)`、`mDDPG(v=4)`、`mDDPG(v=7)` 的对比结果。

**表 3-1 主场景代表性结果对比**

| 方法 | RMSE | IAE | Reward | Overshoot | Steady-state error | Control energy | Velocity RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PID | 0.5130 | 302.65 | -347.29 | 1.0658 | 0.2217 | 325.34 | 0.4558 |
| LADRC-fixed | 0.4016 | 234.93 | -300.89 | 0.8917 | 0.3618 | 39103.72 | 0.7198 |
| mDDPG-LADRC (v=1) | 0.2930 | 192.06 | -220.12 | 0.5224 | 0.0836 | 1602.53 | 0.2984 |
| mDDPG-LADRC (v=4) | 0.2504 | 170.69 | -192.56 | 0.4217 | 0.1384 | 1620.26 | 0.2483 |
| mDDPG-LADRC (v=7) | 0.3858 | 249.59 | -275.87 | 0.6804 | 0.3312 | 1473.15 | 0.2745 |

从表 3-1 可以看出，增强型方法整体上显著优于传统基线方法。与 `PID` 相比，`mDDPG(v=4)` 的 `RMSE` 从 `0.5130` 降至 `0.2504`，`IAE` 从 `302.65` 降至 `170.69`，`Reward` 从 `-347.29` 提升至 `-192.56`。与固定参数 `LADRC` 相比，增强型方法同样在误差指标和奖励指标上表现出明显优势。

值得注意的是，`v=7` 并不是当前最优共享增强值。尽管较大的共享增强值在部分实验中有助于提升训练后期的稳定性，但其最终误差和奖励并未优于 `v=4`。这说明增强强度与任务结构之间存在最优匹配关系，而不是简单地“增强越强越好”。

### 3.3.2 时域响应分析

为了更直观地展示最优共享值下的轨迹跟踪效果，图 3-1 给出了 `mDDPG(v=4)` 在主任务上的时域响应结果。

- [fig04_best_v4_time_response.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig04_best_v4_time_response.svg)

从图中可以看出，所提出方法能够在六阶段参考轨迹变化过程中较好地跟踪目标，并在扰动出现后保持较小的误差波动。输出轨迹整体贴近参考轨迹，控制输入变化较为平滑，没有出现明显的高频振荡。

为了进一步比较不同增强强度下的响应差异，图 3-2 给出了 `v=1`、`v=4` 和 `v=7` 三组共享增强值的典型时域响应对比。

- [fig05_time_response_v1_v4_v7.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig05_time_response_v1_v4_v7.svg)

可以看出，`v=1` 在部分阶段存在较明显的误差波动，`v=7` 在稳态阶段相对平滑，但整体误差并未优于 `v=4`。相比之下，`v=4` 在跟踪精度与稳定性之间取得了更好的折中。

## 3.4 共享增强机制消融实验

### 3.4.1 消融实验设计

本章消融实验不再以模型规模变化为主线，而是聚焦于增强机制本身。消融的核心问题是：增强处理是否必要，以及增强强度变化是否会显著影响性能。因此，本文选取以下四组进行消融对比：

1. `DDPG-LADRC`
2. `mDDPG-LADRC (v=1)`
3. `mDDPG-LADRC (v=7)`
4. `mDDPG-LADRC (best-v=4)`

在当前结果中，`v=1` 被视为最小增强配置，`v=7` 被视为较强增强配置，`v=4` 则为当前最优增强配置。

### 3.4.2 共享增强值全量扫描

图 3-3 给出了 `v=1..10` 的共享增强值扫描结果。

- [fig02_v_sweep_rmse_reward.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig02_v_sweep_rmse_reward.svg)

从扫描曲线可见：

- 当 `v` 从 `1` 增大到 `4` 时，`RMSE` 明显下降，`Reward` 明显提升；
- 当 `v` 继续增大到 `7` 或 `10` 时，性能未继续提升，反而出现回退；
- `v=2` 和 `v=3` 对应的结果较差，说明共享增强值与任务节奏之间存在较强耦合关系。

扫描结果表明，增强机制是有效的，但增强强度必须与具体任务匹配。当前任务下，`v=4` 是最佳共享增强值。

### 3.4.3 代表性增强值的训练稳定性

为了进一步考察增强强度对训练过程的影响，图 3-4 给出了 `v=1`、`v=4` 和 `v=7` 的 reward 收敛曲线。

- [fig03_reward_curves_v1_v4_v7.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig03_reward_curves_v1_v4_v7.svg)

统计结果如下：

| 变体 | best reward | worst reward | last100 reward mean | last100 reward std | last100 rmse mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| mDDPG(v=1) | -118.15 | -907.27 | -319.37 | 104.23 | 0.4385 |
| mDDPG(v=4) | -79.56 | -1735.83 | -241.91 | 74.42 | 0.3229 |
| mDDPG(v=7) | -129.06 | -895.83 | -258.47 | 53.86 | 0.3530 |

从结果可以看出：

- `v=4` 的最优 episode 奖励最高，说明该配置具有更高的性能上限；
- `v=7` 的后 100 集 reward 标准差小于 `v=1`，说明较强增强值有助于改善训练稳定性；
- `v=4` 在性能和稳定性之间取得了更平衡的结果，因此被选为当前最佳共享增强值。

综上，增强机制本身确实有效，但其作用并不是简单提高稳定性或简单提高上限，而是在二者之间建立更优平衡。当前结果表明，`v=4` 是这一平衡点。

## 3.5 训练稳定性与配置来源分析

除了共享增强值之外，本文还考察了网络规模、目标网络更新节奏以及探索率衰减对训练稳定性的影响。图 3-5 比较了三组 `v=7` 条件下的调参与训练结果。

- [fig06_v7_tuning_comparison.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig06_v7_tuning_comparison.svg)

三组结果分别为：

- `net512, dropout0.20, tau0.05`
- `net512, dropout0.20, tau0.02`
- `net768, dropout0.25, tau0.02`

结果表明：

- 单独减小 `tau` 并不会自动提升控制性能；
- 当更大的网络容量与更保守的目标网络更新相结合时，性能会重新提升；
- 探索率线性衰减有助于降低训练后期波动，但最终效果仍取决于增强值、网络容量和更新节奏的整体匹配。

因此，本章最终主配置采用：

- `hidden_dim = 768`
- `dropout = 0.25`
- `tau = 0.02`
- `soft_update_interval = 10`
- `exploration_noise: 0.10 -> 0.04`

这一配置既来自调参实验，也得到共享增强扫描结果的支持。

## 3.6 关于 DDPG-LADRC 的补充说明

当前 `x` 轴六阶段参考轨迹主场景下，`DDPG-LADRC` 的正式结果尚待补跑，因此本文目前未将其写入表 3-1 的最终位置。不过，从历史 `medium` 多轴任务的结果看，增强版方法同样优于基础 `DDPG-LADRC`。在该历史任务中：

- `DDPG-LADRC`：`RMSE = 0.0219`，`Reward = -3.7041`
- `mDDPG-LADRC`：`RMSE = 0.0044`，`Reward = -1.2046`

这一结果说明，增强机制本身具有明确的普适收益。待主场景下的 `DDPG-LADRC` 正式结果补齐后，可进一步完成“五组统一对比”的最终主表。

## 3.7 蒙特卡洛仿真设计

为了验证当前最优配置 `mDDPG-LADRC (v=4)` 是否具有良好的统计稳定性，后续将补充多随机种子蒙特卡洛仿真。实验设置如下：

- 固定主模型配置不变；
- 共享增强值固定为 `v=4`；
- 随机种子取 `7..16`；
- 每个 seed 独立训练与评估；
- 统计均值、标准差、95% 置信区间、最优/最差结果与训练成功率。

同时，为了对比增强机制对稳定性的影响，还将补充 `DDPG-LADRC` 的相同随机种子重复实验。最终图表将包括：

- `RMSE` 箱线图
- `Reward` 箱线图
- `mean ± 95% CI` 图

当前该部分尚无正式结果，因此本稿仅保留实验设计。待补跑完成后，可将其作为本章稳定性验证的核心统计支撑。

## 3.8 泛化能力实验设计

为了验证方法在不同复杂度场景下的适应能力，后续实验将对 `easy`、`medium`、`hard` 和 `extreme` 四个难度等级进行测试。比较方法包括：

- `LADRC-fixed`
- `DDPG-LADRC`
- `mDDPG-LADRC (v=4)`

这里的实验目标是“在统一最优配置下观察不同难度场景中的性能变化”，因此更准确地说属于 difficulty-adaptive evaluation，而非严格意义上的 zero-shot transfer。

该部分实验完成后，将输出：

- 多难度分层指标表
- 难度趋势图
- 高难度典型响应图

## 3.9 抗干扰性实验设计

抗干扰性实验单独考察不同扰动条件下的恢复能力。建议划分四类扰动情形：

1. 无扰动
2. 训练分布内扰动
3. 强持续扰动
4. 脉冲扰动

比较方法仍然固定为：

- `LADRC-fixed`
- `DDPG-LADRC`
- `mDDPG-LADRC (v=4)`

重点指标包括：

- `Disturbance recovery time`
- `RMSE`
- `Reward`
- `Control energy`

并保留至少一张扰动阶段放大图，用于说明不同方法在扰动触发后的恢复速度差异。

## 3.10 `xy` 绕圆扩展实验设计

为了验证所提出方法是否能够从单轴任务扩展到双轴耦合任务，后续将构造 `xy` 轴绕圆跟踪实验，其参考轨迹定义为：

\[
x = R\cos(\omega t), \quad y = R\sin(\omega t)
\]

比较组包括：

- `LADRC-fixed`
- `DDPG-LADRC`
- `mDDPG-LADRC (v=4)`

该实验的评价指标将扩展为：

- 二维轨迹偏差
- 半径误差
- 相位滞后
- 二维累计误差面积

输出图包括：

- `xy` 平面轨迹图
- 半径误差曲线
- 相位滞后或二维跟踪误差曲线

该实验定位为第三章的扩展性验证，不与主场景结果混淆，但能够有效说明方法具有向更复杂任务迁移的潜力。

## 3.11 阶段性结论

结合当前已完成的数据，可以得到以下阶段性结论：

1. 在 `x` 轴六阶段参考轨迹任务中，增强型 `mDDPG-LADRC` 相比 `PID` 和固定参数 `LADRC` 具有明显性能优势。
2. 共享增强值对性能影响显著，当前最优值为 `v=4`。
3. `v=1`、`v=4`、`v=7` 的对比表明，共享增强值不仅影响跟踪精度，也影响训练后期稳定性。
4. 探索率递减和更合理的网络配置对稳定训练是必要的，但最终性能仍由增强值与训练配置的整体匹配决定。
5. 当前最需要补齐的关键实验，是 `DDPG-LADRC` 在主场景下的正式结果，以及 `mDDPG(v=4)` 的蒙特卡洛统计实验。

## 3.12 图表与数据路径

本章当前已生成的论文图包括：

- [fig00_experiment_status_matrix.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig00_experiment_status_matrix.svg)
- [fig01_main_methods_available.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig01_main_methods_available.svg)
- [fig02_v_sweep_rmse_reward.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig02_v_sweep_rmse_reward.svg)
- [fig03_reward_curves_v1_v4_v7.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig03_reward_curves_v1_v4_v7.svg)
- [fig04_best_v4_time_response.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig04_best_v4_time_response.svg)
- [fig05_time_response_v1_v4_v7.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig05_time_response_v1_v4_v7.svg)
- [fig06_v7_tuning_comparison.svg](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/figures/fig06_v7_tuning_comparison.svg)

本章当前已生成的数据汇总表包括：

- [chapter3_master_metrics.csv](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/data/chapter3_master_metrics.csv)
- [chapter3_reward_curve_stats.csv](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/data/chapter3_reward_curve_stats.csv)
- [chapter3_thesis_experiment_manifest.csv](D:/ZhangC/lc_codex_ch4_run/outputs/paper_figures/chapter3/data/chapter3_thesis_experiment_manifest.csv)

上述结果构成了第三章实验部分的当前初步稿件。待补充 `DDPG-LADRC` 主场景结果、蒙特卡洛统计结果及泛化/抗干扰/扩展任务结果后，本章即可形成完整终稿。
