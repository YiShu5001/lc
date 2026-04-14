# Chapter 3 `x` 轴 RL RefLine 并行对比实验分析报告

## 1. 报告目的

本文总结第三章 `x` 轴 `RLcontrolRefLine` 六阶段任务下，三组并行对比实验的配置差异、核心结果、训练稳定性变化，以及后续最值得继续验证的优化方向。

本轮实验统一设置如下：

- 任务：`x` 轴六阶段参考轨迹跟踪
- 控制器：`mDDPG-LADRC`
- 共享值：`v=7`
- 训练轮数：`500 episode`
- 固定 `r=10`
- 状态：`x / vx / pitch / pitch_rate`
- 探索率策略：按 episode 递减
- 快照保存：每 `50` 集保存一次训练轨迹

基线方法保持不变：

- `PID`
- `LADRC`

## 2. 三组实验配置

| 实验 | 目标 | hidden_dim | dropout | tau | soft update interval | exploration |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Exp1 | 慢速线性衰减 | 512 | 0.20 | 0.05 | 20 | 0.10 -> 0.04 |
| Exp2 | 慢速衰减 + 更稳 target update | 512 | 0.20 | 0.02 | 10 | 0.10 -> 0.04 |
| Exp3 | 中等放大模型 | 768 | 0.25 | 0.02 | 10 | 0.10 -> 0.04 |

输出目录分别为：

- `outputs/control/x_axis_rl_refline__exp-v7ep500decay-slow__ep-500__v-7__noise-linear-0.1-to-0.04__net-512__drop-0.2`
- `outputs/control/x_axis_rl_refline__exp-v7ep500decay-slow-tau02__ep-500__v-7__noise-linear-0.1-to-0.04__net-512__drop-0.2`
- `outputs/control/x_axis_rl_refline__exp-v7ep500decay-slow-tau02-net768__ep-500__v-7__noise-linear-0.1-to-0.04__net-768__drop-0.25`

## 3. 核心结果对比

### 3.1 最终评估指标

| 方法 | rmse | iae | reward | overshoot | velocity rmse | control energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PID | 0.5130 | 302.65 | -347.29 | 1.0658 | 0.4558 | 325.34 |
| LADRC | 0.4016 | 234.93 | -300.89 | 0.8917 | 0.7198 | 39103.72 |
| Exp1 | 0.3224 | 208.98 | -249.54 | 0.5186 | 0.4234 | 7128.21 |
| Exp2 | 0.3608 | 230.92 | -274.07 | 0.7227 | 0.4194 | 3958.27 |
| Exp3 | 0.3028 | 194.09 | -227.03 | 0.5403 | 0.3355 | 2519.34 |

### 3.2 训练过程稳定性

| 实验 | best episode | best reward | worst reward | last 100 reward mean | last 100 reward std | last 100 rmse mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exp1 | 307 | -154.37 | -1042.53 | -298.06 | 45.85 | 0.3915 |
| Exp2 | 208 | -165.19 | -1539.40 | -298.05 | 45.49 | 0.4014 |
| Exp3 | 257 | -108.34 | -1156.24 | -238.94 | 63.85 | 0.3150 |

## 4. 结果解读

### 4.1 探索率递减方向是正确的

从前一轮 `0.10 -> 0.02` 的实验到这次三组 `0.10 -> 0.04` 的实验，可以看到一个明确趋势：

- reward 后期波动明显收敛
- 最优 episode 基本都出现在中后段
- 训练过程不再频繁出现“后期整体崩掉”的现象

这说明“探索率随 episode 逐步降低”这个方向是正确的。它改善的不是单次最优值，而是训练后期的可控性和重复性。

### 4.2 `tau=0.02` 单独变小，没有直接带来更好结果

Exp2 相比 Exp1 只改了：

- `tau: 0.05 -> 0.02`
- `soft_update_interval: 20 -> 10`

结果却从：

- `rmse: 0.3224 -> 0.3608`
- `reward: -249.54 -> -274.07`

说明在 `hidden_dim=512` 的情况下，更保守的 target 网络更新并没有自动提升性能，反而让学习更偏保守。这里的原因更像是：

- target update 更稳了
- 但策略更新速度和表示能力没有同步增强
- 于是训练更“稳”，但优化推进不够

### 4.3 放大模型后，保守更新策略开始体现优势

Exp3 在 Exp2 基础上进一步把模型从 `512` 提到 `768`，并把 `dropout` 从 `0.20` 提到 `0.25`。结果明显改善：

- `rmse` 从 `0.3608` 降到 `0.3028`
- `iae` 从 `230.92` 降到 `194.09`
- `reward` 从 `-274.07` 提升到 `-227.03`
- `velocity rmse` 从 `0.4194` 降到 `0.3355`

这说明：

- 模型加大是有价值的
- 但它不是单独起作用，而是和更稳的 target update 组合后才体现优势
- 也就是说，`更大模型 + 更保守更新` 比 `小模型 + 更保守更新` 更匹配

## 5. 模型规模分析

### 5.1 参数量变化

| 模型配置 | actor params | critic params | total params |
| --- | ---: | ---: | ---: |
| `hidden_dim=512, dropout=0.20` | 279,043 | 279,553 | 558,596 |
| `hidden_dim=768, dropout=0.25` | 615,171 | 615,937 | 1,231,108 |

`768` 模型的参数量约为 `512` 模型的 `2.20x`。

### 5.2 模型文件大小

| 模型配置 | best_model.pt |
| --- | ---: |
| `hidden_dim=512, dropout=0.20` | 2,239,605 bytes |
| `hidden_dim=768, dropout=0.25` | 4,929,653 bytes |

从资源开销看，`768` 版本虽然明显更大，但仍然处于可接受范围，没有大到不可训练。结合最终指标，当前结论是：

- 可以继续使用 `768`
- 现阶段不建议直接跳到 `1024`
- 更值得优先优化的是探索率衰减节奏，而不是继续粗暴扩网络

## 6. 与历史最好结果的关系

之前“调大网络后 300 episode”的最好记录为：

- 最佳 `v=7`
- `rmse ≈ 0.2090`
- `reward ≈ -161.31`

本轮三组并行实验都没有超过这一结果。当前最好的 Exp3 仍有差距：

- `rmse: 0.3028`
- `reward: -227.03`

这说明本轮优化主要解决了：

- 训练后期稳定性
- 更稳的探索和更新节奏

但还没有完全恢复到此前最强性能。换句话说，本轮更像是“提高可训练性和可重复性”，而不是“刷新最优上限”。

## 7. 结论

本轮实验可以给出四个明确结论：

1. 探索率随 episode 逐步降低是正确方向，能够明显改善 reward 稳定性。
2. `tau=0.02` 和更频繁 soft update 在 `512` 模型上会让训练更保守，单独使用并不划算。
3. 当模型增大到 `768` 后，保守 target update 才开始体现优势，说明模型容量和更新节奏存在耦合关系。
4. 当前最值得保留的候选配置是 Exp3，而不是 Exp1 或 Exp2。

## 8. 下一轮实验建议

建议下一轮不要继续同时改太多变量，而是围绕 Exp3 做两组对照：

### 方案 A：保持 Exp3，只放慢探索衰减

- 保持：
  - `hidden_dim=768`
  - `dropout=0.25`
  - `tau=0.02`
  - `soft_update_interval=10`
- 调整：
  - 探索率改为更慢衰减
  - 例如前 `300` 集保持较高探索，后 `200` 集再下降到 `0.04`

目标：

- 让模型在中后期仍有一定搜索空间
- 避免过早收缩到保守策略

### 方案 B：保持 Exp3，只回调 `tau`

- 保持：
  - `hidden_dim=768`
  - `dropout=0.25`
  - 探索率 `0.10 -> 0.04`
- 调整：
  - `tau: 0.02 -> 0.03` 或 `0.05`

目标：

- 验证当前性能瓶颈是不是 target update 过于保守
- 判断更大模型是否能承受更积极的参数同步

如果这两组里有一组能把 `reward` 重新拉回 `-200` 附近，同时把 `rmse` 压到 `0.26 ~ 0.28`，就值得再向“逼近历史最好值”继续推进。

## 9. 相关文件

- [Exp1 summary](D:/ZhangC/lc_codex_ch4_run/outputs/control/x_axis_rl_refline__exp-v7ep500decay-slow__ep-500__v-7__noise-linear-0.1-to-0.04__net-512__drop-0.2/summary.json)
- [Exp2 summary](D:/ZhangC/lc_codex_ch4_run/outputs/control/x_axis_rl_refline__exp-v7ep500decay-slow-tau02__ep-500__v-7__noise-linear-0.1-to-0.04__net-512__drop-0.2/summary.json)
- [Exp3 summary](D:/ZhangC/lc_codex_ch4_run/outputs/control/x_axis_rl_refline__exp-v7ep500decay-slow-tau02-net768__ep-500__v-7__noise-linear-0.1-to-0.04__net-768__drop-0.25/summary.json)
- [分析报告 Markdown](D:/ZhangC/lc_codex_ch4_run/docs/chapter3_x_axis_rl_refline_parallel_suite_analysis.md)
