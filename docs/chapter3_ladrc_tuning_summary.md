# 第三章 LADRC 参数整定汇总

## 1. 目标与口径

第三章当前控制口径保持不变：

- 控制器主体仍然是 `LADRC`
- 强化学习后续输出仍然是连续参数动作：
  - `action[0] -> b0`
  - `action[1] -> wc`
  - `action[2] -> k`
- 不做离散模式学习
- 离线整定的作用是先给出若干有物理意义的锚点参数，再据此确定 RL 的连续动作范围

当前已经形成 3 类 `x` 轴参数角色：

1. 常规默认参数
2. 稳态抗扰参数
3. 大误差 + 稳定扰动环境下的快速参数

## 2. 当前默认参数

当前 `default_axis_params.json` 中保留的默认参数为：

- `x`: `b0=30.5, wc=1.5, k=10.5, r=12`
- `y`: `b0=30.5, wc=1.5, k=10.5, r=12`
- `z`: `b0=100.0, wc=0.05, k=0.5, r=2`

其中：

- `x/y` 采用同一组横向位置环参数
- `z` 单独采用保守高度修正参数

## 3. `x` 轴大误差参数

最初用于 `hold_step_hold` 阶跃响应整定、并最终保留为当前默认 `x` 轴基线的参数是：

- `b0=30.5`
- `wc=1.5`
- `k=10.5`
- `wo=15.75`
- `r=12`

这一组参数的意义是：

- 阶跃场景下能够较快接近参考
- `rmse` 和 `steady_state_error` 已达到可接受水平
- 相比更激进的组合，整体可用性更稳定

## 4. `x` 轴稳态抗扰参数

为给 RL 提供“稳态工作点”锚点，又单独进行了 `x` 轴稳态参数整定。

当前稳态参数结果为：

- `b0=30.5`
- `wc=0.8`
- `k=7.0`
- `wo=5.6`
- `r=12`

实验口径：

- 主场景：`x_hold_disturbance_hold`
- 验证场景：`x_small_step_hold`

这组参数的作用不是追求大误差快速靠近，而是：

- 降低稳态振动
- 降低控制变化幅度
- 在有横向扰动时提高保持与恢复表现

对应输出目录：

- `outputs/control_pybullet_manual_tuning/x_steady_tuning/`

关键结果文件：

- `recommended_x_steady_params.json`
- `comparison_against_fast_x.json`
- `comparison_against_pid.json`
- `final_compare/pid_vs_fast_vs_steady_response.svg`

## 5. `x` 轴大误差 + 稳定扰动环境快速参数

随后又在“阶跃 + 稳定扰动环境”下，按最初 `x` 轴那套顺序化流程重新扫描了参数：

1. 先扫 `b0`
2. 再扫 `wc`
3. 最后扫 `k`
4. 再做局部细扫直到找到不再贴边的局部最优点

最终在该环境下得到的局部最优参数为：

- `b0=1.0`
- `wc=8.25`
- `k=4.0`
- `wo=33.0`
- `r=12`

这组参数的意义是：

- 面向大误差且存在稳定外扰的快速工作点
- 更强调快速抵达和扰动下的激进修正
- 不能直接替代默认参数，但非常适合作为 RL 连续动作范围的另一端锚点

对应输出目录：

- `outputs/control_pybullet_manual_tuning/x_disturbed_rescan/`
- `outputs/control_pybullet_manual_tuning/x_disturbed_rescan_local/`
- `outputs/control_pybullet_manual_tuning/x_disturbed_rescan_fine2/`
- `outputs/control_pybullet_manual_tuning/x_disturbed_rescan_final/`

最终推荐以最后一轮为准：

- `outputs/control_pybullet_manual_tuning/x_disturbed_rescan_final/`

## 6. 三线对比结果

目前已经补了 `PID + 当前默认 x 轴 LADRC + 扰动环境最优 x 轴 LADRC` 的三线对比结果。

输出目录：

- `outputs/control_pybullet_manual_tuning/final_compare_disturbed_threeway/x/`

核心文件：

- `pid_current_candidate_response.svg`
- `summary.json`
- `metrics.csv`
- `pid_timeseries.csv`
- `current_ladrc_timeseries.csv`
- `candidate_ladrc_timeseries.csv`

这组图的作用是：

- 直接比较 `PID`
- 比较当前默认 `x` 参数
- 比较“稳定扰动环境下重新扫描得到”的快速参数

## 7. 当前用于 RL 范围设计的 `x` 轴锚点

如果后续只为了给第三章 `DDPG / mDDPG` 确定连续动作范围，那么目前最有价值的不是单一最优点，而是多锚点口径：

### 常规快速锚点

- `b0=30.5`
- `wc=1.5`
- `k=10.5`

### 稳态抗扰锚点

- `b0=30.5`
- `wc=0.8`
- `k=7.0`

### 扰动大误差快速锚点

- `b0=1.0`
- `wc=8.25`
- `k=4.0`

后续 RL 若仍输出连续 `b0 / wc / k`，则动作边界可以围绕这些锚点构造，而不是拍脑袋给范围。

## 8. 当前结论

当前第三章 `x` 轴整定工作可以先形成下面的结论：

- 默认在线路上保留：
  - `x/y = 30.5 / 1.5 / 10.5 / 12`
- 稳态抗扰补充锚点为：
  - `x_steady = 30.5 / 0.8 / 7.0 / 12`
- 大误差 + 稳定扰动快速锚点为：
  - `x_disturbed_fast = 1.0 / 8.25 / 4.0 / 12`

后续若继续推进 RL 参数范围设计，应优先做：

1. 用这三组 `x` 轴锚点推导连续动作边界
2. 再把同样思路迁移到 `y`
3. `z` 轴继续单独口径处理

## 9. `x` 轴 `r` 的统一口径

为判断 `r` 是否可以在“大误差快速阶段”和“稳态抗扰阶段”共用，已经额外做了 `x` 轴 `r` 均衡扫描。

扫描口径：

- 快速场景：`hold_step_hold`
- 稳态场景：`x_hold_disturbance_hold`
- 稳态验证场景：`x_small_step_hold`

固定参数：

- 快速参数组：`b0=30.5, wc=1.5, k=10.5`
- 稳态参数组：`b0=30.5, wc=0.8, k=7.0`

扫描范围：

- `r = 4, 6, 8, 10, 12, 15, 18, 20, 25, 30`

当前结果：

- 推荐共享 `r = 10`

对应输出目录：

- `outputs/control_pybullet_manual_tuning/x_r_balance/`

关键文件：

- `r_balance_metrics.csv`
- `recommended_r_balance.json`
- `summary.json`
- `figures/pid_vs_fast_vs_steady_r_balance.svg`

需要特别说明的是：

- 在 `x_hold_disturbance_hold` 这种“参考基本不变化”的场景下，`r` 的影响本身就很弱
- 因为 `r` 主要作用于 TD，对参考变化速度进行整形
- 所以真正把 `r` 区分出来的，主要还是：
  - 大误差阶跃场景
  - 小阶跃稳态验证场景

当前可以先统一采用：

- `x / y` 默认固定 `r = 10`

并让后续 RL 先只学习：

- `b0`
- `wc`
- `k`

如果后续发现：

- `wc / k` 已经无法同时兼顾快速性和柔和性

再考虑把 `r` 放进 RL 范围中做第四维连续动作。
