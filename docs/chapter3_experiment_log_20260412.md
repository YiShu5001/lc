# 第三章实验记录（2026-04-12）

## 1. 文档目的

这份文档记录 2026-04-12 当天围绕第三章控制层所做的主要仿真实验、排错过程、关键结论和最终保留结果。

它的定位不是简单罗列命令，而是帮助后续写论文、复现实验和回顾第三章细节时，快速回答下面几个问题：

1. 今天到底验证了哪些控制场景
2. 哪些结果是可信的，哪些结果后来被推翻
3. RL-LADRC 为什么一度出现“训练期很好、回放很差”的现象
4. 当前第三章应该引用哪些结果，应该沿着哪条主线继续做

## 2. 今日实验主线概览

今天的第三章工作可以概括为四条主线：

1. 重新确认固定控制器基线  
   重点是确认 `PID` 和固定参数 `LADRC` 在无扰动 PyBullet 环境下的真实表现，并排除“环境本身坏掉”的可能。

2. 重新验证 RL-LADRC 的训练链是否可信  
   重点是确认第三章的 RL 训练、deterministic eval 和正式 compare 是否真的在同一条 PyBullet rollout 链上。

3. 追查“训练指标很好，正式回放很差”的根因  
   这一步最终定位到：问题不只是 reward，也不是单纯模型没学会，而是 checkpoint 恢复状态不完整，尤其是 `_normalizer` 没有被一致地保存和恢复。

4. 在修复后的主链上，重新做随机悬停扰动训练和正式 compare  
   最终把结果收敛到一个可信结论：在 `0.004N` 的随机悬停扰动下，修复后的 `DDPG-LADRC(v=2)` 已经可以稳定地回到接近甚至略优于固定 `PID/LADRC` 的水平。

## 3. 固定控制器基线确认

### 3.1 无外扰基线

在无外扰 PyBullet 环境中，使用四段参考路线：

- `+0.5 m/s` 前进 `2 s`
- 悬停 `1 s`
- `-0.6 m/s` 反向 `2 s`
- 末尾静止 `0.5 s`

结果目录：

- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_vs_ladrc_no_disturbance_compare\x\20260412_105149`

关键指标如下：

| 控制器 | RMSE | MAE | velocity_RMSE |
|---|---:|---:|---:|
| PID | 0.05817 | 0.03897 | 0.24971 |
| LADRC(0.5-opt) | 0.05843 | 0.05104 | 0.21482 |
| LADRC(0.6-opt) | 0.06650 | 0.05103 | 0.30465 |
| LADRC(switched) | 1.29688 | 1.18059 | 0.91544 |

这里有两个重要结论：

1. `PID` 与 `LADRC(0.5-opt)` 在无外扰条件下几乎处于同一水平，这说明固定控制器侧本身是正常的。
2. “按参考段硬切换 LADRC 参数”的方案明显失败，因此后续第三章主线不再把它当成可信方法分支。

## 4. RL-LADRC 无外扰验证

### 4.1 小范围动作验证主链是否可靠

为了先确认环境和 RL 主链是否可靠，我们曾在无外扰条件下，将动作范围限制在 `0.5-opt` 参数周围一个很小的邻域内。

结果目录：

- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_no_disturbance_mddpg_retrain\20260412_v1_small_range_reliable_check`

这次训练的动作范围是：

- `r: 62.5 ~ 63.5`
- `b0: 23.5 ~ 25.5`
- `omega_c: 2.75 ~ 3.15`
- `k: 6.8 ~ 7.9`

训练期最优结果：

- `best_eval_rmse = 0.05818`
- `best_eval_mae = 0.05131`
- `best_eval_velocity_rmse = 0.20547`

这一步的意义非常大，因为它说明：

- 在环境正确、动作范围够小、参数中心合理时，`DDPG-LADRC(v=1)` 是能完整走完轨迹的；
- 也就是说，第三章 RL 主链不是天然不工作，而是后续更大范围训练时出现了链路或选模问题。

### 4.2 无外扰扩展训练

随后在无外扰环境下，我们把 shared value 扩展到 `v=1..5` 并训练 `300` episodes。

关键正式结果目录：

- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_no_disturbance_mddpg_retrain\20260412_v1_to_v5_300eps_reexpanded\best_v_compare\bestv_compare_20260412_150029`

最优 shared value 为 `v=2`，正式回放指标为：

| 控制器 | RMSE | MAE | velocity_RMSE |
|---|---:|---:|---:|
| PID | 0.05817 | 0.03897 | 0.24971 |
| LADRC(0.5-opt) | 0.05843 | 0.05104 | 0.21482 |
| DDPG-LADRC(best v=2) | 0.05732 | 0.04923 | 0.22568 |

结论：

- `DDPG-LADRC(v=2)` 在无外扰条件下已经能够达到接近固定 `PID/LADRC` 的水平；
- 在 `RMSE` 上甚至略优于 `PID` 和固定 `LADRC`；
- 但在 `MAE` 上仍未全面超过 `PID`。

## 5. 随机悬停扰动扫描

### 5.1 扰动设计的调整

为了让第三章更符合抗扰研究目标，后续实验把随机扰动加入到了中间悬停段。最终采用的扰动窗口是：

- 悬停前留白约 `0.2 s`
- 扰动持续约 `0.6 s`
- 悬停后再留白约 `0.2 s`

离散实现为：

- 前留白 `10` steps
- 扰动 `28` steps
- 后留白 `10` steps

### 5.2 扰动幅值扫描结果

在这个随机悬停扰动设置下，我们先扫描了 `0.003N ~ 0.009N`。

结果目录组：

- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_ladrc_ddpg_random_hover_disturb_compare\x\hover_gap_rand_0p003`
- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_ladrc_ddpg_random_hover_disturb_compare\x\hover_gap_rand_0p004`
- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_ladrc_ddpg_random_hover_disturb_compare\x\hover_gap_rand_0p005`
- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_ladrc_ddpg_random_hover_disturb_compare\x\hover_gap_rand_0p006`
- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_ladrc_ddpg_random_hover_disturb_compare\x\hover_gap_rand_0p007`
- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_ladrc_ddpg_random_hover_disturb_compare\x\hover_gap_rand_0p008`
- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_ladrc_ddpg_random_hover_disturb_compare\x\hover_gap_rand_0p009`

关键 RMSE 如下：

| 扰动幅值 | PID | LADRC(0.5-opt) | DDPG-LADRC(best v=2) |
|---|---:|---:|---:|
| 0.003N | 0.05756 | 0.05883 | 0.05792 |
| 0.004N | 0.06207 | 0.05801 | 0.05737 |
| 0.005N | 0.05976 | 0.65649 | 0.05570 |
| 0.006N | 0.06149 | 0.05895 | 0.05632 |
| 0.007N | 0.05757 | 0.05772 | 0.05926 |
| 0.008N | 0.05993 | 0.05823 | 0.05870 |
| 0.009N | 0.06009 | 0.34175 | 0.11831 |

这组扫描带来的判断是：

1. `0.001N ~ 0.002N` 基本已经接近无扰动口径，区分度不足。
2. `0.005N` 及以上时，固定 `LADRC(0.5-opt)` 会开始出现明显的不稳定性。
3. `0.004N` 是一个比较合适的训练与验证扰动强度：
   - 已经能看出扰动影响；
   - 但又没有把系统推到过于极端的失稳区间。

因此，后续训练主线最终锁定在 `0.004N`。

## 6. RL-LADRC 训练链排错过程

### 6.1 第一类问题：环境链路不一致

在今天的前半段，RL 训练和正式 compare 曾出现明显不一致：

- 训练期 deterministic eval 看起来不错；
- 但正式回放时模型像是“换了一个”，表现明显变差。

后来确认有一段时期，训练、deterministic eval 和正式 compare 没有严格走同一条 rollout 逻辑，这会直接导致第三章实验无法解释。

这一步最终通过统一 `run_policy_episode()` 主链得到修复。

### 6.2 第二类问题：native LADRC 参数更新导致 channel 重建

随后又发现，RL 每步调参时，如果采用“重建 channel”的方式更新 native LADRC 参数，会导致：

- TD/ESO 内部状态被清空；
- 控制器看起来像“每一步都失忆”；
- 即使动作很小，轨迹也会异常。

这一步最终改成“原位更新参数，不重建 channel”，并通过 zero-delta 实验完成验证。

### 6.3 第三类问题：checkpoint 恢复状态不完整

这是今天最关键、也是最容易误判的一类问题。

表现是：

- 训练期间最优轨迹图看起来很好；
- 但是从磁盘重新加载 `x_policy_best.pt` 后，正式 compare 明显恶化。

后续定位到真正的问题不是“文件路径选错”，而是：

- checkpoint 只保存了 actor/critic 权重；
- 但没有完整保存并恢复策略内部状态，尤其是 `_normalizer`；
- 导致训练期 policy 和正式 compare 重新加载的 policy 虽然权重相同，但输入尺度已经不同；
- 于是同一个模型看起来像“换了个脑子”。

后来补齐保存和恢复的状态包括：

- `_normalizer`
- `_last_action`
- `_hold_counter`
- `_current_expl_noise`

同时正式 compare 改成显式：

- `torch.load(..., weights_only=False)`

这一修复完成后，训练期最优和正式回放终于重新对上。

## 7. 随机扰动训练的可信主结果

### 7.1 修复后 500 episode 的 `v=2` 结果

修复后的关键训练目录：

- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_random_hover_disturbance_mddpg_retrain\20260412_random_hover_disturb_v2_500eps_narrowed_noise0p2_0p004_posonly_multiseed_normfix`

配置为：

- 扰动：`0.004N`
- reward：`-|pos_error|`
- 多 seed 评估：`7, 17, 27`
- shared value：`v=2`
- 参数范围：
  - `r: 58~68`
  - `b0: 20~30`
  - `omega_c: 2.4~3.5`
  - `k: 5.8~9.2`

训练期 best eval 指标：

- `RMSE = 0.05366`
- `MAE = 0.04661`
- `velocity_RMSE = 0.21172`

正式 compare 目录：

- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_ladrc_ddpg_random_hover_disturb_compare\x\20260412_random_hover_disturb_0p004_bestv2_narrowed_noise0p2_posonly_multiseed_normfix_compare`

正式回放指标：

| 控制器 | RMSE | MAE | velocity_RMSE |
|---|---:|---:|---:|
| PID | 0.06207 | 0.04197 | 0.25546 |
| LADRC(0.5-opt) | 0.05801 | 0.05060 | 0.21721 |
| DDPG-LADRC(v=2) | 0.05714 | 0.04953 | 0.22158 |

这个结果非常关键，因为它说明：

- 在修复后的环境与 checkpoint 链路上，训练期最优与正式回放已经重新一致；
- `DDPG-LADRC(v=2)` 不再“训练期有效、回放失效”；
- 在 `0.004N` 扰动下，RL 已经能达到接近甚至略优于固定基线的水平。

### 7.2 修复后 300 episode 的 `v=1,2,3,4` 对比

为了进一步降低训练成本，我们随后又在相同设置下重训了 `v=1,2,3,4`，每个 `300` episodes。

训练总目录：

- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_random_hover_disturbance_mddpg_retrain\20260412_random_hover_disturb_v1234_300eps_narrowed_noise0p2_0p004_posonly_multiseed_normfix`

best eval 指标如下：

| v | RMSE | MAE | velocity_RMSE | score |
|---|---:|---:|---:|---:|
| 1 | 0.06571 | 0.05867 | 0.21831 | 0.12990 |
| 2 | 0.05400 | 0.04704 | 0.21097 | 0.11266 |
| 3 | 0.05404 | 0.04695 | 0.21921 | 0.11431 |
| 4 | 0.06062 | 0.05350 | 0.20863 | 0.12107 |

最优仍然是 `v=2`。

正式 compare 目录：

- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_ladrc_ddpg_random_hover_disturb_compare\x\20260412_random_hover_disturb_0p004_bestv2_narrowed_noise0p2_posonly_multiseed_normfix_300eps_compare`

正式回放指标：

| 控制器 | RMSE | MAE | velocity_RMSE |
|---|---:|---:|---:|
| PID | 0.06207 | 0.04197 | 0.25546 |
| LADRC(0.5-opt) | 0.05801 | 0.05060 | 0.21721 |
| DDPG-LADRC(v=2, 300eps) | 0.05406 | 0.04674 | 0.23341 |

结论：

- `v=2` 在 300 episodes 下仍然保持最优；
- `RMSE` 已经明显优于 `PID` 和固定 `LADRC`；
- `MAE` 也优于固定 `LADRC`，但仍未超过 `PID`；
- 因此当前最准确的表述是：
  - `DDPG-LADRC(v=2)` 已经在随机扰动恢复任务中展现出明显竞争力；
  - 但还不能简单表述为“全面优于 PID”。

## 8. 当前可信结论

截至今天实验结束，可以确认以下几点。

### 8.1 关于环境与训练链

- 第三章当前的 PyBullet 训练、deterministic eval 和正式 compare 已经统一到同一条 rollout 主链。
- 训练期和正式回放之前的巨大落差，主要不是“环境不同”，而是 checkpoint 恢复状态不完整。
- 在补齐策略内部状态保存与恢复之后，这个问题已经被修复。

### 8.2 关于固定控制器

- `PID` 和固定 `LADRC(0.5-opt)` 在无扰动环境下均表现正常；
- `LADRC(0.5-opt)` 在随机悬停扰动下整体上比 `PID` 更有恢复潜力，但在某些较大扰动幅值下也会出现明显不稳定；
- 固定参数 `LADRC` 对扰动场景的鲁棒性并不总是优于 `PID`，这也是第三章引入在线调参的意义所在。

### 8.3 关于 RL-LADRC

- 在修复后的主链上，`v=2` 是当前最稳定、最有代表性的 shared value；
- `v=2` 在无扰动和 `0.004N` 随机扰动下都已经能够达到很接近固定基线的水平；
- 在 `RMSE` 上，`DDPG-LADRC(v=2)` 已经可以超过固定 `PID`；
- 但在 `MAE` 上仍未全面超过 `PID`，因此第三章目前更适合表述为：
  - RL-LADRC 已经实现了稳定可用；
  - 并在部分指标上展现出优于传统固定控制器的潜力。

## 9. 当前推荐引用结果

如果现在需要在论文或汇报中引用今天的结果，建议优先使用以下目录。

固定控制器基线：

- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_vs_ladrc_no_disturbance_compare\x\20260412_105149`

无扰动 RL 可信结果：

- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_no_disturbance_mddpg_retrain\20260412_v1_to_v5_300eps_reexpanded`

随机扰动 RL 可信结果：

- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_random_hover_disturbance_mddpg_retrain\20260412_random_hover_disturb_v2_500eps_narrowed_noise0p2_0p004_posonly_multiseed_normfix`
- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_random_hover_disturbance_mddpg_retrain\20260412_random_hover_disturb_v1234_300eps_narrowed_noise0p2_0p004_posonly_multiseed_normfix`
- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_ladrc_ddpg_random_hover_disturb_compare\x\20260412_random_hover_disturb_0p004_bestv2_narrowed_noise0p2_posonly_multiseed_normfix_compare`
- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_ladrc_ddpg_random_hover_disturb_compare\x\20260412_random_hover_disturb_0p004_bestv2_narrowed_noise0p2_posonly_multiseed_normfix_300eps_compare`

## 10. 下一步建议

从今天的结果看，后续第三章最值得继续推进的方向有两个。

1. reward 从纯位置误差改成位置误差加一个较小速度项  
   例如 `-|pos_error| - 0.1*|vel_error|`，因为今天的结果已经表明“纯位置误差”虽然能学出不错的 RMSE，但在 `MAE` 和恢复平衡上仍有继续改进空间。

2. 继续围绕 `v=2` 做细化，而不是再盲目扩大 `v` 搜索范围  
   当前所有可信结果都在说明，`v=2` 是第三章目前最值得继续深挖的 shared value。

一句话总结今天的实验：

今天不是简单“又跑了一堆结果”，而是把第三章 RL-LADRC 从“结果看起来时好时坏、解释不通”推进到了“链路可信、结果可复核、结论可写进论文”的状态。
