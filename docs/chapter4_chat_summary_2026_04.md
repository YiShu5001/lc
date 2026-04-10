# 第四章与相关工作对话总结（2026-04）

本文档用于汇总本轮围绕第三章控制约束、第四章课程环境、第四章神经网络重构、实验参数、测试与汇报材料的连续讨论结果。

## 1. 第三章控制层相关结论

### 1.1 PyBullet 频率口径

已确认第三章 PyBullet 控制链路的时间尺度为：

- `simulation_freq_hz = 240`
- `control_freq_hz = 48`
- `rl_freq_hz = 8`

结论：

- 第四章如果要和第三章“高层 RL 交互节奏”对齐，应优先参考 `8 Hz`
- 第四章当前规划环境仍使用 `step_dt = 0.1`，即 `10 Hz`

### 1.2 第三章速度参数的语义澄清

已明确第三章 `primary_speed_range / reverse_speed_range` 的含义：

- 它们是 **单轴控制任务中的参考速度设定**
- 它们用于生成目标参考速度轨迹
- 它们 **不是** Crazyflie 的物理飞行上限
- 它们 **不是** 当前无人机状态速度的硬限制

这一点后续在第四章设计中不再混用。

### 1.3 第三章速度修改已撤回

中途曾将第三章 `x/y` 参考速度改为 `±1 m/s`，后续已按要求恢复到原设定，不作为最终口径。

## 2. 第四章环境与课程学习相关结论

### 2.1 第四章动作语义

已统一为：

- 第四章 actor 输出的是高层速度命令
- 语义为 `vx, vy`
- 动作边界固定为 `[-0.8, 0.8] m/s`

### 2.2 课程环境需要考虑的核心约束

已整理出第四章课程环境应重点确认的内容：

- `step_dt / horizon / episode_duration`
- 动作边界 `[-0.8, 0.8]`
- 是否增加 `delta_v_max`
- `workspace_limit`
- 障碍半径与邻机安全距离
- success 判据
- 课程阶段目标速度尺度

### 2.3 参数表文档已形成

已写成完整参数表：

- [chapter4_experiment_parameter_tables.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter4_experiment_parameter_tables.md)

内容包括：

- 仿真环境设计与强化学习接口
- 奖励函数设定
- 课程学习三阶段九情景
- 神经网络参数
- 经验池参数
- 实验前核对清单

## 3. 第四章神经网络设计相关结论

### 3.1 两套网络方案

已明确分为两套：

- `MLP` 基线
- `Transformer` 主模型

### 3.2 mask 方案

已确认采用：

- “零填充 + 显式 mask”

具体做法：

- `MLP`
  - 输入展平后附加 `obstacle_mask / neighbor_mask`
- `Transformer`
  - token 仍零填充
  - attention 使用 `key_padding_mask`

不采用“仅靠 0 输入自动充当 mask”的口径。

### 3.3 actor / critic 分工

已确认：

- actor 保留“先避障，后协同”的两阶段语义
- 协同只在最后一层残差修正体现
- critic 不做全体无人机动作理解
- critic 不保留独立邻机协同理解分支
- 邻机在 critic 中按“局部动态风险体”并入 obstacle-like token

### 3.4 大模型版参数口径

最终按用户要求上调为大模型版本：

- `MLP` 主隐藏层 `>= 800`
- `Transformer embed_dim >= 800`
- `Transformer ff_dim >= 800`

当前默认口径为：

- `mlp_large`
- `transformer_large`

## 4. 已落地的代码重构

### 4.1 配置分离

已拆出网络配置层：

- [planning_config.py](/C:/context_mine/mine_code/GIT_Projects/lc/src/lc/planning/configs/planning_config.py)

新增：

- `PlanningNetworkConfig`
- `MLPBaselineConfig`
- `TransformerActorConfig`
- `LocalRiskCriticConfig`
- `build_planning_network_config()`

### 4.2 环境输入扩展

已在：

- [swarm.py](/C:/context_mine/mine_code/GIT_Projects/lc/src/lc/planning/envs/swarm.py)

完成：

- `self_state` 扩为 6 维
- `obstacles` 扩为 4 维 token
- 新增 `obstacle_mask`
- 新增 `neighbor_mask`

### 4.3 MLP 基线

已在：

- [mlp_baseline.py](/C:/context_mine/mine_code/GIT_Projects/lc/src/lc/planning/models/mlp_baseline.py)

完成：

- 大宽度 MLP actor
- 结构化观测展平函数
- mask 特征拼接

### 4.4 Transformer 主模型

已在：

- [multi_uav_model.py](/C:/context_mine/mine_code/GIT_Projects/lc/src/lc/planning/models/multi_uav_model.py)

完成：

- 显式 mask 的 attention
- 两阶段 actor
- `no_collab` 和 `no_mask` 消融配置支持

### 4.5 Critic 重构

已新增：

- [local_risk_critic.py](/C:/context_mine/mine_code/GIT_Projects/lc/src/lc/planning/critics/local_risk_critic.py)

并保留：

- [structured_critic.py](/C:/context_mine/mine_code/GIT_Projects/lc/src/lc/planning/critics/structured_critic.py)

当前支持：

- `local_risk` critic
- `mlp` critic
- `structured_legacy` critic

### 4.6 trainer 和 compare

已在：

- [planning_trainer.py](/C:/context_mine/mine_code/GIT_Projects/lc/src/lc/planning/trainers/planning_trainer.py)
- [compare.py](/C:/context_mine/mine_code/GIT_Projects/lc/src/lc/planning/experiments/compare.py)

完成：

- 按 `network_version` 装配 actor/critic
- 记录 `network_version / actor_type / critic_type / actor_param_count / critic_param_count / config_snapshot`
- 保留旧方法名兼容
- 加入新网络版本组

## 5. 对照试验与消融分组

当前推荐主组：

- `mlp_large`
- `transformer_large`

当前推荐结构消融：

- `transformer_large_no_collab`
- `transformer_large_no_mask`
- `transformer_large_mlp_critic`
- `transformer_large_old_critic`

当前推荐容量消融：

- `mlp_800`
- `transformer_800`
- `transformer_896`

相关说明文档：

- [chapter4_network_design.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter4_network_design.md)
- [chapter4_network_versions.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter4_network_versions.md)
- [chapter4_ablation_visualization_guide.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter4_ablation_visualization_guide.md)

## 6. 测试与回归

### 6.1 已补的专项测试

- [test_planning_network_configs.py](/C:/context_mine/mine_code/GIT_Projects/lc/tests/test_planning_network_configs.py)
- [test_planning_network_execution.py](/C:/context_mine/mine_code/GIT_Projects/lc/tests/test_planning_network_execution.py)
- [test_planning_compare_outputs.py](/C:/context_mine/mine_code/GIT_Projects/lc/tests/test_planning_compare_outputs.py)

### 6.2 已验证通过的测试

已通过：

- `python -m pytest tests\test_planning_network_configs.py -q`
- `python -m pytest tests\test_planning_network_execution.py -q`
- `python -m pytest tests\test_planning_compare_outputs.py -q`

### 6.3 回归文档与脚本

已补：

- [chapter4_network_regression_tests.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter4_network_regression_tests.md)
- [run_chapter4_network_regression.py](/C:/context_mine/mine_code/GIT_Projects/lc/scripts/run_chapter4_network_regression.py)

脚本已实际跑通。

## 7. 已形成的汇报/整理类文档

本轮对话过程中已形成或补充的核心文档包括：

- [chapter4_group_meeting_report.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter4_group_meeting_report.md)
- [chapter4_experiment_parameter_tables.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter4_experiment_parameter_tables.md)
- [chapter4_network_design.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter4_network_design.md)
- [chapter4_network_versions.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter4_network_versions.md)
- [chapter4_network_regression_tests.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter4_network_regression_tests.md)
- [chapter4_ablation_visualization_guide.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter4_ablation_visualization_guide.md)

## 8. 当前状态总结

到目前为止，第四章这条线已经从“单一结构化 actor + 旧式 critic”推进到：

- 有配置分离
- 有网络版本管理
- 有 MLP / Transformer 双方案
- 有显式 mask
- 有轻量 local-risk critic
- 有对照试验和消融试验矩阵
- 有专项回归测试和回归脚本

仍需后续继续关注的点：

- 大模型口径下训练成本和稳定性
- 是否进一步收缩 compare 的正式汇报矩阵
- 是否把第四章参数总表中的神经网络部分同步到最新实现口径
