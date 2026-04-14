# 第三章论文实验包说明

## 1. 文档目的

这份文档用于说明第三章正式论文实验包的结构、可信结果来源，以及论文中各图表与现有数据目录之间的对应关系。

第三章实验包由下面这个脚本统一生成：

`D:\ZhangC\lc\experiments\control\chapter3_thesis_suite.py`

生成目录固定为：

`D:\ZhangC\lc\outputs\chapter3_thesis_package\<timestamp>\`

这份实验包的目标不是重新发明实验，而是把当前已经验证可信的第三章 PyBullet 主链结果整理成：
- 可直接引用的表格
- 可直接插入论文的图
- 可追溯来源的 manifest 文件

## 2. 当前可信范围

当前只纳入已经确认可信的第三章 PyBullet 主链结果，核心代码链为：

- `D:\ZhangC\lc\src\lc\control\simulators\pybullet_runner.py`
- `D:\ZhangC\lc\src\lc\control\trainers\pybullet_axis_trainer.py`
- `D:\ZhangC\lc\src\lc\control\controllers\pybullet_variants.py`

以下历史阶段结果默认不纳入论文主结果：
- zero-delta fixedopt rewrite 之前的 RL 结果
- native LADRC 参数更新仍会重建 channel 时得到的结果
- checkpoint 未完整恢复 policy 内部状态时得到的 compare 结果
- 单 seed 随机扰动选模阶段、且已被多 seed 结果推翻的中间输出

## 3. 与论文章节的对应关系

### 3.1 本章控制架构与 LADRC 位置控制器设计

建议引用：
- `fig1_ladrc_framework.png`
- `fig5_ddpg_ladrc_framework.png`

建议配套表格：
- `rl_action_space_table.csv`
- `rl_training_config_table.csv`

本节建议重点说明：
- LADRC 仍然是底层位置控制器主体
- 强化学习负责在线调参，不替代控制器
- 当前动作空间统一为 `[r, b0, omega_c, k]`

### 3.2 传统整定与参数矛盾分析

建议引用：
- `fig2_step_or_short_response_compare.png`
- `fig3_disturbance_recovery_compare.png`
- `fig4_tracking_error_compare.png`

建议配套表格：
- `table_ladrc_single_speed_retune.csv`
- `table_ladrc_multispeed_best_params.csv`
- `table_baseline_no_disturbance.csv`

主要来源目录：
- `D:\ZhangC\lc\outputs\control_pybullet\x_ladrc_retune_short_speed_r_scan`
- `D:\ZhangC\lc\outputs\control_pybullet\x_multispeed_ladrc_retune_vs_pid`
- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_vs_ladrc_no_disturbance_compare`

说明：
- 当前“阶跃/短参考响应图”用已验证的短参考重整定实验近似支撑
- 如果后续论文必须严格展示标准阶跃工况，可以在此基础上单独补实验

### 3.3 强化学习联合参数重整定方法

建议引用：
- `fig5_ddpg_ladrc_framework.png`

建议配套表格：
- `rl_action_space_table.csv`
- `rl_training_config_table.csv`
- `shared_value_summary.csv`

主要来源目录：
- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_no_disturbance_mddpg_retrain\20260412_v1_to_v5_300eps_reexpanded`
- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_random_hover_disturbance_mddpg_retrain\20260412_random_hover_disturb_v1234_300eps_narrowed_noise0p2_0p004_posonly_multiseed_normfix`

当前建议结论：
- 在无扰动与 `0.004N` 随机悬停扰动两个主场景下，当前可信主线的推荐 shared value 都是 `v=2`

### 3.4 跨时间样本增强机制

建议引用：
- `fig6_temporal_enhancement_framework.png`

建议配套表格：
- `table_rl_shared_value_summary.csv`

说明重点：
- 当前这一节先用已有 shared value 结果支撑
- 强调 `stack_size`、`action_hold_steps`、`n_step` 与 shared value 的作用语义
- 当前不建议在正文中虚构一整套尚未补跑的大规模纯机制消融矩阵

### 3.5 实验设计与结果分析

建议引用：
- `fig7_training_reward_curve.png`
- `fig8_control_smoothness_compare.png`
- `fig9_overall_performance_compare.png`

建议配套表格：
- `table_rl_best_v2_no_disturbance.csv`
- `table_rl_best_v2_random_0p004.csv`
- `table_random_disturbance_scan_0p003_to_0p009.csv`
- `table_overall_metrics_summary.csv`

主要正式对比目录：
- 无扰动：
  `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_no_disturbance_mddpg_retrain\20260412_v1_to_v5_300eps_reexpanded\best_v_compare\bestv_compare_20260412_150029`
- `0.004N` 随机悬停扰动：
  `D:\ZhangC\lc\outputs\control_pybullet\x_pid_ladrc_ddpg_random_hover_disturb_compare\x\20260412_random_hover_disturb_0p004_bestv2_narrowed_noise0p2_posonly_multiseed_normfix_300eps_compare`

## 4. 图表意图说明

- `fig1`：用于交代 LADRC 本体结构和参数位置
- `fig2`：用于说明传统整定在短参考工况下的响应差异
- `fig3`：用于说明随机扰动恢复差异
- `fig4`：用于说明传统方法在误差层面的限制
- `fig5`：用于说明 DDPG-LADRC 在线重整定框架
- `fig6`：用于说明跨时间样本增强机制
- `fig7`：用于展示训练收敛过程
- `fig8`：用于展示控制输出平滑性，当前用 RPM 变化量作为平滑性代理指标
- `fig9`：用于做综合性能归一化比较

## 5. 推荐引用规则

当前第三章正式推荐引用的结果主线是：

1. 固定控制器与 LADRC 传统整定结果
2. 无扰动场景下的可信 RL 最优结果 `best v=2`
3. `0.004N` 随机悬停扰动场景下的可信 RL 最优结果 `best v=2`

除非后续有新的正式实验补充，否则不建议再回头引用更早的中间排错结果。

## 6. 实验包内部结构

实验包内固定包含：
- `3_2_traditional_tuning`
- `3_3_rl_ladrc_method`
- `3_4_temporal_enhancement`
- `3_5_experiments`
- `tables`
- `figures`
- `summaries`

其中后续写论文时最重要的索引文件是：
- `chapter3_experiment_manifest.json`
- `chapter3_metrics_manifest.json`
- `chapter3_figure_manifest.json`

如果后面要继续补图、补表或改编号，优先沿着这份说明和 manifest 做，而不是重新翻聊天记录找数据源。
