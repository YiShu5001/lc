# 第三章仿真代码与实验结果归档说明

## 1. 文档目的

这份文档用于整理第三章当前的仿真代码主链与实验结果目录，先给出逻辑归档说明，后续再按本文档执行物理归档。

本轮只做说明，不移动、不删除任何代码或结果目录。

本文档要回答四个问题：

1. 当前第三章可信代码主链是什么
2. 哪些实验脚本是当前主入口，哪些只是历史试验或排错入口
3. 哪些 `outputs` 结果可以继续作为论文或汇报主结果引用
4. 哪些代码和结果应归档，且不应继续作为第三章主结论依据

## 2. 可信主链结论

### 2.1 当前第三章可信代码主链

当前第三章控制层的可信主链，以 `src/lc/control` 下以下模块为准：

- `D:\ZhangC\lc\src\lc\control\configs`
- `D:\ZhangC\lc\src\lc\control\controllers`
- `D:\ZhangC\lc\src\lc\control\envs`
- `D:\ZhangC\lc\src\lc\control\policies`
- `D:\ZhangC\lc\src\lc\control\simulators`
- `D:\ZhangC\lc\src\lc\control\trainers`

这些模块构成了当前第三章 PyBullet 仿真闭环的核心实现。

### 2.2 当前第三章可信核心文件

下列文件属于当前可信 PyBullet 主链的核心：

- `D:\ZhangC\lc\src\lc\control\simulators\pybullet_runner.py`
- `D:\ZhangC\lc\src\lc\control\trainers\pybullet_axis_trainer.py`
- `D:\ZhangC\lc\src\lc\control\controllers\pybullet_variants.py`
- `D:\ZhangC\lc\src\lc\control\policies\mddpg_control.py`
- `D:\ZhangC\lc\src\lc\control\configs\control_config.py`
- `D:\ZhangC\lc\src\lc\control\configs\pybullet_control_config.py`
- `D:\ZhangC\lc\src\lc\control\envs\pybullet_axis_env.py`

这些文件当前承担的职责分别是：

- `pybullet_runner.py`：统一训练、评估、正式对比的 PyBullet rollout 主链
- `pybullet_axis_trainer.py`：RL 训练主调度与 checkpoint 保存
- `pybullet_variants.py`：PyBullet 控制器创建、参数写入与原位更新
- `mddpg_control.py`：第三章 RL-LADRC 策略接口
- `control_config.py`：LADRC 动作边界与 anchor 配置
- `pybullet_control_config.py`：PyBullet 实验配置与评估 seed 等运行参数
- `pybullet_axis_env.py`：第三章单轴环境接口与奖励侧定义

### 2.3 当前不作为主链的代码

下列内容属于历史链路、参考资产或过渡实现，不作为当前第三章 PyBullet 主训练链：

- `D:\ZhangC\lc\src\lc\control\RLcontrolRefLine`

结论：

- 这个目录可以保留为历史参考
- 但不应再作为第三章当前仿真训练与正式对比的主实现入口

## 3. 代码脚本分层

### 3.1 当前建议保留的主入口脚本

这些脚本仍可作为第三章当前主入口或可信实验入口继续使用。

固定控制器/LADRC 对比与整定：

- `D:\ZhangC\lc\experiments\control\run_pybullet_x_pid_vs_ladrc_refline_compare.py`
- `D:\ZhangC\lc\experiments\control\run_pybullet_x_multispeed_ladrc_retune_vs_pid.py`
- `D:\ZhangC\lc\experiments\control\run_pybullet_x_pid_vs_ladrc_no_disturbance_compare.py`
- `D:\ZhangC\lc\experiments\control\run_pybullet_x_pid_ladrc_ddpg_random_hover_disturb_compare.py`

RL 训练主入口：

- `D:\ZhangC\lc\experiments\control\run_pybullet_x_refline_no_disturbance_mddpg_retrain.py`
- `D:\ZhangC\lc\experiments\control\run_pybullet_x_refline_random_hover_disturbance_mddpg_retrain.py`
- `D:\ZhangC\lc\experiments\control\run_pybullet_x_refline_random_hover_disturbance_mddpg_retrain_v136_reexpanded.py`

### 3.2 可保留但不建议再作为主入口的脚本

这些脚本可以继续保留参考价值，但不建议继续作为论文主结果入口：

- `D:\ZhangC\lc\experiments\control\run_pybullet_x_speed_sweep_pid_vs_ladrc_r63_compare.py`
- `D:\ZhangC\lc\experiments\control\run_pybullet_x_speed_sweep_pid_vs_ladrc_no_td_compare.py`
- `D:\ZhangC\lc\experiments\control\run_pybullet_x_no_td_retune_short_speed.py`
- `D:\ZhangC\lc\experiments\control\run_pybullet_highspeed_midhold_sine_disturbance.py`
- `D:\ZhangC\lc\experiments\control\chapter3_rl_refline_suite.py`

这些脚本多数用于：

- 某一阶段的整定
- 某种局部假设验证
- 与当前主链不同的旧实验组织方式

### 3.3 待归档脚本

这些脚本主要属于历史排错入口、过渡 compare 脚本或已被后续主链替代的脚本，后续建议归档：

- `D:\ZhangC\lc\experiments\control\run_pybullet_x_pid_ladrc_ddpg_mddpg_compare.py`
- `D:\ZhangC\lc\experiments\control\run_pybullet_x_pid_ladrc_ddpg_mddpg_no_disturbance_compare.py`
- `D:\ZhangC\lc\experiments\control\run_pybullet_x_rl_triplet_random_hover_compare.py`
- `D:\ZhangC\lc\experiments\control\run_pybullet_x_refline_sine_disturbance_mddpg_v_sweep.py`

原因：

- 与当前修复后的主链不一致
- 或者仅服务于中间阶段排错
- 或者已被更稳定的正式入口替代

## 4. 实验结果可信分界线

### 4.1 当前可信分界线

第三章 RL 仿真结果的可信分界线如下：

1. `zero-delta fixedopt rewrite` 之前的 RL 结果，默认不可信或仅作排错记录
2. native LADRC 参数更新从“重建 channel”修到“原位更新”之后，结果可信度提升
3. checkpoint 完整保存/恢复策略内部状态之后，RL 训练与正式回放才重新对齐

这里的关键修复包括：

- 环境 rollout 统一到同一条 PyBullet 主链
- native LADRC 参数更新不再每步重建 channel
- checkpoint 完整保存和恢复：
  - `_normalizer`
  - `_last_action`
  - `_hold_counter`
  - `_current_expl_noise`
- 正式 compare 使用显式完整 checkpoint 加载

### 4.2 历史结果失效原因索引

历史结果失效原因可归并为以下几类：

1. 环境链路不一致
2. native LADRC 参数更新会重建 channel
3. checkpoint 未完整保存/恢复策略内部状态
4. 单 seed 随机扰动选模不稳
5. reward/动作范围仍在探索期，仅作方法调试

只要历史结果属于上述任一类，并且被后续修复后结果推翻，就不应再引用为第三章主结论。

## 5. 实验结果白名单

### 5.1 当前主保留白名单

以下结果目录可作为当前第三章主结果继续保留和引用。

固定控制器/LADRC 标定主结果：

- `D:\ZhangC\lc\outputs\control_pybullet\x_ladrc_retune_short_speed_r_scan`
- `D:\ZhangC\lc\outputs\control_pybullet\x_multispeed_ladrc_retune_vs_pid`
- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_vs_ladrc_no_disturbance_compare`

无扰动 RL 主链可信结果：

- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_no_disturbance_mddpg_retrain\20260412_v1_zero_delta_fixedopt_rewrite`
- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_no_disturbance_mddpg_retrain\20260412_v1_to_v5_300eps_reexpanded`

随机扰动 RL 主链可信结果：

- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_random_hover_disturbance_mddpg_retrain\20260412_random_hover_disturb_v2_500eps_narrowed_noise0p2_0p004_posonly_multiseed_normfix`
- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_random_hover_disturbance_mddpg_retrain\20260412_random_hover_disturb_v1234_300eps_narrowed_noise0p2_0p004_posonly_multiseed_normfix`
- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_ladrc_ddpg_random_hover_disturb_compare\x\20260412_random_hover_disturb_0p004_bestv2_narrowed_noise0p2_posonly_multiseed_normfix_compare`
- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_ladrc_ddpg_random_hover_disturb_compare\x\20260412_random_hover_disturb_0p004_bestv2_narrowed_noise0p2_posonly_multiseed_normfix_300eps_compare`

### 5.2 可参考但不作为主结论

以下结果可以保留参考价值，但不应作为第三章当前论文主结论：

- `D:\ZhangC\lc\outputs\control_pybullet\x_speed_sweep_pid_vs_ladrc_r63_compare`
- `D:\ZhangC\lc\outputs\control_pybullet\x_speed_sweep_pid_vs_ladrc_no_td_compare`
- `D:\ZhangC\lc\outputs\control_pybullet\x_speed_sweep_td_vs_no_td_summary`
- `D:\ZhangC\lc\outputs\control_pybullet\x_no_td_retune_short_speed`
- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_vs_ladrc_refline_compare`

这些结果主要用于：

- 早期整定方向判断
- no-TD 对比
- 中间阶段分析

## 6. 待归档结果清单

以下目录后续建议归档，原因是它们要么属于环境修复前结果，要么属于被后续 normfix/multiseed 主链推翻的中间试验结果。

### 6.1 待归档 RL 结果目录

- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_sine_disturbance_mddpg_v_sweep`
- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_pid_ladrc_ddpg_mddpg_compare`
- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_pid_ladrc_ddpg_mddpg_no_disturbance_compare`
- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_env_consistency_check_v1`
- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_rl_triplet_random_hover_compare`
- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_random_hover_disturbance_mddpg_retrain\20260412_random_hover_disturb_v13579_500eps`
- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_random_hover_disturbance_mddpg_retrain\20260412_random_hover_disturb_v136_500eps_reexpanded`
- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_random_hover_disturbance_mddpg_retrain\20260412_random_hover_disturb_v136_500eps_reexpanded_noise0p2`
- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_random_hover_disturbance_mddpg_retrain\20260412_random_hover_disturb_v6_500eps_narrowed_noise0p2`
- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_random_hover_disturbance_mddpg_retrain\20260412_random_hover_disturb_v6_500eps_narrowed_noise0p2_0p004_posonly`
- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_no_disturbance_mddpg_retrain\20260412_no_disturbance_locked_bounds_64_66`
- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_no_disturbance_mddpg_retrain\20260412_no_disturbance_narrow_bounds_full`
- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_no_disturbance_mddpg_retrain\20260412_no_disturbance_posvel_reward_only`
- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_no_disturbance_mddpg_retrain\20260412_no_disturbance_v1_v5_full`
- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_no_disturbance_mddpg_retrain\20260412_v1_retrain_after_envfix`
- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_no_disturbance_mddpg_retrain\20260412_v1_zero_delta`
- `D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_no_disturbance_mddpg_retrain\20260412_v1_zero_delta_fixedopt`

### 6.2 待归档 compare 结果目录

- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_ladrc_ddpg_random_hover_disturb_compare\x\20260412_random_hover_disturb_bestv3_compare`
- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_ladrc_ddpg_random_hover_disturb_compare\x\20260412_random_hover_disturb_bestv6_compare`
- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_ladrc_ddpg_random_hover_disturb_compare\x\20260412_random_hover_disturb_bestv6_compare_noise0p2`
- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_ladrc_ddpg_random_hover_disturb_compare\x\20260412_random_hover_disturb_bestv6_narrowed_noise0p2_compare`
- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_ladrc_ddpg_random_hover_disturb_compare\x\20260412_random_hover_disturb_0p004_bestv6_narrowed_noise0p2_compare`
- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_ladrc_ddpg_random_hover_disturb_compare\x\20260412_random_hover_disturb_0p004_bestv6_narrowed_noise0p2_posonly_compare`
- `D:\ZhangC\lc\outputs\control_pybullet\x_pid_ladrc_ddpg_random_hover_disturb_compare\x\20260412_random_hover_disturb_0p004_bestv2_narrowed_noise0p2_posonly_multiseed_compare`

原因：

- 被后续 `normfix` 和多 seed compare 主链替代
- 或属于 reward/探索率/范围排错过程目录

## 7. 当前使用建议

如果现在继续推进第三章工作，默认应遵守以下规则：

1. 代码实现默认从 `src/lc/control` 当前主链继续，不回到历史链路
2. RL 训练默认从：
   - `run_pybullet_x_refline_random_hover_disturbance_mddpg_retrain_v136_reexpanded.py`
   - 或 `run_pybullet_x_refline_no_disturbance_mddpg_retrain.py`
   继续推进
3. 正式 compare 默认使用：
   - `run_pybullet_x_pid_ladrc_ddpg_random_hover_disturb_compare.py`
4. 论文或汇报引用结果时，只从本文档白名单目录中选
5. 若新增实验结果，应先判断其是否跨过：
   - 环境统一
   - zero-delta rewrite
   - checkpoint state 完整恢复
   - 多 seed 评估
   这四个可信门槛

## 8. 后续物理归档执行原则

后续如果执行物理归档，应按以下顺序进行：

1. 先保留本文档中的代码主链白名单
2. 再保留本文档中的实验结果白名单
3. 将“待归档脚本”移动到专门 archive 目录
4. 将“待归档结果目录”移动到专门 archive 目录
5. 不按目录时间先后粗暴删除

一句话原则：

后续所有第三章代码与结果归档，都以本文档为准，而不是以目录新旧或文件多少为准。
