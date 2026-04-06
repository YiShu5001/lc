# 第四章失效排查日志模板

## 1. 实验基本信息

- 日期：
- 实验人：
- 目标：
- 备注：

## 2. 运行配置

### 2.1 训练配置

- `episodes`:
- `eval_episodes`:
- `seed`:
- `actor_variant`:
- `use_curriculum`:
- `use_pyramid_per`:
- `use_uniform_replay`:

### 2.2 课程配置

- `window_size`:
- `decision_window`:
- `stable_windows_required`:
- `rollback_windows_required`:
- `reward_thresholds`:
- `success_thresholds`:
- `std_thresholds`:

### 2.3 回放配置

- `avoidance_old_fraction`:
- `cooperation_old_fraction`:
- `avoidance_sample_ratio`:
- `cooperation_sample_ratio`:
- `avoidance_priority_mode`:
- `cooperation_priority_mode`:
- `rare_priority_mode`:

### 2.4 网络与训练超参数

- `actor_lr`:
- `critic_lr`:
- `gamma`:
- `tau`:
- `policy_noise`:
- `noise_clip`:
- `exploration_noise`:
- `policy_delay`:
- `batch_size`:
- `warmup_steps`:

## 3. 输出文件

- `output_dir`:
- `summary.json`:
- `metrics.csv`:
- `training_history.csv`:
- 关键图表：

## 4. 课程推进记录

### 4.1 访问过的课程环境

- 环境链：
- 最终停留环境：
- 是否发生晋级：
- 是否发生回退：

### 4.2 关键窗口统计

- 最近 `reward_mean`:
- 最近 `success_mean`:
- 最近 `reward_std`:
- 当前阶段判定：

## 5. 奖励分解观察

- `target_reward`:
- `avoidance_reward`:
- `collaboration_reward`:
- `recovery_reward`:
- `smoothness_penalty`:
- `consistency_penalty`:
- `success_bonus`:

结论：

- 是否有明确上升项：
- 是否有长期无效项：
- 是否存在奖励错配：

## 6. Replay 观察

### 6.1 当前回放后端

- `current_replay_type`:
- `last_sample_sources`:

### 6.2 guidance

- `buffer_size`:
- `old_pool_size`:
- `sample_counts`:

### 6.3 avoidance

- `bucket_sizes`:
- `bucket_fractions`:
- `sampling_counts`:
- `old_pool_size`:
- `old_pool_sample_count`:

### 6.4 cooperation

- `bucket_sizes`:
- `bucket_fractions`:
- `sampling_counts`:
- `old_pool_size`:
- `old_pool_sample_count`:

结论：

- 是否存在层为空：
- 是否存在 old pool 几乎不生效：
- 是否存在采样偏斜：

## 7. 优化与损失观察

- `final_actor_loss`:
- `final_critic_loss`:
- loss 是否稳定：
- 是否出现明显震荡：

结论：

- 更像网络问题 / replay 问题 / 奖励问题 / 环境过难：

## 8. 本轮诊断结论

### 8.1 主要问题

1.
2.
3.

### 8.2 下轮只改的参数

1.
2.
3.

### 8.3 不改的部分

1.
2.
3.
