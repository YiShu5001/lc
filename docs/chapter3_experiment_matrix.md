# 第三章实验矩阵

## 1. 用途

本文档用于给另一个 AI 或协作者直接提供第三章实验设计清单，避免实现过程中反复自行决定：

- 跑哪些方法组
- 跑哪些难度
- 输出哪些图
- 哪些实验必须优先完成

## 2. 方法组矩阵

### 核心方法组

必须保留：

- `pid`
- `ladrc`
- `ddpg_ladrc`
- `mddpg_ladrc`

说明：

- `pid` 是传统控制基线
- `ladrc` 是鲁棒控制基线
- `ddpg_ladrc` 是第三章默认主方法
- `mddpg_ladrc` 是增强版对比方法

### 机制消融组

建议保留：

- `no_state_stack`
- `no_action_hold`
- `no_n_step`

说明：

- 用于验证 `mDDPG` 三项增强是否真的有效

## 3. 场景复杂度矩阵

### 难度等级

固定按 4 个等级组织：

- `easy`
- `medium`
- `hard`
- `extreme`

### 复杂度维度

每个难度等级应至少覆盖以下维度：

- 参考轨迹复杂度
- 扰动强度
- 扰动出现频率
- 任务时长
- 噪声或建模不确定性

如果后续扩展到更真实对象，也可继续加入：

- 控制对象参数摄动
- 外部风扰变化

## 4. 最低必须完成的实验

### 实验 A：四方法主对比

目的：

- 验证 `DDPG-LADRC` 与 `mDDPG-LADRC` 相对传统方法的效果提升

对比组：

- `pid`
- `ladrc`
- `ddpg_ladrc`
- `mddpg_ladrc`

场景：

- `easy / medium / hard`

输出：

- `IAE`
- `RMSE`
- 超调量
- 调节时间
- 稳态误差
- 控制能量
- 扰动恢复时间

### 实验 B：mDDPG 增强机制消融

目的：

- 验证三项增强机制分别带来的贡献

对比组：

- `mddpg_ladrc`
- `no_state_stack`
- `no_action_hold`
- `no_n_step`

场景：

- `medium`
- `hard`

输出：

- `IAE`
- `RMSE`
- 收敛速度

### 实验 C：DDPG vs mDDPG 主消融

目的：

- 证明增强版相对基础版的增益

对比组：

- `ddpg_ladrc`
- `mddpg_ladrc`

场景：

- `medium`
- `hard`

输出：

- `IAE`
- `RMSE`
- 控制能量
- 奖励曲线

### 实验 D：复杂度泛化实验

目的：

- 验证方法在不同扰动与轨迹复杂度下的稳定性

对比组：

- `ladrc`
- `ddpg_ladrc`
- `mddpg_ladrc`

场景：

- `easy`
- `medium`
- `hard`
- `extreme`

输出：

- 难度分层指标表
- 泛化性能图

## 5. 图表矩阵

### 必须输出的图

- `training_reward_curve`
- `training_actor_loss_curve`
- `training_critic_loss_curve`
- `metric_bar_iae`
- `metric_bar_rmse`
- `metric_bar_control_energy`
- `mechanism_ablation_bar`
- `time_response_output`
- `time_response_error`
- `time_response_control`
- `time_response_disturbance`

### 推荐补充的图

- 四方法统一时域对比图
- 多难度综合指标图
- 随机种子平均曲线图

## 6. 实验输出目录建议

建议统一输出为：

- `outputs/control/<difficulty>/`
- `outputs/control/compare/<experiment_name>/`
- `outputs/control/ablations/<experiment_name>/`
- `outputs/control/generalization/<experiment_name>/`

每个实验目录至少包含：

- `summary.json`
- `metrics.csv`
- `training_log.csv`
- `figures/*.svg`

如果是 RL 方法，还建议包含：

- `checkpoints/*.pt`

## 7. 另一位 AI 的执行优先级

若资源有限，建议按如下顺序推进：

1. 四方法主对比
2. `DDPG-LADRC vs mDDPG-LADRC`
3. 三项增强机制消融
4. 复杂度泛化实验
5. 更论文化时域图整理

## 8. 最终验收口径

第三章实验矩阵完成的最低标准是：

- 至少 4 组方法对比
- 至少 3 个机制消融组
- 至少 3 个难度等级
- 至少 6 类图表输出
- 输出目录和日志格式统一
