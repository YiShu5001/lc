# 第四章实验矩阵

## 1. 用途

本文件用于给另一个 AI 或协作者直接提供第四章实验设计清单，避免实现过程中自行决定：

- 用哪些对比组
- 跑哪些复杂度
- 输出哪些图
- 哪些实验必须优先完成

## 2. 方法组矩阵

### 核心方法组

必须保留：

- `task_decomposed`
  第四章主方法，对应 `MultiUAVModel`
- `single_stream_mlp`
  单流基线

### 机制消融组

建议后续补齐：

- `without_curriculum`
  关闭课程学习
- `without_pyramid_per`
  关闭金字塔经验池
- `without_task_decomposition`
  用单流策略替换双阶段语义
- `alt_memory_scoring`
  更换经验评分方式
- `uniform_replay`
  去掉优先级采样

## 3. 场景复杂度矩阵

### 难度等级

固定按 4 个等级组织：

- `easy`
- `medium`
- `hard`
- `extreme`

### 复杂度维度

每个难度等级都应组合以下维度：

- 无人机数量
- 障碍物数量
- 障碍物布局类型
- 障碍物动态性
- 目标动态性
- 环境密度
- 课程阶段

## 4. 最低必须完成的实验

### 实验 A：主方法 vs 基线

目的：

- 证明任务分解结构优于单流 MLP

对比组：

- `task_decomposed`
- `single_stream_mlp`

场景：

- `easy / medium / hard`

输出：

- 成功率
- 碰撞率
- formation error
- reward

### 实验 B：课程学习有效性

目的：

- 证明 curriculum 不只是训练技巧，而能提升复杂场景稳定性

对比组：

- `task_decomposed`
- `without_curriculum`

场景：

- 分阶段课程任务

输出：

- 不同阶段成功率
- 阶段切换时性能变化
- 收敛速度

### 实验 C：Pyramid-PER 有效性

目的：

- 证明多层经验池能提升跨阶段学习稳定性

对比组：

- `task_decomposed`
- `without_pyramid_per`
- `uniform_replay`

输出：

- 成功率
- collision rate
- stage retention
- 旧阶段性能保持情况

### 实验 D：复杂度泛化实验

目的：

- 证明方法不只在单一规模场景有效

场景：

- 无人机数量变化
- 障碍物数量变化
- 动态障碍开启/关闭
- 目标动态开启/关闭

输出：

- 泛化成功率
- 碰撞率
- 占位误差

## 5. 图表矩阵

### 必须输出的图

- `training_reward_curve`
- `success_rate_curve`
- `collision_rate_curve`
- `formation_error_curve`
- `curriculum_transition_curve`
- `complexity_generalization_bar`
- `ablation_bar`
- `trajectory_plot`

### 推荐补充的图

- `attention_heatmap`
- `memory_pool_distribution`
- `stage_retention_plot`

## 6. 实验输出目录建议

建议统一输出为：

- `outputs/planning/<difficulty>/stage_<index>/`
- `outputs/planning/compare/<experiment_name>/`
- `outputs/planning/ablations/<experiment_name>/`
- `outputs/planning/generalization/<experiment_name>/`

每个实验目录至少包含：

- `summary.json`
- `metrics.csv`
- `training_log.csv`
- `figures/*.svg`

## 7. 另一个 AI 的执行优先级

若资源有限，建议另一个 AI 按以下顺序推进：

1. `task_decomposed vs single_stream_mlp`
2. `without_curriculum`
3. `without_pyramid_per`
4. `complexity generalization`
5. `attention heatmap`

## 8. 最终验收口径

第四章实验矩阵完成的最低标准是：

- 至少 2 组方法对比
- 至少 2 个机制消融
- 至少 3 个复杂度等级
- 至少 6 类图表输出
- 输出目录和日志格式统一
