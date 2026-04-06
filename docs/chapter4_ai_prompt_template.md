# 第四章 AI 提示词模板

下面这段提示词可以直接发给另一个 AI，用来继续实现第四章代码主链。

---

你现在接手的是一个多无人机协同规划项目的**第四章代码实现**，请严格基于当前仓库已有的新架构继续推进，不要重写架构，不要回退到旧目录。

## 你的任务定位

你负责的是**第四章规划层**，目标是把当前 `src/lc/planning/` 从“可运行骨架”推进成“可训练、可对比、可绘图、可与系统桥接”的完整实现。

你**不用处理论文正文**，只需要把第四章代码和实验体系补完整。

## 当前仓库中你必须遵守的固定口径

### 1. 主实现不能改

第四章主模型固定为：

- `src/lc/planning/models/multi_uav_model.py`
  其中主类是 `MultiUAVModel`

它代表第四章的正式主方法，不要把它替换成普通 MLP，也不要把主口径改成旧目录实现。

### 2. 输入接口不能改

第四章规划层主输入固定为三类结构化观测：

- `self_state`
- `obstacles`
- `neighbors`

不要把主实现改成只有扁平向量输入。  
如果需要扁平输入，只能作为兼容形式或基线，不应替代主接口。

### 3. 方法语义不能改

第四章主模型语义固定为：

- 第一阶段：避障流
- 第二阶段：协同流

输出固定为：

- `avoid_action`
- `final_action`

### 4. 目录边界不能改

你必须在当前新架构上继续实现：

- `src/lc/planning/`
- `src/lc/envs/`
- `src/lc/analysis/`
- `src/lc/entrypoints/`

不要把新实现写回旧目录：

- `NN/`
- `Trainer/`
- `Reinforce_learning/`

## 你开始前必须先读的文件

请先阅读以下文件，再开始修改：

- `docs/chapter4_execution_plan.md`
- `docs/chapter4_readme.md`
- `docs/chapter4_experiment_matrix.md`
- `src/lc/planning/experiments/compare.py`
- `src/lc/planning/trainers/planning_trainer.py`
- `src/lc/planning/models/multi_uav_model.py`
- `src/lc/planning/envs/swarm.py`
- `src/lc/planning/memory/pyramid.py`
- `src/lc/planning/curriculum/scheduler.py`

## 当前第四章的真实状态

当前已经有：

- 环境骨架
- 输入编码器
- `MultiUAVModel`
- `SingleStreamMLPPolicy`
- `StructuredCritic`
- `CurriculumScheduler`
- `PyramidReplayMemory`
- 简单实验入口
- 简单柱状图输出

但当前**还没有真正补实**：

- actor-critic 训练闭环
- curriculum 与场景切换联动
- Pyramid-PER 优先级和采样逻辑
- 完整奖励系统
- 完整对比实验
- 论文级图表输出

## 你的优先任务

请按下面顺序执行，不要跳步。

### 阶段 1：补齐训练闭环

你要做的事：

- 在 `PlanningTrainer` 中实现真实 actor-critic 训练：
  - 经验采集
  - replay sampling
  - critic loss
  - actor loss
  - target update 或等价稳定机制
- 让 `StructuredCritic` 实际进入训练闭环
- 输出训练日志：
  - `episode`
  - `reward`
  - `success_rate`
  - `collision_rate`
  - `formation_error`
  - `actor_loss`
  - `critic_loss`

完成标准：

- 第四章不再只是前向评估，而是能真正训练。

### 阶段 2：补齐课程学习

你要做的事：

- 让 `CurriculumScheduler` 和环境难度切换联动
- 将课程阶段固定为：
  - `easy`
  - `medium`
  - `hard`
  - `extreme`
- 课程切换依据至少包含：
  - 成功率
  - 碰撞率
  - 稳定/恢复类指标
- 在实验输出中记录：
  - 当前阶段
  - 阶段切换历史
  - 各阶段平均指标

完成标准：

- 第四章训练输出可以明确看到课程阶段演化。

### 阶段 3：补齐 Pyramid-PER

你要做的事：

- 将 `PyramidReplayMemory` 从三层 bucket 补成真正可用的多层经验池
- 至少支持：
  - `high`
  - `medium`
  - `low`
- 增加样本优先级逻辑，至少考虑：
  - 风险/碰撞
  - 协同误差
  - 稀有事件
  - TD 误差或等价优先项
- 增加：
  - 分层采样比例
  - 旧阶段样本保留
  - 阶段切换后的样本重用

完成标准：

- 第四章能输出各层经验池的样本统计和采样占比。

### 阶段 4：补齐奖励系统

请将第四章奖励拆成清晰子项，至少包括：

- `target_reward`
- `avoidance_reward`
- `collaboration_reward`
- `recovery_reward`
- `smoothness_penalty`
- `consistency_penalty`

并在训练日志与实验汇总中输出奖励分项。

### 阶段 5：补齐对比实验与消融实验

至少保留这些实验组：

- `task_decomposed`
- `single_stream_mlp`
- `without_curriculum`
- `without_pyramid_per`
- `uniform_replay`

至少保留这些复杂度维度：

- 无人机数量变化
- 障碍物数量变化
- 障碍物动态性变化
- 目标动态性变化
- 课程阶段变化

### 阶段 6：补齐图表输出

至少输出这些图：

- 收敛曲线
- 成功率 / 碰撞率曲线
- 编队误差 / 占位误差图
- 课程切换图
- 消融图
- 复杂度泛化图
- 轨迹图
- 注意力热力图

## 你不应该做的事

- 不要重写 `src/lc/planning` 整体结构
- 不要把主方法替换成 `MLP`
- 不要回到旧目录重做
- 不要把三输入结构改掉
- 不要把第四章主方法改成别的命名口径

## 输出要求

你完成每一步后，必须同步更新：

- 代码
- 实验输出
- README 或说明文档

最终你需要明确告诉我：

1. 你改了哪些主模块
2. 第四章现在新增了哪些实验能力
3. 跑了哪些测试或 smoke
4. 还有哪些剩余缺口

## 补充说明

如果你在实现过程中发现当前仓库状态与你的预期不一致，不要擅自大改架构，先优先遵守这些文件里的现有口径：

- `docs/chapter4_execution_plan.md`
- `docs/chapter4_readme.md`
- `docs/chapter4_experiment_matrix.md`

如果你遇到不明确的地方，优先做“最小改动、保留当前主架构”的方案。

---

## 简短版提示词

如果只想发一段短提示，可以直接发下面这版：

```text
请继续实现这个仓库的第四章规划层代码，严格基于当前 src/lc/planning 架构继续做，不要重写架构，也不要回到旧目录。主方法固定为 MultiUAVModel，主输入固定为 self_state / obstacles / neighbors，语义固定为先避障后协同。请优先补齐：1. actor-critic 训练闭环；2. curriculum 与环境切换联动；3. Pyramid-PER 的优先级与分层采样；4. 奖励分项；5. 对比实验与消融实验；6. 论文图表输出。开始前先阅读 docs/chapter4_execution_plan.md、docs/chapter4_readme.md、docs/chapter4_experiment_matrix.md。不要把主实现降级成普通 MLP。```

