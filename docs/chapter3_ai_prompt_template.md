# 第三章 AI 提示词模板

下面这段提示词可以直接发给另一个 AI，用来继续实现第三章代码主链。

---

你现在接手的是这个仓库的**第三章控制层代码实现**，请严格基于当前已经存在的 `src/lc/control/` 新架构继续推进，不要重写架构，也不要回退到旧目录。

## 你的任务定位

你负责的是**第三章控制层**，目标是把当前 `src/lc/control/` 从“已经可运行的研究骨架”继续推进成“更稳定、可复现、可对比、可直接服务论文图表”的完整实现。

你不用处理论文正文，只需要继续补强第三章代码、实验和图表体系。

## 你必须遵守的固定口径

### 1. 主方法不能改

第三章默认主方法固定为：

- `DDPG-LADRC`

增强版对比方法固定为：

- `mDDPG-LADRC`

不要把默认主方法改回 `mDDPG-LADRC`，也不要新增新的主命名口径。

### 2. 被控对象不能随意改

当前第三章默认任务固定为：

- 单轴位置跟踪
- 扰动抑制

不要直接改成别的主任务，除非是向后兼容扩展。

### 3. 强化学习动作不能改

强化学习当前固定为在线调节 `LADRC` 位置环三个参数：

- `omega_c`
- `omega_o`
- `b0`

不要改成“RL 直接输出控制量”的主实现。

### 4. 奖励函数不能擅自复杂化

当前奖励固定为：

```text
reward = -|position_error|
```

不要擅自把它改成复杂多项奖励，除非明确作为新实验分支保留。

### 5. 新实现目录边界不能改

你必须继续在这些目录中实现：

- `src/lc/control/`
- `src/lc/envs/`
- `src/lc/analysis/`
- `src/lc/entrypoints/`

不要把新实现写回旧目录：

- `Gym_env/`
- `Trainer/`
- `Reinforce_learning/`
- `NN/`

## 你开始前必须先读的文件

请先阅读以下文件，再开始修改：

- `docs/chapter3_execution_plan.md`
- `docs/chapter3_readme.md`
- `docs/chapter3_experiment_matrix.md`
- `docs/chapter3_method_figures.md`
- `src/lc/control/trainers/control_trainer.py`
- `src/lc/control/experiments/compare.py`
- `src/lc/control/envs/tracking.py`
- `src/lc/control/controllers/ladrc.py`
- `src/lc/control/controllers/adaptive_ladrc.py`
- `src/lc/control/policies/mddpg_control.py`

## 当前第三章的真实状态

当前已经有：

- `PID / LADRC / DDPG-LADRC / mDDPG-LADRC`
- 单轴跟踪与扰动抑制环境
- 基本训练闭环
- 训练日志与 checkpoint 导出
- 主方法对比图
- 机制消融图
- 时域响应图
- 方法插图

但当前还可以继续补强：

- 训练稳定性与多随机种子支持
- `mDDPG` 三项增强的细节验证
- 更完整的复杂度泛化实验
- 更统一的论文图表风格
- 更明确的实验配置与复现说明

## 你的优先任务

请按下面顺序执行，不要跳步。

### 阶段 1：补强 DDPG-LADRC 与 mDDPG-LADRC 训练稳定性

你要做的事：

- 检查并补强 replay buffer、target network、loss 更新逻辑
- 补强训练日志导出
- 增加更稳定的训练历史记录
- 视情况增加多随机种子运行入口

完成标准：

- 训练日志足以分析收敛曲线
- 方法对比不再只依赖单次结果

### 阶段 2：补强 mDDPG 三项增强机制

你要做的事：

- 明确状态堆叠的缓存与刷新逻辑
- 明确动作保持的执行逻辑
- 明确 `n-step` 回报构造逻辑
- 保持三项机制都可以独立开关

完成标准：

- `no_state_stack`
- `no_action_hold`
- `no_n_step`

三类消融都能独立运行并输出结果。

### 阶段 3：补强实验矩阵

至少保留这些方法组：

- `pid`
- `ladrc`
- `ddpg_ladrc`
- `mddpg_ladrc`

至少保留这些实验：

- 四方法主对比
- `DDPG-LADRC vs mDDPG-LADRC`
- 三项增强机制消融
- 多复杂度泛化实验

### 阶段 4：补强图表输出

至少输出这些图：

- 训练奖励曲线
- actor loss 曲线
- critic loss 曲线
- `IAE / RMSE / control_energy` 对比图
- 三项机制消融图
- 时域响应图

如果有余力，再补：

- 四方法统一时域对比图
- 多难度汇总图

## 你不应该做的事

- 不要重写 `src/lc/control` 整体结构
- 不要把主方法改成别的命名口径
- 不要把 RL 主语义改成直接输出控制量
- 不要回到旧目录重做
- 不要擅自把奖励改成复杂多目标主设定

## 输出要求

你每完成一步后，必须同步更新：

- 代码
- 实验输出
- README 或说明文档

最终你需要明确告诉我：

1. 你改了哪些主模块
2. 第三章现在新增了哪些实验能力
3. 跑了哪些测试或 smoke
4. 还有哪些剩余缺口

## 补充说明

如果你在实现过程中发现当前仓库状态与你预期不完全一致，不要擅自大改架构，优先遵守这些文件里的现有口径：

- `docs/chapter3_execution_plan.md`
- `docs/chapter3_readme.md`
- `docs/chapter3_experiment_matrix.md`

如果遇到不明确的地方，优先选择“最小改动、保留当前主架构”的方案。

---

## 简短版提示词

如果只想发一段短提示，可以直接发下面这版：

```text
请继续实现这个仓库的第三章控制层代码，严格基于当前 src/lc/control 架构继续做，不要重写架构，也不要回到旧目录。默认主方法固定为 DDPG-LADRC，增强版对比方法是 mDDPG-LADRC，默认任务固定为单轴位置跟踪 + 扰动抑制，强化学习固定为在线调节 LADRC 的 omega_c / omega_o / b0，奖励固定为 reward = -|position_error|。请优先补强：1. DDPG 与 mDDPG 训练稳定性和日志；2. 状态堆叠、动作保持、n-step 回报三项增强机制；3. 四方法对比和多复杂度实验；4. 论文图表输出。开始前先阅读 docs/chapter3_execution_plan.md、docs/chapter3_readme.md、docs/chapter3_experiment_matrix.md。
```
