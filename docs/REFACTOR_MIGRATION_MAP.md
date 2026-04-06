# 重构迁移清单

## 文档目的

这份文档是 `docs/REFACTOR_DIRECTORY_DESIGN.md` 的执行版。

它回答 6 个问题：

1. 当前文件在哪里
2. 未来应该迁移到哪里
3. 是保留、迁移、归档还是重写
4. 对应论文章节是什么
5. 迁移优先级是什么
6. 迁移时需要注意什么

## 状态标签

- `迁移`：保留核心逻辑，迁到新目录并修 import
- `重写`：保留概念和接口，但建议重写实现
- `归档`：不再作为主干使用，转移到 `experiments/legacy/` 或 `legacy/`
- `保留`：短期无需迁移，先保留

## 优先级说明

- `P0`：论文主贡献主路径，必须先处理
- `P1`：主流程依赖项，紧随其后
- `P2`：通用基础设施，后续处理
- `P3`：历史实验、临时脚本、样例、资源

---

## 一、P0：论文主贡献模块

### 1. 第 3 章：TSA-LADRC / RL-LADRC

| 当前路径 | 新路径 | 状态 | 论文章节 | 说明 |
|---|---|---:|---|---|
| `Gym_env/LADRC_Controller.py` | `src/lc/control/controllers/ladrc.py` | 迁移 | 第3章 控制器设计 | 作为低层控制器主实现 |
| `Gym_env/gym_pybullet_drones/control/LADRC.py` | `src/lc/env/pybullet_drones/control/ladrc_backend.py` | 迁移 | 第3章 控制器实现支撑 | 与通用控制器逻辑分层 |
| `Reinforce_learning/RLg/TSA_LADRC.py` | `src/lc/control/controllers/rl_ladrc_adapter.py` | 重写 | 第3章 RL-LADRC 调参机制 | 建议从“算法脚本”重构成“控制调参模块” |
| `Gym_env/gym_pybullet_drones/envs/TSA_LADRC_Env.py` | `src/lc/control/envs/tsa_ladrc_env.py` | 迁移 | 第3章 控制层实验 | 控制层专用环境 |
| `examples/train_tsa_ladrc.py` | `experiments/control/train_tsa_ladrc.py` | 迁移 | 第3章 实验 | 作为控制层主实验入口 |
| `Trainer/HGCTrainer.py` | `src/lc/training/trainers/hgc_trainer.py` | 重写 | 第3章 / 系统集成 | 需要去掉对旧目录的强耦合 |

### 迁移备注

- `TSA-LADRC` 不应继续放在 `RLg/` 这类通用算法目录里。
- 它在论文中是控制层方法，而不是通用 RL 算法。
- 重构时应拆为：
  - 控制器主逻辑
  - RL 参数调节逻辑
  - TSA 时间调度逻辑
  - 控制层实验入口

---

### 2. 第 4 章：Pyramid-PER / 多层经验池

| 当前路径 | 新路径 | 状态 | 论文章节 | 说明 |
|---|---|---:|---|---|
| `Reinforce_learning/buffers/PyramidPER.py` | `src/lc/planning/memory/pyramid_per.py` | 迁移 | 第4章 Pyramid-PER | 论文主贡献，优先迁移 |
| `Reinforce_learning/buffers/multi_level/MultiLevelBuffer.py` | `src/lc/planning/memory/multi_level_buffer.py` | 迁移 | 第4章 经验池结构 | 保留核心逻辑 |
| `Reinforce_learning/buffers/multi_level/BaseCoveragePool.py` | `src/lc/planning/memory/pools/base_coverage_pool.py` | 迁移 | 第4章 多层池 | |
| `Reinforce_learning/buffers/multi_level/DifficultyFocusPool.py` | `src/lc/planning/memory/pools/difficulty_focus_pool.py` | 迁移 | 第4章 多层池 | |
| `Reinforce_learning/buffers/multi_level/KeyEventPool.py` | `src/lc/planning/memory/pools/key_event_pool.py` | 迁移 | 第4章 多层池 | |
| `Reinforce_learning/buffers/MultiLevelBufferConfig.py` | `src/lc/planning/configs/memory_configs.py` | 迁移 | 第4章 配置 | |
| `Reinforce_learning/buffers/metrics/BaseMetric.py` | `src/lc/planning/memory/metrics/base.py` | 迁移 | 第4章 指标定义 | |
| `Reinforce_learning/buffers/metrics/TDMetric.py` | `src/lc/planning/memory/metrics/td_metric.py` | 迁移 | 第4章 TD/Error 相关 | |
| `Reinforce_learning/buffers/metrics/RiskMetric.py` | `src/lc/planning/memory/metrics/risk_metric.py` | 迁移 | 第4章 风险维度 | |
| `Reinforce_learning/buffers/metrics/NoveltyMetric.py` | `src/lc/planning/memory/metrics/novelty_metric.py` | 迁移 | 第4章 新颖性维度 | |
| `Reinforce_learning/buffers/metrics/CollaborationMetric.py` | `src/lc/planning/memory/metrics/collaboration_metric.py` | 迁移 | 第4章 协同维度 | |
| `Reinforce_learning/buffers/filters/BaseFilter.py` | `src/lc/planning/memory/filters/base.py` | 迁移 | 第4章 样本过滤 | |
| `Reinforce_learning/buffers/filters/LearningFilter.py` | `src/lc/planning/memory/filters/learning_filter.py` | 迁移 | 第4章 样本过滤 | |
| `Reinforce_learning/buffers/filters/PriorityFilter.py` | `src/lc/planning/memory/filters/priority_filter.py` | 迁移 | 第4章 样本过滤 | |
| `Reinforce_learning/buffers/filters/ValueFilter.py` | `src/lc/planning/memory/filters/value_filter.py` | 迁移 | 第4章 样本过滤 | |
| `Reinforce_learning/buffers/samplers/PrioritizedSampler.py` | `src/lc/planning/memory/samplers/prioritized_sampler.py` | 迁移 | 第4章 采样器 | |
| `Reinforce_learning/buffers/samplers/MultiPoolSampler.py` | `src/lc/planning/memory/samplers/multi_pool_sampler.py` | 迁移 | 第4章 采样器 | |
| `examples/train_multilevel_buffer.py` | `experiments/planning/train_multilevel_buffer.py` | 迁移 | 第4章 实验 | 主实验脚本之一 |

### 迁移备注

- 这部分与第 4 章内容绑定最强，建议形成独立包：
  - `memory/`
  - `metrics/`
  - `filters/`
  - `samplers/`
  - `pools/`
- 不建议继续把它作为 `Reinforce_learning/buffers/` 下的一个“普通 buffer”。

---

### 3. 第 4 章：Task-Decomposed Actor / Transformer 规划模型

| 当前路径 | 新路径 | 状态 | 论文章节 | 说明 |
|---|---|---:|---|---|
| `NN/TaskDecomposedActor.py` | `src/lc/planning/models/task_decomposed_actor.py` | 迁移 | 第4章 Task-Decomposed Actor | 核心主模型 |
| `NN/MultiUAVModel.py` | `src/lc/planning/models/multi_uav_model.py` | 迁移 | 第4章 主策略模型 | |
| `NN/obstacle_branch.py` | `src/lc/planning/models/obstacle_branch.py` | 迁移 | 第4章 避障分支 | |
| `NN/collaborative_branch.py` | `src/lc/planning/models/collaborative_branch.py` | 迁移 | 第4章 协同分支 | |
| `NN/embeddings.py` | `src/lc/planning/models/embeddings.py` | 迁移 | 第4章 输入嵌入 | |
| `NN/components.py` | `src/lc/planning/models/components.py` | 迁移 | 第4章 Transformer 组件 | |
| `NN/BaseNN.py` | `src/lc/rl/models/base.py` | 迁移 | 通用基础层 | 只保留抽象接口 |
| `NN/model_factory.py` | `src/lc/rl/models/factory.py` | 重写 | 通用基础层 | 需要按新分层重写 |

### 迁移备注

- `TaskDecomposedActor.py` 当前既包含核心结构，又混有示例代码，迁移后要拆分：
  - 模型定义
  - 单元测试
  - demo/example
- `BaseNN.py` 应降级成通用接口层，不再承载论文方法含义。

---

### 4. 第 4 章：课程学习

| 当前路径 | 新路径 | 状态 | 论文章节 | 说明 |
|---|---|---:|---|---|
| `Trainer/curriculum/BaseCurriculum.py` | `src/lc/planning/curriculum/base.py` | 迁移 | 第4章 Curriculum | |
| `Trainer/curriculum/__init__.py` | `src/lc/planning/curriculum/__init__.py` | 迁移 | 第4章 Curriculum | |

### 迁移备注

- 当前只看到了基类；如果后续还有动态课程、阶段调度逻辑，应直接归入 `planning/curriculum/`。
- 课程学习在论文里是规划层一部分，不应继续放在 trainer 内部。

---

## 二、P1：主流程依赖模块

### 5. 训练器与训练编排

| 当前路径 | 新路径 | 状态 | 论文章节 | 说明 |
|---|---|---:|---|---|
| `Trainer/BaseTrainer.py` | `src/lc/training/trainers/base_trainer.py` | 迁移 | 支撑全篇实验 | |
| `Trainer/OffPolicyTrainer.py` | `src/lc/training/trainers/off_policy_trainer.py` | 迁移 | 第3/4章实验主流程 | |
| `Trainer/HGCTrainer.py` | `src/lc/training/trainers/hgc_trainer.py` | 重写 | 第3章主流程 | 与旧目录强耦合 |
| `Trainer/callbacks.py` | `src/lc/training/callbacks/callbacks.py` | 迁移 | 通用训练支撑 | |
| `Trainer/rewards/BaseReward.py` | `src/lc/training/rewards/base_reward.py` | 迁移 | 奖励体系基础 | |

### 迁移备注

- trainer 应只负责训练调度，不应继续承担课程学习、环境语义或论文方法逻辑。
- 如果奖励在规划层高度专用，后续再分流到 `planning/rewards/`。

---

### 6. 环境与工厂

| 当前路径 | 新路径 | 状态 | 论文章节 | 说明 |
|---|---|---:|---|---|
| `Gym_env/BaseEnv.py` | `src/lc/env/base_env.py` | 迁移 | 全篇实验基础设施 | |
| `Gym_env/wrappers/GymnasiumWrapper.py` | `src/lc/env/wrappers/gymnasium_wrapper.py` | 迁移 | 全篇实验基础设施 | |
| `Gym_env/factories/PyBulletDronesFactory.py` | `src/lc/env/factories/pybullet_factory.py` | 迁移 | 第3/4章实验支撑 | |
| `Gym_env/gym_pybullet_drones/` | `src/lc/env/pybullet_drones/` | 迁移 | 仿真环境底座 | 整体迁移，先不改逻辑 |

### 迁移备注

- `Gym_env/` 未来不再作为顶层目录存在。
- 区分：
  - 方法专用环境 -> `control/envs` 或 `planning/envs`
  - 通用仿真基础设施 -> `env/pybullet_drones`

---

### 7. 系统层

| 当前路径 | 新路径 | 状态 | 论文章节 | 说明 |
|---|---|---:|---|---|
| `core_architecture/ControlLayer.py` | `src/lc/system/control_layer.py` | 迁移 | 分层架构总述 | |
| `core_architecture/PlanningLayer.py` | `src/lc/system/planning_layer.py` | 迁移 | 分层架构总述 | |
| `core_architecture/SystemIntegration.py` | `src/lc/system/system_integration.py` | 迁移 | 分层架构总述 | |

### 迁移备注

- 这部分和论文叙述直接对应，建议尽快从 `core_architecture/` 迁出。
- 迁出后，系统层会更像一个“论文结构映射层”，便于整体维护。

---

## 三、P2：通用基础设施

### 8. 通用 RL 算法与工厂

| 当前路径 | 新路径 | 状态 | 论文章节 | 说明 |
|---|---|---:|---|---|
| `Reinforce_learning/Basealgos.py` | `src/lc/rl/algorithms/base.py` | 迁移 | 通用基础 | |
| `Reinforce_learning/algo_factory.py` | `src/lc/rl/factory.py` | 迁移 | 通用基础 | |
| `Reinforce_learning/exploration/BaseExploration.py` | `src/lc/rl/exploration/base.py` | 迁移 | 通用基础 | |
| `NN/action_dists.py` | `src/lc/rl/distributions/action_dists.py` | 迁移 | 通用基础 | |
| `Reinforce_learning/RLg/PPO.py` | `src/lc/rl/algorithms/ppo.py` | 迁移 | 通用基础 | |
| `Reinforce_learning/RLg/TD3.py` | `src/lc/rl/algorithms/td3.py` | 迁移 | 通用基础 | |
| `Reinforce_learning/RLg/SAC.py` | `src/lc/rl/algorithms/sac.py` | 迁移 | 通用基础 | |
| `Reinforce_learning/RLg/DDPG.py` | `src/lc/rl/algorithms/ddpg.py` | 迁移 | 通用基础 | |
| `Reinforce_learning/RLg/DDPG_refactored.py` | `src/lc/rl/algorithms/ddpg_refactored.py` | 迁移 | 通用基础 | 先保留双版本 |
| `Reinforce_learning/RLg/DQN.py` | `src/lc/rl/algorithms/dqn.py` | 迁移 | 通用基础 | |
| `Reinforce_learning/RLg/A2C.py` | `src/lc/rl/algorithms/a2c.py` | 迁移 | 通用基础 | |
| `Reinforce_learning/RLg/Agent.py` | `src/lc/rl/legacy/agent.py` | 归档 | 历史实现 | |
| `Reinforce_learning/RLg/network.py` | `src/lc/rl/legacy/network.py` | 归档 | 历史实现 | |
| `Reinforce_learning/RLg/Buffer.py` | `src/lc/rl/legacy/buffer.py` | 归档 | 历史实现 | |
| `Reinforce_learning/RLg/MaBuffer.py` | `src/lc/rl/legacy/ma_buffer.py` | 归档 | 历史实现 | |
| `Reinforce_learning/RLg/prioritized_memory.py` | `src/lc/rl/legacy/prioritized_memory.py` | 归档 | 历史实现 | |
| `Reinforce_learning/RLg/SumTree.py` | `src/lc/rl/legacy/sum_tree.py` | 归档 | 历史实现 | |
| `Reinforce_learning/RLg/attention.py` | `src/lc/rl/legacy/attention.py` | 归档 | 历史实验 | |
| `Reinforce_learning/RLg/LeMer.py` | `src/lc/rl/legacy/lemer.py` | 归档 | 历史实验 | |

### 迁移备注

- `RLg/` 明显是历史堆积目录，建议大部分进入：
  - `rl/algorithms/`
  - `rl/legacy/`
- 论文主方法不要继续依赖 `legacy` 目录。

---

### 9. 配置与入口

| 当前路径 | 新路径 | 状态 | 论文章节 | 说明 |
|---|---|---:|---|---|
| `configs/default_configs.py` | `src/lc/configs/default.py` | 迁移 | 全局配置 | |
| `main.py` | `src/lc/training/orchestration/run_pipeline.py` 或保留根入口 | 重写 | 全局入口 | 应转成统一编排入口 |
| `pyproject.toml` | 原位保留 | 保留 | 项目配置 | |

### 迁移备注

- `main.py` 最终应只承担入口职责。
- 真正的逻辑应下沉到 `src/lc/...`。

---

### 10. 工具与绘图

| 当前路径 | 新路径 | 状态 | 论文章节 | 说明 |
|---|---|---:|---|---|
| `utils/logger.py` | `src/lc/utils/logger.py` | 迁移 | 通用支撑 | |
| `utils/draw_pic.py` | `src/lc/utils/plotting.py` | 迁移 | 图表支撑 | |
| `utils/draw_thesis_plots.py` | `experiments/plotting/draw_thesis_plots.py` | 迁移 | 论文图表 | |
| `utils/__init__.py` | `src/lc/utils/__init__.py` | 迁移 | 通用支撑 | |

---

## 四、P3：实验脚本、样例、临时资源

### 11. 训练脚本与示例

| 当前路径 | 新路径 | 状态 | 论文章节 | 说明 |
|---|---|---:|---|---|
| `examples/train_pybullet.py` | `experiments/integrated/train_pybullet.py` | 迁移 | 综合实验 | |
| `examples/train_cartpole.py` | `experiments/legacy/train_cartpole.py` | 归档 | 非论文主线 | |
| `Gym_env/examples/*.py` | `experiments/legacy/gym_env_examples/` | 归档 | 历史调试样例 | 大部分不是论文主线 |
| `Gym_env/examples/results/*` | `outputs/legacy/gym_env_results/` | 归档 | 历史输出 | |
| `Gym_env/examples/*.xlsx` | `outputs/legacy/control_tuning_tables/` | 归档 | 早期调参数据 | |

### 迁移备注

- 这部分不要优先投入大量重构时间。
- 先归档，等主干迁移完成后再清理。

---

### 12. 论文辅助脚本

| 当前路径 | 新路径 | 状态 | 论文章节 | 说明 |
|---|---|---:|---|---|
| `build_tex.py` | `scripts/build_tex.py` | 迁移 | 论文工程 | |
| `append_bib.py` | `scripts/append_bib.py` | 迁移 | 参考文献 | |
| `fetch_papers.py` | `scripts/fetch_papers.py` | 迁移 | 文献抓取 | |

---

## 五、第一阶段建议创建的新目录骨架

在真正迁移代码前，先创建这些目录：

```text
src/lc/control/controllers/
src/lc/control/envs/
src/lc/control/configs/
src/lc/planning/curriculum/
src/lc/planning/memory/
src/lc/planning/memory/pools/
src/lc/planning/memory/metrics/
src/lc/planning/memory/filters/
src/lc/planning/memory/samplers/
src/lc/planning/models/
src/lc/planning/envs/
src/lc/planning/rewards/
src/lc/planning/configs/
src/lc/rl/algorithms/
src/lc/rl/exploration/
src/lc/rl/distributions/
src/lc/rl/models/
src/lc/rl/legacy/
src/lc/training/trainers/
src/lc/training/callbacks/
src/lc/training/rewards/
src/lc/training/orchestration/
src/lc/env/wrappers/
src/lc/env/factories/
src/lc/env/pybullet_drones/
src/lc/system/
src/lc/configs/
src/lc/utils/
experiments/control/
experiments/planning/
experiments/integrated/
experiments/legacy/
experiments/plotting/
outputs/checkpoints/
outputs/logs/
outputs/figures/
outputs/tables/
outputs/videos/
outputs/legacy/
tests/control/
tests/planning/
tests/integration/
tests/smoke/
scripts/
```

---

## 六、建议的执行顺序

### 第一步：搭骨架

先创建新目录，不迁移逻辑。

目标：

- 让新结构可落地
- 不破坏现有训练流程

### 第二步：迁移论文主线

按顺序迁移：

1. `TaskDecomposedActor` 及规划模型
2. `PyramidPER` 与多层经验池
3. `TSA-LADRC` 控制链路

原因：

- 这三块最直接对应论文贡献
- 它们决定后续代码与论文同步效率

### 第三步：迁移训练与环境基础设施

包括：

- trainer
- env base
- wrappers
- factory
- configs

### 第四步：迁移脚本与输出

包括：

- experiments
- plotting
- outputs

### 第五步：归档历史代码

把不再作为主干的代码迁入：

- `experiments/legacy/`
- `src/lc/rl/legacy/`
- `outputs/legacy/`

---

## 七、后续执行时的规则

开始真正重构后，每次只迁移一个“完整小块”，不要一次横扫全仓。

推荐任务粒度：

- 一次迁移一个模型模块
- 一次迁移一个 memory 子系统
- 一次迁移一个 trainer
- 一次迁移一个实验入口

每次迁移完成后至少要检查：

1. import 是否可用
2. 配置路径是否可用
3. 训练脚本是否仍能启动
4. 该模块对应的论文表述是否仍然准确

---

## 八、建议的下一步

最合理的下一步是：

1. 先创建新目录骨架
2. 先迁移 `TaskDecomposedActor` 相关模型链

原因：

- 它是第 4 章核心表达
- 当前边界比较清楚
- 迁移后最容易建立新目录范式

如果之后继续执行，建议顺序是：

1. `TaskDecomposedActor` 相关模型链
2. `PyramidPER` 与多层经验池
3. `TSA-LADRC` 控制链
4. trainer / env / factory
5. experiments / outputs / legacy 清理
