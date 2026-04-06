# 重构目录设计

## 目标

本项目后续重构不应继续围绕历史目录名展开，例如 `NN/`、`Gym_env/`、`Reinforce_learning/`、`Trainer/` 这种“技术层切分”。

根据论文内容，更合理的重构目标是按“研究问题 + 方法层次 + 实验支撑”来组织代码，使代码结构直接映射论文主线：

1. 第 3 章：低层控制 `RL-LADRC + TSA`
2. 第 4 章：高层规划 `Curriculum + Pyramid-PER + Task-Decomposed Actor`
3. 系统集成：控制层与规划层耦合
4. 实验与论文产物：训练脚本、图表、结果、论文引用

因此，新的目录设计应满足：

- 代码结构能够直接对应论文结构
- 每个方法模块有清晰边界
- 环境、模型、算法、训练器不再分散在多个平级目录里
- 后续代码改动时，能快速定位对应论文章节

## 当前结构的主要问题

### 1. 目录按技术类型拆分，未按研究方法拆分

当前核心目录：

- `Gym_env/`
- `NN/`
- `Reinforce_learning/`
- `Trainer/`
- `core_architecture/`

这会导致一个完整方法横跨多个目录。

例如：

- `TSA-LADRC` 同时散落在环境、控制器、算法、训练脚本中
- `Pyramid-PER` 分散在 buffer、metrics、samplers、trainer 中
- `Task-Decomposed Actor` 分散在 `NN/` 和训练逻辑里

对论文驱动项目来说，这种布局会让“方法 = 多目录拼图”，不利于后续重构。

### 2. 历史代码和目标方法混杂

当前仓库里既有：

- 通用 RL 算法
- 早期实验脚本
- PyBullet 示例
- 论文主方法实现

这些内容混在一起，容易造成：

- 主干方法和实验草稿难区分
- 论文描述与实际主版本代码难对应
- 后续迁移时不清楚哪些文件应该保留、归档或重写

### 3. 论文与代码映射关系没有体现在目录上

用户的真实需求是“论文和代码双更新”，因此目录本身应服务于这种工作模式。

理想状态下，看到目录就能回答：

- 第 3 章代码在哪
- 第 4 章代码在哪
- 实验脚本在哪
- 图表产出在哪
- 系统集成层在哪

当前结构不能直接做到这一点。

## 重构原则

### 原则 1：按论文方法域组织，而不是按技术类别组织

优先按下面几个研究域组织：

- 控制层
- 规划层
- 记忆与课程学习
- 多智能体模型
- 系统集成
- 实验

### 原则 2：保留共用基础层，但把方法实现上浮

保留必要的共用基础设施，例如：

- 通用环境接口
- 通用训练循环
- 通用算法基类
- 通用网络组件

但论文主方法应有独立目录，不要继续隐藏在基础层中。

### 原则 3：实验目录必须和方法目录解耦

训练脚本、调参脚本、结果文件、图表脚本应该进入独立实验层，而不是散落在 `examples/`、`Gym_env/examples/`、`utils/`。

### 原则 4：论文产出需要可追踪到代码模块

未来每个论文章节最好能映射到一个代码包：

- 第 3 章 <-> `src/lc/control/`
- 第 4 章 <-> `src/lc/planning/`
- 系统集成 <-> `src/lc/system/`
- 实验章节 <-> `experiments/`

## 建议的新目录结构

建议将项目逐步迁移到下面的结构：

```text
lc/
├─ src/
│  └─ lc/
│     ├─ control/
│     │  ├─ controllers/
│     │  │  ├─ ladrc.py
│     │  │  ├─ rl_ladrc_adapter.py
│     │  │  └─ tsa_scheduler.py
│     │  ├─ envs/
│     │  │  ├─ tsa_ladrc_env.py
│     │  │  └─ control_env_base.py
│     │  ├─ configs/
│     │  │  └─ control_configs.py
│     │  └─ README.md
│     │
│     ├─ planning/
│     │  ├─ curriculum/
│     │  │  ├─ base.py
│     │  │  └─ dynamic_curriculum.py
│     │  ├─ memory/
│     │  │  ├─ pyramid_per.py
│     │  │  ├─ multi_level_buffer.py
│     │  │  ├─ filters/
│     │  │  ├─ metrics/
│     │  │  └─ samplers/
│     │  ├─ models/
│     │  │  ├─ task_decomposed_actor.py
│     │  │  ├─ multi_uav_model.py
│     │  │  ├─ obstacle_branch.py
│     │  │  ├─ collaborative_branch.py
│     │  │  ├─ embeddings.py
│     │  │  └─ components.py
│     │  ├─ envs/
│     │  │  ├─ multi_uav_planning_env.py
│     │  │  └─ planning_env_base.py
│     │  ├─ rewards/
│     │  │  ├─ obstacle_avoidance_reward.py
│     │  │  └─ cooperation_reward.py
│     │  ├─ configs/
│     │  │  └─ planning_configs.py
│     │  └─ README.md
│     │
│     ├─ rl/
│     │  ├─ algorithms/
│     │  │  ├─ base.py
│     │  │  ├─ td3.py
│     │  │  ├─ sac.py
│     │  │  ├─ ppo.py
│     │  │  └─ ddpg.py
│     │  ├─ exploration/
│     │  │  └─ base.py
│     │  ├─ distributions/
│     │  │  └─ action_dists.py
│     │  └─ factory.py
│     │
│     ├─ training/
│     │  ├─ trainers/
│     │  │  ├─ base_trainer.py
│     │  │  ├─ off_policy_trainer.py
│     │  │  └─ hgc_trainer.py
│     │  ├─ callbacks/
│     │  │  └─ callbacks.py
│     │  └─ orchestration/
│     │     └─ run_pipeline.py
│     │
│     ├─ env/
│     │  ├─ base_env.py
│     │  ├─ wrappers/
│     │  │  └─ gymnasium_wrapper.py
│     │  ├─ factories/
│     │  │  └─ pybullet_factory.py
│     │  └─ pybullet_drones/
│     │
│     ├─ system/
│     │  ├─ control_layer.py
│     │  ├─ planning_layer.py
│     │  └─ system_integration.py
│     │
│     ├─ configs/
│     │  ├─ base.py
│     │  ├─ default.py
│     │  └─ experiment_registry.py
│     │
│     └─ utils/
│        ├─ logger.py
│        ├─ plotting.py
│        └─ io.py
│
├─ experiments/
│  ├─ control/
│  │  ├─ train_tsa_ladrc.py
│  │  ├─ eval_tsa_ladrc.py
│  │  └─ configs/
│  ├─ planning/
│  │  ├─ train_pyramid_per.py
│  │  ├─ train_task_decomposed_actor.py
│  │  └─ configs/
│  ├─ integrated/
│  │  ├─ train_full_stack.py
│  │  └─ configs/
│  └─ legacy/
│
├─ outputs/
│  ├─ checkpoints/
│  ├─ logs/
│  ├─ figures/
│  ├─ tables/
│  └─ videos/
│
├─ papers/
│  ├─ chapters/
│  ├─ figures/
│  ├─ reference.bib
│  └─ ...
│
├─ docs/
│  ├─ REFACTOR_DIRECTORY_DESIGN.md
│  ├─ CODE_THESIS_SYNC_MAP.md
│  └─ ...
│
├─ tests/
│  ├─ control/
│  ├─ planning/
│  ├─ integration/
│  └─ smoke/
│
├─ scripts/
│  ├─ build_tex.py
│  ├─ append_bib.py
│  └─ fetch_papers.py
│
├─ main.py
├─ pyproject.toml
└─ AGENTS.md
```

## 目录设计解释

## `src/lc/control/`

对应论文第 3 章，聚焦低层控制。

应放入：

- LADRC 控制器
- RL 对 LADRC 参数的调节逻辑
- TSA 机制
- 控制层专用环境
- 控制层配置

这样做的好处是，第 3 章的代码会在一个包内闭环，而不是拆在：

- `Gym_env/`
- `Reinforce_learning/RLg/`
- `examples/`

之间。

## `src/lc/planning/`

对应论文第 4 章，聚焦高层协同规划。

应放入：

- Curriculum Learning
- Pyramid-PER
- 多层记忆池
- Task-Decomposed Actor
- Transformer 组件
- 多无人机协同规划环境
- 规划层奖励和配置

这是后续代码重构的重点区域，因为当前这部分是最分散的。

## `src/lc/rl/`

这是论文无关但项目必需的通用 RL 基础设施层。

应放入：

- PPO / TD3 / SAC / DDPG 等算法实现
- exploration 策略
- action distribution
- algorithm factory

这层应该保持“通用”，不要再放论文特定逻辑。

例如：

- `TSA-LADRC` 不应留在通用 `rl/algorithms/` 里
- `Pyramid-PER` 不应作为通用 buffer 直接暴露在最上层

它们属于论文方法层，应放在 `control/` 或 `planning/`。

## `src/lc/training/`

负责训练编排，不负责方法定义。

应放入：

- trainer 基类
- off-policy / HGC trainer
- callbacks
- 训练入口 orchestration

它的职责是“调度”，不是“定义控制方法”或“定义网络结构”。

## `src/lc/env/`

这是统一环境基础设施层。

应放入：

- 环境基类
- gymnasium wrapper
- factory
- pybullet drones 适配层

它是 control/planning 的基础支撑，而不是方法本身。

## `src/lc/system/`

对应论文中的分层系统集成表达。

应放入：

- ControlLayer
- PlanningLayer
- SystemIntegration

未来如果做完整双层系统联合仿真或联调，这里会成为主干。

## `experiments/`

后续所有实验脚本都建议迁移到这里。

按研究问题拆开：

- `experiments/control/`
- `experiments/planning/`
- `experiments/integrated/`

旧的零散脚本先移入 `experiments/legacy/`，避免混乱。

## `outputs/`

把产出和源码分离。

当前仓库里结果文件、视频、图像、npy 很容易散落在 `examples/` 或子目录中。
后续统一放到：

- checkpoint
- logs
- figures
- tables
- videos

这样论文画图、实验复现实验、结果归档都会更清晰。

## 旧目录到新目录的迁移建议

下面是第一版迁移映射。

### 控制相关

- `Gym_env/LADRC_Controller.py`
  -> `src/lc/control/controllers/ladrc.py`

- `Gym_env/gym_pybullet_drones/control/LADRC.py`
  -> `src/lc/control/controllers/ladrc.py` 或 `src/lc/env/pybullet_drones/control/ladrc_backend.py`

- `Reinforce_learning/RLg/TSA_LADRC.py`
  -> `src/lc/control/controllers/rl_ladrc_adapter.py`

- `Gym_env/gym_pybullet_drones/envs/TSA_LADRC_Env.py`
  -> `src/lc/control/envs/tsa_ladrc_env.py`

### 规划相关

- `NN/TaskDecomposedActor.py`
  -> `src/lc/planning/models/task_decomposed_actor.py`

- `NN/MultiUAVModel.py`
  -> `src/lc/planning/models/multi_uav_model.py`

- `NN/obstacle_branch.py`
  -> `src/lc/planning/models/obstacle_branch.py`

- `NN/collaborative_branch.py`
  -> `src/lc/planning/models/collaborative_branch.py`

- `NN/embeddings.py`
  -> `src/lc/planning/models/embeddings.py`

- `NN/components.py`
  -> `src/lc/planning/models/components.py`

### 记忆与课程学习

- `Reinforce_learning/buffers/PyramidPER.py`
  -> `src/lc/planning/memory/pyramid_per.py`

- `Reinforce_learning/buffers/multi_level/*`
  -> `src/lc/planning/memory/`

- `Reinforce_learning/buffers/metrics/*`
  -> `src/lc/planning/memory/metrics/`

- `Reinforce_learning/buffers/filters/*`
  -> `src/lc/planning/memory/filters/`

- `Reinforce_learning/buffers/samplers/*`
  -> `src/lc/planning/memory/samplers/`

- `Trainer/curriculum/*`
  -> `src/lc/planning/curriculum/`

### 通用 RL

- `Reinforce_learning/Basealgos.py`
  -> `src/lc/rl/algorithms/base.py`

- `Reinforce_learning/algo_factory.py`
  -> `src/lc/rl/factory.py`

- `Reinforce_learning/RLg/PPO.py`
  -> `src/lc/rl/algorithms/ppo.py`

- `Reinforce_learning/RLg/TD3.py`
  -> `src/lc/rl/algorithms/td3.py`

- `Reinforce_learning/RLg/SAC.py`
  -> `src/lc/rl/algorithms/sac.py`

- `Reinforce_learning/RLg/DDPG.py`
  -> `src/lc/rl/algorithms/ddpg.py`

### 训练器

- `Trainer/BaseTrainer.py`
  -> `src/lc/training/trainers/base_trainer.py`

- `Trainer/OffPolicyTrainer.py`
  -> `src/lc/training/trainers/off_policy_trainer.py`

- `Trainer/HGCTrainer.py`
  -> `src/lc/training/trainers/hgc_trainer.py`

- `Trainer/callbacks.py`
  -> `src/lc/training/callbacks/callbacks.py`

### 系统层

- `core_architecture/ControlLayer.py`
  -> `src/lc/system/control_layer.py`

- `core_architecture/PlanningLayer.py`
  -> `src/lc/system/planning_layer.py`

- `core_architecture/SystemIntegration.py`
  -> `src/lc/system/system_integration.py`

### 环境基础设施

- `Gym_env/BaseEnv.py`
  -> `src/lc/env/base_env.py`

- `Gym_env/wrappers/GymnasiumWrapper.py`
  -> `src/lc/env/wrappers/gymnasium_wrapper.py`

- `Gym_env/factories/PyBulletDronesFactory.py`
  -> `src/lc/env/factories/pybullet_factory.py`

## 重构阶段建议

不要一次性物理迁移全部代码，建议分阶段进行。

### 阶段 1：先建立新骨架，不迁移逻辑

先创建目录和空 `__init__.py`，并写清每层职责。

目标：

- 确立未来结构
- 不影响现有训练流程
- 为后续迁移提供落点

### 阶段 2：先迁移论文主方法

优先迁移：

1. `TSA-LADRC`
2. `Pyramid-PER`
3. `Task-Decomposed Actor`

原因：

- 它们是论文主贡献
- 后续代码与论文同步最依赖这三块

### 阶段 3：迁移通用基础设施

包括：

- 通用 trainer
- 通用算法
- wrappers / factory
- 配置系统

### 阶段 4：清理历史目录

在新目录稳定后，再考虑：

- 删除重复实现
- 归档 legacy 脚本
- 替换 import 路径
- 补测试

## 推荐下一步

最合理的下一步不是立刻大搬家，而是先做两件事：

1. 创建新目录骨架
2. 做一份“旧文件 -> 新路径”的迁移清单

完成这两步后，才能开始安全地逐块迁移代码。

## 与论文的直接映射

后续建议固定以下映射关系：

- 第 3 章 -> `src/lc/control/`
- 第 4 章 -> `src/lc/planning/`
- 分层架构总述 -> `src/lc/system/`
- 实验章节 -> `experiments/`
- 图表与论文结果 -> `outputs/figures/` 和 `papers/`

这样以后不管是改论文还是改代码，定位都会快很多。

## 建议的下一份文档

完成目录设计后，下一步建议建立：

- `docs/REFACTOR_MIGRATION_MAP.md`

内容包括：

- 当前文件路径
- 新文件路径
- 是否保留
- 是否归档
- 是否需要重写
- 对应论文章节

这会成为后续真正执行重构迁移的操作清单。
