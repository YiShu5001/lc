# 新生项目代码集介绍

## 1. 文档定位

这份文档用于介绍当前 `lc` 仓库正在形成的“新生项目代码集”。

这里的“新生项目代码集”不是一个已经完全独立的新仓库，而是指：

- 在当前仓库内部，以 `src/lc/` 为核心逐步重建的新主代码体系；
- 旧目录继续保留，用于兼容、过渡和回溯；
- 代码、实验、论文三条线围绕同一套方法结构逐步收敛。

这份文档的目标不是写历史，而是说明“现在这套新代码集已经到了什么程度、有哪些入口、后续该怎么继续接着做”。

## 2. 当前项目的核心目标

当前项目围绕两条主研究线展开：

### 第三章主线

控制层主线，核心是：

- LADRC
- RL-LADRC / TSA-LADRC
- 控制参数动态调节
- 控制实验接口与轨迹跟踪

### 第四章主线

规划层主线，核心是：

- Task-Decomposed Actor
- MultiUAVModel
- Pyramid-PER
- MultiLevelBuffer
- curriculum learning
- 从单无人机到多无人机的协同演化

### 联动要求

这不是“代码仓库”和“论文仓库”并排存在，而是一个必须代码和论文双同步的研究工程：

- 改方法，要能落到代码模块；
- 改代码，要能回指论文章节；
- 新实验，要同时有实验入口、接口层和论文映射。

## 3. 新生项目代码集的主结构

当前新主结构如下：

```text
src/lc/
  control/
  planning/
  rl/
  training/
  env/
  system/
  configs/
  utils/

experiments/
  control/
  planning/
  integrated/
  legacy/

tests/
  smoke/
  control/
  planning/
  integration/

outputs/
  checkpoints/
  logs/
  figures/
  tables/
  videos/
```

## 4. 当前已经落地的核心模块

### 控制层

当前已落地：

- [chapter3_interfaces.py](C:/context_mine/mine_code/GIT_Projects/lc/src/lc/control/chapter3_interfaces.py)

提供的能力：

- 第三章实验配置对象
- LADRC 控制器组装函数
- 第三章控制实验接口
- 参考轨迹构造函数

### 规划层模型

当前已落地：

- [task_decomposed_actor.py](C:/context_mine/mine_code/GIT_Projects/lc/src/lc/planning/models/task_decomposed_actor.py)
- [multi_uav_model.py](C:/context_mine/mine_code/GIT_Projects/lc/src/lc/planning/models/multi_uav_model.py)
- [obstacle_branch.py](C:/context_mine/mine_code/GIT_Projects/lc/src/lc/planning/models/obstacle_branch.py)
- [collaborative_branch.py](C:/context_mine/mine_code/GIT_Projects/lc/src/lc/planning/models/collaborative_branch.py)
- [embeddings.py](C:/context_mine/mine_code/GIT_Projects/lc/src/lc/planning/models/embeddings.py)
- [components.py](C:/context_mine/mine_code/GIT_Projects/lc/src/lc/planning/models/components.py)

### 规划层实验接口

当前已落地：

- [chapter4_interfaces.py](C:/context_mine/mine_code/GIT_Projects/lc/src/lc/planning/chapter4_interfaces.py)

提供的能力：

- 第四章实验配置对象
- actor + replay memory 的实验 bundle
- 第四章规划前向接口

### 记忆池主结构

当前已落地：

- [pyramid_per.py](C:/context_mine/mine_code/GIT_Projects/lc/src/lc/planning/memory/pyramid_per.py)
- [multi_level_buffer.py](C:/context_mine/mine_code/GIT_Projects/lc/src/lc/planning/memory/multi_level_buffer.py)
- [memory_configs.py](C:/context_mine/mine_code/GIT_Projects/lc/src/lc/planning/configs/memory_configs.py)

### 记忆池内层子模块

这一轮已经进入新路径：

- `filters/`
- `metrics/`
- `pools/`
- `samplers/`
- `sum_tree.py`

也就是说，`planning/memory` 这一条链现在已经不只是壳子，而是开始形成完整内部结构。

### 环境演化接口

当前已落地：

- [evolution_scenarios.py](C:/context_mine/mine_code/GIT_Projects/lc/src/lc/env/evolution_scenarios.py)

提供的能力：

- 课程式场景定义
- 障碍物难度渐进
- 从单无人机到多无人机的阶段进化

### 第三章与第四章桥接层

当前已落地：

- [chapter34_bridge.py](C:/context_mine/mine_code/GIT_Projects/lc/src/lc/system/chapter34_bridge.py)

提供的能力：

- 第三章控制结果接口
- 第四章规划输入接口
- 控制层到规划层的统一桥接管线

## 5. 当前已有的实验入口

### 第三章实验入口

- [run_chapter3_experiment.py](C:/context_mine/mine_code/GIT_Projects/lc/experiments/control/run_chapter3_experiment.py)

### 第四章实验入口

- [run_chapter4_experiment.py](C:/context_mine/mine_code/GIT_Projects/lc/experiments/planning/run_chapter4_experiment.py)

### 三四章联调入口

- [run_chapter34_demo.py](C:/context_mine/mine_code/GIT_Projects/lc/experiments/integrated/run_chapter34_demo.py)

这三个脚本我已经实际跑通过，当前能够作为后续开发的统一入口使用。

## 6. 当前已有的测试入口

### Smoke 测试

- [test_chapter34_interfaces.py](C:/context_mine/mine_code/GIT_Projects/lc/tests/smoke/test_chapter34_interfaces.py)
- [run_key_smoke_tests.py](C:/context_mine/mine_code/GIT_Projects/lc/tests/smoke/run_key_smoke_tests.py)

当前 smoke 测试已通过，覆盖内容包括：

- 第三章接口
- 第四章接口
- 课程式环境进化
- 第三章与第四章桥接流程

## 7. 旧路径与新路径的关系

当前仓库仍然处于“迁移中”状态，因此必须把文件分成三类来理解。

### A. 新主实现

这些文件是后续真正应继续演进的主代码：

- `src/lc/control/*`
- `src/lc/planning/models/*`
- `src/lc/planning/memory/*`
- `src/lc/planning/configs/*`
- `src/lc/system/chapter34_bridge.py`

### B. 兼容桥

这些文件保留旧路径，但已经把导出或实现指向新路径：

- `NN/BaseNN.py`
- `NN/TaskDecomposedActor.py`
- `NN/MultiUAVModel.py`
- `Reinforce_learning/buffers/PyramidPER.py`
- `Reinforce_learning/buffers/MultiLevelBufferConfig.py`
- `Reinforce_learning/buffers/multi_level/MultiLevelBuffer.py`
- `Reinforce_learning/buffers/filters/*`
- `Reinforce_learning/buffers/metrics/*`
- `Reinforce_learning/buffers/multi_level/BaseCoveragePool.py`
- `Reinforce_learning/buffers/multi_level/DifficultyFocusPool.py`
- `Reinforce_learning/buffers/multi_level/KeyEventPool.py`

### C. 仍未完成迁移的旧主逻辑

这些目录还没有被新结构完全接管：

- `Trainer/*`
- `Gym_env/*`
- `core_architecture/*`
- `Reinforce_learning/RLg/*`

## 8. 当前完整度判断

根据当前代码状态，可以粗略理解为：

- `planning/models`：第一批核心实现已成型
- `planning/memory`：已经进入完整链路迁移阶段
- `control`：已有章节实验接口，但底层 env 仍依赖旧路径
- `system`：已有三四章桥接层，但不是最终系统集成层
- `training`：新目录骨架已建立，但主 trainer 还未迁入
- `env`：已有课程式场景接口，但尚未正式接管旧环境工厂

换句话说：

现在已经具备“按章节和实验继续开发”的能力，但还没有完成“旧工程彻底退场”的能力。

## 9. 这套新生项目代码集现在能做什么

当前这套代码集已经能承担以下任务：

- 第三章控制实验接口开发
- 第四章规划实验接口开发
- 课程式场景难度设计
- 单无人机到多无人机进化接口设计
- 第三章与第四章的联调桥接
- smoke 测试验证
- 论文与代码映射的增量更新

## 10. 当前还没有完全做完的部分

有几类问题我这里明确标出来，避免误判进度。

### 训练层未迁完

`src/lc/training/` 还没有真正接管：

- BaseTrainer
- OffPolicyTrainer
- HGCTrainer
- callbacks

### 环境层未接完

虽然已有课程式环境演化接口，但还没正式接入旧：

- PyBullet 工厂
- TSA-LADRC 环境
- 多无人机真实仿真环境

### 系统层仍是桥接阶段

当前的 [chapter34_bridge.py](C:/context_mine/mine_code/GIT_Projects/lc/src/lc/system/chapter34_bridge.py) 是章节级打通接口，不是最终系统级控制-规划-环境闭环。

### memory 还有一个已知风险

`MultiLevelBuffer.filter_and_transfer()` 当前还没有“已转移去重”机制，多次调用会重复把同一批经验往后一级池里转。这不影响现在的实验入口和 smoke 测试，但在训练器迁移前需要修。

## 11. 后续最合理的推进顺序

接下来建议按下面顺序继续推进：

1. 迁 `src/lc/training/`
2. 把新章节接口接入训练主入口
3. 把课程式场景接入环境工厂
4. 让第三章控制环境和第四章规划环境形成正式联调流程
5. 最后再收 `system` 和 `legacy`

## 12. 后续开发时的使用原则

后续开发时，建议默认遵循这几条规则：

### 规则 1

新增代码优先写到 `src/lc/`。

### 规则 2

旧路径优先做兼容桥，不优先扩写新逻辑。

### 规则 3

每次改动尽量围绕一个小链路推进：

- 一个模型链
- 一个 memory 链
- 一个 trainer 链
- 一个 env 链
- 一个章节实验链

### 规则 4

每次修改都要检查它在论文中的落点：

- 第三章控制
- 第四章规划
- 课程学习
- Pyramid-PER
- Task-Decomposed Actor
- 系统联调

## 13. 一句话总结

当前的“新生项目代码集”已经不是空目录，也不只是目录设计稿，而是一套已经具备章节接口、实验入口、测试入口、课程式场景和章节桥接能力的可继续演化的研究代码主结构；它已经能承接后续重构，但还没有完全替代旧工程。
