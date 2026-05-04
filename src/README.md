# src/lc Code Readme

## 1. 文档目的

这份文档用于统一说明 `src/lc/` 主链代码当前的组织方式、章节归属、模块职责、实现状态，以及后续需要继续修正的第三章与第四章内容。

本仓库当前应以 `src/lc/` 作为主实现链来理解：

- 论文是方法规格
- `src/lc/` 是方法落地后的主代码
- `Gym_env/`、`NN/`、`Reinforce_learning/` 等旧目录主要作为历史参考，不应再被当作默认主链

## 2. 顶层结构

`src/lc/` 当前主要包含以下模块：

- `analysis/`
- `common/`
- `control/`
- `entrypoints/`
- `envs/`
- `integration/`
- `planning/`
- `rl/`

其中最重要的章节对应关系是：

- 第三章：`src/lc/control/`
- 第四章：`src/lc/planning/`
- 第三章与第四章桥接：`src/lc/integration/`

## 3. 模块总览

### 3.1 `src/lc/common/`

通用基础设施层，给控制、规划和桥接模块提供公共能力。

主要子目录：

- `config/`：通用配置定义
- `io/`：文件读写、导出与序列化支持
- `registry/`：模块注册与工厂能力
- `types/`：公共类型声明
- `utils/`：随机种子、路径、数值工具等

作用：

- 减少第三章和第四章之间的重复工具代码
- 提供统一的数据结构与通用辅助逻辑

### 3.2 `src/lc/envs/`

全局环境抽象层，不专属于某一章。

主要子目录：

- `base/`：基础环境协议
- `generators/`：环境或样本生成工具
- `gymnasium/`：Gymnasium 相关兼容层
- `metrics/`：通用控制/规划指标计算
- `pybullet_uav/`：PyBullet 无人机相关环境支持
- `scenarios/`：场景配置与难度预设

作用：

- 为控制和规划提供统一的场景、指标与基础环境能力

### 3.3 `src/lc/rl/`

强化学习算法基础层。

主要子目录：

- `algorithms/`
- `buffers/`
- `core/`
- `exploration/`

作用：

- 为第三章控制层 RL 和第四章规划层 RL 提供可复用的算法组件
- 当前已包含 `mDDPG` 等控制侧算法基础，以及规划侧训练所需的公共 RL 能力

### 3.4 `src/lc/analysis/`

结果分析与报告层。

主要子目录：

- `compare/`
- `plotting/`
- `reports/`

作用：

- 面向实验结果整理、指标比较和报告输出
- 当前仍偏轻量，更多工作还集中在章节目录内部的 plotting/experiments 中

### 3.5 `src/lc/entrypoints/`

统一命令行入口层。

作用：

- 暴露训练、对比实验、PyBullet 对比等可运行入口
- 降低直接手写脚本调用的成本

### 3.6 `src/lc/integration/`

第三章控制层与第四章规划层的桥接层。

主要子目录：

- `experiments/`
- `interfaces/`
- `pipeline/`
- `plotting/`

作用：

- 将规划层输出映射到控制层接口
- 支持章节间联调与示范实验

当前状态：

- 已有桥接框架和基础实验入口
- 仍属于“接口打通中”的阶段，不是最终完整系统闭环

## 4. 第三章代码解读：`src/lc/control/`

### 4.1 当前定位

第三章主问题是控制层实现，论文口径应理解为：

- 控制器主体仍然是 `LADRC`
- 强化学习负责在线调参，而不是替代控制器本体
- 核心目标是鲁棒性、抗扰性、恢复能力、平滑性和参数自适应

### 4.2 目录组成

`src/lc/control/` 当前包含：

- `baselines/`
- `configs/`
- `controllers/`
- `envs/`
- `experiments/`
- `io/`
- `plotting/`
- `policies/`
- `reference_generators/`
- `rewards/`
- `simulators/`
- `trainers/`

### 4.3 主要职责

#### `configs/`

负责第三章实验配置。

当前包括：

- 控制实验基础配置
- PyBullet 控制实验配置
- 单轴 `LADRC` 参数整定配置

#### `controllers/`

负责控制器实现与适配。

当前核心包括：

- `pid.py`：轻量 PID 控制器
- `ladrc.py`：基础 LADRC 控制器
- `adaptive_ladrc.py`：RL 增强 LADRC 参数接口
- `ladrc_channels.py`：多轴 LADRC 参数组织与默认参数集
- `pybullet_variants.py`：PyBullet 下的控制器组合与单轴替换逻辑

其中当前第三章最关键的修正是：

- PyBullet 控制器已经从“近似手写映射”切回到“原生 `DSLPIDControl` + 单轴位置环 LADRC 替换”
- 单轴固定参数现在显式使用用户原代码中的 `x/y/z` LADRC 参数

#### `envs/`

负责控制环境封装。

当前包括：

- PyBullet 评测/训练环境：如 `pybullet_axis_env.py`、`pybullet_eval_env.py`

#### `reference_generators/`

负责参考轨迹构造。

当前重点是：

- 单轴分段匀速递推位置参考轨迹
- 支持 `x/y/z` 三轴分别构造单轴场景

#### `simulators/`

负责一次控制实验/episode 的底层执行。

当前最重要的是：

- `pybullet_runner.py`

它承担：

- 对接真实 `gym_pybullet_drones`
- 控制频率与 RL 频率解耦
- 时序数据记录

#### `policies/`

负责第三章控制侧 RL 策略封装。

当前包含：

- `mddpg_control.py`
- 状态堆叠与动作保持支持

当前动作口径已统一为：

- `action[0] -> b0`
- `action[1] -> wc`
- `action[2] -> k`

并在控制器内部恢复：

- `omega_o = k * omega_c`

#### `trainers/`

负责训练流程、参数整定和实验组织。

当前包括：

- PyBullet 单轴训练/评测/整定 trainer

#### `experiments/`

负责实验编排入口。

当前包括：

- 控制对比实验
- 多难度泛化实验
- PyBullet 对比实验
- 单轴 LADRC 参数整定实验

#### `plotting/`

负责第三章图表输出。

当前支持：

- 训练曲线
- 控制指标对比
- PyBullet 单轴响应图
- 热力图
- 敏感性分析图

### 4.4 当前第三章完整度判断

从“是否具备一条可运行主链”来看，第三章已经基本成立。

当前已经有：

- PyBullet 控制主链
- `PID / LADRC / DDPG-LADRC / mDDPG-LADRC` 对比结构
- PyBullet 单轴递推轨迹实验链
- 单轴 LADRC 参数整定实验
- 图表和结构化输出
- 训练/评测/CLI 入口

但它还不是最终完成版。

### 4.5 第三章后续需要修正的重点

后续建议继续修正以下内容：

1. 固定参数 `LADRC` 整定逻辑还需要更严肃

- 当前虽然已经接回真实 `DSLPIDControl` 基线
- 但 `LADRC` 搜索范围、评分规则和验证工况仍需要进一步工程化

2. `RL bounds` 的尺度仍需和真实参数量级重新对齐

- 当前 `b0 / wc / k` 的可学习范围推导还带有旧尺度残留
- 需要完全按真实 PyBullet 参数尺度重新定义

3. 第三章正式实验口径固定为 PyBullet

- 原轻量 `tracking.py` 环境已退役，不再作为第三章底层控制主环境
- 正式结论应基于 PyBullet 真实结果

4. 第三章最终图表仍需筛选

- 当前生成的图比较多
- 还需要收敛出适合最终实验汇报的一组核心图

5. 多随机种子统计和正式实验矩阵仍需补齐

- 当前 smoke 和单次结果较多
- 论文级结论仍需均值、方差和多难度统计

## 5. 第四章代码解读：`src/lc/planning/`

### 5.1 当前定位

第四章主问题是规划层实现，当前论文口径应理解为：

- 主模型是 `MultiUAVModel`
- 输入是 `self_state / obstacles / neighbors`
- 语义是“两阶段：先避障，后协同”
- 基线模型只作为对比，不替代主方法

### 5.2 目录组成

`src/lc/planning/` 当前包含：

- `baselines/`
- `configs/`
- `critics/`
- `curriculum/`
- `encoders/`
- `envs/`
- `experiments/`
- `memory/`
- `models/`
- `plotting/`
- `rewards/`
- `trainers/`

### 5.3 主要职责

#### `models/`

规划层 actor 主体。

当前核心是：

- `multi_uav_model.py`

职责：

- 融合 `self_state / obstacles / neighbors`
- 表达“先安全，再协同”的结构化决策语义

#### `encoders/`

负责不同输入分支编码。

作用：

- 将自状态、障碍物、邻机信息编码成统一特征

#### `critics/`

负责规划层 critic。

当前核心是结构化 critic，用于评估动作质量和训练 actor。

#### `memory/`

负责经验回放系统。

当前重点是：

- 金字塔经验池/多层回放
- 稀有样本与关键样本保留

#### `curriculum/`

负责训练课程调度。

作用：

- 管理 guidance / avoidance / cooperation 等阶段
- 按训练表现推进或回退

#### `rewards/`

负责规划层奖励拆分。

当前主要包括：

- 到达目标
- 避障
- 协同
- 平滑性
- 恢复项

#### `envs/`

负责规划层环境封装。

当前更偏抽象多 UAV 规划环境，而不是高保真联合物理仿真。

#### `trainers/`

负责 actor-critic 主训练流程。

当前核心是：

- `planning_trainer.py`

职责包括：

- 训练闭环
- curriculum 切换
- memory 采样
- actor/critic 更新
- 训练日志记录

#### `experiments/`

负责第四章实验组织。

当前已经支持：

- 主方法与基线对比
- 若干机制消融
- 输出统一指标和图表

#### `plotting/`

负责第四章论文化图表输出。

### 5.4 当前第四章完整度判断

第四章已经具备相当完整的代码骨架，并且很多训练逻辑已经落地，不再只是空目录。

当前已具备：

- 多输入主模型
- 结构化 critic
- curriculum 训练逻辑
- 经验回放系统
- 奖励拆分
- 实验入口
- 绘图输出

但第四章距离“最终高质量完稿版本”也还差几块关键内容。

### 5.5 第四章后续需要修正的重点

后续建议继续修正以下内容：

1. 规划环境仍偏抽象

- 当前环境更适合方法验证
- 若要更强工程说服力，需要进一步接近真实多机任务设定

2. Actor/Critic 与论文语义仍需继续逐项对齐

- 需要继续核对网络分支、交互方式和阶段语义是否完全贴合论文设定

3. memory/curriculum 的论文级实验还可以更完整

- 当前已有实现，但还需要更系统的消融和统计

4. 第四章实验结果组织还需继续规范化

- 输出已经较丰富
- 但最终还需要更统一的实验矩阵、种子统计和标准图表集

5. 与控制层的接口还需更严格定义

- 规划输出到控制输入的映射目前仍偏轻量桥接
- 后续要为第三章与第四章联调提供更稳定接口

## 6. 桥接代码解读：`src/lc/integration/`

桥接层当前主要负责：

- 统一第三章和第四章接口
- 将规划输出映射到控制层执行
- 支持章节联调演示

当前状态：

- 已有结构
- 已有基础实验
- 还不算完整成型的最终系统链

后续主要需要：

- 收紧接口协议
- 提高控制层与规划层之间的时序一致性
- 增加完整联调 smoke 与结果分析

## 7. 当前主链的整体判断

现在的 `src/lc/` 已经不是“只有框架”的阶段，而是：

- 第三章主链可运行
- 第四章主链可训练
- 桥接层已具备基础联调能力

但当前仍属于“研究实现持续收敛中”的阶段，不应误解为所有章节都已经最终定型。

## 8. 后续开发优先级建议

建议后续按下面顺序推进：

1. 继续把第三章 PyBullet 控制实验做稳

- 固定参数 `LADRC`
- RL 参数边界
- 多种子统计
- 最终对比图筛选

2. 继续补强第四章主方法实验闭环

- 更强的环境设定
- 更完整的消融矩阵
- 更稳的训练统计

3. 再加强第三章和第四章桥接

- 统一接口
- 做章节联调
- 补集成实验图表

## 9. 推荐阅读顺序

如果后续继续开发，建议按这个顺序理解代码：

### 第三章

1. `src/lc/control/controllers/`
2. `src/lc/control/configs/`
3. `src/lc/control/reference_generators/`
4. `src/lc/control/simulators/`
5. `src/lc/control/trainers/`
6. `src/lc/control/experiments/`

### 第四章

1. `src/lc/planning/models/`
2. `src/lc/planning/critics/`
3. `src/lc/planning/rewards/`
4. `src/lc/planning/memory/`
5. `src/lc/planning/curriculum/`
6. `src/lc/planning/trainers/`
7. `src/lc/planning/experiments/`

### 桥接

1. `src/lc/integration/interfaces/`
2. `src/lc/integration/pipeline/`
3. `src/lc/integration/experiments/`

## 10. 一句话结论

`src/lc/` 现在已经形成了“控制层、规划层、桥接层”三部分组成的论文实现主链；下一阶段最重要的不是再扩目录，而是把第三章控制实验做稳、把第四章实验做完整、再把两章真正桥接起来。
