# Chat3 Readme

## 1. 第三章的统一定位

第三章在本仓库里只有一个主链，不存在“两个平行主链”。

唯一的主链是：

- `src/lc/control/`

这条主链承载的是第三章控制层方法实现，也就是：

- 单轴位置跟踪与扰动抑制
- `PID / LADRC / DDPG-LADRC / mDDPG-LADRC` 对比
- RL 在线调节 `LADRC` 参数
- 训练、评测、图表、实验输出

之前提到的 PyBullet 相关内容，不应理解为“第二条主链”，而应理解为：

- 第三章主链中的一个实验子模块
- 用于把第三章控制层方法接到更接近无人机接口的仿真执行层

所以正确理解是：

- 主链只有一个：`src/lc/control/`
- PyBullet 是这条主链中的实验扩展，不是独立章节链路

---

## 2. 第三章现在的代码组成

第三章当前代码可以按功能拆成下面几部分。

### 2.1 配置层

位置：

- `src/lc/control/configs/control_config.py`
- `src/lc/control/configs/pybullet_control_config.py`

作用：

- 定义第三章基础实验配置
- 定义 PyBullet 分轴实验配置
- 管理训练轮数、评测轮数、难度、输出目录、频率等参数

### 2.2 控制器层

位置：

- `src/lc/control/controllers/pid.py`
- `src/lc/control/controllers/ladrc.py`
- `src/lc/control/controllers/adaptive_ladrc.py`
- `src/lc/control/controllers/ladrc_channels.py`
- `src/lc/control/controllers/pybullet_variants.py`

作用：

- `pid.py`
  提供第三章的 PID 基线控制器

- `ladrc.py`
  提供第三章的基础 LADRC 控制器

- `adaptive_ladrc.py`
  提供 RL 调参包装器
  当前 RL 动作已经改成输出：
  - `b0`
  - `wc`
  - `k = wo / wc`

  并由控制器内部恢复：
  - `wo = k * wc`

- `ladrc_channels.py`
  用于统一管理多通道 LADRC 参数集

- `pybullet_variants.py`
  提供 PyBullet 实验中的三种控制器组合：
  - `pid_pos_att`
  - `ladrc_pos_pid_att`
  - `ladrc_pos_att`

### 2.3 环境层

位置：

- `src/lc/control/envs/tracking.py`
- `src/lc/control/envs/pybullet_axis_env.py`
- `src/lc/control/envs/pybullet_eval_env.py`

作用：

- `tracking.py`
  是第三章的轻量主训练环境
  用于单轴位置跟踪与扰动抑制

- `pybullet_axis_env.py`
  是第三章 PyBullet 分轴 RL 训练环境
  用于 `x / y / z` 三轴分别训练

- `pybullet_eval_env.py`
  用于统一评测三类控制器在同一批参考轨迹上的表现

### 2.4 RL 策略层

位置：

- `src/lc/control/policies/mddpg_control.py`
- `src/lc/control/policies/stacking.py`

作用：

- `mddpg_control.py`
  封装第三章 RL-LADRC 智能体
  目前无论 `DDPG-LADRC` 还是 `mDDPG-LADRC`，动作语义都统一为：
  - `b0`
  - `wc`
  - `k`

- `stacking.py`
  提供 `mDDPG` 所需的状态堆叠能力

### 2.5 训练器层

位置：

- `src/lc/control/trainers/control_trainer.py`
- `src/lc/control/trainers/pybullet_axis_trainer.py`

作用：

- `control_trainer.py`
  是第三章主训练器
  负责：
  - `PID / LADRC / DDPG-LADRC / mDDPG-LADRC`
  - 训练日志
  - 消融实验
  - 对比实验

- `pybullet_axis_trainer.py`
  是第三章 PyBullet 分轴实验训练器
  负责：
  - `x / y / z` 分轴训练
  - 三类控制器评测
  - 结构化实验输出

### 2.6 实验编排层

位置：

- `src/lc/control/experiments/compare.py`
- `src/lc/control/experiments/pybullet_compare.py`
- `experiments/control/run_chapter3_experiment.py`

作用：

- `compare.py`
  是第三章主实验入口
  支持：
  - 主方法对比
  - 机制消融
  - 多难度泛化

- `pybullet_compare.py`
  是第三章 PyBullet 实验入口
  支持：
  - 分轴训练
  - 控制器基准对比
  - 全流程实验

- `run_chapter3_experiment.py`
  是项目级的控制实验入口脚本

### 2.7 仿真执行与参考轨迹层

位置：

- `src/lc/control/reference_generators/piecewise_velocity.py`
- `src/lc/control/simulators/pybullet_runner.py`

作用：

- `piecewise_velocity.py`
  生成第三章 PyBullet 子模块所需的递推参考轨迹
  轨迹口径是：
  - 多阶段
  - 匀速
  - 直线
  - 分轴

- `pybullet_runner.py`
  统一封装 PyBullet 或回退仿真执行过程

### 2.8 输出与绘图层

位置：

- `src/lc/control/io/pybullet_artifacts.py`
- `src/lc/control/plotting/plots.py`
- `src/lc/control/plotting/pybullet_plots.py`

作用：

- `plots.py`
  输出第三章主链图表

- `pybullet_plots.py`
  输出第三章 PyBullet 子模块图表

- `pybullet_artifacts.py`
  统一保存：
  - `summary.json`
  - `metrics.csv`
  - `timeseries.csv`
  - `reference.csv`
  - `figures/*`
  同时兼容旧 Logger 风格导出

### 2.9 测试层

位置：

- `tests/test_new_architecture.py`
- `tests/test_chapter3_action_parameterization.py`
- `tests/test_chapter3_pybullet_reference.py`
- `tests/test_chapter3_pybullet_protocol.py`

作用：

- 验证第三章主实验输出是否存在
- 验证 RL 动作参数化是否正确
- 验证递推参考轨迹生成是否正确
- 验证 PyBullet 协议、目录、图表与日志输出是否正常

---

## 3. 第三章当前已经实现了什么

第三章目前已经具备以下能力。

### 3.1 方法层

- `PID`
- `LADRC`
- `DDPG-LADRC`
- `mDDPG-LADRC`

并且：

- RL 不直接输出控制量
- RL 负责在线调节 `LADRC`
- 当前 RL 动作定义为：
  - `b0`
  - `wc`
  - `k`

### 3.2 环境与任务层

- 单轴位置跟踪
- 扰动抑制
- 多难度轻量控制环境
- `x / y / z` 三轴分轴递推参考轨迹场景

### 3.3 实验层

- 四方法主对比
- `DDPG-LADRC vs mDDPG-LADRC`
- 三项增强机制消融
- 多难度泛化实验
- PyBullet 分轴控制器对比实验

### 3.4 输出层

- `summary.json`
- `metrics.csv`
- `training_ddpg.csv`
- `training_mddpg.csv`
- `ablation_metrics.csv`
- `checkpoints/*.pt`
- 图表输出

### 3.5 图表层

已经支持生成：

- 训练奖励曲线
- actor loss 曲线
- critic loss 曲线
- 时域响应图
- 误差图
- 控制量图
- 控制器对比图
- 分轴轨迹图

---

## 4. 第三章当前完整度判断

如果从“是否已经形成完整章节代码链”来看，第三章已经不是骨架，而是一套可运行的章节实现。

可以这样判断：

- 方法主链完整度：`80%`
- 工程可运行完整度：`85%`
- 论文级实验完整度：`65%`
- PyBullet 实验子模块完整度：`65%`

这意味着：

- 第三章已经可以运行
- 已经有训练、评测、图表、输出
- 已经可以做方法对比和分轴控制实验

但它还没有达到“答辩/论文最终版就绪”的程度

---

## 5. 第三章当前最缺什么

第三章接下来最需要补强的，不是继续简单加文件，而是下面这些关键能力。

### 5.1 固定参数 LADRC 本体还不够强

这是最重要的问题。

当前虽然 RL-LADRC 已经形成链路，但如果固定参数 `LADRC` 本身表现明显不如 `PID`，那么：

- RL 训练结论会变弱
- 方法优势会不扎实
- 第三章会更像“RL 补救控制器”，而不是“RL 增强控制器”

因此下一步必须继续加强：

- `LADRC` 参数整定
- `b0 / wc / k` 口径下的参数边界
- `x / y / z` 三轴参数分离
- 固定参数基线与 PID 的公平比较

### 5.2 主链训练任务与 PyBullet 场景还没有完全统一

当前：

- 主链轻量环境还是单轴抽象环境
- PyBullet 子模块已经是 `x / y / z` 分轴递推轨迹

所以第三章还需要进一步统一任务口径：

- 把主链训练环境也逐步对齐到分轴场景
- 统一参考轨迹定义
- 统一动作语义
- 统一评估指标

### 5.3 多随机种子统计还不够

目前可以跑实验，但统计层面还不够强。

第三章还应补：

- 多随机种子重复实验
- 均值与标准差
- 置信区间
- 训练稳定性统计图

否则论文级说服力还不够。

### 5.4 真实 PyBullet 深度接入还不够

当前 PyBullet 子模块已经建立了实验协议和输出结构，但还存在一个现实情况：

- 当前环境若缺依赖，会走回退仿真逻辑

所以第三章后续还需要：

- 确认真正使用 `gym_pybullet_drones` 跑通完整实验
- 跑通真实 backend 下的 `train / eval / full`
- 对齐真实状态量与控制器输出
- 明确真实 PyBullet 实验结果

### 5.5 第三章最终主结论还需要收敛

当前第三章已经有：

- 四方法主链对比
- 三类 PyBullet 控制器组合

但最后还需要明确：

- 第三章论文主结论到底以哪组实现为主
- 工程主实现是否以 `LADRC位置 + PID姿态` 为主
- `LADRC位置 + LADRC姿态` 是否作为完整对照而不是默认主方法

这一点需要后续在实验整理时明确下来。

---

## 6. 现在对第三章的正确理解

第三章现在不是“两条主链”，而是“一条主链 + 一个实验扩展模块”。

正确理解应当是：

- 主链：
  - `src/lc/control/`

- 其中包含：
  - 轻量训练环境
  - 方法对比与消融
  - RL-LADRC 主方法
  - PyBullet 分轴实验扩展

也就是说：

- 第三章只有一个实现主链
- PyBullet 是第三章主链中的实验执行扩展
- 它不是第二条独立章节代码线

---

## 7. 第三章后续建议优先级

如果按实际研发优先级，建议下一步这样推进：

1. 先把固定参数 `LADRC` 调强  
   目标：至少在若干工况下不明显弱于 `PID`

2. 把第三章主链训练任务统一到分轴递推轨迹口径  
   目标：主链和 PyBullet 子模块不再各讲各的任务

3. 做多随机种子统计与统一实验矩阵  
   目标：把第三章结果变成论文级实验结果

4. 跑通真实 `gym_pybullet_drones` backend  
   目标：让 PyBullet 子模块不只是协议完整，而是真实闭环完整

---

## 8. 一句话总结

第三章现在已经形成了一套可运行、可训练、可评测、可绘图的控制层实现主链；它只有一个主链，就是 `src/lc/control/`，PyBullet 部分只是这条主链中的实验扩展模块。当前最需要补强的，是固定参数 `LADRC` 的质量、主链任务口径统一、多随机种子统计，以及真实 PyBullet 仿真闭环。
