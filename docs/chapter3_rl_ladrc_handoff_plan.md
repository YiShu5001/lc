# 第三章 RL-LADRC 对接实施文档

## 1. 目标

本文件用于把当前已经完成的单轴 `LADRC` 固定参数整定结果，对接到第三章主链强化学习实现中，供另一个线程直接执行。

本轮对接目标固定为：

- 保留第三章论文口径：`RL` 只做 `LADRC` 在线调参，不直接替代控制器
- 统一参数口径为 `b0 / wc / k`
- 将当前手工整定结果作为 `RL` 的初始参数与范围来源
- 明确主链 RL 与 `PyBullet` 整定模块之间的接口关系

## 2. 当前事实与来源

### 2.1 已完成的固定参数来源

当前固定参数整定模块位于：

- [src/control/Tuning_ladrc](/C:/context_mine/mine_code/GIT_Projects/lc/src/control/Tuning_ladrc)

当前参数文件：

- [default_axis_params.json](/C:/context_mine/mine_code/GIT_Projects/lc/src/control/Tuning_ladrc/default_axis_params.json)

当前默认参数：

- `x: b0=31, wc=1.55, k=11, r=30`
- `y: b0=31, wc=1.55, k=11, r=30`
- `z: b0=100, wc=0.05, k=0.5, r=2`

### 2.2 当前第三章 RL 主链

当前第三章 RL 主链位于：

- [src/lc/control/controllers/adaptive_ladrc.py](/C:/context_mine/mine_code/GIT_Projects/lc/src/lc/control/controllers/adaptive_ladrc.py)
- [src/lc/control/envs/tracking.py](/C:/context_mine/mine_code/GIT_Projects/lc/src/lc/control/envs/tracking.py)
- [src/lc/control/policies/mddpg_control.py](/C:/context_mine/mine_code/GIT_Projects/lc/src/lc/control/policies/mddpg_control.py)
- [src/lc/control/trainers/control_trainer.py](/C:/context_mine/mine_code/GIT_Projects/lc/src/lc/control/trainers/control_trainer.py)

当前已经成立的 RL 口径：

- 动作维度为 `3`
- 动作顺序固定为：
  - `action[0] -> b0`
  - `action[1] -> wc`
  - `action[2] -> k`
- 控制器内部按：
  - `wo = k * wc`

## 3. 现有接口与问题

### 3.1 固定参数模块接口

#### 参数加载接口

文件：

- [parameter_loader.py](/C:/context_mine/mine_code/GIT_Projects/lc/src/control/Tuning_ladrc/parameter_loader.py)

当前接口：

- `load_axis_parameter_file(path) -> dict[str, AxisLADRCParameters]`
- `build_single_axis_ladrc_bundle(axis, parameter_file) -> ControllerBundle`

当前数据结构：

- `AxisLADRCParameters`
  - `axis: str`
  - `b0: float`
  - `wc: float`
  - `k: float`
  - `r: float`
  - `wo` 为只读派生属性

这部分已经足够作为 RL 初始化参数来源，不需要重写。

### 3.2 RL 控制器接口

文件：

- [adaptive_ladrc.py](/C:/context_mine/mine_code/GIT_Projects/lc/src/lc/control/controllers/adaptive_ladrc.py)

当前接口：

- `AdaptiveLADRCController.base: LADRCController`
- `AdaptiveLADRCController.adapt(action)`
- `AdaptiveLADRCController.step(reference, measurement, dt)`

当前问题：

- `AdaptiveLADRCController` 的默认动作边界仍是通用手写值：
  - `b0_bounds = (0.3, 2.5)`
  - `omega_c_bounds = (2.0, 15.0)`
  - `k_bounds = (2.0, 6.0)`
- 这和当前真实整定结果完全不匹配
- 尤其 `x/y/z` 三轴现在量级差异很大，不能再用一套统一范围

### 3.3 RL 训练器接口

文件：

- [control_trainer.py](/C:/context_mine/mine_code/GIT_Projects/lc/src/lc/control/trainers/control_trainer.py)

当前接口与问题：

- `ControlTrainer.build_ladrc_controller(axis, params=None)`
- `ControlTrainer.tuned_ladrc_params`
- `ControlTrainer._default_axis_params(axis)`

当前问题：

- `_default_axis_params()` 仍是轻量手写值：
  - `x: b0=1.0, omega_c=5.5, k=3.0`
  - `y: b0=1.0, omega_c=5.5, k=3.2`
  - `z: b0=1.2, omega_c=4.5, k=3.8`
- 这些值与当前 `PyBullet` 手工整定结果不一致
- `ControlTrainer` 还没有从 `src/control/Tuning_ladrc/default_axis_params.json` 读取基线参数

### 3.4 RL 环境接口

文件：

- [tracking.py](/C:/context_mine/mine_code/GIT_Projects/lc/src/lc/control/envs/tracking.py)

当前接口：

- 观测：`8` 维
- 动作：`3` 维，说明为 `normalized_b0, normalized_wc, normalized_k`
- 环境支持按 `axis` 切换

当前问题：

- 环境动作语义已经对，但缺少“每轴独立动作边界”的配置入口
- 后续如果 `RL` 要真正消费 `x/y/z` 的不同参数量级，需要把边界下沉到环境/agent 初始化层

## 4. 建议的对接方案

### 4.1 新增统一配置桥接文件

建议新增：

- `src/lc/control/configs/ladrc_rl_bridge.py`

作用：

- 从 [default_axis_params.json](/C:/context_mine/mine_code/GIT_Projects/lc/src/control/Tuning_ladrc/default_axis_params.json) 读取 `x/y/z` 基线参数
- 产出主链 RL 可直接使用的配置对象

建议新增数据结构：

- `AxisRLInitConfig`
  - `axis`
  - `baseline_b0`
  - `baseline_wc`
  - `baseline_k`
  - `baseline_wo`
  - `r`
  - `b0_bounds`
  - `wc_bounds`
  - `k_bounds`

- `LADRCRLBridgeConfig`
  - `x: AxisRLInitConfig`
  - `y: AxisRLInitConfig`
  - `z: AxisRLInitConfig`

### 4.2 边界生成规则

本轮不要再手写拍脑袋范围，直接按当前固定参数生成一版可执行边界。

默认规则建议：

- `b0_bounds = [0.8 * baseline_b0, 1.2 * baseline_b0]`
- `wc_bounds = [0.8 * baseline_wc, 1.2 * baseline_wc]`
- `k_bounds = [0.7 * baseline_k, 1.3 * baseline_k]`

对 `z` 轴单独限制：

- `k_bounds` 下限允许更小，但不低于 `0.3`
- `wc_bounds` 不能落到 `0`

这只是 RL 第一轮可执行边界，不等于最终最优边界。

### 4.3 修改 `AdaptiveLADRCController`

文件：

- [adaptive_ladrc.py](/C:/context_mine/mine_code/GIT_Projects/lc/src/lc/control/controllers/adaptive_ladrc.py)

需要改成：

- 支持构造时传入 `baseline_params`
- 支持构造时传入 `axis_specific_bounds`
- `base` 初始参数不再靠默认构造，而是显式从桥接配置注入

建议接口改为：

- `AdaptiveLADRCController(base: LADRCController, b0_bounds, omega_c_bounds, k_bounds)`
- 新增工厂函数：
  - `build_adaptive_ladrc_for_axis(axis, bridge_config)`

### 4.4 修改 `ControlTrainer`

文件：

- [control_trainer.py](/C:/context_mine/mine_code/GIT_Projects/lc/src/lc/control/trainers/control_trainer.py)

需要改成：

- 初始化时加载一次 `LADRCRLBridgeConfig`
- `build_ladrc_controller(axis)` 默认从桥接配置取固定参数
- `_default_axis_params(axis)` 不再硬编码，改为桥接配置
- `_train_agent()` 在创建 `ControlLADRLAgent` 后，用该轴的固定参数初始化 `agent.controller.base`
- 同时传入该轴对应的 `b0_bounds / wc_bounds / k_bounds`

### 4.5 修改 `ControlLADRLAgent`

文件：

- [mddpg_control.py](/C:/context_mine/mine_code/GIT_Projects/lc/src/lc/control/policies/mddpg_control.py)

需要改成：

- 构造时支持接收一个已初始化好的 `AdaptiveLADRCController`
- 或支持接收 `axis_rl_init_config`

建议接口改为：

- `ControlLADRLAgent(..., controller: AdaptiveLADRCController | None = None)`

如果外部未传入，则按旧逻辑构造；如果传入，则直接使用外部控制器。

### 4.6 主链 RL 和手工整定模块的职责边界

要明确给执行线程：

- `src/control/Tuning_ladrc/`
  - 是参数整定与人工验证模块
  - 输出固定参数和范围来源

- `src/lc/control/`
  - 是第三章论文主链 RL 实现
  - 消费整定结果，不重新定义参数口径

后续不允许再在 `src/lc/control/trainers/control_trainer.py` 内部写另一套脱节的默认 `LADRC` 参数。

## 5. 新接口定义

### 5.1 参数桥接读取接口

建议新增：

- `load_ladrc_rl_bridge_config(path: str | Path) -> LADRCRLBridgeConfig`

输入：

- `default_axis_params.json` 路径

输出：

- 三轴统一桥接配置对象

### 5.2 RL 控制器工厂接口

建议新增：

- `build_adaptive_ladrc_for_axis(axis: str, bridge_config: LADRCRLBridgeConfig) -> AdaptiveLADRCController`

行为：

- 用该轴固定参数初始化 `LADRCController`
- 用该轴动作边界初始化 `AdaptiveLADRCController`

### 5.3 训练器对接接口

建议在 `ControlTrainer` 中新增：

- `load_axis_rl_init(axis: str) -> AxisRLInitConfig`

用途：

- 让 `evaluate_ladrc`
- `evaluate_ddpg_ladrc`
- `evaluate_mddpg_ladrc`
- `_train_agent`

全部共享同一份基线来源。

## 6. 对接后的执行流程

另一个线程拿到这份文档后，执行顺序固定为：

1. 新增 `ladrc_rl_bridge.py`
2. 把 `default_axis_params.json` 接进桥接配置
3. 修改 `AdaptiveLADRCController` 支持轴级基线与边界
4. 修改 `ControlLADRLAgent` 支持外部注入控制器
5. 修改 `ControlTrainer` 使用桥接配置替换硬编码默认值
6. 跑三类最小验证

## 7. 验收测试

至少要补这些测试：

### 7.1 配置桥接测试

- `x/y/z` 三轴能从 `default_axis_params.json` 正确读出
- `wo = k * wc`
- 每轴边界按规则生成

### 7.2 控制器初始化测试

- `AdaptiveLADRCController` 创建后基线参数就是该轴固定参数
- 动作 `[-1, 1]` 能正确映射到该轴边界

### 7.3 训练器对接测试

- `ControlTrainer.build_ladrc_controller('x')` 使用桥接配置，不再走旧硬编码
- `_train_agent()` 初始化的 `agent.controller.base` 与轴基线一致

### 7.4 端到端 smoke

- `x` 轴 RL 训练 smoke 能跑通
- `y` 轴 RL 训练 smoke 能跑通
- `z` 轴 RL 训练 smoke 能跑通

这里不要求一次就优于固定参数，只要求接口通、口径一致、初始化正确。

## 8. 重要默认假设

- 本轮只做第三章主链 RL 对接，不改第四章代码
- 本轮不把 `PyBullet` 手工整定模块并入 `src/lc/control`
- 当前固定参数文件仍是：
  - `x: 31 / 1.55 / 11 / 30`
  - `y: 31 / 1.55 / 11 / 30`
  - `z: 100 / 0.05 / 0.5 / 2`
- `x` 轴二次整定得到的 `31 / 1.55 / 11 / 15` 暂不覆盖默认参数文件
- RL 第一轮先消费当前默认固定参数，不等待后续更多人工整定

