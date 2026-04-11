# RLcontrolRefLine

这个目录用于集中定义第三章 `x/y` 轴 `RL-LADRC` 参数自整定训练的“参考线任务”。

核心目标不是改控制器，而是明确训练时每个 episode 到底让智能体学什么：

- 单轴静止保持
- 单轴匀速前进
- 扰动下保持
- 反向匀速
- 扰动恢复
- 末端再次静止

## 默认 6 段任务

固定顺序如下，不允许打乱：

1. `hold_start`
2. `forward_constant_velocity`
3. `disturbance_hold`
4. `reverse_constant_velocity`
5. `disturbance_recovery`
6. `hold_end`

## 默认范围

面向 `x/y` 轴的默认值如下：

- 阶段1 静止保持：`0.8s ~ 1.2s`
- 阶段2 匀速前进：速度 `0.25 ~ 0.55`，时长 `1.2s ~ 2.0s`
- 阶段3 干扰保持：扰动 `0.06 ~ 0.16`，时长 `1.0s ~ 1.8s`
- 阶段4 反向匀速：速度 `-0.50 ~ -0.20`，时长 `1.2s ~ 2.0s`
- 阶段5 扰动恢复：`0.8s ~ 1.5s`
- 阶段6 末端静止保持：`0.8s ~ 1.2s`

总时长默认 `8s`，控制频率 `100Hz`，RL 频率 `10Hz`。

## 随机化规则

- 阶段顺序固定
- 每个 episode 的阶段时长会在范围内随机采样
- 匀速前进与反向匀速的速度会在范围内随机采样
- 干扰保持阶段的扰动幅值会在范围内随机采样
- 如果采样后的总时长不等于目标总时长，代码会自动缩放到整条 episode
- 每段最短时长默认不小于 `0.6s`

## 主要接口

- `build_default_xy_task_config(axis)`
  返回默认 `x/y` 轴任务模板。后续如果你想改范围，优先改这里。
- `sample_phase_plan(config, seed=None)`
  只采样阶段计划，不生成完整曲线。适合调试“时间节点是否合理”。
- `build_refline_episode(config, seed=None)`
  生成完整 episode，包括参考位置、参考速度和扰动。
- `adapt_episode_to_tracking_inputs(bundle)`
  把生成结果转成统一的参考位置、参考速度和扰动数组，供 PyBullet 训练/评测链路消费。

## 推荐修改点

如果你后续要调整训练问题，通常只需要改以下位置：

- 改阶段顺序或增删阶段：
  `task_spec.py` 中的 `PhaseKind`
  `builders.py` 中的 `build_default_xy_task_config()`
- 改时长范围、速度范围、扰动范围：
  `builders.py` 中每个 `PhaseSpec`
- 改扰动恢复方式：
  `AxisRLRefLineTaskConfig.disturbance_decay_mode`
- 改是否允许随机：
  `AxisRLRefLineTaskConfig.enable_randomization`
  或单个 `PhaseSpec` 上的 `randomize_*`

## 与主链的关系

- 本目录是“任务规格和生成器”
- `pybullet_axis_env.py` / `pybullet_eval_env.py` 负责消费这些结果，并把它们用于 PyBullet 环境 roll-out
- `pybullet_axis_trainer.py` 负责在每个 episode 开始时按轴采样一个新任务

## 简单示例

```python
from lc.control.RLcontrolRefLine import build_default_xy_task_config, build_refline_episode

config = build_default_xy_task_config("x")
episode = build_refline_episode(config, seed=7)

print(episode.phase_table)
print(episode.reference_velocity[:20])
print(episode.disturbance[:20])
```

## 说明

- 当前默认主任务是 `x/y` 轴
- `z` 轴允许构建，但不是本轮优先任务
- 奖励定义不在这个目录里改；当前主链仍由环境侧维护“位置误差 + 速度误差”奖励
