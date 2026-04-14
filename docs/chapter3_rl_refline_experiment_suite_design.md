# Chapter 3 `x` 轴 RL RefLine 三类实验设计与执行说明

## 1. 目标

本文给出第三章三类关键实验的外置编排方案：

- 消融对比实验
- 蒙特卡洛重复实验
- 泛化能力检验实验

设计原则是：

- 不修改现有训练主代码
- 直接复用当前已有的 `run_control_comparison(...)` 能力
- 所有结果写入独立目录，避免覆盖
- 每类实验都能单独执行，也能统一调度

对应的外置总控脚本是：

- [chapter3_rl_refline_suite.py](D:/ZhangC/lc_codex_ch4_run/experiments/control/chapter3_rl_refline_suite.py)

## 2. 为什么这样安排

### 2.1 消融实验

目的不是重新做一遍主实验，而是回答“当前最好结果到底来自哪一部分设计”。

在不改原代码的前提下，最稳妥的做法不是强行往现有框架里插新的分支，而是直接把当前最优配置拆成几组外置对照：

- `best_config`
- `no_decay`
- `smaller_model`
- `faster_target`

这样能直接回答三个高价值问题：

- 探索率衰减是否真的提升了稳定性
- 模型变大是否确实带来了收益
- 更保守的 target update 是否必要

### 2.2 蒙特卡洛实验

目的不是追求单次最优值，而是回答“这个结果是不是偶然”。

这里最关键的是不要把多个 seed 混在一个目录里，否则后续分析很难拆分。因此外置脚本按 seed 一次一目录跑，后面再做汇总。

推荐默认 `10` 个 seed：

- `7, 8, 9, 10, 11, 12, 13, 14, 15, 16`

这样可以直接做：

- reward 分布
- rmse 分布
- 95% 区间
- 异常 seed 排查

### 2.3 泛化实验

现有代码里虽然已经有 `run_control_generalization(...)`，但它在构造 scoped config 时没有完整继承当前最优模型超参数。因此如果想严格保持“当前最优配置”，更稳妥的做法是外置脚本直接按 difficulty 循环调用 `run_control_comparison(...)`。

这样可以确保以下配置不丢失：

- `hidden_dim=768`
- `dropout_p=0.25`
- `tau=0.02`
- `soft_update_interval=10`
- `exploration_noise_schedule="linear"`
- `exploration_noise_start=0.1`
- `exploration_noise_end=0.04`

## 3. 三类实验的优化细节

### 3.1 消融实验建议

固定：

- `axes=("x",)`
- `reference_profile_mode="rl_refline_six_phase"`
- `train_episodes=500`
- `compare_episodes=5`
- `snapshot_interval=50`
- 共享值固定为当前最优 `v`

建议先用当前阶段已知最优 `v=7`，等你这轮 `v=1..10` 全量扫描跑完后，再把 `--best-v` 替换为最新结果。

四组配置如下：

| 组别 | hidden_dim | dropout | tau | soft interval | noise |
| --- | ---: | ---: | ---: | ---: | --- |
| best_config | 768 | 0.25 | 0.02 | 10 | linear 0.10 -> 0.04 |
| no_decay | 768 | 0.25 | 0.02 | 10 | fixed 0.10 |
| smaller_model | 512 | 0.20 | 0.02 | 10 | linear 0.10 -> 0.04 |
| faster_target | 768 | 0.25 | 0.05 | 20 | linear 0.10 -> 0.04 |

重点看：

- `rmse`
- `reward`
- `last100 reward mean/std`
- `control energy`

### 3.2 蒙特卡洛实验建议

固定主模型配置不变，只改变 seed。

推荐输出：

- 每个 seed 独立目录
- 每个 seed 的 `summary.json`
- 汇总 `manifest.json`

后续分析重点：

- 平均 `rmse`
- 平均 `reward`
- 方差和极差
- 最差 seed 是否仍能优于 `LADRC`

### 3.3 泛化实验建议

外置脚本默认跑四个 difficulty：

- `easy`
- `medium`
- `hard`
- `extreme`

每个 difficulty 单独一目录，保持同一套最优超参数。这样得到的是“同策略配置在不同复杂度场景下的适应性”，而不是重新设计四套模型。

重点观察：

- `rmse` 随 difficulty 的劣化趋势
- `reward` 是否仍优于 `LADRC`
- `overshoot` 和 `control_energy` 是否在高难度下恶化过快

## 4. 外置执行方式

### 4.1 只看计划，不实际跑

```powershell
$env:PYTHONPATH='D:\ZhangC\lc_codex_ch4_run\src'
D:\anaconda3\envs\drone\python.exe D:\ZhangC\lc_codex_ch4_run\experiments\control\chapter3_rl_refline_suite.py --suite all --best-v 7 --dry-run
```

### 4.2 单独跑消融实验

```powershell
$env:PYTHONPATH='D:\ZhangC\lc_codex_ch4_run\src'
D:\anaconda3\envs\drone\python.exe D:\ZhangC\lc_codex_ch4_run\experiments\control\chapter3_rl_refline_suite.py --suite ablation --best-v 7 --tag chapter3_ablation_run1
```

### 4.3 单独跑蒙特卡洛实验

```powershell
$env:PYTHONPATH='D:\ZhangC\lc_codex_ch4_run\src'
D:\anaconda3\envs\drone\python.exe D:\ZhangC\lc_codex_ch4_run\experiments\control\chapter3_rl_refline_suite.py --suite monte_carlo --best-v 7 --tag chapter3_mc_run1
```

### 4.4 单独跑泛化实验

```powershell
$env:PYTHONPATH='D:\ZhangC\lc_codex_ch4_run\src'
D:\anaconda3\envs\drone\python.exe D:\ZhangC\lc_codex_ch4_run\experiments\control\chapter3_rl_refline_suite.py --suite generalization --best-v 7 --tag chapter3_gen_run1
```

## 5. 输出结果示意图

下图给出三类实验的组织关系和输出结构：

![Chapter3 Experiment Suite](D:/ZhangC/lc_codex_ch4_run/docs/figures/chapter3/chapter3_rl_refline_experiment_suite.svg)

## 6. 建议执行顺序

推荐顺序如下：

1. 先完成当前 `v=1..10` 全量扫描，锁定最终 `best_v`
2. 再跑消融实验
3. 然后跑蒙特卡洛实验
4. 最后跑泛化实验

原因是后面三类实验都应固定在同一个最终共享值上，否则论文里会出现配置口径不一致的问题。
