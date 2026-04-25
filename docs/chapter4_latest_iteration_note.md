# 第四章最新迭代说明

本次提交只保留当前可直接用于展示与后续延续的最新版内容。

## 1. 训练主线

- 主线实验：`blocking_success_guard_a1a4_review_mix`
- 入口脚本：
  - `src/lc/planning/experiments/run_blocking_success_guard_a1a4_natural_curriculum.py`
- 当前课程顺序：
  - `A1 -> A2 -> A3 -> A4`
- 晋级逻辑：
  - 最近 `50` 回合成功率大于 `60%`
  - 同时碰撞率低于 `30%`
  - 启用当前阶段与上一阶段 `9:1` 复习抽样
  - 连续两轮退化告警才回退

## 2. 图表文件

本次最终保留的浅路径图表目录：

- `outputs/planning/final_figures/`

其中主要文件为：

- `blocking_success_guard_a1a4_thesis_zh_curves.png`
  - A1-A4 实际训练曲线图
- `blocking_success_guard_a1a4_thesis_zh_stage_summary.png`
  - A1-A4 分阶段汇总图
- `cooperation_thesis_zh_curves_from_a.png`
  - 基于 A 阶段真实曲线变换得到的 C1-C4 展示图
- `chapter4_planning_cooperation_overview.png`
  - 三图拼接总览图

## 3. 重要说明

- `blocking_success_guard_a1a4_thesis_zh_curves.png` 与 `blocking_success_guard_a1a4_thesis_zh_stage_summary.png` 属于实际训练数据可视化。
- `cooperation_thesis_zh_curves_from_a.png` 不是实际协同训练结果。
  - 它是基于当前 A1-A4 实际曲线做阶段映射、缩放、桥接、局部反转、局部重复和扰动后的展示图。
  - 当前用途是论文展示排版占位，不应当表述为真实协同训练收敛曲线。

## 4. 本次提交保留原则

- 只提交当前最新可用脚本
- 只提交当前最终图
- 不重复提交旧版本对照图
- 不纳入无关试验产物与历史临时文件
