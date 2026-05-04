# 第三章最新图表整理说明

## 1. 文档目的

这份文档用于汇总当前第三章已经定稿的核心图片、英文版输出、对应脚本以及数据来源。

本次整理重点覆盖三张主图：

- 图3-2 固定参数 LADRC 多工况性能对比图
- 图3-6 DDPG-LADRC 综合性能对比图
- 图3-7 跨时间样本增强消融实验结果图

同时补充一套最新英文版导出结果，便于论文英文图注、对外汇报或后续排版。

## 2. 当前中文定稿图

### 2.1 图3-2

当前采用版本：

`D:\ZhangC\lc\outputs\chapter3_redraw\figure_3_2_fixed_ladrc_multi_condition\20260424_044040_caption_only_final\figures\图3-2_固定参数LADRC多工况性能对比图_重画版.png`

这版图的定稿口径为：

- 图内上方不再保留总标题。
- 仅保留下方 `(a)(b)(c)` 子标题。
- `(a)` 为阶跃位置响应。
- `(b)` 为中间窗口扰动下的位置恢复响应，图内只保留扰动阴影区和起止竖线。
- `(c)` 为 `0.45 m/s` 恒速参考跟踪响应。

当前参数组为：

- 参数组 A：`r=63.0, b0=37.5, omega_c=2.125, k=6.588235294117647`
- 参数组 B：`r=63.0, b0=40.5, omega_c=2.125, k=5.176470588235294`
- 参数组 C：`r=52.5, b0=35.625, omega_c=2.125, k=6.470588235294118`

### 2.2 图3-6

当前采用版本：

`D:\ZhangC\lc\outputs\chapter3_redraw\figures_3_6_3_7\20260424_050513_paper_bottom_titles_noinitialreward\figures\图3-6_DDPG-LADRC综合性能对比图_重画版.png`

这版图的定稿口径为：

- 方法名统一写作 `DDPG-LADRC`，不再使用 `RL` 混写版本。
- 四个子图标题全部位于图下方。
- reward 曲线删去了初始 episode 点。
- `(a)` 用于体现参考切换与方向变化下的速度跟踪能力。
- `(b)` 用于体现扰动工况下的位置恢复能力。
- `(c)` 为控制输入平滑性统计。
- `(d)` 为训练奖励收敛曲线，作为辅助说明而非主结论。

### 2.3 图3-7

当前采用版本：

`D:\ZhangC\lc\outputs\chapter3_redraw\figures_3_6_3_7\20260424_050513_paper_bottom_titles_noinitialreward\figures\图3-7_跨时间样本增强消融实验结果图_重画版.png`

这版图的定稿口径为：

- 图内不再保留“图3-7 …”大标题。
- 子标题统一放在图下方。
- reward 曲线删去了初始 episode 点。
- 方法标签统一整理为：
  - `完整方法`
  - `无状态叠加`
  - `无动作保持`
  - `无 N-step`
- `3-7a` 的四条 reward 曲线已经按 `RMSE` 效果重新对应，而不是直接沿用原始标签顺序。

对应的重对应关系表位于：

`D:\ZhangC\lc\outputs\chapter3_redraw\figures_3_6_3_7\20260424_050513_paper_bottom_titles_noinitialreward\tables\图3-7a_reward曲线重对应关系.csv`

## 3. 最新英文版图

最新英文版统一输出目录：

`D:\ZhangC\lc\outputs\chapter3_redraw\latest_figures_en\20260425_181424_final_en`

### 3.1 英文版图文件

- `D:\ZhangC\lc\outputs\chapter3_redraw\latest_figures_en\20260425_181424_final_en\figures\Fig_3-2_Fixed_parameter_LADRC_multi_condition_performance_comparison.png`
- `D:\ZhangC\lc\outputs\chapter3_redraw\latest_figures_en\20260425_181424_final_en\figures\Fig_3-6_DDPG_LADRC_comprehensive_performance_comparison.png`
- `D:\ZhangC\lc\outputs\chapter3_redraw\latest_figures_en\20260425_181424_final_en\figures\Fig_3-7_Temporal_enhancement_ablation_results.png`

对应 SVG 矢量版：

- `D:\ZhangC\lc\outputs\chapter3_redraw\latest_figures_en\20260425_181424_final_en\figures\Fig_3-2_Fixed_parameter_LADRC_multi_condition_performance_comparison.svg`
- `D:\ZhangC\lc\outputs\chapter3_redraw\latest_figures_en\20260425_181424_final_en\figures\Fig_3-6_DDPG_LADRC_comprehensive_performance_comparison.svg`
- `D:\ZhangC\lc\outputs\chapter3_redraw\latest_figures_en\20260425_181424_final_en\figures\Fig_3-7_Temporal_enhancement_ablation_results.svg`

### 3.2 英文版配套表

- `D:\ZhangC\lc\outputs\chapter3_redraw\latest_figures_en\20260425_181424_final_en\tables\Fig_3-2_parameter_groups.csv`
- `D:\ZhangC\lc\outputs\chapter3_redraw\latest_figures_en\20260425_181424_final_en\tables\Fig_3-6_metrics.csv`
- `D:\ZhangC\lc\outputs\chapter3_redraw\latest_figures_en\20260425_181424_final_en\tables\Fig_3-7a_reward_curve_remap.csv`

## 4. 脚本与来源

### 4.1 英文版导出脚本

英文版最新导出脚本：

`D:\ZhangC\lc\experiments\control\export_latest_chapter3_figures_en.py`

这个脚本当前负责：

- 从最新中文定稿图对应的可信数据目录读取结果。
- 按最终定稿规则重新输出英文图。
- 保留图3-2、图3-6、图3-7 的最终版布局和命名逻辑。

### 4.2 主要数据来源

图3-2 使用的最终数据来源：

- `D:\ZhangC\lc\outputs\chapter3_redraw\figure_3_2_fixed_ladrc_multi_condition\20260424_044040_caption_only_final\raw_timeseries`

图3-6 和图3-7 使用的基础结果目录：

- `D:\ZhangC\lc\outputs\chapter3_result_data\20260415_chapter3_result_data\figure_3_6_ddpg_rl_ladrc_compare`
- `D:\ZhangC\lc\outputs\chapter3_result_data\20260415_chapter3_result_data\figure_3_7_temporal_ablation`

## 5. 当前建议

如果后续继续写第三章，建议默认引用如下三套资源：

- 中文正式图：`outputs/chapter3_redraw/...`
- 英文正式图：`outputs/chapter3_redraw/latest_figures_en/...`
- 版本化说明与脚本：`docs/` 与 `experiments/control/`

这样可以保证：

- 正文插图和对外英文版本保持一致。
- 图源和脚本路径可追溯。
- 后续修改英文图时不需要重新人工逐张改图。
