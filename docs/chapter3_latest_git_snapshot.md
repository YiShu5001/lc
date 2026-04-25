# 第三章最新主链版本说明

## 目的

这份说明只记录当前建议进入 Git 的第三章最新主链代码与配套脚本，不包含中间实验输出、图片导出目录和缓存文件。

本次版本管理遵循两个原则：

- 只保留迭代后的最新实现入口与主链代码。
- 不把 `outputs/` 下的结果图、临时重画目录、`__pycache__` 和 `.matplotlib*` 缓存提交到 Git。

## 当前建议保留的核心代码

### 1. 控制主链

- `D:\ZhangC\lc\src\lc\control\configs\control_config.py`
- `D:\ZhangC\lc\src\lc\control\controllers\pybullet_variants.py`
- `D:\ZhangC\lc\src\lc\control\simulators\pybullet_runner.py`
- `D:\ZhangC\lc\src\lc\control\trainers\pybullet_axis_trainer.py`
- `D:\ZhangC\lc\src\lc\control\experiments\compare.py`
- `D:\ZhangC\lc\src\lc\rl\algorithms\mddpg\policy.py`

这一组文件对应当前第三章可信 PyBullet 控制闭环，包括：

- LADRC 与 PID 的位置环接入。
- DDPG-LADRC 参数在线调节。
- 训练、确定性评估、正式对比共用的 rollout 主链。
- 随机扰动、动作边界、anchor + delta 参数解码。
- 与跨时间增强相关的 trainer / policy 更新逻辑。

### 2. 实验与导出入口

建议保留的最新实验脚本包括：

- `D:\ZhangC\lc\experiments\control\run_pybullet_x_refline_no_disturbance_mddpg_retrain.py`
- `D:\ZhangC\lc\experiments\control\run_pybullet_x_refline_random_hover_disturbance_mddpg_retrain.py`
- `D:\ZhangC\lc\experiments\control\run_pybullet_x_refline_random_hover_disturbance_mddpg_retrain_v136_reexpanded.py`
- `D:\ZhangC\lc\experiments\control\run_pybullet_x_temporal_ablation_suite.py`
- `D:\ZhangC\lc\experiments\control\run_pybullet_xy_axis_transfer_debug.py`
- `D:\ZhangC\lc\experiments\control\run_pybullet_xy_ladrc_circle_demo.py`
- `D:\ZhangC\lc\experiments\control\run_pybullet_y_and_xy_ladrc_motion_check.py`
- `D:\ZhangC\lc\experiments\control\run_pybullet_y_pid_ladrc_ddpg_random_hover_disturb_compare.py`
- `D:\ZhangC\lc\experiments\control\run_pybullet_y_refline_random_hover_disturbance_mddpg_retrain_v3_state6.py`

### 3. 第三章图表重画与导出脚本

建议保留的最新图表脚本包括：

- `D:\ZhangC\lc\experiments\control\export_chapter3_plot_data.py`
- `D:\ZhangC\lc\experiments\control\plot_chapter3_result_figures.py`
- `D:\ZhangC\lc\experiments\control\plot_chapter3_result_figures_en.py`
- `D:\ZhangC\lc\experiments\control\redraw_ladrc_tuning_stage_figures.py`
- `D:\ZhangC\lc\experiments\control\redraw_chapter3_figure_3_2.py`
- `D:\ZhangC\lc\experiments\control\redraw_chapter3_figures_3_6_3_7.py`
- `D:\ZhangC\lc\experiments\control\chapter3_export_figures_and_notes.py`

这些脚本负责：

- 从可信实验结果导出论文图表数据。
- 重画第三章关键图片。
- 生成中英文版本图表。
- 导出论文正文版和实验记录版说明材料。

## 本次不建议进入 Git 的内容

以下内容默认保留在本地，不进入这次提交：

- `D:\ZhangC\lc\outputs\chapter3_redraw\`
- `D:\ZhangC\lc\outputs\chapter3_paper_assets\`
- `D:\ZhangC\lc\outputs\chapter3_result_data\`
- `D:\ZhangC\lc\outputs\control_pybullet\`
- `D:\ZhangC\lc\outputs\control_pybullet_rl\`
- 各级 `__pycache__`
- `.pyc`
- `.matplotlib*`

原因是这些目录主要保存：

- 重画后的图片文件。
- 训练与对比实验输出。
- 中间表格和临时汇总结果。
- 绘图缓存与解释器缓存。

它们适合本地留档，不适合作为这次“只保留最新迭代代码”的 Git 版本内容。

## 使用建议

如果后续还要继续第三章工作，建议默认从下面两类入口继续：

- 算法与训练主链：`src/lc/control/` 与 `src/lc/rl/algorithms/mddpg/`
- 正式实验与论文图表：`experiments/control/`

如果后续需要再次归档，只需要在这份说明的基础上继续筛选最新脚本，而不必回收所有历史输出目录。
