# 项目总交接入口文档

## 1. 文档定位

本文档用于作为当前项目的统一交接入口，帮助你或另一个 AI 快速理解：

- 现在项目做到哪里了
- 第三章和第四章分别有哪些现成材料
- 每一章应该先看什么
- 后续推进时不能改的主口径是什么

本文档不承担细节说明，而是承担“总导航页”的作用。

## 2. 当前项目主线

当前项目已经按新的 `src/lc/` 主链建立起两条章节实现路径：

- 第三章控制层
- 第四章规划层

当前推荐把整个项目理解为三层：

1. 第三章控制层  
   负责单轴位置跟踪、扰动抑制、`LADRC` 与强化学习调参结合。

2. 第四章规划层  
   负责多无人机协同规划、避障、任务分解、课程学习和经验池机制。

3. 系统桥接层  
   负责把规划层输出映射到控制层输入，形成完整系统链路。

## 3. 第三章交接入口

第三章当前推荐阅读顺序如下：

1. [chapter3_readme.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter3_readme.md)  
   用于快速理解第三章当前代码结构、方法口径、实验输出和完成度。

2. [chapter3_execution_plan.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter3_execution_plan.md)  
   用于理解第三章后续应该如何继续推进，以及哪些内容不能随意改。

3. [chapter3_experiment_matrix.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter3_experiment_matrix.md)  
   用于查看第三章对比实验、消融实验和复杂度实验矩阵。

4. [chapter3_ai_prompt_template.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter3_ai_prompt_template.md)  
   用于直接交给另一个 AI 继续执行第三章任务。

5. [chapter3_method_figures.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter3_method_figures.md)  
   用于查看第三章方法插图说明。

第三章方法插图资源位于：

- [ddpg_ladrc_framework.svg](/C:/context_mine/mine_code/GIT_Projects/lc/docs/figures/chapter3/ddpg_ladrc_framework.svg)
- [mddpg_enhancement.svg](/C:/context_mine/mine_code/GIT_Projects/lc/docs/figures/chapter3/mddpg_enhancement.svg)

### 第三章固定口径

第三章当前不能随意改动的主口径为：

- 默认主方法：`DDPG-LADRC`
- 增强版对比：`mDDPG-LADRC`
- 默认任务：单轴位置跟踪 + 扰动抑制
- 强化学习动作：在线调节 `omega_c / omega_o / b0`
- 奖励函数：`reward = -|position_error|`

## 4. 第四章交接入口

第四章当前推荐阅读顺序如下：

1. [chapter4_readme.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter4_readme.md)  
   用于快速理解第四章当前代码结构、方法口径、实验输出和完成度。

2. [chapter4_execution_plan.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter4_execution_plan.md)  
   用于理解第四章下一步应如何继续补实训练闭环和实验体系。

3. [chapter4_experiment_matrix.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter4_experiment_matrix.md)  
   用于查看第四章对比实验、消融实验、课程实验和复杂度实验矩阵。

4. [chapter4_ai_prompt_template.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter4_ai_prompt_template.md)  
   用于直接交给另一个 AI 继续执行第四章任务。

### 第四章固定口径

第四章当前不能随意改动的主口径为：

- 主方法：`MultiUAVModel`
- 主输入结构：`self_state / obstacles / neighbors`
- 语义结构：先避障、后协同
- 主要输出：`avoid_action / final_action`
- 主链目录：`src/lc/planning/`

## 5. 代码主入口参考

如果需要从代码角度进入，建议优先查看：

### 第三章

- `src/lc/control/trainers/control_trainer.py`
- `src/lc/control/experiments/compare.py`
- `src/lc/control/envs/tracking.py`
- `src/lc/control/controllers/adaptive_ladrc.py`
- `src/lc/control/policies/mddpg_control.py`

### 第四章

- `src/lc/planning/trainers/planning_trainer.py`
- `src/lc/planning/experiments/compare.py`
- `src/lc/planning/models/multi_uav_model.py`
- `src/lc/planning/envs/swarm.py`
- `src/lc/planning/memory/pyramid.py`
- `src/lc/planning/curriculum/scheduler.py`

## 6. 推荐交接方式

如果你要把任务交给另一个 AI，建议按下面方式进行：

### 只交第三章

直接发：

- [chapter3_readme.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter3_readme.md)
- [chapter3_execution_plan.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter3_execution_plan.md)
- [chapter3_experiment_matrix.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter3_experiment_matrix.md)
- [chapter3_ai_prompt_template.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter3_ai_prompt_template.md)

### 只交第四章

直接发：

- [chapter4_readme.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter4_readme.md)
- [chapter4_execution_plan.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter4_execution_plan.md)
- [chapter4_experiment_matrix.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter4_experiment_matrix.md)
- [chapter4_ai_prompt_template.md](/C:/context_mine/mine_code/GIT_Projects/lc/docs/chapter4_ai_prompt_template.md)

### 同时交第三章和第四章

建议先发本文档，再附对应章节的四件套。

## 7. 当前项目真实判断

当前项目不是“空架构”，也不是“已全部完成”。

更准确的状态是：

- 第三章已经形成可运行主链，具备方法对比、机制消融、图表输出和方法插图。
- 第四章已经形成可运行骨架，主模型、环境、课程学习、经验池、实验和绘图入口已建立，但仍需要继续补实真实训练闭环和完整实验体系。
- 两章都已经形成适合交给另一个 AI 继续推进的文档体系。

## 8. 建议的后续推进顺序

如果后续继续推进，建议按这个顺序：

1. 继续补实第四章训练闭环、课程学习和 Pyramid-PER  
   因为第四章当前缺口比第三章更大。

2. 继续增强第三章实验稳定性和论文图表质量  
   因为第三章已经能跑，下一步更多是做深做稳。

3. 最后再做系统桥接层的联合实验与总文档整理  
   这样能避免在章节本体还没稳定前过早做端到端包装。

## 9. 建议新增的总任务分发方式

如果你之后还要继续拆分任务给不同 AI，建议固定成三类：

- 第三章 AI：只负责 `src/lc/control/` 与第三章文档
- 第四章 AI：只负责 `src/lc/planning/` 与第四章文档
- 集成 AI：负责 `src/lc/integration/`、统一 README、总实验汇总和系统级图表

这样可以避免不同 AI 相互覆盖主口径。
