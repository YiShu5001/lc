# AGENTS.md

## 项目核心定位

这个仓库当前的核心目标不是维护“代码与论文双向同步”，而是：

- 以论文方法和章节设计为源头
- 把论文中的算法、模块、训练流程、实验设定翻译为可运行的代码实现

代理默认应把自己当作“论文方法实现助手”，而不是论文编辑器。

## 第一原则

在本仓库中，默认优先级是：

1. 理清论文在说什么
2. 把论文口径落实为当前代码架构下的实现
3. 保持实现尽量贴近论文方法本身

不需要默认去处理“代码会不会反过来影响论文表述”这类双向同步问题，除非用户明确要求。

## 你要服务的核心任务

代理在这里最重要的职责是：

- 把论文中的算法结构翻译成代码模块
- 把论文中的训练流程翻译成 trainer / memory / curriculum / reward / experiment 代码
- 把论文中的实验设计翻译成可运行的实验入口和输出
- 把论文中的系统关系翻译成模块接口和桥接逻辑

换句话说：

- 论文是“方法规格”
- 代码是“实现落地”

## 当前仓库的实现主链

当前代码主链以 `src/lc/` 为准：

- `src/lc/control/`
- `src/lc/planning/`
- `src/lc/integration/`
- `src/lc/envs/`
- `src/lc/analysis/`
- `src/lc/entrypoints/`

配套入口与验证：

- `experiments/control/`
- `experiments/planning/`
- `experiments/integrated/`
- `tests/`

默认情况下，新的实现应优先写入这些目录。

## 当前仓库的论文口径来源

当前仓库内可直接作为“论文方法规格”的主要文件在 `docs/`：

- `docs/chapter3_readme.md`
- `docs/chapter4_readme.md`
- `docs/chapter4_execution_plan.md`
- `docs/chapter4_experiment_matrix.md`
- `docs/chapter3_chapter4_full_guide.md`
- `docs/chapter4_ai_prompt_template.md`
- `README.md`

这些文件的主要用途不是让代理去润色论文，而是帮助代理搞清楚：

- 章节方法到底是什么
- 目标算法的结构和语义是什么
- 哪些模块已实现，哪些只是骨架
- 接下来应该优先补哪一部分实现

## 历史目录的使用方式

以下目录默认视为历史实现、参考实现或待迁移资产：

- `Gym_env/`
- `NN/`
- `Reinforce_learning/`
- `Trainer/`
- `configs/`
- `main.py`

它们的默认作用是：

- 提供旧版算法细节参考
- 提供可迁移逻辑
- 帮助理解历史实现思路

但默认不要把它们当作优先修改目标，除非：

- 用户明确要求改旧链路
- 当前 `src/lc` 还缺该能力，而你需要从旧实现中搬运或对照

## 第三章实现规则

第三章默认对应控制层实现。

当前主要实现区域：

- `src/lc/control/controllers/`
- `src/lc/control/envs/`
- `src/lc/control/policies/`
- `src/lc/control/trainers/`
- `src/lc/control/experiments/`
- `src/lc/control/plotting/`

实现第三章时，默认遵守以下论文口径：

- 控制器主体仍然是 `LADRC`
- 强化学习负责在线调参，而不是完全替代控制器
- 重点是鲁棒性、抗扰性、恢复能力、平滑性和参数自适应

如果你在做第三章实现，优先把论文内容翻译成：

- 控制器接口
- 参数调节动作空间
- 状态设计
- 奖励设计
- 训练流程
- 对比实验
- 图表输出

## 第四章实现规则

第四章默认对应规划层实现。

当前主要实现区域：

- `src/lc/planning/envs/`
- `src/lc/planning/encoders/`
- `src/lc/planning/models/`
- `src/lc/planning/critics/`
- `src/lc/planning/memory/`
- `src/lc/planning/curriculum/`
- `src/lc/planning/rewards/`
- `src/lc/planning/trainers/`
- `src/lc/planning/experiments/`
- `src/lc/planning/plotting/`

实现第四章时，默认遵守以下论文口径：

- 主模型是 `MultiUAVModel`
- 主输入是 `self_state / obstacles / neighbors`
- 方法语义是“两阶段：先避障，后协同”
- 基线模型只作为对比，不替代主方法

如果你在做第四章实现，优先把论文内容翻译成：

- 网络结构
- actor-critic 训练闭环
- curriculum 逻辑
- Pyramid-PER 或多层经验池逻辑
- 奖励拆分
- 对比实验和消融实验
- 论文化图表输出

## 系统桥接规则

当论文描述涉及“规划层输出如何进入控制层”时，再去处理桥接实现。

主要区域：

- `src/lc/integration/`
- `experiments/integrated/`
- `tests/smoke/test_chapter34_interfaces.py`

桥接实现的目标是把论文中的层级关系变成代码接口，而不是额外创造新的方法叙事。

## 默认工作方式

对非琐碎任务，代理默认按下面顺序工作：

1. 先确认任务属于第三章、第四章还是系统桥接。
2. 先读对应 `docs/` 文件，弄清论文方法口径。
3. 再读最小必要的 `src/lc/` 文件，确认当前实现状态。
4. 在现有主链上补实现，而不是重写整个架构。
5. 必要时参考旧目录，但实现优先落在 `src/lc/`。
6. 最终明确说明：
   - 你把论文中的哪些内容翻译成了代码
   - 改了哪些模块
   - 还缺哪些论文能力尚未落地

## 默认不是重点的事情

除非用户明确要求，否则以下事项不是默认重点：

- 主动修改论文正文
- 主动维护论文与代码的双向一致性说明
- 为局部代码改动同步大面积改文档
- 把任务转成“文档治理”或“仓库叙事治理”

在这个仓库里，默认应把时间花在“论文算法实现”上。

## 代码实现风格要求

实现论文方法时请遵守：

- 优先保留当前 `src/lc` 架构
- 优先做局部、渐进、可验证的实现补全
- 不要随意重命名论文主方法
- 不要把主方法偷偷降级成简单基线
- 不要因为旧代码更完整，就直接回退到旧目录重做

如果当前代码只是骨架，正确做法通常是：

- 在骨架上补齐训练、记忆、奖励、实验和绘图

而不是：

- 推倒重写

## 推荐起点

### 做第三章实现时，优先看

- `docs/chapter3_readme.md`
- `docs/chapter3_chapter4_full_guide.md`
- `src/lc/control/trainers/control_trainer.py`
- `src/lc/control/experiments/compare.py`

### 做第四章实现时，优先看

- `docs/chapter4_readme.md`
- `docs/chapter4_execution_plan.md`
- `docs/chapter4_experiment_matrix.md`
- `docs/chapter4_ai_prompt_template.md`
- `src/lc/planning/trainers/planning_trainer.py`
- `src/lc/planning/models/multi_uav_model.py`
- `src/lc/planning/experiments/compare.py`

### 做桥接实现时，优先看

- `docs/chapter3_chapter4_full_guide.md`
- `src/lc/integration/pipeline/bridge.py`
- `experiments/integrated/run_chapter34_demo.py`

## 交付期望

在这个仓库里，一个好的最终结果通常应说明：

- 本次落地了哪一段论文算法
- 对应改了哪些代码模块
- 跑了哪些测试或最小验证
- 还有哪些论文中的能力尚未实现

## 一句话总结

本仓库里，代理的默认职责是“先读懂论文方法，再把论文内容翻译成当前 `src/lc` 主链中的代码实现”。
