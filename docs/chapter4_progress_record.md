# 第四章当前进度记录

## 1. 文档用途

本文档用于记录第四章当前代码实现的真实状态，避免后续继续开发时出现这些问题：

- 不清楚已经完成了什么
- 不清楚哪些只是临时实现
- 不清楚下一步应该优先补哪里
- 不清楚哪些地方需要用户补充论文口径

这是一份“继续开发交接文档”，以当前代码为准，不以旧 README 为准。

## 2. 当前结论

第四章目前已经从“纯骨架”推进到了“能训练、能对比、能出图”的阶段，但离论文级最终实现还差一段距离。

当前最准确的判断是：

- 主链已经打通
- 训练闭环已经存在
- 主 Actor 已按图片重建为两阶段注意力结构
- 对比实验和图表已经能输出
- 但环境、奖励、课程策略、经验池打分和训练算法仍然是“论文导向的近似实现”，还不是最终论文定稿版

## 3. 已完成内容

### 3.1 第四章主 Actor 已重建

当前主模型文件：

- `src/lc/planning/models/multi_uav_model.py`

已完成内容：

- 支持三类结构化输入：
  - `self_state`
  - `obstacles`
  - `neighbors`
- 支持两阶段结构：
  - 第一阶段：避障
  - 第二阶段：协作
- 第一阶段先对 `self + obstacles` 做嵌入和多头注意力
- 第二阶段将第一阶段得到的安全特征 `safe_feature` 送入协作分支，再与邻居信息建模
- 输出：
  - `avoid_action`
  - `final_action`
- 当前输出激活已经调整为：
  - `tanh`
  - 动作范围 `[-1, 1]`

当前还额外支持：

- `dict` 输入
- 扁平 `flat tensor` 输入
- `policy_stages()`，可直接拿到分阶段中间特征
- `last_attention`，便于后续做注意力可视化

### 3.2 训练闭环已补齐

当前主训练器文件：

- `src/lc/planning/trainers/planning_trainer.py`

已完成内容：

- 经验采集
- replay sampling
- critic update
- actor update
- target network soft update
- 训练历史记录
- 结构化主模型训练
- MLP 基线训练

当前训练器已经不是只会前向评估的骨架，而是一个可运行的连续动作训练器。

当前进一步完成：

- 已从基础 actor-critic 方向收敛为 `TD3` 风格训练框架
- 已接入双 Critic：
  - `critic_1`
  - `critic_2`
- 已接入：
  - target policy smoothing
  - delayed policy update
  - twin critic min target

### 3.3 Curriculum 已接入训练流程

当前文件：

- `src/lc/planning/curriculum/scheduler.py`

已完成内容：

- 固定课程主任务阶段：
  - `guidance`
  - `avoidance`
  - `cooperation`
- 根据如下指标调整阶段：
  - `success_rate`
  - `collision_rate`
  - `occupancy_error`
- 记录：
  - 当前阶段
  - 阶段切换历史
  - 分阶段平均指标

### 3.4 Pyramid Replay 已做成可用版本

当前文件：

- `src/lc/planning/memory/pyramid.py`

已完成内容：

- `SumTree`
- 标准 `PrioritizedReplayBuffer`
- 三层金字塔式经验回放：
  - `td_error`
  - `stage_balanced`
  - `rare_event`
- 分层采样比例
- 优先级回写更新
- 样本统计输出
- 阶段保留统计

注意：

- 当前实现已经具备 priority update 链路，但各层优先级公式仍然是论文口径下的近似实现，不是最终定稿公式。

### 3.5 奖励已拆分为六个子项

当前文件：

- `src/lc/planning/rewards/planning_reward.py`

已完成内容：

- `target_reward`
- `avoidance_reward`
- `collaboration_reward`
- `recovery_reward`
- `smoothness_penalty`
- `consistency_penalty`

当前训练和实验输出里已经能看到这些子项的均值。

### 3.6 第四章实验矩阵已扩成五组

当前文件：

- `src/lc/planning/experiments/compare.py`

当前已能跑的组：

- `task_decomposed`
- `single_stream_mlp`
- `without_curriculum`
- `without_pyramid_per`
- `uniform_replay`

### 3.7 图表输出已扩展

当前文件：

- `src/lc/planning/plotting/plots.py`

当前已能输出：

- `ablation_comparison.svg`
- `convergence_curve.svg`
- `success_collision_curve.svg`
- `formation_occupancy_curve.svg`
- `curriculum_schedule.svg`
- `complexity_generalization.svg`
- `trajectory.svg`
- `attention_heatmap.svg`

同时还保留了基础柱状图：

- `success_rate.svg`
- `collision_rate.svg`
- `formation_error.svg`
- `reward.svg`

### 3.8 已验证通过

已跑过的验证：

- `PYTHONPATH=...\\src python -m pytest tests/test_new_architecture.py -q`

当前结果：

- `3 passed`

并且第四章实验入口已直接跑通：

- `run_planning_comparison()`

当前输出目录示例：

- `outputs/planning/medium/stage_1/`

当前进一步验证过：

- Actor 支持：
  - `dict` 输入
  - 扁平 `flat tensor` 输入
- Actor 输出已固定为：
  - 二维速度动作 `(vx, vy)`
  - 范围 `[-1, 1]`

## 4. 当前仍缺少的内容

### 4.1 训练算法口径还没有最终定版

该项目前已由用户确认并按建议推进为：

- `TD3`

### 4.2 环境还是论文导向的近似环境

当前文件：

- `src/lc/planning/envs/swarm.py`

当前能用于训练和对比，但还不是高保真正式环境。

当前已确认并已落实到代码的大方向：

- 平面二维任务
- 不考虑 `z` 轴
- `self_state`：
  - 目标相对位置 `(dx, dy)`
  - 自身速度 `(vx, vy)`
- `obstacles`：
  - 最近 `n` 个障碍物 `(x, y, r)`
- `neighbors`：
  - 最近 `n` 个邻机 `(x, y)`
- 动作输出：
  - 二维速度向量 `(vx, vy)`
- 障碍物：
  - 固定直径柱体抽象

现在仍缺少你给出的明确信息：

- 无人机真实状态定义
- 障碍物状态定义
- 邻居状态定义
- 目标状态如何进入观测
- 动态障碍和动态目标的真实运动模型
- episode 终止条件的正式定义
- success 判定的正式定义

### 4.3 奖励公式仍是近似设计

虽然奖励已经拆项，但目前仍缺：

- 每个子项的论文正式公式
- 每个子项的权重
- 哪些项是奖励，哪些项是惩罚
- 是否存在终止奖励 / 成功奖励 / 碰撞一次性惩罚

如果你不给正式口径，我只能继续做工程上合理但不一定等于论文原意的版本。

### 4.4 Curriculum 规则还需要论文确认

当前阶段切换逻辑是可运行的，但不是最终论文定稿逻辑。

还缺少你补充：

- 每个阶段具体对应什么场景配置
- 升级阈值
- 降级阈值
- 是否允许降级
- 是否要保留阶段驻留时间
- 是否按滑动窗口统计指标

### 4.5 Pyramid-PER 还不是最终论文版

当前 replay memory 已可用，但还缺少这些定稿信息：

- 优先级正式公式
- 是否必须使用 TD 误差作为主权重
- 是否要按阶段单独建池
- 旧阶段样本保留比例
- 分层采样比例是否固定
- 是否需要 importance sampling 修正

### 4.6 注意力热力图目前只是代理输出

当前图能生成，但严格说还不是论文中的正式 attention 可解释结果。

如果你希望它成为论文图，需要你确认：

- 热力图要看哪一层
- 看哪一个 head
- 展示障碍注意力、邻居注意力，还是两者都展示
- 用单样本图还是批量平均图

### 4.7 结果数值目前不能作为论文最终结论

当前训练和实验已经能跑，但当前数值主要用于：

- 检查主链是否打通
- 检查实验输出是否完整
- 检查模块是否联动

还不能直接作为论文最终实验结论，因为环境、奖励、训练算法、课程逻辑都还没被你最终拍板。

## 5. 当前代码里最值得继续写的模块

如果现在继续推进第四章，我建议优先顺序是：

1. 定稿训练算法口径
2. 定稿观测与动作语义
3. 定稿奖励公式
4. 定稿 curriculum 阶段与切换规则
5. 定稿 Pyramid-PER 优先级与采样
6. 最后再做论文级图和正式实验批量运行

## 6. 当前文档状态说明

当前这些文档已经明显落后于代码状态：

- `docs/chapter4_readme.md`
- `docs/chapter4_execution_plan.md`
- `docs/chapter4_experiment_matrix.md`

它们仍然把很多内容写成“未完成骨架”，但代码里实际上已经补了一部分。

建议：

- 保留这些文档作为最初设计记录
- 后续以本文档作为“当前真实进度记录”

## 7. 现在最需要你补充确认的点

下面这些点中，已有一部分已经由用户确认，剩余部分继续保留待定。

### 7.1 训练算法

- 当前建议：
  - `TD3`
- 选择原因：
  - 第四章是连续动作输出
  - Actor / Critic 都是注意力结构，函数逼近能力较强，更容易出现 Q 值高估
  - 相比基础 DDPG，`TD3` 更适合做当前这种连续动作、多模块耦合、训练容易震荡的规划任务
  - `TD3` 的双 Critic、target policy smoothing、delayed policy update 更适合后续课程学习和三层经验回放联动
- 当前状态：
  - 代码里暂时还是 DDPG 风格闭环
  - 后续应收敛为 `TD3`

### 7.2 观测定义

- 已确认：
  - `self_state` 应包含：
    - 目标点相对自身位置 `(dx, dy)`
    - 自身速度 `(vx, vy)`
  - `obstacles` 应包含最近 `n` 个障碍物：
    - 每个障碍物 `(x, y, r)`
  - `neighbors` 应包含最近 `n` 个邻机：
    - 每个邻机 `(x, y)`
- 当前理解：
  - 第四章不考虑 `z` 轴
  - 目标信息默认并入 `self_state`

### 7.3 动作定义

- 已确认：
  - 最终动作输出为二维速度向量：
    - `vx`
    - `vy`
- 当前理解：
  - 规划层输出的是二维平面内的高层速度意图
  - 不再使用“四维动作”版本
- 当前待实现细节：
  - `avoid_action` 是否也直接定义为二维安全速度参考
  - `final_action` 是否为 `safe_action + coop_residual` 的最终融合速度向量
  - 目前按这个理解继续实现是合理的

### 7.4 奖励定义

- 六个奖励子项的正式公式或文字规则是什么？
- 每一项的权重是多少？

### 7.5 Curriculum

- 已确认的大方向：
  - 课程主任务按三阶段组织：
    - `定位/占位`
    - `避障`
    - `合作`
  - 每个阶段内部难度递增
  - 递增方式包括：
    - 定位从易到难
    - 障碍物从少到多
    - 协作从易到难
  - 障碍物默认可设为固定直径柱体
  - 第四章规划仅考虑二维平面
- 当前还缺少：
  - 每一阶段对应的具体场景参数
  - 切换阈值
  - 滑动窗口长度
  - 是否允许降级退避的精确规则

### 7.6 Pyramid-PER

- 已确认的大方向：
  - 第四章经验回放不是普通单层 PER
  - 而是“三层金字塔结构”
  - 第 1 层：
    - TD 误差主导
  - 第 2 层：
    - 重新计算 TD
    - 小于额定值的可过滤
    - 再叠加样本重要程度
    - 且重要程度与课程阶段相关
  - 第 3 层：
    - 稀有事件程度主导
- 这意味着后续应优先补：
  - `SumTree`
  - 标准 `PER`
  - 再叠加三层金字塔筛选逻辑
- 当前还缺少：
  - 每层容量比例
  - 每层优先级正式公式
- 第二层“额定值”的定义
- 稀有事件的正式判据

## 4.8 当前实验结果仍然只是“开发验证结果”

当前 `outputs/planning/...` 里的结果已经可以用于：

- 检查代码主链是否打通
- 检查 TD3 + Actor/Critic + replay 是否联动
- 检查图表和日志是否完整输出

但还不能用于论文最终实验结论，原因包括：

- 奖励权重未定稿
- 课程阈值未定稿
- 金字塔 PER 各层公式未定稿
- 环境仍是论文导向抽象环境，不是最终高保真版本

### 7.7 注意力图

- 论文里最终要展示哪种热力图？

## 8. 一句话总结

第四章目前已经不再是空骨架，而是一个“主链打通、关键口径已部分确认、接下来应按 TD3 + 二维速度动作 + 三阶段课程 + 三层金字塔 PER 收敛”的实现版本。
