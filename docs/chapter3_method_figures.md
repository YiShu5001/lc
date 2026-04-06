# 第三章方法插图说明

## 图 1：DDPG-LADRC 总体框架图

- 文件：
  [ddpg_ladrc_framework.svg](/C:/context_mine/mine_code/GIT_Projects/lc/docs/figures/chapter3/ddpg_ladrc_framework.svg)
- 图题：
  `图3-x DDPG-LADRC单轴位置跟踪控制框架`
- 图注：
  `该框架将深度确定性策略梯度与LADRC位置环相结合。智能体基于跟踪状态输出三个参数增量，在线调整ωc、ωo与b0，自适应LADRC再生成控制输入，从而实现单轴位置跟踪与扰动抑制。`

框内文字固定为：
- `参考输入 r(t)`
- `扰动 d(t)`
- `单轴被控对象`
- `系统输出 y(t)`
- `状态构造`
- `位置误差 e、速度 v、扰动项、参考值、归一化时间`
- `DDPG Actor`
- `三参数动作`
- `Δωc, Δωo, Δb0`
- `自适应 LADRC 位置环`
- `控制量 u(t)`
- `奖励函数`
- `r = -|位置误差|`
- `Replay Buffer`
- `Critic`
- `目标网络`
- `软更新`

箭头关系固定为：
- `参考输入 -> 自适应 LADRC`
- `扰动 -> 单轴被控对象`
- `自适应 LADRC -> 控制量 -> 单轴被控对象 -> 系统输出`
- `系统输出 -> 状态构造 -> DDPG Actor`
- `DDPG Actor -> 三参数动作 -> 自适应 LADRC`
- `系统输出 -> 奖励函数`
- `状态、动作、奖励、下一状态 -> Replay Buffer`
- `Replay Buffer -> Critic`
- `Replay Buffer -> Actor`
- `Actor/Critic -> 目标网络更新`

图中唯一创新点注释固定为：
- `强化学习在线调节位置环参数，而非直接生成控制指令`

## 图 2：mDDPG 增强机制图

- 文件：
  [mddpg_enhancement.svg](/C:/context_mine/mine_code/GIT_Projects/lc/docs/figures/chapter3/mddpg_enhancement.svg)
- 图题：
  `图3-y mDDPG增强机制示意图`
- 图注：
  `mDDPG在基础DDPG-LADRC框架上引入状态堆叠、动作保持和n-step回报三项增强机制，以提升策略对短时历史信息、控制平滑性与时序收益的利用能力。`

框内文字固定为：
- `DDPG-LADRC`
- `单步状态 / 每步动作更新 / 1-step回报`
- `mDDPG-LADRC`
- `n步状态堆叠 / 动作保持 / n-step回报`
- `过去 n 个状态`
- `状态增强`
- `将过去n个时刻状态拼接为策略输入`
- `动作保持`
- `动作保持 n 个采样时刻`
- `同一参数动作保持n个采样时刻`
- `奖励自举`
- `过去 n 步奖励`
- `基于过去n个时刻奖励构造n-step目标`
- `LADRC 参数持续生效`
- `增强经验样本`
- `Replay Buffer`
- `Actor/Critic 更新`

箭头关系固定为：
- `过去 n 个状态 -> 状态增强 -> Actor 输入`
- `Actor 输出 -> 动作保持 n 个采样时刻 -> LADRC 参数持续生效`
- `过去 n 步奖励 -> 奖励自举 -> 增强经验样本`
- `状态增强 + 动作保持 + n-step目标 -> Replay Buffer`
- `Replay Buffer -> Actor/Critic 更新`

## 文件落点

本次插图默认落在：
- `docs/figures/chapter3/`

后续如果要进论文目录，可复制到：
- `papers/figures/chapter3/`
