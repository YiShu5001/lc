# Gym_env 环境模块文档

## 目录结构

```
Gym_env/
├── BaseEnv.py                    # 基础环境接口定义
├── __init__.py                   # 模块初始化文件
├── examples/                     # 示例代码目录
│   ├── pid.py                    # PID控制器轨迹跟踪示例
│   ├── learn.py                  # 强化学习训练示例（PPO）
│   ├── downwash.py               # 下降气流效应演示
│   ├── pid_velocity.py           # 速度控制示例
│   ├── cf.py                     # Crazyflie软件在环控制示例
│   ├── beta.py                   # Betaflight控制示例
│   ├── funx_att.py               # 姿态控制示例
│   ├── 0ma_to_1.py               # 多智能体跟随示例
│   ├── debug.py                  # PyBullet调试工具
│   ├── 调试控制参数.py            # 控制参数调试工具
│   ├── 控制参数/                  # 控制参数输出目录
│   │   └── *.xlsx                # 控制器参数Excel文件
│   └── results/                  # 仿真结果输出目录
│       └── *.csv, *.npy          # 日志数据和结果文件
└── gym_pybullet_drones/          # PyBullet无人机仿真环境
    ├── envs/                     # 环境实现
    │   ├── BaseAviary.py         # 基础无人机环境类
    │   ├── BaseAviarySimple.py   # 简化版基础环境
    │   ├── BaseRLAviary.py       # 强化学习基础环境
    │   ├── HoverAviary.py        # 单智能体悬停任务
    │   ├── MultiHoverAviary.py    # 多智能体悬停任务
    │   ├── VelocityAviary.py     # 速度控制任务
    │   ├── BetaAviary.py         # Beta测试环境
    │   ├── CFAviary.py           # Crazyflie无人机环境
    │   ├── CtrlAviary.py         # 控制器测试环境
    │   ├── 00RLAviary.py         # RL环境示例
    │   └── 00添加物体.py          # 添加障碍物示例
    ├── control/                  # 控制器模块
    │   ├── BaseControl.py        # 控制器基类
    │   ├── DSLPIDControl.py      # DSL PID控制器
    │   ├── LADRC.py              # 线性自抗扰控制器
    │   ├── CTBRControl.py        # CTBR控制器
    │   └── adrc_pid.py           # ADRC-PID控制器
    ├── utils/                    # 工具函数
    │   ├── enums.py              # 枚举类型定义
    │   ├── utils.py              # 通用工具函数
    │   ├── Logger.py             # 日志记录器
    │   └── RLg/                  # 强化学习工具
    │       └── Buffer.py         # 经验回放缓冲区
    └── assets/                   # 资源文件（URDF模型等）
```

## 核心模块说明

### 1. BaseEnv.py - 基础环境接口

定义了强化学习训练框架所需的环境接口标准。

#### 1.1 EnvConfig 配置类

环境配置数据类，用于存储环境相关参数。

**属性：**

- `env_id` (str): 环境标识符
- `num_envs` (int): 并行环境数量，默认为1（单环境）
- `seed` (int): 随机种子，默认为0
- `capture_video` (bool): 是否录制视频，默认为False
- `run_name` (str): 日志/视频命名，默认为"exp"
- `max_episode_steps` (Optional[int]): 可选的最大回合步数，用于强制截断

**使用示例：**

```python
from Gym_env import EnvConfig

cfg = EnvConfig(
    env_id="HoverAviary-v0",
    num_envs=4,
    seed=42,
    capture_video=True,
    run_name="hover_experiment",
    max_episode_steps=1000
)
```

#### 1.2 VectorEnvLike 抽象基类

向量化环境的抽象接口，定义了训练代码期望的最小接口规范。不绑定特定的gymnasium实现，便于替换后端。

**抽象属性：**

- `num_envs` (int): 返回并行环境数量
- `obs_shape` (Tuple[int, ...]): 返回观测维度形状，例如 `(4,)` 或 `(84, 84, 3)`
- `action_shape` (Tuple[int, ...]): 返回动作张量形状
  - 离散动作：`()` 或 `(1,)`
  - 连续动作：`(act_dim,)`
- `is_discrete` (bool): 返回动作空间是否为离散类型
- `action_dim` (int): 返回动作维度/动作数量
  - 离散：动作个数 n
  - 连续：动作维度 act_dim

**抽象方法：**

##### `reset(seed: Optional[int] = None) -> np.ndarray`

重置环境到初始状态。

**参数：**

- `seed` (Optional[int]): 可选的随机种子

**返回：**

- `obs` (np.ndarray): 初始观测，形状为 `(num_envs, *obs_shape)`

##### `step(actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]`

与环境交互一步。

**参数：**

- `actions` (np.ndarray): 动作数组
  - 离散动作：形状为 `(num_envs,)`，类型为 int
  - 连续动作：形状为 `(num_envs, action_dim)`，类型为 float

**返回：**

- `next_obs` (np.ndarray): 下一步观测，形状为 `(num_envs, *obs_shape)`
- `rewards` (np.ndarray): 奖励值，形状为 `(num_envs,)`
- `dones` (np.ndarray): 回合结束标志，形状为 `(num_envs,)`，True表示该环境回合结束
- `infos` (dict): 额外信息字典，可包含episode return、length等

**使用示例：**

```python
from Gym_env import VectorEnvLike
import numpy as np

class MyEnv(VectorEnvLike):
    # 实现所有抽象属性和方法
    @property
    def num_envs(self) -> int:
        return 4
    
    @property
    def obs_shape(self):
        return (12,)
    
    # ... 其他属性实现
    
    def reset(self, seed=None):
        # 实现重置逻辑
        return np.zeros((self.num_envs, *self.obs_shape))
    
    def step(self, actions):
        # 实现步进逻辑
        return obs, rewards, dones, {}
```

#### 1.3 EnvFactory 环境工厂基类

环境工厂抽象基类，负责根据 `EnvConfig` 构造 `VectorEnvLike` 对象。可以创建不同的工厂实现（如GymEnvFactory、PettingZooFactory等）。

**方法：**

##### `__init__(cfg: EnvConfig)`

初始化工厂。

**参数：**

- `cfg` (EnvConfig): 环境配置对象

##### `build() -> VectorEnvLike`（抽象方法）

构造并返回一个可用于训练的向量化环境对象。

**返回：**

- `VectorEnvLike`: 向量化环境实例

**使用示例：**

```python
from Gym_env import EnvFactory, EnvConfig, VectorEnvLike

class GymEnvFactory(EnvFactory):
    def build(self) -> VectorEnvLike:
        # 根据cfg创建并返回环境
        return gym.make(self.cfg.env_id)

cfg = EnvConfig(env_id="HoverAviary-v0")
factory = GymEnvFactory(cfg)
env = factory.build()
```

---

### 2. gym_pybullet_drones/ - PyBullet无人机仿真环境

基于PyBullet物理引擎的无人机仿真环境，支持单智能体和多智能体强化学习任务。

#### 2.1 环境类型 (envs/)

##### BaseAviary - 基础无人机环境类

所有无人机环境的基类，继承自 `gymnasium.Env`。

**主要功能：**

- 初始化PyBullet物理引擎
- 加载无人机URDF模型
- 管理多无人机实例
- 处理物理仿真步进
- 提供观测和状态信息
- 支持障碍物添加
- 支持视频录制

**关键参数：**

- `drone_model` (DroneModel): 无人机模型类型（CF2X、CF2P、RACE）
- `num_drones` (int): 无人机数量
- `neighbourhood_radius` (float): 邻域半径，用于计算无人机邻接矩阵
- `initial_xyzs` (ndarray): 初始位置，形状为 `(NUM_DRONES, 3)`
- `initial_rpys` (ndarray): 初始姿态（欧拉角），形状为 `(NUM_DRONES, 3)`
- `physics` (Physics): 物理引擎类型（PYB、DYN、PYB_GND等）
- `pyb_freq` (int): PyBullet仿真频率，默认240Hz
- `ctrl_freq` (int): 控制频率，默认240Hz
- `gui` (bool): 是否显示GUI
- `record` (bool): 是否录制视频
- `obstacles` (bool): 是否添加障碍物
- `user_debug_gui` (bool): 是否显示调试GUI（轴、RPM滑块等）
- `vision_attributes` (bool): 是否为视觉观测分配属性

**关键方法：**

- `reset()`: 重置环境
- `step()`: 执行一步仿真
- `close()`: 关闭环境
- `_computeObs()`: 计算观测（需子类实现）
- `_computeReward()`: 计算奖励（需子类实现）
- `_computeDone()`: 判断是否结束（需子类实现）
- `_computeInfo()`: 计算额外信息（需子类实现）

##### BaseRLAviary - 强化学习基础环境

继承自 `BaseAviary`，专门为强化学习设计的环境基类。

**主要特性：**

- 支持多种动作类型（RPM、PID、VEL等）
- 支持多种观测类型（KIN、RGB）
- 集成PID控制器（当动作类型为PID或VEL时）
- 动作缓冲区管理（最近0.5秒的动作历史）

**关键参数：**

- `obs` (ObservationType): 观测类型
  - `KIN`: 运动学信息（位置、速度、姿态等）
  - `RGB`: RGB相机图像
- `act` (ActionType): 动作类型
  - `RPM`: 直接控制四个电机的RPM
  - `PID`: 三维位置控制（输入目标位置）
  - `VEL`: 速度控制（三维速度方向+速度系数）
  - `ONE_D_RPM`: 一维RPM控制（所有电机相同）
  - `ONE_D_PID`: 一维PID控制

**动作空间：**

- `RPM` / `VEL`: Box形状 `(NUM_DRONES, 4)`
- `PID`: Box形状 `(NUM_DRONES, 3)`
- `ONE_D_RPM` / `ONE_D_PID`: Box形状 `(NUM_DRONES, 1)`

**观测空间：**

- `KIN`: Box形状，包含位置、速度、姿态、角速度等信息
- `RGB`: Box形状 `(NUM_DRONES, H, W, 3)`，RGB图像

##### HoverAviary - 单智能体悬停任务

单智能体强化学习环境，任务目标是让无人机悬停在指定位置。

**奖励函数：**

- 基于位置误差的奖励
- 鼓励稳定悬停

**使用示例：**

```python
from gym_pybullet_drones.envs import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

env = HoverAviary(
    drone_model=DroneModel.CF2X,
    initial_xyzs=np.array([[0, 0, 1]]),
    physics=Physics.PYB,
    pyb_freq=240,
    ctrl_freq=30,
    gui=True,
    obs=ObservationType.KIN,
    act=ActionType.RPM
)
```

##### MultiHoverAviary - 多智能体悬停任务

多智能体强化学习环境，支持多个无人机同时执行悬停任务。

**特点：**

- 支持leader-follower模式
- 每个无人机有独立的目标位置
- 默认回合长度为8秒

**使用示例：**

```python
from gym_pybullet_drones.envs import MultiHoverAviary

env = MultiHoverAviary(
    drone_model=DroneModel.CF2X,
    num_drones=3,
    neighbourhood_radius=np.inf,
    obs=ObservationType.KIN,
    act=ActionType.RPM
)
```

##### VelocityAviary - 速度控制任务

速度控制任务环境，训练无人机按照指定速度飞行。

##### BetaAviary - Beta测试环境

用于测试和实验的环境。

##### CFAviary - Crazyflie无人机环境

专门针对Crazyflie无人机的环境配置。

##### CtrlAviary - 控制器测试环境

用于测试不同控制器的环境。

#### 2.2 控制器模块 (control/)

##### BaseControl - 控制器基类

所有控制器的基类，定义了控制器的通用接口。

**主要方法：**

- `__init__(drone_model, g=9.8)`: 初始化控制器
- `reset()`: 重置控制器状态
- `computeControl()`: 计算控制输出（需子类实现）
- `computeControlFromState()`: 从状态计算控制（接口方法）

**关键属性：**

- `DRONE_MODEL`: 无人机模型类型
- `GRAVITY`: 重力（M*g）
- `KF`: 推力系数（RPM到推力的转换系数）
- `KM`: 力矩系数（RPM到力矩的转换系数）

##### DSLPIDControl - DSL PID控制器

基于DSL（Dynamic Soaring Logic）的PID控制器实现。

**功能：**

- 位置控制
- 姿态控制
- 速度控制

##### LADRC - 线性自抗扰控制器

线性自抗扰控制（Linear Active Disturbance Rejection Control）实现。

##### CTBRControl - CTBR控制器

CTBR控制算法实现。

##### adrc_pid.py - ADRC-PID控制器

结合ADRC和PID的混合控制器。

#### 2.3 工具模块 (utils/)

##### enums.py - 枚举类型定义

定义了系统中使用的所有枚举类型。

**DroneModel - 无人机模型枚举：**

- `CF2X`: Bitcraze Crazyflie 2.0 X型配置
- `CF2P`: Bitcraze Crazyflie 2.0 +型配置
- `RACE`: Racer无人机 X型配置

**Physics - 物理引擎类型枚举：**

- `PYB`: 基础PyBullet物理更新
- `DYN`: 显式动力学模型
- `PYB_GND`: PyBullet物理更新（含地面效应）
- `PYB_DRAG`: PyBullet物理更新（含空气阻力）
- `PYB_DW`: PyBullet物理更新（含下降气流）
- `PYB_GND_DRAG_DW`: PyBullet物理更新（含地面效应、阻力和下降气流）

**ActionType - 动作类型枚举：**

- `RPM`: RPM控制（四个电机）
- `PID`: PID位置控制
- `VEL`: 速度控制（使用PID）
- `ONE_D_RPM`: 一维RPM控制
- `ONE_D_PID`: 一维PID控制

**ObservationType - 观测类型枚举：**

- `KIN`: 运动学信息（姿态、线速度、角速度等）
- `RGB`: RGB相机图像

**ImageType - 图像类型枚举：**

- `RGB`: 红绿蓝（含alpha通道）
- `DEP`: 深度图
- `SEG`: 基于对象ID的分割图
- `BW`: 黑白图

##### utils.py - 通用工具函数

**sync(i, start_time, timestep)**
同步仿真步进与实时时钟。

**参数：**

- `i` (int): 当前仿真迭代次数
- `start_time` (timestamp): 仿真开始时间戳
- `timestep` (float): 期望的实时步长

**功能：** 调用 `time.sleep()` 暂停运行过快的循环，使其与期望的实时步长同步。

**str2bool(val)**
将字符串转换为布尔值。

**参数：**

- `val` (str | bool): 输入值（可能是字符串）

**返回：**

- `bool`: 解释为True或False

**支持的字符串：** 'yes', 'true', 't', 'y', '1' → True；'no', 'false', 'f', 'n', '0' → False

##### Logger.py - 日志记录器

用于记录仿真过程中的数据和事件。

##### RLg/Buffer.py - 经验回放缓冲区

强化学习经验回放缓冲区实现，用于存储和采样经验数据。

---

## 使用示例

### 基本使用

```python
from gym_pybullet_drones.envs import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
import numpy as np

# 创建环境
env = HoverAviary(
    drone_model=DroneModel.CF2X,
    initial_xyzs=np.array([[0, 0, 1]]),
    physics=Physics.PYB,
    pyb_freq=240,
    ctrl_freq=30,
    gui=True,
    obs=ObservationType.KIN,
    act=ActionType.RPM
)

# 重置环境
obs, info = env.reset()

# 运行一个回合
for _ in range(1000):
    # 随机动作（实际使用时应该由策略网络生成）
    action = env.action_space.sample()
    
    # 执行一步
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

# 关闭环境
env.close()
```

### 多智能体使用

```python
from gym_pybullet_drones.envs import MultiHoverAviary

env = MultiHoverAviary(
    drone_model=DroneModel.CF2X,
    num_drones=3,
    obs=ObservationType.KIN,
    act=ActionType.RPM
)

obs, info = env.reset()
# obs形状: (3, obs_dim) - 3个无人机的观测

# 动作形状: (3, 4) - 3个无人机，每个4个电机RPM
actions = env.action_space.sample()
obs, rewards, terminated, truncated, info = env.step(actions)
```

### 与训练框架集成

```python
from Gym_env import EnvConfig, EnvFactory, VectorEnvLike
from gym_pybullet_drones.envs import HoverAviary

class GymPybulletFactory(EnvFactory):
    def build(self) -> VectorEnvLike:
        # 创建并包装环境
        env = HoverAviary(
            drone_model=DroneModel.CF2X,
            gui=self.cfg.capture_video,
            obs=ObservationType.KIN,
            act=ActionType.RPM
        )
        # 包装为VectorEnvLike接口
        return WrappedEnv(env)

cfg = EnvConfig(
    env_id="HoverAviary-v0",
    num_envs=1,
    seed=42
)
factory = GymPybulletFactory(cfg)
env = factory.build()
```

---

## 注意事项

1. **频率设置**：`pyb_freq` 必须是 `ctrl_freq` 的整数倍
2. **动作空间**：不同动作类型对应不同的动作空间维度
3. **观测空间**：KIN观测包含位置、速度、姿态等信息，维度取决于无人机数量
4. **多智能体**：多无人机环境返回的观测和奖励都是数组形式
5. **物理引擎**：不同物理引擎类型会影响仿真的真实性和计算开销
6. **GUI模式**：启用GUI会显著降低仿真速度，建议训练时关闭

---

## 依赖项

- `gymnasium`: OpenAI Gym接口
- `pybullet`: 物理仿真引擎
- `numpy`: 数值计算
- `PIL`: 图像处理（用于RGB观测）

---

## 扩展开发

### 创建自定义环境

1. 继承 `BaseRLAviary` 或 `BaseAviary`
2. 实现 `_computeObs()`, `_computeReward()`, `_computeDone()`, `_computeInfo()` 方法
3. 在 `__init__` 中设置任务特定的参数

### 创建自定义控制器

1. 继承 `BaseControl`
2. 实现 `computeControl()` 方法
3. 在环境初始化时传入控制器实例

---

---

## 3. Examples 示例代码

`examples/` 目录包含了多个实用的示例脚本，演示如何使用环境模块进行仿真、控制和强化学习训练。

### 3.1 控制器示例

#### pid.py - PID控制器轨迹跟踪示例

演示如何使用 `DSLPIDControl` 控制器进行轨迹跟踪控制。

**主要功能：**

- 使用 `CtrlAviary` 环境进行仿真
- 实现圆形、直线、固定点等多种轨迹跟踪
- 支持多无人机同时控制
- 记录和可视化仿真结果

**关键特性：**

- 可配置轨迹类型（圆形、直线、固定点）
- 支持自定义轨迹周期和路径点数量
- 使用Logger记录仿真数据
- 支持数据导出为CSV格式
- 自动生成可视化图表

**使用示例：**

```bash
python pid.py --num_drones 1 --duration_sec 12 --gui True
```

**参数说明：**

- `--drone`: 无人机模型（默认：CF2X）
- `--num_drones`: 无人机数量（默认：1）
- `--physics`: 物理引擎类型（默认：pyb_gnd_drag_dw）
- `--simulation_freq_hz`: 仿真频率（默认：240Hz）
- `--control_freq_hz`: 控制频率（默认：48Hz）
- `--duration_sec`: 仿真时长（默认：12秒）
- `--gui`: 是否显示GUI（默认：True）
- `--plot`: 是否绘制结果图表（默认：True）

#### funx_att.py - 姿态控制示例

演示姿态控制功能，使用LADRC控制器进行姿态跟踪。

**主要功能：**

- 姿态控制演示
- 使用LADRC（线性自抗扰控制）控制器
- 支持下降气流效应（PYB_DW物理引擎）

#### 调试控制参数.py - 控制参数调试工具

用于调试和优化控制器参数的脚本。

**主要功能：**

- 测试不同控制器参数配置
- 记录控制参数变化过程
- 导出控制参数到Excel文件
- 分析控制器性能

**输出文件：**

- `控制参数/yaw无人机L.xlsx`: 无人机控制参数日志
- `控制参数/yaw控制器L.xlsx`: LADRC控制器参数日志

### 3.2 强化学习示例

#### learn.py - 强化学习训练示例

演示如何使用 `stable-baselines3` 进行强化学习训练。

**主要功能：**

- 使用PPO算法训练单智能体或多智能体策略
- 支持 `HoverAviary` 和 `MultiHoverAviary` 环境
- 自动保存最佳模型
- 评估训练后的策略性能
- 支持KIN和RGB两种观测类型
- 支持多种动作类型（RPM、PID、VEL、ONE_D_RPM、ONE_D_PID）

**使用示例：**

```bash
# 单智能体训练
python learn.py --multiagent false

# 多智能体训练
python learn.py --multiagent true --gui False
```

**关键特性：**

- 自动回调机制，达到目标奖励后停止训练
- 定期评估策略性能
- 保存最佳模型和最终模型
- 训练后自动测试并可视化结果
- 支持视频录制

**参数说明：**

- `--multiagent`: 是否使用多智能体环境（默认：True）
- `--gui`: 是否显示GUI（默认：True）
- `--record_video`: 是否录制视频（默认：False）
- `--output_folder`: 输出文件夹（默认：results）

**训练目标奖励：**

- ONE_D_RPM动作类型：单智能体474.15，多智能体949.5
- 其他动作类型：单智能体467.0，多智能体920.0

### 3.3 物理效应演示

#### downwash.py - 下降气流效应演示

演示多无人机之间的下降气流（downwash）效应。

**主要功能：**

- 展示无人机之间的空气动力学相互作用
- 使用 `PYB_DW` 物理引擎模拟下降气流
- 演示多无人机编队飞行时的相互影响
- 支持圆形轨迹跟踪

**使用示例：**

```bash
python downwash.py --duration_sec 40
```

**物理效应说明：**

- 当一架无人机位于另一架无人机上方时，会产生下降气流
- 下方的无人机会受到额外的向下的力
- 影响无人机的稳定性和控制性能

#### 0ma_to_1.py - 多智能体跟随示例

演示多智能体之间的跟随行为，展示leader-follower模式。

**主要功能：**

- 实现多无人机跟随控制
- 后一架无人机跟随前一架无人机的位置
- 演示多智能体协调控制

### 3.4 速度控制示例

#### pid_velocity.py - 速度控制示例

演示如何使用 `VelocityAviary` 环境进行速度控制。

**主要功能：**

- 使用速度输入控制无人机
- 内部PID控制器自动跟踪目标速度
- 支持多无人机同时控制
- 可配置不同的速度轨迹

**使用示例：**

```bash
python pid_velocity.py --num_drones 4 --duration_sec 5
```

**速度输入格式：**

- 形状：`(num_drones, 4)`
- 内容：`[vx, vy, vz, speed_coefficient]`
  - `vx, vy, vz`: 速度方向向量（归一化）
  - `speed_coefficient`: 速度系数（0-1之间）

### 3.5 特殊环境示例

#### cf.py - Crazyflie软件在环控制

演示Crazyflie无人机的软件在环（SITL）控制。

**主要功能：**

- 使用 `CFAviary` 环境
- 支持完整状态命令发送
- 实现复杂轨迹跟踪
- 与真实Crazyflie固件接口兼容

**前置要求：**

- 需要安装 `pycffirmware`
- 参考脚本注释中的安装说明

**使用示例：**

```bash
python cf.py --simulation_freq_hz 500 --control_freq_hz 25
```

#### beta.py - Betaflight控制示例

演示与Betaflight固件的集成控制。

**主要功能：**

- 使用 `BetaAviary` 环境
- 支持Betaflight SITL模式
- 从CSV文件读取轨迹
- 支持多无人机同时控制

**前置要求：**

- 需要设置Betaflight SITL环境
- 使用 `clone_bfs.sh` 脚本创建多个Betaflight实例

**使用示例：**

```bash
# 首先设置Betaflight环境
./gym_pybullet_drones/assets/clone_bfs.sh 2

# 然后运行示例
python beta.py --num_drones 2
```

### 3.6 调试工具

#### debug.py - PyBullet调试工具

用于调试PyBullet基本功能的简单脚本。

**主要功能：**

- 测试PyBullet基本功能
- 演示物体加载和物理仿真
- 显示坐标轴和调试信息
- 测试外部力和力矩施加

**使用场景：**

- 验证PyBullet安装是否正确
- 测试物理引擎基本功能
- 调试物体位置和姿态

### 3.7 示例文件总结

| 文件名 | 主要功能 | 使用的环境 | 使用的控制器 |
|--------|---------|-----------|-------------|
| `pid.py` | PID轨迹跟踪 | `CtrlAviary` | `DSLPIDControl` |
| `learn.py` | 强化学习训练 | `HoverAviary` / `MultiHoverAviary` | PPO算法 |
| `downwash.py` | 下降气流效应 | `CtrlAviary` (PYB_DW) | `LADRCControl` / `DSLPIDControl` |
| `pid_velocity.py` | 速度控制 | `VelocityAviary` | 内置PID |
| `cf.py` | Crazyflie SITL | `CFAviary` | `CTBRControl` |
| `beta.py` | Betaflight控制 | `BetaAviary` | `CTBRControl` |
| `funx_att.py` | 姿态控制 | `CtrlAviary` | `LADRCControl` |
| `0ma_to_1.py` | 多智能体跟随 | `CtrlAviary` | `DSLPIDControl` |
| `debug.py` | PyBullet调试 | 直接使用PyBullet | 无 |
| `调试控制参数.py` | 参数调试 | `CtrlAviary` | `LADRCControl` |

### 3.8 运行示例的通用步骤

1. **设置环境参数**

   ```bash
   # 根据需要修改脚本中的默认参数
   # 或使用命令行参数覆盖
   ```

2. **运行脚本**

   ```bash
   python <script_name>.py [--参数名 参数值]
   ```

3. **查看结果**
   - GUI模式：实时查看仿真过程
   - 日志文件：`results/` 目录下的CSV和日志文件
   - 图表：如果启用plot选项，会自动生成可视化图表

4. **分析数据**
   - 使用Logger保存的数据进行分析
   - 导出Excel文件进行进一步处理
   - 查看生成的图表

### 3.9 示例代码结构

所有示例脚本通常包含以下部分：

1. **导入模块**

   ```python
   from gym_pybullet_drones.envs import ...
   from gym_pybullet_drones.control import ...
   from gym_pybullet_drones.utils import ...
   ```

2. **默认参数定义**

   ```python
   DEFAULT_DRONE = DroneModel("cf2x")
   DEFAULT_GUI = True
   # ...
   ```

3. **主函数 `run()`**
   - 初始化环境
   - 初始化控制器
   - 初始化Logger
   - 运行仿真循环
   - 保存和可视化结果

4. **命令行参数解析**

   ```python
   parser = argparse.ArgumentParser(...)
   # 添加参数定义
   ```

5. **主程序入口**

   ```python
   if __name__ == "__main__":
       ARGS = parser.parse_args()
       run(**vars(ARGS))
   ```

### 3.10 常见问题

**Q: 如何修改轨迹类型？**
A: 在 `pid.py` 中修改 `TARGET_POS` 的计算逻辑，取消注释对应的轨迹类型。

**Q: 如何调整控制器参数？**
A: 在控制器初始化时传入参数，或修改控制器类的默认参数。

**Q: 如何保存仿真视频？**
A: 使用 `--record_video True` 参数，视频会保存在 `results/` 目录。

**Q: 如何提高仿真速度？**
A: 关闭GUI (`--gui False`)，降低仿真频率，或使用更简单的物理引擎。

**Q: 如何分析仿真数据？**
A: 使用Logger保存的数据，可以导出为CSV格式，然后用Excel或Python进行分析。

---

## 版本信息

- 基于 PyBullet 物理引擎
- 兼容 Gymnasium API
- 支持单智能体和多智能体强化学习
