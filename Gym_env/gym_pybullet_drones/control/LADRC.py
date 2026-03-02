import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class LADRC:
    """
    控制器运行的流程为： 根据当前计算出的状态量，通过TD、LESO、PD三个模块，计算出控制量u。
    结合目标状态值 传入TD，经过微分处理得到 v1, v2
    控制系统得到状态， 状态观测器的到估计值 x1, x2, x3
    传入LESO，经过PD计算得到 x1, x2, x3
    传入PD，计算出控制量u
    """
    def __init__(self, wc, b0, w0, h, r):
        try:
            self.wc = wc
            self.b0 = b0
            self.w0 = w0
            # TD
            self.h = h  # 事实上就是采样时间
            self.v1 = 0
            self.v2 = 0
            # LESO
            self.l1 = 0
            self.l2 = 0
            self.l3 = 0
            self.x1 = 0
            self.x2 = 0
            self.x3 = 0
            # PD
            self.kp = 0
            self.kd = 0

            self.r = r
            self.u = 0.0
            self.u0 = 0.0
        except Exception as e:
            print("LADRC初始化错误:", e)
            raise
    def up_para(self, wc, b0, w0):
        self.wc = wc
        self.b0 = b0
        self.w0 = w0

    def TD(self, target):
        fh = -self.r * self.r * (self.v1 - target) - 2 * self.r * self.v2
        self.v1 += self.v2 * self.h
        self.v2 += fh * self.h
        return self.v1, self.v2

    def LESO(self, current):
        self.l1 = 3 * self.w0
        self.l2 = 3 * self.w0 * self.w0
        self.l3 = self.w0 * self.w0 * self.w0

        err = current - self.x1
        self.x1 += (self.x2 + self.l1 * err) * self.h
        self.x2 += (self.x3 + self.l2 * err + self.b0 * self.u) * self.h
        self.x3 += self.l3 * err * self.h
        return self.x1, self.x2, self.x3

    def PD(self):
        self.kp = self.wc * self.wc
        self.kd = 2 * self.wc

        e1 = self.v1 - self.x1
        e2 = self.v2 - self.x2
        self.u0 = self.kp * e1 + self.kd * e2
        # 也补充上了最后一点的u0转变为u的过程
        self.u = (self.u0 - self.x3) / self.b0
        return self.u

    def __call__(self, target, current):
        target = np.array(target)
        current = np.array(current)
        self.TD(target)
        self.LESO(current)
        self.PD()
        return self.u
    def reset(self):
        self.u = 0



class LADRCControl(BaseControl):
    """
    控制器的参数导入还是按照通道来吧，每个通道为一组参数
    [wc, w0, b0, h, r]
    """
    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8,
                 *para: np.array
                 ):
        super().__init__(drone_model=drone_model, g=g)

        # 添加调试打印
        print("[DEBUG] 开始初始化 LADRC 控制器")

        # 保存参数
        self.logger ={
            'target_thrust': [],
            'target_thrust_x': [],
            'target_thrust_y': [],
            'target_thrust_z': [],
            'thrust': [],
            'target_rotation': [],
            'target_euler': [],
            'pos': [],
            'pos_x': [],
            'pos_y': [],
            'pos_z': [],
            'target_x': [],
            'target_y': [],
            'target_z': [],
        }

        # 型号兼容性检查（当前仅支持 Crazyflie 2.0 系列）
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()
        # 1. 默认参数：6个通道 [X, Y, Z, 横滚, 俯仰, 偏航]，每个通道为 [wc, b0, w0]
        wc = 1.2
        default_para = [
            np.array([1.2, 1.2, 80.5, 0.005, 50.0]),  # X通道默认参数
            np.array([1.2, 1.2, 80.5, 0.005, 50.0]),  # Y通道默认参数
            np.array([0.9, 1., 120.02, 0.005, 100.0]),  # Z通道默认参数（略高带宽以提升高度控制响应）
            np.array([20.0, 20.0, 20.0, 0.005, 50.0]),  # 横滚通道默认参数
            np.array([wc, 100.,wc*4, 0.005, 50.0]),  # 俯仰通道默认参数
            np.array([20.0, 2.0, 20.0, 0.005, 50.0])  # 偏航通道默认参数（较低响应避免震荡）
        ]
        # 位置控制 PID 系数（三维：X/Y/Z 轴独立参数）
        self.P_COEFF_FOR = np.array([.4, .4, 1.25])    # 比例项系数（单位：N/m）
        self.I_COEFF_FOR = np.array([.05, .05, .05])   # 积分项系数（单位：N/(m·s)）
        self.D_COEFF_FOR = np.array([.2, .2, .5])      # 微分项系数（单位：N/(m/s))

        # 姿态控制 PID 系数（三维：Roll/Pitch/Yaw 轴独立参数）
        self.P_COEFF_TOR = np.array([70000., 70000., 60000.])
        self.I_COEFF_TOR = np.array([.0, .0, 500.])
        self.D_COEFF_TOR = np.array([20000., 20000., 12000.])
        # 2. 优先使用用户传入的para，若未传入或长度不足则补充默认参数
        if len(para) >= 6:
            used_para = para[:6]  # 取前6个通道参数
            print("[INFO] 使用用户传入的LADRC参数")
        else:
            used_para = default_para  # 使用默认参数
            print("[INFO] 未传入足够LADRC参数，使用默认值")
        # --------------------------------------------------------------------

        # 设置位置和姿态的LADRC控制器（使用used_para初始化）
        self.con_X = LADRC(*used_para[0])  # X轴位置控制器
        self.con_Y = LADRC(*used_para[1])  # Y轴位置控制器
        self.con_Z = LADRC(*used_para[2])  # Z轴位置控制器
        self.con_roll = LADRC(*used_para[3])  # 横滚角控制器
        self.con_pitch = LADRC(*used_para[4])  # 俯仰角控制器
        self.con_yaw = LADRC(*used_para[5])  # 偏航角控制器
        print(f"[DEBUG] con_X 初始化完成: {hasattr(self, 'con_X')}")
        print(f"[DEBUG] con_Y 初始化完成: {hasattr(self, 'con_Y')}")
        print(f"[DEBUG] con_Z 初始化完成: {hasattr(self, 'con_Z')}")
        print(f"[DEBUG] con_roll 初始化完成: {hasattr(self, 'con_roll')}")
        print(f"[DEBUG] con_pitch 初始化完成: {hasattr(self, 'con_pitch')}")
        print(f"[DEBUG] con_yaw 初始化完成: {hasattr(self, 'con_yaw')}")
        # PWM 信号转换参数（将 PID 输出转换为电机转速）
        self.PWM2RPM_SCALE = 0.2685  # 比例系数：PWM 单位转换为 RPM 的缩放因子
        self.PWM2RPM_CONST = 4070.3  # 偏移量：PWM 基础值对应的 RPM
        self.MIN_PWM = 20000  # 最小有效 PWM 值（保护电机）
        self.MAX_PWM = 65535  # 最大有效 PWM 值（对应最大推力）

        # 混控矩阵配置（将推力/力矩转换为四个电机的控制量）
        if self.DRONE_MODEL == DroneModel.CF2X:
            # X 型布局混控矩阵（支持横滚、俯仰、偏航力矩耦合）
            self.MIXER_MATRIX = np.array([
                [-.5, -.5, -1],  # 电机 1 的力矩分配系数
                [-.5, .5, 1],  # 电机 2
                [.5, .5, -1],  # 电机 3
                [.5, -.5, 1]  # 电机 4
            ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            # + 型布局混控矩阵（简化横滚/俯仰控制）
            self.MIXER_MATRIX = np.array([
                [0, -1, -1],  # 电机 1
                [+1, 0, 1],  # 电机 2
                [0, 1, -1],  # 电机 3
                [-1, 0, 1]  # 电机 4
            ])
        self.reset()
    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.
            所有 变量都被重置为零。
        """
        super().reset()
        # 参数初始化
        #### Store the last roll, pitch, and yaw ###################
        self.last_rpy = np.zeros(3)
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.con_X.reset()
        self.con_Y.reset()
        self.con_Z.reset()
        self.con_roll.reset()
        self.con_pitch.reset()
        self.con_yaw.reset()
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)



    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """
        计算控制 ， 除了位置Target是确定的，其他的目标都是0，也是后续由控制自由设定的
        输入：
        control_timestep: 控制时间步长
        cur_pos: 当前位置
        cur_quat: 当前四元数
        cur_vel: 当前速度
        cur_ang_vel: 当前角速度
        target_pos: 目标位置
        target_rpy: 目标欧拉角
        target_vel: 目标速度
        target_rpy_rates: 目标角速度
        输出：
        rpm: 电机转速
        pos_e: 位置误差
        yaw_e: 偏航误差
        """
        self.control_counter += 1
        thrust, computed_target_rpy, pos_e = self._dslPIDPositionControl(control_timestep,
                                                                         cur_pos,
                                                                         cur_quat,
                                                                         cur_vel,
                                                                         target_pos,
                                                                         target_rpy,
                                                                         target_vel
                                                                         )
        rpm = self._dslPIDAttitudeControl(control_timestep, thrust, cur_quat, computed_target_rpy, target_rpy_rates)
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]

    def _dslLADRCPositionControl(self,
                               control_timestep,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel
                               ):
        """位置控制
        输出参数：
        thrust: 目标推力，z轴
        computed_target_rpy: 计算出的目标姿态 ， 滚转角，俯仰角，偏航角
        pos_e: 位置误差
        Returns
        -------
        float
            The target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
        float
            The current position error.

        """
        # 1. 坐标系转换：根据四元数获取当前机体系旋转矩阵
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        # 计算误差
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        # 积分项处理  上下限
        self.integral_pos_e = self.integral_pos_e + pos_e * control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)
        # 使用LADRC计算目标推力
        u0 = self.con_X(target_pos[0],cur_pos[0])
        u1 = self.con_Y(target_pos[1],cur_pos[1])
        u2 = self.con_Z(target_pos[2],cur_pos[2])
        target_thrust = np.array([u0,u1,u2])+ np.array([0, 0, self.GRAVITY])
        # 将世界坐标系的目标推力投影到机体Z轴方向     （TODO： 姿态解算）
        # cur_rotation[:,2] 表示机体Z轴在世界坐标系中的方向向量
        # np.dot() 计算目标推力在机体Z轴方向的投影标量值
        # max(0., ...) 确保推力不为负（无人机无法产生向下推力）
        # 5. 推力方向投影（转换到机体Z轴）
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:, 2]))
        # 6. 推力到PWM的转换（考虑电机模型参数）
        thrust = (math.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        # 7. 目标姿态生成（根据推力方向）
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        #### Target rotation #######################################
        # 将目标旋转矩阵 target_rotation 转换为 XYZ 顺序的欧拉角（滚转、俯仰、偏航），作为目标姿态 target_euler
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        if np.any(np.abs(target_euler) > math.pi):
            print("\n[ERROR] ctrl it", self.control_counter,
                  "in Control._dslPIDPositionControl(), values outside range [-pi,pi]")
        # thrust：电机 PWM 控制信号（沿机体 Z 轴的推力）；
        # target_euler：目标姿态（滚转、俯仰、偏航，用于后续姿态控制）；
        # pos_e：当前位置误差。
        # 保存数据
        self.logger["target_thrust"].append(target_thrust)
        self.logger["target_thrust_x"].append(target_thrust[0])
        self.logger["target_thrust_y"].append(target_thrust[1])
        self.logger["target_thrust_z"].append(target_thrust[2])
        self.logger["thrust"].append(thrust)
        self.logger["target_rotation"].append(target_rotation)
        self.logger["target_euler"].append(target_euler)
        self.logger["pos"].append(cur_pos)
        self.logger["pos_x"].append(cur_pos[0])
        self.logger["pos_y"].append(cur_pos[1])
        self.logger["pos_z"].append(cur_pos[2])
        self.logger["target_pos"].append(target_pos)
        self.logger["target_pos_x"].append(target_pos[0])
        self.logger["target_pos_y"].append(target_pos[1])
        self.logger["target_pos_z"].append(target_pos[2])
        return thrust, target_euler, pos_e

    ################################################################################

    def _dslLADRCAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               target_euler,
                               target_rpy_rates
                               ):
        """
        姿态控制:
        输入：
        thrust: 目标推力，z轴
        cur_quat: 当前四元数
        target_euler: 目标姿态，滚转角，俯仰角，偏航角
        target_rpy_rates: 目标滚转角速度，俯仰角速度，偏航角速度
        输出：
        rpm: 电机PWM控制信号
        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """
        # 获取当前姿态欧拉角
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))

        # LADRC控制器计算三轴力矩（仅需目标值和当前值）
        roll_torque = self.con_roll(target_euler[0], cur_rpy[0])  # 横滚通道LADRC
        pitch_torque = self.con_pitch(target_euler[1], cur_rpy[1])  # 俯仰通道LADRC
        yaw_torque = self.con_yaw(target_euler[2], cur_rpy[2])  # 偏航通道LADRC
        #### 组合目标力矩并限幅 ####################################
        target_torques = np.array([roll_torque, pitch_torque, yaw_torque])

        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

    ################################################################################

    def _one23DInterface(self,
                         thrust
                         ):
        """Utility function interfacing 1, 2, or 3D thrust input use cases.

        Parameters
        ----------
        thrust : ndarray
            Array of floats of length 1, 2, or 4 containing a desired thrust input.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the PWM (not RPMs) to apply to each of the 4 motors.

        """
        DIM = len(np.array(thrust))
        pwm = np.clip((np.sqrt(np.array(thrust) / (self.KF * (4 / DIM))) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE,
                      self.MIN_PWM, self.MAX_PWM)
        if DIM in [1, 4]:
            return np.repeat(pwm, 4 / DIM)
        elif DIM == 2:
            return np.hstack([pwm, np.flip(pwm)])
        else:
            print("[ERROR] in DSLPIDControl._one23DInterface()")
            exit()

    def _dslPIDPositionControl(self,
                               control_timestep,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel
                               ):
        """DSL's CF2.x PID position control.

        """
        # 1. 坐标系转换：根据四元数获取当前机体系旋转矩阵
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        # 计算误差
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        # 积分项处理  上下限
        self.integral_pos_e = self.integral_pos_e + pos_e * control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)
        #### PID target thrust  推力计算#####################################
        #  target_thrust = (P项 * 位置误差) + (I项 * 积分误差) + (D项 * 速度误差) + 重力补偿
        target_thrust = np.multiply(self.P_COEFF_FOR, pos_e) \
                        + np.multiply(self.I_COEFF_FOR, self.integral_pos_e) \
                        + np.multiply(self.D_COEFF_FOR, vel_e) + np.array([0, 0, self.GRAVITY])
        # 将世界坐标系的目标推力投影到机体Z轴方向     （TODO： 姿态解算）
        # cur_rotation[:,2] 表示机体Z轴在世界坐标系中的方向向量
        # np.dot() 计算目标推力在机体Z轴方向的投影标量值
        # max(0., ...) 确保推力不为负（无人机无法产生向下推力）
        # 5. 推力方向投影（转换到机体Z轴）
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:, 2]))
        # 6. 推力到PWM的转换（考虑电机模型参数）
        thrust = (math.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        # 7. 目标姿态生成（根据推力方向）
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        #### Target rotation #######################################
        # 将目标旋转矩阵 target_rotation 转换为 XYZ 顺序的欧拉角（滚转、俯仰、偏航），作为目标姿态 target_euler
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        if np.any(np.abs(target_euler) > math.pi):
            print("\n[ERROR] ctrl it", self.control_counter,
                  "in Control._dslPIDPositionControl(), values outside range [-pi,pi]")
        # thrust：电机 PWM 控制信号（沿机体 Z 轴的推力）；
        # target_euler：目标姿态（滚转、俯仰、偏航，用于后续姿态控制）；
        # pos_e：当前位置误差。
        # 记录日志
        self.logger["target_thrust"].append(target_thrust)
        self.logger["target_thrust_x"].append(target_thrust[0])
        self.logger["target_thrust_y"].append(target_thrust[1])
        self.logger["target_thrust_z"].append(target_thrust[2])
        self.logger["thrust"].append(thrust)
        self.logger["target_rotation"].append(target_rotation)
        self.logger["target_euler"].append(target_euler)
        self.logger["pos"].append(cur_pos)
        self.logger["pos_x"].append(cur_pos[0])
        self.logger["pos_y"].append(cur_pos[1])
        self.logger["pos_z"].append(cur_pos[2])
        self.logger["target_pos"].append(target_pos)
        self.logger["target_pos_x"].append(target_pos[0])
        self.logger["target_pos_y"].append(target_pos[1])
        self.logger["target_pos_z"].append(target_pos[2])
        return thrust, target_euler, pos_e

    ################################################################################

    def _dslPIDAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               target_euler,
                               target_rpy_rates
                               ):
        """DSL's CF2.x PID attitude control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        thrust : float
            The target thrust along the drone z-axis.

        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        target_euler : ndarray
            (3,1)-shaped array of floats containing the computed target Euler angles.
        target_rpy_rates : ndarray
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w, x, y, z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()), cur_rotation) - np.dot(cur_rotation.transpose(),
                                                                                    target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy) / control_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e * control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
        #### PID target torques ####################################
        target_torques = - np.multiply(self.P_COEFF_TOR, rot_e) \
                         + np.multiply(self.D_COEFF_TOR, rpy_rates_e) \
                         + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

out = []
out_hat = []
out_dt_hat = []
out_dt = []
out_f = []
if __name__ == '__main__':
    ladrc = LADRC(3.3, 1, 3.4, 0.005, 50)
    # 二阶被控对象参数
    zeta = 0.2         # 阻尼比
    omega_n = 0.8      # 固有频率
    K = 2.2            # 增益
    x = 0.0            # 系统输出
    x_dot = 0.0        # 系统速度

    i = 0
    target = 200

    for i in range(1000):
        # 1. 使用当前控制量u更新二阶被控对象状态
        # 二阶系统微分方程: x'' = -2ζωₙx' - ωₙ²x + Kωₙ²u + d(t)
        disturbance = 100 * np.sin(i * 0.1) if i > 200 else 0  # 200步后加入正弦扰动
        x_ddot = -2 * zeta * omega_n * x_dot - omega_n ** 2 * x + K * omega_n ** 2 * ladrc.u + disturbance

        # 欧拉积分更新状态
        x_dot += x_ddot * ladrc.h
        x += x_dot * ladrc.h

        # 2. LADRC控制器计算 (修复原代码中LESO缺少current参数的问题)
        ladrc(target, x)

        # 3. 记录数据
        out = np.append(out, x)  # 系统实际输出
        out_hat = np.append(out_hat, ladrc.v1)  # 观测器估计输出
        out_dt_hat = np.append(out_dt_hat, ladrc.x2)  # 观测器估计速度
        out_dt = np.append(out_dt, ladrc.v2)  # TD微分输出
        out_f = np.append(out_f, ladrc.x3)  # 观测器估计扰动
    # 设置中文显示
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # 补充绘图代码
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 12))
    t = np.arange(0, len(out) * ladrc.h, ladrc.h)  # 时间轴

    # 1. 系统输出与观测器估计对比
    axes[0].plot(t, out, 'r-', label='实际输出 y ')
    axes[0].plot(t, out_hat, 'b--', label='观测器估计输出 z1')
    axes[0].set_title('系统输出与观测器估计对比')
    axes[0].set_xlabel('时间 (s)')
    axes[0].set_ylabel('输出值')
    axes[0].legend()
    axes[0].grid(True)

    # 2. 目标跟踪效果
    axes[1].plot(t, out, 'r-', label='实际输出')
    axes[1].axhline(y=target, color='g', linestyle='--', label='目标值')
    axes[1].set_title('目标跟踪效果')
    axes[1].set_xlabel('时间 (s)')
    axes[1].set_ylabel('输出值')
    axes[1].legend()
    axes[1].grid(True)

    # 3. 速度跟踪对比 (TD微分输出 vs 观测器速度估计)
    axes[2].plot(t, out_dt, 'r-', label='TD微分输出 v2')
    axes[2].plot(t, out_dt, 'b--', label='TD微分输出 v1')
    axes[2].set_title('速度跟踪对比')
    axes[2].set_xlabel('时间 (s)')
    axes[2].set_ylabel('速度值')
    axes[2].legend()
    axes[2].grid(True)

    # 4. 扩展状态观测器扰动估计
    axes[3].plot(t, out_f, 'm-', label='扰动估计值 z3')
    axes[3].set_title('扩展状态观测器扰动估计')
    axes[3].set_xlabel('时间 (s)')
    axes[3].set_ylabel('扰动值')
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()
