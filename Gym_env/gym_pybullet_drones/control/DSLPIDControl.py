import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class DSLPIDControl(BaseControl):
    """PID control class for Crazyflies.

    Based on work conducted at UTIAS' DSL. Contributors: SiQi Zhou, James Xu, 
    Tracy Du, Mario Vukosavljev, Calvin Ngan, and Jingyuan Hou.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """PID 控制器初始化方法

        主要完成以下初始化工作：
        1. 校验无人机型号兼容性（仅支持 CF2X/CF2P）
        2. 设置位置控制 PID 参数（FOR 前缀参数）
        3. 设置姿态控制 PID 参数（TOR 前缀参数）
        4. 配置 PWM 与 RPM 的转换参数
        5. 根据机型选择混控矩阵

        Parameters
        ----------
        drone_model : DroneModel
            无人机型号枚举值，决定使用的动力学参数和混控矩阵
        g : float, optional
            重力加速度，默认使用 9.8 m/s²

        Raises
        ------
        SystemExit
            当传入不支持的无人机型号时立即终止程序
        """
        super().__init__(drone_model=drone_model, g=g)
        # 型号兼容性检查（当前仅支持 Crazyflie 2.0 系列）
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()

        # 位置控制 PID 系数（三维：X/Y/Z 轴独立参数）
        self.P_COEFF_FOR = np.array([.4, .4, 1.25])    # 比例项系数（单位：N/m）
        self.I_COEFF_FOR = np.array([.05, .05, .05])   # 积分项系数（单位：N/(m·s)）
        self.D_COEFF_FOR = np.array([.2, .2, .5])      # 微分项系数（单位：N/(m/s))

        # 姿态控制 PID 系数（三维：Roll/Pitch/Yaw 轴独立参数）
        self.P_COEFF_TOR = np.array([7., 7., 6.])  # 比例项（单位：KN·m/rad）
        self.I_COEFF_TOR = np.array([.0, .0, 0.05])            # 积分项（单位：KN·m/(rad·s)）
        self.D_COEFF_TOR = np.array([2., 2., 1.2])  # 微分项（单位：KN·m/(rad/s)）

        # PWM 信号转换参数（将 PID 输出转换为电机转速）
        self.PWM2RPM_SCALE = 0.2685    # 比例系数：PWM 单位转换为 RPM 的缩放因子
        self.PWM2RPM_CONST = 4070.3    # 偏移量：PWM 基础值对应的 RPM
        self.MIN_PWM = 20000           # 最小有效 PWM 值（保护电机）
        self.MAX_PWM = 65535           # 最大有效 PWM 值（对应最大推力）

        # 混控矩阵配置（将推力/力矩转换为四个电机的控制量）
        if self.DRONE_MODEL == DroneModel.CF2X:
            # X 型布局混控矩阵（支持横滚、俯仰、偏航力矩耦合）
            self.MIXER_MATRIX = np.array([
                                    [-.5, -.5, -1],  # 电机 1 的力矩分配系数
                                    [-.5,  .5,  1],  # 电机 2
                                    [.5, .5, -1],    # 电机 3
                                    [.5, -.5,  1]    # 电机 4
                                    ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            # + 型布局混控矩阵（简化横滚/俯仰控制）
            self.MIXER_MATRIX = np.array([
                                    [0, -1,  -1],  # 电机 1
                                    [+1, 0, 1],    # 电机 2
                                    [0,  1,  -1],   # 电机 3
                                    [-1, 0, 1]     # 电机 4
                                    ])
        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.
            所有 变量都被重置为零。
        """
        super().reset()
        #### Store the last roll, pitch, and yaw ###################
        self.last_rpy = np.zeros(3)
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    ################################################################################
    
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
        """Computes the PID control action (as RPMs) for a single drone.

        This methods sequentially calls `_dslPIDPositionControl()` and `_dslPIDAttitudeControl()`.
        Parameter `cur_ang_vel` is unused.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

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
        rpm = self._dslPIDAttitudeControl(control_timestep,
                                          thrust,
                                          cur_quat,
                                          computed_target_rpy,
                                          target_rpy_rates
                                          )
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]
    
    ################################################################################

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

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
            位置 x ，y ，z 分别对应于无人机在世界坐标系中的 x 、y 、z 轴方向的位置。
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
            四元数 (w, x, y, z)，用于表示无人机的当前旋转状态。 或者 x, y, z, w 顺序。
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
            速度 vx ，vy ，vz 分别对应于无人机在世界坐标系中的 x 、y 、z 轴方向的速度。
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray
            (3,1)-shaped array of floats containing the desired velocity.

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
        self.integral_pos_e = self.integral_pos_e + pos_e*control_timestep
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
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:,2]))
        # 6. 推力到PWM的转换（考虑电机模型参数）
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
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
            print("\n[ERROR] ctrl it", self.control_counter, "in Control._dslPIDPositionControl(), values outside range [-pi,pi]")
        # thrust：电机 PWM 控制信号（沿机体 Z 轴的推力）；
        # target_euler：目标姿态（滚转、俯仰、偏航，用于后续姿态控制）；
        # pos_e：当前位置误差。
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
        w,x,y,z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()),cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) 
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy)/control_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e*control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
        #### PID target torques ####################################
        target_torques = - np.multiply(self.P_COEFF_TOR, rot_e) \
                         + np.multiply(self.D_COEFF_TOR, rpy_rates_e) \
                         + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        target_torques = np.clip(target_torques*10000, -3200, 3200)
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
        pwm = np.clip((np.sqrt(np.array(thrust)/(self.KF*(4/DIM)))-self.PWM2RPM_CONST)/self.PWM2RPM_SCALE, self.MIN_PWM, self.MAX_PWM)
        if DIM in [1, 4]:
            return np.repeat(pwm, 4/DIM)
        elif DIM==2:
            return np.hstack([pwm, np.flip(pwm)])
        else:
            print("[ERROR] in DSLPIDControl._one23DInterface()")
            exit()
