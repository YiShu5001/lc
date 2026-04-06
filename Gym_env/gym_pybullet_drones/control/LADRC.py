import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel
from Gym_env.LADRC_Controller import LADRC, LADRCConfig

class LADRCControl(BaseControl):
    """
    LADRC Controller for PyBullet Drones.
    Utilizes 6 independent LADRC channels for [X, Y, Z, Roll, Pitch, Yaw].
    """
    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8,
                 *para
                 ):
        super().__init__(drone_model=drone_model, g=g)

        print("[DEBUG] 初始化 LADRC 控制器")

        self.logger = {
            'target_thrust': [], 'target_thrust_x': [], 'target_thrust_y': [], 'target_thrust_z': [],
            'thrust': [], 'target_rotation': [], 'target_euler': [],
            'pos': [], 'pos_x': [], 'pos_y': [], 'pos_z': [],
            'target': [], 'target_x': [], 'target_y': [], 'target_z': [],
        }

        if self.DRONE_MODEL not in [DroneModel.CF2X, DroneModel.CF2P]:
            print("[ERROR] in LADRCControl.__init__(), LADRCControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()

        step_size = self.CTRL_TIMESTEP if hasattr(self, 'CTRL_TIMESTEP') else 0.005

        # Default configuration: [omega_c, b0, omega_o, r] for each channel
        default_configs = [
            [1.1, 220., 9.6, 30.0],  # X
            [1.3, 220., 9.6, 30.0],  # Y
            [15., 300., 110., 30.0], # Z
            [6., 250., 30., 50.0],   # Roll
            [6., 250., 30., 50.0],   # Pitch
            [1.2, 150., 6.0, 50.0]   # Yaw
        ]
        
        used_configs = []
        if len(para) >= 6:
            print("[INFO] 使用用户传入的LADRC参数")
            for p_arr in para[:6]:
                # Assuming p_arr = [omega_c, b0, omega_o, step_size, r]
                # Fallback to default step_size if not provided or just use standard mapping
                r_val = p_arr[4] if len(p_arr) > 4 else 50.0
                used_configs.append([p_arr[0], p_arr[1], p_arr[2], r_val])
        else:
            print("[INFO] 未传入足够LADRC参数，使用默认值")
            used_configs = default_configs

        self.con_X = LADRC(LADRCConfig(omega_c=used_configs[0][0], b0=used_configs[0][1], omega_o=used_configs[0][2], step_size=step_size, r=used_configs[0][3]))
        self.con_Y = LADRC(LADRCConfig(omega_c=used_configs[1][0], b0=used_configs[1][1], omega_o=used_configs[1][2], step_size=step_size, r=used_configs[1][3]))
        self.con_Z = LADRC(LADRCConfig(omega_c=used_configs[2][0], b0=used_configs[2][1], omega_o=used_configs[2][2], step_size=step_size, r=used_configs[2][3]))
        self.con_roll = LADRC(LADRCConfig(omega_c=used_configs[3][0], b0=used_configs[3][1], omega_o=used_configs[3][2], step_size=step_size, r=used_configs[3][3]))
        self.con_pitch = LADRC(LADRCConfig(omega_c=used_configs[4][0], b0=used_configs[4][1], omega_o=used_configs[4][2], step_size=step_size, r=used_configs[4][3]))
        self.con_yaw = LADRC(LADRCConfig(omega_c=used_configs[5][0], b0=used_configs[5][1], omega_o=used_configs[5][2], step_size=step_size, r=used_configs[5][3]))

        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535

        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([
                [-.5, -.5, -1],  
                [-.5, .5, 1],  
                [.5, .5, -1],  
                [.5, -.5, 1]  
            ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                [0, -1, -1],  
                [+1, 0, 1],  
                [0, 1, -1],  
                [-1, 0, 1]  
            ])
        self.reset()

    def reset(self):
        """Resets the control classes."""
        super().reset()
        self.last_rpy = np.zeros(3)
        self.last_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.con_X.reset()
        self.con_Y.reset()
        self.con_Z.reset()
        self.con_roll.reset()
        self.con_pitch.reset()
        self.con_yaw.reset()

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
        self.control_counter += 1
        thrust, computed_target_rpy, pos_e = self._dslLADRCPositionControl(
            control_timestep, cur_pos, cur_quat, cur_vel, target_pos, target_rpy, target_vel)
        
        rpm = self._dslLADRCAttitudeControl(
            control_timestep, thrust, cur_quat, computed_target_rpy, target_rpy_rates)
        
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
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        
        # Position LADRC updates
        u0 = self.con_X.update(target_pos[0], cur_pos[0])
        u1 = self.con_Y.update(target_pos[1], cur_pos[1])
        u2 = self.con_Z.update(target_pos[2], cur_pos[2])
        
        target_thrust = np.array([u0, u1, u2]) + np.array([0, 0, self.GRAVITY])
        
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:, 2]))
        thrust = (math.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        
        # Compute target orientation
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        
        # Logging
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
        self.logger["target"].append(target_pos)
        self.logger["target_x"].append(target_pos[0])
        self.logger["target_y"].append(target_pos[1])
        self.logger["target_z"].append(target_pos[2])
        
        return thrust, target_euler, pos_e

    def _dslLADRCAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               target_euler,
                               target_rpy_rates
                               ):
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))

        roll_torque = self.con_roll.update(target_euler[0], cur_rpy[0])
        pitch_torque = self.con_pitch.update(target_euler[1], cur_rpy[1])
        yaw_torque = self.con_yaw.update(target_euler[2], cur_rpy[2])
        
        target_torques = np.array([roll_torque, pitch_torque, yaw_torque]) * 10000.0
        target_torques = np.clip(target_torques, -3200, 3200)
        
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
