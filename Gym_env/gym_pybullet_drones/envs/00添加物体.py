import os
from sys import platform
import time
import collections
from datetime import datetime
import xml.etree.ElementTree as etxml
import pkg_resources
from PIL import Image
# import pkgutil
# egl = pkgutil.get_loader('eglRenderer')
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from prompt_toolkit.key_binding.bindings.named_commands import self_insert
from pydantic_core.core_schema import none_schema

from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType


class BaseAviary(gym.Env):
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 vision_attributes=False,
                 output_folder='results'
                 ):

        #### Constants   基础内容 #############################################
        self.G = 9.8
        self.RAD2DEG = 180/np.pi
        self.DEG2RAD = np.pi/180
        self.CTRL_FREQ = ctrl_freq     # 控制频率
        self.PYB_FREQ = pyb_freq       # 仿真频率
        if self.PYB_FREQ % self.CTRL_FREQ != 0:
            raise ValueError('[ERROR] in BaseAviary.__init__(), pyb_freq is not divisible by env_freq.')
        self.PYB_STEPS_PER_CTRL = int(self.PYB_FREQ / self.CTRL_FREQ)
        self.CTRL_TIMESTEP = 1. / self.CTRL_FREQ   #步长是频率的倒数
        self.PYB_TIMESTEP = 1. / self.PYB_FREQ
        #### Parameters  参数信息：障碍物信息，无人机个数，临近距离，无人机模型 ############################################
        self.Obs = {}
        self.NUM_DRONES = num_drones
        self.NEIGHBOURHOOD_RADIUS = neighbourhood_radius
        self.DRONE_MODEL = drone_model
        self.GUI = gui
        self.RECORD = record
        self.PHYSICS = physics
        self.OBSTACLES = obstacles
        self.USER_DEBUG = user_debug_gui
        self.URDF = self.DRONE_MODEL.value + ".urdf"
        self.OUTPUT_FOLDER = output_folder

        #### Load the drone properties from the .urdf file #########               模型属性
        '''
        质量 、臂展 、 推重比 、转动惯量矩阵 、转动惯量逆矩阵 、 推力系数 、力矩系数、 collision 碰撞体（h 高度，r 半径， z_off z轴偏移量）
        最大速度、 地面效应 、 螺旋桨半径 、空气阻力、 下降气流系数（DW_coeff 123）
        '''
        self.M, \
        self.L, \
        self.THRUST2WEIGHT_RATIO, \
        self.J, \
        self.J_INV, \
        self.KF, \
        self.KM, \
        self.COLLISION_H,\
        self.COLLISION_R, \
        self.COLLISION_Z_OFFSET, \
        self.MAX_SPEED_KMH, \
        self.GND_EFF_COEFF, \
        self.PROP_RADIUS, \
        self.DRAG_COEFF, \
        self.DW_COEFF_1, \
        self.DW_COEFF_2, \
        self.DW_COEFF_3 = self._parseURDFParameters()
        # 绘制初始信息
        print("[INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf:\n[INFO] m {:f}, L {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f},\n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f},\n[INFO] gnd_eff_coeff {:f}, prop_radius {:f},\n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f},\n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f}".format(
            self.M, self.L, self.J[0,0], self.J[1,1], self.J[2,2], self.KF, self.KM, self.THRUST2WEIGHT_RATIO, self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS, self.DRAG_COEFF[0], self.DRAG_COEFF[2], self.DW_COEFF_1, self.DW_COEFF_2, self.DW_COEFF_3))
        #### Compute constants ####  计算系统环境的常量 （上面是模型的参数，下面就是产生的力、转速之类的量）
        '''
        重力、悬停的旋转速度、最大转速（RPM）、最大推力（THRUST）、xy，z最大旋转力矩（TORQUE）、地面效应参数
        '''
        self.GRAVITY = self.G*self.M
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4*self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (4*self.KF))
        self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/np.sqrt(2)
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MAX_XY_TORQUE = (self.L*self.KF*self.MAX_RPM**2)
        elif self.DRONE_MODEL == DroneModel.RACE:
            self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/np.sqrt(2)
        self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt((15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)
        #### Create attributes for vision tasks ####################  打开/创建数据存储地址 ， 保存图像
        if self.RECORD:
            self.ONBOARD_IMG_PATH = os.path.join(self.OUTPUT_FOLDER, "recording_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
            os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH), exist_ok=True)
        self.VISION_ATTR = vision_attributes
        if self.VISION_ATTR:
            self.IMG_RES = np.array([64, 48])
            self.IMG_FRAME_PER_SEC = 24
            self.IMG_CAPTURE_FREQ = int(self.PYB_FREQ/self.IMG_FRAME_PER_SEC)
            self.rgb = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4)))
            self.dep = np.ones(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            self.seg = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            if self.IMG_CAPTURE_FREQ%self.PYB_STEPS_PER_CTRL != 0:
                print("[ERROR] in BaseAviary.__init__(), PyBullet and control frequencies incompatible with the desired video capture frame rate ({:f}Hz)".format(self.IMG_FRAME_PER_SEC))
                exit()
            if self.RECORD:
                for i in range(self.NUM_DRONES):
                    os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH+"/drone_"+str(i)+"/"), exist_ok=True)
        #### Connect to PyBullet   连接到仿真gui（图形系统）或者数字系统###################################
        '''
        该部分内容 全部都是解释两种情况下  解读图像数据的过程 对我无用 参数 ： ret；RECORD
        '''
        if self.GUI:
            #### With debug GUI ####################################### 其实我觉得没必要，可能图像分析需要通过这个增强数据信息#
            self.CLIENT = p.connect(p.GUI) # p.connect(p.GUI, options="--opengl2")
            ##   禁用预览缓冲区 RGB、深度和分割  三个通道，   数据输出的设定 保存 图像（ret）, 旋转速度 SLIDERS
            for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]:
                p.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
            p.resetDebugVisualizerCamera(cameraDistance=3,
                                         cameraYaw=-30,
                                         cameraPitch=-30,
                                         cameraTargetPosition=[0, 0, 0],
                                         physicsClientId=self.CLIENT
                                         )
            ret = p.getDebugVisualizerCamera(physicsClientId=self.CLIENT)
            print("viewMatrix", ret[2])
            print("projectionMatrix", ret[3])
            if self.USER_DEBUG:
                #### Add input sliders to the GUI ##########################
                self.SLIDERS = -1*np.ones(4)
                for i in range(4):
                    self.SLIDERS[i] = p.addUserDebugParameter("Propeller "+str(i)+" RPM", 0, self.MAX_RPM, self.HOVER_RPM, physicsClientId=self.CLIENT)
                self.INPUT_SWITCH = p.addUserDebugParameter("Use GUI RPM", 9999, -1, 0, physicsClientId=self.CLIENT)
        else:
            #### Without debug GUI #####################################
            self.CLIENT = p.connect(p.DIRECT)
            #### Uncomment the following line to use EGL Render Plugin #
            #### Instead of TinyRender (CPU-based) in PYB's Direct mode
            # if platform == "linux": p.setAdditionalSearchPath(pybullet_data.getDataPath()); plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin"); print("plugin=", plugin)
            # 保存图像的说明，所以对我的数据分析都用不到，就是单纯的 CLIENT = p.connect(p gui/direct)
            if self.RECORD:
                #### Set the camera parameters to save frames in DIRECT mode
                self.VID_WIDTH=int(640)
                self.VID_HEIGHT=int(480)
                self.FRAME_PER_SEC = 24
                self.CAPTURE_FREQ = int(self.PYB_FREQ/self.FRAME_PER_SEC)
                self.CAM_VIEW = p.computeViewMatrixFromYawPitchRoll(distance=3,
                                                                    yaw=-30,
                                                                    pitch=-30,
                                                                    roll=0,
                                                                    cameraTargetPosition=[0, 0, 0],
                                                                    upAxisIndex=2,
                                                                    physicsClientId=self.CLIENT
                                                                    )
                self.CAM_PRO = p.computeProjectionMatrixFOV(fov=60.0,
                                                            aspect=self.VID_WIDTH/self.VID_HEIGHT,
                                                            nearVal=0.1,
                                                            farVal=1000.0
                                                            )
        #### Set initial poses #####################################
        if initial_xyzs is None:    # 初始位置
            self.INIT_XYZS = np.vstack([np.array([x*4*self.L for x in range(self.NUM_DRONES)]), \
                                        np.array([y*4*self.L for y in range(self.NUM_DRONES)]), \
                                        np.ones(self.NUM_DRONES) * (self.COLLISION_H/2-self.COLLISION_Z_OFFSET+.1)]).transpose().reshape(self.NUM_DRONES, 3)
        elif np.array(initial_xyzs).shape == (self.NUM_DRONES,3):
            self.INIT_XYZS = initial_xyzs
        else:
            print("[ERROR] invalid initial_xyzs in BaseAviary.__init__(), try initial_xyzs.reshape(NUM_DRONES,3)")
        # 初始 角度
        if initial_rpys is None:
            self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
        elif np.array(initial_rpys).shape == (self.NUM_DRONES, 3):
            self.INIT_RPYS = initial_rpys
        else:
            print("[ERROR] invalid initial_rpys in BaseAviary.__init__(), try initial_rpys.reshape(NUM_DRONES,3)")
        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        #### Housekeeping  加载状态 ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()

    def _housekeeping(self):
        """Housekeeping function.  状态信息初始化，联系物理引擎

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.

        """
        #### Initialize/reset counters and zero-valued variables ###
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.first_render_call = True
        self.X_AX = -1*np.ones(self.NUM_DRONES)
        self.Y_AX = -1*np.ones(self.NUM_DRONES)
        self.Z_AX = -1*np.ones(self.NUM_DRONES)
        self.GUI_INPUT_TEXT = -1*np.ones(self.NUM_DRONES)
        self.USE_GUI_RPM=False
        self.last_input_switch = 0
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))
        self.gui_input = np.zeros(4)
        #### Initialize the drones kinemaatic information ##########
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        if self.PHYSICS == Physics.DYN:
            self.rpy_rates = np.zeros((self.NUM_DRONES, 3))
        #### Set PyBullet's parameters #############################
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
        #### Load ground plane, drone and obstacles models #########
        #TODO：  设置无人机
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)

        self.DRONE_IDS = np.array([p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+self.URDF),
                                              self.INIT_XYZS[i,:],
                                              p.getQuaternionFromEuler(self.INIT_RPYS[i,:]),
                                              flags = p.URDF_USE_INERTIA_FROM_FILE,
                                              physicsClientId=self.CLIENT
                                              ) for i in range(self.NUM_DRONES)])
        #### Remove default damping  TODO： 阻尼设定 #################################
        # for i in range(self.NUM_DRONES):
        #     p.changeDynamics(self.DRONE_IDS[i], -1, linearDamping=0, angularDamping=0)
        #### Show the frame of reference of the drone, note that ###
        #### It severly slows down the GUI TODO： 展示无人机自身的三位向量 #########################
        if self.GUI and self.USER_DEBUG:
            for i in range(self.NUM_DRONES):
                self._showDroneLocalAxes(i)
        #### Disable collisions between drones' and the ground plane
        #### E.g., to start a drone at [0,0,0] #####################
        # for i in range(self.NUM_DRONES):
            # p.setCollisionFilterPair(bodyUniqueIdA=self.PLANE_ID, bodyUniqueIdB=self.DRONE_IDS[i], linkIndexA=-1, linkIndexB=-1, enableCollision=0, physicsClientId=self.CLIENT)
        if self.OBSTACLES:
            self.Obs = {}
            self._addObstacles()
            self._getObstacle()

    def _addObstacles(self):
        """Adds obstacles to the environment.
        """
        #TODO：  障碍物设置
        self.Obs[1] = self._AddObstacle(nth=1, pos=[0, 0, 4], shape=[2, 2, 0.4], color=[1, 0, 0, 1], mass=0,
                              ClientId=self.CLIENT)
        self.Obs[2] = self._AddObstacle(nth=2, pos=[1.5, 1.5, 2], shape=[0.5, 0.5, 2], color=[1, 0, 0, 1], mass=0,
                              ClientId=self.CLIENT)
        self.Obs[3] = self._AddObstacle(nth=3, pos=[-1.5, -1.5, 2], shape=[0.5, 0.5, 2], color=[1, 0, 0, 1], mass=0,
                              ClientId=self.CLIENT)


    def _AddObstacle(nth: int = None, pos=[0, 0, 2], shape=[0.5, 0.5, 0.5], color=[1, 0, 0, 1], mass=2, Fixed=True,
                     ClientId=None):
        # 创建障碍物
        visualShapeId = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=shape,
            rgbaColor=color  # 红色，不透明
        )
        collisionShapeId = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=shape
        )
        self.Obs[nth] = p.createMultiBody(
            baseMass=mass,  # 1kg
            baseCollisionShapeIndex=collisionShapeId,
            baseVisualShapeIndex=visualShapeId,
            basePosition=pos,  # 初始位置（x,y,z）
            useFixedBase=Fixed,
            physicsClientId=ClientId
        )
    def _getObstacle(self,):
        for i, value in self.Obs:
            self.obs_pos[i], self.obs_quat[i] = p.getBasePositionAndOrientation(value, physicsClientId=self.CLIENT)
            self.obs_rpy[i] = p.getEulerFromQuaternion(self.quat[i])
            self.obs_vel[i], self.obs_ang_vang_v[i] = p.getBaseVelocity(value, physicsClientId=self.CLIENT)
    def _getObstacleStateVector(self,
                             nth_drone
                             ):
        state = np.hstack([self.obs_pos[nth_drone, :], self.obs_quat[nth_drone, :], self.obs_rpy[nth_drone, :],
                           self.obs_vel[nth_drone, :], self.obs_ang_vang_v[nth_drone, :]])
        return state.reshape(16,)

########################   目标点， 有点的控制加入 resetBaseVelocity
    def _setTarget(self, pos=[0, 5, 0], shape=[0.1, 0.1, 0.1], color=[0, 0, 0, 1], mass=0.1, ClientId= None):
        ClientId = self.CLIENT if ClientId is None else ClientId
        # 创建障碍物
        visualShapeId = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=shape,
            rgbaColor=color  # 红色，不透明
        )
        collisionShapeId = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=shape
        )
        self.Tar = p.createMultiBody(
            baseMass=mass,  # 1kg
            baseCollisionShapeIndex=collisionShapeId,
            baseVisualShapeIndex=visualShapeId,
            basePosition=pos,  # 初始位置（x,y,z）
            useMaximalCoordinates=True,
            physicsClientId=ClientId
        )
        return self.Tar

    #### 控制目标点的运动
    def _moveTarget(self, pos = np.array([0, 0, 0]), id=None,speed = 3, ClientId=None):
        ClientId = self.CLIENT if ClientId is None else ClientId
        id = self.Tar if id is None else id
        # 目标点位置
        position_tolerance = 0.1  # 到达目标点的距离阈值

        # 获取箱子当前位置
        current_pos, _ = p.getBasePositionAndOrientation(id, physicsClientId=ClientId)
        current_pos = np.array(current_pos)

        # 计算到目标点的方向和距离
        target_pos = pos
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)
        print('当前位置{}，目标位置{}，距离{}'.format(current_pos, target_pos, distance))
        # 如果接近目标点，切换到下一个目标
        if distance < position_tolerance:
            print(f"到达点 {pos}")
        # 计算移动速度向量
        if distance > 0:
            direction_normalized = direction / distance
            velocity = direction_normalized * min(speed, distance * 10)
        else:
            velocity = [0, 0, 0]
        # 应用速度（重置为新的速度）
        p.resetBaseVelocity(id, linearVelocity=velocity)
        p.applyExternalForce(id, -1, [0, 0, 0.98], [0, 0, 0], p.WORLD_FRAME)  #克服重力

