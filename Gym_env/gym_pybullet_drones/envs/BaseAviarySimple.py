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
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType


class BaseAviary(gym.Env):
    """Base class for "drone aviary" Gym environments."""

    # metadata = {'render.modes': ['human']}
    
    ################################################################################

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
                 obstacles=True,
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
        #### Parameters ############################################
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
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()

    
    ################################################################################

    def reset(self,
              seed : int = None,
              options : dict = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """

        # TODO : initialize random number generator with seed

        p.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()

        #### Return the initial observation ########################
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info
    
    ################################################################################

    def step(self,
             action
             ):

        #### Read the GUI's input parameters #######################
        '''    手动模式  USER_DEBUG, 当使用鼠标点击会触发 '''
        if self.GUI and self.USER_DEBUG:
            current_input_switch = p.readUserDebugParameter(self.INPUT_SWITCH, physicsClientId=self.CLIENT)
            if current_input_switch > self.last_input_switch:
                self.last_input_switch = current_input_switch
                self.USE_GUI_RPM = True if self.USE_GUI_RPM == False else False
        if self.USE_GUI_RPM:
            for i in range(4):
                self.gui_input[i] = p.readUserDebugParameter(int(self.SLIDERS[i]), physicsClientId=self.CLIENT)
            clipped_action = np.tile(self.gui_input, (self.NUM_DRONES, 1))
            if self.step_counter%(self.PYB_FREQ/2) == 0:
                self.GUI_INPUT_TEXT = [p.addUserDebugText("Using GUI RPM",
                                                          textPosition=[0, 0, 0],
                                                          textColorRGB=[1, 0, 0],
                                                          lifeTime=1,
                                                          textSize=2,
                                                          parentObjectUniqueId=self.DRONE_IDS[i],
                                                          parentLinkIndex=-1,
                                                          replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i]),
                                                          physicsClientId=self.CLIENT
                                                          ) for i in range(self.NUM_DRONES)]
        #### Save, preprocess, and clip the action to the max. RPM #
        else:
            clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))

        #### Repeat for as many as the aggregate physics steps #####
        for _ in range(self.PYB_STEPS_PER_CTRL):  # 每次控制频率  仿真频率/控制频率 ， 一次动作保持多少步长
            #### Update and store the drones kinematic info for certain
            #### Between aggregate steps for certain types of update ###
            if self.PYB_STEPS_PER_CTRL > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                self._updateAndStoreKinematicInformation()
            #### Step the simulation using the desired physics update
            # 不同仿真环境调用不同的形式
            # ##
            for i in range (self.NUM_DRONES):
                if self.PHYSICS == Physics.PYB:
                    self._physics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.DYN:
                    self._dynamics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_GND:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DRAG:
                    self._physics(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DW:
                    self._physics(clipped_action[i, :], i)
                    self._downwash(i)
                elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                    self._downwash(i)
            #### PyBullet computes the new state, unless Physics.DYN ###
            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.CLIENT)
            #### Save the last applied action (e.g. to compute drag) ###
            self.last_clipped_action = clipped_action
        #### 更新无人机的物理信息 #####
        self._updateAndStoreKinematicInformation()
        #### 返还值 #############################
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)
        return obs, reward, terminated, truncated, info
    
    ################################################################################
    
    def render(self,
               mode='human',
               close=False
               ):
        """Prints a textual output of the environment.
               文本模式渲染状态信息，  显示位置/速度/姿态等信息
        Parameters
        ----------
        mode : str, optional
            Unused.
        close : bool, optional
            Unused.

        """
        if self.first_render_call and not self.GUI:
            print("[WARNING] BaseAviary.render() is implemented as text-only, re-initialize the environment using Aviary(gui=True) to use PyBullet's graphical interface")
            self.first_render_call = False
        print("\n[INFO] BaseAviary.render() ——— it {:04d}".format(self.step_counter),
              "——— wall-clock time {:.1f}s,".format(time.time()-self.RESET_TIME),
              "simulation time {:.1f}s@{:d}Hz ({:.2f}x)".format(self.step_counter*self.PYB_TIMESTEP, self.PYB_FREQ, (self.step_counter*self.PYB_TIMESTEP)/(time.time()-self.RESET_TIME)))
        for i in range (self.NUM_DRONES):
            print("[INFO] BaseAviary.render() ——— drone {:d}".format(i),
                  "——— x {:+06.2f}, y {:+06.2f}, z {:+06.2f}".format(self.pos[i, 0], self.pos[i, 1], self.pos[i, 2]),
                  "——— velocity {:+06.2f}, {:+06.2f}, {:+06.2f}".format(self.vel[i, 0], self.vel[i, 1], self.vel[i, 2]),
                  "——— roll {:+06.2f}, pitch {:+06.2f}, yaw {:+06.2f}".format(self.rpy[i, 0]*self.RAD2DEG, self.rpy[i, 1]*self.RAD2DEG, self.rpy[i, 2]*self.RAD2DEG),
                  "——— angular velocity {:+06.4f}, {:+06.4f}, {:+06.4f} ——— ".format(self.ang_v[i, 0], self.ang_v[i, 1], self.ang_v[i, 2]))
    
    ################################################################################

    def close(self):
        """Terminates the environment.
        """
        if self.RECORD and self.GUI:
            p.stopStateLogging(self.VIDEO_ID, physicsClientId=self.CLIENT)
        p.disconnect(physicsClientId=self.CLIENT)
    
    ################################################################################
    #  添加障碍物
    def _addObstacles(self):
        """Adds obstacles to the environment.
        """
        # TODO：  障碍物设置
        self.Obs[1] = self._AddObstacle(pos=[0, 0, 1], shape=[1, 0.1, 0.1], color=[1, 0, 0, 1], mass=2,
                                        ClientId=self.CLIENT)
        self.Obs[2] = self._AddObstacle( pos=[1, 0, 0.5], shape=[0.1, 0.1, 0.5], color=[1, 0, 0, 1], mass=2,
                                        ClientId=self.CLIENT)
        self.Obs[3] = self._AddObstacle(pos=[-1, 0, 0.5], shape=[0.1, 0.1,0.5], color=[1, 0, 0, 1], mass=2,
                                        ClientId=self.CLIENT)

    def _AddObstacle(nth: int = None, pos=[0, 0, 2], shape=[0.5, 0.5, 0.5], color=[1, 0, 0, 1], mass=2,
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

        return p.createMultiBody(
            baseMass=mass,  # 1kg
            baseCollisionShapeIndex=collisionShapeId,
            baseVisualShapeIndex=visualShapeId,
            basePosition=pos,  # 初始位置（x,y,z）
            physicsClientId=ClientId
        )
    ################################################################################
    def getPyBulletClient(self):
        """Returns the PyBullet Client Id.

        Returns
        -------
        int:
            The PyBullet Client Id.

        """
        return self.CLIENT
    
    ################################################################################

    def getDroneIds(self):
        """Return the Drone Ids.

        Returns
        -------
        ndarray:
            (NUM_DRONES,)-shaped array of ints containing the drones' ids.

        """
        return self.DRONE_IDS
    
    ################################################################################

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
        self.Obs = {}
        if self.OBSTACLES:
            self._addObstacles()
    
    ################################################################################

    def _updateAndStoreKinematicInformation(self):
        """Updates and stores the drones kinemaatic information.
更新并存储无人机动力学信息。
        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).

        """
        for i in range (self.NUM_DRONES):
            self.pos[i], self.quat[i] = p.getBasePositionAndOrientation(self.DRONE_IDS[i], physicsClientId=self.CLIENT)
            self.rpy[i] = p.getEulerFromQuaternion(self.quat[i])
            self.vel[i], self.ang_v[i] = p.getBaseVelocity(self.DRONE_IDS[i], physicsClientId=self.CLIENT)
    
    ################################################################################

    def _getDroneStateVector(self,
                             nth_drone
                             ):
        """Returns the state vector of the n-th drone.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        Returns
        -------
        ndarray 
            (20,)-shaped array of floats containing the state vector of the n-th drone.
            Check the only line in this method and `_updateAndStoreKinematicInformation()`
            to understand its format.
       ndarray
            (20,)维状态向量，包含：
            [0:3]  位置坐标 x/y/z（米）
            [3:7]  姿态四元数 qw/qx/qy/qz
            [7:10] 欧拉角 roll/pitch/yaw（弧度）
            [10:13] 线速度 vx/vy/vz（米/秒）
            [13:16] 角速度 wx/wy/wz（弧度/秒）
            [16:20] 上一次控制动作（4个电机的RPM值）
        """
        state = np.hstack([self.pos[nth_drone, :], self.quat[nth_drone, :], self.rpy[nth_drone, :],
                           self.vel[nth_drone, :], self.ang_v[nth_drone, :], self.last_clipped_action[nth_drone, :]])
        return state.reshape(20,)

    ################################################################################

    def _getAdjacencyMatrix(self):
        # TODO: 获得一个上三角矩阵 1 距离够近有交互信息 0 距离不够近没有交互信息
        """Computes the adjacency matrix of a multi-drone system.
                计算邻接矩阵
        Attribute NEIGHBOURHOOD_RADIUS is used to determine neighboring relationships.

        Returns
        -------
        ndarray
            i，i 矩阵：表示是否能观测

            (NUM_DRONES, NUM_DRONES)-shaped array of 0's and 1's representing the adjacency matrix
            of the system: adj_mat[i,j] == 1 if (i, j) are neighbors; == 0 otherwise.

        """
        adjacency_mat = np.identity(self.NUM_DRONES)  # 先创建单位矩阵（对角线为1，其余为0）
        for i in range(self.NUM_DRONES - 1):  # 遍历除最后一架外的所有无人机
            for j in range(self.NUM_DRONES - i - 1):  # 遍历当前无人机之后的所有无人机
                # TODO 计算两架无人机之间的欧氏距离
                if np.linalg.norm(self.pos[i, :] - self.pos[j + i + 1, :]) < self.NEIGHBOURHOOD_RADIUS:
                    # 如果距离小于阈值，将矩阵对应位置设为1（对称设置）
                    adjacency_mat[i, j + i + 1] = adjacency_mat[j + i + 1, i] = 1
        return adjacency_mat     # 返回邻接矩阵

    ################################################################################
    def _dynamics(self,
                  rpm,
                  nth_drone
                  ):
        """Explicit dynamics implementation.

        Based on code written at the Dynamic Systems Lab by James Xu.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        参数:
            rpm: 4个电机的转速值(4,)数组
            nth_drone: 目标无人机在列表中的索引
        """
        #### 当前状态 #########################################
        pos = self.pos[nth_drone, :]  # 当前位置(x,y,z)
        quat = self.quat[nth_drone, :]  # 当前姿态(四元数)
        vel = self.vel[nth_drone, :]  # 当前线速度
        rpy_rates = self.rpy_rates[nth_drone, :]  # 当前角速度(roll,pitch,yaw)
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)  # 旋转矩阵
        #### 计算力和扭矩 ############################
        forces = np.array(rpm ** 2) * self.KF  # 计算4个螺旋桨的推力(F=KF*rpm^2)
        thrust = np.array([0, 0, np.sum(forces)])  # 总推力(Z轴方向)
        thrust_world_frame = np.dot(rotation, thrust)  # 将推力从自身坐标系转换到世界坐标系
        force_world_frame = thrust_world_frame - np.array([0, 0, self.GRAVITY])  # 计算净力(减去重力)

        # 计算扭矩
        z_torques = np.array(rpm ** 2) * self.KM  # 计算每个螺旋桨的扭矩
        if self.DRONE_MODEL == DroneModel.RACE:  # 如果是RACE竞速无人机模型，扭矩方向相反
            z_torques = -z_torques
        z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])  # 计算总偏航扭矩

        # 根据无人机型号计算滚转和俯仰扭矩
        if self.DRONE_MODEL == DroneModel.CF2X or self.DRONE_MODEL == DroneModel.RACE:
            x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self.L / np.sqrt(2))
            y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * (self.L / np.sqrt(2))
        elif self.DRONE_MODEL == DroneModel.CF2P:
            x_torque = (forces[1] - forces[3]) * self.L
            y_torque = (-forces[0] + forces[2]) * self.L

        torques = np.array([x_torque, y_torque, z_torque])  # 组合所有扭矩
        torques = torques - np.cross(rpy_rates, np.dot(self.J, rpy_rates))  # 考虑陀螺效应
        rpy_rates_deriv = np.dot(self.J_INV, torques)  # 计算角加速度
        no_pybullet_dyn_accs = force_world_frame / self.M  # 计算线加速度(F=ma)

        #### 更新状态 ##########################################
        vel = vel + self.PYB_TIMESTEP * no_pybullet_dyn_accs  # 更新线速度
        rpy_rates = rpy_rates + self.PYB_TIMESTEP * rpy_rates_deriv  # 更新角速度
        pos = pos + self.PYB_TIMESTEP * vel  # 更新位置
        quat = self._integrateQ(quat, rpy_rates, self.PYB_TIMESTEP)  # 积分更新姿态

        #### 设置无人机的状态和速度信息  ##################################
        p.resetBasePositionAndOrientation(self.DRONE_IDS[nth_drone],
                                          pos,
                                          quat,
                                          physicsClientId=self.CLIENT
                                          )

        p.resetBaseVelocity(self.DRONE_IDS[nth_drone],
                            vel,
                            np.dot(rotation, rpy_rates),
                            physicsClientId=self.CLIENT
                            )
        ####  # 保存当前角速度到类属性，下一个状态 ####
        self.rpy_rates[nth_drone, :] = rpy_rates
    ##################################################################################
    def _physics(self,
                 rpm,
                 nth_drone
                 ):
        """Base PyBullet physics implementation.
        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        rpm的转速
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        第几个无人机
        """
        forces = np.array(rpm**2)*self.KF
        torques = np.array(rpm**2)*self.KM

        if self.DRONE_MODEL == DroneModel.RACE:
            torques = -torques
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
        for i in range(4):
            p.applyExternalForce(
                self.DRONE_IDS[nth_drone],  # 目标物体的唯一ID（这里指第n个无人机）
                i,  # 施加力的连杆索引（0-3对应4个螺旋桨）
                forceObj=[0, 0, forces[i]],  # 施加的力向量（Z轴方向，大小由forces[i]决定）
                posObj=[0, 0, 0],  # 施加力的位置（连杆坐标系原点）
                flags=p.LINK_FRAME,  # 坐标系标志（表示力向量在连杆局部坐标系中定义）
                # LINK_FRAME表示力向量是在螺旋桨连杆的局部坐标系中定义的
                physicsClientId=self.CLIENT  # 物理引擎客户端ID
            )
        p.applyExternalTorque(
            self.DRONE_IDS[nth_drone],  # 目标物体的唯一ID（这里指第n个无人机）
            4,  # 施加扭矩的连杆索引（4对应无人机主体/中心质量）
            torqueObj=[0, 0, z_torque],  # 扭矩向量（Z轴方向，大小由z_torque决定）
            flags=p.LINK_FRAME,  # 坐标系标志（表示扭矩在连杆局部坐标系中定义）
            physicsClientId=self.CLIENT  # 物理引擎客户端ID
        )

    ################################################################################

    def _groundEffect(self, rpm, nth_drone):
        """地面效应模型实现"""
        # 获取无人机所有连杆(4个螺旋桨+主体)的运动学信息
        link_states = p.getLinkStates(
            self.DRONE_IDS[nth_drone],
            linkIndices=[0, 1, 2, 3, 4],  # 0-3:螺旋桨, 4:主体
            computeLinkVelocity=1,  # 计算连杆速度
            computeForwardKinematics=1,  # 计算正向运动学
            physicsClientId=self.CLIENT
        )

        # 提取4个螺旋桨的高度(z坐标)
        prop_heights = np.array([
            link_states[0][0][2],  # 螺旋桨1高度
            link_states[1][0][2],  # 螺旋桨2高度
            link_states[2][0][2],  # 螺旋桨3高度
            link_states[3][0][2]  # 螺旋桨4高度
        ])

        # 限制最小高度，防止除零错误
        prop_heights = np.clip(prop_heights, self.GND_EFF_H_CLIP, np.inf)

        # 计算每个螺旋桨的地面效应力(与RPM平方成正比，与高度平方成反比)
        gnd_effects = np.array(rpm ** 2) * self.KF * self.GND_EFF_COEFF * \
                      (self.PROP_RADIUS / (4 * prop_heights)) ** 2

        # 只在无人机倾斜角度小于90度时应用地面效应
        if np.abs(self.rpy[nth_drone, 0]) < np.pi / 2 and \
                np.abs(self.rpy[nth_drone, 1]) < np.pi / 2:
            # 对每个螺旋桨施加计算得到的地面效应力
            for i in range(4):
                p.applyExternalForce(
                    self.DRONE_IDS[nth_drone],
                    i,  # 螺旋桨索引
                    forceObj=[0, 0, gnd_effects[i]],  # Z轴方向的力
                    posObj=[0, 0, 0],  # 力的作用点(螺旋桨局部坐标系原点)
                    flags=p.LINK_FRAME,  # 使用连杆局部坐标系
                    physicsClientId=self.CLIENT
                )
    
    ################################################################################

    def _drag(self,
              rpm,
              nth_drone
              ):
        """PyBullet implementation of a drag model.

        Based on the the system identification in (Forster, 2015).

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """

        def _drag(self, rpm, nth_drone):
            """空气阻力模型实现"""
            # 获取无人机当前姿态的旋转矩阵（从四元数转换）
            base_rot = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)

            # 计算阻力系数（与电机转速总和成正比）
            drag_factors = -1 * self.DRAG_COEFF * np.sum(np.array(2 * np.pi * rpm / 60))

            # 计算阻力向量（在无人机局部坐标系中）
            drag = np.dot(base_rot.T, drag_factors * np.array(self.vel[nth_drone, :]))

            # 对无人机主体施加计算得到的阻力
            p.applyExternalForce(
                self.DRONE_IDS[nth_drone],  # 目标无人机ID
                4,  # 4表示无人机主体/中心质量
                forceObj=drag,  # 阻力向量
                posObj=[0, 0, 0],  # 作用点（中心点）
                flags=p.LINK_FRAME,  # 使用连杆局部坐标系
                physicsClientId=self.CLIENT
            )
    
    ################################################################################

    def _downwash(self,
                  nth_drone
                  ):
        """PyBullet implementation of a ground effect model.

        Based on experiments conducted at the Dynamic Systems Lab by SiQi Zhou.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        """下洗流效应模型实现"""
        # 遍历所有其他无人机
        for i in range(self.NUM_DRONES):
            # 计算高度差(当前无人机在上方时delta_z为正)
            delta_z = self.pos[i, 2] - self.pos[nth_drone, 2]
            # 计算水平距离
            delta_xy = np.linalg.norm(np.array(self.pos[i, 0:2]) - np.array(self.pos[nth_drone, 0:2]))

            # 只考虑上方无人机(高度差>0)且在10米范围内的下洗流影响
            if delta_z > 0 and delta_xy < 10:
                # 计算下洗流强度系数alpha(与螺旋桨半径和高度差相关)
                alpha = self.DW_COEFF_1 * (self.PROP_RADIUS / (4 * delta_z)) ** 2
                # 计算下洗流范围系数beta(线性依赖于高度差)
                beta = self.DW_COEFF_2 * delta_z + self.DW_COEFF_3
                # 计算下洗流力(高斯分布模型，随水平距离衰减)
                downwash = [0, 0, -alpha * np.exp(-.5 * (delta_xy / beta) ** 2)]

                # 对当前无人机施加下洗流力
                p.applyExternalForce(
                    self.DRONE_IDS[nth_drone],
                    4,  # 作用于无人机主体
                    forceObj=downwash,  # 向下的力
                    posObj=[0, 0, 0],  # 作用在中心点
                    flags=p.LINK_FRAME,  # 使用局部坐标系
                    physicsClientId=self.CLIENT
                )

    ################################################################################



    def _integrateQ(self, quat, omega, dt):
        # 四元数(quat)积分方法，用于根据角速度(omega)和时间步长(dt)来更新四元数姿态 （integrateQ）。
        # 这是无人机动力学仿真中常用的姿态更新方法。
        omega_norm = np.linalg.norm(omega)  # 计算角速度向量的模(大小)
        p, q, r = omega  # 解构角速度分量
        if np.isclose(omega_norm, 0):
            return quat  # 如果角速度接近零，直接返回原四元数
        lambda_ = np.array([
            [ 0,  r, -q, p],
            [-r,  0,  p, q],
            [ q, -p,  0, r],
            [-p, -q, -r, 0]
        ]) * .5                # 这是四元数导数的矩阵表示
        theta = omega_norm * dt / 2  # 计算旋转角度的一半(用于四元数更新)
        quat = np.dot(np.eye(4) * np.cos(theta) + 2 / omega_norm * lambda_ * np.sin(theta), quat)
        return quat   # 返回新的四元数

    ################################################################################

    def _normalizedActionToRPM(self,
                               action
                               ):
        """De-normalizes the [-1, 1] range to the [0, MAX_RPM] range.

        Parameters
        ----------
        action : ndarray
            (4)-shaped array of ints containing an input in the [-1, 1] range.

        Returns
        -------
        ndarray
            (4)-shaped array of ints containing RPMs for the 4 motors in the [0, MAX_RPM] range.

        """
        if np.any(np.abs(action) > 1):
            print("\n[ERROR] it", self.step_counter, "in BaseAviary._normalizedActionToRPM(), out-of-bound action")
        return np.where(action <= 0, (action+1)*self.HOVER_RPM, self.HOVER_RPM + (self.MAX_RPM - self.HOVER_RPM)*action)
        # Non-linear mapping: -1 -> 0, 0 -> HOVER_RPM, 1 -> MAX_RPM`
    
    ################################################################################
    #  绘制每个无人机的局部坐标系
    def _showDroneLocalAxes(self,
                            nth_drone
                            ):
        """

        在图形中设置无人机的三位坐标状态，用于可视化。

        Draws the local frame of the n-th drone in PyBullet's GUI.
在Pybullet的GUI中绘制第n架无人机的本地框架。，，
        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        if self.GUI:
            AXIS_LENGTH = 2*self.L
            self.X_AX[nth_drone] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[AXIS_LENGTH, 0, 0],
                                                      lineColorRGB=[1, 0, 0],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.X_AX[nth_drone]),
                                                      physicsClientId=self.CLIENT
                                                      )
            self.Y_AX[nth_drone] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[0, AXIS_LENGTH, 0],
                                                      lineColorRGB=[0, 1, 0],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.Y_AX[nth_drone]),
                                                      physicsClientId=self.CLIENT
                                                      )
            self.Z_AX[nth_drone] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[0, 0, AXIS_LENGTH],
                                                      lineColorRGB=[0, 0, 1],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.Z_AX[nth_drone]),
                                                      physicsClientId=self.CLIENT
                                                      )
    
    ################################################################################

    # 获取无人机模型基础状态
    def _parseURDFParameters(self):
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.
        URDF文件结构示例
<robot>
  <arm>0.17</arm>                               ➔ L (机臂长度)
  <kf>6.11e-8</kf>                             ➔ KF (推力系数)
  <drag_coeff_xy>0.1</drag_coeff_xy>           ➔ XY平面阻力
  <inertial>
    <mass value="0.5"/>                         ➔ M (质量)
    <inertia ixx="0.003" iyy="0.003" izz="0.005"/> ➔ 转动惯量
  </inertial>
  <collision>
    <origin xyz="0 0 0.02"/>                    ➔ COLLISION_Z_OFFSET
    <cylinder length="0.04" radius="0.08"/>     ➔ 碰撞体尺寸
  </collision>
</robot>

        """
        URDF_TREE = etxml.parse(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+self.URDF)).getroot()
        M = float(URDF_TREE[1][0][1].attrib['value'])
        L = float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib['kf'])
        KM = float(URDF_TREE[0].attrib['km'])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
               GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3
    
    ################################################################################
    
    def _actionSpace(self):
        """Returns the action space of the environment.

        Must be implemented in a subclass.

        """
        print('未设定动作空间')
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Must be implemented in a subclass.

        """
        print('未设定状态空间')
    
    ################################################################################
    
    def _computeObs(self):
        """Returns the current observation of the environment.

        Must be implemented in a subclass.

        """
        print('未定义计算状态值')
    
    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Must be implemented in a subclass.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, to be translated into RPMs.

        """
        print('未设定动作缩放')
        return action
    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Must be implemented in a subclass.

        """
        print('未定义奖励')

    ################################################################################

    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Must be implemented in a subclass.

        """
        print('未定义终止符')
    
    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Must be implemented in a subclass.

        """
        print('未定义截断符')

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Must be implemented in a subclass.

        """
        print('未定义计算info')

    ################################################################################
    #  从无人机的当前位置计算到无人机目的地的中间航路点
    def _calculateNextStep(self, current_position, destination, step_size=1):
        """
        Calculates intermediate waypoint towards drone's destination from drone's current position

        Enables drones to reach distant waypoints without losing control/crashing, and hover on arrival at destintion

        Parameters
        ----------
        current_position : ndarray
            drone's current position from state vector
        destination : ndarray
            drone's target position 
        step_size: int
            distance next waypoint is from current position, default 1

        Returns
        ----------
        next_pos: int 
            intermediate waypoint for drone

        """
        direction = (
            destination - current_position
        )  # Calculate the direction vector
        distance = np.linalg.norm(
            direction
        )  # Calculate the distance to the destination

        if distance <= step_size:
            # If the remaining distance is less than or equal to the step size,
            # return the destination
            return destination

        normalized_direction = (
            direction / distance
        )  # Normalize the direction vector
        next_step = (
            current_position + normalized_direction * step_size
        )  # Calculate the next step
        return next_step

if __name__ == '__main__':
    DEFAULT_DRONES = DroneModel("cf2x")
    DEFAULT_NUM_DRONES = 1
    DEFAULT_PHYSICS = Physics("pyb")
    DEFAULT_GUI = True
    DEFAULT_RECORD_VISION = False
    DEFAULT_PLOT = True
    DEFAULT_USER_DEBUG_GUI = True
    DEFAULT_OBSTACLES = True
    DEFAULT_SIMULATION_FREQ_HZ = 240
    DEFAULT_CONTROL_FREQ_HZ = 48
    DEFAULT_DURATION_SEC = 12
    DEFAULT_OUTPUT_FOLDER = 'results'
    DEFAULT_COLAB = False
    env = BaseAviary(drone_model=DEFAULT_DRONES,
                        num_drones=DEFAULT_NUM_DRONES,
                        physics=DEFAULT_PHYSICS,
                        neighbourhood_radius=10,
                        pyb_freq=DEFAULT_SIMULATION_FREQ_HZ,
                        ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,
                        gui=DEFAULT_GUI,
                        record=False,
                        obstacles=DEFAULT_OBSTACLES,
                        user_debug_gui=DEFAULT_USER_DEBUG_GUI)

    PYB_CLIENT = env.getPyBulletClient()
    action = [[12480.11953626,15057.71953626,14198.51953626, 15057.71953626]]
    time.sleep(2)
    for _ in range(100000):
        env.step(action=action)
    env.close()