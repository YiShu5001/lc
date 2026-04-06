import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gym_pybullet_drones.control.LADRC import LADRCControl

class TSA_LADRC_Env(BaseRLAviary):
    """
    Two-Scale Action (TSA) Reinforcement Learning Environment for LADRC dynamic tuning.
    
    Architecture Decoupling:
    - Environment purely handles Physics (PyBullet), Controller Math (LADRC), and pure Tracking Reward.
    - Two-Scale Mechanism: RL operates at `rl_freq` (e.g., 10Hz), Controller at `ctrl_freq` (e.g., 100Hz).
    - Advanced reward shaping (smoothness penalty) and state stacking should be handled via Wrappers.
    """
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 100,  # 100Hz controller loop
                 rl_freq: int = 10,     # 10Hz RL action loop
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 **kwargs):
        
        self.RL_FREQ = rl_freq
        if ctrl_freq % rl_freq != 0:
            raise ValueError("[ERROR] ctrl_freq must be divisible by rl_freq for TSA-RL Action Holding.")
        self.CTRL_STEPS_PER_RL = int(ctrl_freq / rl_freq)
        
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=ActionType.RPM) # Using raw RPM mapping internally
        
        # Replace default DSLPIDControl controllers with LADRCControl
        self.ctrl = [LADRCControl(drone_model=self.DRONE_MODEL) for _ in range(self.NUM_DRONES)]
        
        # Action space: [dwc, db0] for 6 channels = 12 variables
        self.action_dim = 12
        
        # Configurable boundaries for LADRC tuning
        self.param_bounds = {'omega_c': (1.0, 50.0), 'b0': (10.0, 1000.0)}
        self.max_delta = {'omega_c': 2.0, 'b0': 50.0}
        
        # Standard target points (wrappers or curriculum can update this)
        self.target_pos = np.array([[0.0, 0.0, 1.0] for _ in range(self.NUM_DRONES)])
        self.target_rpy = np.zeros((self.NUM_DRONES, 3))
        
        self.step_counter_rl = 0
        self.last_action = np.zeros((self.NUM_DRONES, self.action_dim))

    def _actionSpace(self):
        """ Continuous action space [-1, 1] for parameter deltas """
        act_lower_bound = np.array([-1 * np.ones(self.action_dim) for _ in range(self.NUM_DRONES)])
        act_upper_bound = np.array([+1 * np.ones(self.action_dim) for _ in range(self.NUM_DRONES)])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    def _observationSpace(self):
        """ 
        Base Observation space (Dim: 50).
        Kinematic(20) + TargetPos(3) + TargetRPY(3) + CurrentParams(12) + LastAction(12) = 50.
        Note: The Algo layer or a Wrapper is responsible for state-stacking (e.g. deque) for temporal awareness.
        """
        obs_dim = 50
        obs_lower_bound = np.array([-np.inf * np.ones(obs_dim) for _ in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([np.inf * np.ones(obs_dim) for _ in range(self.NUM_DRONES)])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self):
        obs = np.zeros((self.NUM_DRONES, 50))
        for i in range(self.NUM_DRONES):
            kin_state = self._getDroneStateVector(i)
            ctrl = self.ctrl[i]
            params = np.array([
                ctrl.con_X.omega_c, ctrl.con_X.b0,
                ctrl.con_Y.omega_c, ctrl.con_Y.b0,
                ctrl.con_Z.omega_c, ctrl.con_Z.b0,
                ctrl.con_roll.omega_c, ctrl.con_roll.b0,
                ctrl.con_pitch.omega_c, ctrl.con_pitch.b0,
                ctrl.con_yaw.omega_c, ctrl.con_yaw.b0
            ])
            obs[i, :] = np.hstack([kin_state, self.target_pos[i], self.target_rpy[i], params, self.last_action[i]])
        return obs.astype(np.float32)

    def step(self, action):
        """ Two-Scale mechanism implementation """
        # Ensure action shape is appropriate
        if len(action.shape) == 1 and self.NUM_DRONES == 1:
            action = np.expand_dims(action, axis=0)
            
        self.last_action = action
        
        # 1. Update Parameters (RL Layer -> Control Layer)
        for k in range(self.NUM_DRONES):
            dwc = action[k, 0::2] * self.max_delta['omega_c']
            db0 = action[k, 1::2] * self.max_delta['b0']
            
            ctrl = self.ctrl[k]
            channels = [ctrl.con_X, ctrl.con_Y, ctrl.con_Z, ctrl.con_roll, ctrl.con_pitch, ctrl.con_yaw]
            for idx, ch in enumerate(channels):
                new_wc = np.clip(ch.omega_c + dwc[idx], self.param_bounds['omega_c'][0], self.param_bounds['omega_c'][1])
                new_b0 = np.clip(ch.b0 + db0[idx], self.param_bounds['b0'][0], self.param_bounds['b0'][1])
                ch.set_params(omega_c=new_wc, b0=new_b0, omega_o=ch.omega_o)

        # 2. Action Holding: Step environment at Control Frequency (e.g. 10x per RL step)
        cumulative_reward = 0
        for _ in range(self.CTRL_STEPS_PER_RL):
            # Inside super().step, self._preprocessAction will be called using the updated params
            obs, reward, terminated, truncated, info = super().step(action)
            
            # Aggregate rewards
            cumulative_reward = cumulative_reward + reward
            
            # Check early stopping conditions during inner loop
            if (isinstance(terminated, np.ndarray) and terminated.any()) or \
               (isinstance(terminated, bool) and terminated):
                break
                
        self.step_counter_rl += 1
        
        # For evaluation/ablation ease, the environment only returns pure tracking accumulation.
        return obs, cumulative_reward, terminated, truncated, info

    def _preprocessAction(self, action):
        """ Control Layer execution. Action represents RPM computation given *current* parameters. """
        rpm = np.zeros((self.NUM_DRONES, 4))
        for k in range(self.NUM_DRONES):
            state = self._getDroneStateVector(k)
            # computeControl uses the internally stored LADRC parameters
            rpm_k, _, _ = self.ctrl[k].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=self.target_pos[k],
                target_rpy=self.target_rpy[k]
            )
            rpm[k, :] = rpm_k
        return rpm

    def _computeReward(self):
        """ Pure mathematical tracking reward. Add smoothing in Wrappers for ablation. """
        rewards = np.zeros(self.NUM_DRONES)
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            pos_error = np.linalg.norm(self.target_pos[i] - state[0:3])
            # Scale tracking error down slightly to prevent huge magnitude
            rewards[i] = - (pos_error * 0.1)
            
        return rewards[0] if self.NUM_DRONES == 1 else rewards

    def _computeTerminated(self):
        terminated = np.zeros(self.NUM_DRONES, dtype=bool)
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            pos = state[0:3]
            # Terminate if drone flies far out of bounds or drops too low
            if np.linalg.norm(pos) > 10.0 or pos[2] < 0.05:
                terminated[i] = True
        return terminated[0] if self.NUM_DRONES == 1 else terminated

    def _computeTruncated(self):
        # Truncate after 500 RL steps
        trunc = self.step_counter_rl >= 500
        return trunc if self.NUM_DRONES == 1 else np.ones(self.NUM_DRONES, dtype=bool) * trunc

    def _computeInfo(self):
        # Info can be populated with specifics for logging
        return {'target_pos': self.target_pos, 'rl_step': self.step_counter_rl}
