import numpy as np
from dataclasses import dataclass

@dataclass
class LADRCConfig:
    """LADRC Controller Configuration"""
    omega_c: float = 10.0  # Controller bandwidth
    omega_o: float = 30.0  # Observer bandwidth (usually 3~5 * omega_c)
    b0: float = 1.0        # System gain estimate
    step_size: float = 0.01 # Control loop dt

class LESO:
    """Linear Extended State Observer (2nd Order System)"""
    def __init__(self, config: LADRCConfig):
        self.cfg = config
        self.beta1 = 3 * config.omega_o
        self.beta2 = 3 * config.omega_o**2
        self.beta3 = config.omega_o**3
        
        # State: [z1 (y), z2 (y_dot), z3 (f)]
        self.z = np.zeros(3)

    def update(self, y_meas: float, u_prev: float):
        """
        Update observer state
        y_meas: Measured output (e.g., angle)
        u_prev: Previous control input
        """
        err = self.z[0] - y_meas
        
        # Euler integration
        dz1 = self.z[1] - self.beta1 * err
        dz2 = self.z[2] - self.beta2 * err + self.cfg.b0 * u_prev
        dz3 = -self.beta3 * err
        
        self.z[0] += dz1 * self.cfg.step_size
        self.z[1] += dz2 * self.cfg.step_size
        self.z[2] += dz3 * self.cfg.step_size
        
        return self.z

class LADRC:
    """Linear Active Disturbance Rejection Controller"""
    def __init__(self, config: LADRCConfig):
        self.cfg = config
        self.leso = LESO(config)
        
        # Controller gains (PD part)
        self.kp = config.omega_c**2
        self.kd = 2 * config.omega_c
        
        self.u_last = 0.0

    def set_params(self, omega_c: float, b0: float = None):
        """RL Adaptive Interface: Update parameters dynamically"""
        self.cfg.omega_c = omega_c
        self.cfg.omega_o = 3.0 * omega_c # Keep observer faster
        if b0: self.cfg.b0 = b0
        
        # Re-calculate gains
        self.kp = self.cfg.omega_c**2
        self.kd = 2 * self.cfg.omega_c
        
        # Re-calculate LESO gains
        self.leso.beta1 = 3 * self.cfg.omega_o
        self.leso.beta2 = 3 * self.cfg.omega_o**2
        self.leso.beta3 = self.cfg.omega_o**3

    def update(self, ref: float, y_meas: float) -> float:
        """
        Calculate control output
        ref: Target setpoint
        y_meas: Measured system output
        """
        # 1. Update Observer
        z = self.leso.update(y_meas, self.u_last)
        z1, z2, z3 = z[0], z[1], z[2] # est_y, est_dy, est_dist
        
        # 2. Control Law (u0 = kp*e + kd*de)
        error = ref - z1
        d_error = 0 - z2 # Assuming ref_dot is 0 for step/regulation
        
        u0 = self.kp * error + self.kd * d_error
        
        # 3. Disturbance Rejection
        u = (u0 - z3) / self.cfg.b0
        
        self.u_last = u
        return u

    def reset(self):
        self.leso.z = np.zeros(3)
        self.u_last = 0.0
