import numpy as np
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class LADRCConfig:
    """LADRC Controller Configuration (Single Channel)"""
    omega_c: float = 10.0   # Controller bandwidth
    b0: float = 1.0         # System gain estimate
    omega_o: float = 30.0   # Observer bandwidth (usually 3~5 * omega_c)
    step_size: float = 0.01 # Control loop dt
    r: float = 50.0         # Tracking Differentiator speed factor (transient profile)
    use_td: bool = True     # Whether to use Tracking Differentiator

class TD:
    """Tracking Differentiator"""
    def __init__(self, config: LADRCConfig):
        self.cfg = config
        self.v1 = 0.0
        self.v2 = 0.0

    def update(self, target: float):
        if not self.cfg.use_td:
            self.v1 = target
            self.v2 = 0.0
            return self.v1, self.v2
            
        fh = -self.cfg.r**2 * (self.v1 - target) - 2 * self.cfg.r * self.v2
        self.v1 += self.v2 * self.cfg.step_size
        self.v2 += fh * self.cfg.step_size
        return self.v1, self.v2
        
    def reset(self):
        self.v1 = 0.0
        self.v2 = 0.0

class LESO:
    """Linear Extended State Observer (2nd Order System)"""
    def __init__(self, config: LADRCConfig):
        self.cfg = config
        # State: [z1 (y), z2 (y_dot), z3 (f)]
        self.z = np.zeros(3)

    def update(self, y_meas: float, u_prev: float):
        """
        Update observer state
        y_meas: Measured output (e.g., angle or position)
        u_prev: Previous control input
        """
        beta1 = 3 * self.cfg.omega_o
        beta2 = 3 * self.cfg.omega_o**2
        beta3 = self.cfg.omega_o**3

        err = self.z[0] - y_meas
        
        # Euler integration
        dz1 = self.z[1] - beta1 * err
        dz2 = self.z[2] - beta2 * err + self.cfg.b0 * u_prev
        dz3 = -beta3 * err
        
        self.z[0] += dz1 * self.cfg.step_size
        self.z[1] += dz2 * self.cfg.step_size
        self.z[2] += dz3 * self.cfg.step_size
        
        return self.z
        
    def reset(self):
        self.z = np.zeros(3)

class LADRC:
    """Linear Active Disturbance Rejection Controller"""
    def __init__(self, config: LADRCConfig):
        self.cfg = config
        self.td = TD(config)
        self.leso = LESO(config)
        
        self.u_last = 0.0
        self.u0 = 0.0

    def set_params(self, omega_c: float, b0: Optional[float] = None, omega_o: Optional[float] = None):
        """RL Adaptive Interface: Update parameters dynamically"""
        self.cfg.omega_c = omega_c
        self.cfg.omega_o = omega_o if omega_o is not None else 3.0 * omega_c
        if b0 is not None:
            self.cfg.b0 = b0

    def update(self, ref: float, y_meas: float) -> float:
        """
        Calculate control output
        ref: Target setpoint
        y_meas: Measured system output
        """
        # 1. Update Tracking Differentiator
        v1, v2 = self.td.update(ref)
        
        # 2. Update Observer
        z = self.leso.update(y_meas, self.u_last)
        z1, z2, z3 = z[0], z[1], z[2] # est_y, est_dy, est_dist
        
        # 3. Control Law (u0 = kp*e + kd*de)
        kp = self.cfg.omega_c**2
        kd = 2 * self.cfg.omega_c
        
        error = v1 - z1
        d_error = v2 - z2 
        
        self.u0 = kp * error + kd * d_error
        
        # 4. Disturbance Rejection
        u = (self.u0 - z3) / self.cfg.b0
        
        self.u_last = u
        return u

    def __call__(self, ref: float, y_meas: float) -> float:
        return self.update(ref, y_meas)

    def reset(self):
        self.td.reset()
        self.leso.reset()
        self.u_last = 0.0
        self.u0 = 0.0

    # Read-only properties for backward compatibility with old logging scripts
    @property
    def v1(self): return self.td.v1
    
    @property
    def v2(self): return self.td.v2

    @property
    def x1(self): return self.leso.z[0]
    
    @property
    def x2(self): return self.leso.z[1]
    
    @property
    def x3(self): return self.leso.z[2]

    @property
    def u(self): return self.u_last

