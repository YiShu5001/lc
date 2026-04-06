from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PIDController:
    kp: float = 1.2
    ki: float = 0.04
    kd: float = 0.2
    integral_limit: float = 2.0

    integral_error: float = 0.0
    previous_error: float = 0.0

    def reset(self) -> None:
        self.integral_error = 0.0
        self.previous_error = 0.0

    def step(self, reference: float, measurement: float, dt: float) -> float:
        error = reference - measurement
        self.integral_error = max(min(self.integral_error + error * dt, self.integral_limit), -self.integral_limit)
        derivative = (error - self.previous_error) / max(dt, 1e-6)
        self.previous_error = error
        return self.kp * error + self.ki * self.integral_error + self.kd * derivative

