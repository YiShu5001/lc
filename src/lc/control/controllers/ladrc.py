from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LADRCController:
    """Lightweight chapter-3 LADRC position controller with `r / b0 / wc / k` semantics."""

    r: float = 10.0
    b0: float = 1.0
    omega_c: float = 6.0
    k: float = 3.5
    derivative_gain_ratio: float = 0.35
    disturbance_estimate: float = 0.0
    last_error: float = 0.0

    @property
    def omega_o(self) -> float:
        return float(self.k * self.omega_c)

    @omega_o.setter
    def omega_o(self, value: float) -> None:
        self.k = float(value) / max(self.omega_c, 1e-6)

    def set_parameters(
        self,
        *,
        r: float | None = None,
        b0: float | None = None,
        omega_c: float | None = None,
        k: float | None = None,
    ) -> None:
        if r is not None:
            self.r = float(r)
        if b0 is not None:
            self.b0 = float(b0)
        if omega_c is not None:
            self.omega_c = float(omega_c)
        if k is not None:
            self.k = float(k)

    def snapshot(self) -> dict[str, float]:
        return {
            "r": float(self.r),
            "b0": float(self.b0),
            "omega_c": float(self.omega_c),
            "k": float(self.k),
            "omega_o": float(self.omega_o),
        }

    def reset(self) -> None:
        self.disturbance_estimate = 0.0
        self.last_error = 0.0

    def step(self, reference: float, measurement: float, dt: float) -> float:
        """Execute one LADRC position-control update."""
        error = reference - measurement
        error_rate = (error - self.last_error) / max(dt, 1e-6)
        self.last_error = error
        observer_gain = self.omega_o
        self.disturbance_estimate += dt * observer_gain * (error - self.disturbance_estimate)
        proportional = (self.omega_c**2) * error
        derivative = 2.0 * self.derivative_gain_ratio * self.omega_c * error_rate
        nominal_control = proportional + derivative
        return (nominal_control - self.disturbance_estimate) / max(self.b0, 1e-6)
