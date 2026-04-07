from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lc.control.configs import LADRCActionBounds, get_axis_ladrc_action_bounds

from .ladrc import LADRCController


@dataclass
class AdaptiveLADRCController:
    """Adaptive LADRC wrapper driven by RL outputs."""

    base: LADRCController = field(default_factory=LADRCController)
    b0_bounds: tuple[float, float] = (0.3, 2.5)
    omega_c_bounds: tuple[float, float] = (2.0, 15.0)
    k_bounds: tuple[float, float] = (2.0, 6.0)

    @classmethod
    def from_action_bounds(cls, bounds: LADRCActionBounds) -> "AdaptiveLADRCController":
        controller = cls(
            base=LADRCController(
                b0=float(bounds.fast_anchor.b0),
                omega_c=float(bounds.fast_anchor.wc),
                k=float(bounds.fast_anchor.k),
            ),
            b0_bounds=tuple(float(value) for value in bounds.b0),
            omega_c_bounds=tuple(float(value) for value in bounds.wc),
            k_bounds=tuple(float(value) for value in bounds.k),
        )
        return controller

    @classmethod
    def for_axis(cls, axis: str) -> "AdaptiveLADRCController":
        return cls.from_action_bounds(get_axis_ladrc_action_bounds(axis))

    def reset(self) -> None:
        self.base.reset()

    @property
    def k(self) -> float:
        return float(self.base.k)

    def adapt(self, action: np.ndarray) -> None:
        """Map normalized RL outputs to absolute b0, wc, and k, then derive wo."""
        clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        b0 = _map_from_unit_interval(clipped[0], self.b0_bounds)
        omega_c = _map_from_unit_interval(clipped[1], self.omega_c_bounds)
        k = _map_from_unit_interval(clipped[2], self.k_bounds)
        self.base.set_parameters(b0=float(b0), omega_c=float(omega_c), k=float(k))

    def step(self, reference: float, measurement: float, dt: float) -> float:
        return self.base.step(reference, measurement, dt)


def _map_from_unit_interval(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    scaled = (float(value) + 1.0) * 0.5
    return low + scaled * (high - low)
