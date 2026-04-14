from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lc.control.configs import LADRCActionBounds, get_axis_ladrc_action_bounds

from .ladrc import LADRCController


@dataclass
class AdaptiveLADRCController:
    """Adaptive LADRC wrapper driven by RL outputs."""

    base: LADRCController = field(default_factory=LADRCController)
    r_bounds: tuple[float, float] = (64.0, 66.0)
    b0_bounds: tuple[float, float] = (28.0, 32.0)
    omega_c_bounds: tuple[float, float] = (2.0, 2.5)
    k_bounds: tuple[float, float] = (6.0, 8.0)
    delta_r_bounds: tuple[float, float] = (-1.0, 1.0)
    delta_b0_bounds: tuple[float, float] = (-2.0, 2.0)
    delta_omega_c_bounds: tuple[float, float] = (-0.25, 0.25)
    delta_k_bounds: tuple[float, float] = (-1.0, 1.0)
    anchor_r: float = 65.0
    anchor_b0: float = 30.0
    anchor_omega_c: float = 2.25
    anchor_k: float = 7.0

    @classmethod
    def from_action_bounds(cls, bounds: LADRCActionBounds) -> "AdaptiveLADRCController":
        controller = cls(
            base=LADRCController(
                r=float(bounds.train_anchor.r),
                b0=float(bounds.train_anchor.b0),
                omega_c=float(bounds.train_anchor.wc),
                k=float(bounds.train_anchor.k),
            ),
            r_bounds=tuple(float(value) for value in bounds.r),
            b0_bounds=tuple(float(value) for value in bounds.b0),
            omega_c_bounds=tuple(float(value) for value in bounds.wc),
            k_bounds=tuple(float(value) for value in bounds.k),
            delta_r_bounds=tuple(float(value) for value in bounds.delta_r),
            delta_b0_bounds=tuple(float(value) for value in bounds.delta_b0),
            delta_omega_c_bounds=tuple(float(value) for value in bounds.delta_wc),
            delta_k_bounds=tuple(float(value) for value in bounds.delta_k),
            anchor_r=float(bounds.train_anchor.r),
            anchor_b0=float(bounds.train_anchor.b0),
            anchor_omega_c=float(bounds.train_anchor.wc),
            anchor_k=float(bounds.train_anchor.k),
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

    @property
    def r(self) -> float:
        return float(self.base.r)

    def adapt(self, action: np.ndarray) -> None:
        """Map normalized RL outputs to anchor-centered deltas, then clip to training bounds."""
        clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        r = np.clip(
            self.anchor_r + _map_from_unit_interval(clipped[0], self.delta_r_bounds),
            self.r_bounds[0],
            self.r_bounds[1],
        )
        b0 = np.clip(
            self.anchor_b0 + _map_from_unit_interval(clipped[1], self.delta_b0_bounds),
            self.b0_bounds[0],
            self.b0_bounds[1],
        )
        omega_c = np.clip(
            self.anchor_omega_c + _map_from_unit_interval(clipped[2], self.delta_omega_c_bounds),
            self.omega_c_bounds[0],
            self.omega_c_bounds[1],
        )
        k = np.clip(
            self.anchor_k + _map_from_unit_interval(clipped[3], self.delta_k_bounds),
            self.k_bounds[0],
            self.k_bounds[1],
        )
        self.base.set_parameters(r=float(r), b0=float(b0), omega_c=float(omega_c), k=float(k))

    def step(self, reference: float, measurement: float, dt: float) -> float:
        return self.base.step(reference, measurement, dt)


def _map_from_unit_interval(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    scaled = (float(value) + 1.0) * 0.5
    return low + scaled * (high - low)
