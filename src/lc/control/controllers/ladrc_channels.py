from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy


@dataclass
class SingleChannelLADRCConfig:
    b0: float
    omega_c: float
    k: float
    r: float = 50.0

    @property
    def omega_o(self) -> float:
        return float(self.k * self.omega_c)

    @omega_o.setter
    def omega_o(self, value: float) -> None:
        self.k = float(value / max(self.omega_c, 1e-6))


@dataclass
class AxiswiseLADRCParameterSet:
    x: SingleChannelLADRCConfig
    y: SingleChannelLADRCConfig
    z: SingleChannelLADRCConfig
    roll: SingleChannelLADRCConfig
    pitch: SingleChannelLADRCConfig
    yaw: SingleChannelLADRCConfig

    def axis_config(self, axis: str) -> SingleChannelLADRCConfig:
        return getattr(self, axis)


def load_default_ladrc_parameter_set(variant: str) -> AxiswiseLADRCParameterSet:
    attitude = SingleChannelLADRCConfig(b0=250.0, omega_c=6.0, k=30.0 / 6.0, r=50.0)
    base = AxiswiseLADRCParameterSet(
        x=SingleChannelLADRCConfig(b0=30.5, omega_c=1.5, k=11.0, r=10.0),
        y=SingleChannelLADRCConfig(b0=30.5, omega_c=1.5, k=11.0, r=10.0),
        z=SingleChannelLADRCConfig(b0=300.0, omega_c=15.0, k=110.0 / 15.0, r=30.0),
        roll=SingleChannelLADRCConfig(b0=attitude.b0, omega_c=attitude.omega_c, k=attitude.k, r=attitude.r),
        pitch=SingleChannelLADRCConfig(b0=attitude.b0, omega_c=attitude.omega_c, k=attitude.k, r=attitude.r),
        yaw=SingleChannelLADRCConfig(b0=150.0, omega_c=1.2, k=6.0 / 1.2, r=50.0),
    )
    if variant in {
        "pid_pos_att",
        "ladrc_pos_pid_att",
        "ladrc_pos_att",
        "ladrc_x_pos_pid_att",
        "ladrc_y_pos_pid_att",
        "ladrc_z_pos_pid_att",
    }:
        return base
    raise KeyError(f"Unsupported LADRC parameter set variant: {variant}")


def clone_parameter_set(parameter_set: AxiswiseLADRCParameterSet) -> AxiswiseLADRCParameterSet:
    return deepcopy(parameter_set)


def apply_parameter_deltas(
    parameter_set: AxiswiseLADRCParameterSet,
    axis: str,
    action: tuple[float, float, float] | list[float],
) -> AxiswiseLADRCParameterSet:
    target = parameter_set.axis_config(axis)
    target.b0 = float(max(0.2, min(4.0, target.b0 + float(action[0]))))
    target.omega_c = float(max(0.5, min(12.0, target.omega_c + float(action[1]))))
    target.k = float(max(2.0, min(6.0, target.k + float(action[2]))))
    return parameter_set
