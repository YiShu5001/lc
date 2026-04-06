from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lc.control.controllers.ladrc import LADRCController
from lc.control.controllers.ladrc_channels import (
    AxiswiseLADRCParameterSet,
    clone_parameter_set,
    load_default_ladrc_parameter_set,
)
from lc.control.controllers.pid import PIDController


def _zero_vector() -> np.ndarray:
    return np.zeros(3, dtype=np.float32)


@dataclass
class ControllerBundle:
    name: str
    use_ladrc_position: bool
    use_ladrc_attitude: bool
    position_ladrc_axes: tuple[str, ...] = ()
    parameter_set: AxiswiseLADRCParameterSet = field(
        default_factory=lambda: clone_parameter_set(load_default_ladrc_parameter_set("ladrc_pos_pid_att"))
    )

    def reset(self) -> None:
        raise NotImplementedError

    def compute_control_from_state(
        self,
        control_timestep: float,
        state: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_rpy: np.ndarray | None = None,
        target_rpy_rates: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        raise NotImplementedError

    def uses_ladrc_on_axis(self, axis: str) -> bool:
        return self.use_ladrc_position and (not self.position_ladrc_axes or axis in self.position_ladrc_axes)

    def set_axis_parameters(self, axis: str, *, b0: float | None = None, omega_c: float | None = None, k: float | None = None) -> None:
        target = self.parameter_set.axis_config(axis)
        if b0 is not None:
            target.b0 = float(b0)
        if omega_c is not None:
            target.omega_c = float(omega_c)
        if k is not None:
            target.k = float(k)
        if hasattr(self, "_sync_from_parameter_set"):
            self._sync_from_parameter_set()

    def snapshot_params(self) -> dict[str, float]:
        return {
            "x_b0": self.parameter_set.x.b0,
            "x_omega_c": self.parameter_set.x.omega_c,
            "x_k": self.parameter_set.x.k,
            "y_b0": self.parameter_set.y.b0,
            "y_omega_c": self.parameter_set.y.omega_c,
            "y_k": self.parameter_set.y.k,
            "z_b0": self.parameter_set.z.b0,
            "z_omega_c": self.parameter_set.z.omega_c,
            "z_k": self.parameter_set.z.k,
        }


@dataclass
class _FallbackBundle(ControllerBundle):
    pid_x: PIDController = field(default_factory=lambda: PIDController(kp=0.4, ki=0.05, kd=0.2))
    pid_y: PIDController = field(default_factory=lambda: PIDController(kp=0.4, ki=0.05, kd=0.2))
    pid_z: PIDController = field(default_factory=lambda: PIDController(kp=1.25, ki=0.05, kd=0.5))
    ladrc_x: LADRCController = field(default_factory=LADRCController)
    ladrc_y: LADRCController = field(default_factory=LADRCController)
    ladrc_z: LADRCController = field(default_factory=lambda: LADRCController(omega_c=15.0, k=110.0 / 15.0, b0=300.0))
    att_pid_roll: PIDController = field(default_factory=lambda: PIDController(kp=7.0, ki=0.0, kd=2.0, integral_limit=1.0))
    att_pid_pitch: PIDController = field(default_factory=lambda: PIDController(kp=7.0, ki=0.0, kd=2.0, integral_limit=1.0))
    att_pid_yaw: PIDController = field(default_factory=lambda: PIDController(kp=6.0, ki=0.05, kd=1.2, integral_limit=1.0))
    att_ladrc_roll: LADRCController = field(default_factory=lambda: LADRCController(omega_c=6.0, k=5.0, b0=250.0))
    att_ladrc_pitch: LADRCController = field(default_factory=lambda: LADRCController(omega_c=6.0, k=5.0, b0=250.0))
    att_ladrc_yaw: LADRCController = field(default_factory=lambda: LADRCController(omega_c=1.2, k=5.0, b0=150.0))

    def __post_init__(self) -> None:
        self._sync_from_parameter_set()

    def reset(self) -> None:
        for controller in (
            self.pid_x,
            self.pid_y,
            self.pid_z,
            self.ladrc_x,
            self.ladrc_y,
            self.ladrc_z,
            self.att_pid_roll,
            self.att_pid_pitch,
            self.att_pid_yaw,
            self.att_ladrc_roll,
            self.att_ladrc_pitch,
            self.att_ladrc_yaw,
        ):
            controller.reset()

    def compute_control_from_state(
        self,
        control_timestep: float,
        state: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_rpy: np.ndarray | None = None,
        target_rpy_rates: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        current_pos = state[0:3]
        current_vel = state[10:13]
        current_rpy = state[7:10] if len(state) >= 10 else np.zeros(3, dtype=np.float32)
        target_rpy = target_rpy if target_rpy is not None else _zero_vector()
        pos_error = target_pos - current_pos
        vel_error = target_vel - current_vel

        position_signal = np.zeros(3, dtype=np.float32)
        for index, axis_name in enumerate(("x", "y", "z")):
            if self.uses_ladrc_on_axis(axis_name):
                controller = getattr(self, f"ladrc_{axis_name}")
                base_signal = controller.step(float(target_pos[index]), float(current_pos[index]), control_timestep)
                position_signal[index] = float(base_signal + 0.15 * vel_error[index])
            else:
                controller = getattr(self, f"pid_{axis_name}")
                base_signal = controller.step(float(target_pos[index]), float(current_pos[index]), control_timestep)
                position_signal[index] = float(base_signal + 0.2 * vel_error[index])

        desired_rpy = np.asarray(
            [
                float(np.clip(-0.18 * position_signal[1], -0.45, 0.45)),
                float(np.clip(0.18 * position_signal[0], -0.45, 0.45)),
                float(target_rpy[2]),
            ],
            dtype=np.float32,
        )
        attitude_error = desired_rpy - current_rpy
        if self.use_ladrc_attitude:
            roll_torque = self.att_ladrc_roll.step(float(desired_rpy[0]), float(current_rpy[0]), control_timestep)
            pitch_torque = self.att_ladrc_pitch.step(float(desired_rpy[1]), float(current_rpy[1]), control_timestep)
            yaw_torque = self.att_ladrc_yaw.step(float(desired_rpy[2]), float(current_rpy[2]), control_timestep)
        else:
            roll_torque = self.att_pid_roll.step(float(desired_rpy[0]), float(current_rpy[0]), control_timestep)
            pitch_torque = self.att_pid_pitch.step(float(desired_rpy[1]), float(current_rpy[1]), control_timestep)
            yaw_torque = self.att_pid_yaw.step(float(desired_rpy[2]), float(current_rpy[2]), control_timestep)

        rpm_base = 4300.0 + 280.0 * position_signal[2]
        rpm = np.asarray(
            [
                rpm_base - 120.0 * roll_torque - 120.0 * pitch_torque - 60.0 * yaw_torque,
                rpm_base - 120.0 * roll_torque + 120.0 * pitch_torque + 60.0 * yaw_torque,
                rpm_base + 120.0 * roll_torque + 120.0 * pitch_torque - 60.0 * yaw_torque,
                rpm_base + 120.0 * roll_torque - 120.0 * pitch_torque + 60.0 * yaw_torque,
            ],
            dtype=np.float32,
        )
        rpm = np.clip(rpm, 3600.0, 5600.0)
        return rpm, pos_error.astype(np.float32), float(attitude_error[2])

    def _sync_from_parameter_set(self) -> None:
        self.ladrc_x.set_parameters(
            b0=self.parameter_set.x.b0,
            omega_c=self.parameter_set.x.omega_c,
            k=self.parameter_set.x.k,
        )
        self.ladrc_y.set_parameters(
            b0=self.parameter_set.y.b0,
            omega_c=self.parameter_set.y.omega_c,
            k=self.parameter_set.y.k,
        )
        self.ladrc_z.set_parameters(
            b0=self.parameter_set.z.b0,
            omega_c=self.parameter_set.z.omega_c,
            k=self.parameter_set.z.k,
        )
        self.att_ladrc_roll.set_parameters(
            b0=self.parameter_set.roll.b0,
            omega_c=self.parameter_set.roll.omega_c,
            k=self.parameter_set.roll.k,
        )
        self.att_ladrc_pitch.set_parameters(
            b0=self.parameter_set.pitch.b0,
            omega_c=self.parameter_set.pitch.omega_c,
            k=self.parameter_set.pitch.k,
        )
        self.att_ladrc_yaw.set_parameters(
            b0=self.parameter_set.yaw.b0,
            omega_c=self.parameter_set.yaw.omega_c,
            k=self.parameter_set.yaw.k,
        )


@dataclass
class PIDPositionAttitudeController(_FallbackBundle):
    name: str = "pid_pos_att"
    use_ladrc_position: bool = False
    use_ladrc_attitude: bool = False
    position_ladrc_axes: tuple[str, ...] = ()
    parameter_set: AxiswiseLADRCParameterSet = field(
        default_factory=lambda: clone_parameter_set(load_default_ladrc_parameter_set("pid_pos_att"))
    )


@dataclass
class LADRCPositionPIDAttitudeController(_FallbackBundle):
    name: str = "ladrc_pos_pid_att"
    use_ladrc_position: bool = True
    use_ladrc_attitude: bool = False
    position_ladrc_axes: tuple[str, ...] = ()
    parameter_set: AxiswiseLADRCParameterSet = field(
        default_factory=lambda: clone_parameter_set(load_default_ladrc_parameter_set("ladrc_pos_pid_att"))
    )


@dataclass
class LADRCPositionAttitudeController(_FallbackBundle):
    name: str = "ladrc_pos_att"
    use_ladrc_position: bool = True
    use_ladrc_attitude: bool = True
    position_ladrc_axes: tuple[str, ...] = ()
    parameter_set: AxiswiseLADRCParameterSet = field(
        default_factory=lambda: clone_parameter_set(load_default_ladrc_parameter_set("ladrc_pos_att"))
    )


@dataclass
class SingleAxisLADRCPositionPIDAttitudeController(_FallbackBundle):
    name: str = "ladrc_x_pos_pid_att"
    use_ladrc_position: bool = True
    use_ladrc_attitude: bool = False
    position_ladrc_axes: tuple[str, ...] = ("x",)
    parameter_set: AxiswiseLADRCParameterSet = field(
        default_factory=lambda: clone_parameter_set(load_default_ladrc_parameter_set("ladrc_pos_pid_att"))
    )


def create_controller_bundle(name: str, checkpoint: dict[str, Any] | None = None) -> ControllerBundle:
    if name == "pid_pos_att":
        bundle: ControllerBundle = PIDPositionAttitudeController()
    elif name == "ladrc_pos_pid_att":
        bundle = LADRCPositionPIDAttitudeController()
    elif name == "ladrc_pos_att":
        bundle = LADRCPositionAttitudeController()
    elif name == "ladrc_x_pos_pid_att":
        bundle = SingleAxisLADRCPositionPIDAttitudeController(name=name, position_ladrc_axes=("x",))
    elif name == "ladrc_y_pos_pid_att":
        bundle = SingleAxisLADRCPositionPIDAttitudeController(name=name, position_ladrc_axes=("y",))
    elif name == "ladrc_z_pos_pid_att":
        bundle = SingleAxisLADRCPositionPIDAttitudeController(name=name, position_ladrc_axes=("z",))
    else:
        raise KeyError(f"Unsupported controller variant: {name}")
    if checkpoint:
        snapshot = checkpoint.get("parameter_snapshot", {})
        for axis in ("x", "y", "z"):
            target = bundle.parameter_set.axis_config(axis)
            target.b0 = float(snapshot.get(f"{axis}_b0", target.b0))
            target.omega_c = float(snapshot.get(f"{axis}_omega_c", target.omega_c))
            target.k = float(snapshot.get(f"{axis}_k", target.k))
    if isinstance(bundle, _FallbackBundle):
        bundle._sync_from_parameter_set()
    return bundle
