from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from lc.control.controllers.ladrc import LADRCController
from lc.control.controllers.ladrc_channels import (
    AxiswiseLADRCParameterSet,
    clone_parameter_set,
    load_default_ladrc_parameter_set,
)
from lc.control.controllers.pid import PIDController


_GYM_ENV_ROOT = Path(__file__).resolve().parents[4] / "Gym_env"
if str(_GYM_ENV_ROOT) not in sys.path:
    sys.path.append(str(_GYM_ENV_ROOT))

try:
    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
    from gym_pybullet_drones.utils.enums import DroneModel
    from Gym_env.LADRC_Controller import LADRC, LADRCConfig

    _HAS_NATIVE_CONTROLLERS = True
except Exception:
    DSLPIDControl = object  # type: ignore[assignment]
    DroneModel = None  # type: ignore[assignment]
    LADRC = None  # type: ignore[assignment]
    LADRCConfig = None  # type: ignore[assignment]
    _HAS_NATIVE_CONTROLLERS = False


def _zero_vector() -> np.ndarray:
    return np.zeros(3, dtype=np.float32)


def _default_quaternion() -> np.ndarray:
    return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


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


class _NativeSingleAxisHybridControl(DSLPIDControl):
    def __init__(
        self,
        drone_model: Any,
        parameter_set: AxiswiseLADRCParameterSet,
        ladrc_axes: tuple[str, ...],
        g: float = 9.8,
    ) -> None:
        self.con_X = None
        self.con_Y = None
        self.con_Z = None
        super().__init__(drone_model=drone_model, g=g)
        self.parameter_set = parameter_set
        self.ladrc_axes = tuple(ladrc_axes)
        self._build_ladrc_channels(self.CTRL_TIMESTEP if hasattr(self, "CTRL_TIMESTEP") else 1.0 / 60.0)

    def _build_ladrc_channels(self, step_size: float) -> None:
        self.con_X = self._make_channel(self.parameter_set.x, step_size)
        self.con_Y = self._make_channel(self.parameter_set.y, step_size)
        self.con_Z = self._make_channel(self.parameter_set.z, step_size)

    def _make_channel(self, axis_cfg: Any, step_size: float) -> Any:
        return LADRC(
            LADRCConfig(
                omega_c=axis_cfg.omega_c,
                b0=axis_cfg.b0,
                omega_o=axis_cfg.omega_c * axis_cfg.k,
                step_size=step_size,
                r=axis_cfg.r,
            )
        )

    def sync_from_parameter_set(self, step_size: float | None = None) -> None:
        if step_size is None:
            step_size = self.CTRL_TIMESTEP if hasattr(self, "CTRL_TIMESTEP") else 1.0 / 60.0
        self._build_ladrc_channels(step_size)

    def reset(self) -> None:
        super().reset()
        if self.con_X is not None:
            self.con_X.reset()
        if self.con_Y is not None:
            self.con_Y.reset()
        if self.con_Z is not None:
            self.con_Z.reset()

    def _dslPIDPositionControl(
        self,
        control_timestep: float,
        cur_pos: np.ndarray,
        cur_quat: np.ndarray,
        cur_vel: np.ndarray,
        target_pos: np.ndarray,
        target_rpy: np.ndarray,
        target_vel: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        self.integral_pos_e = self.integral_pos_e + (target_pos - cur_pos) * control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2.0, 2.0)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, 0.15)

        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        target_thrust = (
            np.multiply(self.P_COEFF_FOR, pos_e)
            + np.multiply(self.I_COEFF_FOR, self.integral_pos_e)
            + np.multiply(self.D_COEFF_FOR, vel_e)
            + np.array([0.0, 0.0, self.GRAVITY])
        )

        for index, axis_name in enumerate(("x", "y", "z")):
            if axis_name not in self.ladrc_axes:
                continue
            controller = getattr(self, f"con_{axis_name.upper()}")
            controller.cfg.step_size = control_timestep
            controller.td.cfg.step_size = control_timestep
            controller.leso.cfg.step_size = control_timestep
            ladrc_output = float(controller.update(float(target_pos[index]), float(cur_pos[index])))
            if axis_name == "z":
                # Keep the native PID hover/thrust baseline on altitude and let LADRC
                # act as an additive correction term. Replacing the whole z thrust
                # proved too aggressive for the PyBullet vertical channel.
                target_thrust[index] = target_thrust[index] + ladrc_output
            else:
                target_thrust[index] = ladrc_output

        scalar_thrust = max(0.0, np.dot(target_thrust, cur_rotation[:, 2]))
        thrust = (math.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE

        thrust_norm = float(np.linalg.norm(target_thrust))
        if thrust_norm < 1e-8 or not np.isfinite(thrust_norm):
            target_z_ax = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            target_z_ax = target_thrust / thrust_norm
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0.0])
        cross = np.cross(target_z_ax, target_x_c)
        if np.linalg.norm(cross) < 1e-8:
            target_y_ax = np.array([0.0, 1.0, 0.0])
        else:
            target_y_ax = cross / np.linalg.norm(cross)
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        if np.linalg.norm(target_x_ax) < 1e-8 or not np.all(np.isfinite(target_x_ax)):
            target_x_ax = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            target_x_ax = target_x_ax / np.linalg.norm(target_x_ax)
        if np.linalg.norm(target_y_ax) < 1e-8 or not np.all(np.isfinite(target_y_ax)):
            target_y_ax = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            target_y_ax = target_y_ax / np.linalg.norm(target_y_ax)
        target_rotation = np.vstack([target_x_ax, target_y_ax, target_z_ax]).transpose()
        target_euler = Rotation.from_matrix(target_rotation).as_euler("XYZ", degrees=False)
        return thrust, target_euler, pos_e


class _NativeBundle(ControllerBundle):
    def __init__(
        self,
        *,
        name: str,
        use_ladrc_position: bool,
        use_ladrc_attitude: bool,
        position_ladrc_axes: tuple[str, ...],
        parameter_set: AxiswiseLADRCParameterSet,
    ) -> None:
        super().__init__(
            name=name,
            use_ladrc_position=use_ladrc_position,
            use_ladrc_attitude=use_ladrc_attitude,
            position_ladrc_axes=position_ladrc_axes,
            parameter_set=parameter_set,
        )
        drone_model = DroneModel.CF2X
        if use_ladrc_attitude:
            from gym_pybullet_drones.control.LADRC import LADRCControl

            self.controller = LADRCControl(
                drone_model,
                9.8,
                [
                    parameter_set.x.omega_c,
                    parameter_set.x.b0,
                    parameter_set.x.omega_c * parameter_set.x.k,
                    1.0 / 60.0,
                    parameter_set.x.r,
                ],
                [
                    parameter_set.y.omega_c,
                    parameter_set.y.b0,
                    parameter_set.y.omega_c * parameter_set.y.k,
                    1.0 / 60.0,
                    parameter_set.y.r,
                ],
                [
                    parameter_set.z.omega_c,
                    parameter_set.z.b0,
                    parameter_set.z.omega_c * parameter_set.z.k,
                    1.0 / 60.0,
                    parameter_set.z.r,
                ],
                [
                    parameter_set.roll.omega_c,
                    parameter_set.roll.b0,
                    parameter_set.roll.omega_c * parameter_set.roll.k,
                    1.0 / 60.0,
                    parameter_set.roll.r,
                ],
                [
                    parameter_set.pitch.omega_c,
                    parameter_set.pitch.b0,
                    parameter_set.pitch.omega_c * parameter_set.pitch.k,
                    1.0 / 60.0,
                    parameter_set.pitch.r,
                ],
                [
                    parameter_set.yaw.omega_c,
                    parameter_set.yaw.b0,
                    parameter_set.yaw.omega_c * parameter_set.yaw.k,
                    1.0 / 60.0,
                    parameter_set.yaw.r,
                ],
            )
        elif use_ladrc_position:
            self.controller = _NativeSingleAxisHybridControl(
                drone_model=drone_model,
                parameter_set=parameter_set,
                ladrc_axes=position_ladrc_axes or ("x", "y", "z"),
                g=9.8,
            )
        else:
            self.controller = DSLPIDControl(drone_model=drone_model, g=9.8)
        self._sync_from_parameter_set()

    def reset(self) -> None:
        self.controller.reset()

    def compute_control_from_state(
        self,
        control_timestep: float,
        state: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_rpy: np.ndarray | None = None,
        target_rpy_rates: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        cur_pos = np.asarray(state[0:3], dtype=np.float64)
        cur_quat = np.asarray(state[3:7] if len(state) >= 7 else _default_quaternion(), dtype=np.float64)
        cur_vel = np.asarray(state[10:13] if len(state) >= 13 else np.zeros(3), dtype=np.float64)
        cur_ang_vel = np.asarray(state[13:16] if len(state) >= 16 else np.zeros(3), dtype=np.float64)
        target_rpy = np.asarray(target_rpy if target_rpy is not None else _zero_vector(), dtype=np.float64)
        target_rpy_rates = np.asarray(target_rpy_rates if target_rpy_rates is not None else _zero_vector(), dtype=np.float64)
        rpm, pos_e, yaw_e = self.controller.computeControl(
            control_timestep=control_timestep,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_vel,
            target_pos=np.asarray(target_pos, dtype=np.float64),
            target_rpy=target_rpy,
            target_vel=np.asarray(target_vel, dtype=np.float64),
            target_rpy_rates=target_rpy_rates,
        )
        return np.asarray(rpm, dtype=np.float32), np.asarray(pos_e, dtype=np.float32), float(yaw_e)

    def _sync_from_parameter_set(self) -> None:
        if isinstance(self.controller, _NativeSingleAxisHybridControl):
            self.controller.sync_from_parameter_set()


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
        self.ladrc_x.set_parameters(b0=self.parameter_set.x.b0, omega_c=self.parameter_set.x.omega_c, k=self.parameter_set.x.k)
        self.ladrc_y.set_parameters(b0=self.parameter_set.y.b0, omega_c=self.parameter_set.y.omega_c, k=self.parameter_set.y.k)
        self.ladrc_z.set_parameters(b0=self.parameter_set.z.b0, omega_c=self.parameter_set.z.omega_c, k=self.parameter_set.z.k)
        self.att_ladrc_roll.set_parameters(b0=self.parameter_set.roll.b0, omega_c=self.parameter_set.roll.omega_c, k=self.parameter_set.roll.k)
        self.att_ladrc_pitch.set_parameters(b0=self.parameter_set.pitch.b0, omega_c=self.parameter_set.pitch.omega_c, k=self.parameter_set.pitch.k)
        self.att_ladrc_yaw.set_parameters(b0=self.parameter_set.yaw.b0, omega_c=self.parameter_set.yaw.omega_c, k=self.parameter_set.yaw.k)


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
        default_factory=lambda: clone_parameter_set(load_default_ladrc_parameter_set("ladrc_x_pos_pid_att"))
    )


def _create_native_bundle(name: str, checkpoint: dict[str, Any] | None = None) -> ControllerBundle:
    if name == "pid_pos_att":
        bundle = _NativeBundle(
            name=name,
            use_ladrc_position=False,
            use_ladrc_attitude=False,
            position_ladrc_axes=(),
            parameter_set=clone_parameter_set(load_default_ladrc_parameter_set("pid_pos_att")),
        )
    elif name == "ladrc_pos_pid_att":
        bundle = _NativeBundle(
            name=name,
            use_ladrc_position=True,
            use_ladrc_attitude=False,
            position_ladrc_axes=("x", "y", "z"),
            parameter_set=clone_parameter_set(load_default_ladrc_parameter_set("ladrc_pos_pid_att")),
        )
    elif name == "ladrc_pos_att":
        bundle = _NativeBundle(
            name=name,
            use_ladrc_position=True,
            use_ladrc_attitude=True,
            position_ladrc_axes=("x", "y", "z"),
            parameter_set=clone_parameter_set(load_default_ladrc_parameter_set("ladrc_pos_att")),
        )
    elif name in {"ladrc_x_pos_pid_att", "ladrc_y_pos_pid_att", "ladrc_z_pos_pid_att"}:
        axis = name.split("_")[1]
        bundle = _NativeBundle(
            name=name,
            use_ladrc_position=True,
            use_ladrc_attitude=False,
            position_ladrc_axes=(axis,),
            parameter_set=clone_parameter_set(load_default_ladrc_parameter_set(name)),
        )
    else:
        raise KeyError(f"Unsupported controller variant: {name}")

    if checkpoint:
        snapshot = checkpoint.get("parameter_snapshot", {})
        for axis in ("x", "y", "z"):
            target = bundle.parameter_set.axis_config(axis)
            target.b0 = float(snapshot.get(f"{axis}_b0", target.b0))
            target.omega_c = float(snapshot.get(f"{axis}_omega_c", target.omega_c))
            target.k = float(snapshot.get(f"{axis}_k", target.k))
        if hasattr(bundle, "_sync_from_parameter_set"):
            bundle._sync_from_parameter_set()
    return bundle


def create_controller_bundle(name: str, checkpoint: dict[str, Any] | None = None) -> ControllerBundle:
    if _HAS_NATIVE_CONTROLLERS:
        return _create_native_bundle(name, checkpoint)

    if name == "pid_pos_att":
        bundle: ControllerBundle = PIDPositionAttitudeController()
    elif name == "ladrc_pos_pid_att":
        bundle = LADRCPositionPIDAttitudeController()
    elif name == "ladrc_pos_att":
        bundle = LADRCPositionAttitudeController()
    elif name == "ladrc_x_pos_pid_att":
        bundle = SingleAxisLADRCPositionPIDAttitudeController(
            name=name,
            position_ladrc_axes=("x",),
            parameter_set=clone_parameter_set(load_default_ladrc_parameter_set(name)),
        )
    elif name == "ladrc_y_pos_pid_att":
        bundle = SingleAxisLADRCPositionPIDAttitudeController(
            name=name,
            position_ladrc_axes=("y",),
            parameter_set=clone_parameter_set(load_default_ladrc_parameter_set(name)),
        )
    elif name == "ladrc_z_pos_pid_att":
        bundle = SingleAxisLADRCPositionPIDAttitudeController(
            name=name,
            position_ladrc_axes=("z",),
            parameter_set=clone_parameter_set(load_default_ladrc_parameter_set(name)),
        )
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
