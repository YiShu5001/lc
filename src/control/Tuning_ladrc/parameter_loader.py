from __future__ import annotations

import json
from pathlib import Path

from control.Tuning_ladrc.schemas import AxisLADRCParameters
from lc.control.controllers import ControllerBundle, create_controller_bundle


def load_axis_parameter_file(path: str | Path) -> dict[str, AxisLADRCParameters]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    axis_params: dict[str, AxisLADRCParameters] = {}
    for axis in ("x", "y", "z"):
        values = payload[axis]
        axis_params[axis] = AxisLADRCParameters(
            axis=axis,
            b0=float(values["b0"]),
            wc=float(values["wc"]),
            k=float(values["k"]),
            r=float(values.get("r", 30.0)),
        )
    return axis_params


def build_single_axis_ladrc_bundle(
    axis: str,
    parameter_file: str | Path,
) -> ControllerBundle:
    params = load_axis_parameter_file(parameter_file)[axis]
    bundle = create_controller_bundle(f"ladrc_{axis}_pos_pid_att")
    bundle.set_axis_parameters(axis, b0=params.b0, omega_c=params.wc, k=params.k)
    bundle.parameter_set.axis_config(axis).r = float(params.r)
    if hasattr(bundle, "_sync_from_parameter_set"):
        bundle._sync_from_parameter_set()
    return bundle
