from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lc.control.configs import ArtifactConfig, AxisTrainingConfig, PyBulletControlExperimentConfig
from lc.control.controllers import create_controller_bundle
from lc.control.envs import compute_episode_metrics, run_controller_episode
from lc.control.io import write_metrics_csv, write_reference_csv, write_summary_json, write_timeseries_csv
from lc.control.reference_generators import build_xyz_reference_trajectory, summarize_reference_segments
from lc.control.simulators import close_ctrl_aviary, create_ctrl_aviary
from lc.control.simulators.pybullet_runner import (
    _build_axis_observation,
    _ensure_real_env,
    _timeseries_row,
)

CONTROL_FREQ_HZ = 48
FORWARD_STEPS = 96
HOVER_STEPS = 48
REVERSE_STEPS = 96
FINAL_HOLD_STEPS = 24
STEP_COUNT = FORWARD_STEPS + HOVER_STEPS + REVERSE_STEPS + FINAL_HOLD_STEPS
DURATION_SEC = STEP_COUNT / CONTROL_FREQ_HZ


@dataclass(frozen=True)
class LADRCParams:
    label: str
    r: float
    b0: float
    omega_c: float
    k: float


LADRC_0P5_OPT = LADRCParams(
    label="LADRC(0.5-opt)",
    r=63.0,
    b0=24.3,
    omega_c=2.95,
    k=7.415254237288136,
)

LADRC_0P6_OPT = LADRCParams(
    label="LADRC(0.6-opt)",
    r=59.5,
    b0=27.45,
    omega_c=3.15,
    k=3.08,
)


def _timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_config(output_root: Path) -> PyBulletControlExperimentConfig:
    return PyBulletControlExperimentConfig(
        control_freq_hz=CONTROL_FREQ_HZ,
        duration_sec=DURATION_SEC,
        seed=7,
        train_episodes=1,
        eval_episodes=1,
        compare_episodes=1,
        snapshot_interval=0,
        training_controller_variant="ladrc_x_pos_pid_att",
        artifact=ArtifactConfig(
            output_root=str(output_root),
            export_structured=True,
            export_legacy_logger=True,
            save_figures=True,
            record_video=False,
        ),
        axis_configs=(
            AxisTrainingConfig(
                axis="x",
                initial_position=(0.0, 0.0, 1.0),
                fixed_axes=(0.0, 1.0),
                include_disturbance=False,
                disturbance_scale=0.0,
                disturbance_axis_bias=1.0,
                fixed_stage_lengths=(FORWARD_STEPS, HOVER_STEPS, REVERSE_STEPS, FINAL_HOLD_STEPS),
                fixed_stage_velocities=(0.5, 0.0, -0.6, 0.0),
            ),
        ),
    )


def _apply_x_params(controller, params: LADRCParams) -> None:
    controller.set_axis_parameters(
        "x",
        r=params.r,
        b0=params.b0,
        omega_c=params.omega_c,
        k=params.k,
    )


def _stage_index(step: int) -> int:
    if step < FORWARD_STEPS:
        return 0
    if step < FORWARD_STEPS + HOVER_STEPS:
        return 1
    if step < FORWARD_STEPS + HOVER_STEPS + REVERSE_STEPS:
        return 2
    return 3


def _run_switched_ladrc_episode(
    config: PyBulletControlExperimentConfig,
    reference_bundle,
    forward_params: LADRCParams,
    reverse_params: LADRCParams,
) -> dict[str, object]:
    env = create_ctrl_aviary(config)
    controller = create_controller_bundle("ladrc_x_pos_pid_att")
    try:
        real_env = _ensure_real_env(env, config, reference_bundle)
        obs, _ = real_env.reset(seed=config.seed)
        state = np.asarray(obs[0], dtype=np.float32)
        controller.reset()
        timeseries: list[dict[str, float]] = []
        axis = "x"
        prev_rpm = np.full(4, 4300.0, dtype=np.float32)
        for step in range(config.step_count):
            stage = _stage_index(step)
            if stage <= 1:
                active = forward_params
            else:
                active = reverse_params
            _apply_x_params(controller, active)
            checkpoint = controller.snapshot_params()
            target_pos = reference_bundle.positions[step]
            target_vel = reference_bundle.velocities[step]
            rpm, _, _ = controller.compute_control_from_state(
                control_timestep=config.control_dt,
                state=state,
                target_pos=target_pos,
                target_vel=target_vel,
                target_rpy=np.zeros(3, dtype=np.float32),
                target_rpy_rates=np.zeros(3, dtype=np.float32),
            )
            next_obs, _, terminated, truncated, _ = real_env.step(rpm.reshape(1, 4))
            next_state = np.asarray(next_obs[0], dtype=np.float32)
            pos_error = float(target_pos[0] - next_state[0])
            vel_error = float(target_vel[0] - next_state[10])
            rpm_delta = float(np.mean(np.abs(rpm - prev_rpm)))
            reward = float(-abs(pos_error) - 0.15 * abs(vel_error) - 0.0008 * rpm_delta)
            row = _timeseries_row(
                step,
                config.control_dt,
                axis,
                state,
                target_pos,
                target_vel,
                rpm,
                reward,
                checkpoint,
                env["backend"],
                disturbance=np.zeros(3, dtype=np.float32),
            )
            row["stage_index"] = float(stage)
            row["active_param_set"] = 0.0 if stage <= 1 else 1.0
            row["switched_r"] = float(active.r)
            row["switched_b0"] = float(active.b0)
            row["switched_omega_c"] = float(active.omega_c)
            row["switched_k"] = float(active.k)
            timeseries.append(row)
            state = next_state
            prev_rpm = rpm
            if terminated or truncated:
                break
        metrics = compute_episode_metrics(timeseries, axis)
        metrics["backend"] = env["backend"]
        return {
            "timeseries": timeseries,
            "metrics": metrics,
            "backend": env["backend"],
        }
    finally:
        close_ctrl_aviary(env)


def _plot_tracking(rows_by_label: dict[str, list[dict[str, float]]], output_path: Path) -> Path:
    plt.figure(figsize=(10.8, 5.8))
    first_rows = next(iter(rows_by_label.values()))
    t = [row["time"] for row in first_rows]
    plt.plot(t, [row["target_x"] for row in first_rows], label="Reference", linewidth=2.6, color="black", linestyle="--")
    colors = {
        "PID": "#1f77b4",
        LADRC_0P5_OPT.label: "#d62728",
        LADRC_0P6_OPT.label: "#2ca02c",
        "LADRC(switched)": "#ff7f0e",
    }
    for label, rows in rows_by_label.items():
        plt.plot(t, [row["x"] for row in rows], label=label, linewidth=2.0, color=colors.get(label))
    plt.xlabel("Time (s)")
    plt.ylabel("X Position (m)")
    plt.title("PyBullet X-Axis Tracking Without Disturbance")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _plot_velocity(rows_by_label: dict[str, list[dict[str, float]]], output_path: Path) -> Path:
    plt.figure(figsize=(10.8, 5.8))
    first_rows = next(iter(rows_by_label.values()))
    t = [row["time"] for row in first_rows]
    plt.plot(t, [row["target_vx"] for row in first_rows], label="Reference", linewidth=2.6, color="black", linestyle="--")
    colors = {
        "PID": "#1f77b4",
        LADRC_0P5_OPT.label: "#d62728",
        LADRC_0P6_OPT.label: "#2ca02c",
        "LADRC(switched)": "#ff7f0e",
    }
    for label, rows in rows_by_label.items():
        plt.plot(t, [row["vx"] for row in rows], label=label, linewidth=2.0, color=colors.get(label))
    plt.xlabel("Time (s)")
    plt.ylabel("X Velocity (m/s)")
    plt.title("PyBullet X-Axis Velocity Tracking Without Disturbance")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _plot_error(rows_by_label: dict[str, list[dict[str, float]]], output_path: Path) -> Path:
    plt.figure(figsize=(10.8, 5.8))
    first_rows = next(iter(rows_by_label.values()))
    t = [row["time"] for row in first_rows]
    colors = {
        "PID": "#1f77b4",
        LADRC_0P5_OPT.label: "#d62728",
        LADRC_0P6_OPT.label: "#2ca02c",
        "LADRC(switched)": "#ff7f0e",
    }
    for label, rows in rows_by_label.items():
        plt.plot(t, [row["target_x"] - row["x"] for row in rows], label=label, linewidth=2.0, color=colors.get(label))
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Tracking Error (m)")
    plt.title("PyBullet X-Axis Tracking Error Without Disturbance")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def main() -> None:
    tag = _timestamp_tag()
    output_root = PROJECT_ROOT / "outputs" / "control_pybullet" / "x_pid_vs_ladrc_no_disturbance_compare" / "x" / tag
    config = _build_config(output_root)
    reference_bundle = build_xyz_reference_trajectory(config.axis_config("x"), config, rng=np.random.default_rng(config.seed))

    pid_controller = create_controller_bundle("pid_pos_att")
    pid_result = run_controller_episode(config, pid_controller, reference_bundle)

    ladrc_0p5_controller = create_controller_bundle("ladrc_x_pos_pid_att")
    _apply_x_params(ladrc_0p5_controller, LADRC_0P5_OPT)
    ladrc_0p5_result = run_controller_episode(config, ladrc_0p5_controller, reference_bundle)

    ladrc_0p6_controller = create_controller_bundle("ladrc_x_pos_pid_att")
    _apply_x_params(ladrc_0p6_controller, LADRC_0P6_OPT)
    ladrc_0p6_result = run_controller_episode(config, ladrc_0p6_controller, reference_bundle)

    ladrc_switched_result = _run_switched_ladrc_episode(config, reference_bundle, LADRC_0P5_OPT, LADRC_0P6_OPT)

    rows_by_label = {
        "PID": list(pid_result["timeseries"]),
        LADRC_0P5_OPT.label: list(ladrc_0p5_result["timeseries"]),
        LADRC_0P6_OPT.label: list(ladrc_0p6_result["timeseries"]),
        "LADRC(switched)": list(ladrc_switched_result["timeseries"]),
    }

    figures_dir = output_root / "figures"
    tracking_figure = _plot_tracking(rows_by_label, figures_dir / "tracking_compare.png")
    velocity_figure = _plot_velocity(rows_by_label, figures_dir / "velocity_compare.png")
    error_figure = _plot_error(rows_by_label, figures_dir / "error_compare.png")

    write_reference_csv(output_root / "reference.csv", reference_bundle)
    write_timeseries_csv(output_root / "pid_timeseries.csv", rows_by_label["PID"])
    write_timeseries_csv(output_root / "ladrc_0p5_opt_timeseries.csv", rows_by_label[LADRC_0P5_OPT.label])
    write_timeseries_csv(output_root / "ladrc_0p6_opt_timeseries.csv", rows_by_label[LADRC_0P6_OPT.label])
    write_timeseries_csv(output_root / "ladrc_switched_timeseries.csv", rows_by_label["LADRC(switched)"])

    metric_rows = [
        {"controller": "PID", **dict(pid_result["metrics"])},
        {"controller": LADRC_0P5_OPT.label, **dict(ladrc_0p5_result["metrics"])},
        {"controller": LADRC_0P6_OPT.label, **dict(ladrc_0p6_result["metrics"])},
        {"controller": "LADRC(switched)", **dict(ladrc_switched_result["metrics"])},
    ]
    write_metrics_csv(output_root / "metrics.csv", metric_rows)
    write_summary_json(
        output_root / "summary.json",
        {
            "backend": "gym_env",
            "disturbance": "none",
            "include_disturbance": False,
            "reference_segments": summarize_reference_segments(reference_bundle),
            "scenario_definition": {
                "control_freq_hz": CONTROL_FREQ_HZ,
                "duration_sec": DURATION_SEC,
                "fixed_stage_lengths": [FORWARD_STEPS, HOVER_STEPS, REVERSE_STEPS, FINAL_HOLD_STEPS],
                "fixed_stage_velocities": [0.5, 0.0, -0.6, 0.0],
            },
            "controllers": {
                "PID": {"source": "pid_pos_att"},
                LADRC_0P5_OPT.label: {
                    "source": "previous_0.5_mps_r_scan_best",
                    "r": LADRC_0P5_OPT.r,
                    "b0": LADRC_0P5_OPT.b0,
                    "omega_c": LADRC_0P5_OPT.omega_c,
                    "k": LADRC_0P5_OPT.k,
                },
                LADRC_0P6_OPT.label: {
                    "source": "previous_0.6_mps_multispeed_best",
                    "r": LADRC_0P6_OPT.r,
                    "b0": LADRC_0P6_OPT.b0,
                    "omega_c": LADRC_0P6_OPT.omega_c,
                    "k": LADRC_0P6_OPT.k,
                },
                "LADRC(switched)": {
                    "switch_rule": {
                        "stage_0_forward_0p5": LADRC_0P5_OPT.label,
                        "stage_1_hover": LADRC_0P5_OPT.label,
                        "stage_2_reverse_minus_0p6": LADRC_0P6_OPT.label,
                        "stage_3_final_hold": LADRC_0P6_OPT.label,
                    }
                },
            },
            "metrics": metric_rows,
            "figures": [str(tracking_figure), str(velocity_figure), str(error_figure)],
        },
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "metrics": metric_rows,
                "figures": [str(tracking_figure), str(velocity_figure), str(error_figure)],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
