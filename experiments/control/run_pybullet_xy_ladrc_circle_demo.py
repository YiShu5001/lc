from __future__ import annotations

import argparse
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
from lc.control.io import write_reference_csv, write_summary_json, write_timeseries_csv
from lc.control.reference_generators.piecewise_velocity import ReferenceBundle
from lc.control.simulators import close_ctrl_aviary, create_ctrl_aviary, run_evaluation_episode


CONTROL_FREQ_HZ = 48
DURATION_SEC = 8.0
STEP_COUNT = int(CONTROL_FREQ_HZ * DURATION_SEC)


@dataclass(frozen=True)
class XYMetrics:
    xy_rmse: float
    xy_mae: float
    radial_rmse: float


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
        training_controller_variant="ladrc_pos_pid_att",
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
                initial_position=(0.6, 0.0, 1.0),
                fixed_axes=(0.0, 1.0),
                include_disturbance=False,
                disturbance_scale=0.0,
            ),
        ),
    )


def _build_circle_reference(config: PyBulletControlExperimentConfig, radius: float) -> ReferenceBundle:
    times = np.arange(config.step_count, dtype=np.float32) * config.control_dt
    omega = 2.0 * np.pi / max(config.duration_sec, 1e-6)
    positions = np.zeros((config.step_count, 3), dtype=np.float32)
    velocities = np.zeros((config.step_count, 3), dtype=np.float32)

    positions[:, 0] = radius * np.cos(omega * times)
    positions[:, 1] = radius * np.sin(omega * times)
    positions[:, 2] = 1.0

    velocities[:, 0] = -radius * omega * np.sin(omega * times)
    velocities[:, 1] = radius * omega * np.cos(omega * times)
    velocities[:, 2] = 0.0

    return ReferenceBundle(
        axis="x",
        positions=positions,
        velocities=velocities,
        stage_slices=(slice(0, config.step_count),),
        stage_velocities=(0.0,),
    )


def _apply_xy_ladrc_params(controller) -> None:
    params = {"r": 63.0, "b0": 24.3, "omega_c": 2.95, "k": 7.415254237288136}
    controller.set_axis_parameters("x", **params)
    controller.set_axis_parameters("y", **params)


def _compute_xy_metrics(rows: list[dict[str, float]]) -> XYMetrics:
    x_err = np.asarray([float(row["target_x"]) - float(row["x"]) for row in rows], dtype=float)
    y_err = np.asarray([float(row["target_y"]) - float(row["y"]) for row in rows], dtype=float)
    xy_err = np.sqrt(x_err**2 + y_err**2)
    target_r = np.asarray(
        [np.sqrt(float(row["target_x"]) ** 2 + float(row["target_y"]) ** 2) for row in rows],
        dtype=float,
    )
    actual_r = np.asarray(
        [np.sqrt(float(row["x"]) ** 2 + float(row["y"]) ** 2) for row in rows],
        dtype=float,
    )
    radial_err = target_r - actual_r
    return XYMetrics(
        xy_rmse=float(np.sqrt(np.mean(xy_err**2))),
        xy_mae=float(np.mean(np.abs(xy_err))),
        radial_rmse=float(np.sqrt(np.mean(radial_err**2))),
    )


def _plot_xy_trajectory(rows_by_label: dict[str, list[dict[str, float]]], output_path: Path) -> Path:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8.4, 7.4))
    first_rows = next(iter(rows_by_label.values()))
    ax.plot(
        [row["target_x"] for row in first_rows],
        [row["target_y"] for row in first_rows],
        linestyle="--",
        color="#2f2f2f",
        linewidth=2.6,
        label="圆形参考轨迹",
    )
    color_map = {
        "PID": "#1f77b4",
        "固定LADRC": "#b22222",
    }
    for label, rows in rows_by_label.items():
        ax.plot(
            [row["x"] for row in rows],
            [row["y"] for row in rows],
            linewidth=2.2,
            color=color_map.get(label, "#444444"),
            label=label,
        )
    ax.scatter(first_rows[0]["target_x"], first_rows[0]["target_y"], color="#2ca02c", s=55, zorder=4)
    ax.annotate("起点", (first_rows[0]["target_x"], first_rows[0]["target_y"]), xytext=(8, 8), textcoords="offset points")
    ax.set_xlabel("X方向位置 / m")
    ax.set_ylabel("Y方向位置 / m")
    ax.set_title("PyBullet圆形参考轨迹下的二维平面跟踪图")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.28)
    ax.legend(frameon=True, framealpha=0.95)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real PyBullet circular XY-plane tracking demo.")
    parser.add_argument("--radius", type=float, default=0.6)
    parser.add_argument("--tag", default=_timestamp_tag())
    args = parser.parse_args()

    output_root = PROJECT_ROOT / "outputs" / "control_pybullet" / "xy_circle_tracking_demo" / args.tag
    config = _build_config(output_root)
    reference_bundle = _build_circle_reference(config, radius=float(args.radius))

    env = create_ctrl_aviary(config)
    try:
        pid_controller = create_controller_bundle("pid_pos_att")
        ladrc_controller = create_controller_bundle("ladrc_pos_pid_att")
        _apply_xy_ladrc_params(ladrc_controller)

        pid_artifacts = run_evaluation_episode(env, pid_controller, reference_bundle, axis="x", config=config)
        ladrc_artifacts = run_evaluation_episode(env, ladrc_controller, reference_bundle, axis="x", config=config)
    finally:
        close_ctrl_aviary(env)

    rows_by_label = {
        "PID": list(pid_artifacts.timeseries),
        "固定LADRC": list(ladrc_artifacts.timeseries),
    }

    figures_dir = output_root / "figures"
    trajectory_figure = _plot_xy_trajectory(rows_by_label, figures_dir / "xy_circle_tracking_real_data.png")

    write_reference_csv(output_root / "reference.csv", reference_bundle)
    write_timeseries_csv(output_root / "pid_timeseries.csv", rows_by_label["PID"])
    write_timeseries_csv(output_root / "ladrc_timeseries.csv", rows_by_label["固定LADRC"])

    summary = {
        "backend": pid_artifacts.backend,
        "radius": float(args.radius),
        "duration_sec": config.duration_sec,
        "control_freq_hz": config.control_freq_hz,
        "controllers": {
            "PID": {
                "xy_metrics": _compute_xy_metrics(rows_by_label["PID"]).__dict__,
            },
            "固定LADRC": {
                "params": {"r": 63.0, "b0": 24.3, "omega_c": 2.95, "k": 7.415254237288136},
                "xy_metrics": _compute_xy_metrics(rows_by_label["固定LADRC"]).__dict__,
            },
        },
        "figures": [str(trajectory_figure)],
    }
    write_summary_json(output_root / "summary.json", summary)
    print(json.dumps({"output_root": str(output_root), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
