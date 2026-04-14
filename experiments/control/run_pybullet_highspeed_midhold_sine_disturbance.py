from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lc.common.io import ensure_dir
from lc.control.configs import ArtifactConfig, AxisTrainingConfig, PyBulletControlExperimentConfig
from lc.control.controllers import create_controller_bundle
from lc.control.envs import run_controller_episode
from lc.control.io import write_metrics_csv, write_reference_csv, write_summary_json, write_timeseries_csv
from lc.control.reference_generators import build_xyz_reference_trajectory, summarize_reference_segments


def _build_config(output_root: Path) -> PyBulletControlExperimentConfig:
    control_freq_hz = 48
    start_hold = 5
    forward_steps = 120
    mid_hold = 139
    reverse_steps = 120
    breeze_force_n = 0.02
    x_axis = AxisTrainingConfig(
        axis="x",
        initial_position=(0.0, 0.0, 1.0),
        fixed_axes=(0.0, 1.0),
        primary_speed_range=(0.6, 0.6),
        reverse_speed_range=(-0.6, -0.6),
        stage_duration_range=(0.1, 0.1),
        include_disturbance=True,
        disturbance_scale=breeze_force_n,
        disturbance_axis_bias=1.0,
        stage_count=4,
        fixed_stage_lengths=(start_hold, forward_steps, mid_hold, reverse_steps),
        fixed_stage_velocities=(0.0, 0.6, 0.0, -0.6),
        disturbance_step_window=(start_hold + forward_steps, start_hold + forward_steps + mid_hold),
        disturbance_frequency_rad=(2.0 * np.pi) / 24.0,
    )
    return PyBulletControlExperimentConfig(
        gui=False,
        seed=7,
        duration_sec=8.0,
        eval_episodes=3,
        artifact=ArtifactConfig(output_root=str(output_root), record_video=True),
        axis_configs=(
            x_axis,
            AxisTrainingConfig(axis="y"),
            AxisTrainingConfig(axis="z", fixed_axes=(0.0, 0.0), initial_position=(0.0, 0.0, 1.0)),
        ),
    )


def _compute_stage_metrics(rows: list[dict[str, float]], stage_segments: list[dict[str, float]]) -> list[dict[str, float]]:
    metrics: list[dict[str, float]] = []
    for segment in stage_segments:
        start = int(segment["start"])
        stop = int(segment["stop"])
        chunk = rows[start:stop]
        errors = [float(row["target_x"]) - float(row["x"]) for row in chunk]
        if not errors:
            rmse = 0.0
            mae = 0.0
        else:
            rmse = float(np.sqrt(np.mean(np.square(errors))))
            mae = float(np.mean(np.abs(errors)))
        metrics.append(
            {
                "stage": int(segment["stage"]),
                "start": start,
                "stop": stop,
                "velocity": float(segment["velocity"]),
                "rmse": rmse,
                "mae": mae,
            }
        )
    return metrics


def _plot_all(
    output_root: Path,
    pid_rows: list[dict[str, float]],
    default_rows: list[dict[str, float]],
    tuned_rows: list[dict[str, float]],
    disturbance_window: tuple[int, int],
) -> list[Path]:
    figures_dir = ensure_dir(output_root / "figures")
    times = np.asarray([float(row["time"]) for row in pid_rows])
    reference = np.asarray([float(row["target_x"]) for row in pid_rows])
    pid_output = np.asarray([float(row["x"]) for row in pid_rows])
    default_output = np.asarray([float(row["x"]) for row in default_rows])
    tuned_output = np.asarray([float(row["x"]) for row in tuned_rows])

    plt.rcParams.update(
        {
            "font.family": "Microsoft YaHei",
            "axes.unicode_minus": False,
            "figure.figsize": (10.5, 5.8),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    out_paths: list[Path] = []

    fig, ax = plt.subplots()
    ax.plot(times, reference, color="#222222", linewidth=2.4, label="Reference")
    ax.plot(times, pid_output, color="#d62728", linewidth=1.9, label="PID")
    ax.plot(times, default_output, color="#1f77b4", linewidth=1.9, label="LADRC(default)")
    ax.plot(times, tuned_output, color="#2ca02c", linewidth=2.0, label="LADRC(tuned)")
    ax.axvspan(times[disturbance_window[0]], times[min(disturbance_window[1] - 1, len(times) - 1)], color="#f7e1a0", alpha=0.25)
    ax.set_title("PyBullet High-Speed X Tracking With Mid-Hold Sine Disturbance")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    p1 = figures_dir / "tracking_three_way.svg"
    p2 = figures_dir / "tracking_three_way.png"
    fig.savefig(p1, bbox_inches="tight")
    fig.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    out_paths.extend([p1, p2])

    fig, ax = plt.subplots()
    ax.plot(times, reference - pid_output, color="#d62728", linewidth=1.8, label="PID error")
    ax.plot(times, reference - default_output, color="#1f77b4", linewidth=1.8, label="LADRC(default) error")
    ax.plot(times, reference - tuned_output, color="#2ca02c", linewidth=2.0, label="LADRC(tuned) error")
    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.axvspan(times[disturbance_window[0]], times[min(disturbance_window[1] - 1, len(times) - 1)], color="#f7e1a0", alpha=0.25)
    ax.set_title("Tracking Error Under Mid-Hold Disturbance")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.legend(frameon=False)
    fig.tight_layout()
    p3 = figures_dir / "tracking_error_three_way.svg"
    p4 = figures_dir / "tracking_error_three_way.png"
    fig.savefig(p3, bbox_inches="tight")
    fig.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close(fig)
    out_paths.extend([p3, p4])

    start_idx = max(disturbance_window[0] - 12, 0)
    stop_idx = min(disturbance_window[1] + 12, len(times))
    fig, ax = plt.subplots()
    ax.plot(times[start_idx:stop_idx], reference[start_idx:stop_idx], color="#222222", linewidth=2.4, label="Reference")
    ax.plot(times[start_idx:stop_idx], pid_output[start_idx:stop_idx], color="#d62728", linewidth=1.9, label="PID")
    ax.plot(times[start_idx:stop_idx], default_output[start_idx:stop_idx], color="#1f77b4", linewidth=1.9, label="LADRC(default)")
    ax.plot(times[start_idx:stop_idx], tuned_output[start_idx:stop_idx], color="#2ca02c", linewidth=2.0, label="LADRC(tuned)")
    ax.axvspan(times[disturbance_window[0]], times[min(disturbance_window[1] - 1, len(times) - 1)], color="#f7e1a0", alpha=0.25)
    ax.set_title("Zoomed Mid-Hold Disturbance Response")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    p5 = figures_dir / "disturbance_zoom_three_way.svg"
    p6 = figures_dir / "disturbance_zoom_three_way.png"
    fig.savefig(p5, bbox_inches="tight")
    fig.savefig(p6, dpi=300, bbox_inches="tight")
    plt.close(fig)
    out_paths.extend([p5, p6])

    return out_paths


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = ensure_dir(
        Path("outputs") / "control_pybullet" / "x_highspeed_midhold_sine_disturbance" / "x" / stamp
    )
    cfg = _build_config(output_root)
    reference_bundle = build_xyz_reference_trajectory(cfg.axis_config("x"), cfg, rng=np.random.default_rng(cfg.seed))
    segments = summarize_reference_segments(reference_bundle)
    disturbance_window = cfg.axis_config("x").disturbance_step_window or (0, 0)

    pid_controller = create_controller_bundle("pid_pos_att")
    default_ladrc_controller = create_controller_bundle("ladrc_x_pos_pid_att")
    tuned_ladrc_controller = create_controller_bundle("ladrc_x_pos_pid_att")
    tuned_ladrc_controller.set_axis_parameters("x", b0=0.8, omega_c=6.6, k=3.2)

    pid_cfg = replace(cfg, artifact=replace(cfg.artifact, video_output_dir=str(output_root / "videos" / "pid")))
    default_cfg = replace(cfg, artifact=replace(cfg.artifact, video_output_dir=str(output_root / "videos" / "ladrc_default")))
    tuned_cfg = replace(cfg, artifact=replace(cfg.artifact, video_output_dir=str(output_root / "videos" / "ladrc_tuned")))

    pid_result = run_controller_episode(pid_cfg, pid_controller, reference_bundle)
    default_result = run_controller_episode(default_cfg, default_ladrc_controller, reference_bundle)
    tuned_result = run_controller_episode(tuned_cfg, tuned_ladrc_controller, reference_bundle)

    write_reference_csv(output_root / "reference.csv", reference_bundle)
    write_timeseries_csv(output_root / "pid_timeseries.csv", list(pid_result["timeseries"]))
    write_timeseries_csv(output_root / "default_ladrc_timeseries.csv", list(default_result["timeseries"]))
    write_timeseries_csv(output_root / "tuned_ladrc_timeseries.csv", list(tuned_result["timeseries"]))
    write_metrics_csv(
        output_root / "metrics.csv",
        [
            {"controller": "pid_pos_att", **pid_result["metrics"]},
            {"controller": "ladrc_x_pos_pid_att_default", **default_result["metrics"]},
            {"controller": "ladrc_x_pos_pid_att_tuned", **tuned_result["metrics"]},
        ],
    )

    pid_stage_metrics = _compute_stage_metrics(list(pid_result["timeseries"]), segments)
    default_stage_metrics = _compute_stage_metrics(list(default_result["timeseries"]), segments)
    tuned_stage_metrics = _compute_stage_metrics(list(tuned_result["timeseries"]), segments)
    write_metrics_csv(
        output_root / "stage_metrics.csv",
        [
            {"controller": "pid_pos_att", **row} for row in pid_stage_metrics
        ]
        + [{"controller": "ladrc_x_pos_pid_att_default", **row} for row in default_stage_metrics]
        + [{"controller": "ladrc_x_pos_pid_att_tuned", **row} for row in tuned_stage_metrics],
    )

    figure_paths = _plot_all(
        output_root,
        list(pid_result["timeseries"]),
        list(default_result["timeseries"]),
        list(tuned_result["timeseries"]),
        disturbance_window,
    )

    summary = {
        "backend": pid_result["backend"],
        "scenario_name": "x_highspeed_midhold_sine_disturbance",
        "scenario_definition": {
            "speed_profile": {"forward": 0.6, "reverse": -0.6},
            "stage_lengths_steps": list(cfg.axis_config("x").fixed_stage_lengths or ()),
            "stage_lengths_seconds": [round(length / cfg.control_freq_hz, 6) for length in (cfg.axis_config("x").fixed_stage_lengths or ())],
            "disturbance_mode": "pybullet_external_force_sine",
            "disturbance_axis": "x",
            "disturbance_amplitude_n": float(cfg.axis_config("x").disturbance_scale),
            "disturbance_step_window": list(disturbance_window),
            "disturbance_frequency_rad": float(cfg.axis_config("x").disturbance_frequency_rad),
        },
        "reference_segments": segments,
        "controllers": {
            "pid_pos_att": pid_result["metrics"],
            "ladrc_x_pos_pid_att_default": default_result["metrics"],
            "ladrc_x_pos_pid_att_tuned": tuned_result["metrics"],
        },
        "stage_metrics": {
            "pid_pos_att": pid_stage_metrics,
            "ladrc_x_pos_pid_att_default": default_stage_metrics,
            "ladrc_x_pos_pid_att_tuned": tuned_stage_metrics,
        },
        "tuned_params": {"b0": 0.8, "omega_c": 6.6, "k": 3.2, "source": "current_disturbance_tuned"},
        "figures": [str(path) for path in figure_paths],
        "video_dirs": {
            "pid_pos_att": str(output_root / "videos" / "pid"),
            "ladrc_x_pos_pid_att_default": str(output_root / "videos" / "ladrc_default"),
            "ladrc_x_pos_pid_att_tuned": str(output_root / "videos" / "ladrc_tuned"),
        },
    }
    write_summary_json(output_root / "summary.json", summary)
    (output_root / "summary_readable.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_root), "backend": pid_result["backend"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
