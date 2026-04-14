from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lc.common.io import ensure_dir
from lc.control.configs import ArtifactConfig, AxisTrainingConfig, PyBulletControlExperimentConfig
from lc.control.controllers import create_controller_bundle
from lc.control.envs import run_controller_episode
from lc.control.io import write_metrics_csv, write_reference_csv, write_summary_json, write_timeseries_csv
from lc.control.reference_generators import build_xyz_reference_trajectory


@dataclass(frozen=True)
class FixedLADRCCandidate:
    label: str
    b0: float
    omega_c: float
    r: float
    omega_o: float | None = None
    k: float | None = None

    @property
    def resolved_k(self) -> float:
        if self.k is not None:
            return float(self.k)
        return float(self.omega_o or 0.0) / max(float(self.omega_c), 1e-6)

    @property
    def resolved_omega_o(self) -> float:
        if self.omega_o is not None:
            return float(self.omega_o)
        return float(self.resolved_k * self.omega_c)


PID_LABEL = "pid_pos_att"
WITH_R_LABEL = "ladrc_retuned_r63"
NO_R_LABEL = "ladrc_retuned_no_r_search"
SWEEP_SPEEDS = tuple(round(step * 0.1, 1) for step in range(1, 9))
TYPICAL_SPEEDS = (0.1, 0.5, 0.8)

RETUNED_WITH_R63 = FixedLADRCCandidate(
    label=WITH_R_LABEL,
    b0=24.3,
    omega_c=2.95,
    omega_o=21.875,
    r=63.0,
)
RETUNED_WITHOUT_R_SEARCH = FixedLADRCCandidate(
    label=NO_R_LABEL,
    b0=29.75,
    omega_c=6.6,
    k=4.8,
    r=10.0,
)


def _build_config(output_root: Path, speed_mps: float, *, record_video: bool) -> PyBulletControlExperimentConfig:
    x_axis = AxisTrainingConfig(
        axis="x",
        initial_position=(0.0, 0.0, 1.0),
        fixed_axes=(0.0, 1.0),
        primary_speed_range=(speed_mps, speed_mps),
        reverse_speed_range=(-speed_mps, -speed_mps),
        stage_duration_range=(2.0, 2.0),
        include_disturbance=False,
        disturbance_scale=0.0,
        stage_count=1,
        fixed_stage_lengths=(96,),
        fixed_stage_velocities=(speed_mps,),
    )
    return PyBulletControlExperimentConfig(
        gui=False,
        seed=7,
        duration_sec=2.0,
        eval_episodes=3,
        artifact=ArtifactConfig(output_root=str(output_root), record_video=record_video),
        axis_configs=(
            x_axis,
            AxisTrainingConfig(axis="y"),
            AxisTrainingConfig(axis="z", fixed_axes=(0.0, 0.0), initial_position=(0.0, 0.0, 1.0)),
        ),
    )


def _load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("arial.ttf", "msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _color(hex_value: str) -> tuple[int, int, int]:
    hex_value = hex_value.lstrip("#")
    return tuple(int(hex_value[index:index + 2], 16) for index in (0, 2, 4))


def _draw_dashed_line(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], color: tuple[int, int, int]) -> None:
    for start, stop in zip(points[::2], points[1::2]):
        draw.line([start, stop], fill=color, width=2)


def _render_line_chart(
    output_path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_values: np.ndarray,
    series: list[tuple[str, np.ndarray, str]],
    zero_line: bool = False,
) -> Path:
    width = 1280
    height = 720
    margin_left = 110
    margin_right = 40
    margin_top = 90
    margin_bottom = 100
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    all_y = np.concatenate([np.asarray(values, dtype=np.float64) for _, values, _ in series])
    if zero_line:
        all_y = np.concatenate([all_y, np.asarray([0.0])])
    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values)) if len(x_values) > 1 else float(np.min(x_values) + 1.0)
    y_min = float(np.min(all_y))
    y_max = float(np.max(all_y))
    if abs(y_max - y_min) < 1e-9:
        y_min -= 1.0
        y_max += 1.0
    y_pad = 0.08 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    def map_x(value: float) -> float:
        ratio = 0.0 if x_max == x_min else (value - x_min) / (x_max - x_min)
        return plot_left + ratio * plot_width

    def map_y(value: float) -> float:
        ratio = 0.0 if y_max == y_min else (value - y_min) / (y_max - y_min)
        return plot_bottom - ratio * plot_height

    image = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    title_font = _load_font(30)
    axis_font = _load_font(22)
    tick_font = _load_font(18)
    legend_font = _load_font(20)

    draw.rounded_rectangle(
        (plot_left, plot_top, plot_right, plot_bottom),
        radius=14,
        outline=(205, 208, 212),
        width=2,
        fill=(255, 255, 255),
    )
    for index in range(6):
        y_value = y_min + (y_max - y_min) * index / 5.0
        y_pixel = map_y(y_value)
        draw.line((plot_left, y_pixel, plot_right, y_pixel), fill=(232, 234, 237), width=1)
        draw.text((18, y_pixel - 10), f"{y_value:.3f}", font=tick_font, fill=(75, 75, 75))
    for index in range(8):
        x_value = x_min + (x_max - x_min) * index / 7.0
        x_pixel = map_x(x_value)
        draw.line((x_pixel, plot_top, x_pixel, plot_bottom), fill=(238, 239, 241), width=1)
        draw.text((x_pixel - 18, plot_bottom + 14), f"{x_value:.1f}", font=tick_font, fill=(75, 75, 75))

    if zero_line and y_min <= 0.0 <= y_max:
        zero_y = map_y(0.0)
        dashed_points = [(x, zero_y) for x in np.linspace(plot_left, plot_right, 40)]
        _draw_dashed_line(draw, dashed_points, (100, 100, 100))

    for label, values, color in series:
        pts = [(map_x(float(x_val)), map_y(float(y_val))) for x_val, y_val in zip(x_values, values)]
        draw.line(pts, fill=_color(color), width=4, joint="curve")

    draw.text((plot_left, 26), title, font=title_font, fill=(25, 25, 25))
    draw.text((width // 2 - 64, height - 52), x_label, font=axis_font, fill=(35, 35, 35))
    draw.text((24, plot_top - 44), y_label, font=axis_font, fill=(35, 35, 35))

    legend_x = plot_left
    legend_y = height - 88
    for label, _, color in series:
        draw.rounded_rectangle((legend_x, legend_y + 4, legend_x + 26, legend_y + 18), radius=4, fill=_color(color))
        draw.text((legend_x + 34, legend_y - 2), label, font=legend_font, fill=(40, 40, 40))
        legend_x += 300

    image.save(output_path)
    return output_path


def _controller_color(label: str) -> str:
    return {
        PID_LABEL: "#d62728",
        WITH_R_LABEL: "#2ca02c",
        NO_R_LABEL: "#1f77b4",
    }[label]


def _controller_display_name(label: str) -> str:
    return {
        PID_LABEL: "PID",
        WITH_R_LABEL: "LADRC(r=63)",
        NO_R_LABEL: "LADRC(no-r)",
    }[label]


def _apply_fixed_ladrc(candidate: FixedLADRCCandidate):
    controller = create_controller_bundle("ladrc_x_pos_pid_att")
    controller.set_axis_parameters(
        "x",
        b0=candidate.b0,
        omega_c=candidate.omega_c,
        omega_o=candidate.resolved_omega_o,
        r=candidate.r,
    )
    return controller


def _run_single_controller(
    cfg: PyBulletControlExperimentConfig,
    speed_dir: Path,
    speed_mps: float,
    label: str,
    reference_bundle,
    *,
    record_video: bool,
) -> dict[str, object]:
    artifact = replace(
        cfg.artifact,
        record_video=record_video,
        video_output_dir=str(speed_dir / "videos" / label) if record_video else None,
    )
    run_cfg = replace(cfg, artifact=artifact)
    if label == PID_LABEL:
        controller = create_controller_bundle("pid_pos_att")
    elif label == WITH_R_LABEL:
        controller = _apply_fixed_ladrc(RETUNED_WITH_R63)
    elif label == NO_R_LABEL:
        controller = _apply_fixed_ladrc(RETUNED_WITHOUT_R_SEARCH)
    else:
        raise KeyError(f"Unknown controller label: {label}")
    result = run_controller_episode(run_cfg, controller, reference_bundle)
    metrics = dict(result["metrics"])
    metrics["speed_mps"] = float(speed_mps)
    metrics["controller"] = label
    metrics["backend"] = str(result["backend"])
    return {
        "result": result,
        "metrics": metrics,
    }


def _plot_speed_metric_curves(figures_dir: Path, metric_rows: list[dict[str, float]]) -> list[Path]:
    paths: list[Path] = []
    x_values = np.asarray(SWEEP_SPEEDS, dtype=np.float64)
    metric_names = (
        ("rmse", "Speed Sweep: RMSE", "RMSE", "speed_rmse_curve.png"),
        ("mae", "Speed Sweep: MAE", "MAE", "speed_mae_curve.png"),
        ("velocity_rmse", "Speed Sweep: Velocity RMSE", "Velocity RMSE", "speed_velocity_rmse_curve.png"),
    )
    for metric_key, title, y_label, filename in metric_names:
        series: list[tuple[str, np.ndarray, str]] = []
        for label in (PID_LABEL, WITH_R_LABEL, NO_R_LABEL):
            values = [
                float(
                    next(
                        row[metric_key]
                        for row in metric_rows
                        if float(row["speed_mps"]) == float(speed) and row["controller"] == label
                    )
                )
                for speed in SWEEP_SPEEDS
            ]
            series.append((_controller_display_name(label), np.asarray(values, dtype=np.float64), _controller_color(label)))
        path = figures_dir / filename
        _render_line_chart(
            path,
            title=title,
            x_label="Speed (m/s)",
            y_label=y_label,
            x_values=x_values,
            series=series,
            zero_line=False,
        )
        paths.append(path)
    return paths


def _plot_typical_speed_figure(
    figures_dir: Path,
    speed_mps: float,
    controller_rows: dict[str, list[dict[str, float]]],
    *,
    value_key: str,
    target_key: str,
    title_suffix: str,
    y_label: str,
    filename_suffix: str,
    zero_line: bool = False,
) -> Path:
    times = np.asarray([float(row["time"]) for row in controller_rows[PID_LABEL]], dtype=np.float64)
    series: list[tuple[str, np.ndarray, str]] = [
        ("Reference", np.asarray([float(row[target_key]) for row in controller_rows[PID_LABEL]], dtype=np.float64), "#222222"),
        (_controller_display_name(PID_LABEL), np.asarray([float(row[value_key]) for row in controller_rows[PID_LABEL]], dtype=np.float64), _controller_color(PID_LABEL)),
        (_controller_display_name(WITH_R_LABEL), np.asarray([float(row[value_key]) for row in controller_rows[WITH_R_LABEL]], dtype=np.float64), _controller_color(WITH_R_LABEL)),
        (_controller_display_name(NO_R_LABEL), np.asarray([float(row[value_key]) for row in controller_rows[NO_R_LABEL]], dtype=np.float64), _controller_color(NO_R_LABEL)),
    ]
    path = figures_dir / f"speed_{speed_mps:.1f}_{filename_suffix}.png"
    _render_line_chart(
        path,
        title=f"{speed_mps:.1f} m/s {title_suffix}",
        x_label="Time (s)",
        y_label=y_label,
        x_values=times,
        series=series,
        zero_line=zero_line,
    )
    return path


def main() -> None:
    axis = "x"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = ensure_dir(Path("outputs") / "control_pybullet" / "x_speed_sweep_pid_vs_ladrc_r63_compare" / axis / stamp)
    figures_dir = ensure_dir(output_root / "figures")

    metric_rows: list[dict[str, float]] = []
    typical_rows: dict[float, dict[str, list[dict[str, float]]]] = {}
    generated_videos: dict[str, dict[str, str]] = {}

    for speed_mps in SWEEP_SPEEDS:
        speed_dir = ensure_dir(output_root / "speeds" / f"{speed_mps:.1f}")
        record_video = speed_mps in TYPICAL_SPEEDS
        cfg = _build_config(speed_dir, speed_mps, record_video=record_video)
        reference_bundle = build_xyz_reference_trajectory(cfg.axis_config(axis), cfg, rng=np.random.default_rng(cfg.seed))
        write_reference_csv(speed_dir / "reference.csv", reference_bundle)

        speed_controller_rows: dict[str, list[dict[str, float]]] = {}
        speed_metrics: list[dict[str, float]] = []
        speed_video_paths: dict[str, str] = {}
        for label in (PID_LABEL, WITH_R_LABEL, NO_R_LABEL):
            controller_run = _run_single_controller(
                cfg,
                speed_dir,
                speed_mps,
                label,
                reference_bundle,
                record_video=record_video,
            )
            result = controller_run["result"]
            rows = list(result["timeseries"])
            metrics = dict(controller_run["metrics"])
            speed_controller_rows[label] = rows
            speed_metrics.append(metrics)
            metric_rows.append(metrics)
            write_timeseries_csv(speed_dir / f"{label}_timeseries.csv", rows)
            if record_video:
                speed_video_paths[label] = str(speed_dir / "videos" / label / "episode.gif")

        write_metrics_csv(speed_dir / "metrics.csv", speed_metrics)
        if record_video:
            typical_rows[float(speed_mps)] = speed_controller_rows
            generated_videos[f"{speed_mps:.1f}"] = speed_video_paths

    write_metrics_csv(output_root / "metrics_by_speed.csv", metric_rows)

    figure_paths = _plot_speed_metric_curves(figures_dir, metric_rows)
    for speed_mps in TYPICAL_SPEEDS:
        controller_rows = typical_rows[float(speed_mps)]
        figure_paths.append(
            _plot_typical_speed_figure(
                figures_dir,
                speed_mps,
                controller_rows,
                value_key="x",
                target_key="target_x",
                title_suffix="Position Tracking",
                y_label="Position (m)",
                filename_suffix="position_tracking",
            )
        )
        figure_paths.append(
            _plot_typical_speed_figure(
                figures_dir,
                speed_mps,
                controller_rows,
                value_key="vx",
                target_key="target_vx",
                title_suffix="Velocity Tracking",
                y_label="Velocity (m/s)",
                filename_suffix="velocity_tracking",
                zero_line=True,
            )
        )

    summary = {
        "axis": axis,
        "scenario_name": "x_speed_sweep_pid_vs_ladrc_r63_compare",
        "backend": "gym_env",
        "speed_grid_mps": list(SWEEP_SPEEDS),
        "typical_speeds_mps": list(TYPICAL_SPEEDS),
        "route_definition": {
            "duration_sec": 2.0,
            "steps": 96,
            "fixed_stage_lengths": [96],
            "mode": "single_forward_segment",
        },
        "controllers": {
            PID_LABEL: {"variant": "pid_pos_att"},
            WITH_R_LABEL: {
                "variant": "ladrc_x_pos_pid_att",
                "b0": RETUNED_WITH_R63.b0,
                "omega_c": RETUNED_WITH_R63.omega_c,
                "omega_o": RETUNED_WITH_R63.resolved_omega_o,
                "k": RETUNED_WITH_R63.resolved_k,
                "r": RETUNED_WITH_R63.r,
            },
            NO_R_LABEL: {
                "variant": "ladrc_x_pos_pid_att",
                "b0": RETUNED_WITHOUT_R_SEARCH.b0,
                "omega_c": RETUNED_WITHOUT_R_SEARCH.omega_c,
                "omega_o": RETUNED_WITHOUT_R_SEARCH.resolved_omega_o,
                "k": RETUNED_WITHOUT_R_SEARCH.resolved_k,
                "r": RETUNED_WITHOUT_R_SEARCH.r,
            },
        },
        "figures": [str(path) for path in figure_paths],
        "videos": generated_videos,
        "metrics_by_speed_csv": str(output_root / "metrics_by_speed.csv"),
        "speed_dirs": {f"{speed:.1f}": str(output_root / "speeds" / f"{speed:.1f}") for speed in SWEEP_SPEEDS},
    }
    write_summary_json(output_root / "summary.json", summary)
    (output_root / "summary_readable.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "speed_count": len(SWEEP_SPEEDS),
                "typical_speeds": list(TYPICAL_SPEEDS),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
