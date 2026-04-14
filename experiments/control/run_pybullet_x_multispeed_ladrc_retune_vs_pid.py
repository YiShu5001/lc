from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
from lc.control.reference_generators import build_xyz_reference_trajectory, summarize_reference_segments
from lc.control.trainers import PyBulletAxisTrainer


@dataclass(frozen=True)
class Candidate:
    name: str
    b0: float
    omega_c: float
    omega_o: float
    r: float

    @property
    def k(self) -> float:
        return float(self.omega_o) / max(float(self.omega_c), 1e-6)


PID_LABEL = "pid_pos_att"
DEFAULT_LABEL = "ladrc_x_pos_pid_att_default"
RETUNED_LABEL = "ladrc_x_pos_pid_att_retuned"
SWEEP_SPEEDS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4)
TYPICAL_SPEEDS = (0.2, 0.8, 1.4)
MOVE_STEPS = 96
HOLD_STEPS = 24
TOTAL_STEPS = MOVE_STEPS + HOLD_STEPS
DURATION_SEC = TOTAL_STEPS / 48.0


def _candidate_dict(candidate: Candidate) -> dict[str, float]:
    return {
        "b0": float(candidate.b0),
        "omega_c": float(candidate.omega_c),
        "omega_o": float(candidate.omega_o),
        "k": float(candidate.k),
        "r": float(candidate.r),
    }


def _build_config(output_root: Path, speed_mps: float, *, record_video: bool) -> PyBulletControlExperimentConfig:
    x_axis = AxisTrainingConfig(
        axis="x",
        initial_position=(0.0, 0.0, 1.0),
        fixed_axes=(0.0, 1.0),
        primary_speed_range=(speed_mps, speed_mps),
        reverse_speed_range=(-speed_mps, -speed_mps),
        stage_duration_range=(2.0, 0.5),
        include_disturbance=False,
        disturbance_scale=0.0,
        stage_count=2,
        fixed_stage_lengths=(MOVE_STEPS, HOLD_STEPS),
        fixed_stage_velocities=(speed_mps, 0.0),
    )
    return PyBulletControlExperimentConfig(
        gui=False,
        seed=7,
        duration_sec=DURATION_SEC,
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
        draw.text((18, y_pixel - 10), f"{y_value:.2f}", font=tick_font, fill=(75, 75, 75))
    for index in range(6):
        x_value = x_min + (x_max - x_min) * index / 5.0
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
    draw.text((width // 2 - 42, height - 52), x_label, font=axis_font, fill=(35, 35, 35))
    draw.text((24, plot_top - 44), y_label, font=axis_font, fill=(35, 35, 35))

    legend_x = plot_left
    legend_y = height - 88
    for label, _, color in series:
        draw.rounded_rectangle((legend_x, legend_y + 4, legend_x + 26, legend_y + 18), radius=4, fill=_color(color))
        draw.text((legend_x + 34, legend_y - 2), label, font=legend_font, fill=(40, 40, 40))
        legend_x += 210

    image.save(output_path)
    return output_path


def _build_fallback_gif(video_dir: Path, rows: list[dict[str, float]], title: str) -> Path:
    video_dir.mkdir(parents=True, exist_ok=True)
    gif_path = video_dir / "episode.gif"
    width = 960
    height = 540
    rail_y = 290
    left = 90
    right = width - 90
    x_values = [float(row["x"]) for row in rows]
    ref_values = [float(row["target_x"]) for row in rows]
    x_min = min(x_values + ref_values)
    x_max = max(x_values + ref_values)
    if abs(x_max - x_min) < 1e-9:
        x_min -= 1.0
        x_max += 1.0
    pad = 0.15 * (x_max - x_min)
    x_min -= pad
    x_max += pad
    title_font = _load_font(26)
    body_font = _load_font(20)
    small_font = _load_font(16)

    def map_x(value: float) -> int:
        ratio = 0.0 if x_max == x_min else (value - x_min) / (x_max - x_min)
        return int(round(left + ratio * (right - left)))

    frames: list[Image.Image] = []
    trail: list[tuple[int, int]] = []
    for index, row in enumerate(rows):
        image = Image.new("RGB", (width, height), (247, 248, 250))
        draw = ImageDraw.Draw(image)
        current_x = float(row["x"])
        target_x = float(row["target_x"])
        trail.append((map_x(current_x), rail_y))

        draw.text((32, 24), title, font=title_font, fill=(25, 25, 25))
        draw.text((32, 64), f"time={float(row['time']):.2f}s  x={current_x:.3f}m  target={target_x:.3f}m", font=body_font, fill=(55, 55, 55))
        draw.text((32, 96), "Gray: reference   Red: controller output", font=small_font, fill=(85, 85, 85))

        draw.rounded_rectangle((left - 26, rail_y - 60, right + 26, rail_y + 60), radius=20, outline=(210, 214, 218), width=2, fill=(255, 255, 255))
        draw.line((left, rail_y, right, rail_y), fill=(120, 126, 132), width=4)

        for marker in np.linspace(x_min, x_max, 7):
            marker_x = map_x(float(marker))
            draw.line((marker_x, rail_y - 12, marker_x, rail_y + 12), fill=(170, 175, 180), width=2)
            draw.text((marker_x - 16, rail_y + 22), f"{marker:.2f}", font=small_font, fill=(95, 95, 95))

        if len(trail) > 1:
            draw.line(trail, fill=(214, 39, 40), width=3)

        ref_px = map_x(target_x)
        cur_px = map_x(current_x)
        draw.ellipse((ref_px - 12, rail_y - 12, ref_px + 12, rail_y + 12), fill=(34, 34, 34))
        draw.ellipse((cur_px - 14, rail_y - 44, cur_px + 14, rail_y - 16), fill=(214, 39, 40))
        draw.text((ref_px - 18, rail_y - 42), "Ref", font=small_font, fill=(34, 34, 34))
        draw.text((cur_px - 18, rail_y - 78), "Out", font=small_font, fill=(214, 39, 40))

        draw.rounded_rectangle((90, 380, width - 90, 470), radius=18, outline=(220, 223, 227), width=2, fill=(255, 255, 255))
        draw.text((116, 404), f"step {index + 1}/{len(rows)}", font=body_font, fill=(40, 40, 40))
        draw.text((300, 404), f"error = {target_x - current_x:+.3f} m", font=body_font, fill=(40, 40, 40))
        frames.append(image)

    duration_ms = max(int(round(1000 / 12)), 1)
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    for frame in frames:
        frame.close()
    return gif_path


def _ensure_video(output_root: Path, subdir: str, rows: list[dict[str, float]], title: str) -> Path:
    video_dir = output_root / "videos" / subdir
    gif_path = video_dir / "episode.gif"
    if gif_path.exists():
        return gif_path
    return _build_fallback_gif(video_dir, rows, title)


def _compute_risk_flags(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "end_vx_mean": 0.0,
            "negative_tail_velocity": 0.0,
            "max_tail_pitch_abs": 0.0,
            "tail_rpm_edge_fraction": 0.0,
            "tail_same_rpm_fraction": 0.0,
            "unstable_tail": 1.0,
            "hold_steady_state_error": 1.0e12,
        }
    tail = rows[-8:]
    hold_rows = rows[MOVE_STEPS:]
    tail_vx = np.asarray([float(row["vx"]) for row in tail], dtype=np.float64)
    tail_pitch = np.asarray([abs(float(row["pitch"])) for row in tail], dtype=np.float64)
    tail_rpm = np.asarray(
        [[float(row["rpm0"]), float(row["rpm1"]), float(row["rpm2"]), float(row["rpm3"])] for row in tail],
        dtype=np.float64,
    )
    tail_mean_rpm = np.mean(tail_rpm, axis=1)
    edge_mask = np.logical_or(np.min(tail_rpm, axis=1) <= 10000.0, np.max(tail_rpm, axis=1) >= 19000.0)
    same_rpm_mask = np.max(np.abs(tail_rpm - tail_mean_rpm[:, None]), axis=1) <= 5.0
    hold_error = [abs(float(row["target_x"]) - float(row["x"])) for row in hold_rows]
    negative_tail_velocity = float(np.mean(tail_vx) < -0.05)
    max_tail_pitch_abs = float(np.max(tail_pitch))
    tail_rpm_edge_fraction = float(np.mean(edge_mask.astype(np.float64)))
    tail_same_rpm_fraction = float(np.mean(same_rpm_mask.astype(np.float64)))
    unstable_tail = float(
        negative_tail_velocity > 0.5
        or max_tail_pitch_abs > 0.8
        or tail_rpm_edge_fraction > 0.5
        or tail_same_rpm_fraction > 0.5
    )
    return {
        "end_vx_mean": float(np.mean(tail_vx)),
        "negative_tail_velocity": negative_tail_velocity,
        "max_tail_pitch_abs": max_tail_pitch_abs,
        "tail_rpm_edge_fraction": tail_rpm_edge_fraction,
        "tail_same_rpm_fraction": tail_same_rpm_fraction,
        "unstable_tail": unstable_tail,
        "hold_steady_state_error": float(np.mean(hold_error)) if hold_error else 0.0,
    }


def _custom_score(metrics: dict[str, float], pid_metrics: dict[str, float] | None = None) -> float:
    rmse = float(metrics["rmse"])
    mae = float(metrics["mae"])
    velocity_rmse = float(metrics.get("velocity_rmse", 0.0))
    steady_state = float(metrics["steady_state_error"])
    hold_steady = float(metrics.get("hold_steady_state_error", steady_state))
    overshoot = float(metrics["overshoot"])
    score = 0.46 * rmse + 0.18 * mae + 0.16 * velocity_rmse + 0.12 * steady_state + 0.06 * hold_steady + 0.02 * overshoot
    if pid_metrics is not None and float(pid_metrics.get("rmse", 0.0)) > 0.0:
        score += 0.18 * max(rmse - float(pid_metrics["rmse"]), 0.0)
        score += 0.08 * max(mae - float(pid_metrics["mae"]), 0.0)
    if float(metrics.get("negative_tail_velocity", 0.0)) > 0.5:
        score += 10.0
    if float(metrics.get("max_tail_pitch_abs", 0.0)) > 0.8:
        score += 6.0
    if float(metrics.get("tail_rpm_edge_fraction", 0.0)) > 0.5:
        score += 6.0
    if float(metrics.get("tail_same_rpm_fraction", 0.0)) > 0.5:
        score += 6.0
    if float(metrics.get("unstable_tail", 0.0)) > 0.5:
        score += 25.0
    return float(score)


def _evaluate_candidate(
    trainer: PyBulletAxisTrainer,
    axis: str,
    candidate: Candidate,
    *,
    episodes: int,
    pid_metrics: dict[str, float],
) -> dict[str, float]:
    original_config = trainer.config
    trainer.config = replace(
        trainer.config,
        artifact=replace(
            trainer.config.artifact,
            record_video=False,
            video_output_dir=None,
        ),
    )
    try:
        result = trainer.evaluate_single_axis_ladrc_variant(
            axis,
            _candidate_dict(candidate),
            difficulty="medium",
            episodes=episodes,
        )
    finally:
        trainer.config = original_config
    row = dict(result["metrics"])
    best_timeseries = list(result.get("result", {}).get("timeseries", []))
    risk_flags = _compute_risk_flags(best_timeseries)
    row.update(risk_flags)
    row["name"] = candidate.name
    row["b0"] = float(candidate.b0)
    row["omega_c"] = float(candidate.omega_c)
    row["omega_o"] = float(candidate.omega_o)
    row["k"] = float(candidate.k)
    row["r"] = float(candidate.r)
    row["score"] = _custom_score(row, pid_metrics)
    return row


def _build_r_scan_candidates(default_axis: object) -> list[Candidate]:
    r_values = (10.0, 20.0, 30.0, 40.0, 50.0, 70.0, 90.0)
    return [
        Candidate(
            name=f"r_scan_r{r_value:g}",
            b0=float(default_axis.b0),
            omega_c=float(default_axis.omega_c),
            omega_o=float(default_axis.omega_c) * float(default_axis.k),
            r=float(r_value),
        )
        for r_value in r_values
    ]


def _build_coarse_candidates(default_axis: object, seed_r: float) -> list[Candidate]:
    b0_values = (18.0, 24.0, 30.5, 37.5, 45.0)
    wc_values = (2.0, 3.5, 5.0, 6.5, 8.0)
    k_values = (2.0, 3.5, 5.0, 6.5, 8.0)
    candidates: list[Candidate] = []
    seen: set[tuple[float, float, float, float]] = set()
    default_candidate = Candidate(
        "repo_default_seeded_r",
        float(default_axis.b0),
        float(default_axis.omega_c),
        float(default_axis.omega_c) * float(default_axis.k),
        float(seed_r),
    )
    for candidate in [default_candidate]:
        key = (round(candidate.b0, 6), round(candidate.omega_c, 6), round(candidate.omega_o, 6), round(candidate.r, 6))
        seen.add(key)
        candidates.append(candidate)
    for b0 in b0_values:
        for omega_c in wc_values:
            for k in k_values:
                omega_o = omega_c * k
                candidate = Candidate(
                    name=f"coarse_b{b0:g}_wc{omega_c:g}_k{k:g}_r{seed_r:g}",
                    b0=float(b0),
                    omega_c=float(omega_c),
                    omega_o=float(omega_o),
                    r=float(seed_r),
                )
                key = (round(candidate.b0, 6), round(candidate.omega_c, 6), round(candidate.omega_o, 6), round(candidate.r, 6))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
    return candidates


def _build_refine_candidates(top_rows: list[dict[str, float]]) -> list[Candidate]:
    b0_scales = (0.90, 0.98, 1.00, 1.08)
    wc_scales = (0.90, 0.98, 1.00, 1.08)
    k_scales = (0.88, 0.96, 1.00, 1.10)
    r_scales = (0.85, 0.95, 1.00, 1.12)
    candidates: list[Candidate] = []
    seen: set[tuple[float, float, float, float]] = set()
    for row in top_rows:
        for b0_scale in b0_scales:
            for wc_scale in wc_scales:
                for k_scale in k_scales:
                    for r_scale in r_scales:
                        omega_c = max(float(row["omega_c"]) * wc_scale, 0.1)
                        k = max(float(row["k"]) * k_scale, 0.1)
                        candidate = Candidate(
                            name=f"refine_{row['name']}_b{b0_scale:.2f}_wc{wc_scale:.2f}_k{k_scale:.2f}_r{r_scale:.2f}",
                            b0=max(float(row["b0"]) * b0_scale, 0.1),
                            omega_c=omega_c,
                            omega_o=max(omega_c * k, 0.1),
                            r=max(float(row["r"]) * r_scale, 1.0),
                        )
                        key = (round(candidate.b0, 6), round(candidate.omega_c, 6), round(candidate.omega_o, 6), round(candidate.r, 6))
                        if key in seen:
                            continue
                        seen.add(key)
                        candidates.append(candidate)
    return candidates


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
            velocity_rmse = 0.0
        else:
            rmse = float(np.sqrt(np.mean(np.square(errors))))
            mae = float(np.mean(np.abs(errors)))
            velocity_rmse = float(
                np.sqrt(np.mean([(float(row["target_vx"]) - float(row["vx"])) ** 2 for row in chunk]))
            )
        metrics.append(
            {
                "stage": int(segment["stage"]),
                "start": start,
                "stop": stop,
                "velocity": float(segment["velocity"]),
                "rmse": rmse,
                "mae": mae,
                "velocity_rmse": velocity_rmse,
            }
        )
    return metrics


def _plot_speed_metric_curves(figures_dir: Path, metric_rows: list[dict[str, float]]) -> list[Path]:
    paths: list[Path] = []
    x_values = np.asarray(SWEEP_SPEEDS, dtype=np.float64)
    metric_names = (
        ("rmse", "Speed Sweep: RMSE", "RMSE", "speed_rmse_curve.png"),
        ("mae", "Speed Sweep: MAE", "MAE", "speed_mae_curve.png"),
        ("velocity_rmse", "Speed Sweep: Velocity RMSE", "Velocity RMSE", "speed_velocity_rmse_curve.png"),
        ("hold_steady_state_error", "Speed Sweep: Hold Steady-State Error", "Hold Error (m)", "speed_hold_error_curve.png"),
    )
    color_map = {PID_LABEL: "#d62728", DEFAULT_LABEL: "#1f77b4", RETUNED_LABEL: "#2ca02c"}
    label_map = {PID_LABEL: "PID", DEFAULT_LABEL: "LADRC(default)", RETUNED_LABEL: "LADRC(retuned)"}
    for metric_key, title, y_label, filename in metric_names:
        series: list[tuple[str, np.ndarray, str]] = []
        for controller in (PID_LABEL, DEFAULT_LABEL, RETUNED_LABEL):
            values = [
                float(next(row[metric_key] for row in metric_rows if float(row["speed_mps"]) == float(speed) and row["controller"] == controller))
                for speed in SWEEP_SPEEDS
            ]
            series.append((label_map[controller], np.asarray(values, dtype=np.float64), color_map[controller]))
        path = figures_dir / filename
        _render_line_chart(path, title=title, x_label="Speed (m/s)", y_label=y_label, x_values=x_values, series=series)
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
        ("PID", np.asarray([float(row[value_key]) for row in controller_rows[PID_LABEL]], dtype=np.float64), "#d62728"),
        ("LADRC(default)", np.asarray([float(row[value_key]) for row in controller_rows[DEFAULT_LABEL]], dtype=np.float64), "#1f77b4"),
        ("LADRC(retuned)", np.asarray([float(row[value_key]) for row in controller_rows[RETUNED_LABEL]], dtype=np.float64), "#2ca02c"),
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


def _run_speedpoint(output_root: Path, speed_mps: float) -> dict[str, object]:
    axis = "x"
    speed_dir = ensure_dir(output_root / "speeds" / f"{speed_mps:.1f}")
    record_video = speed_mps in TYPICAL_SPEEDS
    cfg = _build_config(speed_dir, speed_mps, record_video=record_video)
    trainer = PyBulletAxisTrainer(cfg)
    default_axis = create_controller_bundle("ladrc_x_pos_pid_att").parameter_set.axis_config(axis)
    reference_bundle = build_xyz_reference_trajectory(cfg.axis_config(axis), cfg, rng=np.random.default_rng(cfg.seed))
    segments = summarize_reference_segments(reference_bundle)

    pid_cfg = replace(cfg, artifact=replace(cfg.artifact, video_output_dir=str(speed_dir / "videos" / "pid")))
    pid_controller = create_controller_bundle(PID_LABEL)
    pid_result = run_controller_episode(pid_cfg, pid_controller, reference_bundle)
    pid_metrics = dict(pid_result["metrics"])
    pid_metrics.update(_compute_risk_flags(list(pid_result["timeseries"])))
    pid_metrics["controller"] = PID_LABEL
    pid_metrics["speed_mps"] = float(speed_mps)
    pid_metrics["score"] = _custom_score(pid_metrics)

    r_scan_rows = sorted(
        [_evaluate_candidate(trainer, axis, candidate, episodes=1, pid_metrics=pid_metrics) for candidate in _build_r_scan_candidates(default_axis)],
        key=lambda row: float(row["score"]),
    )
    best_r = float(r_scan_rows[0]["r"])
    coarse_rows = sorted(
        [_evaluate_candidate(trainer, axis, candidate, episodes=1, pid_metrics=pid_metrics) for candidate in _build_coarse_candidates(default_axis, best_r)],
        key=lambda row: float(row["score"]),
    )
    refine_seeds = sorted(r_scan_rows[:2] + coarse_rows[:3], key=lambda row: float(row["score"]))[:3]
    refine_rows = sorted(
        [_evaluate_candidate(trainer, axis, candidate, episodes=2, pid_metrics=pid_metrics) for candidate in _build_refine_candidates(refine_seeds)],
        key=lambda row: float(row["score"]),
    )

    best_row = min(refine_rows + coarse_rows + r_scan_rows, key=lambda row: float(row["score"]))
    tuned_candidate = Candidate(
        name=f"retuned_speed_{speed_mps:.1f}",
        b0=float(best_row["b0"]),
        omega_c=float(best_row["omega_c"]),
        omega_o=float(best_row["omega_o"]),
        r=float(best_row["r"]),
    )

    default_controller = create_controller_bundle("ladrc_x_pos_pid_att")
    tuned_controller = create_controller_bundle("ladrc_x_pos_pid_att")
    tuned_controller.set_axis_parameters(axis, b0=tuned_candidate.b0, omega_c=tuned_candidate.omega_c, omega_o=tuned_candidate.omega_o, r=tuned_candidate.r)

    default_cfg = replace(cfg, artifact=replace(cfg.artifact, video_output_dir=str(speed_dir / "videos" / "ladrc_default")))
    tuned_cfg = replace(cfg, artifact=replace(cfg.artifact, video_output_dir=str(speed_dir / "videos" / "ladrc_retuned")))
    default_result = run_controller_episode(default_cfg, default_controller, reference_bundle)
    tuned_result = run_controller_episode(tuned_cfg, tuned_controller, reference_bundle)

    default_metrics = dict(default_result["metrics"])
    default_metrics.update(_compute_risk_flags(list(default_result["timeseries"])))
    default_metrics["controller"] = DEFAULT_LABEL
    default_metrics["speed_mps"] = float(speed_mps)
    default_metrics["score"] = _custom_score(default_metrics, pid_metrics)

    tuned_metrics = dict(tuned_result["metrics"])
    tuned_metrics.update(_compute_risk_flags(list(tuned_result["timeseries"])))
    tuned_metrics["controller"] = RETUNED_LABEL
    tuned_metrics["speed_mps"] = float(speed_mps)
    tuned_metrics["score"] = _custom_score(tuned_metrics, pid_metrics)

    write_reference_csv(speed_dir / "reference.csv", reference_bundle)
    write_timeseries_csv(speed_dir / "pid_timeseries.csv", list(pid_result["timeseries"]))
    write_timeseries_csv(speed_dir / "default_ladrc_timeseries.csv", list(default_result["timeseries"]))
    write_timeseries_csv(speed_dir / "retuned_ladrc_timeseries.csv", list(tuned_result["timeseries"]))
    write_metrics_csv(speed_dir / "metrics.csv", [pid_metrics, default_metrics, tuned_metrics])
    write_metrics_csv(speed_dir / "candidate_r_scan.csv", r_scan_rows)
    write_metrics_csv(speed_dir / "candidate_coarse.csv", coarse_rows)
    write_metrics_csv(speed_dir / "candidate_refine.csv", refine_rows)

    pid_stage_metrics = _compute_stage_metrics(list(pid_result["timeseries"]), segments)
    default_stage_metrics = _compute_stage_metrics(list(default_result["timeseries"]), segments)
    tuned_stage_metrics = _compute_stage_metrics(list(tuned_result["timeseries"]), segments)
    write_metrics_csv(
        speed_dir / "stage_metrics.csv",
        [{"controller": PID_LABEL, **row} for row in pid_stage_metrics]
        + [{"controller": DEFAULT_LABEL, **row} for row in default_stage_metrics]
        + [{"controller": RETUNED_LABEL, **row} for row in tuned_stage_metrics],
    )

    video_paths: dict[str, str] = {}
    if record_video:
        video_paths = {
            PID_LABEL: str(_ensure_video(speed_dir, "pid", list(pid_result["timeseries"]), f"PID {speed_mps:.1f} m/s tracking")),
            DEFAULT_LABEL: str(_ensure_video(speed_dir, "ladrc_default", list(default_result["timeseries"]), f"Default LADRC {speed_mps:.1f} m/s tracking")),
            RETUNED_LABEL: str(_ensure_video(speed_dir, "ladrc_retuned", list(tuned_result["timeseries"]), f"Retuned LADRC {speed_mps:.1f} m/s tracking")),
        }

    speed_summary = {
        "speed_mps": float(speed_mps),
        "route_definition": {
            "duration_sec": float(cfg.duration_sec),
            "steps": int(cfg.step_count),
            "fixed_stage_lengths": [MOVE_STEPS, HOLD_STEPS],
            "fixed_stage_velocities": [float(speed_mps), 0.0],
        },
        "reference_segments": segments,
        "default_repo_params": {
            "b0": float(default_axis.b0),
            "omega_c": float(default_axis.omega_c),
            "omega_o": float(default_axis.omega_c * default_axis.k),
            "k": float(default_axis.k),
            "r": float(default_axis.r),
        },
        "retuned_candidate": asdict(tuned_candidate) | {"k": float(tuned_candidate.k), "omega_o": float(tuned_candidate.omega_o)},
        "controllers": {
            PID_LABEL: pid_metrics,
            DEFAULT_LABEL: default_metrics,
            RETUNED_LABEL: tuned_metrics,
        },
        "stage_metrics": {
            PID_LABEL: pid_stage_metrics,
            DEFAULT_LABEL: default_stage_metrics,
            RETUNED_LABEL: tuned_stage_metrics,
        },
        "acceptance": {
            "tuned_beats_default": bool(float(tuned_metrics["rmse"]) < float(default_metrics["rmse"]) and float(tuned_metrics["velocity_rmse"]) < float(default_metrics["velocity_rmse"])),
            "rmse_ratio_to_pid": float(tuned_metrics["rmse"]) / max(float(pid_metrics["rmse"]), 1e-9),
        },
        "videos": video_paths,
    }
    write_summary_json(speed_dir / "summary.json", speed_summary)
    (speed_dir / "summary_readable.json").write_text(json.dumps(speed_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "speed_mps": float(speed_mps),
        "pid_metrics": pid_metrics,
        "default_metrics": default_metrics,
        "retuned_metrics": tuned_metrics,
        "retuned_candidate": tuned_candidate,
        "speed_dir": str(speed_dir),
        "video_paths": video_paths,
        "controller_rows": {
            PID_LABEL: list(pid_result["timeseries"]),
            DEFAULT_LABEL: list(default_result["timeseries"]),
            RETUNED_LABEL: list(tuned_result["timeseries"]),
        },
    }


def main() -> None:
    axis = "x"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = ensure_dir(Path("outputs") / "control_pybullet" / "x_multispeed_ladrc_retune_vs_pid" / axis / stamp)
    figures_dir = ensure_dir(output_root / "figures")

    metric_rows: list[dict[str, float]] = []
    best_param_rows: list[dict[str, float]] = []
    videos: dict[str, dict[str, str]] = {}
    typical_rows: dict[float, dict[str, list[dict[str, float]]]] = {}
    speed_summaries: dict[str, dict[str, object]] = {}

    for speed_mps in SWEEP_SPEEDS:
        speed_result = _run_speedpoint(output_root, speed_mps)
        pid_metrics = dict(speed_result["pid_metrics"])
        default_metrics = dict(speed_result["default_metrics"])
        retuned_metrics = dict(speed_result["retuned_metrics"])
        tuned_candidate: Candidate = speed_result["retuned_candidate"]

        metric_rows.extend([pid_metrics, default_metrics, retuned_metrics])
        best_param_rows.append(
            {
                "speed_mps": float(speed_mps),
                "r": float(tuned_candidate.r),
                "b0": float(tuned_candidate.b0),
                "omega_c": float(tuned_candidate.omega_c),
                "k": float(tuned_candidate.k),
                "omega_o": float(tuned_candidate.omega_o),
                "pid_rmse": float(pid_metrics["rmse"]),
                "retuned_rmse": float(retuned_metrics["rmse"]),
                "pid_mae": float(pid_metrics["mae"]),
                "retuned_mae": float(retuned_metrics["mae"]),
                "pid_velocity_rmse": float(pid_metrics["velocity_rmse"]),
                "retuned_velocity_rmse": float(retuned_metrics["velocity_rmse"]),
                "rmse_ratio_to_pid": float(retuned_metrics["rmse"]) / max(float(pid_metrics["rmse"]), 1e-9),
            }
        )
        speed_key = f"{speed_mps:.1f}"
        speed_summaries[speed_key] = {
            "speed_dir": speed_result["speed_dir"],
            "retuned_candidate": asdict(tuned_candidate) | {"k": float(tuned_candidate.k), "omega_o": float(tuned_candidate.omega_o)},
            "pid_rmse": float(pid_metrics["rmse"]),
            "retuned_rmse": float(retuned_metrics["rmse"]),
            "rmse_ratio_to_pid": float(retuned_metrics["rmse"]) / max(float(pid_metrics["rmse"]), 1e-9),
        }
        if speed_mps in TYPICAL_SPEEDS:
            typical_rows[float(speed_mps)] = speed_result["controller_rows"]
            videos[speed_key] = speed_result["video_paths"]

    write_metrics_csv(output_root / "metrics_by_speed.csv", metric_rows)
    write_metrics_csv(output_root / "best_params_by_speed.csv", best_param_rows)

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
        "scenario_name": "x_multispeed_ladrc_retune_vs_pid",
        "backend": "gym_env",
        "speed_grid_mps": list(SWEEP_SPEEDS),
        "typical_speeds_mps": list(TYPICAL_SPEEDS),
        "route_definition": {
            "duration_sec": DURATION_SEC,
            "steps": TOTAL_STEPS,
            "fixed_stage_lengths": [MOVE_STEPS, HOLD_STEPS],
            "fixed_stage_velocities_template": ["speed_mps", 0.0],
            "mode": "forward_then_zero_hold",
        },
        "search_definition": {
            "parameters": ["r", "b0", "omega_c", "k"],
            "r_scan_values": [10.0, 20.0, 30.0, 40.0, 50.0, 70.0, 90.0],
            "b0_range": [18.0, 45.0],
            "omega_c_range": [2.0, 9.0],
            "k_range": [2.0, 8.0],
        },
        "figures": [str(path) for path in figure_paths],
        "videos": videos,
        "metrics_by_speed_csv": str(output_root / "metrics_by_speed.csv"),
        "best_params_by_speed_csv": str(output_root / "best_params_by_speed.csv"),
        "speed_summaries": speed_summaries,
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
