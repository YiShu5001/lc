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


@dataclass(frozen=True)
class Candidate:
    name: str
    b0: float
    omega_c: float
    omega_o: float

    @property
    def k(self) -> float:
        return float(self.omega_o) / max(float(self.omega_c), 1e-6)


BASELINE_CANDIDATE = Candidate(
    name="baseline_no_td",
    b0=29.75,
    omega_c=6.6,
    omega_o=31.68,
)


def _build_config(output_root: Path) -> PyBulletControlExperimentConfig:
    x_axis = AxisTrainingConfig(
        axis="x",
        initial_position=(0.0, 0.0, 1.0),
        fixed_axes=(0.0, 1.0),
        primary_speed_range=(0.5, 0.5),
        reverse_speed_range=(-0.5, -0.5),
        stage_duration_range=(2.0, 2.0),
        include_disturbance=False,
        disturbance_scale=0.0,
        stage_count=1,
        fixed_stage_lengths=(96,),
        fixed_stage_velocities=(0.5,),
    )
    return PyBulletControlExperimentConfig(
        gui=False,
        seed=7,
        duration_sec=2.0,
        eval_episodes=3,
        artifact=ArtifactConfig(output_root=str(output_root), record_video=True),
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
        points = [(map_x(float(x_value)), map_y(float(y_value))) for x_value, y_value in zip(x_values, values)]
        draw.line(points, fill=_color(color), width=4, joint="curve")

    draw.text((plot_left, 26), title, font=title_font, fill=(25, 25, 25))
    draw.text((width // 2 - 64, height - 52), x_label, font=axis_font, fill=(35, 35, 35))
    draw.text((24, plot_top - 44), y_label, font=axis_font, fill=(35, 35, 35))

    legend_x = plot_left
    legend_y = height - 88
    for label, _, color in series:
        draw.rounded_rectangle((legend_x, legend_y + 4, legend_x + 26, legend_y + 18), radius=4, fill=_color(color))
        draw.text((legend_x + 34, legend_y - 2), label, font=legend_font, fill=(40, 40, 40))
        legend_x += 250

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


def _disable_td_for_axis(controller, axis: str) -> None:
    native_controller = getattr(controller, "controller", None)
    if native_controller is None:
        return
    channel = getattr(native_controller, f"con_{axis.upper()}", None)
    if channel is None:
        return
    channel.cfg.use_td = False
    if hasattr(channel, "td") and hasattr(channel.td, "cfg"):
        channel.td.cfg.use_td = False


def _compute_risk_flags(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "end_vx_mean": 0.0,
            "negative_tail_velocity": 0.0,
            "max_tail_pitch_abs": 0.0,
            "tail_rpm_edge_fraction": 0.0,
            "tail_same_rpm_fraction": 0.0,
            "unstable_tail": 1.0,
        }
    tail = rows[-8:]
    tail_vx = np.asarray([float(row["vx"]) for row in tail], dtype=np.float64)
    tail_pitch = np.asarray([abs(float(row["pitch"])) for row in tail], dtype=np.float64)
    tail_rpm = np.asarray(
        [[float(row["rpm0"]), float(row["rpm1"]), float(row["rpm2"]), float(row["rpm3"])] for row in tail],
        dtype=np.float64,
    )
    tail_mean_rpm = np.mean(tail_rpm, axis=1)
    edge_mask = np.logical_or(np.min(tail_rpm, axis=1) <= 10000.0, np.max(tail_rpm, axis=1) >= 19000.0)
    same_rpm_mask = np.max(np.abs(tail_rpm - tail_mean_rpm[:, None]), axis=1) <= 5.0
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
    }


def _custom_score(metrics: dict[str, float]) -> float:
    score = (
        0.56 * float(metrics["rmse"])
        + 0.18 * float(metrics["mae"])
        + 0.18 * float(metrics["velocity_rmse"])
        + 0.08 * float(metrics["steady_state_error"])
    )
    if float(metrics.get("negative_tail_velocity", 0.0)) > 0.5:
        score += 8.0
    if float(metrics.get("max_tail_pitch_abs", 0.0)) > 0.8:
        score += 5.0
    if float(metrics.get("tail_rpm_edge_fraction", 0.0)) > 0.5:
        score += 5.0
    if float(metrics.get("tail_same_rpm_fraction", 0.0)) > 0.5:
        score += 5.0
    if float(metrics.get("unstable_tail", 0.0)) > 0.5:
        score += 25.0
    return float(score)


def _apply_candidate(controller, axis: str, candidate: Candidate) -> None:
    controller.set_axis_parameters(
        axis,
        b0=float(candidate.b0),
        omega_c=float(candidate.omega_c),
        omega_o=float(candidate.omega_o),
    )
    _disable_td_for_axis(controller, axis)


def _evaluate_ladrc_candidate(
    cfg: PyBulletControlExperimentConfig,
    axis: str,
    candidate: Candidate,
    *,
    episodes: int,
) -> dict[str, float]:
    episode_rows: list[dict[str, float]] = []
    episode_metrics: list[dict[str, float]] = []
    for episode in range(episodes):
        eval_cfg = replace(
            cfg,
            seed=int(cfg.seed + episode),
            artifact=replace(
                cfg.artifact,
                record_video=False,
                video_output_dir=None,
            ),
        )
        episode_reference = build_xyz_reference_trajectory(
            eval_cfg.axis_config(axis),
            eval_cfg,
            rng=np.random.default_rng(eval_cfg.seed),
        )
        controller = create_controller_bundle("ladrc_x_pos_pid_att")
        _apply_candidate(controller, axis, candidate)
        result = run_controller_episode(eval_cfg, controller, episode_reference)
        rows = list(result["timeseries"])
        metrics = dict(result["metrics"])
        metrics.update(_compute_risk_flags(rows))
        episode_rows.append(rows)
        episode_metrics.append(metrics)
    averaged: dict[str, float] = {}
    for key in episode_metrics[0].keys():
        sample = episode_metrics[0][key]
        if isinstance(sample, (int, float, np.floating)):
            averaged[key] = float(np.mean([float(row[key]) for row in episode_metrics]))
        else:
            averaged[key] = sample
    averaged["name"] = candidate.name
    averaged["b0"] = float(candidate.b0)
    averaged["omega_c"] = float(candidate.omega_c)
    averaged["omega_o"] = float(candidate.omega_o)
    averaged["k"] = float(candidate.k)
    averaged["score"] = _custom_score(averaged)
    return averaged


def _build_coarse_candidates() -> list[Candidate]:
    b0_values = (18.0, 24.0, 29.75, 35.0, 40.0, 45.0)
    omega_c_values = (3.5, 4.5, 5.5, 6.6, 7.5, 9.0)
    omega_o_values = (18.0, 24.0, 31.68, 36.0, 45.0)
    candidates: list[Candidate] = [BASELINE_CANDIDATE]
    seen = {(round(BASELINE_CANDIDATE.b0, 6), round(BASELINE_CANDIDATE.omega_c, 6), round(BASELINE_CANDIDATE.omega_o, 6))}
    for b0 in b0_values:
        for omega_c in omega_c_values:
            for omega_o in omega_o_values:
                key = (round(float(b0), 6), round(float(omega_c), 6), round(float(omega_o), 6))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    Candidate(
                        name=f"coarse_b{b0:g}_wc{omega_c:g}_wo{omega_o:g}",
                        b0=float(b0),
                        omega_c=float(omega_c),
                        omega_o=float(omega_o),
                    )
                )
    return candidates


def _build_refine_candidates(seed_rows: list[dict[str, float]]) -> list[Candidate]:
    b0_scales = (0.90, 0.97, 1.00, 1.05, 1.12)
    wc_scales = (0.88, 0.95, 1.00, 1.06, 1.12)
    wo_scales = (0.85, 0.93, 1.00, 1.08, 1.16)
    candidates: list[Candidate] = []
    seen: set[tuple[float, float, float]] = set()
    for row in seed_rows:
        for b0_scale in b0_scales:
            for wc_scale in wc_scales:
                for wo_scale in wo_scales:
                    candidate = Candidate(
                        name=f"refine_{row['name']}_b{b0_scale:.2f}_wc{wc_scale:.2f}_wo{wo_scale:.2f}",
                        b0=max(float(row["b0"]) * b0_scale, 0.1),
                        omega_c=max(float(row["omega_c"]) * wc_scale, 0.1),
                        omega_o=max(float(row["omega_o"]) * wo_scale, 0.1),
                    )
                    key = (round(candidate.b0, 6), round(candidate.omega_c, 6), round(candidate.omega_o, 6))
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
        rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0
        mae = float(np.mean(np.abs(errors))) if errors else 0.0
        velocity_rmse = (
            float(np.sqrt(np.mean([(float(row["target_vx"]) - float(row["vx"])) ** 2 for row in chunk])))
            if errors
            else 0.0
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


def _plot_all(
    output_root: Path,
    pid_rows: list[dict[str, float]],
    default_rows: list[dict[str, float]],
    tuned_rows: list[dict[str, float]],
) -> list[Path]:
    figures_dir = ensure_dir(output_root / "figures")
    times = np.asarray([float(row["time"]) for row in pid_rows], dtype=np.float64)
    reference = np.asarray([float(row["target_x"]) for row in pid_rows], dtype=np.float64)
    pid_output = np.asarray([float(row["x"]) for row in pid_rows], dtype=np.float64)
    default_output = np.asarray([float(row["x"]) for row in default_rows], dtype=np.float64)
    tuned_output = np.asarray([float(row["x"]) for row in tuned_rows], dtype=np.float64)
    target_vx = np.asarray([float(row["target_vx"]) for row in pid_rows], dtype=np.float64)
    pid_vx = np.asarray([float(row["vx"]) for row in pid_rows], dtype=np.float64)
    default_vx = np.asarray([float(row["vx"]) for row in default_rows], dtype=np.float64)
    tuned_vx = np.asarray([float(row["vx"]) for row in tuned_rows], dtype=np.float64)

    paths: list[Path] = []
    p1 = figures_dir / "tracking_three_way.png"
    _render_line_chart(
        p1,
        title="No-TD 0.5 m/s Route: Position Tracking",
        x_label="Time (s)",
        y_label="Position (m)",
        x_values=times,
        series=[
            ("Reference", reference, "#222222"),
            ("PID", pid_output, "#d62728"),
            ("LADRC(no TD baseline)", default_output, "#1f77b4"),
            ("LADRC(no TD retuned)", tuned_output, "#2ca02c"),
        ],
    )
    paths.append(p1)

    p2 = figures_dir / "tracking_error_three_way.png"
    _render_line_chart(
        p2,
        title="No-TD 0.5 m/s Route: Position Error",
        x_label="Time (s)",
        y_label="Position Error (m)",
        x_values=times,
        series=[
            ("PID error", reference - pid_output, "#d62728"),
            ("Baseline error", reference - default_output, "#1f77b4"),
            ("Retuned error", reference - tuned_output, "#2ca02c"),
        ],
        zero_line=True,
    )
    paths.append(p2)

    p3 = figures_dir / "velocity_tracking_three_way.png"
    _render_line_chart(
        p3,
        title="No-TD 0.5 m/s Route: Velocity Tracking",
        x_label="Time (s)",
        y_label="Velocity (m/s)",
        x_values=times,
        series=[
            ("Target vx", target_vx, "#222222"),
            ("PID vx", pid_vx, "#d62728"),
            ("Baseline vx", default_vx, "#1f77b4"),
            ("Retuned vx", tuned_vx, "#2ca02c"),
        ],
        zero_line=True,
    )
    paths.append(p3)
    return paths


def main() -> None:
    axis = "x"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = ensure_dir(Path("outputs") / "control_pybullet" / "x_no_td_retune_short_speed" / axis / stamp)
    cfg = _build_config(output_root)
    reference_bundle = build_xyz_reference_trajectory(cfg.axis_config(axis), cfg, rng=np.random.default_rng(cfg.seed))
    segments = summarize_reference_segments(reference_bundle)

    pid_controller = create_controller_bundle("pid_pos_att")
    pid_cfg = replace(cfg, artifact=replace(cfg.artifact, video_output_dir=str(output_root / "videos" / "pid")))
    pid_result = run_controller_episode(pid_cfg, pid_controller, reference_bundle)
    pid_metrics = dict(pid_result["metrics"])
    pid_metrics.update(_compute_risk_flags(list(pid_result["timeseries"])))

    coarse_candidates = _build_coarse_candidates()
    coarse_rows = [_evaluate_ladrc_candidate(cfg, axis, candidate, episodes=1) for candidate in coarse_candidates]
    coarse_rows = sorted(coarse_rows, key=lambda row: float(row["score"]))

    refine_seeds = coarse_rows[:5]
    refine_candidates = _build_refine_candidates(refine_seeds)
    refine_rows = [_evaluate_ladrc_candidate(cfg, axis, candidate, episodes=2) for candidate in refine_candidates]
    refine_rows = sorted(refine_rows, key=lambda row: float(row["score"]))

    best_row = min(refine_rows + coarse_rows, key=lambda row: float(row["score"]))
    tuned_candidate = Candidate(
        name="retuned_no_td_short_speed",
        b0=float(best_row["b0"]),
        omega_c=float(best_row["omega_c"]),
        omega_o=float(best_row["omega_o"]),
    )

    default_controller = create_controller_bundle("ladrc_x_pos_pid_att")
    _apply_candidate(default_controller, axis, BASELINE_CANDIDATE)
    tuned_controller = create_controller_bundle("ladrc_x_pos_pid_att")
    _apply_candidate(tuned_controller, axis, tuned_candidate)

    default_cfg = replace(cfg, artifact=replace(cfg.artifact, video_output_dir=str(output_root / "videos" / "ladrc_default_no_td")))
    tuned_cfg = replace(cfg, artifact=replace(cfg.artifact, video_output_dir=str(output_root / "videos" / "ladrc_retuned_no_td")))
    default_result = run_controller_episode(default_cfg, default_controller, reference_bundle)
    tuned_result = run_controller_episode(tuned_cfg, tuned_controller, reference_bundle)

    default_metrics = dict(default_result["metrics"])
    default_metrics.update(_compute_risk_flags(list(default_result["timeseries"])))
    tuned_metrics = dict(tuned_result["metrics"])
    tuned_metrics.update(_compute_risk_flags(list(tuned_result["timeseries"])))

    pid_score = _custom_score(pid_metrics)
    default_score = _custom_score(default_metrics)
    tuned_score = _custom_score(tuned_metrics)

    write_reference_csv(output_root / "reference.csv", reference_bundle)
    write_timeseries_csv(output_root / "pid_timeseries.csv", list(pid_result["timeseries"]))
    write_timeseries_csv(output_root / "default_no_td_timeseries.csv", list(default_result["timeseries"]))
    write_timeseries_csv(output_root / "retuned_no_td_timeseries.csv", list(tuned_result["timeseries"]))
    write_metrics_csv(
        output_root / "metrics.csv",
        [
            {"controller": "pid_pos_att", **pid_metrics, "score": pid_score},
            {"controller": "ladrc_x_pos_pid_att_no_td_baseline", **default_metrics, "score": default_score},
            {"controller": "ladrc_x_pos_pid_att_no_td_retuned", **tuned_metrics, "score": tuned_score},
        ],
    )
    write_metrics_csv(output_root / "candidate_coarse.csv", coarse_rows)
    write_metrics_csv(output_root / "candidate_refine.csv", refine_rows)

    pid_stage_metrics = _compute_stage_metrics(list(pid_result["timeseries"]), segments)
    default_stage_metrics = _compute_stage_metrics(list(default_result["timeseries"]), segments)
    tuned_stage_metrics = _compute_stage_metrics(list(tuned_result["timeseries"]), segments)
    write_metrics_csv(
        output_root / "stage_metrics.csv",
        [{"controller": "pid_pos_att", **row} for row in pid_stage_metrics]
        + [{"controller": "ladrc_x_pos_pid_att_no_td_baseline", **row} for row in default_stage_metrics]
        + [{"controller": "ladrc_x_pos_pid_att_no_td_retuned", **row} for row in tuned_stage_metrics],
    )

    figure_paths = _plot_all(
        output_root,
        list(pid_result["timeseries"]),
        list(default_result["timeseries"]),
        list(tuned_result["timeseries"]),
    )
    pid_gif = _ensure_video(output_root, "pid", list(pid_result["timeseries"]), "PID No-TD 0.5 m/s Tracking")
    default_gif = _ensure_video(output_root, "ladrc_default_no_td", list(default_result["timeseries"]), "Baseline No-TD LADRC 0.5 m/s Tracking")
    tuned_gif = _ensure_video(output_root, "ladrc_retuned_no_td", list(tuned_result["timeseries"]), "Retuned No-TD LADRC 0.5 m/s Tracking")

    tuned_beats_default = bool(
        float(tuned_metrics["rmse"]) < float(default_metrics["rmse"])
        and float(tuned_metrics["velocity_rmse"]) < float(default_metrics["velocity_rmse"])
    )
    tuned_beats_pid = bool(float(tuned_metrics["rmse"]) < float(pid_metrics["rmse"]))

    summary = {
        "axis": axis,
        "backend": pid_result["backend"],
        "scenario_name": "x_no_td_retune_short_speed",
        "scenario_definition": {
            "control_freq_hz": int(cfg.control_freq_hz),
            "duration_sec": float(cfg.duration_sec),
            "current_tuning_speed_mps": 0.5,
            "current_tuning_duration_sec": 2.0,
            "use_td": False,
            "reference_route": [
                {"duration_sec": 2.0, "velocity_mps": 0.5, "description": "single_forward_segment"},
            ],
            "fixed_stage_lengths": list(cfg.axis_config(axis).fixed_stage_lengths or ()),
            "fixed_stage_velocities": list(cfg.axis_config(axis).fixed_stage_velocities or ()),
        },
        "reference_segments": segments,
        "controllers": {
            "pid_pos_att": {**pid_metrics, "score": pid_score},
            "ladrc_x_pos_pid_att_no_td_baseline": {**default_metrics, "score": default_score},
            "ladrc_x_pos_pid_att_no_td_retuned": {**tuned_metrics, "score": tuned_score},
        },
        "stage_metrics": {
            "pid_pos_att": pid_stage_metrics,
            "ladrc_x_pos_pid_att_no_td_baseline": default_stage_metrics,
            "ladrc_x_pos_pid_att_no_td_retuned": tuned_stage_metrics,
        },
        "tuning": {
            "search_dimensions": ["b0", "omega_c", "omega_o"],
            "baseline_candidate": asdict(BASELINE_CANDIDATE) | {"k": float(BASELINE_CANDIDATE.k), "use_td": False},
            "retuned_candidate": asdict(tuned_candidate) | {"k": float(tuned_candidate.k), "use_td": False},
            "top_coarse_candidates": coarse_rows[:10],
            "top_refine_candidates": refine_rows[:10],
        },
        "acceptance": {
            "tuned_beats_default": tuned_beats_default,
            "tuned_beats_pid": tuned_beats_pid,
            "pid_rmse": float(pid_metrics["rmse"]),
            "baseline_rmse": float(default_metrics["rmse"]),
            "retuned_rmse": float(tuned_metrics["rmse"]),
            "pid_velocity_rmse": float(pid_metrics["velocity_rmse"]),
            "baseline_velocity_rmse": float(default_metrics["velocity_rmse"]),
            "retuned_velocity_rmse": float(tuned_metrics["velocity_rmse"]),
        },
        "figures": [str(path) for path in figure_paths],
        "video_dirs": {
            "pid_pos_att": str(output_root / "videos" / "pid"),
            "ladrc_x_pos_pid_att_no_td_baseline": str(output_root / "videos" / "ladrc_default_no_td"),
            "ladrc_x_pos_pid_att_no_td_retuned": str(output_root / "videos" / "ladrc_retuned_no_td"),
        },
        "videos": {
            "pid_pos_att": str(pid_gif),
            "ladrc_x_pos_pid_att_no_td_baseline": str(default_gif),
            "ladrc_x_pos_pid_att_no_td_retuned": str(tuned_gif),
        },
    }
    write_summary_json(output_root / "summary.json", summary)
    (output_root / "summary_readable.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "backend": pid_result["backend"],
                "retuned_candidate": asdict(tuned_candidate) | {"k": float(tuned_candidate.k), "use_td": False},
                "tuned_beats_default": tuned_beats_default,
                "tuned_beats_pid": tuned_beats_pid,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
