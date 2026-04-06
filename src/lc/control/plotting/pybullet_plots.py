from __future__ import annotations

from pathlib import Path
from typing import Iterable

from lc.common.io import ensure_dir


def plot_axis_tracking(run_rows: list[dict[str, float]], output_dir: str | Path) -> Path:
    return _line_plot(
        run_rows,
        output_dir,
        "axis_tracking.png",
        "Position Tracking",
        [("x", "target_x"), ("y", "target_y"), ("z", "target_z")],
    )


def plot_axis_velocity(run_rows: list[dict[str, float]], output_dir: str | Path) -> Path:
    return _line_plot(
        run_rows,
        output_dir,
        "axis_velocity.png",
        "Velocity Tracking",
        [("vx", "target_vx"), ("vy", "target_vy"), ("vz", "target_vz")],
    )


def plot_axis_error(run_rows: list[dict[str, float]], output_dir: str | Path) -> Path:
    rows = []
    for row in run_rows:
        rows.append(
            {
                "time": row["time"],
                "pos_error_x": row["target_x"] - row["x"],
                "pos_error_y": row["target_y"] - row["y"],
                "pos_error_z": row["target_z"] - row["z"],
                "vel_error_x": row["target_vx"] - row["vx"],
                "vel_error_y": row["target_vy"] - row["vy"],
                "vel_error_z": row["target_vz"] - row["vz"],
            }
        )
    return _line_plot(
        rows,
        output_dir,
        "axis_error.png",
        "Tracking Error",
        [("pos_error_x", None), ("pos_error_y", None), ("pos_error_z", None)],
    )


def plot_attitude_response(run_rows: list[dict[str, float]], output_dir: str | Path) -> Path:
    return _line_plot(
        run_rows,
        output_dir,
        "attitude_response.png",
        "Attitude Response",
        [("roll", None), ("pitch", None), ("yaw", None)],
    )


def plot_control_effort(run_rows: list[dict[str, float]], output_dir: str | Path) -> Path:
    return _line_plot(
        run_rows,
        output_dir,
        "control_effort.png",
        "Motor RPM",
        [("rpm0", None), ("rpm1", None), ("rpm2", None), ("rpm3", None)],
    )


def plot_controller_comparison(metric_rows: list[dict[str, float]], output_dir: str | Path) -> Path:
    target = ensure_dir(output_dir)
    labels = [str(row["controller"]) for row in metric_rows]
    values = [float(row["rmse"]) for row in metric_rows]
    return _bar_plot(target / "controller_comparison.png", "Controller Comparison (RMSE)", labels, values)


def plot_training_curves(history_rows: list[dict[str, float]], output_dir: str | Path) -> Path:
    return _line_plot(
        history_rows,
        output_dir,
        "training_curves.png",
        "Training Reward",
        [("reward", None)],
        x_key="episode",
    )


def plot_pid_vs_best_ladrc_response(
    pid_rows: list[dict[str, float]],
    ladrc_rows: list[dict[str, float]],
    axis: str,
    output_dir: str | Path,
) -> Path:
    left = _project_axis_timeseries(pid_rows, axis, "pid")
    right = _project_axis_timeseries(ladrc_rows, axis, "ladrc")
    rows = []
    for pid_row, ladrc_row in zip(left, right):
        rows.append(
            {
                "time": pid_row["time"],
                "reference": pid_row["reference"],
                "pid_output": pid_row["output"],
                "ladrc_output": ladrc_row["output"],
            }
        )
    return _line_plot(
        rows,
        output_dir,
        "pid_vs_best_ladrc_time_response.png",
        f"PID vs Best LADRC ({axis})",
        [("reference", None), ("pid_output", None), ("ladrc_output", None)],
    )


def plot_metric_heatmap(
    rows: list[dict[str, float]],
    x_key: str,
    y_key: str,
    value_key: str,
    output_dir: str | Path,
    filename: str,
    title: str,
) -> Path:
    target = ensure_dir(output_dir)
    x_values = sorted({float(row[x_key]) for row in rows})
    y_values = sorted({float(row[y_key]) for row in rows})
    if not x_values or not y_values:
        return _bar_plot(target / filename, title, ["empty"], [0.0])
    width = 720
    height = 480
    margin = 60
    cell_w = (width - 2 * margin) / max(len(x_values), 1)
    cell_h = (height - 2 * margin) / max(len(y_values), 1)
    values = [float(row[value_key]) for row in rows]
    min_v = min(values)
    max_v = max(values)
    if abs(max_v - min_v) < 1e-6:
        max_v = min_v + 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2}" y="26" text-anchor="middle" font-size="18">{title}</text>',
    ]
    for row in rows:
        x = x_values.index(float(row[x_key]))
        y = y_values.index(float(row[y_key]))
        value = float(row[value_key])
        color_ratio = (value - min_v) / (max_v - min_v)
        red = int(235 * color_ratio)
        green = int(190 * (1.0 - color_ratio))
        blue = 210
        rect_x = margin + x * cell_w
        rect_y = margin + (len(y_values) - 1 - y) * cell_h
        parts.append(
            f'<rect x="{rect_x:.1f}" y="{rect_y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="rgb({red},{green},{blue})" stroke="#ffffff" />'
        )
        parts.append(
            f'<text x="{rect_x + cell_w / 2:.1f}" y="{rect_y + cell_h / 2:.1f}" text-anchor="middle" font-size="11">{value:.3f}</text>'
        )
    for idx, value in enumerate(x_values):
        x = margin + idx * cell_w + cell_w / 2
        parts.append(f'<text x="{x:.1f}" y="{height - 18}" text-anchor="middle" font-size="11">{value:.2f}</text>')
    for idx, value in enumerate(reversed(y_values)):
        y = margin + idx * cell_h + cell_h / 2
        parts.append(f'<text x="24" y="{y:.1f}" text-anchor="middle" font-size="11">{value:.2f}</text>')
    parts.append("</svg>")
    out = target / filename
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def plot_single_factor_sensitivity(
    rows: list[dict[str, float]],
    factor_key: str,
    metric_key: str,
    output_dir: str | Path,
    filename: str,
    title: str,
) -> Path:
    projected = [{factor_key: float(row[factor_key]), metric_key: float(row[metric_key])} for row in rows]
    return _line_plot(projected, output_dir, filename, title, [(metric_key, None)], x_key=factor_key)


def _line_plot(
    rows: list[dict[str, float]],
    output_dir: str | Path,
    filename: str,
    title: str,
    series: list[tuple[str, str | None]],
    x_key: str = "time",
) -> Path:
    target = ensure_dir(output_dir)
    width = 720
    height = 420
    margin = 50
    x_values = [float(row[x_key]) for row in rows] if rows else [0.0]
    all_values = []
    for left, right in series:
        all_values.extend(float(row[left]) for row in rows)
        if right:
            all_values.extend(float(row[right]) for row in rows)
    min_y = min(all_values) if all_values else -1.0
    max_y = max(all_values) if all_values else 1.0
    if abs(max_y - min_y) < 1e-6:
        max_y = min_y + 1.0
    min_x = min(x_values)
    max_x = max(x_values)
    if abs(max_x - min_x) < 1e-6:
        max_x = min_x + 1.0
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="18">{title}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333" />',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333" />',
    ]
    legend_y = 42
    for index, (left, right) in enumerate(series):
        color = colors[index % len(colors)]
        points = []
        for row in rows:
            x = margin + (width - 2 * margin) * ((float(row[x_key]) - min_x) / (max_x - min_x))
            y = height - margin - (height - 2 * margin) * ((float(row[left]) - min_y) / (max_y - min_y))
            points.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}" />')
        parts.append(f'<text x="{width - 160}" y="{legend_y}" font-size="12" fill="{color}">{left}</text>')
        legend_y += 16
        if right:
            dashed = []
            for row in rows:
                x = margin + (width - 2 * margin) * ((float(row[x_key]) - min_x) / (max_x - min_x))
                y = height - margin - (height - 2 * margin) * ((float(row[right]) - min_y) / (max_y - min_y))
                dashed.append(f"{x:.1f},{y:.1f}")
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-dasharray="6,4" stroke-width="1.5" points="{" ".join(dashed)}" />'
            )
            parts.append(f'<text x="{width - 160}" y="{legend_y}" font-size="12" fill="{color}">{right}</text>')
            legend_y += 16
    parts.append("</svg>")
    out = target / filename
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def _bar_plot(path: Path, title: str, labels: list[str], values: list[float]) -> Path:
    width = 720
    height = 420
    margin = 50
    max_value = max(values) if values else 1.0
    if max_value <= 0:
        max_value = 1.0
    bar_width = max(int((width - 2 * margin) / max(len(values), 1) * 0.55), 24)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="18">{title}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333" />',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333" />',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        x = margin + index * (bar_width + 30)
        bar_height = int((height - 2 * margin) * (value / max_value))
        y = height - margin - bar_height
        parts.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="#4f81bd" />')
        parts.append(f'<text x="{x + bar_width / 2}" y="{height - margin + 18}" font-size="12" text-anchor="middle">{label}</text>')
        parts.append(f'<text x="{x + bar_width / 2}" y="{y - 6}" font-size="12" text-anchor="middle">{value:.3f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _project_axis_timeseries(rows: list[dict[str, float]], axis: str, label: str) -> list[dict[str, float]]:
    target_key = f"target_{axis}"
    output_key = axis
    return [
        {
            "time": float(row["time"]),
            "reference": float(row[target_key]),
            "output": float(row[output_key]),
            "label": label,
        }
        for row in rows
    ]
