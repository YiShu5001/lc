from __future__ import annotations

from pathlib import Path

from lc.common.io import ensure_dir


def plot_control_comparison(results: dict[str, dict[str, float]], output_dir: str | Path) -> list[Path]:
    """绘制第三章四组方法的核心指标对比图。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []
    for metric in [
        "mae",
        "rmse",
        "iae",
        "overshoot",
        "settling_time",
        "steady_state_error",
        "control_energy",
        "control_variation",
        "reward",
    ]:
        labels = list(results.keys())
        values = [results[label][metric] for label in labels]
        out = _save_bar_chart(output_dir, metric, labels, values, "#4f81bd")
        artifacts.append(out)
    return artifacts


def plot_control_training_curves(training_logs: dict[str, list[dict[str, float]]], output_dir: str | Path) -> list[Path]:
    """绘制 DDPG-LADRC 与 mDDPG-LADRC 的训练曲线。"""
    target = ensure_dir(output_dir)
    artifacts: list[Path] = []
    for metric in ("reward", "mae", "rmse", "steady_state_error", "actor_loss", "critic_loss"):
        out = target / f"{metric}_curve.svg"
        series = {name: [row[metric] for row in rows] for name, rows in training_logs.items()}
        _write_svg_line_chart(out, f"Training curve: {metric}", series)
        artifacts.append(out)
    return artifacts


def plot_control_ablation(results: dict[str, dict[str, float]], output_dir: str | Path) -> Path:
    """绘制 DDPG-LADRC 与 mDDPG-LADRC 的主消融对比图。"""
    target = ensure_dir(output_dir)
    labels = ["ddpg_ladrc", "mddpg_ladrc"]
    values = [results[label]["iae"] for label in labels]
    out = target / "ddpg_vs_mddpg_iae.svg"
    _write_svg_bar_chart(out, "Ablation: DDPG-LADRC vs mDDPG-LADRC (IAE)", labels, values, "#d95f02")
    return out


def plot_control_mechanism_ablation(results: dict[str, dict[str, float]], output_dir: str | Path) -> Path:
    """绘制三个增强机制去除后的消融对比图。"""
    target = ensure_dir(output_dir)
    labels = list(results.keys())
    values = [results[label]["iae"] for label in labels]
    out = target / "mechanism_ablation_iae.svg"
    _write_svg_bar_chart(out, "Ablation: remove enhancement mechanisms (IAE)", labels, values, "#7570b3")
    return out


def plot_time_response(trajectories: dict[str, dict[str, list[float]]], output_dir: str | Path) -> list[Path]:
    """绘制第三章典型时域响应图。"""
    target = ensure_dir(output_dir)
    artifacts: list[Path] = []
    for metric in ("output", "error", "velocity_error", "control", "disturbance"):
        out = target / f"time_response_{metric}.svg"
        series = {name: values[metric] for name, values in trajectories.items()}
        if metric == "output" and trajectories:
            first = next(iter(trajectories.values()))
            series = {"reference": first["reference"], **series}
        _write_svg_line_chart(out, f"Time response: {metric}", series)
        artifacts.append(out)
    return artifacts


def plot_control_generalization(results: dict[str, dict[str, dict[str, float]]], output_dir: str | Path) -> list[Path]:
    """Plot per-difficulty generalization curves for chapter 3 control methods."""
    target = ensure_dir(output_dir)
    artifacts: list[Path] = []
    difficulties = list(results.keys())
    if not difficulties:
        return artifacts
    methods = list(next(iter(results.values())).keys())
    for metric in ("iae", "rmse", "steady_state_error", "control_energy", "reward"):
        out = target / f"generalization_{metric}.svg"
        series = {method: [results[difficulty][method][metric] for difficulty in difficulties] for method in methods}
        _write_svg_line_chart(out, f"Generalization: {metric}", series, x_labels=difficulties)
        artifacts.append(out)
    return artifacts


def plot_mddpg_shared_value_sweep(
    sweep_rows: list[dict[str, float | int]],
    output_dir: str | Path,
) -> list[Path]:
    """Plot x-axis mDDPG shared-value sweep curves for chapter-3 RLcontrolRefLine experiments."""
    target = ensure_dir(output_dir)
    if not sweep_rows:
        return []
    sorted_rows = sorted(sweep_rows, key=lambda row: int(row["shared_value"]))
    shared_values = [str(int(row["shared_value"])) for row in sorted_rows]
    artifacts: list[Path] = []
    for metric in ("reward", "rmse", "steady_state_error", "iae"):
        out = target / f"mddpg_shared_value_{metric}.svg"
        series = {
            metric: [float(row[metric]) for row in sorted_rows],
        }
        _write_svg_line_chart(out, f"mDDPG shared-v sweep: {metric}", series, x_labels=shared_values)
        artifacts.append(out)
    return artifacts


def _save_bar_chart(output_dir: Path, metric: str, labels: list[str], values: list[float], color: str) -> Path:
    out = output_dir / f"{metric}.svg"
    _write_svg_bar_chart(out, f"Control comparison: {metric}", labels, values, color)
    return out


def _write_svg_bar_chart(path: Path, title: str, labels: list[str], values: list[float], color: str) -> None:
    width = 640
    height = 420
    margin = 60
    max_value = max(max(values), 1e-6)
    bar_width = max(40, int((width - 2 * margin) / max(len(values), 1) * 0.6))
    gap = bar_width // 2
    bars = []
    texts = [f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="18">{title}</text>']
    for index, (label, value) in enumerate(zip(labels, values)):
        x = margin + index * (bar_width + gap)
        bar_height = int((height - 2 * margin) * (value / max_value if max_value else 0.0))
        y = height - margin - bar_height
        bars.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{color}" />')
        texts.append(f'<text x="{x + bar_width / 2}" y="{height - margin + 20}" text-anchor="middle" font-size="12">{label}</text>')
        texts.append(f'<text x="{x + bar_width / 2}" y="{max(y - 8, 40)}" text-anchor="middle" font-size="12">{value:.3f}</text>')
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white" />',
            f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333" />',
            f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333" />',
            *bars,
            *texts,
            "</svg>",
        ]
    )
    path.write_text(svg, encoding="utf-8")


def _write_svg_line_chart(
    path: Path,
    title: str,
    series: dict[str, list[float]],
    x_labels: list[str] | None = None,
) -> None:
    width = 720
    height = 420
    margin = 60
    all_values = [value for rows in series.values() for value in rows]
    min_value = min(all_values) if all_values else 0.0
    max_value = max(all_values) if all_values else 1.0
    if abs(max_value - min_value) < 1e-6:
        max_value = min_value + 1.0
    colors = ["#4f81bd", "#c0504d", "#9bbb59", "#8064a2", "#f0ad4e"]
    items: list[str] = []
    legend: list[str] = [f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="18">{title}</text>']
    for idx, (name, values) in enumerate(series.items()):
        points = []
        count = max(len(values) - 1, 1)
        for index, value in enumerate(values):
            x = margin + (width - 2 * margin) * (index / count)
            y = height - margin - (height - 2 * margin) * ((value - min_value) / (max_value - min_value))
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[idx % len(colors)]
        items.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}" />')
        legend.append(f'<text x="{width - 180}" y="{50 + idx * 18}" font-size="12" fill="{color}">{name}</text>')
    axis_labels: list[str] = []
    if x_labels:
        count = max(len(x_labels) - 1, 1)
        for index, label in enumerate(x_labels):
            x = margin + (width - 2 * margin) * (index / count)
            axis_labels.append(
                f'<text x="{x:.1f}" y="{height - margin + 20}" text-anchor="middle" font-size="12">{label}</text>'
            )
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white" />',
            f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333" />',
            f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333" />',
            *items,
            *legend,
            *axis_labels,
            "</svg>",
        ]
    )
    path.write_text(svg, encoding="utf-8")
