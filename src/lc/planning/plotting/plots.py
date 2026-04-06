from __future__ import annotations

from pathlib import Path

from lc.common.io import ensure_dir


def plot_planning_comparison(
    results: dict[str, dict[str, float]],
    output_dir: Path,
    histories: dict[str, list[dict[str, float | int | str]]] | None = None,
    stage_history: list[dict[str, float | int | str]] | None = None,
    complexity_results: dict[str, dict[str, float]] | None = None,
    trajectory: dict[str, list[float] | list[list[float]]] | None = None,
    attention_proxy: dict[str, list[float]] | None = None,
    replay_stats: dict[str, object] | None = None,
) -> list[Path]:
    target = ensure_dir(output_dir)
    figures = [
        _save_bar_chart(
            target,
            "ablation_comparison",
            list(results.keys()),
            [values.get("reward", 0.0) for values in results.values()],
            "#2c7fb8",
            "Ablation reward comparison",
        )
    ]
    if histories and "task_decomposed" in histories:
        primary_history = histories["task_decomposed"]
        figures.append(_save_line_chart(target, "convergence_curve", primary_history, ("reward",), "#1d91c0"))
        figures.append(
            _save_multi_line_chart(
                target,
                "success_collision_curve",
                primary_history,
                ("success_rate", "collision_rate"),
                ("#2ca25f", "#de2d26"),
            )
        )
        figures.append(
            _save_multi_line_chart(
                target,
                "formation_occupancy_curve",
                primary_history,
                ("formation_error", "occupancy_error"),
                ("#7a0177", "#3182bd"),
            )
        )
        figures.append(
            _save_multi_line_chart(
                target,
                "reward_components_curve",
                primary_history,
                ("target_reward", "avoidance_reward", "collaboration_reward", "recovery_reward"),
                ("#1b9e77", "#d95f02", "#7570b3", "#66a61e"),
            )
        )
    if stage_history:
        figures.append(_save_stage_chart(target, "curriculum_schedule", stage_history))
        figures.append(_save_stage_retention_chart(target, "stage_retention_curve", stage_history))
    if complexity_results:
        figures.append(
            _save_bar_chart(
                target,
                "complexity_generalization",
                list(complexity_results.keys()),
                [metrics.get("reward", 0.0) for metrics in complexity_results.values()],
                "#dd8452",
                "Complexity generalization reward",
            )
        )
    if trajectory:
        figures.append(_save_trajectory_chart(target, "trajectory", trajectory))
    if attention_proxy:
        figures.append(_save_heatmap(target, "attention_heatmap", attention_proxy))
    if replay_stats:
        figures.append(_save_replay_distribution(target, "replay_distribution", replay_stats))
        figures.append(_save_old_pool_mix_chart(target, "old_pool_mix_curve", replay_stats))
    figures.extend(_save_experiment_subset_charts(target, results))
    return figures


def _save_bar_chart(
    output_dir: Path,
    metric: str,
    labels: list[str],
    values: list[float],
    color: str,
    title: str,
) -> Path:
    figure_path = output_dir / f"{metric}.svg"
    _write_svg_bar_chart(figure_path, title, labels, values, color)
    return figure_path


def _save_line_chart(output_dir: Path, metric: str, rows: list[dict[str, float | int | str]], fields: tuple[str, ...], color: str) -> Path:
    return _save_multi_line_chart(output_dir, metric, rows, fields, (color,))


def _save_multi_line_chart(
    output_dir: Path,
    metric: str,
    rows: list[dict[str, float | int | str]],
    fields: tuple[str, ...],
    colors: tuple[str, ...],
) -> Path:
    figure_path = output_dir / f"{metric}.svg"
    width = 680
    height = 420
    margin = 50
    values = [float(row.get(field, 0.0)) for field in fields for row in rows]
    min_value = min(values) if values else 0.0
    max_value = max(values) if values else 1.0
    span = max(max_value - min_value, 1e-6)
    polylines = []
    legend = []
    for index, field in enumerate(fields):
        points = []
        color = colors[min(index, len(colors) - 1)]
        for row_index, row in enumerate(rows):
            x = margin + (width - 2 * margin) * row_index / max(len(rows) - 1, 1)
            value = float(row.get(field, 0.0))
            y = height - margin - (height - 2 * margin) * ((value - min_value) / span)
            points.append(f"{x:.1f},{y:.1f}")
        polylines.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(points)}" />')
        legend.append(
            f'<text x="{width - margin - 120}" y="{30 + index * 18}" font-size="12" fill="{color}">{field}</text>'
        )
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white" />',
            f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333" />',
            f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333" />',
            f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="18">{metric.replace("_", " ").title()}</text>',
            *polylines,
            *legend,
            "</svg>",
        ]
    )
    figure_path.write_text(svg, encoding="utf-8")
    return figure_path


def _save_stage_chart(output_dir: Path, metric: str, rows: list[dict[str, float | int | str]]) -> Path:
    figure_path = output_dir / f"{metric}.svg"
    width = 680
    height = 300
    margin = 40
    points = []
    for index, row in enumerate(rows):
        x = margin + (width - 2 * margin) * index / max(len(rows) - 1, 1)
        y = height - margin - 60 * float(row.get("new_stage", 0))
        points.append(f"{x:.1f},{y:.1f}")
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white" />',
            f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="18">Curriculum stage transitions</text>',
            f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333" />',
            f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333" />',
            f'<polyline fill="none" stroke="#88419d" stroke-width="3" points="{" ".join(points)}" />',
            "</svg>",
        ]
    )
    figure_path.write_text(svg, encoding="utf-8")
    return figure_path


def _save_trajectory_chart(output_dir: Path, metric: str, trajectory: dict[str, list[float] | list[list[float]]]) -> Path:
    figure_path = output_dir / f"{metric}.svg"
    width = 480
    height = 480
    margin = 40
    points = trajectory.get("trajectory", [])
    coords = []
    for point in points:
        if not isinstance(point, list) or len(point) < 2:
            continue
        x = margin + (width - 2 * margin) * (float(point[0]) + 1.0) / 2.0
        y = height - margin - (height - 2 * margin) * (float(point[1]) + 1.0) / 2.0
        coords.append(f"{x:.1f},{y:.1f}")
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white" />',
            f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="18">Trajectory</text>',
            f'<rect x="{margin}" y="{margin}" width="{width - 2 * margin}" height="{height - 2 * margin}" fill="none" stroke="#333" />',
            f'<polyline fill="none" stroke="#2171b5" stroke-width="3" points="{" ".join(coords)}" />',
            "</svg>",
        ]
    )
    figure_path.write_text(svg, encoding="utf-8")
    return figure_path


def _save_heatmap(output_dir: Path, metric: str, attention_proxy: dict[str, list[float]]) -> Path:
    figure_path = output_dir / f"{metric}.svg"
    rows = [
        ("obstacle", _flatten_heatmap_values(attention_proxy.get("obstacle_attention", [0.0]))),
        ("neighbor", _flatten_heatmap_values(attention_proxy.get("neighbor_attention", [0.0]))),
    ]
    cell_size = 42
    width = 120 + cell_size * max(len(values) for _, values in rows)
    height = 120 + cell_size * len(rows)
    rects = []
    labels = ['<text x="20" y="28" font-size="18">Attention heatmap</text>']
    for row_index, (label, values) in enumerate(rows):
        labels.append(f'<text x="20" y="{90 + row_index * cell_size}" font-size="12">{label}</text>')
        for col_index, value in enumerate(values):
            intensity = max(0, min(255, int(255 * float(value))))
            color = f"rgb({255 - intensity},{240 - intensity // 2},{255})"
            x = 90 + col_index * cell_size
            y = 60 + row_index * cell_size
            rects.append(f'<rect x="{x}" y="{y}" width="{cell_size - 4}" height="{cell_size - 4}" fill="{color}" />')
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white" />',
            *labels,
            *rects,
            "</svg>",
        ]
    )
    figure_path.write_text(svg, encoding="utf-8")
    return figure_path


def _save_replay_distribution(output_dir: Path, metric: str, replay_stats: dict[str, object]) -> Path:
    figure_path = output_dir / f"{metric}.svg"
    labels: list[str] = []
    values: list[float] = []
    guidance = replay_stats.get("guidance", {}) if isinstance(replay_stats, dict) else {}
    if isinstance(guidance, dict):
        labels.extend(["guidance_buffer", "guidance_old"])
        values.extend([float(guidance.get("buffer_size", 0.0)), float(guidance.get("old_pool_size", 0.0))])
    for prefix in ("avoidance", "cooperation"):
        stage_stats = replay_stats.get(prefix, {}) if isinstance(replay_stats, dict) else {}
        if not isinstance(stage_stats, dict):
            continue
        bucket_sizes = stage_stats.get("bucket_sizes", {})
        if isinstance(bucket_sizes, dict):
            for name, size in bucket_sizes.items():
                labels.append(f"{prefix}_{name}")
                values.append(float(size))
        labels.append(f"{prefix}_old")
        values.append(float(stage_stats.get("old_pool_size", 0.0)))
    _write_svg_bar_chart(figure_path, "Replay bucket distribution", labels, values, "#6a51a3")
    return figure_path


def _save_stage_retention_chart(output_dir: Path, metric: str, rows: list[dict[str, float | int | str]]) -> Path:
    counts = {"guidance": 0.0, "avoidance": 0.0, "cooperation": 0.0}
    for row in rows:
        stage_name = str(row.get("new_stage_name", ""))
        if stage_name in counts:
            counts[stage_name] += 1.0
    figure_path = output_dir / f"{metric}.svg"
    _write_svg_bar_chart(figure_path, "Stage retention count", list(counts.keys()), list(counts.values()), "#3182bd")
    return figure_path


def _save_old_pool_mix_chart(output_dir: Path, metric: str, replay_stats: dict[str, object]) -> Path:
    labels = ["guidance_old", "avoidance_old", "cooperation_old"]
    values = [
        float(replay_stats.get("guidance", {}).get("old_pool_hit_rate", 0.0)) if isinstance(replay_stats.get("guidance", {}), dict) else 0.0,
        float(replay_stats.get("avoidance", {}).get("old_pool_hit_rate", 0.0)) if isinstance(replay_stats.get("avoidance", {}), dict) else 0.0,
        float(replay_stats.get("cooperation", {}).get("old_pool_hit_rate", 0.0)) if isinstance(replay_stats.get("cooperation", {}), dict) else 0.0,
    ]
    figure_path = output_dir / f"{metric}.svg"
    _write_svg_bar_chart(figure_path, "Old-pool mix rate", labels, values, "#e6550d")
    return figure_path


def _save_experiment_subset_charts(output_dir: Path, results: dict[str, dict[str, float]]) -> list[Path]:
    figures: list[Path] = []
    priority_methods = [name for name in results if "priority" in name]
    if priority_methods:
        figures.append(
            _save_bar_chart(
                output_dir,
                "priority_mode_comparison",
                priority_methods,
                [results[name].get("reward", 0.0) for name in priority_methods],
                "#756bb1",
                "Priority mode comparison",
            )
        )
    ratio_methods = [name for name in results if "ratio" in name or "old_mix" in name]
    if ratio_methods:
        figures.append(
            _save_bar_chart(
                output_dir,
                "sample_ratio_comparison",
                ratio_methods,
                [results[name].get("reward", 0.0) for name in ratio_methods],
                "#31a354",
                "Sample ratio and old-mix comparison",
            )
        )
    if "task_decomposed" in results:
        reward_fields = [
            "target_reward",
            "avoidance_reward",
            "collaboration_reward",
            "recovery_reward",
            "success_bonus",
        ]
        figures.append(
            _save_bar_chart(
                output_dir,
                "reward_breakdown_bar",
                reward_fields,
                [results["task_decomposed"].get(name, 0.0) for name in reward_fields],
                "#2b8cbe",
                "Task-decomposed reward breakdown",
            )
        )
    return figures


def _flatten_heatmap_values(values: list[float] | list[list[float]]) -> list[float]:
    flattened: list[float] = []
    for value in values:
        if isinstance(value, list):
            flattened.extend(_flatten_heatmap_values(value))
        else:
            flattened.append(float(value))
    return flattened or [0.0]


def _write_svg_bar_chart(path: Path, title: str, labels: list[str], values: list[float], color: str) -> None:
    width = 640
    height = 420
    margin = 60
    max_value = max(max(values), 1e-6)
    bar_width = max(32, int((width - 2 * margin) / max(len(values), 1) * 0.6))
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
