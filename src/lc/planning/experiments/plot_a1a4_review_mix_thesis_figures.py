from __future__ import annotations

import csv
import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUN_DIR = Path(
    "outputs/planning/manual_block_training_shadow_runs/"
    "blocking_success_guard_a1a4_review_mix_1200cap/"
    "blocking_success_guard_a1a4_review_mix"
)


def _as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    result = np.zeros_like(values, dtype=float)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = float(np.mean(values[start : index + 1]))
    return result


def _load_history(history_csv: Path) -> dict[str, np.ndarray]:
    with history_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    episodes = np.array([int(_as_float(row, "episode")) + 1 for row in rows], dtype=int)
    reward = np.array([_as_float(row, "reward") for row in rows], dtype=float)
    success = np.array([_as_float(row, "episode_success", _as_float(row, "success")) for row in rows], dtype=float)
    collision = np.array([_as_float(row, "episode_collision", _as_float(row, "collision")) for row in rows], dtype=float)
    timeout = np.array([_as_float(row, "episode_timeout", _as_float(row, "timeout")) for row in rows], dtype=float)
    out_of_bounds = np.array([_as_float(row, "episode_out_of_bounds", _as_float(row, "out_of_bounds")) for row in rows], dtype=float)
    final_target_distance = np.array([_as_float(row, "final_target_distance") for row in rows], dtype=float)
    env_names = [str(row.get("episode_env") or row.get("curriculum_env") or "") for row in rows]
    return {
        "episodes": episodes,
        "reward": reward,
        "success": success,
        "collision": collision,
        "timeout": timeout,
        "out_of_bounds": out_of_bounds,
        "final_target_distance": final_target_distance,
        "env_names": np.array(env_names, dtype=object),
    }


def _load_transitions(transition_json: Path) -> list[dict[str, object]]:
    if not transition_json.exists():
        return []
    payload = json.loads(transition_json.read_text(encoding="utf-8"))
    transitions = payload.get("transitions", [])
    return transitions if isinstance(transitions, list) else []


def _stage_spans(episodes: np.ndarray, transitions: list[dict[str, object]], current_envs: np.ndarray) -> list[tuple[int, int, str]]:
    if len(episodes) == 0:
        return []
    spans: list[tuple[int, int, str]] = []
    start_episode = int(episodes[0])
    start_env = str(current_envs[0])
    for item in transitions:
        boundary = int(item.get("episode_count", 0))
        next_env = str(item.get("to_env", ""))
        spans.append((start_episode, boundary, start_env))
        start_episode = boundary
        start_env = next_env
    spans.append((start_episode, int(episodes[-1]), start_env))
    return spans


def _shade_stages(ax: plt.Axes, spans: list[tuple[int, int, str]]) -> None:
    colors = {
        "avoidance_A1_static_single": "#eef6ff",
        "avoidance_A2_static_multi": "#eef8ef",
        "avoidance_A3_dynamic_few": "#fff6ea",
        "avoidance_A4_dynamic_multi_target": "#f8efff",
    }
    labels = {
        "avoidance_A1_static_single": "A1",
        "avoidance_A2_static_multi": "A2",
        "avoidance_A3_dynamic_few": "A3",
        "avoidance_A4_dynamic_multi_target": "A4",
    }
    for start, end, env_name in spans:
        if not env_name:
            continue
        ax.axvspan(start, end, color=colors.get(env_name, "#f5f5f5"), alpha=0.55, linewidth=0)
        ylim = ax.get_ylim()
        ypos = ylim[1] - (ylim[1] - ylim[0]) * 0.06
        ax.text(
            (start + end) / 2.0,
            ypos,
            labels.get(env_name, env_name),
            ha="center",
            va="top",
            fontsize=10,
            color="#555555",
        )


def _apply_thesis_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.linewidth": 1.0,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.unicode_minus": False,
        }
    )


def plot_thesis_figures(
    run_dir: Path,
    prefix: str = "blocking_success_guard_a1a4_thesis",
    *,
    chinese: bool = False,
) -> list[Path]:
    _apply_thesis_style()
    history_csv = run_dir / "live_block_history.csv"
    transition_json = run_dir / "env_transition_history.json"
    if not history_csv.exists():
        raise FileNotFoundError(f"Missing history CSV: {history_csv}")

    series = _load_history(history_csv)
    transitions = _load_transitions(transition_json)

    episodes = series["episodes"]
    reward = series["reward"]
    success = series["success"]
    collision = series["collision"]
    timeout = series["timeout"]
    out_of_bounds = series["out_of_bounds"]
    final_target_distance = series["final_target_distance"]
    env_names = series["env_names"]
    spans = _stage_spans(episodes, transitions, env_names)

    reward_roll = _rolling_mean(reward, 20)
    success_roll = _rolling_mean(success, 50)
    collision_roll = _rolling_mean(collision, 50)
    timeout_roll = _rolling_mean(timeout, 50)
    oob_roll = _rolling_mean(out_of_bounds, 50)
    distance_roll = _rolling_mean(final_target_distance, 20)

    outputs: list[Path] = []

    fig1, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
    axes[0].plot(episodes, reward, color="#b9d6f2", linewidth=0.9, alpha=0.55)
    axes[0].plot(episodes, reward_roll, color="#2166ac", linewidth=2.0)
    axes[0].set_ylabel("回合回报" if chinese else "Episode return")
    axes[0].grid(alpha=0.18, linestyle="--", linewidth=0.6)

    axes[1].plot(episodes, success_roll, color="#1a9850", linewidth=2.0, label="成功率" if chinese else "Success rate")
    axes[1].plot(episodes, collision_roll, color="#d73027", linewidth=1.8, label="碰撞率" if chinese else "Collision rate")
    axes[1].plot(episodes, timeout_roll, color="#e6ab02", linewidth=1.8, label="超时率" if chinese else "Timeout rate")
    axes[1].plot(episodes, oob_roll, color="#7570b3", linewidth=1.6, label="越界率" if chinese else "Out-of-bounds rate")
    axes[1].set_ylabel("滚动比例" if chinese else "Rolling rate")
    axes[1].set_xlabel("训练回合" if chinese else "Episode")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.18, linestyle="--", linewidth=0.6)

    for ax in axes:
        _shade_stages(ax, spans)

    handles, labels = axes[1].get_legend_handles_labels()
    fig1.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.5, 0.505),
        ncol=4,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.2,
    )
    fig1.tight_layout()
    fig1.subplots_adjust(hspace=0.34)
    for ext in ("png", "svg", "pdf"):
        path = run_dir / f"{prefix}_curves.{ext}"
        fig1.savefig(path, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig1)

    stage_map = {
        "avoidance_A1_static_single": "A1",
        "avoidance_A2_static_multi": "A2",
        "avoidance_A3_dynamic_few": "A3",
        "avoidance_A4_dynamic_multi_target": "A4",
    }
    order = ["A1", "A2", "A3", "A4"]
    agg = {key: {"count": 0, "success": 0, "collision": 0, "timeout": 0, "oob": 0, "distance": []} for key in order}
    with history_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        env_name = str(row.get("episode_env") or row.get("curriculum_env") or "")
        stage = stage_map.get(env_name)
        if stage is None:
            continue
        agg[stage]["count"] += 1
        agg[stage]["success"] += 1 if _as_float(row, "episode_success", _as_float(row, "success")) > 0.5 else 0
        agg[stage]["collision"] += 1 if _as_float(row, "episode_collision", _as_float(row, "collision")) > 0.5 else 0
        agg[stage]["timeout"] += 1 if _as_float(row, "episode_timeout", _as_float(row, "timeout")) > 0.5 else 0
        agg[stage]["oob"] += 1 if _as_float(row, "episode_out_of_bounds", _as_float(row, "out_of_bounds")) > 0.5 else 0
        agg[stage]["distance"].append(_as_float(row, "final_target_distance"))

    success_rate = [agg[s]["success"] / agg[s]["count"] if agg[s]["count"] else 0.0 for s in order]
    collision_rate = [agg[s]["collision"] / agg[s]["count"] if agg[s]["count"] else 0.0 for s in order]
    timeout_rate = [agg[s]["timeout"] / agg[s]["count"] if agg[s]["count"] else 0.0 for s in order]
    oob_rate = [agg[s]["oob"] / agg[s]["count"] if agg[s]["count"] else 0.0 for s in order]
    mean_distance = [float(np.mean(agg[s]["distance"])) if agg[s]["distance"] else 0.0 for s in order]
    counts = [agg[s]["count"] for s in order]

    fig2, axes2 = plt.subplots(2, 1, figsize=(6.6, 6.0), sharex=True)
    x = np.arange(len(order))
    width = 0.18
    axes2[0].bar(x - 1.5 * width, success_rate, width=width, color="#1a9850", label="成功" if chinese else "Success")
    axes2[0].bar(x - 0.5 * width, collision_rate, width=width, color="#d73027", label="碰撞" if chinese else "Collision")
    axes2[0].bar(x + 0.5 * width, timeout_rate, width=width, color="#e6ab02", label="超时" if chinese else "Timeout")
    axes2[0].bar(x + 1.5 * width, oob_rate, width=width, color="#7570b3", label="越界" if chinese else "Out-of-bounds")
    axes2[0].set_ylabel("比例" if chinese else "Rate")
    axes2[0].set_ylim(0.0, 1.0)
    axes2[0].legend(loc="upper right", ncol=2, frameon=False)
    axes2[0].grid(alpha=0.18, axis="y", linestyle="--", linewidth=0.6)
    for idx, count in enumerate(counts):
        axes2[0].text(idx, 0.97, f"n={count}", ha="center", va="top", fontsize=9, color="#555555")

    axes2[1].plot(x, mean_distance, color="#5e3c99", marker="o", markersize=5, linewidth=2.0)
    axes2[1].set_ylabel("平均最终距离" if chinese else "Mean final distance")
    axes2[1].set_xlabel("课程阶段" if chinese else "Curriculum stage")
    axes2[1].set_xticks(x)
    axes2[1].set_xticklabels(order)
    axes2[1].grid(alpha=0.18, linestyle="--", linewidth=0.6)

    fig2.tight_layout()
    for ext in ("png", "svg", "pdf"):
        path = run_dir / f"{prefix}_stage_summary.{ext}"
        fig2.savefig(path, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig2)
    return outputs


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--prefix", type=str, default="blocking_success_guard_a1a4_thesis")
    parser.add_argument("--chinese", action="store_true")
    args = parser.parse_args()
    outputs = plot_thesis_figures(args.run_dir.resolve(), prefix=args.prefix, chinese=bool(args.chinese))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
