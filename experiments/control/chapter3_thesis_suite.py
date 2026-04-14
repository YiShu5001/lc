from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@dataclass(frozen=True)
class SourceBundle:
    label: str
    path: Path
    description: str


def ensure_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}")
    return path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def to_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def mean(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return sum(valid) / len(valid) if valid else float("nan")


def control_variation_from_timeseries(path: Path) -> float:
    rows = read_csv(path)
    rpm_keys = ["rpm0", "rpm1", "rpm2", "rpm3"]
    previous: list[float] | None = None
    diffs: list[float] = []
    for row in rows:
        current = [to_float(row[key]) for key in rpm_keys]
        if previous is not None:
            diffs.append(sum(abs(cur - prev) for cur, prev in zip(current, previous)) / len(rpm_keys))
        previous = current
    return mean(diffs)


def disturbance_abs_mean_from_timeseries(path: Path) -> float:
    rows = read_csv(path)
    if not rows or "disturbance_x" not in rows[0]:
        return 0.0
    return mean([abs(to_float(row["disturbance_x"])) for row in rows])


def normalise_series(values: list[float]) -> list[float]:
    maximum = max(values) if values else 1.0
    if maximum <= 0:
        maximum = 1.0
    return [value / maximum for value in values]


def make_box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, facecolor: str) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor=facecolor, edgecolor="#233044", linewidth=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, color="#0f172a")


def arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.add_patch(
        FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->", mutation_scale=12, linewidth=1.5, color="#334155")
    )


def draw_framework_figure(
    path: Path,
    title: str,
    boxes: list[tuple[float, float, float, float, str, str]],
    arrows: list[tuple[float, float, float, float]],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    for item in boxes:
        make_box(ax, *item)
    for item in arrows:
        arrow(ax, *item)
    ax.set_title(title, fontsize=14, pad=16)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_reward_curve(path: Path, histories: list[tuple[str, Path]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, history_path in histories:
        rows = read_csv(history_path)
        episodes = [int(float(row["episode"])) for row in rows]
        avg_rewards = [to_float(row["average_reward"]) for row in rows]
        ax.plot(episodes, avg_rewards, linewidth=2, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_control_smoothness(path: Path, scenario_to_values: dict[str, dict[str, float]]) -> None:
    scenarios = list(scenario_to_values.keys())
    controllers = list(next(iter(scenario_to_values.values())).keys())
    x = list(range(len(controllers)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, scenario in enumerate(scenarios):
        values = [scenario_to_values[scenario][controller] for controller in controllers]
        shifted = [value + (idx - 0.5) * width for value in x]
        ax.bar(shifted, values, width=width, label=scenario)
    ax.set_xticks(x)
    ax.set_xticklabels(controllers)
    ax.set_ylabel("Mean RPM Variation")
    ax.set_title("Control Smoothness Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_overall_performance(path: Path, scenario_rows: dict[str, list[dict[str, Any]]]) -> None:
    metrics = ["rmse", "mae", "velocity_rmse"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4.8))
    for ax, metric in zip(axes, metrics):
        labels: list[str] = []
        values: list[float] = []
        colors: list[str] = []
        for scenario, rows in scenario_rows.items():
            metric_values = [to_float(row[metric]) for row in rows]
            normalized = normalise_series(metric_values)
            for row, value in zip(rows, normalized):
                labels.append(f"{scenario}\\n{row['controller']}")
                values.append(value)
                colors.append("#3b82f6" if "DDPG" in row["controller"] else ("#ef4444" if "PID" in row["controller"] else "#10b981"))
        ax.bar(range(len(labels)), values, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_title(metric.upper())
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Overall Performance Comparison", fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def extract_single_speed_rows(summary_readable: dict[str, Any]) -> list[dict[str, Any]]:
    controllers = summary_readable["controllers"]
    tuning = summary_readable["tuning"]
    default_params = tuning["default_repo_params"]
    retuned = tuning["retuned_candidate"]
    mapping = [
        ("PID", "pid_pos_att", default_params),
        ("LADRC(default)", "ladrc_x_pos_pid_att_default", default_params),
        ("LADRC(retuned)", "ladrc_x_pos_pid_att_retuned", retuned),
    ]
    rows: list[dict[str, Any]] = []
    for label, key, params in mapping:
        metrics = controllers[key]
        rows.append(
            {
                "controller": label,
                "r": params["r"],
                "b0": params["b0"],
                "omega_c": params["omega_c"],
                "k": params["k"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "velocity_rmse": metrics["velocity_rmse"],
                "reward": metrics["reward"],
                "control_variation": metrics["control_variation"],
                "score": metrics["score"],
            }
        )
    return rows


def summarise_rl_runs(summary_json: dict[str, Any], scenario: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_rows = summary_json.get("summary_rows")
    if source_rows:
        for row in source_rows:
            rows.append(
                {
                    "scenario": scenario,
                    "shared_value": row["shared_value"],
                    "train_episodes": row["train_episodes"],
                    "average_reward": row["average_reward"],
                    "best_reward": row["best_reward"],
                    "best_eval_rmse": row["best_eval_rmse"],
                    "best_eval_mae": row["best_eval_mae"],
                    "best_eval_velocity_rmse": row["best_eval_velocity_rmse"],
                    "best_eval_score": row["best_eval_score"],
                    "best_checkpoint_path": row["best_checkpoint_path"],
                    "output_dir": row["output_dir"],
                }
            )
        return rows
    for run in summary_json["runs"]:
        best = run["best_eval_metrics"]
        rows.append(
            {
                "scenario": scenario,
                "shared_value": run["shared_value"],
                "train_episodes": summary_json["train_episodes"],
                "average_reward": run["average_reward"],
                "best_reward": "",
                "best_eval_rmse": best["rmse"],
                "best_eval_mae": best["mae"],
                "best_eval_velocity_rmse": best["velocity_rmse"],
                "best_eval_score": best["score"],
                "best_checkpoint_path": run["best_checkpoint_path"],
                "output_dir": run["output_dir"],
            }
        )
    return rows


def export_chapter3_package(output_root: Path, output_tag: str) -> dict[str, Any]:
    package_root = output_root / output_tag
    section_dirs = {
        "3_2_traditional_tuning": package_root / "3_2_traditional_tuning",
        "3_3_rl_ladrc_method": package_root / "3_3_rl_ladrc_method",
        "3_4_temporal_enhancement": package_root / "3_4_temporal_enhancement",
        "3_5_experiments": package_root / "3_5_experiments",
        "tables": package_root / "tables",
        "figures": package_root / "figures",
        "summaries": package_root / "summaries",
    }
    for directory in section_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    sources = {
        "single_speed_retune_summary": SourceBundle("single_speed_retune_summary", ensure_exists(PROJECT_ROOT / "outputs" / "control_pybullet" / "x_ladrc_retune_short_speed_r_scan" / "x" / "20260411_185832" / "summary_readable.json"), "单速度 0.5 m/s LADRC 重整定可信结果"),
        "single_speed_tracking_figure": SourceBundle("single_speed_tracking_figure", ensure_exists(PROJECT_ROOT / "outputs" / "control_pybullet" / "x_ladrc_retune_short_speed_r_scan" / "x" / "20260411_185832" / "figures" / "tracking_three_way.png"), "单速度短参考轨迹图"),
        "single_speed_error_figure": SourceBundle("single_speed_error_figure", ensure_exists(PROJECT_ROOT / "outputs" / "control_pybullet" / "x_ladrc_retune_short_speed_r_scan" / "x" / "20260411_185832" / "figures" / "tracking_error_three_way.png"), "单速度短参考误差图"),
        "multispeed_best_params": SourceBundle("multispeed_best_params", ensure_exists(PROJECT_ROOT / "outputs" / "control_pybullet" / "x_multispeed_ladrc_retune_vs_pid" / "x" / "20260411_214152" / "best_params_by_speed.csv"), "多速度 LADRC 最优参数表"),
        "baseline_no_disturbance_metrics": SourceBundle("baseline_no_disturbance_metrics", ensure_exists(PROJECT_ROOT / "outputs" / "control_pybullet" / "x_pid_vs_ladrc_no_disturbance_compare" / "x" / "20260412_105149" / "metrics.csv"), "无扰动固定控制器对比指标"),
        "baseline_no_disturbance_error": SourceBundle("baseline_no_disturbance_error", ensure_exists(PROJECT_ROOT / "outputs" / "control_pybullet" / "x_pid_vs_ladrc_no_disturbance_compare" / "x" / "20260412_105149" / "figures" / "error_compare.png"), "无扰动固定控制器误差图"),
        "no_dist_rl_summary": SourceBundle("no_dist_rl_summary", ensure_exists(PROJECT_ROOT / "outputs" / "control_pybullet_rl" / "x_refline_no_disturbance_mddpg_retrain" / "20260412_v1_to_v5_300eps_reexpanded" / "summary.json"), "无扰动 RL-LADRC 摘要"),
        "no_dist_rl_compare_metrics": SourceBundle("no_dist_rl_compare_metrics", ensure_exists(PROJECT_ROOT / "outputs" / "control_pybullet_rl" / "x_refline_no_disturbance_mddpg_retrain" / "20260412_v1_to_v5_300eps_reexpanded" / "best_v_compare" / "bestv_compare_20260412_150029" / "metrics.csv"), "无扰动 RL-LADRC 正式对比指标"),
        "random_rl_summary": SourceBundle("random_rl_summary", ensure_exists(PROJECT_ROOT / "outputs" / "control_pybullet_rl" / "x_refline_random_hover_disturbance_mddpg_retrain" / "20260412_random_hover_disturb_v1234_300eps_narrowed_noise0p2_0p004_posonly_multiseed_normfix" / "summary.json"), "0.004N 随机扰动 RL-LADRC 摘要"),
        "random_rl_compare_metrics": SourceBundle("random_rl_compare_metrics", ensure_exists(PROJECT_ROOT / "outputs" / "control_pybullet" / "x_pid_ladrc_ddpg_random_hover_disturb_compare" / "x" / "20260412_random_hover_disturb_0p004_bestv2_narrowed_noise0p2_posonly_multiseed_normfix_300eps_compare" / "metrics.csv"), "0.004N 随机扰动正式对比指标"),
        "random_rl_compare_tracking": SourceBundle("random_rl_compare_tracking", ensure_exists(PROJECT_ROOT / "outputs" / "control_pybullet" / "x_pid_ladrc_ddpg_random_hover_disturb_compare" / "x" / "20260412_random_hover_disturb_0p004_bestv2_narrowed_noise0p2_posonly_multiseed_normfix_300eps_compare" / "figures" / "tracking_compare.png"), "0.004N 随机扰动正式轨迹图"),
    }
    random_scan_dirs = {amplitude: ensure_exists(PROJECT_ROOT / "outputs" / "control_pybullet" / "x_pid_ladrc_ddpg_random_hover_disturb_compare" / "x" / f"hover_gap_rand_{amplitude}") for amplitude in ("0p003", "0p004", "0p005", "0p006", "0p007", "0p008", "0p009")}

    single_speed_summary = read_json(sources["single_speed_retune_summary"].path)
    no_dist_summary = read_json(sources["no_dist_rl_summary"].path)
    random_summary = read_json(sources["random_rl_summary"].path)
    baseline_rows = read_csv(sources["baseline_no_disturbance_metrics"].path)
    no_dist_best_rows = read_csv(sources["no_dist_rl_compare_metrics"].path)
    random_best_rows = read_csv(sources["random_rl_compare_metrics"].path)
    multispeed_rows = read_csv(sources["multispeed_best_params"].path)
    single_speed_rows = extract_single_speed_rows(single_speed_summary)
    rl_shared_rows = summarise_rl_runs(no_dist_summary, "no_disturbance") + summarise_rl_runs(random_summary, "random_hover_0p004")

    random_scan_rows: list[dict[str, Any]] = []
    for amplitude, directory in random_scan_dirs.items():
        for row in read_csv(directory / "metrics.csv"):
            record = dict(row)
            record["amplitude_n"] = float(amplitude.replace("p", "."))
            random_scan_rows.append(record)

    tables_manifest: dict[str, Any] = {}
    figures_manifest: dict[str, Any] = {}

    def register_table(name: str, rows: list[dict[str, Any]], fields: list[str], source_refs: list[str]) -> Path:
        path = section_dirs["tables"] / name
        write_csv(path, rows, fields)
        tables_manifest[name] = {"path": str(path), "sources": source_refs, "rows": len(rows)}
        return path

    def register_figure(name: str, source_refs: list[str], copy_from: Path | None = None) -> Path:
        path = section_dirs["figures"] / name
        if copy_from is not None:
            copy_file(copy_from, path)
        figures_manifest[name] = {"path": str(path), "sources": source_refs}
        return path

    register_table("table_ladrc_single_speed_retune.csv", single_speed_rows, list(single_speed_rows[0].keys()), ["single_speed_retune_summary"])
    register_table("table_ladrc_multispeed_best_params.csv", multispeed_rows, list(multispeed_rows[0].keys()), ["multispeed_best_params"])
    register_table("table_baseline_no_disturbance.csv", baseline_rows, list(baseline_rows[0].keys()), ["baseline_no_disturbance_metrics"])
    register_table("table_random_disturbance_scan_0p003_to_0p009.csv", random_scan_rows, list(random_scan_rows[0].keys()), [f"hover_gap_rand_{key}" for key in random_scan_dirs])
    register_table("table_rl_shared_value_summary.csv", rl_shared_rows, list(rl_shared_rows[0].keys()), ["no_dist_rl_summary", "random_rl_summary"])
    register_table("shared_value_summary.csv", rl_shared_rows, list(rl_shared_rows[0].keys()), ["no_dist_rl_summary", "random_rl_summary"])
    register_table("table_rl_best_v2_no_disturbance.csv", no_dist_best_rows, list(no_dist_best_rows[0].keys()), ["no_dist_rl_compare_metrics"])
    register_table("table_rl_best_v2_random_0p004.csv", random_best_rows, list(random_best_rows[0].keys()), ["random_rl_compare_metrics"])

    rl_action_rows = [
        {
            "scenario": "no_disturbance",
            "action_order": "r,b0,omega_c,k",
            "r_bounds": no_dist_summary["action_bounds"]["r"],
            "b0_bounds": no_dist_summary["action_bounds"]["b0"],
            "omega_c_bounds": no_dist_summary["action_bounds"]["omega_c"],
            "k_bounds": no_dist_summary["action_bounds"]["k"],
            "delta_r": no_dist_summary["delta_bounds"]["r"],
            "delta_b0": no_dist_summary["delta_bounds"]["b0"],
            "delta_omega_c": no_dist_summary["delta_bounds"]["omega_c"],
            "delta_k": no_dist_summary["delta_bounds"]["k"],
            "anchor_r": no_dist_summary["train_anchor"]["r"],
            "anchor_b0": no_dist_summary["train_anchor"]["b0"],
            "anchor_omega_c": no_dist_summary["train_anchor"]["omega_c"],
            "anchor_k": no_dist_summary["train_anchor"]["k"],
        },
        {
            "scenario": "random_hover_0p004",
            "action_order": "r,b0,omega_c,k",
            "r_bounds": random_summary["action_bounds"]["r"],
            "b0_bounds": random_summary["action_bounds"]["b0"],
            "omega_c_bounds": random_summary["action_bounds"]["omega_c"],
            "k_bounds": random_summary["action_bounds"]["k"],
            "delta_r": random_summary["delta_bounds"]["r"],
            "delta_b0": random_summary["delta_bounds"]["b0"],
            "delta_omega_c": random_summary["delta_bounds"]["omega_c"],
            "delta_k": random_summary["delta_bounds"]["k"],
            "anchor_r": random_summary["train_anchor"]["r"],
            "anchor_b0": random_summary["train_anchor"]["b0"],
            "anchor_omega_c": random_summary["train_anchor"]["omega_c"],
            "anchor_k": random_summary["train_anchor"]["k"],
        },
    ]
    register_table("rl_action_space_table.csv", rl_action_rows, list(rl_action_rows[0].keys()), ["no_dist_rl_summary", "random_rl_summary"])

    rl_training_config_rows = [
        {
            "scenario": "no_disturbance",
            "train_episodes": no_dist_summary["train_episodes"],
            "compare_episodes": no_dist_summary["compare_episodes"],
            "shared_values": ",".join(str(value) for value in no_dist_summary["shared_values"]),
            "hidden_dim": no_dist_summary["network_config"]["hidden_dim"],
            "dropout_p": no_dist_summary["network_config"]["dropout_p"],
            "tau": no_dist_summary["network_config"]["tau"],
            "soft_update_interval": no_dist_summary["network_config"]["soft_update_interval"],
            "exploration_schedule": no_dist_summary["network_config"]["exploration_noise_schedule"],
            "exploration_start": no_dist_summary["network_config"]["exploration_noise_start"],
            "exploration_end": no_dist_summary["network_config"]["exploration_noise_end"],
            "batch_size": no_dist_summary["network_config"]["batch_size"],
            "snapshot_interval": no_dist_summary["network_config"]["snapshot_interval"],
        },
        {
            "scenario": "random_hover_0p004",
            "train_episodes": random_summary["train_episodes"],
            "compare_episodes": random_summary["compare_episodes"],
            "shared_values": ",".join(str(value) for value in random_summary["shared_values"]),
            "hidden_dim": 768,
            "dropout_p": 0.25,
            "tau": 0.02,
            "soft_update_interval": 10,
            "exploration_schedule": random_summary["exploration_noise"]["schedule"],
            "exploration_start": random_summary["exploration_noise"]["start"],
            "exploration_end": random_summary["exploration_noise"]["end"],
            "batch_size": 128,
            "snapshot_interval": 20,
        },
    ]
    register_table("rl_training_config_table.csv", rl_training_config_rows, list(rl_training_config_rows[0].keys()), ["no_dist_rl_summary", "random_rl_summary"])

    no_dist_ddpg_ts = PROJECT_ROOT / "outputs" / "control_pybullet_rl" / "x_refline_no_disturbance_mddpg_retrain" / "20260412_v1_to_v5_300eps_reexpanded" / "best_v_compare" / "bestv_compare_20260412_150029" / "ddpg-ladrc_best_v_2" / "timeseries.csv"
    no_dist_ladrc_ts = PROJECT_ROOT / "outputs" / "control_pybullet_rl" / "x_refline_no_disturbance_mddpg_retrain" / "20260412_v1_to_v5_300eps_reexpanded" / "best_v_compare" / "bestv_compare_20260412_150029" / "ladrc0.5-opt" / "timeseries.csv"
    no_dist_pid_ts = PROJECT_ROOT / "outputs" / "control_pybullet_rl" / "x_refline_no_disturbance_mddpg_retrain" / "20260412_v1_to_v5_300eps_reexpanded" / "best_v_compare" / "bestv_compare_20260412_150029" / "pid" / "timeseries.csv"
    random_ddpg_ts = PROJECT_ROOT / "outputs" / "control_pybullet" / "x_pid_ladrc_ddpg_random_hover_disturb_compare" / "x" / "20260412_random_hover_disturb_0p004_bestv2_narrowed_noise0p2_posonly_multiseed_normfix_300eps_compare" / "ddpg_best_timeseries.csv"
    random_ladrc_ts = PROJECT_ROOT / "outputs" / "control_pybullet" / "x_pid_ladrc_ddpg_random_hover_disturb_compare" / "x" / "20260412_random_hover_disturb_0p004_bestv2_narrowed_noise0p2_posonly_multiseed_normfix_300eps_compare" / "ladrc_0p5_opt_timeseries.csv"
    random_pid_ts = PROJECT_ROOT / "outputs" / "control_pybullet" / "x_pid_ladrc_ddpg_random_hover_disturb_compare" / "x" / "20260412_random_hover_disturb_0p004_bestv2_narrowed_noise0p2_posonly_multiseed_normfix_300eps_compare" / "pid_timeseries.csv"

    overall_rows: list[dict[str, Any]] = []
    ts_map = {
        ("no_disturbance", "PID"): no_dist_pid_ts,
        ("no_disturbance", "LADRC(0.5-opt)"): no_dist_ladrc_ts,
        ("no_disturbance", "DDPG-LADRC(best v=2)"): no_dist_ddpg_ts,
        ("no_disturbance", "DDPG-LADRC (best v=2)"): no_dist_ddpg_ts,
        ("random_hover_0p004", "PID"): random_pid_ts,
        ("random_hover_0p004", "LADRC(0.5-opt)"): random_ladrc_ts,
        ("random_hover_0p004", "DDPG-LADRC (v=2, 0.004N, pos-only, multiseed, normfix, 300eps)"): random_ddpg_ts,
    }
    for scenario, rows in [("no_disturbance", no_dist_best_rows), ("random_hover_0p004", random_best_rows)]:
        for row in rows:
            ts_path = ts_map[(scenario, row["controller"])]
            overall_rows.append(
                {
                    "scenario": scenario,
                    "controller": row["controller"],
                    "rmse": row["rmse"],
                    "mae": row["mae"],
                    "velocity_rmse": row["velocity_rmse"],
                    "reward": row.get("reward", ""),
                    "control_variation": control_variation_from_timeseries(ts_path),
                    "disturbance_abs_mean": row.get("disturbance_abs_mean", disturbance_abs_mean_from_timeseries(ts_path)),
                }
            )
    register_table("table_overall_metrics_summary.csv", overall_rows, list(overall_rows[0].keys()), ["no_dist_rl_compare_metrics", "random_rl_compare_metrics"])

    fig1 = register_figure("fig1_ladrc_framework.png", ["single_speed_retune_summary"])
    draw_framework_figure(
        fig1,
        "Figure 1. LADRC Position Control Framework",
        [
            (0.4, 1.6, 1.6, 0.8, "Reference\nx*, vx*", "#dbeafe"),
            (2.4, 1.6, 1.7, 0.8, "TD\n(r)", "#e0f2fe"),
            (4.6, 1.6, 1.8, 0.8, "SEF\n(omega_c)", "#dcfce7"),
            (6.9, 1.6, 1.9, 0.8, "ESO\n(b0, k)", "#fde68a"),
            (9.3, 1.6, 1.8, 0.8, "PyBullet Drone", "#fecaca"),
        ],
        [(2.0, 2.0, 2.4, 2.0), (4.1, 2.0, 4.6, 2.0), (6.4, 2.0, 6.9, 2.0), (8.8, 2.0, 9.3, 2.0)],
    )
    register_figure("fig2_step_or_short_response_compare.png", ["single_speed_tracking_figure"], sources["single_speed_tracking_figure"].path)
    register_figure("fig3_disturbance_recovery_compare.png", ["random_rl_compare_tracking"], sources["random_rl_compare_tracking"].path)
    register_figure("fig4_tracking_error_compare.png", ["baseline_no_disturbance_error"], sources["baseline_no_disturbance_error"].path)
    fig5 = register_figure("fig5_ddpg_ladrc_framework.png", ["no_dist_rl_summary", "random_rl_summary"])
    draw_framework_figure(
        fig5,
        "Figure 5. DDPG-LADRC Online Retuning Framework",
        [
            (0.4, 1.6, 1.6, 0.8, "State\n[x,v,ref]", "#dbeafe"),
            (2.4, 1.6, 1.7, 0.8, "Actor", "#ddd6fe"),
            (4.6, 1.6, 1.9, 0.8, "Action\n[r,b0,wc,k]", "#fecdd3"),
            (7.0, 1.6, 1.7, 0.8, "LADRC", "#dcfce7"),
            (9.1, 1.6, 2.0, 0.8, "PyBullet Env\n+ Reward", "#fde68a"),
        ],
        [(2.0, 2.0, 2.4, 2.0), (4.1, 2.0, 4.6, 2.0), (6.5, 2.0, 7.0, 2.0), (8.7, 2.0, 9.1, 2.0)],
    )
    fig6 = register_figure("fig6_temporal_enhancement_framework.png", ["random_rl_summary", "no_dist_rl_summary"])
    draw_framework_figure(
        fig6,
        "Figure 6. Temporal Sample Enhancement Mechanism",
        [
            (0.4, 1.6, 1.7, 0.8, "State Stack\n(stack_size)", "#dbeafe"),
            (2.6, 1.6, 1.8, 0.8, "Action Hold\n(action_hold)", "#dcfce7"),
            (4.9, 1.6, 1.8, 0.8, "Replay Buffer", "#fde68a"),
            (7.2, 1.6, 1.8, 0.8, "N-Step Return", "#fecaca"),
            (9.5, 1.6, 1.6, 0.8, "Policy Update", "#ddd6fe"),
        ],
        [(2.1, 2.0, 2.6, 2.0), (4.4, 2.0, 4.9, 2.0), (6.7, 2.0, 7.2, 2.0), (9.0, 2.0, 9.5, 2.0)],
    )
    fig7 = register_figure("fig7_training_reward_curve.png", ["random_rl_summary"])
    random_training_histories = [(f"v={run['shared_value']}", Path(run["output_dir"]) / "training_history.csv") for run in random_summary["runs"]]
    plot_reward_curve(fig7, random_training_histories, "Figure 7. Training Convergence Curves (Random Hover Disturbance)")
    fig8 = register_figure("fig8_control_smoothness_compare.png", ["no_dist_rl_compare_metrics", "random_rl_compare_metrics"])
    plot_control_smoothness(
        fig8,
        {
            "No Disturbance": {
                "PID": control_variation_from_timeseries(no_dist_pid_ts),
                "LADRC": control_variation_from_timeseries(no_dist_ladrc_ts),
                "DDPG-LADRC": control_variation_from_timeseries(no_dist_ddpg_ts),
            },
            "Random 0.004N": {
                "PID": control_variation_from_timeseries(random_pid_ts),
                "LADRC": control_variation_from_timeseries(random_ladrc_ts),
                "DDPG-LADRC": control_variation_from_timeseries(random_ddpg_ts),
            },
        },
    )
    fig9 = register_figure("fig9_overall_performance_compare.png", ["no_dist_rl_compare_metrics", "random_rl_compare_metrics"])
    plot_overall_performance(fig9, {"No Disturbance": no_dist_best_rows, "Random 0.004N": random_best_rows})

    write_json(
        section_dirs["3_2_traditional_tuning"] / "section_manifest.json",
        {
            "section": "3.2",
            "tables": ["table_ladrc_single_speed_retune.csv", "table_ladrc_multispeed_best_params.csv", "table_baseline_no_disturbance.csv"],
            "figures": ["fig2_step_or_short_response_compare.png", "fig3_disturbance_recovery_compare.png", "fig4_tracking_error_compare.png"],
        },
    )
    write_json(
        section_dirs["3_3_rl_ladrc_method"] / "section_manifest.json",
        {
            "section": "3.3",
            "tables": ["rl_action_space_table.csv", "rl_training_config_table.csv", "shared_value_summary.csv"],
            "figures": ["fig5_ddpg_ladrc_framework.png"],
        },
    )
    write_json(
        section_dirs["3_4_temporal_enhancement"] / "section_manifest.json",
        {
            "section": "3.4",
            "tables": ["table_rl_shared_value_summary.csv"],
            "figures": ["fig6_temporal_enhancement_framework.png"],
            "notes": {
                "stack_size": "由 shared value 与时序增强机制共同影响。",
                "action_hold_steps": "体现在 trainer 与 policy 的跨时间动作保持逻辑里。",
                "n_step": "由增强回报计算和 replay 采样共同决定。",
            },
        },
    )
    write_json(
        section_dirs["3_5_experiments"] / "section_manifest.json",
        {
            "section": "3.5",
            "tables": [
                "table_rl_best_v2_no_disturbance.csv",
                "table_rl_best_v2_random_0p004.csv",
                "table_overall_metrics_summary.csv",
                "table_random_disturbance_scan_0p003_to_0p009.csv",
            ],
            "figures": ["fig7_training_reward_curve.png", "fig8_control_smoothness_compare.png", "fig9_overall_performance_compare.png"],
        },
    )

    experiment_manifest = {
        "generated_at": datetime.now().isoformat(),
        "package_root": str(package_root),
        "trusted_sources": {name: {"path": str(bundle.path), "description": bundle.description} for name, bundle in sources.items()},
        "random_scan_sources": {key: str(value) for key, value in random_scan_dirs.items()},
        "recommended_results": {"no_disturbance_best_v": 2, "random_hover_0p004_best_v": 2},
        "notes": [
            "本导出包只复用 2026-04-12 已经验证可信的 PyBullet 第三章主链结果。",
            "zero-delta rewrite 与 checkpoint state 完整恢复之前的 RL 结果没有纳入正式论文包。",
            "3.2 的阶跃响应使用当前短参考重整定实验近似支撑；若后续论文要求严格标准阶跃，可单独补实验。",
        ],
    }
    metrics_manifest = {
        "single_speed_rows": len(single_speed_rows),
        "multispeed_rows": len(multispeed_rows),
        "baseline_no_disturbance_rows": len(baseline_rows),
        "random_scan_rows": len(random_scan_rows),
        "shared_value_rows": len(rl_shared_rows),
        "overall_summary_rows": len(overall_rows),
    }
    write_json(section_dirs["summaries"] / "chapter3_experiment_manifest.json", experiment_manifest)
    write_json(section_dirs["summaries"] / "chapter3_metrics_manifest.json", metrics_manifest)
    write_json(section_dirs["summaries"] / "chapter3_figure_manifest.json", figures_manifest)

    return {
        "package_root": str(package_root),
        "tables": tables_manifest,
        "figures": figures_manifest,
        "summaries": {
            "chapter3_experiment_manifest.json": str(section_dirs["summaries"] / "chapter3_experiment_manifest.json"),
            "chapter3_metrics_manifest.json": str(section_dirs["summaries"] / "chapter3_metrics_manifest.json"),
            "chapter3_figure_manifest.json": str(section_dirs["summaries"] / "chapter3_figure_manifest.json"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a thesis-ready Chapter 3 experiment package.")
    parser.add_argument("--output-tag", default="", help="Optional directory tag. Default uses a timestamp.")
    parser.add_argument("--reuse-existing-results", action="store_true", help="Accepted for compatibility. The current suite always reuses trusted existing outputs.")
    parser.add_argument("--export-format", default="csv+png+json", help="Export format hint. Currently supports csv+png+json only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.export_format != "csv+png+json":
        raise ValueError("Only csv+png+json is currently supported.")
    output_tag = args.output_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = PROJECT_ROOT / "outputs" / "chapter3_thesis_package"
    result = export_chapter3_package(output_root, output_tag)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
