from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"D:\ZhangC\lc")
SHORT_SPEED_DIR = ROOT / r"outputs\control_pybullet\x_ladrc_retune_short_speed_r_scan\x\20260411_185832"
MULTISPEED_DIR = ROOT / r"outputs\control_pybullet\x_multispeed_ladrc_retune_vs_pid\x\20260411_214152"
OUTPUT_ROOT = ROOT / r"outputs\chapter3_redraw\ladrc_tuning_stage"


FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]

COLORS = {
    "reference": "#2B2B2B",
    "pid": "#1F77B4",
    "default_ladrc": "#C44E52",
    "retuned_ladrc": "#2CA02C",
    "accent": "#DD8452",
}


@dataclass(frozen=True)
class ControllerLabel:
    key: str
    display_name: str
    color: str
    csv_name: str


CONTROLLERS = [
    ControllerLabel("pid", "PID", COLORS["pid"], "pid_timeseries.csv"),
    ControllerLabel("default_ladrc", "默认LADRC", COLORS["default_ladrc"], "default_ladrc_timeseries.csv"),
    ControllerLabel("retuned_ladrc", "重整定LADRC", COLORS["retuned_ladrc"], "retuned_ladrc_timeseries.csv"),
]


def configure_matplotlib() -> None:
    matplotlib.use("Agg")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = FONT_CANDIDATES
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = "#D9D9D9"
    plt.rcParams["grid.linewidth"] = 0.6
    plt.rcParams["grid.alpha"] = 0.75
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_reference_csv(path: Path, control_freq_hz: float = 48.0) -> pd.DataFrame:
    data = pd.read_csv(path)
    if "time" not in data.columns and "step" in data.columns:
        data["time"] = data["step"].astype(float) / float(control_freq_hz)
    return data


def build_output_dir(tag: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    out_dir = OUTPUT_ROOT / f"{stamp}{suffix}"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "summaries").mkdir(parents=True, exist_ok=True)
    return out_dir


def add_param_box(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.015,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#BFBFBF", "alpha": 0.95},
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def short_speed_param_text(summary: dict) -> str:
    tuning = summary["tuning"]
    default = tuning["default_repo_params"]
    retuned = tuning["retuned_candidate"]
    scenario = summary["scenario_definition"]
    return (
        "实验参数\n"
        f"后端: {summary['backend']}\n"
        f"控制频率: {scenario['control_freq_hz']} Hz\n"
        f"总时长: {scenario['duration_sec']:.1f} s\n"
        f"参考速度: {scenario['current_tuning_speed_mps']:.1f} m/s\n"
        f"阶段步长: {scenario['fixed_stage_lengths']}\n"
        f"默认参数: r={default['r']}, b0={default['b0']}, wc={default['omega_c']}, k={default['k']}\n"
        f"重整定参数: r={retuned['r']}, b0={retuned['b0']}, wc={retuned['omega_c']:.2f}, k={retuned['k']:.3f}"
    )


def multispeed_param_text(summary: dict) -> str:
    route = summary["route_definition"]
    search = summary["search_definition"]
    speed_grid = ", ".join(f"{v:.1f}" for v in summary["speed_grid_mps"])
    return (
        "实验参数\n"
        f"后端: {summary['backend']}\n"
        f"控制频率: 48 Hz\n"
        f"总时长: {route['duration_sec']:.1f} s\n"
        f"速度集合: [{speed_grid}] m/s\n"
        f"阶段步长: {route['fixed_stage_lengths']}\n"
        f"r扫描: {search['r_scan_values']}\n"
        f"b0范围: [{search['b0_range'][0]}, {search['b0_range'][1]}]\n"
        f"wc范围: [{search['omega_c_range'][0]}, {search['omega_c_range'][1]}]\n"
        f"k范围: [{search['k_range'][0]}, {search['k_range'][1]}]"
    )


def plot_short_speed_tracking(short_dir: Path, out_dir: Path, summary: dict) -> None:
    reference = load_reference_csv(
        short_dir / "reference.csv",
        control_freq_hz=summary["scenario_definition"]["control_freq_hz"],
    )
    traces = {item.key: pd.read_csv(short_dir / item.csv_name) for item in CONTROLLERS}
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 11), sharex=True)
    param_text = short_speed_param_text(summary)

    axes[0].plot(reference["time"], reference["target_x"], "--", color=COLORS["reference"], linewidth=2.2, label="参考轨迹")
    for item in CONTROLLERS:
        trace = traces[item.key]
        axes[0].plot(trace["time"], trace["x"], color=item.color, linewidth=2.0, label=item.display_name)
    axes[0].set_ylabel("位置 x / m")
    axes[0].legend(loc="upper left", ncol=4, frameon=False)
    add_param_box(axes[0], param_text)

    axes[1].plot(reference["time"], reference["target_vx"], "--", color=COLORS["reference"], linewidth=2.2, label="参考速度")
    for item in CONTROLLERS:
        trace = traces[item.key]
        axes[1].plot(trace["time"], trace["vx"], color=item.color, linewidth=2.0, label=item.display_name)
    axes[1].set_ylabel("速度 vx / (m/s)")

    for item in CONTROLLERS:
        trace = traces[item.key]
        error = trace["x"] - trace["target_x"]
        axes[2].plot(trace["time"], error, color=item.color, linewidth=2.0, label=item.display_name)
    axes[2].axhline(0.0, color=COLORS["reference"], linestyle="--", linewidth=1.3)
    axes[2].set_ylabel("跟踪误差 / m")
    axes[2].set_xlabel("时间 / s")

    save_figure(fig, out_dir / "figures" / "x_ladrc_short_speed_tracking_redraw.png")


def plot_r_scan(short_dir: Path, out_dir: Path, summary: dict) -> None:
    data = pd.read_csv(short_dir / "candidate_r_scan.csv").sort_values("r")
    default_r = summary["tuning"]["default_repo_params"]["r"]
    retuned_r = summary["tuning"]["retuned_candidate"]["r"]
    default_row = data.loc[(data["r"] - default_r).abs().idxmin()]
    best_row = data.loc[data["score"].idxmin()]

    fig, ax = plt.subplots(figsize=(9.8, 5.5))
    ax.plot(data["r"], data["score"], color=COLORS["accent"], marker="o", linewidth=2.2, markersize=5.5)
    ax.scatter([default_row["r"]], [default_row["score"]], color=COLORS["default_ladrc"], s=68, zorder=3)
    ax.scatter([best_row["r"]], [best_row["score"]], color=COLORS["retuned_ladrc"], s=68, zorder=3)
    ax.annotate(
        f"默认仓库参数 r={default_row['r']:.0f}\nscore={default_row['score']:.3f}",
        xy=(default_row["r"], default_row["score"]),
        xytext=(-30, 26),
        textcoords="offset points",
        fontsize=9.5,
        arrowprops={"arrowstyle": "->", "color": COLORS["default_ladrc"]},
    )
    ax.annotate(
        f"r扫描最优点 r={best_row['r']:.0f}\nscore={best_row['score']:.3f}",
        xy=(best_row["r"], best_row["score"]),
        xytext=(14, -38),
        textcoords="offset points",
        fontsize=9.5,
        arrowprops={"arrowstyle": "->", "color": COLORS["retuned_ladrc"]},
    )
    ax.axvline(retuned_r, color=COLORS["retuned_ladrc"], linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("TD参数 r")
    ax.set_ylabel("综合评价分数")
    add_param_box(ax, short_speed_param_text(summary))
    save_figure(fig, out_dir / "figures" / "x_ladrc_r_scan_redraw.png")


def plot_short_speed_metrics(short_dir: Path, out_dir: Path) -> None:
    metrics = pd.read_csv(short_dir / "metrics.csv")
    rename_map = {
        "pid_pos_att": "PID",
        "ladrc_x_pos_pid_att_default": "默认LADRC",
        "ladrc_x_pos_pid_att_retuned": "重整定LADRC",
    }
    metrics["controller"] = metrics["controller"].map(rename_map)
    metrics = metrics[["controller", "rmse", "mae", "velocity_rmse", "settling_time", "steady_state_error", "score"]]
    metrics.columns = ["控制器", "RMSE", "MAE", "速度RMSE", "调节时间/步", "稳态误差", "综合分数"]
    metrics.to_csv(out_dir / "tables" / "x_ladrc_short_speed_metrics_redraw.csv", index=False, encoding="utf-8-sig")


def plot_multispeed_params(multi_dir: Path, out_dir: Path, summary: dict) -> None:
    data = pd.read_csv(multi_dir / "best_params_by_speed.csv").sort_values("speed_mps")
    fig, axes = plt.subplots(4, 1, figsize=(10.2, 12), sharex=True)
    params = [
        ("r", "TD参数 r"),
        ("b0", "ESO参数 b0"),
        ("omega_c", "带宽 ωc"),
        ("k", "反馈增益 k"),
    ]
    for ax, (column, ylabel) in zip(axes, params):
        ax.plot(data["speed_mps"], data[column], color=COLORS["retuned_ladrc"], marker="o", linewidth=2.2)
        ax.set_ylabel(ylabel)
    axes[0].set_xlim(data["speed_mps"].min() - 0.03, data["speed_mps"].max() + 0.03)
    axes[-1].set_xlabel("参考速度 / (m/s)")
    add_param_box(axes[0], multispeed_param_text(summary))
    save_figure(fig, out_dir / "figures" / "x_ladrc_multispeed_best_params_redraw.png")


def plot_multispeed_metrics(multi_dir: Path, out_dir: Path, summary: dict) -> None:
    data = pd.read_csv(multi_dir / "best_params_by_speed.csv").sort_values("speed_mps")
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 8.5), sharex=True)

    axes[0].plot(data["speed_mps"], data["pid_rmse"], color=COLORS["pid"], marker="o", linewidth=2.1, label="PID")
    axes[0].plot(data["speed_mps"], data["retuned_rmse"], color=COLORS["retuned_ladrc"], marker="s", linewidth=2.1, label="重整定LADRC")
    axes[0].set_ylabel("位置RMSE / m")
    axes[0].legend(loc="upper left", frameon=False, ncol=2)
    add_param_box(axes[0], multispeed_param_text(summary))

    axes[1].axhline(1.0, color=COLORS["reference"], linestyle="--", linewidth=1.4)
    axes[1].plot(data["speed_mps"], data["rmse_ratio_to_pid"], color=COLORS["accent"], marker="o", linewidth=2.1)
    axes[1].set_ylabel("RMSE相对PID比值")
    axes[1].set_xlabel("参考速度 / (m/s)")

    save_figure(fig, out_dir / "figures" / "x_ladrc_multispeed_rmse_redraw.png")


def plot_typical_speed_panels(multi_dir: Path, out_dir: Path, summary: dict) -> None:
    speeds = summary["typical_speeds_mps"]
    fig, axes = plt.subplots(len(speeds), 2, figsize=(13, 10), sharex=False)

    for row_idx, speed in enumerate(speeds):
        speed_dir = multi_dir / "speeds" / str(speed)
        reference = load_reference_csv(speed_dir / "reference.csv", control_freq_hz=48.0)
        traces = {item.key: pd.read_csv(speed_dir / item.csv_name) for item in CONTROLLERS}
        ax_pos = axes[row_idx, 0]
        ax_vel = axes[row_idx, 1]
        ax_pos.plot(reference["time"], reference["target_x"], "--", color=COLORS["reference"], linewidth=2.0)
        ax_vel.plot(reference["time"], reference["target_vx"], "--", color=COLORS["reference"], linewidth=2.0)
        for item in CONTROLLERS:
            trace = traces[item.key]
            ax_pos.plot(trace["time"], trace["x"], color=item.color, linewidth=1.8)
            ax_vel.plot(trace["time"], trace["vx"], color=item.color, linewidth=1.8)
        ax_pos.set_ylabel(f"{speed:.1f} m/s\n位置 x / m")
        ax_vel.set_ylabel(f"{speed:.1f} m/s\n速度 vx / (m/s)")

    axes[-1, 0].set_xlabel("时间 / s")
    axes[-1, 1].set_xlabel("时间 / s")
    handles = [
        plt.Line2D([], [], color=COLORS["reference"], linestyle="--", linewidth=2.0, label="参考"),
        plt.Line2D([], [], color=COLORS["pid"], linewidth=1.9, label="PID"),
        plt.Line2D([], [], color=COLORS["default_ladrc"], linewidth=1.9, label="默认LADRC"),
        plt.Line2D([], [], color=COLORS["retuned_ladrc"], linewidth=1.9, label="重整定LADRC"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    add_param_box(axes[0, 0], multispeed_param_text(summary))
    save_figure(fig, out_dir / "figures" / "x_ladrc_multispeed_typical_tracking_redraw.png")


def export_param_tables(short_summary: dict, multi_summary: dict, out_dir: Path) -> None:
    short_rows = [
        ["实验阶段", "单速度固定参数整定"],
        ["后端", short_summary["backend"]],
        ["控制频率(Hz)", short_summary["scenario_definition"]["control_freq_hz"]],
        ["总时长(s)", short_summary["scenario_definition"]["duration_sec"]],
        ["参考速度(m/s)", short_summary["scenario_definition"]["current_tuning_speed_mps"]],
        ["阶段长度(steps)", str(short_summary["scenario_definition"]["fixed_stage_lengths"])],
        ["默认参数_r", short_summary["tuning"]["default_repo_params"]["r"]],
        ["默认参数_b0", short_summary["tuning"]["default_repo_params"]["b0"]],
        ["默认参数_wc", short_summary["tuning"]["default_repo_params"]["omega_c"]],
        ["默认参数_k", short_summary["tuning"]["default_repo_params"]["k"]],
        ["重整定参数_r", short_summary["tuning"]["retuned_candidate"]["r"]],
        ["重整定参数_b0", short_summary["tuning"]["retuned_candidate"]["b0"]],
        ["重整定参数_wc", short_summary["tuning"]["retuned_candidate"]["omega_c"]],
        ["重整定参数_k", short_summary["tuning"]["retuned_candidate"]["k"]],
    ]
    pd.DataFrame(short_rows, columns=["参数项", "数值"]).to_csv(
        out_dir / "tables" / "x_ladrc_short_speed_experiment_params.csv",
        index=False,
        encoding="utf-8-sig",
    )

    multi_rows = [
        ["实验阶段", "多速度固定参数整定"],
        ["后端", multi_summary["backend"]],
        ["控制频率(Hz)", 48],
        ["总时长(s)", multi_summary["route_definition"]["duration_sec"]],
        ["速度集合(m/s)", ",".join(str(v) for v in multi_summary["speed_grid_mps"])],
        ["阶段长度(steps)", str(multi_summary["route_definition"]["fixed_stage_lengths"])],
        ["r扫描", ",".join(str(v) for v in multi_summary["search_definition"]["r_scan_values"])],
        ["b0范围", str(multi_summary["search_definition"]["b0_range"])],
        ["wc范围", str(multi_summary["search_definition"]["omega_c_range"])],
        ["k范围", str(multi_summary["search_definition"]["k_range"])],
    ]
    pd.DataFrame(multi_rows, columns=["参数项", "数值"]).to_csv(
        out_dir / "tables" / "x_ladrc_multispeed_experiment_params.csv",
        index=False,
        encoding="utf-8-sig",
    )


def export_manifest(out_dir: Path, short_summary: dict, multi_summary: dict) -> None:
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "short_speed_source": str(SHORT_SPEED_DIR),
        "multispeed_source": str(MULTISPEED_DIR),
        "short_speed_summary": {
            "backend": short_summary["backend"],
            "reference_speed_mps": short_summary["scenario_definition"]["current_tuning_speed_mps"],
            "default_params": short_summary["tuning"]["default_repo_params"],
            "retuned_params": short_summary["tuning"]["retuned_candidate"],
        },
        "multispeed_summary": {
            "backend": multi_summary["backend"],
            "speed_grid_mps": multi_summary["speed_grid_mps"],
            "typical_speeds_mps": multi_summary["typical_speeds_mps"],
            "search_definition": multi_summary["search_definition"],
        },
        "figures": sorted(str(path) for path in (out_dir / "figures").glob("*.png")),
        "tables": sorted(str(path) for path in (out_dir / "tables").glob("*.csv")),
    }
    (out_dir / "summaries" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    note = (
        "# LADRC参数整定阶段重画说明\n\n"
        "本目录使用已确认可信的固定参数整定结果重新绘制第三章插图，不重跑原始实验。\n\n"
        "## 单速度整定实验\n"
        f"- 数据源: `{SHORT_SPEED_DIR}`\n"
        f"- 后端: `{short_summary['backend']}`\n"
        f"- 控制频率: `{short_summary['scenario_definition']['control_freq_hz']} Hz`\n"
        f"- 总时长: `{short_summary['scenario_definition']['duration_sec']} s`\n"
        f"- 参考速度: `{short_summary['scenario_definition']['current_tuning_speed_mps']} m/s`\n"
        f"- 默认参数: `r={short_summary['tuning']['default_repo_params']['r']}, b0={short_summary['tuning']['default_repo_params']['b0']}, omega_c={short_summary['tuning']['default_repo_params']['omega_c']}, k={short_summary['tuning']['default_repo_params']['k']}`\n"
        f"- 重整定参数: `r={short_summary['tuning']['retuned_candidate']['r']}, b0={short_summary['tuning']['retuned_candidate']['b0']}, omega_c={short_summary['tuning']['retuned_candidate']['omega_c']:.2f}, k={short_summary['tuning']['retuned_candidate']['k']:.3f}`\n\n"
        "## 多速度整定实验\n"
        f"- 数据源: `{MULTISPEED_DIR}`\n"
        f"- 速度集合: `{multi_summary['speed_grid_mps']}` m/s\n"
        f"- 典型展示速度: `{multi_summary['typical_speeds_mps']}` m/s\n"
        f"- 参考阶段长度: `{multi_summary['route_definition']['fixed_stage_lengths']}`\n"
        f"- 搜索范围: `r={multi_summary['search_definition']['r_scan_values']}, "
        f"b0={multi_summary['search_definition']['b0_range']}, "
        f"omega_c={multi_summary['search_definition']['omega_c_range']}, "
        f"k={multi_summary['search_definition']['k_range']}`\n"
    )
    (out_dir / "summaries" / "README.md").write_text(note, encoding="utf-8")


def main(tag: str | None = None) -> None:
    configure_matplotlib()
    out_dir = build_output_dir(tag)
    short_summary = load_json(SHORT_SPEED_DIR / "summary_readable.json")
    multi_summary = load_json(MULTISPEED_DIR / "summary_readable.json")

    plot_short_speed_tracking(SHORT_SPEED_DIR, out_dir, short_summary)
    plot_r_scan(SHORT_SPEED_DIR, out_dir, short_summary)
    plot_short_speed_metrics(SHORT_SPEED_DIR, out_dir)
    plot_multispeed_params(MULTISPEED_DIR, out_dir, multi_summary)
    plot_multispeed_metrics(MULTISPEED_DIR, out_dir, multi_summary)
    plot_typical_speed_panels(MULTISPEED_DIR, out_dir, multi_summary)
    export_param_tables(short_summary, multi_summary, out_dir)
    export_manifest(out_dir, short_summary, multi_summary)

    print(out_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Redraw LADRC tuning-stage figures for Chapter 3.")
    parser.add_argument("--tag", default=None, help="Optional suffix for the output directory name.")
    args = parser.parse_args()
    main(args.tag)
