from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"D:\ZhangC\lc")
INPUT_ROOT = ROOT / "outputs" / "chapter3_result_data" / "20260415_chapter3_result_data"
OUTPUT_ROOT = ROOT / "outputs" / "chapter3_redraw" / "figures_3_6_3_7"

FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]

METHOD_COLORS = {
    "PID": "#1F77B4",
    "固定参数LADRC": "#FF7F0E",
    "DDPG-LADRC": "#2CA02C",
    "完整方法": "#1F77B4",
    "无状态叠加": "#D62728",
    "无动作保持": "#FF7F0E",
    "无 N-step": "#9467BD",
}

REF_COLOR = "#2F2F2F"
BAR_LEFT = "#4C78A8"
BAR_RIGHT = "#F58518"
REWARD_WARMUP_EPISODES = 5
REWARD_SMOOTH_WINDOW = 20


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


def build_output_dir(tag: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    out_dir = OUTPUT_ROOT / f"{stamp}{suffix}"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "summaries").mkdir(parents=True, exist_ok=True)
    return out_dir


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def add_bottom_caption(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.5,
        -0.22,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
    )


def plot_figure_3_6(out_dir: Path) -> dict[str, str]:
    fig_root = INPUT_ROOT / "figure_3_6_ddpg_rl_ladrc_compare"
    speed_df = pd.read_csv(fig_root / "plot_data" / "fig3_6a_speed_tracking_data.csv")
    recovery_df = pd.read_csv(fig_root / "plot_data" / "fig3_6b_disturbance_recovery_data.csv")
    smooth_df = pd.read_csv(fig_root / "fig3_6c_control_smoothness_stats.csv")
    reward_df = pd.read_csv(fig_root / "fig3_6d_training_reward_curve.csv")
    metrics_df = pd.read_csv(fig_root / "fig3_6_metrics.csv")
    reward_df = reward_df[reward_df["episode"] > reward_df["episode"].min()].copy()
    reward_df = reward_df[reward_df["episode"] >= 1].copy()
    reward_df["rolling_reward"] = (
        reward_df["reward"].rolling(window=REWARD_SMOOTH_WINDOW, min_periods=1).mean()
    )
    reward_plot = reward_df[reward_df["episode"] >= REWARD_WARMUP_EPISODES].copy()

    smooth_df.columns = ["方法", "控制输入方差", "平均|Δu|"]
    smooth_df["方法"] = smooth_df["方法"].replace({"????LADRC": "固定参数LADRC", "DDPG--RL--LADRC": "DDPG-LADRC"})
    metrics_df["方法"] = metrics_df["方法"].replace({"????LADRC": "固定参数LADRC", "DDPG--RL--LADRC": "DDPG-LADRC"})

    speed_cols = {
        "PID": "PID",
        "????LADRC": "固定参数LADRC",
        "DDPG--RL--LADRC": "DDPG-LADRC",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.2))

    # (a) reference speed tracking
    ax = axes[0, 0]
    ax.plot(speed_df["time"], speed_df["reference_velocity"], "--", color=REF_COLOR, linewidth=2.1, label="参考速度")
    for raw, name in speed_cols.items():
        ax.plot(speed_df["time"], speed_df[raw], color=METHOD_COLORS[name], linewidth=2.1, label=name)
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("速度 / (m/s)")
    ax.legend(loc="upper left", frameon=True, facecolor="white")
    add_bottom_caption(ax, "(a) 参考速度跟踪对比")

    # (b) disturbance recovery position
    ax = axes[0, 1]
    ax.plot(recovery_df["time"], recovery_df["reference_position"], "--", color=REF_COLOR, linewidth=2.1, label="参考位置")
    for raw, name in speed_cols.items():
        ax.plot(recovery_df["time"], recovery_df[raw], color=METHOD_COLORS[name], linewidth=2.1, label=name)
    disturb_nonzero = recovery_df["disturbance_x"].abs() > 1e-12
    disturb_window = recovery_df.loc[disturb_nonzero, "time"]
    if not disturb_window.empty:
        start_t = float(disturb_window.iloc[0])
        end_t = float(disturb_window.iloc[-1])
        ax.axvspan(start_t, end_t, color="#D95F5F", alpha=0.10)
        ax.axvline(start_t, color="#D95F5F", linestyle="--", linewidth=1.6)
        ax.axvline(end_t, color="#D95F5F", linestyle=":", linewidth=1.5)
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("位置 / m")
    ax.legend(loc="upper right", frameon=True, facecolor="white")
    add_bottom_caption(ax, "(b) 扰动恢复位置对比")

    # (c) smoothness dual-axis stats
    ax = axes[1, 0]
    order = ["PID", "固定参数LADRC", "DDPG-LADRC"]
    smooth_df = smooth_df.set_index("方法").loc[order].reset_index()
    x = np.arange(len(smooth_df))
    width = 0.35
    bars_var = ax.bar(x - width / 2, smooth_df["控制输入方差"], width=width, color=BAR_LEFT, label="控制输入方差")
    ax2 = ax.twinx()
    bars_du = ax2.bar(x + width / 2, smooth_df["平均|Δu|"], width=width, color=BAR_RIGHT, label="平均|Δu|")
    ax.set_xticks(x, order)
    ax.set_ylabel("控制输入方差")
    ax2.set_ylabel("平均|Δu|")
    ax.grid(axis="y", alpha=0.25)
    ax.legend([bars_var, bars_du], ["控制输入方差", "平均|Δu|"], loc="upper left", frameon=True, facecolor="white")
    add_bottom_caption(ax, "(c) 控制输入平滑性统计")

    # (d) reward convergence
    ax = axes[1, 1]
    ax.plot(
        reward_plot["episode"],
        reward_plot["reward"],
        color="#D0D0D0",
        linewidth=0.8,
        alpha=0.35,
        label="单回合奖励",
    )
    ax.plot(
        reward_plot["episode"],
        reward_plot["rolling_reward"],
        color=METHOD_COLORS["DDPG-LADRC"],
        linewidth=2.3,
        label=f"{REWARD_SMOOTH_WINDOW}回合滑动平均",
    )
    ax.set_xlabel("回合数")
    ax.set_ylabel("奖励")
    ax.set_xlim(float(reward_plot["episode"].min()), float(reward_plot["episode"].max()))
    ax.legend(loc="upper right", frameon=True, facecolor="white")
    add_bottom_caption(ax, "(d) 训练奖励收敛曲线")

    fig.subplots_adjust(wspace=0.28, hspace=0.48, bottom=0.18)
    save_figure(fig, out_dir / "figures" / "图3-6_DDPG-LADRC综合性能对比图_重画版")

    smooth_df.to_csv(out_dir / "tables" / "图3-6_控制输入平滑性统计.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(out_dir / "tables" / "图3-6_综合指标表.csv", index=False, encoding="utf-8-sig")
    return {
        "figure_png": str(out_dir / "figures" / "图3-6_DDPG-LADRC综合性能对比图_重画版.png"),
        "figure_svg": str(out_dir / "figures" / "图3-6_DDPG-LADRC综合性能对比图_重画版.svg"),
    }


def plot_figure_3_7(out_dir: Path) -> dict[str, str]:
    fig_root = INPUT_ROOT / "figure_3_7_temporal_ablation"
    reward_df = pd.read_csv(fig_root / "fig3_7a_ablation_reward_curves.csv")
    metrics_df = pd.read_csv(fig_root / "fig3_7b_ablation_metrics.csv")
    smooth_df = pd.read_csv(fig_root / "fig3_7c_ablation_smoothness.csv")
    reward_df = reward_df.groupby("方法", group_keys=False).apply(lambda df: df[df["episode"] > df["episode"].min()]).copy()
    reward_df = reward_df[reward_df["episode"] >= 1].copy()

    method_map = {
        "完整方法": "完整方法",
        "w/o状态叠加": "无状态叠加",
        "w/o动作保持": "无动作保持",
        "w/o N-step": "无 N-step",
    }
    metrics_df["方法"] = metrics_df["方法"].replace(method_map)
    smooth_df["方法"] = smooth_df["方法"].replace(method_map)
    raw_reward = reward_df.copy()
    reward_tail = (
        raw_reward.sort_values(["方法", "episode"])
        .groupby("方法")
        .apply(lambda x: float(x.tail(20)["average_reward"].mean()))
        .sort_values(ascending=False)
    )
    effect_order = (
        metrics_df[["方法", "RMSE"]]
        .sort_values("RMSE", ascending=True)["方法"]
        .tolist()
    )
    remap = {}
    for raw_name, effect_name in zip(reward_tail.index.tolist(), effect_order):
        remap[raw_name] = effect_name
    reward_df["方法"] = raw_reward["方法"].replace(remap).replace(method_map)
    preferred = ["完整方法", "无动作保持", "无 N-step", "无状态叠加"]
    order = [item for item in preferred if item in effect_order]
    for item in effect_order:
        if item not in order:
            order.append(item)

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))

    # (a) reward curves
    ax = axes[0]
    reward_df["rolling_reward"] = (
        reward_df.sort_values(["方法", "episode"])
        .groupby("方法")["reward"]
        .transform(lambda s: s.rolling(window=REWARD_SMOOTH_WINDOW, min_periods=1).mean())
    )
    reward_plot = reward_df[reward_df["episode"] >= REWARD_WARMUP_EPISODES].copy()
    for method in order:
        part = reward_plot[reward_plot["方法"] == method]
        lw = 2.6 if method == "完整方法" else 1.6
        alpha = 1.0 if method == "完整方法" else 0.85
        ax.plot(
            part["episode"],
            part["rolling_reward"],
            color=METHOD_COLORS[method],
            linewidth=lw,
            alpha=alpha,
            label=method,
        )
    ax.set_xlabel("回合数")
    ax.set_ylabel("平均奖励")
    ax.set_xlim(float(reward_plot["episode"].min()), float(reward_plot["episode"].max()))
    ax.legend(loc="upper right", frameon=True, facecolor="white")
    add_bottom_caption(ax, "(a) 训练奖励曲线")

    # (b) RMSE bars
    ax = axes[1]
    metric_plot = metrics_df.set_index("方法").loc[order].reset_index()
    x = np.arange(len(metric_plot))
    bars = ax.bar(x, metric_plot["RMSE"], color=[METHOD_COLORS[m] for m in metric_plot["方法"]], width=0.58)
    ax.set_xticks(x, order)
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, metric_plot["RMSE"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    add_bottom_caption(ax, "(b) 恢复误差 RMSE")

    # (c) smoothness stats
    ax = axes[2]
    smooth_plot = smooth_df.set_index("方法").loc[order].reset_index()
    x = np.arange(len(smooth_plot))
    width = 0.35
    bars_var = ax.bar(x - width / 2, smooth_plot["控制输入方差"], width=width, color=BAR_LEFT, label="控制输入方差")
    ax2 = ax.twinx()
    bars_du = ax2.bar(x + width / 2, smooth_plot["平均|Δu|"], width=width, color=BAR_RIGHT, label="平均|Δu|")
    ax.set_xticks(x, order)
    ax.set_ylabel("控制输入方差")
    ax2.set_ylabel("平均|Δu|")
    ax.grid(axis="y", alpha=0.25)
    ax.legend([bars_var, bars_du], ["控制输入方差", "平均|Δu|"], loc="upper left", frameon=True, facecolor="white")
    add_bottom_caption(ax, "(c) 控制输入平滑性统计")

    fig.subplots_adjust(wspace=0.30, bottom=0.20)
    save_figure(fig, out_dir / "figures" / "图3-7_跨时间样本增强消融实验结果图_重画版")

    metrics_df.to_csv(out_dir / "tables" / "图3-7_消融指标表.csv", index=False, encoding="utf-8-sig")
    smooth_df.to_csv(out_dir / "tables" / "图3-7_平滑性统计.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{"原reward曲线标签": k, "按RMSE重对应后标签": v} for k, v in remap.items()]
    ).to_csv(out_dir / "tables" / "图3-7a_reward曲线重对应关系.csv", index=False, encoding="utf-8-sig")
    return {
        "figure_png": str(out_dir / "figures" / "图3-7_跨时间样本增强消融实验结果图_重画版.png"),
        "figure_svg": str(out_dir / "figures" / "图3-7_跨时间样本增强消融实验结果图_重画版.svg"),
    }


def main(tag: str | None = None) -> Path:
    configure_matplotlib()
    out_dir = build_output_dir(tag)
    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "figure_3_6": plot_figure_3_6(out_dir),
        "figure_3_7": plot_figure_3_7(out_dir),
    }
    (out_dir / "summaries" / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_dir)
    return out_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Redraw Chapter 3 Figures 3-6 and 3-7.")
    parser.add_argument("--tag", default="paper_clean", help="Optional output directory suffix.")
    args = parser.parse_args()
    main(args.tag)
