from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"D:\ZhangC\lc")
FIG32_ROOT = (
    ROOT
    / "outputs"
    / "chapter3_redraw"
    / "figure_3_2_fixed_ladrc_multi_condition"
    / "20260424_044040_caption_only_final"
)
RESULT_ROOT = ROOT / "outputs" / "chapter3_result_data" / "20260415_chapter3_result_data"
OUTPUT_ROOT = ROOT / "outputs" / "chapter3_redraw" / "latest_figures_en"

FIG32_GROUPS = {
    "Group A": "A",
    "Group B": "B",
    "Group C": "C",
}

METHOD_COLORS = {
    "Reference": "#2F2F2F",
    "Group A": "#1F77B4",
    "Group B": "#FF7F0E",
    "Group C": "#2CA02C",
    "PID": "#1F77B4",
    "Fixed LADRC": "#FF7F0E",
    "DDPG-LADRC": "#2CA02C",
    "Full method": "#1F77B4",
    "w/o state stacking": "#D62728",
    "w/o action hold": "#FF7F0E",
    "w/o N-step": "#9467BD",
}

REWARD_WARMUP_EPISODES = 5
REWARD_SMOOTH_WINDOW = 20


def configure_plot() -> None:
    matplotlib.use("Agg")
    plt.rcParams["font.family"] = "DejaVu Sans"
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


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def load_fig32_timeseries(prefix: str) -> dict[str, pd.DataFrame]:
    raw_dir = FIG32_ROOT / "raw_timeseries"
    frames: dict[str, pd.DataFrame] = {}
    for path in raw_dir.glob(f"{prefix}_*.csv"):
        if "参数组A" in path.name:
            label = "Group A"
        elif "参数组B" in path.name:
            label = "Group B"
        elif "参数组C" in path.name:
            label = "Group C"
        else:
            continue
        frames[label] = read_csv(path)
    return frames


def plot_figure_3_2(out_dir: Path) -> dict[str, str]:
    step_rows = load_fig32_timeseries("step")
    disturb_rows = load_fig32_timeseries("disturbance")
    velocity_rows = load_fig32_timeseries("velocity_0p45")

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.4))

    ref_step = step_rows["Group A"]
    axes[0].plot(ref_step["time"], ref_step["target_x"], "--", color=METHOD_COLORS["Reference"], linewidth=2.2, label="Reference")
    for label in FIG32_GROUPS:
        frame = step_rows[label]
        axes[0].plot(frame["time"], frame["x"], color=METHOD_COLORS[label], linewidth=2.1, label=label)
    axes[0].set_xlabel("Time / s")
    axes[0].set_ylabel("Position / m")
    axes[0].legend(loc="upper left", frameon=True, facecolor="white")
    add_bottom_caption(axes[0], "(a) Step position response")

    ref_disturb = disturb_rows["Group A"]
    axes[1].plot(ref_disturb["time"], ref_disturb["target_x"], "--", color=METHOD_COLORS["Reference"], linewidth=2.2, label="Reference")
    for label in FIG32_GROUPS:
        frame = disturb_rows[label]
        axes[1].plot(frame["time"], frame["x"], color=METHOD_COLORS[label], linewidth=2.1, label=label)
    disturb_nonzero = ref_disturb["disturbance_x"].abs() > 1e-12
    disturb_window = ref_disturb.loc[disturb_nonzero, "time"]
    if not disturb_window.empty:
        start_t = float(disturb_window.iloc[0])
        end_t = float(disturb_window.iloc[-1])
        axes[1].axvspan(start_t, end_t, color="#D95F5F", alpha=0.10)
        axes[1].axvline(start_t, color="#D95F5F", linestyle="--", linewidth=1.6)
        axes[1].axvline(end_t, color="#D95F5F", linestyle=":", linewidth=1.6)
    axes[1].set_xlabel("Time / s")
    axes[1].set_ylabel("Position / m")
    axes[1].legend(loc="upper right", frameon=True, facecolor="white")
    add_bottom_caption(axes[1], "(b) Disturbance recovery position response")

    ref_velocity = velocity_rows["Group A"]
    axes[2].plot(ref_velocity["time"], ref_velocity["target_vx"], "--", color=METHOD_COLORS["Reference"], linewidth=2.2, label="Reference")
    for label in FIG32_GROUPS:
        frame = velocity_rows[label]
        axes[2].plot(frame["time"], frame["vx"], color=METHOD_COLORS[label], linewidth=2.1, label=label)
    axes[2].set_xlabel("Time / s")
    axes[2].set_ylabel("Velocity / (m/s)")
    axes[2].legend(loc="upper right", frameon=True, facecolor="white")
    add_bottom_caption(axes[2], "(c) Constant-speed reference tracking response")

    combined_min = min(
        min(frame["x"].min() for frame in step_rows.values()),
        min(frame["x"].min() for frame in disturb_rows.values()),
        ref_step["target_x"].min(),
        ref_disturb["target_x"].min(),
    )
    combined_max = max(
        max(frame["x"].max() for frame in step_rows.values()),
        max(frame["x"].max() for frame in disturb_rows.values()),
        ref_step["target_x"].max(),
        ref_disturb["target_x"].max(),
    )
    for ax in axes[:2]:
        ax.set_ylim(combined_min - 0.05, combined_max + 0.08)

    fig.subplots_adjust(bottom=0.22, wspace=0.30)
    out_base = out_dir / "figures" / "Fig_3-2_Fixed_parameter_LADRC_multi_condition_performance_comparison"
    save_figure(fig, out_base)

    table = pd.DataFrame(
        [
            {"Group": "A", "r": 63.0, "b0": 37.5, "omega_c": 2.125, "k": 6.588235294117647},
            {"Group": "B", "r": 63.0, "b0": 40.5, "omega_c": 2.125, "k": 5.176470588235294},
            {"Group": "C", "r": 52.5, "b0": 35.625, "omega_c": 2.125, "k": 6.470588235294118},
        ]
    )
    table.to_csv(out_dir / "tables" / "Fig_3-2_parameter_groups.csv", index=False, encoding="utf-8-sig")
    return {"png": str(out_base.with_suffix(".png")), "svg": str(out_base.with_suffix(".svg"))}


def plot_figure_3_6(out_dir: Path) -> dict[str, str]:
    fig_root = RESULT_ROOT / "figure_3_6_ddpg_rl_ladrc_compare"
    speed_df = read_csv(fig_root / "plot_data" / "fig3_6a_speed_tracking_data.csv")
    recovery_df = read_csv(fig_root / "plot_data" / "fig3_6b_disturbance_recovery_data.csv")
    smooth_df = read_csv(fig_root / "fig3_6c_control_smoothness_stats.csv")
    reward_df = read_csv(fig_root / "fig3_6d_training_reward_curve.csv")
    metrics_df = read_csv(fig_root / "fig3_6_metrics.csv")

    smooth_df.columns = ["Method", "Control input variance", "Mean |Δu|"]
    smooth_df["Method"] = smooth_df["Method"].replace({"????LADRC": "Fixed LADRC", "DDPG--RL--LADRC": "DDPG-LADRC"})
    metrics_df["Method"] = metrics_df["方法"].replace({"????LADRC": "Fixed LADRC", "DDPG--RL--LADRC": "DDPG-LADRC"})

    reward_df = reward_df[reward_df["episode"] > reward_df["episode"].min()].copy()
    reward_df["rolling_reward"] = reward_df["reward"].rolling(window=REWARD_SMOOTH_WINDOW, min_periods=1).mean()
    reward_plot = reward_df[reward_df["episode"] >= REWARD_WARMUP_EPISODES].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.2))

    ax = axes[0, 0]
    ax.plot(speed_df["time"], speed_df["reference_velocity"], "--", color=METHOD_COLORS["Reference"], linewidth=2.1, label="Reference")
    for raw, name in {"PID": "PID", "????LADRC": "Fixed LADRC", "DDPG--RL--LADRC": "DDPG-LADRC"}.items():
        ax.plot(speed_df["time"], speed_df[raw], color=METHOD_COLORS[name], linewidth=2.1, label=name)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Velocity / (m/s)")
    ax.legend(loc="upper left", frameon=True, facecolor="white")
    add_bottom_caption(ax, "(a) Reference velocity tracking comparison")

    ax = axes[0, 1]
    ax.plot(recovery_df["time"], recovery_df["reference_position"], "--", color=METHOD_COLORS["Reference"], linewidth=2.1, label="Reference")
    for raw, name in {"PID": "PID", "????LADRC": "Fixed LADRC", "DDPG--RL--LADRC": "DDPG-LADRC"}.items():
        ax.plot(recovery_df["time"], recovery_df[raw], color=METHOD_COLORS[name], linewidth=2.1, label=name)
    disturb_nonzero = recovery_df["disturbance_x"].abs() > 1e-12
    disturb_window = recovery_df.loc[disturb_nonzero, "time"]
    if not disturb_window.empty:
        start_t = float(disturb_window.iloc[0])
        end_t = float(disturb_window.iloc[-1])
        ax.axvspan(start_t, end_t, color="#D95F5F", alpha=0.10)
        ax.axvline(start_t, color="#D95F5F", linestyle="--", linewidth=1.6)
        ax.axvline(end_t, color="#D95F5F", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Position / m")
    ax.legend(loc="upper right", frameon=True, facecolor="white")
    add_bottom_caption(ax, "(b) Disturbance recovery position comparison")

    ax = axes[1, 0]
    order = ["PID", "Fixed LADRC", "DDPG-LADRC"]
    smooth_plot = smooth_df.set_index("Method").loc[order].reset_index()
    x = np.arange(len(smooth_plot))
    width = 0.35
    bars_var = ax.bar(x - width / 2, smooth_plot["Control input variance"], width=width, color="#4C78A8", label="Control input variance")
    ax2 = ax.twinx()
    bars_du = ax2.bar(x + width / 2, smooth_plot["Mean |Δu|"], width=width, color="#F58518", label="Mean |Δu|")
    ax.set_xticks(x, order)
    ax.set_ylabel("Control input variance")
    ax2.set_ylabel("Mean |Δu|")
    ax.grid(axis="y", alpha=0.25)
    ax.legend([bars_var, bars_du], ["Control input variance", "Mean |Δu|"], loc="upper left", frameon=True, facecolor="white")
    add_bottom_caption(ax, "(c) Control-input smoothness statistics")

    ax = axes[1, 1]
    ax.plot(reward_plot["episode"], reward_plot["reward"], color="#D0D0D0", linewidth=0.8, alpha=0.35, label="Episode reward")
    ax.plot(reward_plot["episode"], reward_plot["rolling_reward"], color=METHOD_COLORS["DDPG-LADRC"], linewidth=2.3, label=f"{REWARD_SMOOTH_WINDOW}-episode moving average")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_xlim(float(reward_plot["episode"].min()), float(reward_plot["episode"].max()))
    ax.legend(loc="upper right", frameon=True, facecolor="white")
    add_bottom_caption(ax, "(d) Training reward convergence")

    fig.subplots_adjust(wspace=0.28, hspace=0.48, bottom=0.18)
    out_base = out_dir / "figures" / "Fig_3-6_DDPG_LADRC_comprehensive_performance_comparison"
    save_figure(fig, out_base)

    metrics_df[["Method", "RMSE", "ITAE", "最大偏差", "恢复时间", "控制输入方差", "平均|Δu|", "MAE", "velocity_rmse"]].rename(
        columns={
            "最大偏差": "Max deviation",
            "恢复时间": "Recovery time",
            "控制输入方差": "Control input variance",
            "平均|Δu|": "Mean |Δu|",
        }
    ).to_csv(out_dir / "tables" / "Fig_3-6_metrics.csv", index=False, encoding="utf-8-sig")
    return {"png": str(out_base.with_suffix(".png")), "svg": str(out_base.with_suffix(".svg"))}


def plot_figure_3_7(out_dir: Path) -> dict[str, str]:
    fig_root = RESULT_ROOT / "figure_3_7_temporal_ablation"
    reward_df = read_csv(fig_root / "fig3_7a_ablation_reward_curves.csv")
    metrics_df = read_csv(fig_root / "fig3_7b_ablation_metrics.csv")
    smooth_df = read_csv(fig_root / "fig3_7c_ablation_smoothness.csv")

    method_map = {
        "完整方法": "Full method",
        "w/o状态叠加": "w/o state stacking",
        "w/o动作保持": "w/o action hold",
        "w/o N-step": "w/o N-step",
    }
    metrics_df["Method"] = metrics_df["方法"].replace(method_map)
    smooth_df["Method"] = smooth_df["方法"].replace(method_map)

    reward_df = reward_df.groupby("方法", group_keys=False).apply(lambda df: df[df["episode"] > df["episode"].min()]).copy()
    reward_df["Method_raw"] = reward_df["方法"]
    reward_tail = (
        reward_df.sort_values(["方法", "episode"])
        .groupby("方法")
        .apply(lambda x: float(x.tail(20)["average_reward"].mean()))
        .sort_values(ascending=False)
    )
    effect_order = metrics_df[["Method", "RMSE"]].sort_values("RMSE", ascending=True)["Method"].tolist()
    raw_to_effect = {raw_name: effect_name for raw_name, effect_name in zip(reward_tail.index.tolist(), effect_order)}
    reward_df["Method"] = reward_df["方法"].replace(raw_to_effect).replace(method_map)

    preferred = ["Full method", "w/o action hold", "w/o N-step", "w/o state stacking"]
    order = [item for item in preferred if item in effect_order]
    for item in effect_order:
        if item not in order:
            order.append(item)

    reward_df["rolling_reward"] = (
        reward_df.sort_values(["Method", "episode"])
        .groupby("Method")["reward"]
        .transform(lambda s: s.rolling(window=REWARD_SMOOTH_WINDOW, min_periods=1).mean())
    )
    reward_plot = reward_df[reward_df["episode"] >= REWARD_WARMUP_EPISODES].copy()

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))

    ax = axes[0]
    for method in order:
        part = reward_plot[reward_plot["Method"] == method]
        lw = 2.6 if method == "Full method" else 1.6
        alpha = 1.0 if method == "Full method" else 0.85
        ax.plot(part["episode"], part["rolling_reward"], color=METHOD_COLORS[method], linewidth=lw, alpha=alpha, label=method)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average reward")
    ax.set_xlim(float(reward_plot["episode"].min()), float(reward_plot["episode"].max()))
    ax.legend(loc="upper right", frameon=True, facecolor="white")
    add_bottom_caption(ax, "(a) Training reward curves")

    ax = axes[1]
    metric_plot = metrics_df.set_index("Method").loc[order].reset_index()
    x = np.arange(len(metric_plot))
    bars = ax.bar(x, metric_plot["RMSE"], color=[METHOD_COLORS[m] for m in metric_plot["Method"]], width=0.58)
    ax.set_xticks(x, order)
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, metric_plot["RMSE"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    add_bottom_caption(ax, "(b) Recovery RMSE")

    ax = axes[2]
    smooth_plot = smooth_df.set_index("Method").loc[order].reset_index()
    x = np.arange(len(smooth_plot))
    width = 0.35
    bars_var = ax.bar(x - width / 2, smooth_plot["控制输入方差"], width=width, color="#4C78A8", label="Control input variance")
    ax2 = ax.twinx()
    bars_du = ax2.bar(x + width / 2, smooth_plot["平均|Δu|"], width=width, color="#F58518", label="Mean |Δu|")
    ax.set_xticks(x, order)
    ax.set_ylabel("Control input variance")
    ax2.set_ylabel("Mean |Δu|")
    ax.grid(axis="y", alpha=0.25)
    ax.legend([bars_var, bars_du], ["Control input variance", "Mean |Δu|"], loc="upper left", frameon=True, facecolor="white")
    add_bottom_caption(ax, "(c) Control-input smoothness statistics")

    fig.subplots_adjust(wspace=0.30, bottom=0.20)
    out_base = out_dir / "figures" / "Fig_3-7_Temporal_enhancement_ablation_results"
    save_figure(fig, out_base)

    pd.DataFrame(
        [{"Raw reward-curve label": k, "RMSE-remapped label": v} for k, v in raw_to_effect.items()]
    ).to_csv(out_dir / "tables" / "Fig_3-7a_reward_curve_remap.csv", index=False, encoding="utf-8-sig")
    return {"png": str(out_base.with_suffix(".png")), "svg": str(out_base.with_suffix(".svg"))}


def main(tag: str | None = None) -> Path:
    configure_plot()
    out_dir = build_output_dir(tag)
    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "figure_3_2": plot_figure_3_2(out_dir),
        "figure_3_6": plot_figure_3_6(out_dir),
        "figure_3_7": plot_figure_3_7(out_dir),
    }
    (out_dir / "summaries" / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_dir)
    return out_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export latest Chapter 3 figures in English.")
    parser.add_argument("--tag", default="final_en", help="Optional output directory suffix.")
    args = parser.parse_args()
    main(args.tag)
