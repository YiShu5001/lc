from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"D:\ZhangC\lc")
DEFAULT_INPUT_ROOT = ROOT / "outputs" / "chapter3_result_data" / "20260415_chapter3_result_data"
DEFAULT_ABLATION_ROOT = (
    ROOT
    / "outputs"
    / "control_pybullet_rl"
    / "x_temporal_ablation_suite"
    / "20260414_temporal_ablation_ep250"
)


def _configure_chinese_font() -> None:
    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((name for name in preferred if name in available), None)
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 220


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _save(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _method_color_map() -> dict[str, str]:
    return {
        "参考": "#222222",
        "参数组A": "#1f77b4",
        "参数组B": "#ff7f0e",
        "参数组C": "#2ca02c",
        "PID": "#1f77b4",
        "固定参数LADRC": "#ff7f0e",
        "DDPG--RL--LADRC": "#2ca02c",
        "完整方法": "#1f77b4",
        "w/o状态叠加": "#d62728",
        "w/o动作保持": "#ff7f0e",
        "w/o N-step": "#9467bd",
    }


def _plot_figure_3_2(input_root: Path, out_dir: Path) -> None:
    fig_root = input_root / "figure_3_2_fixed_ladrc_multi_condition"
    step_df = _read_csv(fig_root / "plot_data" / "fig3_2a_step_response_data.csv")
    disturb_df = _read_csv(fig_root / "plot_data" / "fig3_2b_disturbance_recovery_data.csv")
    error_df = _read_csv(fig_root / "plot_data" / "fig3_2c_constant_speed_error_data.csv")

    colors = _method_color_map()
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))

    # a. step response
    axes[0].plot(step_df.iloc[:, 0], step_df.iloc[:, 1], "--", color=colors["参考"], linewidth=2.0, label="参考")
    for idx, label in enumerate(["参数组A", "参数组B", "参数组C"], start=2):
        axes[0].plot(step_df.iloc[:, 0], step_df.iloc[:, idx], linewidth=2.0, color=colors[label], label=label)
    axes[0].set_title("（a）阶跃响应对比")
    axes[0].set_xlabel("时间 / s")
    axes[0].set_ylabel("位置 / m")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9)

    # b. disturbance recovery
    axes[1].plot(disturb_df.iloc[:, 0], disturb_df.iloc[:, 1], "--", color=colors["参考"], linewidth=2.0, label="参考")
    for idx, label in enumerate(["参数组A", "参数组B", "参数组C"], start=3):
        axes[1].plot(disturb_df.iloc[:, 0], disturb_df.iloc[:, idx], linewidth=2.0, color=colors[label], label=label)
    nonzero = disturb_df.iloc[:, 2].abs() > 1e-12
    if nonzero.any():
        inj_t = disturb_df.loc[nonzero, disturb_df.columns[0]]
        axes[1].axvline(inj_t.iloc[0], color="#d62728", linestyle="--", linewidth=1.6, label="扰动注入")
        axes[1].axvline(inj_t.iloc[-1], color="#d62728", linestyle=":", linewidth=1.4, label="扰动结束")
        axes[1].axvspan(inj_t.iloc[0], inj_t.iloc[-1], color="#d62728", alpha=0.08)
    axes[1].set_title("（b）扰动恢复对比")
    axes[1].set_xlabel("时间 / s")
    axes[1].set_ylabel("位置 / m")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9, ncol=2)

    # c. tracking error
    for idx, label in enumerate(["参数组A", "参数组B", "参数组C"], start=2):
        axes[2].plot(error_df.iloc[:, 0], error_df.iloc[:, idx], linewidth=2.0, color=colors[label], label=label)
    axes[2].axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    axes[2].set_title("（c）恒速参考跟踪误差对比")
    axes[2].set_xlabel("时间 / s")
    axes[2].set_ylabel("误差 / m")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=9)

    fig.suptitle("图3-2 固定参数LADRC多工况性能对比图", fontsize=15, y=1.03)
    _save(fig, out_dir / "图3-2_固定参数LADRC多工况性能对比图")


def _plot_figure_3_5(input_root: Path, out_dir: Path) -> None:
    timeline = _read_csv(input_root / "figure_3_5_timeline" / "experiment_timeline.csv")
    ref_rows = timeline[timeline["type"] == "reference_change"].copy()
    dist_start = timeline[timeline["type"] == "disturbance_start"]["time_s"].dropna()
    dist_end = timeline[timeline["type"] == "disturbance_end"]["time_s"].dropna()
    recovery_rows = timeline[timeline["type"] == "recovery_interval"].copy()

    stage_names = ["前进阶段", "悬停阶段", "反向阶段", "末段保持"]
    stage_starts = ref_rows["time_s"].dropna().tolist()[:4]
    stage_ends = ref_rows["time_s"].dropna().tolist()[1:5]

    fig, ax = plt.subplots(figsize=(12.5, 3.8))
    y_ref, y_dist, y_recover = 3.0, 2.0, 1.0

    stage_colors = ["#4c78a8", "#72b7b2", "#f58518", "#54a24b"]
    for idx, (name, start, end) in enumerate(zip(stage_names, stage_starts, stage_ends)):
        ax.barh(y_ref, end - start, left=start, height=0.35, color=stage_colors[idx], edgecolor="white")
        ax.text((start + end) / 2, y_ref, name, ha="center", va="center", fontsize=9, color="white")

    if not dist_start.empty and not dist_end.empty:
        ax.barh(y_dist, dist_end.iloc[0] - dist_start.iloc[0], left=dist_start.iloc[0], height=0.35, color="#e45756")
        ax.text((dist_start.iloc[0] + dist_end.iloc[0]) / 2, y_dist, "随机外力扰动", ha="center", va="center", fontsize=9, color="white")
        ax.axvline(dist_start.iloc[0], color="#e45756", linestyle="--", linewidth=1.4)
        ax.axvline(dist_end.iloc[0], color="#e45756", linestyle=":", linewidth=1.4)

    for _, row in recovery_rows.iterrows():
        if math.isnan(row.get("start_time_s", np.nan)) or math.isnan(row.get("end_time_s", np.nan)):
            continue
        ax.barh(y_recover, row["end_time_s"] - row["start_time_s"], left=row["start_time_s"], height=0.26, color="#59a14f", alpha=0.75)
    if not recovery_rows.empty:
        ax.text(
            float(recovery_rows.iloc[0]["start_time_s"] + recovery_rows.iloc[0]["end_time_s"]) / 2,
            y_recover,
            "恢复阶段",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
        )

    ax.set_yticks([y_ref, y_dist, y_recover])
    ax.set_yticklabels(["参考工况", "扰动注入", "恢复区间"])
    ax.set_xlabel("时间 / s")
    ax.set_title("图3-5 控制层实验工况与扰动注入时间线图")
    ax.set_xlim(0.0, max(stage_ends))
    ax.grid(axis="x", alpha=0.25)
    _save(fig, out_dir / "图3-5_控制层实验工况与扰动注入时间线图")


def _plot_figure_3_6(input_root: Path, out_dir: Path) -> None:
    fig_root = input_root / "figure_3_6_ddpg_rl_ladrc_compare"
    speed_df = _read_csv(fig_root / "plot_data" / "fig3_6a_speed_tracking_data.csv")
    recovery_df = _read_csv(fig_root / "plot_data" / "fig3_6b_disturbance_recovery_data.csv")
    smooth_df = _read_csv(fig_root / "fig3_6c_control_smoothness_stats.csv")
    reward_df = _read_csv(fig_root / "fig3_6d_training_reward_curve.csv")

    smooth_df.columns = ["方法", "控制输入方差", "平均|Δu|"]
    colors = _method_color_map()
    label_map = {
        speed_df.columns[2]: "PID",
        speed_df.columns[3]: "固定参数LADRC",
        speed_df.columns[4]: "DDPG--RL--LADRC",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # a speed tracking
    axes[0, 0].plot(speed_df.iloc[:, 0], speed_df.iloc[:, 1], "--", color=colors["参考"], linewidth=2.0, label="参考速度")
    for col in speed_df.columns[2:]:
        label = label_map[col]
        axes[0, 0].plot(speed_df.iloc[:, 0], speed_df[col], linewidth=2.0, label=label, color=colors[label])
    axes[0, 0].set_title("（a）参考速度跟踪曲线")
    axes[0, 0].set_xlabel("时间 / s")
    axes[0, 0].set_ylabel("速度 / m/s")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=9)

    # b disturbance recovery
    label_map_b = {
        recovery_df.columns[3]: "PID",
        recovery_df.columns[4]: "固定参数LADRC",
        recovery_df.columns[5]: "DDPG--RL--LADRC",
    }
    axes[0, 1].plot(recovery_df.iloc[:, 0], recovery_df.iloc[:, 1], "--", color=colors["参考"], linewidth=2.0, label="参考位置")
    for col in recovery_df.columns[3:]:
        label = label_map_b[col]
        axes[0, 1].plot(recovery_df.iloc[:, 0], recovery_df[col], linewidth=2.0, label=label, color=colors[label])
    nonzero = recovery_df.iloc[:, 2].abs() > 1e-12
    if nonzero.any():
        inj_t = recovery_df.loc[nonzero, recovery_df.columns[0]]
        axes[0, 1].axvline(inj_t.iloc[0], color="#d62728", linestyle="--", linewidth=1.5, label="扰动注入")
        axes[0, 1].axvline(inj_t.iloc[-1], color="#d62728", linestyle=":", linewidth=1.4, label="扰动结束")
        axes[0, 1].axvspan(inj_t.iloc[0], inj_t.iloc[-1], color="#d62728", alpha=0.08)
    axes[0, 1].set_title("（b）扰动恢复曲线")
    axes[0, 1].set_xlabel("时间 / s")
    axes[0, 1].set_ylabel("位置 / m")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8, ncol=2)

    # c smoothness
    x = np.arange(len(smooth_df))
    width = 0.36
    ax_c = axes[1, 0]
    bars1 = ax_c.bar(x - width / 2, smooth_df["控制输入方差"], width=width, color="#4c78a8", label="控制输入方差")
    ax_c2 = ax_c.twinx()
    bars2 = ax_c2.bar(x + width / 2, smooth_df["平均|Δu|"], width=width, color="#f58518", label="平均|Δu|")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(smooth_df["方法"], rotation=0)
    ax_c.set_title("（c）控制输入平滑性")
    ax_c.set_ylabel("控制输入方差")
    ax_c2.set_ylabel("平均|Δu|")
    ax_c.grid(axis="y", alpha=0.2)
    ax_c.legend([bars1, bars2], ["控制输入方差", "平均|Δu|"], loc="upper left", fontsize=9)

    # d reward convergence
    axes[1, 1].plot(reward_df["episode"], reward_df["reward"], color="#bdbdbd", linewidth=1.0, alpha=0.6, label="单回合奖励")
    axes[1, 1].plot(reward_df["episode"], reward_df["average_reward"], color="#2ca02c", linewidth=2.0, label="平均奖励")
    axes[1, 1].set_title("（d）DDPG训练奖励收敛曲线")
    axes[1, 1].set_xlabel("回合数")
    axes[1, 1].set_ylabel("奖励")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=9)

    fig.suptitle("图3-6 DDPG--RL--LADRC综合性能对比图", fontsize=15, y=1.02)
    _save(fig, out_dir / "图3-6_DDPG--RL--LADRC综合性能对比图")


def _plot_figure_3_7(input_root: Path, out_dir: Path, ablation_root: Path) -> None:
    fig_root = input_root / "figure_3_7_temporal_ablation"
    reward_df = _read_csv(fig_root / "fig3_7a_ablation_reward_curves.csv")
    metrics_df = _read_csv(fig_root / "fig3_7b_ablation_metrics.csv")
    smooth_df = _read_csv(fig_root / "fig3_7c_ablation_smoothness.csv")

    reward_df.columns = ["方法", "episode", "reward", "average_reward"]
    metrics_df.columns = ["方法", "RMSE", "ITAE", "最大偏差", "恢复时间", "控制输入方差", "平均|Δu|", "MAE", "velocity_rmse"]
    smooth_df.columns = ["方法", "控制输入方差", "平均|Δu|"]

    method_order = ["完整方法", "w/o状态叠加", "w/o动作保持", "w/o N-step"]
    colors = _method_color_map()

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))

    # a reward curves
    for method in method_order:
        part = reward_df[reward_df["方法"] == method]
        axes[0].plot(part["episode"], part["average_reward"], linewidth=2.0, label=method, color=colors[method])
    axes[0].set_title("（a）训练奖励曲线")
    axes[0].set_xlabel("回合数")
    axes[0].set_ylabel("平均奖励")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9)

    # b rmse bar
    metric_plot = metrics_df.set_index("方法").loc[method_order].reset_index()
    x = np.arange(len(metric_plot))
    axes[1].bar(x, metric_plot["RMSE"], color=[colors[m] for m in metric_plot["方法"]], width=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metric_plot["方法"], rotation=15)
    axes[1].set_title("（b）恢复误差对比")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(axis="y", alpha=0.25)

    # c smoothness
    smooth_plot = smooth_df.set_index("方法").loc[method_order].reset_index()
    x2 = np.arange(len(smooth_plot))
    width = 0.35
    axes[2].bar(x2 - width / 2, smooth_plot["控制输入方差"], width=width, color="#4c78a8", label="控制输入方差")
    ax2b = axes[2].twinx()
    ax2b.bar(x2 + width / 2, smooth_plot["平均|Δu|"], width=width, color="#f58518", label="平均|Δu|")
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels(smooth_plot["方法"], rotation=15)
    axes[2].set_title("（c）控制输入平滑性")
    axes[2].set_ylabel("控制输入方差")
    ax2b.set_ylabel("平均|Δu|")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(["控制输入方差", "平均|Δu|"], loc="upper left", fontsize=9)

    fig.suptitle("图3-7 跨时间样本增强消融实验结果图", fontsize=15, y=1.03)
    _save(fig, out_dir / "图3-7_跨时间样本增强消融实验结果图")

    # Chinese heatmaps
    summary_path = ablation_root / "temporal_ablation_summary.csv"
    if summary_path.exists():
        summary_df = _read_csv(summary_path)
        summary_df["家族"] = summary_df["family"].map(
            {
                "single_action_hold": "仅动作保持",
                "single_state_stack": "仅状态叠加",
                "single_n_step": "仅N步自举",
                "pair_action_hold_state_stack": "动作保持+状态叠加",
                "pair_action_hold_n_step": "动作保持+N步自举",
                "pair_state_stack_n_step": "状态叠加+N步自举",
                "full_temporal": "完整方法",
            }
        )
        reward_pivot = summary_df.pivot(index="家族", columns="k", values="reward_loss_rate_vs_full")
        rmse_pivot = summary_df.pivot(index="家族", columns="k", values="best_eval_rmse")
        for pivot, title, cmap, out_name in [
            (reward_pivot, "消融实验奖励损失率热力图", "YlOrRd", "图3-7_消融实验奖励损失率热力图"),
            (rmse_pivot, "消融实验RMSE热力图", "YlGnBu", "图3-7_消融实验RMSE热力图"),
        ]:
            fig_h, ax_h = plt.subplots(figsize=(8.5, 5.2))
            im = ax_h.imshow(pivot.values, cmap=cmap, aspect="auto")
            ax_h.set_xticks(np.arange(pivot.shape[1]))
            ax_h.set_xticklabels([f"k={v}" for v in pivot.columns])
            ax_h.set_yticks(np.arange(pivot.shape[0]))
            ax_h.set_yticklabels(pivot.index)
            ax_h.set_title(title)
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    text = "—" if pd.isna(val) else f"{val:.3f}"
                    ax_h.text(j, i, text, ha="center", va="center", fontsize=8, color="black")
            fig_h.colorbar(im, ax=ax_h, shrink=0.9)
            _save(fig_h, out_dir / out_name)


def _copy_existing_heatmaps(input_root: Path, out_dir: Path) -> None:
    src_root = input_root / "figure_3_7_temporal_ablation"
    mapping = {
        "ablation_heatmap_reward_loss.png": "原始_消融奖励损失率热力图.png",
        "ablation_heatmap_rmse.png": "原始_消融RMSE热力图.png",
    }
    for src_name, dst_name in mapping.items():
        src = src_root / src_name
        if src.exists():
            shutil.copy2(src, out_dir / dst_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot final chapter 3 figures from exported data.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    _configure_chinese_font()

    input_root = args.input_root.resolve()
    out_dir = (args.output_dir or (input_root / "final_figures")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_figure_3_2(input_root, out_dir)
    _plot_figure_3_5(input_root, out_dir)
    _plot_figure_3_6(input_root, out_dir)
    _plot_figure_3_7(input_root, out_dir, args.ablation_root.resolve())
    _copy_existing_heatmaps(input_root, out_dir)

    manifest = {
        "input_root": str(input_root),
        "ablation_root": str(args.ablation_root.resolve()),
        "output_dir": str(out_dir),
        "figures": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
    }
    (out_dir / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
