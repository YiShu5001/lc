from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
PAPER_ROOT = OUTPUT_ROOT / "paper_figures" / "chapter3"
DATA_ROOT = PAPER_ROOT / "data"
FIG_ROOT = PAPER_ROOT / "figures"

SCAN_DIR = OUTPUT_ROOT / "control" / "x_axis_rl_refline__exp-bestcfg-scan-v1-to-v10__ep-500__v-1-10__noise-linear-0.1-to-0.04__net-768__drop-0.25"

COLORS = {
    "pid": "#4C78A8",
    "ladrc": "#F58518",
    "ddpg": "#54A24B",
    "mddpg_v1": "#E45756",
    "mddpg_v4": "#72B7B2",
    "mddpg_v7": "#B279A2",
    "tune_a": "#4C78A8",
    "tune_b": "#F58518",
    "tune_c": "#54A24B",
}

PID_BASELINE_DIR = OUTPUT_ROOT / "control_pybullet_compare_fixed" / "x"


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "lines.linewidth": 2.2,
        }
    )


def _save(fig: plt.Figure, stem: str) -> dict[str, str]:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    svg = FIG_ROOT / f"{stem}.svg"
    png = FIG_ROOT / f"{stem}.png"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"svg": str(svg), "png": str(png)}


def plot_status_matrix(status_df: pd.DataFrame) -> dict[str, str]:
    order = list(status_df["experiment_id"])
    status_to_value = {"available_data": 3, "partial": 2, "needs_run": 1, "planned": 0}
    values = [status_to_value.get(item, 0) for item in status_df["status"]]
    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(range(len(order)), values, color=["#72B7B2" if v == 3 else "#F2CF63" if v == 2 else "#E45756" if v == 1 else "#B0B7C3" for v in values])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylim(-0.1, 3.5)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["planned", "needs run", "partial", "available"])
    ax.set_title("Chapter 3 experiment readiness matrix")
    for bar, title in zip(bars, status_df["title"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.06, title, rotation=90, ha="center", va="bottom", fontsize=8)
    return _save(fig, "fig00_experiment_status_matrix")


def plot_main_available(master_df: pd.DataFrame) -> dict[str, str]:
    subset = master_df[
        (master_df["source_group"] == "x_refline_best_scan")
        & (master_df["variant"].isin(["pid", "ladrc", "mddpg_v1", "mddpg_v4", "mddpg_v7"]))
    ].copy()
    subset["display"] = subset["variant"].map(
        {
            "pid": "PID",
            "ladrc": "LADRC",
            "mddpg_v1": "mDDPG(v=1)",
            "mddpg_v4": "mDDPG(v=4)",
            "mddpg_v7": "mDDPG(v=7)",
        }
    )
    metrics = [("rmse", "RMSE"), ("iae", "IAE"), ("reward", "Reward")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, (column, title) in zip(axes, metrics):
        ax.bar(
            subset["display"],
            subset[column],
            color=[COLORS["pid"], COLORS["ladrc"], COLORS["mddpg_v1"], COLORS["mddpg_v4"], COLORS["mddpg_v7"]],
        )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Available x-axis RL RefLine main comparison")
    return _save(fig, "fig01_main_methods_available")


def plot_v_sweep() -> dict[str, str]:
    sweep = pd.read_csv(SCAN_DIR / "mddpg_shared_value_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].plot(sweep["shared_value"], sweep["rmse"], marker="o", color=COLORS["mddpg_v4"])
    axes[0].scatter([4], [float(sweep.loc[sweep["shared_value"] == 4, "rmse"].iloc[0])], color="#E45756", s=70, zorder=3)
    axes[0].set_title("Shared value sweep: RMSE")
    axes[0].set_xlabel("Shared value v")
    axes[0].set_ylabel("RMSE")
    axes[1].plot(sweep["shared_value"], sweep["reward"], marker="o", color=COLORS["mddpg_v7"])
    axes[1].scatter([4], [float(sweep.loc[sweep["shared_value"] == 4, "reward"].iloc[0])], color="#E45756", s=70, zorder=3)
    axes[1].set_title("Shared value sweep: Reward")
    axes[1].set_xlabel("Shared value v")
    axes[1].set_ylabel("Reward")
    fig.suptitle("mDDPG shared enhancement sweep on x-axis RL RefLine")
    return _save(fig, "fig02_v_sweep_rmse_reward")


def plot_reward_curves() -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    mapping = {1: COLORS["mddpg_v1"], 4: COLORS["mddpg_v4"], 7: COLORS["mddpg_v7"]}
    for shared_value in (1, 4, 7):
        frame = pd.read_csv(SCAN_DIR / f"training_mddpg_v{shared_value}.csv")
        reward = frame["reward"].astype(float)
        smooth = reward.rolling(window=10, min_periods=1).mean()
        ax.plot(frame["episode"], smooth, label=f"mDDPG(v={shared_value})", color=mapping[shared_value])
    ax.set_title("Reward curves of selected shared values")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (10-episode rolling mean)")
    ax.legend(frameon=False)
    return _save(fig, "fig03_reward_curves_v1_v4_v7")


def plot_best_time_response() -> dict[str, str]:
    frame = pd.read_csv(SCAN_DIR / "v4" / "eval_timeseries.csv")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(frame["step"], frame["reference"], label="Reference", color="#111111")
    axes[0].plot(frame["step"], frame["output"], label="Output", color=COLORS["mddpg_v4"])
    axes[0].set_ylabel("Position")
    axes[0].legend(frameon=False)
    axes[0].set_title("Best shared value v=4 time response")
    axes[1].plot(frame["step"], frame["error"], color=COLORS["mddpg_v4"])
    axes[1].set_ylabel("Error")
    axes[2].plot(frame["step"], frame["control"], color=COLORS["mddpg_v4"])
    axes[2].plot(frame["step"], frame["disturbance"], color="#E45756", alpha=0.7, label="Disturbance")
    axes[2].set_ylabel("Control / Disturbance")
    axes[2].set_xlabel("Step")
    axes[2].legend(frameon=False)
    return _save(fig, "fig04_best_v4_time_response")


def plot_selected_time_responses() -> dict[str, str]:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for shared_value, color in [(1, COLORS["mddpg_v1"]), (4, COLORS["mddpg_v4"]), (7, COLORS["mddpg_v7"])]:
        frame = pd.read_csv(SCAN_DIR / f"v{shared_value}" / "eval_timeseries.csv")
        axes[0].plot(frame["step"], frame["output"], color=color, label=f"Output v={shared_value}")
        axes[1].plot(frame["step"], frame["error"], color=color, label=f"Error v={shared_value}")
    ref = pd.read_csv(SCAN_DIR / "v4" / "eval_timeseries.csv")
    axes[0].plot(ref["step"], ref["reference"], color="#111111", linestyle="--", label="Reference")
    axes[0].set_ylabel("Output")
    axes[0].set_title("Selected shared values: trajectory comparison")
    axes[0].legend(frameon=False, ncol=4)
    axes[1].set_ylabel("Error")
    axes[1].set_xlabel("Step")
    axes[1].legend(frameon=False, ncol=3)
    return _save(fig, "fig05_time_response_v1_v4_v7")


def plot_v7_tuning(master_df: pd.DataFrame) -> dict[str, str]:
    subset = master_df[master_df["source_group"] == "v7_tuning_study"].copy()
    subset["display"] = subset["variant"].map(
        {
            "v7_decay_slow_net512_tau005": "net512 tau0.05",
            "v7_decay_slow_net512_tau002": "net512 tau0.02",
            "v7_decay_slow_net768_tau002": "net768 tau0.02",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    cols = [("rmse", "RMSE"), ("reward", "Reward"), ("control_energy", "Control energy")]
    colors = [COLORS["tune_a"], COLORS["tune_b"], COLORS["tune_c"]]
    for ax, (column, title) in zip(axes, cols):
        ax.bar(subset["display"], subset[column], color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("v=7 tuning study with existing data")
    return _save(fig, "fig06_v7_tuning_comparison")


def plot_pid_tracking_effect() -> dict[str, str]:
    frame = pd.read_csv(PID_BASELINE_DIR / "pid_timeseries.csv")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(frame["time"], frame["target_x"], color="#111111", linestyle="--", label="Reference")
    axes[0].plot(frame["time"], frame["x"], color=COLORS["pid"], label="PID output")
    axes[0].set_ylabel("Position")
    axes[0].set_title("PID tracking effect on x-axis reference task")
    axes[0].legend(frameon=False)

    pos_error = frame["target_x"] - frame["x"]
    axes[1].plot(frame["time"], pos_error, color=COLORS["pid"], label="Position error")
    axes[1].axhline(0.0, color="#444444", linewidth=1.0, linestyle=":")
    axes[1].set_ylabel("Error")
    axes[1].legend(frameon=False)

    axes[2].plot(frame["time"], frame["target_vx"], color="#111111", linestyle="--", label="Reference velocity")
    axes[2].plot(frame["time"], frame["vx"], color=COLORS["pid"], label="PID velocity")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Velocity")
    axes[2].legend(frameon=False)

    return _save(fig, "fig07_pid_tracking_effect")


def main() -> None:
    _apply_style()
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(DATA_ROOT / "chapter3_master_metrics.csv")
    status = pd.read_csv(DATA_ROOT / "chapter3_thesis_experiment_manifest.csv")

    manifest = {
        "fig00": plot_status_matrix(status),
        "fig01": plot_main_available(master),
        "fig02": plot_v_sweep(),
        "fig03": plot_reward_curves(),
        "fig04": plot_best_time_response(),
        "fig05": plot_selected_time_responses(),
        "fig06": plot_v7_tuning(master),
        "fig07": plot_pid_tracking_effect(),
    }
    manifest_path = DATA_ROOT / "chapter3_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figure_manifest": str(manifest_path), "figures": manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
