from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.control import export_chapter3_plot_data as export_mod


ROOT = Path(r"D:\ZhangC\lc")
OUTPUT_ROOT = ROOT / "outputs" / "chapter3_redraw" / "figure_3_2_fixed_ladrc_multi_condition"

FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]

COLORS = {
    "reference": "#2F2F2F",
    "A": "#1F77B4",
    "B": "#FF7F0E",
    "C": "#2CA02C",
    "shade": "#D95F5F",
}


@dataclass(frozen=True)
class FigureGroup:
    code: str
    label: str
    r: float
    b0: float
    omega_c: float
    k: float


GROUPS = (
    FigureGroup("A", "参数组A", 63.0, 37.5, 2.125, 6.588235294117647),
    FigureGroup("B", "参数组B", 63.0, 40.5, 2.125, 5.176470588235294),
    FigureGroup("C", "参数组C", 52.5, 35.625, 2.125, 6.470588235294118),
)


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
    (out_dir / "raw_timeseries").mkdir(parents=True, exist_ok=True)
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


def build_velocity_reference(target_speed: float = 0.45, total_seconds: float = 6.0) -> tuple[export_mod.ReferenceBundle, dict[str, float]]:
    step_count = int(round(total_seconds * export_mod.CONTROL_FREQ_HZ))
    hold_steps = int(round(1.0 * export_mod.CONTROL_FREQ_HZ))
    speed_steps = int(round(3.0 * export_mod.CONTROL_FREQ_HZ))
    tail_steps = step_count - hold_steps - speed_steps
    velocities_x = np.concatenate(
        [
            np.zeros(hold_steps, dtype=np.float32),
            np.full(speed_steps, target_speed, dtype=np.float32),
            np.zeros(max(tail_steps, 0), dtype=np.float32),
        ]
    )
    positions_x = export_mod.integrate_velocity_profile(0.0, velocities_x, export_mod.DT)
    bundle = export_mod._reference_bundle_from_position_and_velocity(positions_x, velocities_x)
    return bundle, {
        "target_speed_mps": target_speed,
        "constant_speed_start_time": hold_steps * export_mod.DT,
        "constant_speed_end_time": (hold_steps + speed_steps) * export_mod.DT,
        "total_duration": total_seconds,
    }


def evaluate_group(config: export_mod.PyBulletControlExperimentConfig, bundle: export_mod.ReferenceBundle, group: FigureGroup) -> list[dict[str, float]]:
    fixed_group = export_mod.FixedLADRCGroup(group.code, group.label, group.r, group.b0, group.omega_c, group.k)
    return export_mod._evaluate_fixed_group(config, bundle, fixed_group)


def export_raw_timeseries(out_dir: Path, prefix: str, rows_by_group: dict[str, list[dict[str, float]]]) -> None:
    for label, rows in rows_by_group.items():
        pd.DataFrame(rows).to_csv(out_dir / "raw_timeseries" / f"{prefix}_{label}.csv", index=False, encoding="utf-8-sig")


def main(tag: str | None = None) -> Path:
    configure_matplotlib()
    out_dir = build_output_dir(tag)

    step_bundle, step_info = export_mod.build_step_reference()
    disturb_bundle, disturb_info = export_mod.build_hold_disturbance_reference()
    velocity_bundle, velocity_info = build_velocity_reference(0.45, 6.0)

    step_config = export_mod._build_config(
        output_root=out_dir / "_artifacts" / "step",
        duration_sec=export_mod.STEP_SCENARIO_SECONDS,
        include_disturbance=False,
        disturbance_scale=0.0,
    )
    disturbance_scale_n = 0.008
    disturb_config = export_mod._build_config(
        output_root=out_dir / "_artifacts" / "disturbance",
        duration_sec=export_mod.MAIN_DURATION,
        include_disturbance=True,
        disturbance_scale=disturbance_scale_n,
        fixed_stage_lengths=export_mod.MAIN_STAGE_LENGTHS,
        fixed_stage_velocities=(0.0, 0.0, 0.0, 0.0),
    )
    velocity_config = export_mod._build_config(
        output_root=out_dir / "_artifacts" / "velocity",
        duration_sec=velocity_info["total_duration"],
        include_disturbance=False,
        disturbance_scale=0.0,
    )

    step_rows: dict[str, list[dict[str, float]]] = {}
    disturb_rows: dict[str, list[dict[str, float]]] = {}
    velocity_rows: dict[str, list[dict[str, float]]] = {}

    for group in GROUPS:
        step_rows[group.label] = evaluate_group(step_config, step_bundle, group)
        disturb_rows[group.label] = evaluate_group(disturb_config, disturb_bundle, group)
        velocity_rows[group.label] = evaluate_group(velocity_config, velocity_bundle, group)

    export_raw_timeseries(out_dir, "step", step_rows)
    export_raw_timeseries(out_dir, "disturbance", disturb_rows)
    export_raw_timeseries(out_dir, "velocity_0p45", velocity_rows)

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.4))

    # (a) step position response
    step_time = pd.DataFrame(next(iter(step_rows.values())))["time"]
    step_ref = pd.DataFrame(next(iter(step_rows.values())))["target_x"]
    axes[0].plot(step_time, step_ref, "--", color=COLORS["reference"], linewidth=2.2, label="参考")
    for group in GROUPS:
        frame = pd.DataFrame(step_rows[group.label])
        axes[0].plot(frame["time"], frame["x"], color=COLORS[group.code], linewidth=2.1, label=group.label)
    axes[0].set_xlabel("时间 / s")
    axes[0].set_ylabel("位置 / m")
    axes[0].legend(loc="upper left", frameon=True, facecolor="white")
    add_bottom_caption(axes[0], "(a) 阶跃位置响应")

    # (b) disturbance recovery position response
    disturb_frame0 = pd.DataFrame(next(iter(disturb_rows.values())))
    axes[1].plot(disturb_frame0["time"], disturb_frame0["target_x"], "--", color=COLORS["reference"], linewidth=2.2, label="参考")
    for group in GROUPS:
        frame = pd.DataFrame(disturb_rows[group.label])
        axes[1].plot(frame["time"], frame["x"], color=COLORS[group.code], linewidth=2.1, label=group.label)
    disturb_nonzero = disturb_frame0["disturbance_x"].abs() > 1e-12
    disturb_window = disturb_frame0.loc[disturb_nonzero, "time"]
    if not disturb_window.empty:
        start_t = float(disturb_window.iloc[0])
        end_t = float(disturb_window.iloc[-1])
        axes[1].axvspan(start_t, end_t, color=COLORS["shade"], alpha=0.10)
        axes[1].axvline(start_t, color=COLORS["shade"], linestyle="--", linewidth=1.6)
        axes[1].axvline(end_t, color=COLORS["shade"], linestyle=":", linewidth=1.6)
    axes[1].set_xlabel("时间 / s")
    axes[1].set_ylabel("位置 / m")
    axes[1].legend(loc="upper right", frameon=True, facecolor="white")
    add_bottom_caption(axes[1], "(b) 扰动恢复位置响应")

    # unify y meaning for first two panels
    combined_min = min(
        min(pd.DataFrame(v)["x"].min() for v in step_rows.values()),
        min(pd.DataFrame(v)["x"].min() for v in disturb_rows.values()),
        step_ref.min(),
        disturb_frame0["target_x"].min(),
    )
    combined_max = max(
        max(pd.DataFrame(v)["x"].max() for v in step_rows.values()),
        max(pd.DataFrame(v)["x"].max() for v in disturb_rows.values()),
        step_ref.max(),
        disturb_frame0["target_x"].max(),
    )
    for ax in axes[:2]:
        ax.set_ylim(combined_min - 0.05, combined_max + 0.08)

    # (c) velocity tracking at 0.45 m/s
    velocity_frame0 = pd.DataFrame(next(iter(velocity_rows.values())))
    axes[2].plot(velocity_frame0["time"], velocity_frame0["target_vx"], "--", color=COLORS["reference"], linewidth=2.2, label="参考")
    for group in GROUPS:
        frame = pd.DataFrame(velocity_rows[group.label])
        axes[2].plot(frame["time"], frame["vx"], color=COLORS[group.code], linewidth=2.1, label=group.label)
    axes[2].set_xlabel("时间 / s")
    axes[2].set_ylabel("速度 / (m/s)")
    axes[2].legend(loc="upper right", frameon=True, facecolor="white")
    add_bottom_caption(axes[2], "(c) 恒速参考跟踪响应")

    fig.subplots_adjust(bottom=0.22, wspace=0.30)
    save_figure(fig, out_dir / "figures" / "图3-2_固定参数LADRC多工况性能对比图_重画版")

    pd.DataFrame(
        [
            {"参数项": "后端", "数值": "gym_env"},
            {"参数项": "控制频率(Hz)", "数值": export_mod.CONTROL_FREQ_HZ},
            {"参数项": "阶跃位置幅值(m)", "数值": step_info["step_amplitude"]},
            {"参数项": "阶跃开始时间(s)", "数值": step_info["step_start_time"]},
            {"参数项": "扰动保持位置(m)", "数值": disturb_info["hold_position"]},
            {"参数项": "扰动幅值(N)", "数值": disturbance_scale_n},
            {"参数项": "扰动开始时间(s)", "数值": disturb_info["disturbance_start_time"]},
            {"参数项": "扰动结束时间(s)", "数值": disturb_info["disturbance_end_time"]},
            {"参数项": "速度跟踪目标(m/s)", "数值": velocity_info["target_speed_mps"]},
            {"参数项": "速度跟踪开始时间(s)", "数值": velocity_info["constant_speed_start_time"]},
            {"参数项": "速度跟踪结束时间(s)", "数值": velocity_info["constant_speed_end_time"]},
        ]
    ).to_csv(out_dir / "tables" / "图3-2_实验参数说明.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([group.__dict__ for group in GROUPS]).to_csv(
        out_dir / "tables" / "图3-2_参数组说明.csv",
        index=False,
        encoding="utf-8-sig",
    )

    velocity_stats = []
    for group in GROUPS:
        frame = export_mod._timeseries_frame(velocity_rows[group.label])
        velocity_stats.append(
            {
                "method": group.label,
                "RMSE": export_mod._rmse(frame),
                "MAE": export_mod._mae(frame),
                "ITAE": export_mod._itae(frame),
                "最大偏差": export_mod._max_deviation(frame),
                "控制输入方差": export_mod._control_input_variance(frame),
                "平均|Δu|": export_mod._avg_abs_delta_u(frame),
                "velocity_rmse": export_mod._velocity_rmse(frame),
            }
        )
    pd.DataFrame(velocity_stats).to_csv(out_dir / "tables" / "图3-2_0p45速度跟踪统计.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "groups": [group.__dict__ for group in GROUPS],
        "step_info": step_info,
        "disturbance_info": disturb_info,
        "disturbance_scale_n": disturbance_scale_n,
        "velocity_info": velocity_info,
        "figure_path": str(out_dir / "figures" / "图3-2_固定参数LADRC多工况性能对比图_重画版.png"),
    }
    (out_dir / "summaries" / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_dir)
    return out_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Redraw Chapter 3 Figure 3-2 with cleaner layout.")
    parser.add_argument("--tag", default="paper_style_clean", help="Optional output directory suffix.")
    args = parser.parse_args()
    main(args.tag)
