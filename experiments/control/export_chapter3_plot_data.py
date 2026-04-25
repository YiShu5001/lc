from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lc.control.configs import ArtifactConfig, AxisTrainingConfig, PyBulletControlExperimentConfig
from lc.control.controllers import create_controller_bundle
from lc.control.envs import run_controller_episode
from lc.control.io import write_metrics_csv, write_summary_json, write_timeseries_csv
from lc.control.reference_generators.piecewise_velocity import ReferenceBundle, integrate_velocity_profile
from lc.control.simulators import close_ctrl_aviary, create_ctrl_aviary, run_policy_episode
from lc.rl.algorithms import MDDPGConfig, MDDPGPolicy

CONTROL_FREQ_HZ = 48
DT = 1.0 / CONTROL_FREQ_HZ
MAIN_STAGE_LENGTHS = (96, 48, 96, 24)
MAIN_STAGE_VELOCITIES = (0.5, 0.0, -0.6, 0.0)
MAIN_STEP_COUNT = sum(MAIN_STAGE_LENGTHS)
MAIN_DURATION = MAIN_STEP_COUNT / CONTROL_FREQ_HZ
DISTURBANCE_WINDOW = (106, 134)
EVAL_SEEDS = (7, 17, 27)
STEP_SCENARIO_SECONDS = 6.0
CONST_SCENARIO_SECONDS = 6.0

TRUSTED_NO_DISTURB_COMPARE = (
    PROJECT_ROOT / "outputs" / "control_pybullet" / "x_pid_vs_ladrc_no_disturbance_compare" / "x" / "20260412_105149"
)
TRUSTED_RANDOM_COMPARE = (
    PROJECT_ROOT
    / "outputs"
    / "control_pybullet"
    / "x_pid_ladrc_ddpg_random_hover_disturb_compare"
    / "x"
    / "20260412_random_hover_disturb_0p004_bestv2_narrowed_noise0p2_posonly_multiseed_normfix_300eps_compare"
)
TRUSTED_RANDOM_TRAIN_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "control_pybullet_rl"
    / "x_refline_random_hover_disturbance_mddpg_retrain"
    / "20260412_random_hover_disturb_v1234_300eps_narrowed_noise0p2_0p004_posonly_multiseed_normfix"
    / "v_2"
    / "train"
    / "ladrc_x_pos_pid_att"
    / "x"
    / "20260412_185851"
)
ABLATION_ROOT = (
    PROJECT_ROOT / "outputs" / "control_pybullet_rl" / "x_temporal_ablation_suite" / "20260414_temporal_ablation_ep250"
)
MULTISPEED_TABLE = (
    PROJECT_ROOT
    / "outputs"
    / "chapter3_thesis_package"
    / "20260412_thesis_package"
    / "tables"
    / "table_ladrc_multispeed_best_params.csv"
)


@dataclass(frozen=True)
class FixedLADRCGroup:
    label: str
    chinese_label: str
    r: float
    b0: float
    omega_c: float
    k: float


FIXED_GROUPS = (
    FixedLADRCGroup("A", "参数组A", 63.0, 24.3, 2.95, 7.415254237288136),
    FixedLADRCGroup("B", "参数组B", 59.5, 27.45, 3.15, 3.08),
    FixedLADRCGroup("C", "参数组C", 59.5, 32.94, 3.43, 3.85),
)


def _timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _step_count(seconds: float) -> int:
    return int(round(seconds * CONTROL_FREQ_HZ))


def _build_output_root(tag: str | None) -> Path:
    stem = tag or _timestamp_tag()
    return PROJECT_ROOT / "outputs" / "chapter3_result_data" / stem


def _artifact_root(output_root: Path) -> ArtifactConfig:
    return ArtifactConfig(
        output_root=str(output_root),
        export_structured=True,
        export_legacy_logger=False,
        save_figures=False,
        record_video=False,
    )


def _build_axis_config(*, include_disturbance: bool, disturbance_scale: float, fixed_stage_lengths=(), fixed_stage_velocities=()) -> AxisTrainingConfig:
    return AxisTrainingConfig(
        axis="x",
        initial_position=(0.0, 0.0, 1.0),
        fixed_axes=(0.0, 1.0),
        include_disturbance=include_disturbance,
        disturbance_scale=disturbance_scale,
        disturbance_axis_bias=1.0,
        disturbance_step_window=DISTURBANCE_WINDOW,
        disturbance_mode="random_uniform",
        fixed_stage_lengths=fixed_stage_lengths,
        fixed_stage_velocities=fixed_stage_velocities,
        stage_count=4,
    )


def _build_config(*, output_root: Path, duration_sec: float, include_disturbance: bool, disturbance_scale: float, fixed_stage_lengths=(), fixed_stage_velocities=(), seed: int = 7, eval_seeds=EVAL_SEEDS) -> PyBulletControlExperimentConfig:
    return PyBulletControlExperimentConfig(
        control_freq_hz=CONTROL_FREQ_HZ,
        duration_sec=duration_sec,
        seed=seed,
        train_episodes=1,
        eval_episodes=1,
        compare_episodes=1,
        eval_seeds=tuple(eval_seeds),
        snapshot_interval=0,
        training_controller_variant="ladrc_x_pos_pid_att",
        artifact=_artifact_root(output_root),
        axis_configs=(
            _build_axis_config(
                include_disturbance=include_disturbance,
                disturbance_scale=disturbance_scale,
                fixed_stage_lengths=fixed_stage_lengths,
                fixed_stage_velocities=fixed_stage_velocities,
            ),
        ),
    )


def _reference_bundle_from_position_and_velocity(positions_x: np.ndarray, velocities_x: np.ndarray) -> ReferenceBundle:
    step_count = len(positions_x)
    positions = np.zeros((step_count, 3), dtype=np.float32)
    velocities = np.zeros((step_count, 3), dtype=np.float32)
    positions[:, 2] = 1.0
    positions[:, 0] = positions_x.astype(np.float32)
    velocities[:, 0] = velocities_x.astype(np.float32)
    return ReferenceBundle(
        axis="x",
        positions=positions,
        velocities=velocities,
        stage_slices=(slice(0, step_count),),
        stage_velocities=(float(velocities_x[0]) if step_count else 0.0,),
    )


def build_step_reference() -> tuple[ReferenceBundle, dict[str, float]]:
    step_count = _step_count(STEP_SCENARIO_SECONDS)
    step_start = _step_count(1.0)
    amplitude = 0.6
    positions_x = np.zeros(step_count, dtype=np.float32)
    positions_x[step_start:] = amplitude
    velocities_x = np.zeros(step_count, dtype=np.float32)
    return _reference_bundle_from_position_and_velocity(positions_x, velocities_x), {
        "step_start_time": step_start * DT,
        "step_amplitude": amplitude,
    }


def build_hold_disturbance_reference() -> tuple[ReferenceBundle, dict[str, float]]:
    positions_x = np.full(MAIN_STEP_COUNT, 0.4, dtype=np.float32)
    velocities_x = np.zeros(MAIN_STEP_COUNT, dtype=np.float32)
    return _reference_bundle_from_position_and_velocity(positions_x, velocities_x), {
        "hold_position": 0.4,
        "disturbance_start_time": DISTURBANCE_WINDOW[0] * DT,
        "disturbance_end_time": DISTURBANCE_WINDOW[1] * DT,
    }


def build_constant_speed_reference() -> tuple[ReferenceBundle, dict[str, float]]:
    step_count = _step_count(CONST_SCENARIO_SECONDS)
    hold_steps = _step_count(1.0)
    speed_steps = _step_count(3.0)
    tail_steps = step_count - hold_steps - speed_steps
    velocities_x = np.concatenate(
        [
            np.zeros(hold_steps, dtype=np.float32),
            np.full(speed_steps, 0.5, dtype=np.float32),
            np.zeros(max(tail_steps, 0), dtype=np.float32),
        ]
    )
    positions_x = integrate_velocity_profile(0.0, velocities_x, DT)
    return _reference_bundle_from_position_and_velocity(positions_x, velocities_x), {
        "constant_speed_start_time": hold_steps * DT,
        "constant_speed_end_time": (hold_steps + speed_steps) * DT,
        "target_speed_mps": 0.5,
    }


def _apply_ladrc_group(controller, group: FixedLADRCGroup) -> None:
    controller.set_axis_parameters("x", r=group.r, b0=group.b0, omega_c=group.omega_c, k=group.k)


def _load_policy(checkpoint_path: Path) -> MDDPGPolicy:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = MDDPGConfig(**dict(checkpoint["policy_state"]["config"]))
    policy = MDDPGPolicy(cfg)
    policy.actor.load_state_dict(checkpoint["policy_state"]["actor"])
    if "actor_target" in checkpoint["policy_state"]:
        policy.actor_target.load_state_dict(checkpoint["policy_state"]["actor_target"])
    if "critic" in checkpoint["policy_state"]:
        policy.critic.load_state_dict(checkpoint["policy_state"]["critic"])
    if "critic_target" in checkpoint["policy_state"]:
        policy.critic_target.load_state_dict(checkpoint["policy_state"]["critic_target"])
    policy.reset()
    normalizer = checkpoint["policy_state"].get("normalizer")
    if normalizer is not None:
        policy._normalizer = np.asarray(normalizer, dtype=np.float32)
    last_action = checkpoint["policy_state"].get("last_action")
    if last_action is not None:
        policy._last_action = np.asarray(last_action, dtype=np.float32)
    if "hold_counter" in checkpoint["policy_state"]:
        policy._hold_counter = int(checkpoint["policy_state"]["hold_counter"])
    if "current_expl_noise" in checkpoint["policy_state"]:
        policy._current_expl_noise = float(checkpoint["policy_state"]["current_expl_noise"])
    return policy


def _run_policy_checkpoint_episode(config: PyBulletControlExperimentConfig, reference_bundle: ReferenceBundle, checkpoint_path: Path) -> list[dict[str, float]]:
    env = create_ctrl_aviary(config)
    controller = create_controller_bundle("ladrc_x_pos_pid_att")
    policy = _load_policy(checkpoint_path)
    try:
        artifacts = run_policy_episode(
            env,
            policy,
            controller,
            reference_bundle,
            axis="x",
            config=config,
            explore=False,
            store_transitions=False,
            n_step=1,
        )
        return list(artifacts.timeseries)
    finally:
        close_ctrl_aviary(env)


def _evaluate_fixed_group(config: PyBulletControlExperimentConfig, reference_bundle: ReferenceBundle, group: FixedLADRCGroup) -> list[dict[str, float]]:
    controller = create_controller_bundle("ladrc_x_pos_pid_att")
    _apply_ladrc_group(controller, group)
    result = run_controller_episode(config, controller, reference_bundle)
    return list(result["timeseries"])


def _evaluate_pid(config: PyBulletControlExperimentConfig, reference_bundle: ReferenceBundle) -> list[dict[str, float]]:
    controller = create_controller_bundle("pid_pos_att")
    result = run_controller_episode(config, controller, reference_bundle)
    return list(result["timeseries"])


def _timeseries_frame(rows: list[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _mean_control(frame: pd.DataFrame) -> np.ndarray:
    return frame[["rpm0", "rpm1", "rpm2", "rpm3"]].astype(float).mean(axis=1).to_numpy(dtype=float)


def _control_input_variance(frame: pd.DataFrame) -> float:
    u = _mean_control(frame)
    return float(np.var(u)) if len(u) else 0.0


def _avg_abs_delta_u(frame: pd.DataFrame) -> float:
    u = _mean_control(frame)
    return float(np.mean(np.abs(np.diff(u)))) if len(u) > 1 else 0.0


def _itae(frame: pd.DataFrame, axis: str = "x") -> float:
    err = np.abs(frame[f"target_{axis}"].astype(float) - frame[axis].astype(float)).to_numpy(dtype=float)
    t = frame["time"].astype(float).to_numpy(dtype=float)
    return float(np.sum(t * err) * DT)


def _rmse(frame: pd.DataFrame, axis: str = "x") -> float:
    err = frame[f"target_{axis}"].astype(float) - frame[axis].astype(float)
    return float(np.sqrt(np.mean(np.square(err.to_numpy(dtype=float)))))


def _mae(frame: pd.DataFrame, axis: str = "x") -> float:
    err = frame[f"target_{axis}"].astype(float) - frame[axis].astype(float)
    return float(np.mean(np.abs(err.to_numpy(dtype=float))))


def _max_deviation(frame: pd.DataFrame, axis: str = "x") -> float:
    err = frame[f"target_{axis}"].astype(float) - frame[axis].astype(float)
    return float(np.max(np.abs(err.to_numpy(dtype=float))))


def _velocity_rmse(frame: pd.DataFrame, axis: str = "x") -> float:
    err = frame[f"target_v{axis}"].astype(float) - frame[f"v{axis}"].astype(float)
    return float(np.sqrt(np.mean(np.square(err.to_numpy(dtype=float)))))


def _steady_state_error(frame: pd.DataFrame, axis: str = "x", tail_seconds: float = 0.5) -> float:
    tail_count = max(int(round(tail_seconds * CONTROL_FREQ_HZ)), 1)
    err = np.abs(frame[f"target_{axis}"].astype(float) - frame[axis].astype(float)).to_numpy(dtype=float)
    return float(np.mean(err[-tail_count:]))


def _settling_time(frame: pd.DataFrame, *, axis: str = "x", event_start_time: float, tolerance: float, stable_seconds: float = 0.25) -> float:
    times = frame["time"].astype(float).to_numpy(dtype=float)
    err = np.abs(frame[f"target_{axis}"].astype(float) - frame[axis].astype(float)).to_numpy(dtype=float)
    start_idx = int(np.searchsorted(times, event_start_time, side="left"))
    stable_count = max(int(round(stable_seconds * CONTROL_FREQ_HZ)), 1)
    for idx in range(start_idx, len(err)):
        window = err[idx : idx + stable_count]
        if len(window) < stable_count:
            break
        if np.all(window <= tolerance):
            return float(times[idx] - event_start_time)
    return float("nan")


def _recovery_time(frame: pd.DataFrame, *, axis: str = "x", disturbance_end_time: float, tolerance: float, stable_seconds: float = 0.25) -> float:
    return _settling_time(frame, axis=axis, event_start_time=disturbance_end_time, tolerance=tolerance, stable_seconds=stable_seconds)


def _overshoot(frame: pd.DataFrame, *, axis: str = "x", step_start_time: float, target_level: float) -> float:
    times = frame["time"].astype(float).to_numpy(dtype=float)
    values = frame[axis].astype(float).to_numpy(dtype=float)
    start_idx = int(np.searchsorted(times, step_start_time, side="left"))
    peak = float(np.max(values[start_idx:])) if start_idx < len(values) else float(values[-1])
    return float(max(peak - target_level, 0.0))


def _scenario_metric_row(frame: pd.DataFrame, *, method: str, scenario: str, tolerance: float, step_start_time: float | None = None, target_level: float | None = None, disturbance_end_time: float | None = None) -> dict[str, object]:
    row = {
        "scenario": scenario,
        "method": method,
        "RMSE": _rmse(frame),
        "MAE": _mae(frame),
        "ITAE": _itae(frame),
        "最大偏差": _max_deviation(frame),
        "控制输入方差": _control_input_variance(frame),
        "平均|Δu|": _avg_abs_delta_u(frame),
        "velocity_rmse": _velocity_rmse(frame),
        "steady_state_error": _steady_state_error(frame),
    }
    row["调节时间"] = _settling_time(frame, event_start_time=step_start_time, tolerance=tolerance) if step_start_time is not None else float("nan")
    row["超调量"] = _overshoot(frame, step_start_time=step_start_time, target_level=float(target_level or 0.0)) if step_start_time is not None else float("nan")
    row["恢复时间"] = _recovery_time(frame, disturbance_end_time=disturbance_end_time, tolerance=tolerance) if disturbance_end_time is not None else float("nan")
    return row


def _copy_if_exists(src: Path, dst: Path) -> Path | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _export_fixed_ladrc_multicondition(output_root: Path) -> dict[str, object]:
    base_dir = output_root / "figure_3_2_fixed_ladrc_multi_condition"
    (base_dir / "plot_data").mkdir(parents=True, exist_ok=True)
    (base_dir / "raw_timeseries").mkdir(parents=True, exist_ok=True)
    step_bundle, step_info = build_step_reference()
    disturb_bundle, disturb_info = build_hold_disturbance_reference()
    speed_bundle, speed_info = build_constant_speed_reference()
    step_config = _build_config(output_root=base_dir / "_artifacts" / "step", duration_sec=STEP_SCENARIO_SECONDS, include_disturbance=False, disturbance_scale=0.0)
    disturb_config = _build_config(
        output_root=base_dir / "_artifacts" / "disturbance",
        duration_sec=MAIN_DURATION,
        include_disturbance=True,
        disturbance_scale=0.004,
        fixed_stage_lengths=MAIN_STAGE_LENGTHS,
        fixed_stage_velocities=(0.0, 0.0, 0.0, 0.0),
    )
    speed_config = _build_config(output_root=base_dir / "_artifacts" / "constant_speed", duration_sec=CONST_SCENARIO_SECONDS, include_disturbance=False, disturbance_scale=0.0)

    step_rows: dict[str, list[dict[str, float]]] = {}
    disturb_rows: dict[str, list[dict[str, float]]] = {}
    speed_rows: dict[str, list[dict[str, float]]] = {}
    metric_rows: list[dict[str, object]] = []
    group_table: list[dict[str, object]] = []

    for group in FIXED_GROUPS:
        group_table.append({"group": group.label, "label": group.chinese_label, "r": group.r, "b0": group.b0, "omega_c": group.omega_c, "k": group.k})
        step_timeseries = _evaluate_fixed_group(step_config, step_bundle, group)
        disturb_timeseries = _evaluate_fixed_group(disturb_config, disturb_bundle, group)
        speed_timeseries = _evaluate_fixed_group(speed_config, speed_bundle, group)
        step_rows[group.chinese_label] = step_timeseries
        disturb_rows[group.chinese_label] = disturb_timeseries
        speed_rows[group.chinese_label] = speed_timeseries
        write_timeseries_csv(base_dir / "raw_timeseries" / f"step_{group.label}.csv", step_timeseries)
        write_timeseries_csv(base_dir / "raw_timeseries" / f"disturbance_{group.label}.csv", disturb_timeseries)
        write_timeseries_csv(base_dir / "raw_timeseries" / f"constant_speed_{group.label}.csv", speed_timeseries)
        metric_rows.append(_scenario_metric_row(_timeseries_frame(step_timeseries), method=group.chinese_label, scenario="step_response", tolerance=max(0.02 * step_info["step_amplitude"], 0.01), step_start_time=step_info["step_start_time"], target_level=step_info["step_amplitude"]))
        metric_rows.append(_scenario_metric_row(_timeseries_frame(disturb_timeseries), method=group.chinese_label, scenario="disturbance_recovery", tolerance=0.05, disturbance_end_time=disturb_info["disturbance_end_time"]))
        metric_rows.append(_scenario_metric_row(_timeseries_frame(speed_timeseries), method=group.chinese_label, scenario="constant_speed_tracking", tolerance=0.05))

    pd.DataFrame({"time": [row["time"] for row in next(iter(step_rows.values()))], "reference": [row["target_x"] for row in next(iter(step_rows.values()))], **{label: [row["x"] for row in rows] for label, rows in step_rows.items()}}).to_csv(base_dir / "plot_data" / "fig3_2a_step_response_data.csv", index=False)
    pd.DataFrame({"time": [row["time"] for row in next(iter(disturb_rows.values()))], "reference": [row["target_x"] for row in next(iter(disturb_rows.values()))], "disturbance_x": [row.get("disturbance_x", 0.0) for row in next(iter(disturb_rows.values()))], **{label: [row["x"] for row in rows] for label, rows in disturb_rows.items()}}).to_csv(base_dir / "plot_data" / "fig3_2b_disturbance_recovery_data.csv", index=False)
    pd.DataFrame({"time": [row["time"] for row in next(iter(speed_rows.values()))], "reference_velocity": [row["target_vx"] for row in next(iter(speed_rows.values()))], **{f"{label}_error": [row["target_x"] - row["x"] for row in rows] for label, rows in speed_rows.items()}}).to_csv(base_dir / "plot_data" / "fig3_2c_constant_speed_error_data.csv", index=False)

    write_metrics_csv(base_dir / "fixed_ladrc_statistics.csv", metric_rows)
    write_metrics_csv(base_dir / "fixed_ladrc_param_groups.csv", group_table)
    write_summary_json(base_dir / "summary.json", {"parameter_groups": group_table, "step": step_info, "disturbance": disturb_info, "constant_speed": speed_info})
    return {"root": str(base_dir), "statistics_csv": str(base_dir / "fixed_ladrc_statistics.csv")}


def _export_timeline(output_root: Path) -> dict[str, object]:
    base_dir = output_root / "figure_3_5_timeline"
    rows = [
        {"type": "reference_change", "label": "前进段开始", "time_s": 0.0, "step": 0, "enabled": True},
        {"type": "reference_change", "label": "悬停段开始", "time_s": 96 * DT, "step": 96, "enabled": True},
        {"type": "reference_change", "label": "反向段开始", "time_s": (96 + 48) * DT, "step": 144, "enabled": True},
        {"type": "reference_change", "label": "末段保持开始", "time_s": (96 + 48 + 96) * DT, "step": 240, "enabled": True},
        {"type": "reference_change", "label": "实验结束", "time_s": MAIN_DURATION, "step": MAIN_STEP_COUNT, "enabled": True},
        {"type": "disturbance_start", "label": "随机外力注入开始", "time_s": DISTURBANCE_WINDOW[0] * DT, "step": DISTURBANCE_WINDOW[0], "enabled": True},
        {"type": "disturbance_end", "label": "随机外力注入结束", "time_s": DISTURBANCE_WINDOW[1] * DT, "step": DISTURBANCE_WINDOW[1], "enabled": True},
        {"type": "model_mismatch_start", "label": "模型失配开始", "time_s": float("nan"), "step": float("nan"), "enabled": False},
        {"type": "model_mismatch_end", "label": "模型失配结束", "time_s": float("nan"), "step": float("nan"), "enabled": False},
        {"type": "recovery_interval", "label": "扰动后悬停恢复阶段", "start_time_s": DISTURBANCE_WINDOW[1] * DT, "end_time_s": 144 * DT, "start_step": DISTURBANCE_WINDOW[1], "end_step": 144, "enabled": True},
        {"type": "recovery_interval", "label": "扰动后全局恢复阶段", "start_time_s": DISTURBANCE_WINDOW[1] * DT, "end_time_s": MAIN_DURATION, "start_step": DISTURBANCE_WINDOW[1], "end_step": MAIN_STEP_COUNT, "enabled": True},
    ]
    base_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(base_dir / "experiment_timeline.csv", index=False)
    write_summary_json(base_dir / "experiment_timeline.json", {"control_freq_hz": CONTROL_FREQ_HZ, "stage_lengths": list(MAIN_STAGE_LENGTHS), "stage_velocities": list(MAIN_STAGE_VELOCITIES), "disturbance_scale_n": 0.004, "disturbance_mode": "random_uniform", "disturbance_window_steps": list(DISTURBANCE_WINDOW), "timeline": rows})
    return {"csv": str(base_dir / "experiment_timeline.csv"), "json": str(base_dir / "experiment_timeline.json")}


def _metric_row_from_frame(frame: pd.DataFrame, method: str, disturbance_end_time: float | None = None) -> dict[str, object]:
    return {
        "方法": method,
        "RMSE": _rmse(frame),
        "ITAE": _itae(frame),
        "最大偏差": _max_deviation(frame),
        "恢复时间": _recovery_time(frame, disturbance_end_time=disturbance_end_time, tolerance=0.05) if disturbance_end_time is not None else float("nan"),
        "控制输入方差": _control_input_variance(frame),
        "平均|Δu|": _avg_abs_delta_u(frame),
        "MAE": _mae(frame),
        "velocity_rmse": _velocity_rmse(frame),
    }


def _evaluate_main_compare_multiseed(output_root: Path, checkpoint_path: Path):
    compare_root = output_root / "_artifacts" / "main_compare"
    metric_accum = {"PID": [], "固定参数LADRC": [], "DDPG--RL--LADRC": []}
    representative_rows = {}
    from lc.control.reference_generators import build_xyz_reference_trajectory

    for eval_seed in EVAL_SEEDS:
        scenario_config = _build_config(
            output_root=compare_root / f"seed_{eval_seed}",
            duration_sec=MAIN_DURATION,
            include_disturbance=True,
            disturbance_scale=0.004,
            fixed_stage_lengths=MAIN_STAGE_LENGTHS,
            fixed_stage_velocities=MAIN_STAGE_VELOCITIES,
            seed=eval_seed,
            eval_seeds=(eval_seed,),
        )
        reference_bundle = build_xyz_reference_trajectory(scenario_config.axis_config("x"), scenario_config, rng=np.random.default_rng(eval_seed))
        pid_rows = _evaluate_pid(scenario_config, reference_bundle)
        ladrc_controller = create_controller_bundle("ladrc_x_pos_pid_att")
        ladrc_controller.set_axis_parameters("x", r=63.0, b0=24.3, omega_c=2.95, k=7.415254237288136)
        ladrc_rows = list(run_controller_episode(scenario_config, ladrc_controller, reference_bundle)["timeseries"])
        ddpg_rows = _run_policy_checkpoint_episode(scenario_config, reference_bundle, checkpoint_path)
        if eval_seed == EVAL_SEEDS[0]:
            representative_rows = {"PID": pid_rows, "固定参数LADRC": ladrc_rows, "DDPG--RL--LADRC": ddpg_rows}
        for label, rows in {"PID": pid_rows, "固定参数LADRC": ladrc_rows, "DDPG--RL--LADRC": ddpg_rows}.items():
            metric_accum[label].append(_metric_row_from_frame(_timeseries_frame(rows), label, disturbance_end_time=DISTURBANCE_WINDOW[1] * DT))

    metric_rows = []
    for label, rows in metric_accum.items():
        metric_rows.append(
            {
                "方法": label,
                "RMSE": float(np.mean([float(row["RMSE"]) for row in rows])),
                "ITAE": float(np.mean([float(row["ITAE"]) for row in rows])),
                "最大偏差": float(np.mean([float(row["最大偏差"]) for row in rows])),
                "恢复时间": float(np.nanmean([float(row["恢复时间"]) for row in rows])),
                "控制输入方差": float(np.mean([float(row["控制输入方差"]) for row in rows])),
                "平均|Δu|": float(np.mean([float(row["平均|Δu|"]) for row in rows])),
                "MAE": float(np.mean([float(row["MAE"]) for row in rows])),
                "velocity_rmse": float(np.mean([float(row["velocity_rmse"]) for row in rows])),
            }
        )
    return representative_rows, metric_rows


def _export_ddpg_compare(output_root: Path) -> dict[str, object]:
    base_dir = output_root / "figure_3_6_ddpg_rl_ladrc_compare"
    (base_dir / "plot_data").mkdir(parents=True, exist_ok=True)
    (base_dir / "raw_timeseries").mkdir(parents=True, exist_ok=True)
    rows_by_method = {
        "PID": pd.read_csv(TRUSTED_RANDOM_COMPARE / "pid_timeseries.csv"),
        "????LADRC": pd.read_csv(TRUSTED_RANDOM_COMPARE / "ladrc_0p5_opt_timeseries.csv"),
        "DDPG--RL--LADRC": pd.read_csv(TRUSTED_RANDOM_COMPARE / "ddpg_best_timeseries.csv"),
    }

    _copy_if_exists(TRUSTED_RANDOM_COMPARE / "pid_timeseries.csv", base_dir / "raw_timeseries" / "pid_timeseries.csv")
    _copy_if_exists(TRUSTED_RANDOM_COMPARE / "ladrc_0p5_opt_timeseries.csv", base_dir / "raw_timeseries" / "fixed_ladrc_timeseries.csv")
    _copy_if_exists(TRUSTED_RANDOM_COMPARE / "ddpg_best_timeseries.csv", base_dir / "raw_timeseries" / "ddpg_rl_ladrc_timeseries.csv")
    _copy_if_exists(TRUSTED_RANDOM_COMPARE / "reference.csv", base_dir / "reference.csv")

    first_frame = next(iter(rows_by_method.values()))
    pd.DataFrame(
        {
            "time": first_frame["time"],
            "reference_velocity": first_frame["target_vx"],
            **{method: frame["vx"] for method, frame in rows_by_method.items()},
        }
    ).to_csv(base_dir / "plot_data" / "fig3_6a_speed_tracking_data.csv", index=False)
    pd.DataFrame(
        {
            "time": first_frame["time"],
            "reference_position": first_frame["target_x"],
            "disturbance_x": first_frame["disturbance_x"],
            **{method: frame["x"] for method, frame in rows_by_method.items()},
        }
    ).to_csv(base_dir / "plot_data" / "fig3_6b_disturbance_recovery_data.csv", index=False)

    smooth_rows = []
    metric_rows = []
    for method, frame in rows_by_method.items():
        smooth_rows.append({"??": method, "??????": _control_input_variance(frame), "??|?u|": _avg_abs_delta_u(frame)})
        metric_rows.append(_metric_row_from_frame(frame, method, disturbance_end_time=DISTURBANCE_WINDOW[1] * DT))
    write_metrics_csv(base_dir / "fig3_6c_control_smoothness_stats.csv", smooth_rows)
    pd.read_csv(TRUSTED_RANDOM_TRAIN_DIR / "training_history.csv").to_csv(base_dir / "fig3_6d_training_reward_curve.csv", index=False)
    write_metrics_csv(base_dir / "fig3_6_metrics.csv", metric_rows)
    write_summary_json(
        base_dir / "summary.json",
        {
            "source_compare_root": str(TRUSTED_RANDOM_COMPARE),
            "source_train_dir": str(TRUSTED_RANDOM_TRAIN_DIR),
            "disturbance_scale_n": 0.004,
            "disturbance_window_steps": list(DISTURBANCE_WINDOW),
        },
    )
    return {"root": str(base_dir), "metrics": str(base_dir / "fig3_6_metrics.csv")}

def _ablation_method_specs(summary_df: pd.DataFrame):
    mapping = [
        ("完整方法", "full_temporal", 3),
        ("w/o状态叠加", "pair_action_hold_n_step", 3),
        ("w/o动作保持", "pair_state_stack_n_step", 3),
        ("w/o N-step", "pair_action_hold_state_stack", 3),
    ]
    specs = []
    for chinese_name, family, k in mapping:
        row = summary_df[(summary_df["family"] == family) & (summary_df["k"].astype(int) == k)].iloc[0]
        specs.append((chinese_name, family, Path(row["best_checkpoint_path"]), Path(row["output_dir"])))
    return specs


def _evaluate_ablation_methods(output_root: Path, method_specs):
    compare_root = output_root / "_artifacts" / "ablation_compare"
    representative_rows = {}
    metric_accum = {name: [] for name, *_ in method_specs}
    from lc.control.reference_generators import build_xyz_reference_trajectory

    for eval_seed in EVAL_SEEDS:
        scenario_config = _build_config(
            output_root=compare_root / f"seed_{eval_seed}",
            duration_sec=MAIN_DURATION,
            include_disturbance=True,
            disturbance_scale=0.004,
            fixed_stage_lengths=MAIN_STAGE_LENGTHS,
            fixed_stage_velocities=MAIN_STAGE_VELOCITIES,
            seed=eval_seed,
            eval_seeds=(eval_seed,),
        )
        reference_bundle = build_xyz_reference_trajectory(scenario_config.axis_config("x"), scenario_config, rng=np.random.default_rng(eval_seed))
        for chinese_name, _family, checkpoint_path, _output_dir in method_specs:
            rows = _run_policy_checkpoint_episode(scenario_config, reference_bundle, checkpoint_path)
            if eval_seed == EVAL_SEEDS[0]:
                representative_rows[chinese_name] = rows
            metric_accum[chinese_name].append(_metric_row_from_frame(_timeseries_frame(rows), chinese_name, disturbance_end_time=DISTURBANCE_WINDOW[1] * DT))

    metric_rows = []
    for name, rows in metric_accum.items():
        metric_rows.append(
            {
                "方法": name,
                "RMSE": float(np.mean([float(row["RMSE"]) for row in rows])),
                "ITAE": float(np.mean([float(row["ITAE"]) for row in rows])),
                "最大偏差": float(np.mean([float(row["最大偏差"]) for row in rows])),
                "恢复时间": float(np.nanmean([float(row["恢复时间"]) for row in rows])),
                "控制输入方差": float(np.mean([float(row["控制输入方差"]) for row in rows])),
                "平均|Δu|": float(np.mean([float(row["平均|Δu|"]) for row in rows])),
                "MAE": float(np.mean([float(row["MAE"]) for row in rows])),
                "velocity_rmse": float(np.mean([float(row["velocity_rmse"]) for row in rows])),
            }
        )
    return representative_rows, metric_rows


def _export_ablation(output_root: Path) -> dict[str, object]:
    base_dir = output_root / "figure_3_7_temporal_ablation"
    summary_df = pd.read_csv(ABLATION_ROOT / "temporal_ablation_summary.csv")
    method_specs = _ablation_method_specs(summary_df)
    reward_df = pd.concat(
        [
            pd.read_csv(output_dir / "training_history.csv").assign(方法=chinese_name)[["方法", "episode", "reward", "average_reward"]]
            for chinese_name, _family, _checkpoint_path, output_dir in method_specs
        ],
        ignore_index=True,
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "raw_timeseries").mkdir(parents=True, exist_ok=True)
    reward_df.to_csv(base_dir / "fig3_7a_ablation_reward_curves.csv", index=False)
    representative_rows, metric_rows = _evaluate_ablation_methods(output_root, method_specs)
    for name, rows in representative_rows.items():
        safe_name = name.replace("/", "_").replace(" ", "").replace("-", "_")
        safe_name = safe_name.replace("状态", "state").replace("动作", "action").replace("完整方法", "full").replace("叠加", "stack").replace("保持", "hold")
        write_timeseries_csv(base_dir / "raw_timeseries" / f"{safe_name}.csv", rows)
    write_metrics_csv(base_dir / "fig3_7b_ablation_metrics.csv", metric_rows)
    write_metrics_csv(base_dir / "fig3_7c_ablation_smoothness.csv", [{"方法": row["方法"], "控制输入方差": row["控制输入方差"], "平均|Δu|": row["平均|Δu|"]} for row in metric_rows])
    _copy_if_exists(ABLATION_ROOT / "figures" / "ablation_heatmap_reward_loss.png", base_dir / "ablation_heatmap_reward_loss.png")
    _copy_if_exists(ABLATION_ROOT / "figures" / "ablation_heatmap_rmse.png", base_dir / "ablation_heatmap_rmse.png")
    write_summary_json(
        base_dir / "summary.json",
        {
            "methods": [{"方法": chinese_name, "family": family, "checkpoint": str(checkpoint_path), "training_dir": str(output_dir)} for chinese_name, family, checkpoint_path, output_dir in method_specs],
            "eval_seeds": list(EVAL_SEEDS),
        },
    )
    return {"root": str(base_dir), "metrics": str(base_dir / "fig3_7b_ablation_metrics.csv")}


def _export_table_3_2(output_root: Path) -> Path:
    compare_metrics = pd.read_csv(output_root / "figure_3_6_ddpg_rl_ladrc_compare" / "fig3_6_metrics.csv")
    ablation_metrics = pd.read_csv(output_root / "figure_3_7_temporal_ablation" / "fig3_7b_ablation_metrics.csv")
    table = pd.concat([compare_metrics, ablation_metrics], ignore_index=True)
    table = table[["方法", "RMSE", "ITAE", "最大偏差", "恢复时间", "控制输入方差", "平均|Δu|"]]
    path = output_root / "tables" / "table_3_2_control_main_quant_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def _export_manifest(output_root: Path, payload: dict[str, object]) -> Path:
    path = output_root / "summaries" / "chapter3_result_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_summary_json(path, payload)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export chapter 3 result data package for plotting.")
    parser.add_argument("--output-tag", default=f"{_timestamp_tag()}_chapter3_result_data")
    args = parser.parse_args()
    output_root = _build_output_root(args.output_tag)
    output_root.mkdir(parents=True, exist_ok=True)

    fixed_payload = _export_fixed_ladrc_multicondition(output_root)
    timeline_payload = _export_timeline(output_root)
    compare_payload = _export_ddpg_compare(output_root)
    ablation_payload = _export_ablation(output_root)
    table_path = _export_table_3_2(output_root)
    manifest = {
        "output_root": str(output_root),
        "figure_3_2": fixed_payload,
        "figure_3_5": timeline_payload,
        "figure_3_6": compare_payload,
        "figure_3_7": ablation_payload,
        "table_3_2": str(table_path),
        "trusted_sources": {
            "no_disturb_compare": str(TRUSTED_NO_DISTURB_COMPARE),
            "random_compare": str(TRUSTED_RANDOM_COMPARE),
            "random_train_dir": str(TRUSTED_RANDOM_TRAIN_DIR),
            "ablation_root": str(ABLATION_ROOT),
            "multispeed_table": str(MULTISPEED_TABLE),
        },
    }
    manifest_path = _export_manifest(output_root, manifest)
    print(json.dumps({"output_root": str(output_root), "manifest": str(manifest_path), "table_3_2": str(table_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
