from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from lc.common.io import ensure_dir, write_json, write_metrics_csv
from lc.control.configs import ArtifactConfig, AxisTrainingConfig, PyBulletControlExperimentConfig, get_axis_ladrc_action_bounds
from lc.control.controllers import ControllerBundle, create_controller_bundle
from lc.control.envs import compute_episode_metrics
from lc.control.io import write_reference_csv, write_timeseries_csv
from lc.control.policies.stacking import stack_state
from lc.control.reference_generators import build_xyz_reference_trajectory
from lc.control.simulators import close_ctrl_aviary, create_ctrl_aviary, run_evaluation_episode, step_controller_loop
from lc.control.simulators.pybullet_runner import (
    _apply_axis_action,
    _apply_real_disturbance,
    _build_axis_observation,
    _disturbance_vector,
    _initial_state,
    _logger_row,
    _timeseries_row,
)

CONTROL_FREQ_HZ = 48
FORWARD_STEPS, HOVER_STEPS, REVERSE_STEPS, FINAL_HOLD_STEPS = 96, 48, 96, 24
HOVER_GAP_STEPS, DISTURBANCE_STEPS = 10, 28
STEP_COUNT = FORWARD_STEPS + HOVER_STEPS + REVERSE_STEPS + FINAL_HOLD_STEPS
DURATION_SEC = STEP_COUNT / CONTROL_FREQ_HZ
DISTURBANCE_SCALE_N = 0.004
DISTURBANCE_WINDOW = (FORWARD_STEPS + HOVER_GAP_STEPS, FORWARD_STEPS + HOVER_GAP_STEPS + DISTURBANCE_STEPS)
STAGE_LENGTHS = (FORWARD_STEPS, HOVER_STEPS, REVERSE_STEPS, FINAL_HOLD_STEPS)
STAGE_VELOCITIES = (0.5, 0.0, -0.6, 0.0)
LADRC_0P5_OPT = {"r": 63.0, "b0": 24.3, "omega_c": 2.95, "k": 7.415254237288136}
BEST_X_V2_CHECKPOINT = PROJECT_ROOT / "outputs" / "control_pybullet_rl" / "x_refline_random_hover_disturbance_mddpg_retrain" / "20260412_random_hover_disturb_v1234_300eps_narrowed_noise0p2_0p004_posonly_multiseed_normfix" / "v_2" / "train" / "ladrc_x_pos_pid_att" / "x" / "20260412_185851" / "checkpoints" / "x_policy_best.pt"


@dataclass(frozen=True)
class Scenario:
    name: str
    axis: str
    variant: str
    disturbed: bool
    kind: str
    checkpoint: Path | None = None


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_config(output_root: Path, axis: str, disturbed: bool, seed: int) -> PyBulletControlExperimentConfig:
    return PyBulletControlExperimentConfig(
        control_freq_hz=CONTROL_FREQ_HZ,
        duration_sec=DURATION_SEC,
        seed=int(seed),
        train_episodes=1,
        eval_episodes=1,
        compare_episodes=1,
        eval_seeds=(int(seed),),
        snapshot_interval=0,
        training_controller_variant=f"ladrc_{axis}_pos_pid_att",
        artifact=ArtifactConfig(output_root=str(output_root), export_structured=True, export_legacy_logger=True, save_figures=True, record_video=False),
        axis_configs=(
            AxisTrainingConfig(
                axis=axis,
                initial_position=(0.0, 0.0, 1.0),
                fixed_axes=(0.0, 1.0),
                include_disturbance=bool(disturbed),
                disturbance_scale=DISTURBANCE_SCALE_N if disturbed else 0.0,
                disturbance_axis_bias=1.0,
                disturbance_mode="random_uniform",
                disturbance_step_window=DISTURBANCE_WINDOW,
                fixed_stage_lengths=STAGE_LENGTHS,
                fixed_stage_velocities=STAGE_VELOCITIES,
            ),
        ),
    )


def apply_ladrc_anchor(controller: ControllerBundle, axis: str) -> None:
    controller.set_axis_parameters(axis, **LADRC_0P5_OPT)


def load_policy(path: Path) -> Any:
    import torch
    from lc.rl.algorithms import MDDPGConfig, MDDPGPolicy
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["policy_state"]
    policy = MDDPGPolicy(MDDPGConfig(**dict(state["config"])))
    policy.actor.load_state_dict(state["actor"])
    for attr, key in (("actor_target", "actor_target"), ("critic", "critic"), ("critic_target", "critic_target")):
        if key in state:
            getattr(policy, attr).load_state_dict(state[key])
    normalizer = state.get("normalizer")
    if normalizer is not None:
        policy._normalizer = np.asarray(normalizer, dtype=np.float32)
    last_action = state.get("last_action")
    if last_action is not None:
        policy._last_action = np.asarray(last_action, dtype=np.float32)
    if "hold_counter" in state:
        policy._hold_counter = int(state["hold_counter"])
    if "current_expl_noise" in state:
        policy._current_expl_noise = float(state["current_expl_noise"])
    policy.reset()
    return policy


def map_to_bounds(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return float(low + (float(value) + 1.0) * 0.5 * (high - low))


def decode_action_debug(action: np.ndarray, axis: str) -> dict[str, object]:
    b = get_axis_ladrc_action_bounds(axis)
    clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    deltas = {
        "r": map_to_bounds(float(clipped[0]), b.delta_r),
        "b0": map_to_bounds(float(clipped[1]), b.delta_b0),
        "omega_c": map_to_bounds(float(clipped[2]), b.delta_wc),
        "k": map_to_bounds(float(clipped[3]), b.delta_k),
    }
    raw = {
        "r": b.train_anchor.r + deltas["r"],
        "b0": b.train_anchor.b0 + deltas["b0"],
        "omega_c": b.train_anchor.wc + deltas["omega_c"],
        "k": b.train_anchor.k + deltas["k"],
    }
    ranges = {"r": b.r, "b0": b.b0, "omega_c": b.wc, "k": b.k}
    out: dict[str, object] = {f"action_{i}": float(action[i]) for i in range(4)}
    out.update({f"clipped_action_{i}": float(clipped[i]) for i in range(4)})
    for key in ("r", "b0", "omega_c", "k"):
        value = float(np.clip(raw[key], ranges[key][0], ranges[key][1]))
        out[f"delta_{key}"] = float(deltas[key])
        out[key] = value
        out[f"clip_hit_{key}"] = bool(abs(value - raw[key]) > 1e-9)
    return out


def obs_names(state_dim: int, stack_size: int) -> list[str]:
    base = ["pos_error", "vel_error", "pos", "vel", "target_pos", "target_vel", "residual", "progress"] if int(state_dim) == 8 else ["pos_error", "vel_error", "coupled_attitude", "coupled_angular_rate", "coupled_attitude_error", "ladrc_disturbance_estimate"]
    return [f"frame_{frame}_{name}" for frame in range(int(stack_size)) for name in base]


def compute_debug_metrics(rows: list[dict[str, float]], axis: str, config: PyBulletControlExperimentConfig) -> dict[str, float]:
    metrics = dict(compute_episode_metrics(rows, axis))
    t = np.asarray([row["time"] for row in rows], dtype=float)
    err = np.asarray([row[f"target_{axis}"] - row[axis] for row in rows], dtype=float)
    verr = np.asarray([row[f"target_v{axis}"] - row[f"v{axis}"] for row in rows], dtype=float)
    rpm = np.asarray([[row[f"rpm{i}"] for i in range(4)] for row in rows], dtype=float)
    drpm = np.diff(rpm, axis=0) if len(rpm) > 1 else np.zeros((1, 4), dtype=float)
    metrics.update(
        {
            "itae": float(np.sum(t * np.abs(err)) * config.control_dt),
            "max_error": float(np.max(np.abs(err))) if len(err) else 0.0,
            "velocity_rmse": float(np.sqrt(np.mean(verr**2))) if len(verr) else 0.0,
            "rpm_variance": float(np.var(rpm)) if rpm.size else 0.0,
            "mean_abs_delta_rpm": float(np.mean(np.abs(drpm))) if drpm.size else 0.0,
            "recovery_time": recovery_time(rows, axis, config),
            "disturbance_abs_mean": float(np.mean(np.abs([row[f"disturbance_{axis}"] for row in rows]))) if rows else 0.0,
        }
    )
    return metrics


def recovery_time(rows: list[dict[str, float]], axis: str, config: PyBulletControlExperimentConfig) -> float:
    axis_cfg = config.axis_config(axis)
    if not axis_cfg.include_disturbance or not axis_cfg.disturbance_step_window:
        return 0.0
    end = int(axis_cfg.disturbance_step_window[1])
    errors = [abs(row[f"target_{axis}"] - row[axis]) for row in rows]
    for idx in range(min(end, len(errors) - 1), len(errors)):
        if all(value <= 0.05 for value in errors[idx:]):
            return float((idx - end) * config.control_dt)
    return float(max(len(errors) - end, 0) * config.control_dt)


def trace_rows(rows: list[dict[str, float]], axis: str) -> dict[str, list[dict[str, object]]]:
    disturbance, attitude, rpm = [], [], []
    for step, row in enumerate(rows):
        disturbance.append({"step": step, "time": row["time"], "disturbance_x": row.get("disturbance_x", 0.0), "disturbance_y": row.get("disturbance_y", 0.0), "disturbance_z": row.get("disturbance_z", 0.0)})
        attitude.append({"step": step, "time": row["time"], "axis": axis, "roll": row.get("roll", 0.0), "pitch": row.get("pitch", 0.0), "yaw": row.get("yaw", 0.0), "roll_rate": row.get("roll_rate", 0.0), "pitch_rate": row.get("pitch_rate", 0.0)})
        rpm.append({"step": step, "time": row["time"], **{f"rpm{i}": row.get(f"rpm{i}", 0.0) for i in range(4)}, "rpm_mean": float(np.mean([row.get(f"rpm{i}", 0.0) for i in range(4)]))})
    return {"disturbance": disturbance, "attitude": attitude, "rpm": rpm}


def run_fixed(s: Scenario, config: PyBulletControlExperimentConfig, ref: Any) -> dict[str, Any]:
    env = create_ctrl_aviary(config)
    controller = create_controller_bundle(s.variant)
    apply_ladrc_anchor(controller, s.axis)
    try:
        artifacts = run_evaluation_episode(env, controller, ref, axis=s.axis, config=config)
    finally:
        close_ctrl_aviary(env)
    rows = list(artifacts.timeseries)
    action_rows = [{"step": i, "time": row["time"], "controller": "fixed_ladrc", **LADRC_0P5_OPT, "clip_hit_r": False, "clip_hit_b0": False, "clip_hit_omega_c": False, "clip_hit_k": False} for i, row in enumerate(rows)]
    return {"backend": artifacts.backend, "timeseries": rows, "logger_rows": list(artifacts.logger_rows), "observation_rows": [], "action_rows": action_rows}


def run_rl_transfer(s: Scenario, config: PyBulletControlExperimentConfig, ref: Any) -> dict[str, Any]:
    if s.checkpoint is None:
        raise ValueError("RL transfer scenario requires a checkpoint")
    env = create_ctrl_aviary(config)
    real_env = env.get("env")
    backend = str(env.get("backend", "fallback"))
    controller = create_controller_bundle(s.variant)
    apply_ladrc_anchor(controller, s.axis)
    policy = load_policy(s.checkpoint)
    axis_index = {"x": 0, "y": 1, "z": 2}[s.axis]
    state = _initial_state(ref)
    history: list[np.ndarray] = []
    prev_rpm = np.full(4, 4300.0, dtype=np.float32)
    rows: list[dict[str, float]] = []
    obs_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    logger_rows: list[dict[str, float]] = []
    names = obs_names(policy.config.state_dim, policy.config.stack_size)
    try:
        for step in range(config.step_count):
            target_pos, target_vel = ref.positions[step], ref.velocities[step]
            obs = _build_axis_observation(state, target_pos, target_vel, s.axis, controller, step=step, step_count=config.step_count, state_dim=int(policy.config.state_dim))
            stacked = stack_state(history, obs.copy(), policy.config.stack_size)
            action = policy.select_action(stacked, explore=False)
            action_debug = decode_action_debug(action, s.axis)
            params = _apply_axis_action(controller, s.axis, action)
            checkpoint = controller.snapshot_params()
            disturbance = _disturbance_vector(axis_index, step, config, config.axis_config(s.axis))
            if backend == "gym_env" and real_env is not None:
                rpm, _, _ = controller.compute_control_from_state(config.control_dt, state, target_pos, target_vel, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
                _apply_real_disturbance(real_env, disturbance)
                next_obs, _, terminated, truncated, _ = real_env.step(rpm.reshape(1, 4))
                next_state = np.asarray(next_obs[0], dtype=np.float32)
                done = bool(terminated or truncated or step == config.step_count - 1)
            else:
                next_state, rpm, _, _ = step_controller_loop(state, controller, target_pos=target_pos, target_vel=target_vel, control_dt=config.control_dt, disturbance=disturbance)
                done = step == config.step_count - 1
            reward = -abs(float(target_pos[axis_index] - next_state[axis_index]))
            row = _timeseries_row(step, config.control_dt, s.axis, state, target_pos, target_vel, rpm, reward, checkpoint, backend, disturbance=disturbance)
            rows.append(row)
            logger_rows.append(_logger_row(step, config.control_dt, state, target_pos, target_vel))
            obs_row: dict[str, object] = {"step": step, "time": row["time"], "axis": s.axis}
            for idx, value in enumerate(stacked):
                obs_row[f"obs_{idx}"] = float(value)
                obs_row[f"obs_{idx}_name"] = names[idx] if idx < len(names) else "unknown"
            obs_rows.append(obs_row)
            action_row: dict[str, object] = {"step": step, "time": row["time"], "axis": s.axis, **action_debug, "applied_r": float(params[0]), "applied_b0": float(params[1]), "applied_omega_c": float(params[2]), "applied_k": float(params[3]), "rpm_delta_mean": float(np.mean(np.abs(rpm - prev_rpm)))}
            action_rows.append(action_row)
            prev_rpm = np.asarray(rpm, dtype=np.float32)
            state = next_state
            if done:
                break
    finally:
        close_ctrl_aviary(env)
    return {"backend": backend, "timeseries": rows, "logger_rows": logger_rows, "observation_rows": obs_rows, "action_rows": action_rows, "policy_config": dict(policy.config.__dict__), "checkpoint_path": str(s.checkpoint)}


def write_outputs(sdir: Path, s: Scenario, config: PyBulletControlExperimentConfig, ref: Any, result: dict[str, Any]) -> dict[str, Any]:
    ensure_dir(sdir)
    rows = list(result["timeseries"])
    metrics = compute_debug_metrics(rows, s.axis, config)
    traces = trace_rows(rows, s.axis)
    paths = {
        "reference": write_reference_csv(sdir / "reference.csv", ref),
        "timeseries": write_timeseries_csv(sdir / "timeseries.csv", rows),
        "metrics": write_metrics_csv(sdir / "metrics.csv", [{"scenario": s.name, "axis": s.axis, **metrics}]),
        "disturbance_trace": write_metrics_csv(sdir / "disturbance_trace.csv", traces["disturbance"]),
        "action_params_trace": write_metrics_csv(sdir / "action_params_trace.csv", list(result.get("action_rows", []))),
        "observation_trace": write_metrics_csv(sdir / "observation_trace.csv", list(result.get("observation_rows", []))),
        "attitude_trace": write_metrics_csv(sdir / "attitude_trace.csv", traces["attitude"]),
        "rpm_trace": write_metrics_csv(sdir / "rpm_trace.csv", traces["rpm"]),
    }
    plot_scenario(sdir / "figures", s, rows)
    summary = {
        "scenario": s.name,
        "axis": s.axis,
        "kind": s.kind,
        "controller_variant": s.variant,
        "backend": result.get("backend"),
        "include_disturbance": s.disturbed,
        "disturbance_axis_expected": s.axis if s.disturbed else None,
        "disturbance_scale_n": DISTURBANCE_SCALE_N if s.disturbed else 0.0,
        "disturbance_step_window": list(DISTURBANCE_WINDOW),
        "fixed_stage_lengths": list(STAGE_LENGTHS),
        "fixed_stage_velocities": list(STAGE_VELOCITIES),
        "ladrc_params": LADRC_0P5_OPT,
        "metrics": metrics,
        "policy_config": result.get("policy_config"),
        "checkpoint_path": result.get("checkpoint_path"),
        "observation_names": obs_names(int((result.get("policy_config") or {}).get("state_dim", 8)), int((result.get("policy_config") or {}).get("stack_size", 1))) if s.kind == "rl_transfer" else [],
        "static_checks": static_checks(ref, rows, s),
        "paths": {key: str(value) for key, value in paths.items()},
    }
    write_json(sdir / "summary.json", summary)
    return summary


def static_checks(ref: Any, rows: list[dict[str, float]], s: Scenario) -> dict[str, bool]:
    other = "y" if s.axis == "x" else "x"
    target_axis = np.asarray([row[f"target_{s.axis}"] for row in rows], dtype=float)
    target_other = np.asarray([row[f"target_{other}"] for row in rows], dtype=float)
    dist_axis = np.asarray([row[f"disturbance_{s.axis}"] for row in rows], dtype=float)
    dist_other = np.asarray([row[f"disturbance_{other}"] for row in rows], dtype=float)
    bounds = get_axis_ladrc_action_bounds(s.axis)
    selected_r = np.asarray([row[f"{s.axis}_r"] for row in rows], dtype=float) if rows else np.asarray([])
    selected_b0 = np.asarray([row[f"{s.axis}_b0"] for row in rows], dtype=float) if rows else np.asarray([])
    selected_wc = np.asarray([row[f"{s.axis}_omega_c"] for row in rows], dtype=float) if rows else np.asarray([])
    selected_k = np.asarray([row[f"{s.axis}_k"] for row in rows], dtype=float) if rows else np.asarray([])
    in_bounds = bool(rows and np.all((bounds.r[0] <= selected_r) & (selected_r <= bounds.r[1])) and np.all((bounds.b0[0] <= selected_b0) & (selected_b0 <= bounds.b0[1])) and np.all((bounds.wc[0] <= selected_wc) & (selected_wc <= bounds.wc[1])) and np.all((bounds.k[0] <= selected_k) & (selected_k <= bounds.k[1])))
    return {
        "axis_is_expected": ref.axis == s.axis,
        "target_selected_axis_changes": bool(np.ptp(target_axis) > 1e-6),
        "target_other_axis_constant": bool(np.ptp(target_other) < 1e-6),
        "disturbance_selected_axis_nonzero_when_enabled": bool((not s.disturbed) or np.any(np.abs(dist_axis) > 0.0)),
        "disturbance_other_axis_zero": bool(np.all(np.abs(dist_other) < 1e-12)),
        "ladrc_selected_axis_params_in_bounds": in_bounds,
        "fixed_ladrc_anchor_written": bool(s.kind != "fixed_ladrc" or (rows and abs(float(rows[0][f"{s.axis}_r"]) - LADRC_0P5_OPT["r"]) < 1e-6)),
        "rl_selected_axis_params_updated": bool(s.kind != "rl_transfer" or (len(selected_r) > 0 and np.ptp(selected_r) > 1e-9)),
    }


def plot_lines(path: Path, title: str, ylabel: str, series: list[tuple[str, list[float], list[float], str, str]], hline: float | None = None) -> None:
    plt.figure(figsize=(10.2, 5.4))
    for label, x, y, color, style in series:
        plt.plot(x, y, label=label, color=color, linestyle=style, linewidth=2.0)
    if hline is not None:
        plt.axhline(hline, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    ensure_dir(path.parent)
    plt.savefig(path, dpi=180)
    plt.close()


def plot_scenario(figdir: Path, s: Scenario, rows: list[dict[str, float]]) -> None:
    ensure_dir(figdir)
    axis = s.axis
    t = [row["time"] for row in rows]
    plot_lines(figdir / "tracking.png", f"{s.name} tracking", "Position (m)", [("Reference", t, [row[f"target_{axis}"] for row in rows], "black", "--"), ("Actual", t, [row[axis] for row in rows], "#1f77b4", "-")])
    plot_lines(figdir / "velocity.png", f"{s.name} velocity", "Velocity (m/s)", [("Reference", t, [row[f"target_v{axis}"] for row in rows], "black", "--"), ("Actual", t, [row[f"v{axis}"] for row in rows], "#ff7f0e", "-")])
    plot_lines(figdir / "error.png", f"{s.name} error", "Tracking error (m)", [("Error", t, [row[f"target_{axis}"] - row[axis] for row in rows], "#d62728", "-")], hline=0.0)
    plot_lines(figdir / "attitude.png", f"{s.name} attitude", "Attitude (rad)", [("roll", t, [row["roll"] for row in rows], "#9467bd", "-"), ("pitch", t, [row["pitch"] for row in rows], "#2ca02c", "-")])
    plot_lines(figdir / "disturbance.png", f"{s.name} disturbance", "Force (N)", [("disturbance_x", t, [row["disturbance_x"] for row in rows], "#1f77b4", "-"), ("disturbance_y", t, [row["disturbance_y"] for row in rows], "#d62728", "-")])
    plot_lines(figdir / "params.png", f"{s.name} LADRC parameters", "Parameter", [("r", t, [row[f"{axis}_r"] for row in rows], "#1f77b4", "-"), ("b0", t, [row[f"{axis}_b0"] for row in rows], "#ff7f0e", "-"), ("omega_c", t, [row[f"{axis}_omega_c"] for row in rows], "#2ca02c", "-"), ("k", t, [row[f"{axis}_k"] for row in rows], "#d62728", "-")])
    plot_lines(figdir / "rpm.png", f"{s.name} RPM", "RPM", [(f"rpm{i}", t, [row[f"rpm{i}"] for row in rows], color, "-") for i, color in enumerate(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])])


def plot_comparisons(figdir: Path, summaries: list[dict[str, Any]], rows_by_name: dict[str, list[dict[str, float]]]) -> list[str]:
    ensure_dir(figdir)
    specs = {
        "fixed_ladrc_no_dist_x_vs_y": ["x_ladrc_no_dist", "y_ladrc_no_dist"],
        "fixed_ladrc_dist_x_vs_y": ["x_ladrc_dist", "y_ladrc_dist"],
        "y_ladrc_vs_rl_transfer_no_dist": ["y_ladrc_no_dist", "y_rl_transfer_no_dist"],
        "y_ladrc_vs_rl_transfer_dist": ["y_ladrc_dist", "y_rl_transfer_dist"],
    }
    paths: list[str] = []
    for name, labels in specs.items():
        labels = [label for label in labels if label in rows_by_name]
        if len(labels) < 2:
            continue
        t = [row["time"] for row in rows_by_name[labels[0]]]
        axis_for = {label: ("x" if label.startswith("x_") else "y") for label in labels}
        ref_axis = axis_for[labels[0]]
        tracking = [("Reference", t, [row[f"target_{ref_axis}"] for row in rows_by_name[labels[0]]], "black", "--")]
        error = []
        for idx, label in enumerate(labels):
            rows = rows_by_name[label]
            axis = axis_for[label]
            color = ["#1f77b4", "#d62728", "#2ca02c"][idx]
            tracking.append((label, t, [row[axis] for row in rows], color, "-"))
            error.append((label, t, [row[f"target_{axis}"] - row[axis] for row in rows], color, "-"))
        p1, p2 = figdir / f"{name}_tracking.png", figdir / f"{name}_error.png"
        plot_lines(p1, name.replace("_", " "), "Position (m)", tracking)
        plot_lines(p2, name.replace("_", " ") + " error", "Tracking error (m)", error, hline=0.0)
        paths += [str(p1), str(p2)]
    for metric in ("rmse", "recovery_time", "rpm_variance", "mean_abs_delta_rpm"):
        path = figdir / f"summary_{metric}_bar.png"
        labels = [s["scenario"] for s in summaries]
        values = [float(s["metrics"].get(metric, 0.0)) for s in summaries]
        plt.figure(figsize=(11, 5.6))
        plt.bar(labels, values, color="#4c78a8")
        plt.ylabel(metric)
        plt.xticks(rotation=25, ha="right")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(str(path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run x-to-y LADRC transfer debug comparisons with rich traces.")
    parser.add_argument("--tag", default=f"{timestamp()}_xy_axis_transfer_debug")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", type=Path, default=BEST_X_V2_CHECKPOINT)
    parser.add_argument("--skip-rl", action="store_true")
    args = parser.parse_args()

    out = PROJECT_ROOT / "outputs" / "control_pybullet" / "xy_axis_transfer_debug" / str(args.tag)
    ensure_dir(out)
    scenarios = [
        Scenario("x_ladrc_no_dist", "x", "ladrc_x_pos_pid_att", False, "fixed_ladrc"),
        Scenario("y_ladrc_no_dist", "y", "ladrc_y_pos_pid_att", False, "fixed_ladrc"),
        Scenario("x_ladrc_dist", "x", "ladrc_x_pos_pid_att", True, "fixed_ladrc"),
        Scenario("y_ladrc_dist", "y", "ladrc_y_pos_pid_att", True, "fixed_ladrc"),
    ]
    if not args.skip_rl:
        scenarios += [
            Scenario("y_rl_transfer_no_dist", "y", "ladrc_y_pos_pid_att", False, "rl_transfer", Path(args.checkpoint)),
            Scenario("y_rl_transfer_dist", "y", "ladrc_y_pos_pid_att", True, "rl_transfer", Path(args.checkpoint)),
        ]
    summaries: list[dict[str, Any]] = []
    rows_by_name: dict[str, list[dict[str, float]]] = {}
    for s in scenarios:
        sdir = out / s.name
        cfg = build_config(sdir, s.axis, s.disturbed, args.seed)
        ref = build_xyz_reference_trajectory(cfg.axis_config(s.axis), cfg, rng=np.random.default_rng(args.seed))
        result = run_fixed(s, cfg, ref) if s.kind == "fixed_ladrc" else run_rl_transfer(s, cfg, ref)
        summary = write_outputs(sdir, s, cfg, ref, result)
        summaries.append(summary)
        rows_by_name[s.name] = list(result["timeseries"])
    summary_rows = [{"scenario": s["scenario"], "axis": s["axis"], "kind": s["kind"], "include_disturbance": s["include_disturbance"], **{k: float(v) for k, v in s["metrics"].items() if isinstance(v, (int, float, np.floating))}} for s in summaries]
    write_metrics_csv(out / "summary_metrics.csv", summary_rows)
    figures = plot_comparisons(out / "figures", summaries, rows_by_name)
    manifest = {
        "output_root": str(out),
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "ladrc_params": LADRC_0P5_OPT,
        "action_bounds_y": {"r": list(get_axis_ladrc_action_bounds("y").r), "b0": list(get_axis_ladrc_action_bounds("y").b0), "omega_c": list(get_axis_ladrc_action_bounds("y").wc), "k": list(get_axis_ladrc_action_bounds("y").k)},
        "disturbance": {"mode": "random_uniform", "scale_n": DISTURBANCE_SCALE_N, "window_steps": list(DISTURBANCE_WINDOW), "window_seconds": [DISTURBANCE_WINDOW[0] / CONTROL_FREQ_HZ, DISTURBANCE_WINDOW[1] / CONTROL_FREQ_HZ]},
        "scenarios": summaries,
        "summary_metrics_path": str(out / "summary_metrics.csv"),
        "comparison_figures": figures,
    }
    write_json(out / "summary.json", manifest)
    print(json.dumps({"output_root": str(out), "summary_metrics": str(out / "summary_metrics.csv")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()




