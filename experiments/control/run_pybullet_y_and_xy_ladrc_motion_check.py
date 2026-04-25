from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lc.control.configs import ArtifactConfig, AxisTrainingConfig, PyBulletControlExperimentConfig
from lc.control.controllers import create_controller_bundle
from lc.control.io import write_reference_csv, write_summary_json, write_timeseries_csv
from lc.control.policies.stacking import stack_state
from lc.control.reference_generators.piecewise_velocity import ReferenceBundle
from lc.control.simulators import close_ctrl_aviary, create_ctrl_aviary, run_evaluation_episode, run_policy_episode
from lc.control.simulators.pybullet_runner import _apply_axis_action, _build_axis_observation, _compute_axis_reward, _disturbance_vector, _ensure_real_env, _timeseries_row
from lc.rl.algorithms import MDDPGConfig, MDDPGPolicy


CONTROL_FREQ_HZ = 48
DURATION_SEC = 5.5
STEP_COUNT = int(CONTROL_FREQ_HZ * DURATION_SEC)
STAGE_LENGTHS = (96, 48, 96, 24)
STAGE_VELOCITIES = (0.5, 0.0, -0.6, 0.0)
LADRC_PARAMS = {"r": 63.0, "b0": 24.3, "omega_c": 2.95, "k": 7.415254237288136}
BEST_V2_CHECKPOINT = (
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
    / "checkpoints"
    / "x_policy_best.pt"
)


@dataclass(frozen=True)
class MotionMetrics:
    rmse_x: float
    rmse_y: float
    rmse_xy: float
    mae_xy: float
    max_abs_x: float
    max_abs_y: float
    final_abs_x: float
    final_abs_y: float
    z_rmse: float
    max_abs_roll: float
    max_abs_pitch: float


def _timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_config(output_root: Path, *, training_controller_variant: str) -> PyBulletControlExperimentConfig:
    return PyBulletControlExperimentConfig(
        control_freq_hz=CONTROL_FREQ_HZ,
        duration_sec=DURATION_SEC,
        seed=7,
        train_episodes=1,
        eval_episodes=1,
        compare_episodes=1,
        snapshot_interval=0,
        training_controller_variant=training_controller_variant,
        artifact=ArtifactConfig(
            output_root=str(output_root),
            export_structured=True,
            export_legacy_logger=True,
            save_figures=True,
            record_video=False,
        ),
        axis_configs=(
            AxisTrainingConfig(
                axis="x",
                initial_position=(0.0, 0.0, 1.0),
                fixed_axes=(0.0, 1.0),
                fixed_stage_lengths=STAGE_LENGTHS,
                fixed_stage_velocities=STAGE_VELOCITIES,
                include_disturbance=False,
            ),
            AxisTrainingConfig(
                axis="y",
                initial_position=(0.0, 0.0, 1.0),
                fixed_axes=(0.0, 1.0),
                fixed_stage_lengths=STAGE_LENGTHS,
                fixed_stage_velocities=STAGE_VELOCITIES,
                include_disturbance=False,
            ),
        ),
    )


def _piecewise_positions(initial: float, velocities: tuple[float, ...], lengths: tuple[int, ...], dt: float) -> tuple[np.ndarray, np.ndarray]:
    velocity_profile = np.concatenate(
        [np.full(length, velocity, dtype=np.float32) for length, velocity in zip(lengths, velocities)]
    )
    positions = np.zeros(len(velocity_profile), dtype=np.float32)
    value = float(initial)
    for index, velocity in enumerate(velocity_profile):
        positions[index] = value
        value += float(velocity) * dt
    return positions, velocity_profile


def _build_y_reference(config: PyBulletControlExperimentConfig) -> ReferenceBundle:
    y_pos, y_vel = _piecewise_positions(0.0, STAGE_VELOCITIES, STAGE_LENGTHS, config.control_dt)
    positions = np.zeros((config.step_count, 3), dtype=np.float32)
    velocities = np.zeros((config.step_count, 3), dtype=np.float32)
    positions[:, 2] = 1.0
    positions[:, 1] = y_pos
    velocities[:, 1] = y_vel
    slices = []
    start = 0
    for length in STAGE_LENGTHS:
        stop = start + length
        slices.append(slice(start, stop))
        start = stop
    return ReferenceBundle(axis="y", positions=positions, velocities=velocities, stage_slices=tuple(slices), stage_velocities=STAGE_VELOCITIES)


def _build_xy_reference(config: PyBulletControlExperimentConfig) -> ReferenceBundle:
    x_pos, x_vel = _piecewise_positions(0.0, STAGE_VELOCITIES, STAGE_LENGTHS, config.control_dt)
    y_pos, y_vel = _piecewise_positions(0.0, STAGE_VELOCITIES, STAGE_LENGTHS, config.control_dt)
    positions = np.zeros((config.step_count, 3), dtype=np.float32)
    velocities = np.zeros((config.step_count, 3), dtype=np.float32)
    positions[:, 0] = x_pos
    positions[:, 1] = y_pos
    positions[:, 2] = 1.0
    velocities[:, 0] = x_vel
    velocities[:, 1] = y_vel
    slices = []
    start = 0
    for length in STAGE_LENGTHS:
        stop = start + length
        slices.append(slice(start, stop))
        start = stop
    return ReferenceBundle(axis="x", positions=positions, velocities=velocities, stage_slices=tuple(slices), stage_velocities=STAGE_VELOCITIES)


def _apply_params(controller, axes: tuple[str, ...]) -> None:
    for axis in axes:
        controller.set_axis_parameters(axis, **LADRC_PARAMS)


def _load_policy(checkpoint_path: Path = BEST_V2_CHECKPOINT) -> MDDPGPolicy:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy_state = checkpoint.get("policy_state", checkpoint)
    policy = MDDPGPolicy(MDDPGConfig(**dict(policy_state["config"])))
    policy.actor.load_state_dict(policy_state["actor"])
    policy.critic.load_state_dict(policy_state["critic"])
    policy.actor_target.load_state_dict(policy_state.get("actor_target", policy_state["actor"]))
    policy.critic_target.load_state_dict(policy_state.get("critic_target", policy_state["critic"]))
    policy._normalizer = np.asarray(policy_state.get("normalizer", policy._normalizer), dtype=np.float32).copy()
    policy._last_action = np.asarray(policy_state.get("last_action", policy._last_action), dtype=np.float32).copy()
    policy._hold_counter = int(policy_state.get("hold_counter", 0))
    policy._current_expl_noise = float(policy_state.get("current_expl_noise", 0.0))
    policy.reset()
    return policy


def _run_xy_policy_episode(
    env: dict[str, object],
    policy_x: MDDPGPolicy,
    policy_y: MDDPGPolicy,
    controller,
    reference_bundle: ReferenceBundle,
    config: PyBulletControlExperimentConfig,
) -> tuple[list[dict[str, float]], list[dict[str, float]], list[float]]:
    real_env = _ensure_real_env(env, config, reference_bundle)
    observation, _ = real_env.reset(seed=config.seed)
    state = np.asarray(observation[0], dtype=np.float32)
    histories = {"x": [], "y": []}
    action_trace: list[dict[str, float]] = []
    timeseries: list[dict[str, float]] = []
    rewards: list[float] = []
    prev_rpm = np.full(4, 4300.0, dtype=np.float32)
    axis_indices = {"x": 0, "y": 1}
    policy_x.reset()
    policy_y.reset()
    for step in range(config.step_count):
        target_pos = reference_bundle.positions[step]
        target_vel = reference_bundle.velocities[step]
        raw_actions: dict[str, np.ndarray] = {}
        applied_params: dict[str, np.ndarray] = {}
        for axis, policy in (("x", policy_x), ("y", policy_y)):
            observation_axis = _build_axis_observation(
                state,
                target_pos,
                target_vel,
                axis,
                controller,
                step=step,
                step_count=config.step_count,
                state_dim=int(policy.config.state_dim),
            )
            stacked = stack_state(histories[axis], observation_axis.copy(), policy.config.stack_size)
            action = policy.select_action(stacked, explore=False)
            raw_actions[axis] = action.copy()
            applied_params[axis] = _apply_axis_action(controller, axis, action)
            next_history = list(histories[axis])
            next_history.append(observation_axis.copy())
            histories[axis] = next_history[-policy.config.stack_size :]
        checkpoint = controller.snapshot_params()
        rpm, _, _ = controller.compute_control_from_state(
            control_timestep=config.control_dt,
            state=state,
            target_pos=target_pos,
            target_vel=target_vel,
            target_rpy=np.zeros(3, dtype=np.float32),
            target_rpy_rates=np.zeros(3, dtype=np.float32),
        )
        next_obs, _, terminated, truncated, _ = real_env.step(rpm.reshape(1, 4))
        next_state = np.asarray(next_obs[0], dtype=np.float32)
        rpm_delta = float(np.mean(np.abs(rpm - prev_rpm)))
        per_axis_rewards = []
        for axis, index in axis_indices.items():
            per_axis_rewards.append(
                _compute_axis_reward(
                    float(target_pos[index] - next_state[index]),
                    float(target_vel[index] - next_state[10 + index]),
                    rpm_delta,
                    step=step,
                    step_count=config.step_count,
                    target_vel=float(target_vel[index]),
                    param_delta=0.0,
                )
            )
        reward = float(np.mean(per_axis_rewards))
        rewards.append(reward)
        timeseries.append(
            _timeseries_row(
                step,
                config.control_dt,
                "xy",
                state,
                target_pos,
                target_vel,
                rpm,
                reward,
                checkpoint,
                str(env["backend"]),
                disturbance=np.zeros(3, dtype=np.float32),
            )
        )
        row: dict[str, float] = {"step": float(step), "time": float(step * config.control_dt)}
        for axis in ("x", "y"):
            action = raw_actions[axis]
            params = applied_params[axis]
            row.update(
                {
                    f"{axis}_action_0": float(action[0]),
                    f"{axis}_action_1": float(action[1]),
                    f"{axis}_action_2": float(action[2]),
                    f"{axis}_action_3": float(action[3]),
                    f"{axis}_r": float(params[0]),
                    f"{axis}_b0": float(params[1]),
                    f"{axis}_omega_c": float(params[2]),
                    f"{axis}_k": float(params[3]),
                }
            )
        action_trace.append(row)
        state = next_state
        prev_rpm = rpm
        if bool(terminated or truncated):
            break
    return timeseries, action_trace, rewards


def _metrics(rows: list[dict[str, float]]) -> MotionMetrics:
    target_x = np.asarray([row["target_x"] for row in rows], dtype=float)
    target_y = np.asarray([row["target_y"] for row in rows], dtype=float)
    target_z = np.asarray([row["target_z"] for row in rows], dtype=float)
    x = np.asarray([row["x"] for row in rows], dtype=float)
    y = np.asarray([row["y"] for row in rows], dtype=float)
    z = np.asarray([row["z"] for row in rows], dtype=float)
    roll = np.asarray([row["roll"] for row in rows], dtype=float)
    pitch = np.asarray([row["pitch"] for row in rows], dtype=float)
    err_x = target_x - x
    err_y = target_y - y
    err_xy = np.sqrt(err_x**2 + err_y**2)
    return MotionMetrics(
        rmse_x=float(np.sqrt(np.mean(err_x**2))),
        rmse_y=float(np.sqrt(np.mean(err_y**2))),
        rmse_xy=float(np.sqrt(np.mean(err_xy**2))),
        mae_xy=float(np.mean(np.abs(err_xy))),
        max_abs_x=float(np.max(np.abs(err_x))),
        max_abs_y=float(np.max(np.abs(err_y))),
        final_abs_x=float(abs(err_x[-1])),
        final_abs_y=float(abs(err_y[-1])),
        z_rmse=float(np.sqrt(np.mean((target_z - z) ** 2))),
        max_abs_roll=float(np.max(np.abs(roll))),
        max_abs_pitch=float(np.max(np.abs(pitch))),
    )


def _plot_y(rows_by_label: dict[str, list[dict[str, float]]], output_path: Path) -> Path:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.6), sharex=True)
    colors = {"PID": "#1f77b4", "固定LADRC": "#b22222", "DDPG-LADRC": "#2ca02c"}
    first = next(iter(rows_by_label.values()))
    t = [row["time"] for row in first]
    axes[0].plot(t, [row["target_y"] for row in first], "--", color="#222222", linewidth=2.2, label="参考 y")
    for label, rows in rows_by_label.items():
        axes[0].plot([row["time"] for row in rows], [row["y"] for row in rows], color=colors[label], linewidth=2.0, label=label)
        axes[1].plot(
            [row["time"] for row in rows],
            [row["target_y"] - row["y"] for row in rows],
            color=colors[label],
            linewidth=1.8,
            label=f"{label} 误差",
        )
    axes[0].set_ylabel("y 位置 / m")
    axes[1].set_ylabel("跟踪误差 / m")
    axes[1].set_xlabel("时间 / s")
    axes[0].set_title("y 轴单独运动轨迹跟踪")
    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend(frameon=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_xy(rows_by_label: dict[str, list[dict[str, float]]], output_path: Path) -> Path:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    colors = {"PID": "#1f77b4", "固定LADRC(x/y)": "#b22222", "DDPG-LADRC(x/y)": "#2ca02c"}
    first = next(iter(rows_by_label.values()))
    ax.plot([row["target_x"] for row in first], [row["target_y"] for row in first], "--", color="#222222", linewidth=2.4, label="参考轨迹")
    for label, rows in rows_by_label.items():
        ax.plot([row["x"] for row in rows], [row["y"] for row in rows], color=colors[label], linewidth=2.1, label=label)
    ax.scatter(first[0]["target_x"], first[0]["target_y"], color="#2ca02c", s=60, zorder=4, label="起点")
    ax.set_xlabel("x 位置 / m")
    ax.set_ylabel("y 位置 / m")
    ax.set_title("x/y 同时运动平面轨迹")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_metrics_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check y-axis and xy-axis LADRC position-loop replacement trajectories.")
    parser.add_argument("--tag", default=_timestamp_tag())
    args = parser.parse_args()

    output_root = PROJECT_ROOT / "outputs" / "control_pybullet" / "y_and_xy_ladrc_motion_check" / args.tag
    y_config = _build_config(output_root / "y_single", training_controller_variant="ladrc_y_pos_pid_att")
    xy_config = _build_config(output_root / "xy_together", training_controller_variant="ladrc_xy_pos_pid_att")
    y_ref = _build_y_reference(y_config)
    xy_ref = _build_xy_reference(xy_config)

    y_env = create_ctrl_aviary(y_config)
    try:
        y_pid = create_controller_bundle("pid_pos_att")
        y_ladrc = create_controller_bundle("ladrc_y_pos_pid_att")
        y_ddpg = create_controller_bundle("ladrc_y_pos_pid_att")
        y_policy = _load_policy()
        _apply_params(y_ladrc, ("y",))
        _apply_params(y_ddpg, ("y",))
        y_pid_artifacts = run_evaluation_episode(y_env, y_pid, y_ref, axis="y", config=y_config)
        y_ladrc_artifacts = run_evaluation_episode(y_env, y_ladrc, y_ref, axis="y", config=y_config)
        y_ddpg_artifacts = run_policy_episode(
            y_env,
            y_policy,
            y_ddpg,
            y_ref,
            "y",
            y_config,
            explore=False,
            store_transitions=False,
            n_step=int(y_policy.config.stack_size),
        )
    finally:
        close_ctrl_aviary(y_env)

    xy_env = create_ctrl_aviary(xy_config)
    try:
        xy_pid = create_controller_bundle("pid_pos_att")
        xy_ladrc = create_controller_bundle("ladrc_xy_pos_pid_att")
        xy_ddpg = create_controller_bundle("ladrc_xy_pos_pid_att")
        xy_policy_x = _load_policy()
        xy_policy_y = _load_policy()
        _apply_params(xy_ladrc, ("x", "y"))
        _apply_params(xy_ddpg, ("x", "y"))
        xy_pid_artifacts = run_evaluation_episode(xy_env, xy_pid, xy_ref, axis="x", config=xy_config)
        xy_ladrc_artifacts = run_evaluation_episode(xy_env, xy_ladrc, xy_ref, axis="x", config=xy_config)
        xy_ddpg_timeseries, xy_ddpg_action_trace, xy_ddpg_rewards = _run_xy_policy_episode(
            xy_env,
            xy_policy_x,
            xy_policy_y,
            xy_ddpg,
            xy_ref,
            xy_config,
        )
    finally:
        close_ctrl_aviary(xy_env)

    y_rows = {
        "PID": list(y_pid_artifacts.timeseries),
        "固定LADRC": list(y_ladrc_artifacts.timeseries),
        "DDPG-LADRC": list(y_ddpg_artifacts.timeseries),
    }
    xy_rows = {
        "PID": list(xy_pid_artifacts.timeseries),
        "固定LADRC(x/y)": list(xy_ladrc_artifacts.timeseries),
        "DDPG-LADRC(x/y)": xy_ddpg_timeseries,
    }

    write_reference_csv(output_root / "y_single" / "reference.csv", y_ref)
    write_timeseries_csv(output_root / "y_single" / "pid_timeseries.csv", y_rows["PID"])
    write_timeseries_csv(output_root / "y_single" / "ladrc_y_timeseries.csv", y_rows["固定LADRC"])
    write_timeseries_csv(output_root / "y_single" / "ddpg_ladrc_y_timeseries.csv", y_rows["DDPG-LADRC"])
    write_reference_csv(output_root / "xy_together" / "reference.csv", xy_ref)
    write_timeseries_csv(output_root / "xy_together" / "pid_timeseries.csv", xy_rows["PID"])
    write_timeseries_csv(output_root / "xy_together" / "ladrc_xy_timeseries.csv", xy_rows["固定LADRC(x/y)"])
    write_timeseries_csv(output_root / "xy_together" / "ddpg_ladrc_xy_timeseries.csv", xy_rows["DDPG-LADRC(x/y)"])
    _write_metrics_csv(output_root / "xy_together" / "ddpg_ladrc_xy_action_trace.csv", xy_ddpg_action_trace)

    y_fig = _plot_y(y_rows, output_root / "figures" / "y_axis_single_motion_tracking.png")
    xy_fig = _plot_xy(xy_rows, output_root / "figures" / "xy_together_motion_trajectory.png")

    metrics_rows = [
        {"scenario": "y_single", "controller": "PID", **asdict(_metrics(y_rows["PID"]))},
        {"scenario": "y_single", "controller": "fixed_ladrc_y", **asdict(_metrics(y_rows["固定LADRC"]))},
        {"scenario": "y_single", "controller": "ddpg_ladrc_y", **asdict(_metrics(y_rows["DDPG-LADRC"]))},
        {"scenario": "xy_together", "controller": "PID", **asdict(_metrics(xy_rows["PID"]))},
        {"scenario": "xy_together", "controller": "fixed_ladrc_xy", **asdict(_metrics(xy_rows["固定LADRC(x/y)"]))},
        {"scenario": "xy_together", "controller": "ddpg_ladrc_xy", **asdict(_metrics(xy_rows["DDPG-LADRC(x/y)"]))},
    ]
    _write_metrics_csv(output_root / "metrics_summary.csv", metrics_rows)
    summary = {
        "output_root": str(output_root),
        "config": {
            "control_freq_hz": CONTROL_FREQ_HZ,
            "duration_sec": DURATION_SEC,
            "stage_lengths": STAGE_LENGTHS,
            "stage_velocities": STAGE_VELOCITIES,
            "ladrc_params": LADRC_PARAMS,
            "variants": {
                "y_single": "ladrc_y_pos_pid_att",
                "xy_together": "ladrc_xy_pos_pid_att",
            },
            "ddpg_checkpoint": str(BEST_V2_CHECKPOINT),
        },
        "figures": [str(y_fig), str(xy_fig)],
        "metrics": metrics_rows,
    }
    write_summary_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
