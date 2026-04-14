from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
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
from lc.control.envs import run_controller_episode
from lc.control.io import write_metrics_csv, write_reference_csv, write_summary_json, write_timeseries_csv
from lc.control.policies.stacking import stack_state
from lc.control.reference_generators import build_xyz_reference_trajectory, summarize_reference_segments
from lc.control.simulators import close_ctrl_aviary, create_ctrl_aviary
from lc.control.simulators.pybullet_runner import (
    _apply_axis_action,
    _apply_real_disturbance,
    _build_axis_observation,
    _disturbance_vector,
    _ensure_real_env,
    _timeseries_row,
)
from lc.rl.algorithms import MDDPGConfig, MDDPGPolicy

CONTROL_FREQ_HZ = 48
FORWARD_STEPS = 96
HOVER_STEPS = 48
REVERSE_STEPS = 96
FINAL_HOLD_STEPS = 24
STEP_COUNT = FORWARD_STEPS + HOVER_STEPS + REVERSE_STEPS + FINAL_HOLD_STEPS
DURATION_SEC = STEP_COUNT / CONTROL_FREQ_HZ
DISTURBANCE_WINDOW = (FORWARD_STEPS, FORWARD_STEPS + HOVER_STEPS)
DISTURBANCE_FREQUENCY_RAD = 2.0 * math.pi * 10.0 / CONTROL_FREQ_HZ
RETUNED_RUN_ROOT = Path(
    r"D:\ZhangC\lc\outputs\control_pybullet_rl\x_refline_sine_disturbance_mddpg_v_sweep\20260412_4d_retuned_net_vsweep"
)


@dataclass(frozen=True)
class RLRunSpec:
    label: str
    shared_value: int
    checkpoint_path: Path


def _timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_config(output_root: Path) -> PyBulletControlExperimentConfig:
    return PyBulletControlExperimentConfig(
        control_freq_hz=CONTROL_FREQ_HZ,
        duration_sec=DURATION_SEC,
        seed=7,
        train_episodes=1,
        eval_episodes=1,
        compare_episodes=1,
        snapshot_interval=0,
        training_controller_variant="ladrc_x_pos_pid_att",
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
                include_disturbance=True,
                disturbance_scale=0.03,
                disturbance_axis_bias=1.0,
                fixed_stage_lengths=(FORWARD_STEPS, HOVER_STEPS, REVERSE_STEPS, FINAL_HOLD_STEPS),
                fixed_stage_velocities=(0.5, 0.0, -0.6, 0.0),
                disturbance_step_window=DISTURBANCE_WINDOW,
                disturbance_frequency_rad=DISTURBANCE_FREQUENCY_RAD,
            ),
        ),
    )


def _default_rl_specs() -> list[RLRunSpec]:
    return [
        RLRunSpec(
            label="ddpg_ladrc_v1",
            shared_value=1,
            checkpoint_path=next((RETUNED_RUN_ROOT / "v_1").glob("train/ladrc_x_pos_pid_att/x/*/checkpoints/x_policy_best.pt")),
        ),
        RLRunSpec(
            label="mddpg_ladrc_v5",
            shared_value=5,
            checkpoint_path=next((RETUNED_RUN_ROOT / "v_5").glob("train/ladrc_x_pos_pid_att/x/*/checkpoints/x_policy_best.pt")),
        ),
    ]


def _load_policy(checkpoint_path: Path) -> MDDPGPolicy:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg_dict = dict(checkpoint["policy_state"]["config"])
    policy = MDDPGPolicy(MDDPGConfig(**cfg_dict))
    policy.actor.load_state_dict(checkpoint["policy_state"]["actor"])
    if "actor_target" in checkpoint["policy_state"]:
        policy.actor_target.load_state_dict(checkpoint["policy_state"]["actor_target"])
    if "critic" in checkpoint["policy_state"]:
        policy.critic.load_state_dict(checkpoint["policy_state"]["critic"])
    if "critic_target" in checkpoint["policy_state"]:
        policy.critic_target.load_state_dict(checkpoint["policy_state"]["critic_target"])
    policy.reset()
    return policy


def _run_rl_policy_episode(
    config: PyBulletControlExperimentConfig,
    reference_bundle,
    spec: RLRunSpec,
) -> dict[str, object]:
    env = create_ctrl_aviary(config)
    controller = create_controller_bundle("ladrc_x_pos_pid_att")
    policy = _load_policy(spec.checkpoint_path)
    try:
        real_env = _ensure_real_env(env, config, reference_bundle)
        obs, _ = real_env.reset(seed=config.seed)
        state = np.asarray(obs[0], dtype=np.float32)
        controller.reset()
        policy.reset()
        history: list[np.ndarray] = []
        timeseries: list[dict[str, float]] = []
        axis = "x"
        axis_index = 0
        prev_rpm = np.full(4, 4300.0, dtype=np.float32)
        for step in range(config.step_count):
            target_pos = reference_bundle.positions[step]
            target_vel = reference_bundle.velocities[step]
            observation = _build_axis_observation(state, target_pos, target_vel, axis, step, config.step_count)
            stacked_observation = stack_state(history, observation.copy(), policy.config.stack_size)
            action = policy.select_action(stacked_observation, explore=False)
            checkpoint = controller.snapshot_params()
            _apply_axis_action(controller, axis, action)
            disturbance = _disturbance_vector(axis_index, step, config, config.axis_config(axis))
            rpm, _, _ = controller.compute_control_from_state(
                control_timestep=config.control_dt,
                state=state,
                target_pos=target_pos,
                target_vel=target_vel,
                target_rpy=np.zeros(3, dtype=np.float32),
                target_rpy_rates=np.zeros(3, dtype=np.float32),
            )
            _apply_real_disturbance(real_env, disturbance)
            next_obs, _, terminated, truncated, _ = real_env.step(rpm.reshape(1, 4))
            next_state = np.asarray(next_obs[0], dtype=np.float32)
            pos_error = float(target_pos[axis_index] - next_state[axis_index])
            vel_error = float(target_vel[axis_index] - next_state[10 + axis_index])
            rpm_delta = float(np.mean(np.abs(rpm - prev_rpm)))
            reward = float(-abs(pos_error) - 0.15 * abs(vel_error) - 0.0008 * rpm_delta)
            timeseries.append(
                _timeseries_row(
                    step,
                    config.control_dt,
                    axis,
                    state,
                    target_pos,
                    target_vel,
                    rpm,
                    reward,
                    checkpoint,
                    env["backend"],
                    disturbance=disturbance,
                )
            )
            state = next_state
            prev_rpm = rpm
            if terminated or truncated or step == config.step_count - 1:
                break
        return {"timeseries": timeseries, "backend": env["backend"]}
    finally:
        close_ctrl_aviary(env)


def _compute_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    x_err = np.asarray([row["target_x"] - row["x"] for row in rows], dtype=float)
    vx_err = np.asarray([row["target_vx"] - row["vx"] for row in rows], dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(x_err**2))),
        "mae": float(np.mean(np.abs(x_err))),
        "velocity_rmse": float(np.sqrt(np.mean(vx_err**2))),
        "reward": float(np.mean([row["reward"] for row in rows])) if rows else 0.0,
    }


def _plot_tracking(rows_by_label: dict[str, list[dict[str, float]]], output_path: Path) -> Path:
    plt.figure(figsize=(10.5, 5.6))
    first_rows = next(iter(rows_by_label.values()))
    t = [row["time"] for row in first_rows]
    plt.plot(t, [row["target_x"] for row in first_rows], label="Reference", linewidth=2.4, color="black", linestyle="--")
    colors = {
        "PID": "#1f77b4",
        "LADRC": "#d62728",
        "DDPG-LADRC (v=1)": "#2ca02c",
        "mDDPG-LADRC (v=5)": "#9467bd",
    }
    for label, rows in rows_by_label.items():
        plt.plot(t, [row["x"] for row in rows], label=label, linewidth=2.0, color=colors.get(label))
    plt.xlabel("Time (s)")
    plt.ylabel("X Position (m)")
    plt.title("PyBullet X-Axis Tracking Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _plot_velocity(rows_by_label: dict[str, list[dict[str, float]]], output_path: Path) -> Path:
    plt.figure(figsize=(10.5, 5.6))
    first_rows = next(iter(rows_by_label.values()))
    t = [row["time"] for row in first_rows]
    plt.plot(t, [row["target_vx"] for row in first_rows], label="Reference", linewidth=2.4, color="black", linestyle="--")
    colors = {
        "PID": "#1f77b4",
        "LADRC": "#d62728",
        "DDPG-LADRC (v=1)": "#2ca02c",
        "mDDPG-LADRC (v=5)": "#9467bd",
    }
    for label, rows in rows_by_label.items():
        plt.plot(t, [row["vx"] for row in rows], label=label, linewidth=2.0, color=colors.get(label))
    plt.xlabel("Time (s)")
    plt.ylabel("X Velocity (m/s)")
    plt.title("PyBullet X-Axis Velocity Tracking Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def main() -> None:
    tag = _timestamp_tag()
    output_root = PROJECT_ROOT / "outputs" / "control_pybullet_rl" / "x_pid_ladrc_ddpg_mddpg_compare" / tag
    config = _build_config(output_root)
    reference_bundle = build_xyz_reference_trajectory(config.axis_config("x"), config, rng=np.random.default_rng(config.seed))

    pid_result = run_controller_episode(config, create_controller_bundle("pid_pos_att"), reference_bundle)
    ladrc_result = run_controller_episode(config, create_controller_bundle("ladrc_x_pos_pid_att"), reference_bundle)

    rl_specs = _default_rl_specs()
    ddpg_result = _run_rl_policy_episode(config, reference_bundle, rl_specs[0])
    mddpg_result = _run_rl_policy_episode(config, reference_bundle, rl_specs[1])

    rows_by_label = {
        "PID": list(pid_result["timeseries"]),
        "LADRC": list(ladrc_result["timeseries"]),
        "DDPG-LADRC (v=1)": list(ddpg_result["timeseries"]),
        "mDDPG-LADRC (v=5)": list(mddpg_result["timeseries"]),
    }

    figures_dir = output_root / "figures"
    tracking_figure = _plot_tracking(rows_by_label, figures_dir / "tracking_compare.png")
    velocity_figure = _plot_velocity(rows_by_label, figures_dir / "velocity_compare.png")

    metric_rows: list[dict[str, object]] = []
    for label, rows in rows_by_label.items():
        controller_dir = output_root / label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_")
        write_timeseries_csv(controller_dir / "timeseries.csv", rows)
        metrics = _compute_metrics(rows)
        metrics["controller"] = label
        metric_rows.append(metrics)

    write_reference_csv(output_root / "reference.csv", reference_bundle)
    write_metrics_csv(output_root / "metrics.csv", metric_rows)
    write_summary_json(
        output_root / "summary.json",
        {
            "backend": "gym_env",
            "reference_segments": summarize_reference_segments(reference_bundle),
            "controllers": {
                "PID": {"source": "pid_pos_att"},
                "LADRC": {"source": "ladrc_x_pos_pid_att"},
                "DDPG-LADRC (v=1)": {"checkpoint": str(rl_specs[0].checkpoint_path)},
                "mDDPG-LADRC (v=5)": {"checkpoint": str(rl_specs[1].checkpoint_path)},
            },
            "metrics": metric_rows,
            "figures": [str(tracking_figure), str(velocity_figure)],
        },
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "metrics": metric_rows,
                "figures": [str(tracking_figure), str(velocity_figure)],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
