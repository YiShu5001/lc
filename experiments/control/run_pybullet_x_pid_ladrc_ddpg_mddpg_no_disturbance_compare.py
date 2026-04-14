from __future__ import annotations

import argparse
import json
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
from lc.control.simulators import close_ctrl_aviary, create_ctrl_aviary, run_policy_episode
from lc.control.simulators.pybullet_runner import (
    _apply_axis_action,
)
from lc.rl.algorithms import MDDPGConfig, MDDPGPolicy

CONTROL_FREQ_HZ = 48
FORWARD_STEPS = 96
HOVER_STEPS = 48
REVERSE_STEPS = 96
FINAL_HOLD_STEPS = 24
STEP_COUNT = FORWARD_STEPS + HOVER_STEPS + REVERSE_STEPS + FINAL_HOLD_STEPS
DURATION_SEC = STEP_COUNT / CONTROL_FREQ_HZ


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
                include_disturbance=False,
                disturbance_scale=0.0,
                disturbance_axis_bias=1.0,
                fixed_stage_lengths=(FORWARD_STEPS, HOVER_STEPS, REVERSE_STEPS, FINAL_HOLD_STEPS),
                fixed_stage_velocities=(0.5, 0.0, -0.6, 0.0),
            ),
        ),
    )


def _find_checkpoint(run_root: Path, shared_value: int) -> Path:
    return next((run_root / f"v_{shared_value}").glob("train/ladrc_x_pos_pid_att/x/*/checkpoints/x_policy_best.pt"))


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


def _apply_0p5_opt_params(controller) -> None:
    controller.set_axis_parameters("x", r=63.0, b0=24.3, omega_c=2.95, k=7.415254237288136)


def _run_rl_policy_episode(
    config: PyBulletControlExperimentConfig,
    reference_bundle,
    spec: RLRunSpec,
) -> dict[str, object]:
    env = create_ctrl_aviary(config)
    controller = create_controller_bundle("ladrc_x_pos_pid_att")
    _apply_0p5_opt_params(controller)
    policy = _load_policy(spec.checkpoint_path)
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
        return {"timeseries": list(artifacts.timeseries), "backend": artifacts.backend}
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
        "LADRC(0.5-opt)": "#d62728",
        "DDPG-LADRC (v=1)": "#2ca02c",
        "mDDPG-LADRC (v=5)": "#9467bd",
    }
    for label, rows in rows_by_label.items():
        plt.plot(t, [row["x"] for row in rows], label=label, linewidth=2.0, color=colors.get(label))
    plt.xlabel("Time (s)")
    plt.ylabel("X Position (m)")
    plt.title("No-Disturbance PyBullet X-Axis Tracking Comparison")
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
        "LADRC(0.5-opt)": "#d62728",
        "DDPG-LADRC (v=1)": "#2ca02c",
        "mDDPG-LADRC (v=5)": "#9467bd",
    }
    for label, rows in rows_by_label.items():
        plt.plot(t, [row["vx"] for row in rows], label=label, linewidth=2.0, color=colors.get(label))
    plt.xlabel("Time (s)")
    plt.ylabel("X Velocity (m/s)")
    plt.title("No-Disturbance PyBullet X-Axis Velocity Tracking Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _plot_error(rows_by_label: dict[str, list[dict[str, float]]], output_path: Path) -> Path:
    plt.figure(figsize=(10.5, 5.6))
    first_rows = next(iter(rows_by_label.values()))
    t = [row["time"] for row in first_rows]
    colors = {
        "PID": "#1f77b4",
        "LADRC(0.5-opt)": "#d62728",
        "DDPG-LADRC (v=1)": "#2ca02c",
        "mDDPG-LADRC (v=5)": "#9467bd",
    }
    for label, rows in rows_by_label.items():
        plt.plot(t, [row["target_x"] - row["x"] for row in rows], label=label, linewidth=2.0, color=colors.get(label))
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Tracking Error (m)")
    plt.title("No-Disturbance PyBullet X-Axis Tracking Error Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PID, fixed LADRC, DDPG-LADRC and mDDPG-LADRC on no-disturbance route.")
    parser.add_argument("--run-root", required=True, help="Root output directory of no-disturbance RL retraining results.")
    parser.add_argument("--tag", default=_timestamp_tag())
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_root = PROJECT_ROOT / "outputs" / "control_pybullet_rl" / "x_pid_ladrc_ddpg_mddpg_no_disturbance_compare" / args.tag
    config = _build_config(output_root)
    reference_bundle = build_xyz_reference_trajectory(config.axis_config("x"), config, rng=np.random.default_rng(config.seed))

    pid_result = run_controller_episode(config, create_controller_bundle("pid_pos_att"), reference_bundle)
    ladrc_controller = create_controller_bundle("ladrc_x_pos_pid_att")
    _apply_0p5_opt_params(ladrc_controller)
    ladrc_result = run_controller_episode(config, ladrc_controller, reference_bundle)

    rl_specs = [
        RLRunSpec("DDPG-LADRC (v=1)", 1, _find_checkpoint(run_root, 1)),
        RLRunSpec("mDDPG-LADRC (v=5)", 5, _find_checkpoint(run_root, 5)),
    ]
    ddpg_result = _run_rl_policy_episode(config, reference_bundle, rl_specs[0])
    mddpg_result = _run_rl_policy_episode(config, reference_bundle, rl_specs[1])

    rows_by_label = {
        "PID": list(pid_result["timeseries"]),
        "LADRC(0.5-opt)": list(ladrc_result["timeseries"]),
        "DDPG-LADRC (v=1)": list(ddpg_result["timeseries"]),
        "mDDPG-LADRC (v=5)": list(mddpg_result["timeseries"]),
    }

    figures_dir = output_root / "figures"
    tracking_figure = _plot_tracking(rows_by_label, figures_dir / "tracking_compare.png")
    velocity_figure = _plot_velocity(rows_by_label, figures_dir / "velocity_compare.png")
    error_figure = _plot_error(rows_by_label, figures_dir / "error_compare.png")

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
            "disturbance": "none",
            "reference_segments": summarize_reference_segments(reference_bundle),
            "controllers": {
                "PID": {"source": "pid_pos_att"},
                "LADRC(0.5-opt)": {"source": "fixed_ladrc_0.5_opt", "r": 63.0, "b0": 24.3, "omega_c": 2.95, "k": 7.415254237288136},
                "DDPG-LADRC (v=1)": {"checkpoint": str(rl_specs[0].checkpoint_path)},
                "mDDPG-LADRC (v=5)": {"checkpoint": str(rl_specs[1].checkpoint_path)},
            },
            "metrics": metric_rows,
            "figures": [str(tracking_figure), str(velocity_figure), str(error_figure)],
        },
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "metrics": metric_rows,
                "figures": [str(tracking_figure), str(velocity_figure), str(error_figure)],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
