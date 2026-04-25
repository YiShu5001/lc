from __future__ import annotations

import argparse
import json
import sys
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
from lc.control.reference_generators import build_xyz_reference_trajectory, summarize_reference_segments
from lc.control.simulators import close_ctrl_aviary, create_ctrl_aviary, run_policy_episode
from lc.rl.algorithms import MDDPGConfig, MDDPGPolicy

CONTROL_FREQ_HZ = 48
FORWARD_STEPS = 96
HOVER_STEPS = 48
REVERSE_STEPS = 96
FINAL_HOLD_STEPS = 24
HOVER_GAP_STEPS = 10
DISTURBANCE_STEPS = 28
STEP_COUNT = FORWARD_STEPS + HOVER_STEPS + REVERSE_STEPS + FINAL_HOLD_STEPS
DURATION_SEC = STEP_COUNT / CONTROL_FREQ_HZ

BEST_V2_CHECKPOINT = PROJECT_ROOT / "outputs" / "control_pybullet_rl" / "x_refline_random_hover_disturbance_mddpg_retrain" / "20260412_random_hover_disturb_v1234_300eps_narrowed_noise0p2_0p004_posonly_multiseed_normfix" / "v_2" / "train" / "ladrc_x_pos_pid_att" / "x" / "20260412_185851" / "checkpoints" / "x_policy_best.pt"


def _timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_config(output_root: Path, disturbance_scale: float) -> PyBulletControlExperimentConfig:
    return PyBulletControlExperimentConfig(
        control_freq_hz=CONTROL_FREQ_HZ,
        duration_sec=DURATION_SEC,
        seed=7,
        train_episodes=1,
        eval_episodes=1,
        compare_episodes=1,
        eval_seeds=(7, 17, 27),
        snapshot_interval=0,
        training_controller_variant="ladrc_y_pos_pid_att",
        artifact=ArtifactConfig(
            output_root=str(output_root),
            export_structured=True,
            export_legacy_logger=True,
            save_figures=True,
            record_video=False,
        ),
        axis_configs=(
            AxisTrainingConfig(
                axis="y",
                initial_position=(0.0, 0.0, 1.0),
                fixed_axes=(1.0, 0.0),
                include_disturbance=True,
                disturbance_scale=disturbance_scale,
                disturbance_axis_bias=1.0,
                disturbance_step_window=(FORWARD_STEPS + HOVER_GAP_STEPS, FORWARD_STEPS + HOVER_GAP_STEPS + DISTURBANCE_STEPS),
                disturbance_mode="random_uniform",
                fixed_stage_lengths=(FORWARD_STEPS, HOVER_STEPS, REVERSE_STEPS, FINAL_HOLD_STEPS),
                fixed_stage_velocities=(0.5, 0.0, -0.6, 0.0),
            ),
        ),
    )


def _apply_0p5_opt_params(controller) -> None:
    controller.set_axis_parameters("y", r=63.0, b0=24.3, omega_c=2.95, k=7.415254237288136)


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


def _run_rl_policy_episode(
    config: PyBulletControlExperimentConfig,
    reference_bundle,
    checkpoint_path: Path,
) -> dict[str, object]:
    env = create_ctrl_aviary(config)
    controller = create_controller_bundle("ladrc_y_pos_pid_att")
    _apply_0p5_opt_params(controller)
    policy = _load_policy(checkpoint_path)
    try:
        artifacts = run_policy_episode(
            env,
            policy,
            controller,
            reference_bundle,
            axis="y",
            config=config,
            explore=False,
            store_transitions=False,
            n_step=1,
        )
        return {"timeseries": list(artifacts.timeseries), "backend": artifacts.backend}
    finally:
        close_ctrl_aviary(env)


def _compute_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    x_err = np.asarray([row["target_y"] - row["y"] for row in rows], dtype=float)
    vx_err = np.asarray([row["target_vy"] - row["vy"] for row in rows], dtype=float)
    disturbance_x = np.asarray([row.get("disturbance_y", 0.0) for row in rows], dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(x_err**2))),
        "mae": float(np.mean(np.abs(x_err))),
        "velocity_rmse": float(np.sqrt(np.mean(vx_err**2))),
        "reward": float(np.mean([row["reward"] for row in rows])) if rows else 0.0,
        "disturbance_abs_mean": float(np.mean(np.abs(disturbance_x))) if rows else 0.0,
    }


def _build_color_map(labels: list[str]) -> dict[str, str]:
    color_cycle = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    colors: dict[str, str] = {}
    for idx, label in enumerate(labels):
        colors[label] = color_cycle[idx % len(color_cycle)]
    return colors


def _plot_tracking(rows_by_label: dict[str, list[dict[str, float]]], output_path: Path) -> Path:
    plt.figure(figsize=(10.5, 5.6))
    first_rows = next(iter(rows_by_label.values()))
    t = [row["time"] for row in first_rows]
    plt.plot(t, [row["target_y"] for row in first_rows], label="Reference", linewidth=2.4, color="black", linestyle="--")
    colors = _build_color_map(list(rows_by_label.keys()))
    for label, rows in rows_by_label.items():
        plt.plot(t, [row["y"] for row in rows], label=label, linewidth=2.0, color=colors.get(label))
    plt.xlabel("Time (s)")
    plt.ylabel("Y Position (m)")
    plt.title("PyBullet Y-Axis Tracking With Random Hover Disturbance")
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
    plt.plot(t, [row["target_vy"] for row in first_rows], label="Reference", linewidth=2.4, color="black", linestyle="--")
    colors = _build_color_map(list(rows_by_label.keys()))
    for label, rows in rows_by_label.items():
        plt.plot(t, [row["vy"] for row in rows], label=label, linewidth=2.0, color=colors.get(label))
    plt.xlabel("Time (s)")
    plt.ylabel("Y Velocity (m/s)")
    plt.title("PyBullet Y-Axis Velocity With Random Hover Disturbance")
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
    colors = _build_color_map(list(rows_by_label.keys()))
    for label, rows in rows_by_label.items():
        plt.plot(t, [row["target_y"] - row["y"] for row in rows], label=label, linewidth=2.0, color=colors.get(label))
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Tracking Error (m)")
    plt.title("PyBullet Y-Axis Error With Random Hover Disturbance")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path



def _plot_attitude(rows_by_label: dict[str, list[dict[str, float]]], output_path: Path) -> Path:
    plt.figure(figsize=(10.5, 5.6))
    first_rows = next(iter(rows_by_label.values()))
    t = [row["time"] for row in first_rows]
    colors = _build_color_map(list(rows_by_label.keys()))
    for label, rows in rows_by_label.items():
        plt.plot(t, [row["roll"] for row in rows], label=f"{label}-roll", linewidth=2.0, color=colors.get(label), linestyle='-')
        plt.plot(t, [row["pitch"] for row in rows], label=f"{label}-pitch", linewidth=1.6, color=colors.get(label), linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Attitude (rad)")
    plt.title("PyBullet Y-Axis Roll/Pitch Response")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PID, LADRC and DDPG-LADRC under random hover disturbance in PyBullet.")
    parser.add_argument("--disturbance-scale", type=float, default=0.004)
    parser.add_argument("--tag", default=_timestamp_tag())
    parser.add_argument("--checkpoint", default=str(BEST_V2_CHECKPOINT))
    parser.add_argument("--policy-label", default="DDPG-LADRC")
    args = parser.parse_args()

    tag = args.tag
    output_root = PROJECT_ROOT / "outputs" / "control_pybullet" / "y_pid_ladrc_ddpg_random_hover_disturb_compare" / "y" / tag
    config = _build_config(output_root, float(args.disturbance_scale))
    checkpoint_path = Path(args.checkpoint)
    policy_label = str(args.policy_label)
    seeds = tuple(int(seed) for seed in config.eval_seeds)
    rows_by_label_accum: dict[str, list[list[dict[str, float]]]] = {
        "PID": [],
        "LADRC(0.5-opt)": [],
        policy_label: [],
    }
    metric_rows_accum: dict[str, list[dict[str, float]]] = {
        "PID": [],
        "LADRC(0.5-opt)": [],
        policy_label: [],
    }
    reference_bundle = None
    for eval_seed in seeds:
        reference_bundle = build_xyz_reference_trajectory(config.axis_config("y"), config, rng=np.random.default_rng(eval_seed))
        pid_result = run_controller_episode(config, create_controller_bundle("pid_pos_att"), reference_bundle)
        ladrc_controller = create_controller_bundle("ladrc_y_pos_pid_att")
        _apply_0p5_opt_params(ladrc_controller)
        ladrc_result = run_controller_episode(config, ladrc_controller, reference_bundle)
        ddpg_result = _run_rl_policy_episode(config, reference_bundle, checkpoint_path)
        seed_rows = {
            "PID": list(pid_result["timeseries"]),
            "LADRC(0.5-opt)": list(ladrc_result["timeseries"]),
            policy_label: list(ddpg_result["timeseries"]),
        }
        for label, rows in seed_rows.items():
            rows_by_label_accum[label].append(rows)
            metric_rows_accum[label].append(_compute_metrics(rows))
    rows_by_label = {label: runs[0] for label, runs in rows_by_label_accum.items()}

    figures_dir = output_root / "figures"
    tracking_figure = _plot_tracking(rows_by_label, figures_dir / "tracking_compare.png")
    velocity_figure = _plot_velocity(rows_by_label, figures_dir / "velocity_compare.png")
    error_figure = _plot_error(rows_by_label, figures_dir / "error_compare.png")
    attitude_figure = _plot_attitude(rows_by_label, figures_dir / "attitude_compare.png")

    write_reference_csv(output_root / "reference.csv", reference_bundle)
    write_timeseries_csv(output_root / "pid_timeseries.csv", rows_by_label["PID"])
    write_timeseries_csv(output_root / "ladrc_y_0p5_opt_timeseries.csv", rows_by_label["LADRC(0.5-opt)"])
    write_timeseries_csv(output_root / "ddpg_best_timeseries.csv", rows_by_label[policy_label])

    metric_rows = []
    for label, rows in metric_rows_accum.items():
        metric_rows.append(
            {
                "controller": label,
                "rmse": float(np.mean([row["rmse"] for row in rows])),
                "mae": float(np.mean([row["mae"] for row in rows])),
                "velocity_rmse": float(np.mean([row["velocity_rmse"] for row in rows])),
                "reward": float(np.mean([row["reward"] for row in rows])),
                "disturbance_abs_mean": float(np.mean([row["disturbance_abs_mean"] for row in rows])),
            }
        )
    write_metrics_csv(output_root / "metrics.csv", metric_rows)
    write_summary_json(
        output_root / "summary.json",
        {
            "backend": "gym_env",
            "eval_seeds": list(seeds),
            "disturbance": {
                "mode": "random_uniform",
                "scale_n": float(args.disturbance_scale),
                "window_steps": [FORWARD_STEPS + HOVER_GAP_STEPS, FORWARD_STEPS + HOVER_GAP_STEPS + DISTURBANCE_STEPS],
                "window_seconds": [
                    (FORWARD_STEPS + HOVER_GAP_STEPS) / CONTROL_FREQ_HZ,
                    (FORWARD_STEPS + HOVER_GAP_STEPS + DISTURBANCE_STEPS) / CONTROL_FREQ_HZ,
                ],
                "hover_gap_seconds": HOVER_GAP_STEPS / CONTROL_FREQ_HZ,
                "effective_duration_seconds": DISTURBANCE_STEPS / CONTROL_FREQ_HZ,
                "axis": "y",
                "seed": config.seed,
            },
            "reference_segments": summarize_reference_segments(reference_bundle),
            "controllers": {
                "PID": {"source": "pid_pos_att_y_eval"},
                "LADRC(0.5-opt)": {"source": "fixed_ladrc_0.5_opt_y", "r": 63.0, "b0": 24.3, "omega_c": 2.95, "k": 7.415254237288136},
                policy_label: {"checkpoint": str(checkpoint_path), "source_axis": "y", "target_axis": "y", "checkpoint_type": "best"},
            },
            "metrics": metric_rows,
            "figures": [str(tracking_figure), str(velocity_figure), str(error_figure), str(attitude_figure)],
        },
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "metrics": metric_rows,
                "figures": [str(tracking_figure), str(velocity_figure), str(error_figure), str(attitude_figure)],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
