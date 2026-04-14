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
                disturbance_scale=disturbance_scale,
                disturbance_axis_bias=1.0,
                disturbance_mode="random_uniform",
                disturbance_step_window=(FORWARD_STEPS + HOVER_GAP_STEPS, FORWARD_STEPS + HOVER_GAP_STEPS + DISTURBANCE_STEPS),
                fixed_stage_lengths=(FORWARD_STEPS, HOVER_STEPS, REVERSE_STEPS, FINAL_HOLD_STEPS),
                fixed_stage_velocities=(0.5, 0.0, -0.6, 0.0),
            ),
        ),
    )


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


def _run_rl_policy_episode(config: PyBulletControlExperimentConfig, reference_bundle, checkpoint_path: Path) -> dict[str, object]:
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
        return {"timeseries": list(artifacts.timeseries), "backend": artifacts.backend}
    finally:
        close_ctrl_aviary(env)


def _compute_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    x_err = np.asarray([row["target_x"] - row["x"] for row in rows], dtype=float)
    vx_err = np.asarray([row["target_vx"] - row["vx"] for row in rows], dtype=float)
    disturbance_x = np.asarray([row.get("disturbance_x", 0.0) for row in rows], dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(x_err**2))),
        "mae": float(np.mean(np.abs(x_err))),
        "velocity_rmse": float(np.sqrt(np.mean(vx_err**2))),
        "reward": float(np.mean([row["reward"] for row in rows])) if rows else 0.0,
        "disturbance_abs_mean": float(np.mean(np.abs(disturbance_x))) if rows else 0.0,
    }


def _build_color_map(labels: list[str]) -> dict[str, str]:
    color_cycle = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    return {label: color_cycle[idx % len(color_cycle)] for idx, label in enumerate(labels)}


def _plot_series(rows_by_label: dict[str, list[dict[str, float]]], output_path: Path, key: str, ylabel: str, title: str, target_key: str) -> Path:
    plt.figure(figsize=(10.5, 5.6))
    first_rows = next(iter(rows_by_label.values()))
    t = [row["time"] for row in first_rows]
    plt.plot(t, [row[target_key] for row in first_rows], label="Reference", linewidth=2.4, color="black", linestyle="--")
    colors = _build_color_map(list(rows_by_label.keys()))
    for label, rows in rows_by_label.items():
        plt.plot(t, [row[key] for row in rows], label=label, linewidth=2.0, color=colors.get(label))
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
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
        plt.plot(t, [row["target_x"] - row["x"] for row in rows], label=label, linewidth=2.0, color=colors.get(label))
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Tracking Error (m)")
    plt.title("PyBullet X-Axis Error: RL Triplet Under Random Hover Disturbance")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare three RL checkpoints under random hover disturbance in PyBullet.")
    parser.add_argument("--disturbance-scale", type=float, default=0.008)
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--checkpoint-c", required=True)
    parser.add_argument("--label-a", default="DDPG-LADRC (v=1)")
    parser.add_argument("--label-b", default="DDPG-LADRC (v=3)")
    parser.add_argument("--label-c", default="DDPG-LADRC (v=6)")
    parser.add_argument("--tag", default=_timestamp_tag())
    args = parser.parse_args()

    output_root = PROJECT_ROOT / "outputs" / "control_pybullet_rl" / "x_rl_triplet_random_hover_compare" / args.tag
    config = _build_config(output_root, float(args.disturbance_scale))
    reference_bundle = build_xyz_reference_trajectory(config.axis_config("x"), config, rng=np.random.default_rng(config.seed))

    rows_by_label = {
        str(args.label_a): list(_run_rl_policy_episode(config, reference_bundle, Path(args.checkpoint_a))["timeseries"]),
        str(args.label_b): list(_run_rl_policy_episode(config, reference_bundle, Path(args.checkpoint_b))["timeseries"]),
        str(args.label_c): list(_run_rl_policy_episode(config, reference_bundle, Path(args.checkpoint_c))["timeseries"]),
    }

    figures_dir = output_root / "figures"
    tracking_figure = _plot_series(rows_by_label, figures_dir / "tracking_compare.png", "x", "X Position (m)", "PyBullet X-Axis Tracking: RL Triplet Under Random Hover Disturbance", "target_x")
    velocity_figure = _plot_series(rows_by_label, figures_dir / "velocity_compare.png", "vx", "X Velocity (m/s)", "PyBullet X-Axis Velocity: RL Triplet Under Random Hover Disturbance", "target_vx")
    error_figure = _plot_error(rows_by_label, figures_dir / "error_compare.png")

    write_reference_csv(output_root / "reference.csv", reference_bundle)
    metric_rows = []
    for index, (label, rows) in enumerate(rows_by_label.items(), start=1):
        write_timeseries_csv(output_root / f"rl_{index}_timeseries.csv", rows)
        metric_rows.append({"controller": label, **_compute_metrics(rows)})
    write_metrics_csv(output_root / "metrics.csv", metric_rows)
    write_summary_json(
        output_root / "summary.json",
        {
            "backend": "gym_env",
            "disturbance": {
                "mode": "random_uniform",
                "scale_n": float(args.disturbance_scale),
                "window_steps": [FORWARD_STEPS + HOVER_GAP_STEPS, FORWARD_STEPS + HOVER_GAP_STEPS + DISTURBANCE_STEPS],
                "window_seconds": [
                    (FORWARD_STEPS + HOVER_GAP_STEPS) / CONTROL_FREQ_HZ,
                    (FORWARD_STEPS + HOVER_GAP_STEPS + DISTURBANCE_STEPS) / CONTROL_FREQ_HZ,
                ],
            },
            "reference_segments": summarize_reference_segments(reference_bundle),
            "controllers": [
                {"label": str(args.label_a), "checkpoint": str(Path(args.checkpoint_a))},
                {"label": str(args.label_b), "checkpoint": str(Path(args.checkpoint_b))},
                {"label": str(args.label_c), "checkpoint": str(Path(args.checkpoint_c))},
            ],
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
