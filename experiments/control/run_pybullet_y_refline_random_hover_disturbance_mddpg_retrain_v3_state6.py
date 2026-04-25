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

from lc.common.io import ensure_dir
from lc.control.configs import ArtifactConfig, AxisTrainingConfig, PyBulletControlExperimentConfig, get_axis_ladrc_action_bounds
from lc.control.io import write_summary_json
from lc.control.plotting import plot_reward_curve_collection
from lc.control.trainers import PyBulletAxisTrainer
from lc.rl.algorithms import MDDPGConfig

TRAIN_EPISODES = 300
DEFAULT_HIDDEN_DIM = 768
DEFAULT_DROPOUT_P = 0.25
DEFAULT_TAU = 0.02
DEFAULT_SOFT_UPDATE_INTERVAL = 10
DEFAULT_SNAPSHOT_INTERVAL = 20
DEFAULT_EXPL_NOISE_START = 0.2
DEFAULT_EXPL_NOISE_END = 0.02
DEFAULT_EXPL_NOISE_SCHEDULE = "three_phase"
DEFAULT_BATCH_SIZE = 128
SHARED_VALUE = 3
CONTROL_FREQ_HZ = 48
FORWARD_STEPS = 96
HOVER_STEPS = 48
REVERSE_STEPS = 96
FINAL_HOLD_STEPS = 24
HOVER_GAP_STEPS = 10
DISTURBANCE_STEPS = 28
STEP_COUNT = FORWARD_STEPS + HOVER_STEPS + REVERSE_STEPS + FINAL_HOLD_STEPS
DURATION_SEC = STEP_COUNT / CONTROL_FREQ_HZ
DISTURBANCE_SCALE = 0.004
EVAL_SEEDS = (7, 17, 27)
STATE_TERMS = [
    "pos_error",
    "vel_error",
    "coupled_attitude",
    "coupled_angular_rate",
    "coupled_attitude_error",
    "ladrc_disturbance_estimate",
]


def _timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_config(output_root: Path, train_episodes: int, snapshot_interval: int) -> PyBulletControlExperimentConfig:
    return PyBulletControlExperimentConfig(
        control_freq_hz=CONTROL_FREQ_HZ,
        duration_sec=DURATION_SEC,
        seed=7,
        warmup_steps=64,
        train_episodes=train_episodes,
        eval_episodes=len(EVAL_SEEDS),
        compare_episodes=len(EVAL_SEEDS),
        eval_seeds=EVAL_SEEDS,
        updates_per_step=1,
        batch_size=DEFAULT_BATCH_SIZE,
        snapshot_interval=snapshot_interval,
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
                disturbance_scale=DISTURBANCE_SCALE,
                disturbance_axis_bias=1.0,
                disturbance_mode="random_uniform",
                disturbance_step_window=(FORWARD_STEPS + HOVER_GAP_STEPS, FORWARD_STEPS + HOVER_GAP_STEPS + DISTURBANCE_STEPS),
                fixed_stage_lengths=(FORWARD_STEPS, HOVER_STEPS, REVERSE_STEPS, FINAL_HOLD_STEPS),
                fixed_stage_velocities=(0.5, 0.0, -0.6, 0.0),
            ),
        ),
    )


def _build_policy_config(args: argparse.Namespace) -> MDDPGConfig:
    return MDDPGConfig(
        state_dim=6,
        action_dim=4,
        hidden_dim=int(args.hidden_dim),
        batch_size=int(args.batch_size),
        stack_size=SHARED_VALUE,
        action_hold_steps=SHARED_VALUE,
        gamma=0.95,
        tau=float(args.tau),
        soft_update_interval=int(args.soft_update_interval),
        dropout_p=float(args.dropout_p),
        expl_noise=float(args.exploration_noise_start),
        expl_noise_start=float(args.exploration_noise_start),
        expl_noise_end=float(args.exploration_noise_end),
        expl_noise_schedule=str(args.exploration_noise_schedule),
    )


def _plot_average_reward(history: list[dict[str, float]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5.2))
    episodes = [int(float(row["episode"])) for row in history]
    avg_reward = [float(row["average_reward"]) for row in history]
    reward = [float(row["reward"]) for row in history]
    plt.plot(episodes, avg_reward, linewidth=2.2, label="Average Reward")
    plt.plot(episodes, reward, linewidth=1.0, alpha=0.35, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Y-axis DDPG-LADRC(v=3) Reward Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train y-axis RL-LADRC v=3 with 6D no-progress state under random hover disturbance.")
    parser.add_argument("--episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--dropout-p", type=float, default=DEFAULT_DROPOUT_P)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--soft-update-interval", type=int, default=DEFAULT_SOFT_UPDATE_INTERVAL)
    parser.add_argument("--snapshot-interval", type=int, default=DEFAULT_SNAPSHOT_INTERVAL)
    parser.add_argument("--exploration-noise-start", type=float, default=DEFAULT_EXPL_NOISE_START)
    parser.add_argument("--exploration-noise-end", type=float, default=DEFAULT_EXPL_NOISE_END)
    parser.add_argument("--exploration-noise-schedule", default=DEFAULT_EXPL_NOISE_SCHEDULE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--tag", default=f"{_timestamp_tag()}_v3_300eps_state6_no_progress")
    args = parser.parse_args()

    output_root = PROJECT_ROOT / "outputs" / "control_pybullet_rl" / "y_refline_random_hover_disturbance_mddpg_retrain" / args.tag
    ensure_dir(output_root)
    bounds = get_axis_ladrc_action_bounds("y")
    config = _build_config(output_root / "v_3", train_episodes=int(args.episodes), snapshot_interval=int(args.snapshot_interval))
    trainer = PyBulletAxisTrainer(config)
    result = trainer.train_axis("y", policy_config=_build_policy_config(args), n_step=SHARED_VALUE, shared_value=SHARED_VALUE)
    history = list(result["history"])

    figures_dir = ensure_dir(output_root / "figures")
    reward_curve_collection = plot_reward_curve_collection(
        {"v=3": history},
        figures_dir / "reward_curves_by_shared_value.svg",
        "Y-axis Random-Hover-Disturbance RL-LADRC Reward Curve",
    )
    reward_curve_png = _plot_average_reward(history, figures_dir / "reward_episode_curve.png")
    best_eval_metrics = dict(result.get("best_eval_metrics") or {})
    summary = {
        "backend": result["backend"],
        "axis": "y",
        "controller_variant": "ladrc_y_pos_pid_att",
        "train_episodes": int(args.episodes),
        "eval_seeds": list(EVAL_SEEDS),
        "shared_value": SHARED_VALUE,
        "state_dim": 6,
        "state_terms": STATE_TERMS,
        "progress_removed": True,
        "no_time_or_stage_input": True,
        "no_true_disturbance_input": True,
        "action_dim": 4,
        "action_order": ["r", "b0", "omega_c", "k"],
        "action_bounds": {"r": list(bounds.r), "b0": list(bounds.b0), "omega_c": list(bounds.wc), "k": list(bounds.k)},
        "delta_bounds": {"r": list(bounds.delta_r), "b0": list(bounds.delta_b0), "omega_c": list(bounds.delta_wc), "k": list(bounds.delta_k)},
        "train_anchor": {
            "r": bounds.train_anchor.r,
            "b0": bounds.train_anchor.b0,
            "omega_c": bounds.train_anchor.wc,
            "k": bounds.train_anchor.k,
            "omega_o": bounds.train_anchor.wo,
        },
        "network_config": {
            "hidden_dim": int(args.hidden_dim),
            "dropout_p": float(args.dropout_p),
            "tau": float(args.tau),
            "soft_update_interval": int(args.soft_update_interval),
            "batch_size": int(args.batch_size),
            "gamma": 0.95,
            "exploration_noise_schedule": str(args.exploration_noise_schedule),
            "exploration_noise_start": float(args.exploration_noise_start),
            "exploration_noise_end": float(args.exploration_noise_end),
            "snapshot_interval": int(args.snapshot_interval),
        },
        "scenario_definition": {
            "control_freq_hz": CONTROL_FREQ_HZ,
            "duration_sec": DURATION_SEC,
            "include_disturbance": True,
            "disturbance_axis": "y",
            "disturbance_mode": "random_uniform",
            "disturbance_scale_n": DISTURBANCE_SCALE,
            "disturbance_step_window": [FORWARD_STEPS + HOVER_GAP_STEPS, FORWARD_STEPS + HOVER_GAP_STEPS + DISTURBANCE_STEPS],
            "disturbance_window_seconds": [
                (FORWARD_STEPS + HOVER_GAP_STEPS) / CONTROL_FREQ_HZ,
                (FORWARD_STEPS + HOVER_GAP_STEPS + DISTURBANCE_STEPS) / CONTROL_FREQ_HZ,
            ],
            "fixed_stage_lengths": [FORWARD_STEPS, HOVER_STEPS, REVERSE_STEPS, FINAL_HOLD_STEPS],
            "fixed_stage_velocities": [0.5, 0.0, -0.6, 0.0],
        },
        "checkpoint_path": str(result["checkpoint_path"]),
        "best_checkpoint_path": str(result["best_checkpoint_path"]),
        "output_dir": str(result["output_dir"]),
        "average_reward": float(result["average_reward"]),
        "best_reward": max((float(row["reward"]) for row in history), default=0.0),
        "best_eval_metrics": best_eval_metrics,
        "figures": [str(reward_curve_collection), str(reward_curve_png), *[str(path) for path in result["figures"]]],
    }
    write_summary_json(output_root / "summary.json", summary)
    print(json.dumps({"output_root": str(output_root), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
