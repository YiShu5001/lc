from __future__ import annotations

import argparse
import json
import math
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
from lc.control.configs import (
    ArtifactConfig,
    AxisTrainingConfig,
    PyBulletControlExperimentConfig,
    get_axis_ladrc_action_bounds,
)
from lc.control.io import write_metrics_csv, write_summary_json
from lc.control.plotting import plot_reward_curve_collection
from lc.control.trainers import PyBulletAxisTrainer
from lc.rl.algorithms import MDDPGConfig

TRAIN_EPISODES = 300
COMPARE_EPISODES = 5
SEED_RUNS = 1
DEFAULT_HIDDEN_DIM = 768
DEFAULT_DROPOUT_P = 0.25
DEFAULT_TAU = 0.02
DEFAULT_SOFT_UPDATE_INTERVAL = 10
DEFAULT_SNAPSHOT_INTERVAL = 20
DEFAULT_EXPL_NOISE_START = 0.1
DEFAULT_EXPL_NOISE_END = 0.04
DEFAULT_EXPL_NOISE_SCHEDULE = "linear"
SHARED_VALUES = (1, 3, 5, 7, 9)
CONTROL_FREQ_HZ = 48
FORWARD_STEPS = 96
HOVER_STEPS = 48
REVERSE_STEPS = 96
FINAL_HOLD_STEPS = 24
STEP_COUNT = FORWARD_STEPS + HOVER_STEPS + REVERSE_STEPS + FINAL_HOLD_STEPS
DURATION_SEC = STEP_COUNT / CONTROL_FREQ_HZ
DISTURBANCE_WINDOW = (FORWARD_STEPS, FORWARD_STEPS + HOVER_STEPS)
DISTURBANCE_FREQUENCY_RAD = 2.0 * math.pi * 10.0 / CONTROL_FREQ_HZ


def _timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_config(output_root: Path, train_episodes: int, snapshot_interval: int) -> PyBulletControlExperimentConfig:
    return PyBulletControlExperimentConfig(
        control_freq_hz=CONTROL_FREQ_HZ,
        duration_sec=DURATION_SEC,
        seed=7,
        warmup_steps=64,
        train_episodes=train_episodes,
        eval_episodes=COMPARE_EPISODES,
        compare_episodes=COMPARE_EPISODES,
        updates_per_step=1,
        batch_size=128,
        snapshot_interval=snapshot_interval,
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


def _build_policy_config(
    shared_value: int,
    *,
    hidden_dim: int,
    dropout_p: float,
    tau: float,
    soft_update_interval: int,
    expl_noise_start: float,
    expl_noise_end: float,
    expl_noise_schedule: str,
    batch_size: int,
) -> MDDPGConfig:
    return MDDPGConfig(
        state_dim=6,
        action_dim=4,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        stack_size=shared_value,
        action_hold_steps=shared_value,
        tau=tau,
        soft_update_interval=soft_update_interval,
        dropout_p=dropout_p,
        expl_noise=expl_noise_start,
        expl_noise_start=expl_noise_start,
        expl_noise_end=expl_noise_end,
        expl_noise_schedule=expl_noise_schedule,
    )


def _plot_average_reward(rows: list[dict[str, float]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.8))
    plt.plot([row["shared_value"] for row in rows], [row["average_reward"] for row in rows], marker="o", linewidth=2.0)
    plt.xlabel("Shared Value v")
    plt.ylabel("Average Reward")
    plt.title("PyBullet mDDPG-LADRC Average Reward vs Shared Value")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PyBullet x-axis RL-LADRC shared-value sweep with 4D action space.")
    parser.add_argument("--episodes", type=int, default=TRAIN_EPISODES, help="Training episodes per shared value.")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--dropout-p", type=float, default=DEFAULT_DROPOUT_P)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--soft-update-interval", type=int, default=DEFAULT_SOFT_UPDATE_INTERVAL)
    parser.add_argument("--snapshot-interval", type=int, default=DEFAULT_SNAPSHOT_INTERVAL)
    parser.add_argument("--exploration-noise-start", type=float, default=DEFAULT_EXPL_NOISE_START)
    parser.add_argument("--exploration-noise-end", type=float, default=DEFAULT_EXPL_NOISE_END)
    parser.add_argument("--exploration-noise-schedule", default=DEFAULT_EXPL_NOISE_SCHEDULE)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--shared-values",
        nargs="+",
        type=int,
        default=list(SHARED_VALUES),
        help="Shared values v for mDDPG stack/action-hold/n-step.",
    )
    parser.add_argument("--tag", default=_timestamp_tag(), help="Optional output tag.")
    args = parser.parse_args()

    output_root = PROJECT_ROOT / "outputs" / "control_pybullet_rl" / "x_refline_sine_disturbance_mddpg_v_sweep" / args.tag
    ensure_dir(output_root)
    bounds = get_axis_ladrc_action_bounds("x")
    training_logs: dict[str, list[dict[str, float]]] = {}
    summary_rows: list[dict[str, float | int | str]] = []
    run_rows: list[dict[str, object]] = []

    for shared_value in tuple(int(v) for v in args.shared_values):
        config = _build_config(
            output_root / f"v_{shared_value}",
            train_episodes=int(args.episodes),
            snapshot_interval=int(args.snapshot_interval),
        )
        trainer = PyBulletAxisTrainer(config)
        result = trainer.train_axis(
            "x",
            policy_config=_build_policy_config(
                shared_value,
                hidden_dim=int(args.hidden_dim),
                dropout_p=float(args.dropout_p),
                tau=float(args.tau),
                soft_update_interval=int(args.soft_update_interval),
                expl_noise_start=float(args.exploration_noise_start),
                expl_noise_end=float(args.exploration_noise_end),
                expl_noise_schedule=str(args.exploration_noise_schedule),
                batch_size=int(args.batch_size),
            ),
            n_step=shared_value,
            shared_value=shared_value,
        )
        history = list(result["history"])
        training_logs[f"v={shared_value}"] = history
        average_reward = float(result["average_reward"])
        summary_rows.append(
            {
                "shared_value": shared_value,
                "train_episodes": int(args.episodes),
                "average_reward": average_reward,
                "best_reward": max((float(row["reward"]) for row in history), default=0.0),
                "checkpoint_path": str(result["checkpoint_path"]),
                "best_checkpoint_path": str(result["best_checkpoint_path"]),
                "output_dir": str(result["output_dir"]),
            }
        )
        run_rows.append(
            {
                "shared_value": shared_value,
                "output_dir": str(result["output_dir"]),
                "checkpoint_path": str(result["checkpoint_path"]),
                "best_checkpoint_path": str(result["best_checkpoint_path"]),
                "average_reward": average_reward,
                "history_length": len(history),
                "backend": result["backend"],
                "figures": result["figures"],
            }
        )

    figures_dir = ensure_dir(output_root / "figures")
    reward_curve_path = plot_reward_curve_collection(
        training_logs,
        figures_dir / "reward_curves_by_shared_value.svg",
        "PyBullet RL-LADRC Reward Curves",
    )
    average_reward_plot = _plot_average_reward(
        [
            {
                "shared_value": float(row["shared_value"]),
                "average_reward": float(row["average_reward"]),
            }
            for row in summary_rows
        ],
        figures_dir / "average_reward_by_shared_value.png",
    )
    write_metrics_csv(output_root / "average_reward_summary.csv", [dict(row) for row in summary_rows])
    write_summary_json(
        output_root / "summary.json",
        {
            "backend": "gym_env",
            "axis": "x",
            "seed_runs": SEED_RUNS,
            "train_episodes": int(args.episodes),
            "compare_episodes": COMPARE_EPISODES,
            "action_dim": 4,
            "shared_values": list(args.shared_values),
            "action_order": ["r", "b0", "omega_c", "k"],
            "action_bounds": {
                "r": list(bounds.r),
                "b0": list(bounds.b0),
                "omega_c": list(bounds.wc),
                "k": list(bounds.k),
            },
            "network_config": {
                "hidden_dim": int(args.hidden_dim),
                "dropout_p": float(args.dropout_p),
                "tau": float(args.tau),
                "soft_update_interval": int(args.soft_update_interval),
                "exploration_noise_schedule": str(args.exploration_noise_schedule),
                "exploration_noise_start": float(args.exploration_noise_start),
                "exploration_noise_end": float(args.exploration_noise_end),
                "snapshot_interval": int(args.snapshot_interval),
                "batch_size": int(args.batch_size),
            },
            "reference_route": {
                "fixed_stage_lengths": [FORWARD_STEPS, HOVER_STEPS, REVERSE_STEPS, FINAL_HOLD_STEPS],
                "fixed_stage_velocities": [0.5, 0.0, -0.6, 0.0],
                "control_freq_hz": CONTROL_FREQ_HZ,
                "duration_sec": DURATION_SEC,
            },
            "disturbance": {
                "axis": "x",
                "force_n": 0.03,
                "window_steps": list(DISTURBANCE_WINDOW),
                "duration_sec": HOVER_STEPS / CONTROL_FREQ_HZ,
                "cycles": 10,
                "frequency_rad_per_step": DISTURBANCE_FREQUENCY_RAD,
            },
            "artifacts": {
                "average_reward_summary_csv": str(output_root / "average_reward_summary.csv"),
                "reward_curve_figure": str(reward_curve_path),
                "average_reward_figure": str(average_reward_plot),
            },
            "runs": run_rows,
        },
    )
    write_summary_json(
        output_root / "summary_readable.json",
        {
            "backend": "gym_env",
            "message": "PyBullet four-parameter mDDPG-LADRC shared-value sweep completed.",
            "runs": run_rows,
        },
    )
    print(json.dumps({"output_root": str(output_root), "runs": run_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

