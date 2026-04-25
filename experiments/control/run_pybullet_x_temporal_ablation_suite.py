from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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

from lc.common.io import ensure_dir
from lc.control.configs import ArtifactConfig, AxisTrainingConfig, PyBulletControlExperimentConfig
from lc.control.io import write_metrics_csv, write_summary_json
from lc.control.trainers import PyBulletAxisTrainer
from lc.rl.algorithms import MDDPGConfig


CONTROL_FREQ_HZ = 48
FORWARD_STEPS = 96
HOVER_STEPS = 48
REVERSE_STEPS = 96
FINAL_HOLD_STEPS = 24
HOVER_GAP_STEPS = 10
DISTURBANCE_STEPS = 28
STEP_COUNT = FORWARD_STEPS + HOVER_STEPS + REVERSE_STEPS + FINAL_HOLD_STEPS
DURATION_SEC = STEP_COUNT / CONTROL_FREQ_HZ

DEFAULT_EPISODES = 400
DEFAULT_EVAL_SEEDS = (7, 17, 27)
DEFAULT_FAMILIES = (
    "single_action_hold",
    "single_state_stack",
    "single_n_step",
    "pair_action_hold_state_stack",
    "pair_action_hold_n_step",
    "pair_state_stack_n_step",
    "full_temporal",
)

FAMILY_LABELS = {
    "single_action_hold": "Action Hold Only",
    "single_state_stack": "State Stack Only",
    "single_n_step": "N-Step Only",
    "pair_action_hold_state_stack": "Action Hold + State Stack",
    "pair_action_hold_n_step": "Action Hold + N-Step",
    "pair_state_stack_n_step": "State Stack + N-Step",
    "full_temporal": "Full Temporal",
}


@dataclass(frozen=True)
class AblationRun:
    family: str
    k: int
    action_hold_steps: int
    stack_size: int
    n_step: int


def _timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_runs(selected_families: set[str]) -> list[AblationRun]:
    runs: list[AblationRun] = []
    for k in range(1, 6):
        if "single_action_hold" in selected_families:
            runs.append(AblationRun("single_action_hold", k, k, 1, 1))
        if "single_state_stack" in selected_families:
            runs.append(AblationRun("single_state_stack", k, 1, k, 1))
        if "single_n_step" in selected_families:
            runs.append(AblationRun("single_n_step", k, 1, 1, k))
        if "pair_action_hold_state_stack" in selected_families:
            runs.append(AblationRun("pair_action_hold_state_stack", k, k, k, 1))
        if "pair_action_hold_n_step" in selected_families:
            runs.append(AblationRun("pair_action_hold_n_step", k, k, 1, k))
        if "pair_state_stack_n_step" in selected_families:
            runs.append(AblationRun("pair_state_stack_n_step", k, 1, k, k))
        if "full_temporal" in selected_families:
            runs.append(AblationRun("full_temporal", k, k, k, k))
    return runs


def _build_config(output_root: Path, episodes: int, eval_seeds: tuple[int, ...]) -> PyBulletControlExperimentConfig:
    return PyBulletControlExperimentConfig(
        control_freq_hz=CONTROL_FREQ_HZ,
        duration_sec=DURATION_SEC,
        seed=7,
        warmup_steps=64,
        train_episodes=episodes,
        eval_episodes=len(eval_seeds),
        compare_episodes=len(eval_seeds),
        eval_seeds=eval_seeds,
        updates_per_step=1,
        batch_size=128,
        snapshot_interval=20,
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
                disturbance_scale=0.004,
                disturbance_axis_bias=1.0,
                disturbance_mode="random_uniform",
                disturbance_step_window=(FORWARD_STEPS + HOVER_GAP_STEPS, FORWARD_STEPS + HOVER_GAP_STEPS + DISTURBANCE_STEPS),
                fixed_stage_lengths=(FORWARD_STEPS, HOVER_STEPS, REVERSE_STEPS, FINAL_HOLD_STEPS),
                fixed_stage_velocities=(0.5, 0.0, -0.6, 0.0),
            ),
        ),
    )


def _build_policy_config(run: AblationRun) -> MDDPGConfig:
    return MDDPGConfig(
        state_dim=6,
        action_dim=4,
        hidden_dim=768,
        batch_size=128,
        stack_size=run.stack_size,
        action_hold_steps=run.action_hold_steps,
        gamma=0.95,
        tau=0.02,
        soft_update_interval=10,
        dropout_p=0.25,
        expl_noise=0.2,
        expl_noise_start=0.2,
        expl_noise_end=0.02,
        expl_noise_schedule="three_phase",
    )


def _plot_family_reward_curves(history_logs: dict[str, list[dict[str, float]]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[tuple[str, list[dict[str, float]]]]] = defaultdict(list)
    for key, rows in history_logs.items():
        family, k = key.split(":")
        grouped[family].append((k, rows))
    fig, axes = plt.subplots(len(grouped), 1, figsize=(11, 3.8 * max(len(grouped), 1)), sharex=False)
    if len(grouped) == 1:
        axes = [axes]
    for ax, (family, runs) in zip(axes, grouped.items()):
        for k, rows in sorted(runs, key=lambda item: int(item[0])):
            episodes = [int(float(row["episode"])) for row in rows]
            avg_rewards = [float(row["average_reward"]) for row in rows]
            ax.plot(episodes, avg_rewards, linewidth=2, label=f"k={k}")
        ax.set_title(f"{family} 骞冲潎Reward鏇茬嚎")
        ax.set_xlabel("Episode")
        ax.set_ylabel("????")
        ax.legend(ncol=3, fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_family_eval_curves(eval_logs: dict[str, list[dict[str, float]]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[tuple[str, list[dict[str, float]]]]] = defaultdict(list)
    for key, rows in eval_logs.items():
        family, k = key.split(":")
        grouped[family].append((k, rows))
    fig, axes = plt.subplots(len(grouped), 1, figsize=(11, 3.8 * max(len(grouped), 1)), sharex=False)
    if len(grouped) == 1:
        axes = [axes]
    for ax, (family, runs) in zip(axes, grouped.items()):
        for k, rows in sorted(runs, key=lambda item: int(item[0])):
            episodes = [int(float(row["episode"])) for row in rows]
            rmses = [float(row["rmse"]) for row in rows]
            ax.plot(episodes, rmses, marker="o", linewidth=2, label=f"k={k}")
        ax.set_title(f"{family} Eval RMSE鏇茬嚎")
        ax.set_xlabel("Episode")
        ax.set_ylabel("??RMSE")
        ax.legend(ncol=3, fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_heatmap(rows: list[dict[str, float]], value_key: str, title: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    families = []
    ks = sorted({int(row["k"]) for row in rows})
    for row in rows:
        if row["family"] not in families:
            families.append(row["family"])
    matrix = []
    for family in families:
        family_rows = {int(row["k"]): float(row[value_key]) for row in rows if row["family"] == family}
        matrix.append([family_rows.get(k, float("nan")) for k in ks])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels([FAMILY_LABELS.get(family, family) for family in families])
    ax.set_title(title)
    for i, family_values in enumerate(matrix):
        for j, value in enumerate(family_values):
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run temporal enhancement ablation suite for chapter 3.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--eval-seeds", default="7,17,27")
    parser.add_argument("--output-tag", default=f"{_timestamp_tag()}_temporal_ablation")
    parser.add_argument("--families", nargs="*", default=list(DEFAULT_FAMILIES))
    args = parser.parse_args()

    eval_seeds = tuple(int(part) for part in str(args.eval_seeds).split(",") if part.strip())
    selected_families = set(args.families)
    runs = _build_runs(selected_families)

    output_root = PROJECT_ROOT / "outputs" / "control_pybullet_rl" / "x_temporal_ablation_suite" / args.output_tag
    ensure_dir(output_root)

    summary_rows: list[dict[str, object]] = []
    history_logs: dict[str, list[dict[str, float]]] = {}
    eval_logs: dict[str, list[dict[str, float]]] = {}

    for run in runs:
        family_root = output_root / run.family / f"k_{run.k}"
        config = _build_config(family_root, int(args.episodes), eval_seeds)
        trainer = PyBulletAxisTrainer(config)
        result = trainer.train_axis(
            "x",
            policy_config=_build_policy_config(run),
            n_step=run.n_step,
            shared_value=run.k,
        )
        key = f"{run.family}:{run.k}"
        history_logs[key] = list(result["history"])
        eval_logs[key] = list(result.get("eval_history") or [])
        best_eval_metrics = dict(result.get("best_eval_metrics") or {})
        summary_rows.append(
            {
                "family": run.family,
                "k": run.k,
                "action_hold_steps": run.action_hold_steps,
                "stack_size": run.stack_size,
                "n_step": run.n_step,
                "average_reward": float(result["average_reward"]),
                "best_reward": max((float(row["reward"]) for row in result["history"]), default=0.0),
                "best_eval_rmse": float(best_eval_metrics.get("rmse", 0.0)),
                "best_eval_mae": float(best_eval_metrics.get("mae", 0.0)),
                "best_eval_velocity_rmse": float(best_eval_metrics.get("velocity_rmse", 0.0)),
                "best_eval_score": float(best_eval_metrics.get("score", 0.0)),
                "best_checkpoint_path": str(result["best_checkpoint_path"]),
                "output_dir": str(result["output_dir"]),
            }
        )

    full_reward_by_k = {
        int(row["k"]): float(row["average_reward"])
        for row in summary_rows
        if row["family"] == "full_temporal"
    }
    loss_rows: list[dict[str, object]] = []
    for row in summary_rows:
        baseline = full_reward_by_k.get(int(row["k"]))
        loss_rate = 0.0
        if baseline is not None and abs(baseline) > 1e-9:
            loss_rate = (baseline - float(row["average_reward"])) / abs(baseline)
        row["reward_loss_rate_vs_full"] = float(loss_rate)
        loss_rows.append(
            {
                "family": row["family"],
                "k": row["k"],
                "average_reward": row["average_reward"],
                "full_average_reward": baseline,
                "reward_loss_rate_vs_full": float(loss_rate),
            }
        )

    figures_dir = ensure_dir(output_root / "figures")
    reward_fig = _plot_family_reward_curves(history_logs, figures_dir / "reward_curves_by_family.svg")
    eval_fig = _plot_family_eval_curves(eval_logs, figures_dir / "eval_rmse_curves_by_family.svg")
    loss_heatmap = _plot_heatmap(
        [{k: row[k] for k in ("family", "k", "reward_loss_rate_vs_full")} for row in summary_rows],
        "reward_loss_rate_vs_full",
        "Reward Loss Rate vs Full Temporal Baseline",
        figures_dir / "ablation_heatmap_reward_loss.png",
    )
    rmse_heatmap = _plot_heatmap(
        [{k: row[k] for k in ("family", "k", "best_eval_rmse")} for row in summary_rows],
        "best_eval_rmse",
        "Best Eval RMSE by Ablation Family",
        figures_dir / "ablation_heatmap_rmse.png",
    )

    write_metrics_csv(output_root / "temporal_ablation_summary.csv", summary_rows)
    write_metrics_csv(output_root / "temporal_ablation_loss_rate.csv", loss_rows)
    write_summary_json(
        output_root / "summary.json",
        {
            "episodes": int(args.episodes),
            "eval_seeds": list(eval_seeds),
            "families": sorted(selected_families),
            "environment": {
                "backend": "gym_env",
                "disturbance_mode": "random_uniform",
                "disturbance_scale": 0.004,
                "disturbance_window_steps": [FORWARD_STEPS + HOVER_GAP_STEPS, FORWARD_STEPS + HOVER_GAP_STEPS + DISTURBANCE_STEPS],
                "reward": "-|pos_error|",
                "action_bounds_source": "current credible narrow bounds",
                "exploration_noise": {"start": 0.2, "mid": 0.1, "end": 0.02, "schedule": "three_phase"},
            },
            "n_step_bootstrap_fixed": True,
            "summary_rows": summary_rows,
            "figures": [str(reward_fig), str(eval_fig), str(loss_heatmap), str(rmse_heatmap)],
        },
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "run_count": len(summary_rows),
                "figures": [str(reward_fig), str(eval_fig), str(loss_heatmap), str(rmse_heatmap)],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

