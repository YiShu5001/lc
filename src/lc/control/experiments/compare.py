from __future__ import annotations

from pathlib import Path
from typing import Any

from lc.common.io import ensure_dir, write_json, write_metrics_csv
from lc.common.utils import seed_everything
from lc.control.configs import ControlExperimentConfig
from lc.control.controllers import PIDController
from lc.control.envs import ControlTrackingEnv
from lc.control.plotting import (
    plot_control_ablation,
    plot_control_comparison,
    plot_control_generalization,
    plot_control_mechanism_ablation,
    plot_control_training_curves,
    plot_time_response,
)
from lc.control.trainers import ControlTrainer, save_checkpoint
from lc.envs.scenarios import build_control_scenario


def run_control_comparison(config: ControlExperimentConfig | None = None) -> dict[str, object]:
    """Run the chapter-3 control comparison experiment with axis-wise recursive references."""
    cfg = config or ControlExperimentConfig()
    seed_everything(cfg.seed)
    scenario = build_control_scenario(cfg.difficulty)
    trainer = ControlTrainer(
        env=ControlTrackingEnv(
            scenario=scenario,
            axis=cfg.axes[0],
            seed=cfg.seed,
            episode_length=cfg.episode_length,
            reference_profile_mode=cfg.reference_profile_mode,
        ),
        stack_size=cfg.enhanced_stack_size,
        action_hold_steps=cfg.enhanced_action_hold_steps,
        n_step=cfg.enhanced_n_step,
        warmup_steps=cfg.warmup_steps,
        batch_size=cfg.batch_size,
        updates_per_step=cfg.updates_per_step,
    )
    ladrc_snapshots = trainer.tune_ladrc_axes(cfg.axes, cfg.episodes)
    pid_rows: list[dict[str, object]] = []
    ladrc_rows: list[dict[str, object]] = []
    ddpg_rows: list[dict[str, object]] = []
    mddpg_rows: list[dict[str, object]] = []
    ablation_rows: dict[str, list[dict[str, object]]] = {name: [] for name in ("no_state_stack", "no_action_hold", "no_n_step")}
    ddpg_training_logs: list[list[dict[str, float]]] = []
    mddpg_training_logs: list[list[dict[str, float]]] = []
    representative_trajectories: dict[str, dict[str, list[float]]] = {}
    ddpg_checkpoint: dict[str, object] | None = None
    mddpg_checkpoint: dict[str, object] | None = None

    for seed_index in range(cfg.seed_runs):
        for axis in cfg.axes:
            pid_metrics = trainer.evaluate_pid(cfg.episodes, axis=axis, seed_offset=seed_index * 100)
            ladrc_metrics = trainer.evaluate_ladrc(cfg.episodes, axis=axis, seed_offset=seed_index * 100)
            ddpg_bundle = trainer.run_rl_experiment(
                cfg.train_episodes,
                cfg.compare_episodes,
                enhanced=False,
                axis=axis,
                seed_offset=seed_index * 100,
            )
            mddpg_bundle = trainer.run_rl_experiment(
                cfg.train_episodes,
                cfg.compare_episodes,
                enhanced=True,
                axis=axis,
                seed_offset=seed_index * 100,
            )
            ablations = {
                "no_state_stack": trainer.run_rl_experiment(
                    cfg.train_episodes,
                    cfg.compare_episodes,
                    enhanced=True,
                    axis=axis,
                    seed_offset=seed_index * 100,
                    stack_size_override=1,
                ),
                "no_action_hold": trainer.run_rl_experiment(
                    cfg.train_episodes,
                    cfg.compare_episodes,
                    enhanced=True,
                    axis=axis,
                    seed_offset=seed_index * 100,
                    action_hold_override=1,
                ),
                "no_n_step": trainer.run_rl_experiment(
                    cfg.train_episodes,
                    cfg.compare_episodes,
                    enhanced=True,
                    axis=axis,
                    seed_offset=seed_index * 100,
                    n_step_override=1,
                ),
            }
            pid_rows.append({"seed": seed_index, "axis": axis, **pid_metrics})
            ladrc_rows.append({"seed": seed_index, "axis": axis, **ladrc_metrics})
            ddpg_rows.append({"seed": seed_index, "axis": axis, **ddpg_bundle["metrics"]})
            mddpg_rows.append({"seed": seed_index, "axis": axis, **mddpg_bundle["metrics"]})
            ddpg_training_logs.append(ddpg_bundle["train_history"])
            mddpg_training_logs.append(mddpg_bundle["train_history"])
            if axis not in representative_trajectories:
                representative_trajectories[f"pid_{axis}"] = trainer.collect_controller_trajectory(
                    PIDController(),
                    axis=axis,
                    seed_offset=seed_index * 100,
                )
                representative_trajectories[f"ladrc_{axis}"] = trainer.collect_controller_trajectory(
                    trainer.build_ladrc_controller(axis),
                    axis=axis,
                    seed_offset=seed_index * 100,
                )
                representative_trajectories[f"ddpg_ladrc_{axis}"] = ddpg_bundle["trajectory"]
                representative_trajectories[f"mddpg_ladrc_{axis}"] = mddpg_bundle["trajectory"]
            ddpg_checkpoint = ddpg_bundle["checkpoint"]
            mddpg_checkpoint = mddpg_bundle["checkpoint"]
            for name, bundle in ablations.items():
                ablation_rows[name].append({"seed": seed_index, "axis": axis, **bundle["metrics"]})

    results = {
        "pid": _aggregate_rows(pid_rows),
        "ladrc": _aggregate_rows(ladrc_rows),
        "ddpg_ladrc": _aggregate_rows(ddpg_rows),
        "mddpg_ladrc": _aggregate_rows(mddpg_rows),
    }
    ablations = {name: {"metrics": _aggregate_rows(rows)} for name, rows in ablation_rows.items()}
    axis_results = {
        axis: {
            "pid": _aggregate_rows([row for row in pid_rows if row["axis"] == axis]),
            "ladrc": _aggregate_rows([row for row in ladrc_rows if row["axis"] == axis]),
            "ddpg_ladrc": _aggregate_rows([row for row in ddpg_rows if row["axis"] == axis]),
            "mddpg_ladrc": _aggregate_rows([row for row in mddpg_rows if row["axis"] == axis]),
        }
        for axis in cfg.axes
    }
    out_dir = ensure_dir(Path("outputs") / "control" / cfg.difficulty)
    training_logs = {
        "ddpg_ladrc": _aggregate_training_logs(ddpg_training_logs),
        "mddpg_ladrc": _aggregate_training_logs(mddpg_training_logs),
    }
    write_metrics_csv(out_dir / "training_ddpg.csv", training_logs["ddpg_ladrc"])
    write_metrics_csv(out_dir / "training_mddpg.csv", training_logs["mddpg_ladrc"])
    write_metrics_csv(
        out_dir / "ablation_metrics.csv",
        [{"variant": name, **bundle["metrics"]} for name, bundle in ablations.items()],
    )
    write_metrics_csv(
        out_dir / "metrics.csv",
        [{"method": name, **metrics} for name, metrics in results.items()],
    )
    write_metrics_csv(out_dir / "seed_metrics.csv", _seed_metric_rows(pid_rows, ladrc_rows, ddpg_rows, mddpg_rows))
    write_metrics_csv(
        out_dir / "ladrc_tuning_snapshots.csv",
        [{"axis": axis, **snapshot} for axis, snapshot in ladrc_snapshots.items()],
    )
    write_json(out_dir / "experiment_config.json", _serialize_config(cfg))
    write_json(out_dir / "scenario.json", _serialize_scenario(scenario))
    write_json(out_dir / "control_objective.json", _control_objective_payload(scenario, cfg))
    write_json(
        out_dir / "summary.json",
        {
            "difficulty": cfg.difficulty,
            "primary_method": cfg.primary_method,
            "axes": list(cfg.axes),
            "scenario": _serialize_scenario(scenario),
            "control_objective": _control_objective_payload(scenario, cfg),
            "method_roles": {
                "pid": "classical_baseline",
                "ladrc": "robust_control_baseline",
                "ddpg_ladrc": "primary_rl_method",
                "mddpg_ladrc": "enhanced_rl_method",
            },
            "training_artifacts": {
                "ddpg_log": "training_ddpg.csv",
                "mddpg_log": "training_mddpg.csv",
                "ddpg_checkpoint": "checkpoints/ddpg_ladrc.pt",
                "mddpg_checkpoint": "checkpoints/mddpg_ladrc.pt",
                "ladrc_tuning_snapshots": "ladrc_tuning_snapshots.csv",
                "ablation_metrics": "ablation_metrics.csv",
                "seed_metrics": "seed_metrics.csv",
            },
            "ladrc_baseline": {
                "parameterization": ["b0", "omega_c", "k"],
                "snapshots": ladrc_snapshots,
            },
            "axis_results": axis_results,
            "results": results,
            "ablations": {name: bundle["metrics"] for name, bundle in ablations.items()},
        },
    )
    checkpoints_dir = ensure_dir(out_dir / "checkpoints")
    save_checkpoint(str(checkpoints_dir / "ddpg_ladrc.pt"), ddpg_checkpoint or {})
    save_checkpoint(str(checkpoints_dir / "mddpg_ladrc.pt"), mddpg_checkpoint or {})
    figures = plot_control_comparison(results, out_dir / "figures")
    curve_figures = plot_control_training_curves(training_logs, out_dir / "figures")
    ablation = plot_control_ablation(results, out_dir / "figures")
    mechanism_ablation = plot_control_mechanism_ablation(
        {name: bundle["metrics"] for name, bundle in ablations.items()},
        out_dir / "figures",
    )
    response_figures = plot_time_response(representative_trajectories, out_dir / "figures")
    return {
        "results": results,
        "axis_results": axis_results,
        "ladrc_snapshots": ladrc_snapshots,
        "output_dir": str(out_dir),
        "figures": [str(p) for p in figures + curve_figures + response_figures + [ablation, mechanism_ablation]],
        "training_logs": {name: str(out_dir / f"training_{name.split('_')[0]}.csv") for name in training_logs},
        "artifacts": {
            "experiment_config": str(out_dir / "experiment_config.json"),
            "scenario": str(out_dir / "scenario.json"),
            "control_objective": str(out_dir / "control_objective.json"),
            "ladrc_tuning_snapshots": str(out_dir / "ladrc_tuning_snapshots.csv"),
            "seed_metrics": str(out_dir / "seed_metrics.csv"),
        },
    }


def run_control_generalization(config: ControlExperimentConfig | None = None) -> dict[str, object]:
    """Run chapter-3 generalization experiments across configured difficulties."""
    cfg = config or ControlExperimentConfig()
    generalization_results: dict[str, dict[str, dict[str, float]]] = {}
    for offset, difficulty in enumerate(cfg.difficulty_levels):
        scoped_cfg = ControlExperimentConfig(
            primary_method=cfg.primary_method,
            difficulty=difficulty,
            difficulty_levels=cfg.difficulty_levels,
            axes=cfg.axes,
            episode_length=cfg.episode_length,
            episodes=cfg.episodes,
            seed=cfg.seed + offset,
            seed_runs=cfg.seed_runs,
            compare_episodes=cfg.compare_episodes,
            train_episodes=cfg.train_episodes,
            warmup_steps=cfg.warmup_steps,
            batch_size=cfg.batch_size,
            updates_per_step=cfg.updates_per_step,
            enhanced_stack_size=cfg.enhanced_stack_size,
            enhanced_n_step=cfg.enhanced_n_step,
            enhanced_action_hold_steps=cfg.enhanced_action_hold_steps,
            export_reference_preview=cfg.export_reference_preview,
            reference_profile_mode=cfg.reference_profile_mode,
        )
        bundle = run_control_comparison(scoped_cfg)
        generalization_results[difficulty] = bundle["results"]
    out_dir = ensure_dir(Path("outputs") / "control" / "generalization")
    write_metrics_csv(
        out_dir / "metrics.csv",
        [
            {"difficulty": difficulty, "method": method, **metrics}
            for difficulty, methods in generalization_results.items()
            for method, metrics in methods.items()
        ],
    )
    write_json(out_dir / "experiment_config.json", _serialize_config(cfg))
    write_json(
        out_dir / "summary.json",
        {
            "difficulty_levels": list(cfg.difficulty_levels),
            "methods": ["pid", "ladrc", "ddpg_ladrc", "mddpg_ladrc"],
            "axes": list(cfg.axes),
            "results": generalization_results,
        },
    )
    figures = plot_control_generalization(generalization_results, out_dir / "figures")
    return {
        "output_dir": str(out_dir),
        "results": generalization_results,
        "figures": [str(path) for path in figures],
    }


def _serialize_config(cfg: ControlExperimentConfig) -> dict[str, object]:
    return {
        "primary_method": cfg.primary_method,
        "difficulty": cfg.difficulty,
        "difficulty_levels": list(cfg.difficulty_levels),
        "axes": list(cfg.axes),
        "episode_length": cfg.episode_length,
        "episodes": cfg.episodes,
        "seed": cfg.seed,
        "seed_runs": cfg.seed_runs,
        "compare_episodes": cfg.compare_episodes,
        "train_episodes": cfg.train_episodes,
        "warmup_steps": cfg.warmup_steps,
        "batch_size": cfg.batch_size,
        "updates_per_step": cfg.updates_per_step,
        "enhanced_stack_size": cfg.enhanced_stack_size,
        "enhanced_n_step": cfg.enhanced_n_step,
        "enhanced_action_hold_steps": cfg.enhanced_action_hold_steps,
        "export_reference_preview": cfg.export_reference_preview,
        "reference_profile_mode": cfg.reference_profile_mode,
    }


def _serialize_scenario(scenario: Any) -> dict[str, object]:
    return {
        "difficulty": scenario.difficulty,
        "target_motion": scenario.target_motion,
        "disturbance_level": scenario.disturbance_level,
        "control_frequency_hz": scenario.control_frequency_hz,
        "rl_frequency_hz": scenario.rl_frequency_hz,
        "num_obstacles": scenario.num_obstacles,
        "dynamic_obstacles": scenario.dynamic_obstacles,
        "obstacle_layout": scenario.obstacle_layout,
        "world_scale": scenario.world_scale,
        "density": scenario.density,
    }


def _control_objective_payload(scenario: Any, cfg: ControlExperimentConfig) -> dict[str, object]:
    return {
        "task": "axiswise_recursive_position_tracking_with_disturbance_rejection",
        "controller": "LADRC",
        "rl_role": "online_parameter_tuning",
        "rl_action_targets": ["b0", "omega_c", "k"],
        "derived_parameter_relation": "omega_o = k * omega_c",
        "reward_definition": "-abs(position_error) - 0.15 * abs(velocity_error)",
        "observation_channels": [
            "position_error",
            "velocity_error",
            "current_position",
            "current_velocity",
            "reference_position",
            "reference_velocity",
            "disturbance_proxy",
            "normalized_time",
        ],
        "showcase_targets": [
            "reference_tracking_accuracy",
            "disturbance_rejection",
            "recovery_after_disturbance",
            "control_smoothness",
            "parameter_adaptation",
        ],
        "reference_profile": cfg.reference_profile_mode,
        "axes": list(cfg.axes),
    }


def _aggregate_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    metrics = [key for key in rows[0].keys() if key not in {"seed", "axis"}]
    aggregated: dict[str, float] = {}
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        aggregated[metric] = float(sum(values) / len(values))
        aggregated[f"{metric}_std"] = _std(values)
    return aggregated


def _aggregate_training_logs(logs: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    if not logs:
        return []
    episode_count = min(len(log) for log in logs)
    aggregated: list[dict[str, float]] = []
    for episode_index in range(episode_count):
        row: dict[str, float] = {"episode": float(episode_index)}
        metric_keys = [key for key in logs[0][episode_index].keys() if key != "episode"]
        for key in metric_keys:
            values = [float(log[episode_index][key]) for log in logs]
            row[key] = float(sum(values) / len(values))
            row[f"{key}_std"] = _std(values)
        aggregated.append(row)
    return aggregated


def _seed_metric_rows(*method_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    method_names = ("pid", "ladrc", "ddpg_ladrc", "mddpg_ladrc")
    rows: list[dict[str, object]] = []
    for method_name, entries in zip(method_names, method_rows):
        rows.extend({"method": method_name, **entry} for entry in entries)
    return rows


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance**0.5
