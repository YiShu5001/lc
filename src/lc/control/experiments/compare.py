from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

from lc.common.io import ensure_dir, write_json, write_metrics_csv
from lc.common.utils import seed_everything
from lc.control.RLcontrolRefLine import build_default_xy_task_config
from lc.control.configs import ControlExperimentConfig, get_axis_ladrc_action_bounds
from lc.control.controllers import PIDController
from lc.control.envs import ControlTrackingEnv
from lc.control.plotting import (
    plot_control_ablation,
    plot_control_comparison,
    plot_control_generalization,
    plot_control_mechanism_ablation,
    plot_control_training_curves,
    plot_mddpg_shared_value_sweep,
    plot_reward_curve_collection,
    plot_time_response,
)
from lc.control.trainers import ControlTrainer, save_checkpoint
from lc.envs.scenarios import build_control_scenario


def run_control_comparison(config: ControlExperimentConfig | None = None) -> dict[str, object]:
    """Run chapter-3 control comparison experiments."""
    cfg = config or ControlExperimentConfig()
    if cfg.reference_profile_mode == "rl_refline_six_phase" and tuple(cfg.axes) == ("x",):
        return _run_x_axis_rl_refline_comparison(cfg)
    return _run_default_control_comparison(cfg)


def _run_default_control_comparison(cfg: ControlExperimentConfig) -> dict[str, object]:
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
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        hidden_dim=cfg.hidden_dim,
        dropout_p=cfg.dropout_p,
        tau=cfg.tau,
        soft_update_interval=cfg.soft_update_interval,
        exploration_noise_schedule=cfg.exploration_noise_schedule,
        exploration_noise_start=cfg.exploration_noise_start,
        exploration_noise_end=cfg.exploration_noise_end,
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


def _run_x_axis_rl_refline_comparison(cfg: ControlExperimentConfig) -> dict[str, object]:
    seed_everything(cfg.seed)
    axis = "x"
    scenario = build_control_scenario(cfg.difficulty)
    task_config = build_default_xy_task_config(axis)
    episode_length = int(round(task_config.total_duration_sec * scenario.control_frequency_hz))
    trainer = ControlTrainer(
        env=ControlTrackingEnv(
            scenario=scenario,
            axis=axis,
            seed=cfg.seed,
            episode_length=episode_length,
            reference_profile_mode="rl_refline_six_phase",
        ),
        stack_size=cfg.enhanced_stack_size,
        action_hold_steps=cfg.enhanced_action_hold_steps,
        n_step=cfg.enhanced_n_step,
        warmup_steps=cfg.warmup_steps,
        batch_size=cfg.batch_size,
        updates_per_step=cfg.updates_per_step,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        hidden_dim=cfg.hidden_dim,
        dropout_p=cfg.dropout_p,
        tau=cfg.tau,
        soft_update_interval=cfg.soft_update_interval,
        refline_task_config=task_config,
        exploration_noise_schedule=cfg.exploration_noise_schedule,
        exploration_noise_start=cfg.exploration_noise_start,
        exploration_noise_end=cfg.exploration_noise_end,
    )
    ladrc_snapshots = trainer.tune_ladrc_axes((axis,), cfg.episodes)
    pid_rows: list[dict[str, object]] = []
    ladrc_rows: list[dict[str, object]] = []
    mddpg_rows_by_value: dict[int, list[dict[str, object]]] = {value: [] for value in cfg.mddpg_shared_values}
    mddpg_training_logs_by_value: dict[int, list[list[dict[str, float]]]] = {value: [] for value in cfg.mddpg_shared_values}
    mddpg_snapshots_by_value: dict[int, list[dict[int, dict[str, list[float]]]]] = {value: [] for value in cfg.mddpg_shared_values}
    representative_trajectories: dict[str, dict[str, list[float]]] = {}
    evaluation_trajectories_by_value: dict[int, dict[str, list[float]]] = {}
    checkpoints_by_value: dict[int, dict[str, object]] = {}

    for seed_index in range(cfg.seed_runs):
        seed_offset = seed_index * 100
        pid_metrics = trainer.evaluate_pid(cfg.episodes, axis=axis, seed_offset=seed_offset)
        ladrc_metrics = trainer.evaluate_ladrc(cfg.episodes, axis=axis, seed_offset=seed_offset)
        pid_rows.append({"seed": seed_index, "axis": axis, **pid_metrics})
        ladrc_rows.append({"seed": seed_index, "axis": axis, **ladrc_metrics})
        if "pid_x" not in representative_trajectories:
            representative_trajectories["pid_x"] = trainer.collect_controller_trajectory(
                PIDController(),
                axis=axis,
                seed_offset=seed_offset,
            )
            representative_trajectories["ladrc_x"] = trainer.collect_controller_trajectory(
                trainer.build_ladrc_controller(axis),
                axis=axis,
                seed_offset=seed_offset,
            )
        for shared_value in cfg.mddpg_shared_values:
            bundle = trainer.run_rl_experiment(
                cfg.train_episodes,
                cfg.compare_episodes,
                enhanced=True,
                axis=axis,
                seed_offset=seed_offset,
                stack_size_override=shared_value,
                action_hold_override=shared_value,
                n_step_override=shared_value,
                snapshot_interval=cfg.snapshot_interval,
            )
            mddpg_rows_by_value[shared_value].append({"seed": seed_index, "axis": axis, "shared_value": shared_value, **bundle["metrics"]})
            mddpg_training_logs_by_value[shared_value].append(bundle["train_history"])
            mddpg_snapshots_by_value[shared_value].append(bundle["train_snapshots"])
            evaluation_trajectories_by_value.setdefault(shared_value, bundle["trajectory"])
            checkpoints_by_value.setdefault(shared_value, bundle["checkpoint"])

    mddpg_sweep_rows = [
        {"shared_value": shared_value, **_aggregate_rows(rows)}
        for shared_value, rows in sorted(mddpg_rows_by_value.items())
    ]
    best_mddpg_value = _select_best_mddpg_variant(mddpg_sweep_rows)
    best_mddpg_metrics = _aggregate_rows(mddpg_rows_by_value[best_mddpg_value])
    representative_trajectories["mddpg_ladrc_x"] = evaluation_trajectories_by_value[best_mddpg_value]

    results = {
        "pid": _aggregate_rows(pid_rows),
        "ladrc": _aggregate_rows(ladrc_rows),
        "mddpg_ladrc": best_mddpg_metrics,
    }
    axis_results = {axis: dict(results)}

    out_dir = _prepare_output_dir(cfg.output_subdir or "x_axis_rl_refline", strict=cfg.output_subdir is not None)
    write_metrics_csv(
        out_dir / "metrics.csv",
        [{"method": name, **metrics} for name, metrics in results.items()],
    )
    write_metrics_csv(out_dir / "mddpg_shared_value_sweep.csv", mddpg_sweep_rows)
    write_metrics_csv(
        out_dir / "ladrc_tuning_snapshots.csv",
        [{"axis": axis_name, **snapshot} for axis_name, snapshot in ladrc_snapshots.items()],
    )

    refline_bounds = get_axis_ladrc_action_bounds(axis)
    task_phase_table = [
        {
            "phase": spec.kind.value,
            "duration_range_sec": list(spec.duration_range_sec),
            "reference_velocity_range": list(spec.reference_velocity_range),
            "disturbance_range": list(spec.disturbance_range),
        }
        for spec in task_config.phase_specs
    ]
    write_json(out_dir / "experiment_config.json", _serialize_config(cfg))
    write_json(out_dir / "scenario.json", _serialize_scenario(scenario))
    write_json(out_dir / "control_objective.json", _control_objective_payload(scenario, cfg))
    write_json(
        out_dir / "summary.json",
        {
            "difficulty": cfg.difficulty,
            "primary_method": cfg.primary_method,
            "axes": [axis],
            "reference_profile": "rl_refline_six_phase",
            "task_source": "lc.control.RLcontrolRefLine",
            "task_builder": "build_default_xy_task_config('x')",
            "phase_table": task_phase_table,
            "scenario": _serialize_scenario(scenario),
            "control_objective": _control_objective_payload(scenario, cfg),
            "action_space": {
                "axis": axis,
                "targets": ["b0", "wc", "k"],
                "bounds": {
                    "b0": list(refline_bounds.b0),
                    "wc": list(refline_bounds.wc),
                    "k": list(refline_bounds.k),
                },
                "fixed_r": refline_bounds.fixed_r,
                "fast_anchor": {
                    "b0": refline_bounds.fast_anchor.b0,
                    "wc": refline_bounds.fast_anchor.wc,
                    "k": refline_bounds.fast_anchor.k,
                    "r": refline_bounds.fast_anchor.r,
                    "wo": refline_bounds.fast_anchor.wo,
                },
                "steady_anchor": {
                    "b0": refline_bounds.steady_anchor.b0,
                    "wc": refline_bounds.steady_anchor.wc,
                    "k": refline_bounds.steady_anchor.k,
                    "r": refline_bounds.steady_anchor.r,
                    "wo": refline_bounds.steady_anchor.wo,
                },
            },
            "mddpg_shared_values": list(cfg.mddpg_shared_values),
            "mddpg_best_value": best_mddpg_value,
            "snapshot_interval": cfg.snapshot_interval,
            "saved_training_snapshots": list(range(cfg.snapshot_interval, cfg.train_episodes + 1, cfg.snapshot_interval)),
            "reward_log": f"training_mddpg_v{best_mddpg_value}.csv",
            "best_model": "best_model.pt",
            "exploration_schedule": {
                "type": cfg.exploration_noise_schedule,
                "start": cfg.exploration_noise_start,
                "end": cfg.exploration_noise_end,
            },
            "network_config": {
                "hidden_dim": cfg.hidden_dim,
                "dropout_p": cfg.dropout_p,
                "actor_lr": cfg.actor_lr,
                "critic_lr": cfg.critic_lr,
                "tau": cfg.tau,
                "soft_update_interval": cfg.soft_update_interval,
            },
            "results": results,
            "axis_results": axis_results,
            "mddpg_shared_value_sweep": mddpg_sweep_rows,
            "training_artifacts": {
                "mddpg_logs": {str(v): f"training_mddpg_v{v}.csv" for v in cfg.mddpg_shared_values},
                "mddpg_shared_value_sweep": "mddpg_shared_value_sweep.csv",
                "ladrc_tuning_snapshots": "ladrc_tuning_snapshots.csv",
                "mddpg_checkpoint": "checkpoints/mddpg_ladrc_best.pt",
            },
        },
    )
    checkpoints_dir = ensure_dir(out_dir / "checkpoints")
    save_checkpoint(str(checkpoints_dir / "mddpg_ladrc_best.pt"), checkpoints_by_value[best_mddpg_value])
    save_checkpoint(str(out_dir / "best_model.pt"), checkpoints_by_value[best_mddpg_value])
    write_json(out_dir / "best_mddpg_value.json", {"best_mddpg_value": best_mddpg_value})

    all_training_logs: dict[str, list[dict[str, float]]] = {}
    for shared_value in cfg.mddpg_shared_values:
        v_dir = ensure_dir(out_dir / f"v{shared_value}")
        aggregated_training = _aggregate_training_logs(mddpg_training_logs_by_value[shared_value])
        all_training_logs[f"v{shared_value}"] = aggregated_training
        write_metrics_csv(v_dir / f"training_mddpg_v{shared_value}.csv", aggregated_training)
        write_metrics_csv(out_dir / f"training_mddpg_v{shared_value}.csv", aggregated_training)
        write_json(v_dir / "eval_metrics.json", _aggregate_rows(mddpg_rows_by_value[shared_value]))
        _write_trajectory_csv(v_dir / "eval_timeseries.csv", evaluation_trajectories_by_value[shared_value])
        save_checkpoint(str(v_dir / f"checkpoint_mddpg_v{shared_value}.pt"), checkpoints_by_value[shared_value])
        for snapshot_index, snapshot in sorted(mddpg_snapshots_by_value[shared_value][0].items()):
            _write_trajectory_csv(v_dir / f"train_episode_{snapshot_index:03d}_timeseries.csv", snapshot)
        write_json(
            v_dir / "summary.json",
            {
                "shared_value": shared_value,
                "train_episodes": cfg.train_episodes,
                "compare_episodes": cfg.compare_episodes,
                "metrics": _aggregate_rows(mddpg_rows_by_value[shared_value]),
                "action_bounds": {
                    "b0": list(refline_bounds.b0),
                    "wc": list(refline_bounds.wc),
                    "k": list(refline_bounds.k),
                    "fixed_r": refline_bounds.fixed_r,
                },
            },
        )

    comparison_figures = plot_control_comparison(results, out_dir / "figures")
    training_figures = plot_control_training_curves(
        {f"mddpg_ladrc_v{best_mddpg_value}": all_training_logs[f"v{best_mddpg_value}"]},
        out_dir / "figures",
    )
    reward_all_path = plot_reward_curve_collection(all_training_logs, out_dir / "figures" / "reward_curve_all_v.svg", "mDDPG reward curves")
    response_figures = plot_time_response(representative_trajectories, out_dir / "figures")
    sweep_figures = plot_mddpg_shared_value_sweep(mddpg_sweep_rows, out_dir / "figures")
    best_response_path = out_dir / "figures" / "best_mddpg_time_response.svg"
    output_response_path = out_dir / "figures" / "time_response_output.svg"
    if output_response_path.exists():
        shutil.copyfile(output_response_path, best_response_path)
    for shared_value, logs in all_training_logs.items():
        plot_reward_curve_collection({shared_value: logs}, out_dir / "figures" / f"reward_curve_{shared_value}.svg", f"Reward curve {shared_value}")
    return {
        "results": results,
        "axis_results": axis_results,
        "ladrc_snapshots": ladrc_snapshots,
        "best_mddpg_value": best_mddpg_value,
        "mddpg_shared_value_sweep": mddpg_sweep_rows,
        "output_dir": str(out_dir),
        "figures": [str(path) for path in comparison_figures + training_figures + response_figures + sweep_figures + [reward_all_path, best_response_path]],
        "training_logs": {
            **{f"mddpg_ladrc_v{v}": str(out_dir / f"training_mddpg_v{v}.csv") for v in cfg.mddpg_shared_values},
        },
        "artifacts": {
            "experiment_config": str(out_dir / "experiment_config.json"),
            "scenario": str(out_dir / "scenario.json"),
            "control_objective": str(out_dir / "control_objective.json"),
            "ladrc_tuning_snapshots": str(out_dir / "ladrc_tuning_snapshots.csv"),
            "mddpg_shared_value_sweep": str(out_dir / "mddpg_shared_value_sweep.csv"),
        },
    }


def _prepare_output_dir(subdir: str, strict: bool = False) -> Path:
    out_dir = Path("outputs") / "control" / subdir
    if strict and out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {out_dir}")
    return ensure_dir(out_dir)


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
            mddpg_shared_values=cfg.mddpg_shared_values,
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
        "actor_lr": cfg.actor_lr,
        "critic_lr": cfg.critic_lr,
        "hidden_dim": cfg.hidden_dim,
        "dropout_p": cfg.dropout_p,
        "tau": cfg.tau,
        "soft_update_interval": cfg.soft_update_interval,
        "enhanced_stack_size": cfg.enhanced_stack_size,
        "enhanced_n_step": cfg.enhanced_n_step,
        "enhanced_action_hold_steps": cfg.enhanced_action_hold_steps,
        "mddpg_shared_values": list(cfg.mddpg_shared_values),
        "export_reference_preview": cfg.export_reference_preview,
        "reference_profile_mode": cfg.reference_profile_mode,
        "output_subdir": cfg.output_subdir,
        "snapshot_interval": cfg.snapshot_interval,
        "exploration_noise_schedule": cfg.exploration_noise_schedule,
        "exploration_noise_start": cfg.exploration_noise_start,
        "exploration_noise_end": cfg.exploration_noise_end,
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
    metrics = [key for key in rows[0].keys() if key not in {"seed", "axis", "shared_value"}]
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


def _select_best_mddpg_variant(rows: list[dict[str, object]]) -> int:
    best_row = min(
        rows,
        key=lambda row: (
            float(row["rmse"]),
            float(row["steady_state_error"]),
            -float(row["reward"]),
        ),
    )
    return int(best_row["shared_value"])


def _write_trajectory_csv(path: Path, trajectory: dict[str, list[float]]) -> None:
    keys = list(trajectory.keys())
    rows: list[dict[str, object]] = []
    length = max((len(values) for values in trajectory.values()), default=0)
    for index in range(length):
        row: dict[str, object] = {"step": index}
        for key in keys:
            values = trajectory.get(key, [])
            row[key] = values[index] if index < len(values) else ""
        rows.append(row)
    write_metrics_csv(path, rows)


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance**0.5
