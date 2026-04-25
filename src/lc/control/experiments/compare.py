from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any
from collections import deque

import numpy as np
import torch

from lc.common.io import ensure_dir, write_json, write_metrics_csv
from lc.common.utils import seed_everything
from lc.control.RLcontrolRefLine import build_default_xy_task_config, build_refline_episode
from lc.control.configs import AxisTransferExperimentConfig, ControlExperimentConfig, get_axis_ladrc_action_bounds
from lc.control.controllers import AdaptiveLADRCController, LADRCController, PIDController
from lc.control.envs import ControlTrackingEnv
from lc.control.policies import ControlLADRLAgent, stack_state
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
from lc.envs.metrics import compute_control_metrics
from lc.envs.scenarios import build_control_scenario


def run_control_comparison(config: ControlExperimentConfig | None = None) -> dict[str, object]:
    """Run chapter-3 control comparison experiments."""
    cfg = config or ControlExperimentConfig()
    if cfg.reference_profile_mode == "rl_refline_six_phase" and len(cfg.axes) == 1 and cfg.axes[0] in {"x", "y"}:
        return _run_single_axis_rl_refline_comparison(cfg, cfg.axes[0])
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


def _run_single_axis_rl_refline_comparison(cfg: ControlExperimentConfig, axis: str) -> dict[str, object]:
    seed_everything(cfg.seed)
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
            "task_builder": f"build_default_xy_task_config('{axis}')",
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


def run_y_axis_refline_transfer_suite(
    config: AxisTransferExperimentConfig | None = None,
) -> dict[str, object]:
    cfg = config or AxisTransferExperimentConfig()
    seed_everything(cfg.seed)
    source_dir = _resolve_transfer_source_dir(cfg)
    source_ladrc = _load_transfer_ladrc_params(source_dir, cfg.source_axis)
    shared_values = tuple(dict.fromkeys((*cfg.compare_shared_values, cfg.reference_shared_value)))
    scenario = build_control_scenario(cfg.difficulty)
    suite_out = _prepare_output_dir(
        cfg.output_subdir or "y_axis_refline__exp-xpid-y-transfer-suite",
        strict=cfg.output_subdir is not None,
    )

    baseline = _run_coupled_pid_baseline(cfg, scenario, suite_out / "baseline")
    ladrc = _run_coupled_ladrc_transfer(cfg, scenario, source_ladrc, suite_out / "ladrc_transfer", source_dir)
    rl_runs: dict[str, dict[str, object]] = {}
    for shared_value in shared_values:
        checkpoint = _load_transfer_checkpoint(source_dir, shared_value)
        if checkpoint is None:
            continue
        zero_shot = _run_coupled_rl_transfer(
            cfg,
            scenario,
            suite_out / f"rl_transfer_zero_shot_v{shared_value}",
            checkpoint,
            shared_value=shared_value,
            warm_start=False,
            source_dir=source_dir,
        )
        warm_start = _run_coupled_rl_transfer(
            cfg,
            scenario,
            suite_out / f"rl_transfer_warm_start_v{shared_value}",
            checkpoint,
            shared_value=shared_value,
            warm_start=True,
            source_dir=source_dir,
        )
        rl_runs[f"zero_shot_v{shared_value}"] = zero_shot
        rl_runs[f"warm_start_v{shared_value}"] = warm_start

    suite_metrics = [
        {"method": "y_pid_baseline", **baseline["metrics"]},
        {"method": "y_ladrc_transferred", **ladrc["metrics"]},
    ]
    suite_metrics.extend({"method": name, **bundle["metrics"]} for name, bundle in rl_runs.items())
    write_metrics_csv(suite_out / "metrics.csv", suite_metrics)
    write_json(
        suite_out / "summary.json",
        {
            "x_controller": "pid",
            "y_controller_groups": ["pid", "ladrc_fixed", "rl_policy"],
            "source_axis": cfg.source_axis,
            "target_axis": cfg.target_axis,
            "source_dir": str(source_dir),
            "observation_semantics": "y, vy, roll, roll_rate",
            "baseline": baseline["metrics"],
            "ladrc_transfer": ladrc["metrics"],
            "rl_transfer": {name: bundle["metrics"] for name, bundle in rl_runs.items()},
        },
    )
    figures = plot_control_comparison(
        {row["method"]: {key: value for key, value in row.items() if key != "method"} for row in suite_metrics},
        suite_out / "figures",
    )
    return {
        "output_dir": str(suite_out),
        "source_dir": str(source_dir),
        "results": {
            "baseline": baseline["metrics"],
            "ladrc_transfer": ladrc["metrics"],
            **{name: bundle["metrics"] for name, bundle in rl_runs.items()},
        },
        "figures": [str(path) for path in figures],
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


def _resolve_transfer_source_dir(cfg: AxisTransferExperimentConfig) -> Path:
    for root in cfg.source_output_roots:
        root_path = Path(root)
        for subdir in cfg.source_experiment_subdirs:
            candidate = root_path / subdir
            if candidate.exists():
                return candidate
    raise FileNotFoundError("No source x-axis refline artifact directory found for y-axis transfer.")


def _load_transfer_ladrc_params(source_dir: Path, source_axis: str) -> dict[str, float]:
    snapshot_csv = source_dir / "ladrc_tuning_snapshots.csv"
    if snapshot_csv.exists():
        lines = snapshot_csv.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) >= 2:
            header = lines[0].split(",")
            values = lines[1].split(",")
            row = dict(zip(header, values))
            return {
                "r": 10.0,
                "b0": float(row.get("b0", 30.5)),
                "omega_c": float(row.get("omega_c", 1.5)),
                "k": float(row.get("k", 11.0)),
                "source_axis": source_axis,
            }
    return {"r": 10.0, "b0": 30.5, "omega_c": 1.5, "k": 11.0, "source_axis": source_axis}


def _load_transfer_checkpoint(source_dir: Path, shared_value: int) -> dict[str, object] | None:
    candidates = [
        source_dir / f"v{shared_value}" / f"checkpoint_mddpg_v{shared_value}.pt",
        source_dir / "checkpoints" / "mddpg_ladrc_best.pt",
        source_dir / "best_model.pt",
    ]
    for path in candidates:
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=False)
    return None


def _build_refline_env(scenario: Any, axis: str, seed: int, coupling_gain: float) -> tuple[ControlTrackingEnv, object]:
    task_config = build_default_xy_task_config(axis)
    bundle = build_refline_episode(task_config, seed=seed)
    env = ControlTrackingEnv(
        scenario=scenario,
        axis=axis,
        seed=seed,
        episode_length=len(bundle.time),
        reference_profile_mode="rl_refline_six_phase",
        cross_axis_coupling_gain=coupling_gain,
    )
    return env, bundle


def _episode_metrics_from_env(env: ControlTrackingEnv, total_reward: float) -> dict[str, float]:
    metrics = compute_control_metrics(env.errors, env.controls)
    if env.velocity_errors:
        metrics["velocity_rmse"] = float(np.sqrt(np.mean(np.asarray(env.velocity_errors, dtype=float) ** 2)))
    else:
        metrics["velocity_rmse"] = 0.0
    metrics["reward"] = float(total_reward)
    return metrics


def _trajectory_from_env(env: ControlTrackingEnv) -> dict[str, list[float]]:
    return {
        "reference": list(env.references),
        "reference_velocity": list(env.reference_velocities),
        "error": list(env.errors),
        "velocity_error": list(env.velocity_errors),
        "output": list(env.outputs),
        "control": list(env.controls),
        "disturbance": list(env.disturbances),
        "pitch": list(env.pitches),
        "pitch_rate": list(env.pitch_rates),
        "roll": list(env.rolls),
        "roll_rate": list(env.roll_rates),
        "external_coupling": list(env.external_couplings),
    }


def _aggregate_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return _average_float_rows(rows)


def _average_float_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = list(rows[0].keys())
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def _run_coupled_controller_episode(
    scenario: Any,
    seed: int,
    coupling_gain: float,
    y_controller: PIDController | LADRCController,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    x_env, x_bundle = _build_refline_env(scenario, "x", seed, coupling_gain)
    y_env, y_bundle = _build_refline_env(scenario, "y", seed + 17, coupling_gain)
    x_env.reset(axis="x", seed=seed, external_episode_bundle=x_bundle)
    y_env.reset(axis="y", seed=seed + 17, external_episode_bundle=y_bundle)
    x_controller = PIDController()
    x_controller.reset()
    y_controller.reset()
    total_reward = 0.0
    done = False
    dt = 1.0 / scenario.control_frequency_hz
    while not done:
        x_control = x_controller.step(x_env.reference, x_env.state_axis, dt)
        _, _, x_done, _ = x_env.step(x_control)
        coupling = float(x_env.pitch)
        y_control = y_controller.step(y_env.reference, y_env.state_axis, dt)
        _, reward, done, _ = y_env.step(y_control, external_coupling=coupling)
        total_reward += reward
        done = done or x_done
    return _episode_metrics_from_env(y_env, total_reward), _trajectory_from_env(y_env)


def _build_transfer_agent(
    cfg: AxisTransferExperimentConfig,
    shared_value: int,
    checkpoint: dict[str, object] | None = None,
) -> ControlLADRLAgent:
    hidden_dim = cfg.hidden_dim
    actor_state = checkpoint.get("actor") if checkpoint else None
    if isinstance(actor_state, dict) and "net.0.weight" in actor_state:
        hidden_dim = int(actor_state["net.0.weight"].shape[0])
    agent = ControlLADRLAgent(
        obs_dim=4,
        stack_size=shared_value,
        action_hold_steps=shared_value,
        n_step=shared_value,
        batch_size=cfg.batch_size,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        hidden_dim=hidden_dim,
        dropout_p=cfg.dropout_p,
        tau=cfg.tau,
        soft_update_interval=cfg.soft_update_interval,
        exploration_noise_schedule=cfg.exploration_noise_schedule,
        exploration_noise_start=cfg.exploration_noise_start,
        exploration_noise_end=cfg.exploration_noise_end,
        controller=AdaptiveLADRCController.for_axis(cfg.target_axis),
    )
    if checkpoint:
        actor_state = checkpoint.get("actor")
        critic_state = checkpoint.get("critic")
        if actor_state is not None:
            agent.policy.actor.load_state_dict(actor_state)
            agent.policy.actor_target.load_state_dict(checkpoint.get("actor_target", actor_state))
        if critic_state is not None:
            agent.policy.critic.load_state_dict(critic_state)
            agent.policy.critic_target.load_state_dict(checkpoint.get("critic_target", critic_state))
        if checkpoint.get("normalizer") is not None:
            agent.policy._normalizer = np.asarray(checkpoint["normalizer"], dtype=np.float32)
        if checkpoint.get("ladrc_baseline"):
            baseline = checkpoint["ladrc_baseline"]
            agent.controller.base.set_parameters(
                r=float(baseline.get("r", agent.controller.base.r)),
                b0=float(baseline.get("b0", agent.controller.base.b0)),
                omega_c=float(baseline.get("omega_c", agent.controller.base.omega_c)),
                k=float(baseline.get("k", agent.controller.base.k)),
            )
    return agent


def _run_coupled_rl_eval(
    cfg: AxisTransferExperimentConfig,
    scenario: Any,
    agent: ControlLADRLAgent,
    episodes: int,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    rows: list[dict[str, float]] = []
    representative: dict[str, list[float]] | None = None
    dt = 1.0 / scenario.control_frequency_hz
    for episode in range(episodes):
        seed = cfg.seed + 1000 + episode
        x_env, x_bundle = _build_refline_env(scenario, "x", seed, cfg.coupling_gain)
        y_env, y_bundle = _build_refline_env(scenario, "y", seed + 17, cfg.coupling_gain)
        x_env.reset(axis="x", seed=seed, external_episode_bundle=x_bundle)
        obs = y_env.reset(axis="y", seed=seed + 17, external_episode_bundle=y_bundle)
        x_controller = PIDController()
        x_controller.reset()
        agent.reset()
        history = [obs.copy()]
        total_reward = 0.0
        done = False
        while not done:
            x_control = x_controller.step(x_env.reference, x_env.state_axis, dt)
            _, _, x_done, _ = x_env.step(x_control)
            stacked = stack_state(history, obs, agent.policy.config.stack_size)
            action = agent.act(stacked, explore=False)
            agent.controller.adapt(action)
            y_control = agent.controller.step(y_env.reference, y_env.state_axis, dt)
            obs, reward, done, _ = y_env.step(y_control, external_coupling=float(x_env.pitch))
            history.append(obs.copy())
            total_reward += reward
            done = done or x_done
        rows.append(_episode_metrics_from_env(y_env, total_reward))
        if representative is None:
            representative = _trajectory_from_env(y_env)
    return _aggregate_metric_rows(rows), representative or {}


def _flush_n_step_rollout(
    agent: ControlLADRLAgent,
    rollout: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]],
    n_step: int,
    gamma: float,
    force: bool = False,
) -> None:
    while rollout and (force or len(rollout) >= n_step):
        reward = 0.0
        next_state = rollout[0][3]
        done = rollout[0][4]
        for index, (_, _, step_reward, step_next_state, step_done) in enumerate(rollout):
            if index >= n_step:
                break
            reward += (gamma**index) * step_reward
            next_state = step_next_state
            done = step_done
            if step_done:
                break
        state, action, _, _, _ = rollout.popleft()
        agent.policy.store_transition(state, action, reward, next_state, done)
        if not force:
            break


def _run_coupled_rl_warm_start_training(
    cfg: AxisTransferExperimentConfig,
    scenario: Any,
    agent: ControlLADRLAgent,
) -> list[dict[str, float]]:
    dt = 1.0 / scenario.control_frequency_hz
    total_steps = 0
    history_rows: list[dict[str, float]] = []
    gamma = agent.policy.config.gamma
    for episode in range(cfg.warm_start_episodes):
        seed = cfg.seed + episode
        x_env, x_bundle = _build_refline_env(scenario, "x", seed, cfg.coupling_gain)
        y_env, y_bundle = _build_refline_env(scenario, "y", seed + 17, cfg.coupling_gain)
        x_env.reset(axis="x", seed=seed, external_episode_bundle=x_bundle)
        obs = y_env.reset(axis="y", seed=seed + 17, external_episode_bundle=y_bundle)
        x_controller = PIDController()
        x_controller.reset()
        agent.reset()
        agent.policy.set_exploration_noise(
            cfg.exploration_noise_start
            + (cfg.exploration_noise_end - cfg.exploration_noise_start)
            * float(episode) / float(max(cfg.warm_start_episodes - 1, 1))
        )
        history = [obs.copy()]
        rollout: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque()
        done = False
        total_reward = 0.0
        losses = {"critic_loss": 0.0, "actor_loss": 0.0}
        while not done:
            x_control = x_controller.step(x_env.reference, x_env.state_axis, dt)
            _, _, x_done, _ = x_env.step(x_control)
            stacked_state = stack_state(history, obs, agent.policy.config.stack_size)
            if total_steps >= cfg.warmup_steps:
                action = agent.act(stacked_state, explore=True)
            else:
                action = np.random.uniform(-1.0, 1.0, size=4).astype(np.float32)
            agent.controller.adapt(action)
            y_control = agent.controller.step(y_env.reference, y_env.state_axis, dt)
            next_obs, reward, done, _ = y_env.step(y_control, external_coupling=float(x_env.pitch))
            total_reward += reward
            next_stacked = stack_state(history, next_obs, agent.policy.config.stack_size)
            rollout.append((stacked_state.copy(), action.copy(), reward, next_stacked.copy(), done or x_done))
            _flush_n_step_rollout(agent, rollout, agent.policy.config.stack_size, gamma, force=done or x_done)
            total_steps += 1
            if total_steps >= cfg.warmup_steps:
                losses = agent.policy.update(cfg.updates_per_step)
            obs = next_obs
            history.append(obs.copy())
            done = done or x_done
        episode_metrics = _episode_metrics_from_env(y_env, total_reward)
        history_rows.append(
            {
                "episode": float(episode),
                "reward": float(total_reward),
                "mae": episode_metrics["mae"],
                "rmse": episode_metrics["rmse"],
                "iae": episode_metrics["iae"],
                "steady_state_error": episode_metrics["steady_state_error"],
                "actor_loss": float(losses["actor_loss"]),
                "critic_loss": float(losses["critic_loss"]),
            }
        )
    return history_rows


def _write_transfer_outputs(
    out_dir: Path,
    method_name: str,
    metrics: dict[str, float],
    trajectory: dict[str, list[float]],
    *,
    source_dir: Path,
    source_checkpoint: str | None,
    training_history: list[dict[str, float]] | None = None,
) -> list[str]:
    ensure_dir(out_dir)
    write_metrics_csv(out_dir / "metrics.csv", [{"method": method_name, **metrics}])
    _write_trajectory_csv(out_dir / "eval_timeseries.csv", trajectory)
    write_json(
        out_dir / "summary.json",
        {
            "x_controller": "pid",
            "y_controller": method_name,
            "source_axis": "x",
            "target_axis": "y",
            "source_dir": str(source_dir),
            "source_checkpoint": source_checkpoint,
            "observation_semantics": "y, vy, roll, roll_rate",
            "metrics": metrics,
        },
    )
    figures = plot_time_response({method_name: trajectory}, out_dir / "figures")
    if training_history:
        write_metrics_csv(out_dir / "training_history.csv", training_history)
        figures.extend(plot_control_training_curves({method_name: training_history}, out_dir / "figures"))
    return [str(path) for path in figures]


def _run_coupled_pid_baseline(cfg: AxisTransferExperimentConfig, scenario: Any, out_dir: Path) -> dict[str, object]:
    rows: list[dict[str, float]] = []
    representative: dict[str, list[float]] | None = None
    for episode in range(cfg.baseline_episodes):
        metrics, trajectory = _run_coupled_controller_episode(scenario, cfg.seed + episode, cfg.coupling_gain, PIDController())
        rows.append(metrics)
        if representative is None:
            representative = trajectory
    metrics = _aggregate_metric_rows(rows)
    figures = _write_transfer_outputs(
        out_dir,
        "y_pid_baseline",
        metrics,
        representative or {},
        source_dir=Path("."),
        source_checkpoint=None,
    )
    return {"metrics": metrics, "trajectory": representative or {}, "figures": figures}


def _run_coupled_ladrc_transfer(
    cfg: AxisTransferExperimentConfig,
    scenario: Any,
    ladrc_params: dict[str, float],
    out_dir: Path,
    source_dir: Path,
) -> dict[str, object]:
    controller = LADRCController(
        r=float(ladrc_params.get("r", 10.0)),
        b0=float(ladrc_params.get("b0", 30.5)),
        omega_c=float(ladrc_params.get("omega_c", 1.5)),
        k=float(ladrc_params.get("k", 11.0)),
    )
    rows: list[dict[str, float]] = []
    representative: dict[str, list[float]] | None = None
    for episode in range(cfg.baseline_episodes):
        metrics, trajectory = _run_coupled_controller_episode(scenario, cfg.seed + episode, cfg.coupling_gain, controller)
        rows.append(metrics)
        if representative is None:
            representative = trajectory
    metrics = _aggregate_metric_rows(rows)
    figures = _write_transfer_outputs(
        out_dir,
        "y_ladrc_transferred",
        metrics,
        representative or {},
        source_dir=source_dir,
        source_checkpoint=None,
    )
    return {"metrics": metrics, "trajectory": representative or {}, "figures": figures}


def _run_coupled_rl_transfer(
    cfg: AxisTransferExperimentConfig,
    scenario: Any,
    out_dir: Path,
    checkpoint: dict[str, object],
    *,
    shared_value: int,
    warm_start: bool,
    source_dir: Path,
) -> dict[str, object]:
    agent = _build_transfer_agent(cfg, shared_value, checkpoint)
    training_history: list[dict[str, float]] | None = None
    if warm_start:
        training_history = _run_coupled_rl_warm_start_training(cfg, scenario, agent)
    metrics, trajectory = _run_coupled_rl_eval(cfg, scenario, agent, cfg.eval_episodes)
    figures = _write_transfer_outputs(
        out_dir,
        f"y_rl_{'warm_start' if warm_start else 'zero_shot'}_v{shared_value}",
        metrics,
        trajectory,
        source_dir=source_dir,
        source_checkpoint=str(source_dir / f"v{shared_value}" / f"checkpoint_mddpg_v{shared_value}.pt"),
        training_history=training_history,
    )
    source_training = source_dir / f"training_mddpg_v{shared_value}.csv"
    if source_training.exists():
        shutil.copyfile(source_training, out_dir / "source_training_history.csv")
    final_checkpoint = {
        "actor": agent.policy.actor.state_dict(),
        "critic": agent.policy.critic.state_dict(),
        "actor_target": agent.policy.actor_target.state_dict(),
        "critic_target": agent.policy.critic_target.state_dict(),
        "axis": cfg.target_axis,
        "ladrc_baseline": {
            "r": float(agent.controller.base.r),
            "b0": float(agent.controller.base.b0),
            "omega_c": float(agent.controller.base.omega_c),
            "k": float(agent.controller.base.k),
        },
        "normalizer": agent.policy._normalizer.copy(),
        "last_action": agent.policy._last_action.copy(),
        "hold_counter": int(agent.policy._hold_counter),
        "current_expl_noise": float(agent.policy._current_expl_noise),
    }
    save_checkpoint(str(out_dir / f"checkpoint_v{shared_value}.pt"), final_checkpoint)
    return {"metrics": metrics, "trajectory": trajectory, "figures": figures}


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
