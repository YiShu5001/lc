from __future__ import annotations

from pathlib import Path

from control.Tuning_ladrc.parameter_loader import load_axis_parameter_file
from lc.common.io import ensure_dir, write_json, write_metrics_csv
from lc.common.utils import seed_everything
from lc.control.configs import ArtifactConfig, ControlExperimentConfig, PyBulletControlExperimentConfig
from lc.control.controllers import create_controller_bundle
from lc.control.envs import run_controller_episode
from lc.control.experiments.pybullet_compare import (
    run_pybullet_axis_training,
    run_pybullet_controller_benchmark,
    run_pybullet_full_experiment,
)
from lc.control.io import write_reference_csv, write_timeseries_csv
from lc.control.plotting.pybullet_plots import (
    plot_axis_error,
    plot_control_effort,
    plot_pid_vs_best_ladrc_response,
)
from lc.control.reference_generators import build_xyz_reference_trajectory, summarize_reference_segments
from lc.control.trainers import PyBulletAxisTrainer


def run_control_comparison(
    config: PyBulletControlExperimentConfig | ControlExperimentConfig | None = None,
) -> dict[str, object]:
    """Run the chapter-3 mainline comparison in PyBullet only."""
    cfg = _coerce_pybullet_config(config)
    return run_pybullet_full_experiment(cfg)


def run_control_generalization(
    config: PyBulletControlExperimentConfig | ControlExperimentConfig | None = None,
) -> dict[str, object]:
    """Run chapter-3 PyBullet controller benchmarks across configured difficulties."""
    base_cfg = _coerce_pybullet_config(config)
    difficulty_levels = (
        tuple(config.difficulty_levels)
        if isinstance(config, ControlExperimentConfig)
        else ("easy", "medium", "hard", "extreme")
    )
    trainer = PyBulletAxisTrainer(base_cfg)
    results: dict[str, dict[str, object]] = {}
    metric_rows: list[dict[str, object]] = []
    for difficulty in difficulty_levels:
        scoped_cfg = trainer._with_difficulty(difficulty)
        bundle = run_pybullet_controller_benchmark(scoped_cfg, axis="all", controller="all")
        results[difficulty] = bundle
        for axis, axis_bundle in bundle.items():
            for row in axis_bundle["metrics"]:
                metric_rows.append({"difficulty": difficulty, "axis": axis, **row})
    out_dir = ensure_dir(Path(base_cfg.artifact.output_root) / "generalization")
    write_metrics_csv(out_dir / "metrics.csv", metric_rows)
    write_json(
        out_dir / "summary.json",
        {
            "difficulty_levels": list(difficulty_levels),
            "mode": "pybullet_only",
            "results": results,
        },
    )
    return {
        "output_dir": str(out_dir),
        "results": results,
    }


def run_pid_vs_fixed_ladrc_tracking(
    config: PyBulletControlExperimentConfig | ControlExperimentConfig | None = None,
    axis: str = "x",
    parameter_file: str | Path = Path("src/control/Tuning_ladrc/default_axis_params.json"),
) -> dict[str, object]:
    """Run PID vs fixed-parameter LADRC tracking directly in PyBullet."""
    cfg = _coerce_pybullet_config(config)
    seed_everything(cfg.seed)
    reference_bundle = build_xyz_reference_trajectory(
        cfg.axis_config(axis),
        cfg,
    )
    axis_params = load_axis_parameter_file(parameter_file)[axis]
    pid = create_controller_bundle("pid_pos_att")
    ladrc = create_controller_bundle(f"ladrc_{axis}_pos_pid_att")
    ladrc.set_axis_parameters(axis, b0=axis_params.b0, omega_c=axis_params.wc, k=axis_params.k)

    pid_result = run_controller_episode(cfg, pid, reference_bundle)
    ladrc_result = run_controller_episode(cfg, ladrc, reference_bundle)

    out_dir = ensure_dir(Path(cfg.artifact.output_root) / f"pid_vs_fixed_ladrc_{axis}")
    fig_dir = ensure_dir(out_dir / "figures")
    write_reference_csv(out_dir / "reference.csv", reference_bundle)
    pid_rows = list(pid_result["timeseries"])
    ladrc_rows = list(ladrc_result["timeseries"])
    write_timeseries_csv(out_dir / "pid_timeseries.csv", pid_rows)
    write_timeseries_csv(out_dir / "ladrc_timeseries.csv", ladrc_rows)
    metric_rows = [
        {"controller": "pid_pos_att", **pid_result["metrics"]},
        {"controller": f"ladrc_{axis}_pos_pid_att", **ladrc_result["metrics"]},
    ]
    write_metrics_csv(out_dir / "metrics.csv", metric_rows)
    comparison_fig = plot_pid_vs_best_ladrc_response(pid_rows, ladrc_rows, axis, fig_dir)
    pid_error = plot_axis_error(pid_rows, fig_dir / "pid")
    ladrc_error = plot_axis_error(ladrc_rows, fig_dir / "ladrc")
    pid_effort = plot_control_effort(pid_rows, fig_dir / "pid")
    ladrc_effort = plot_control_effort(ladrc_rows, fig_dir / "ladrc")
    write_json(
        out_dir / "summary.json",
        {
            "mode": "pybullet_only",
            "axis": axis,
            "parameter_source": str(Path(parameter_file)),
            "fixed_ladrc_params": {
                "b0": float(axis_params.b0),
                "omega_c": float(axis_params.wc),
                "k": float(axis_params.k),
                "omega_o": float(axis_params.wc * axis_params.k),
                "r": float(axis_params.r),
            },
            "reference_segments": summarize_reference_segments(reference_bundle),
            "results": {
                "pid": pid_result["metrics"],
                "ladrc": ladrc_result["metrics"],
            },
            "figures": {
                "position_compare": str(comparison_fig),
                "pid_error": str(pid_error),
                "ladrc_error": str(ladrc_error),
                "pid_effort": str(pid_effort),
                "ladrc_effort": str(ladrc_effort),
            },
        },
    )
    return {
        "axis": axis,
        "output_dir": str(out_dir),
        "fixed_ladrc_params": {
            "b0": float(axis_params.b0),
            "omega_c": float(axis_params.wc),
            "k": float(axis_params.k),
        },
        "results": {
            "pid": pid_result["metrics"],
            "ladrc": ladrc_result["metrics"],
        },
        "figures": [
            str(comparison_fig),
            str(pid_error),
            str(ladrc_error),
            str(pid_effort),
            str(ladrc_effort),
        ],
    }


def _coerce_pybullet_config(
    config: PyBulletControlExperimentConfig | ControlExperimentConfig | None,
) -> PyBulletControlExperimentConfig:
    if isinstance(config, PyBulletControlExperimentConfig):
        return config
    if isinstance(config, ControlExperimentConfig):
        duration_sec = max(float(config.episode_length) / 48.0, 1.0)
        return PyBulletControlExperimentConfig(
            duration_sec=duration_sec,
            seed=config.seed,
            train_episodes=config.train_episodes,
            eval_episodes=max(config.compare_episodes, config.episodes, 1),
            updates_per_step=config.updates_per_step,
            batch_size=config.batch_size,
            artifact=ArtifactConfig(output_root="outputs/control_pybullet"),
        )
    return PyBulletControlExperimentConfig()
