from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lc.common.io import ensure_dir
from lc.control.configs import ArtifactConfig, AxisTrainingConfig, PyBulletControlExperimentConfig
from lc.control.controllers import create_controller_bundle
from lc.control.envs import run_controller_episode
from lc.control.io import write_metrics_csv, write_summary_json, write_timeseries_csv
from lc.control.reference_generators import build_xyz_reference_trajectory, summarize_reference_segments
from lc.control.trainers import PyBulletAxisTrainer


@dataclass(frozen=True)
class Candidate:
    name: str
    b0: float
    omega_c: float
    k: float


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    description: str
    axis_config: AxisTrainingConfig


def _candidate_dict(candidate: Candidate) -> dict[str, float]:
    return {"b0": float(candidate.b0), "omega_c": float(candidate.omega_c), "k": float(candidate.k)}


def _evaluate_candidate(
    trainer: PyBulletAxisTrainer,
    axis: str,
    candidate: Candidate,
    *,
    difficulty: str,
    episodes: int,
) -> dict[str, float]:
    result = trainer.evaluate_single_axis_ladrc_variant(
        axis,
        _candidate_dict(candidate),
        difficulty=difficulty,
        episodes=episodes,
    )
    row = dict(result["metrics"])
    row["name"] = candidate.name
    row["difficulty"] = difficulty
    row["b0"] = float(candidate.b0)
    row["omega_c"] = float(candidate.omega_c)
    row["k"] = float(candidate.k)
    return row


def _build_config(output_root: Path, axis_cfg: AxisTrainingConfig) -> PyBulletControlExperimentConfig:
    return PyBulletControlExperimentConfig(
        gui=False,
        seed=7,
        duration_sec=8.0,
        eval_episodes=3,
        artifact=ArtifactConfig(output_root=str(output_root)),
        axis_configs=(
            axis_cfg,
            AxisTrainingConfig(
                axis="y",
                initial_position=(0.0, 0.0, 1.0),
                fixed_axes=(0.0, 1.0),
                primary_speed_range=(0.08, 0.16),
                reverse_speed_range=(-0.14, -0.08),
                stage_duration_range=(1.6, 2.8),
                include_disturbance=False,
                disturbance_scale=0.0,
            ),
            AxisTrainingConfig(
                axis="z",
                initial_position=(0.0, 0.0, 1.0),
                fixed_axes=(0.0, 0.0),
                primary_speed_range=(0.04, 0.08),
                reverse_speed_range=(-0.08, -0.04),
                stage_duration_range=(1.8, 3.0),
                include_disturbance=False,
                disturbance_scale=0.0,
            ),
        ),
    )


def _seed_candidates(default_axis: object) -> list[Candidate]:
    return [
        Candidate("repo_default_x", float(default_axis.b0), float(default_axis.omega_c), float(default_axis.k)),
        Candidate("summary_fast_anchor", 30.5, 1.5, 10.5),
        Candidate("summary_steady_anchor", 30.5, 0.8, 7.0),
        Candidate("summary_disturbed_fast", 1.0, 8.25, 4.0),
    ]


def _search_best_candidate(trainer: PyBulletAxisTrainer, axis: str, default_axis: object) -> tuple[Candidate, list[dict[str, float]], list[dict[str, float]], list[dict[str, float]]]:
    search_rows = [
        _evaluate_candidate(trainer, axis, candidate, difficulty="medium", episodes=3)
        for candidate in _seed_candidates(default_axis)
    ]
    best_seed = min(search_rows, key=lambda row: float(row["score"]))
    best_seed_candidate = Candidate(
        name=str(best_seed["name"]),
        b0=float(best_seed["b0"]),
        omega_c=float(best_seed["omega_c"]),
        k=float(best_seed["k"]),
    )

    local_scales = (0.80, 0.90, 1.0, 1.10, 1.20)
    refine_candidates: list[Candidate] = []
    seen: set[tuple[float, float, float]] = set()
    for b0_scale in local_scales:
        for wc_scale in local_scales:
            for k_scale in local_scales:
                candidate = Candidate(
                    name=f"refine_b{b0_scale:.2f}_w{wc_scale:.2f}_k{k_scale:.2f}",
                    b0=float(max(best_seed_candidate.b0 * b0_scale, 0.1)),
                    omega_c=float(max(best_seed_candidate.omega_c * wc_scale, 0.1)),
                    k=float(max(best_seed_candidate.k * k_scale, 0.1)),
                )
                key = (round(candidate.b0, 6), round(candidate.omega_c, 6), round(candidate.k, 6))
                if key in seen:
                    continue
                seen.add(key)
                refine_candidates.append(candidate)
    refine_rows = [
        _evaluate_candidate(trainer, axis, candidate, difficulty="medium", episodes=3) for candidate in refine_candidates
    ]
    best_refine = min(refine_rows, key=lambda row: float(row["score"]))
    tuned_candidate = Candidate(
        name="optimized_ladrc",
        b0=float(best_refine["b0"]),
        omega_c=float(best_refine["omega_c"]),
        k=float(best_refine["k"]),
    )
    validation_rows = [
        _evaluate_candidate(trainer, axis, tuned_candidate, difficulty="hard", episodes=2),
        _evaluate_candidate(trainer, axis, best_seed_candidate, difficulty="hard", episodes=2),
    ]
    return tuned_candidate, search_rows, refine_rows, validation_rows


def _run_controller_set(
    cfg: PyBulletControlExperimentConfig,
    axis: str,
    tuned_candidate: Candidate,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], object, object]:
    reference_bundle = build_xyz_reference_trajectory(
        cfg.axis_config(axis),
        cfg,
        rng=np.random.default_rng(cfg.seed + 501),
    )
    pid_controller = create_controller_bundle("pid_pos_att")
    default_ladrc_controller = create_controller_bundle("ladrc_x_pos_pid_att")
    tuned_ladrc_controller = create_controller_bundle("ladrc_x_pos_pid_att")
    tuned_ladrc_controller.set_axis_parameters(
        axis,
        b0=tuned_candidate.b0,
        omega_c=tuned_candidate.omega_c,
        k=tuned_candidate.k,
    )
    pid_result = run_controller_episode(cfg, pid_controller, reference_bundle)
    default_result = run_controller_episode(cfg, default_ladrc_controller, reference_bundle)
    tuned_result = run_controller_episode(cfg, tuned_ladrc_controller, reference_bundle)
    return pid_result, default_result, tuned_result, reference_bundle, default_ladrc_controller.parameter_set.axis_config(axis)


def _plot_scenario(output_dir: Path, scenario_name: str, pid_rows: list[dict[str, float]], tuned_rows: list[dict[str, float]]) -> list[Path]:
    figures_dir = ensure_dir(output_dir / "figures")
    times = [row["time"] for row in pid_rows]
    reference = [row["target_x"] for row in pid_rows]
    pid_output = [row["x"] for row in pid_rows]
    tuned_output = [row["x"] for row in tuned_rows]

    plt.rcParams.update(
        {
            "font.family": "Microsoft YaHei",
            "axes.unicode_minus": False,
            "figure.figsize": (10, 5.6),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots()
    ax.plot(times, reference, color="#222222", linewidth=2.3, label="Reference")
    ax.plot(times, pid_output, color="#d62728", linewidth=1.9, label="PID")
    ax.plot(times, tuned_output, color="#2ca02c", linewidth=2.1, label="Tuned LADRC")
    ax.set_title(f"{scenario_name}: PID vs Tuned LADRC")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.legend(frameon=False)
    fig.tight_layout()
    tracking_svg = figures_dir / f"{scenario_name}_pid_vs_tuned_tracking.svg"
    tracking_png = figures_dir / f"{scenario_name}_pid_vs_tuned_tracking.png"
    fig.savefig(tracking_svg, bbox_inches="tight")
    fig.savefig(tracking_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(times, np.asarray(reference) - np.asarray(pid_output), color="#d62728", linewidth=1.8, label="PID error")
    ax.plot(times, np.asarray(reference) - np.asarray(tuned_output), color="#2ca02c", linewidth=2.0, label="Tuned LADRC error")
    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.set_title(f"{scenario_name}: Tracking Error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.legend(frameon=False)
    fig.tight_layout()
    error_svg = figures_dir / f"{scenario_name}_pid_vs_tuned_error.svg"
    error_png = figures_dir / f"{scenario_name}_pid_vs_tuned_error.png"
    fig.savefig(error_svg, bbox_inches="tight")
    fig.savefig(error_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [tracking_svg, tracking_png, error_svg, error_png]


def _plot_suite_overview(output_dir: Path, scenario_rows: list[dict[str, object]]) -> list[Path]:
    figures_dir = ensure_dir(output_dir / "figures")
    names = [str(row["scenario"]) for row in scenario_rows]
    pid_rmse = [float(row["pid_rmse"]) for row in scenario_rows]
    ladrc_rmse = [float(row["ladrc_rmse"]) for row in scenario_rows]
    pid_reward = [float(row["pid_reward"]) for row in scenario_rows]
    ladrc_reward = [float(row["ladrc_reward"]) for row in scenario_rows]

    plt.rcParams.update(
        {
            "font.family": "Microsoft YaHei",
            "axes.unicode_minus": False,
            "figure.figsize": (10, 5.6),
            "axes.grid": True,
            "grid.alpha": 0.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, pid_rmse, width, label="PID", color="#d62728")
    ax.bar(x + width / 2, ladrc_rmse, width, label="Tuned LADRC", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.set_ylabel("RMSE")
    ax.set_title("PID vs Tuned LADRC Across PyBullet Scenarios")
    ax.legend(frameon=False)
    fig.tight_layout()
    rmse_svg = figures_dir / "scenario_rmse_comparison.svg"
    rmse_png = figures_dir / "scenario_rmse_comparison.png"
    fig.savefig(rmse_svg, bbox_inches="tight")
    fig.savefig(rmse_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, pid_reward, width, label="PID", color="#d62728")
    ax.bar(x + width / 2, ladrc_reward, width, label="Tuned LADRC", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.set_ylabel("Average Reward")
    ax.set_title("PID vs Tuned LADRC Reward Across Scenarios")
    ax.legend(frameon=False)
    fig.tight_layout()
    reward_svg = figures_dir / "scenario_reward_comparison.svg"
    reward_png = figures_dir / "scenario_reward_comparison.png"
    fig.savefig(reward_svg, bbox_inches="tight")
    fig.savefig(reward_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [rmse_svg, rmse_png, reward_svg, reward_png]


def _scenario_specs() -> list[ScenarioSpec]:
    base = AxisTrainingConfig(
        axis="x",
        initial_position=(0.0, 0.0, 1.0),
        fixed_axes=(0.0, 1.0),
        primary_speed_range=(0.08, 0.16),
        reverse_speed_range=(-0.14, -0.08),
        stage_duration_range=(1.6, 2.8),
        include_disturbance=False,
        disturbance_scale=0.0,
        stage_count=4,
    )
    return [
        ScenarioSpec("baseline", "Current medium reference without disturbance", base),
        ScenarioSpec(
            "lateral_disturbance",
            "Current reference with sustained x-axis disturbance",
            replace(base, include_disturbance=True, disturbance_scale=0.18, disturbance_axis_bias=1.0),
        ),
        ScenarioSpec(
            "aggressive_reference",
            "Faster and shorter stage transitions to stress transient tracking",
            replace(
                base,
                primary_speed_range=(0.18, 0.28),
                reverse_speed_range=(-0.28, -0.18),
                stage_duration_range=(0.9, 1.5),
                stage_count=4,
            ),
        ),
    ]


def main() -> None:
    axis = "x"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_root = ensure_dir(Path("outputs") / "control_pybullet" / "pid_vs_tuned_ladrc_stress_suite" / axis / stamp)

    scenario_summary_rows: list[dict[str, object]] = []
    all_figure_paths: list[str] = []
    scenario_payloads: dict[str, object] = {}

    for spec in _scenario_specs():
        scenario_root = ensure_dir(suite_root / spec.name)
        cfg = _build_config(scenario_root, spec.axis_config)
        trainer = PyBulletAxisTrainer(cfg)
        default_axis = create_controller_bundle("ladrc_x_pos_pid_att").parameter_set.axis_config(axis)
        tuned_candidate, search_rows, refine_rows, validation_rows = _search_best_candidate(trainer, axis, default_axis)
        pid_result, default_result, tuned_result, reference_bundle, default_axis_after = _run_controller_set(cfg, axis, tuned_candidate)

        write_timeseries_csv(scenario_root / "pid_timeseries.csv", list(pid_result["timeseries"]))
        write_timeseries_csv(scenario_root / "default_ladrc_timeseries.csv", list(default_result["timeseries"]))
        write_timeseries_csv(scenario_root / "tuned_ladrc_timeseries.csv", list(tuned_result["timeseries"]))
        write_metrics_csv(
            scenario_root / "metrics.csv",
            [
                {"controller": "pid_pos_att", **pid_result["metrics"]},
                {"controller": "ladrc_x_pos_pid_att_default", **default_result["metrics"]},
                {"controller": "ladrc_x_pos_pid_att_tuned", **tuned_result["metrics"]},
            ],
        )
        write_metrics_csv(scenario_root / "candidate_screening.csv", search_rows)
        write_metrics_csv(scenario_root / "candidate_refine.csv", refine_rows)
        write_metrics_csv(scenario_root / "candidate_validation.csv", validation_rows)
        figure_paths = _plot_scenario(
            scenario_root,
            spec.name,
            list(pid_result["timeseries"]),
            list(tuned_result["timeseries"]),
        )
        all_figure_paths.extend([str(path) for path in figure_paths])

        pid_rmse = float(pid_result["metrics"]["rmse"])
        ladrc_rmse = float(tuned_result["metrics"]["rmse"])
        pid_reward = float(pid_result["metrics"]["reward"])
        ladrc_reward = float(tuned_result["metrics"]["reward"])
        ladrc_beats_pid = ladrc_rmse < pid_rmse and ladrc_reward >= pid_reward

        scenario_summary_rows.append(
            {
                "scenario": spec.name,
                "description": spec.description,
                "backend": str(pid_result["backend"]),
                "pid_rmse": pid_rmse,
                "ladrc_rmse": ladrc_rmse,
                "pid_reward": pid_reward,
                "ladrc_reward": ladrc_reward,
                "ladrc_beats_pid": 1 if ladrc_beats_pid else 0,
                "best_b0": tuned_candidate.b0,
                "best_omega_c": tuned_candidate.omega_c,
                "best_k": tuned_candidate.k,
            }
        )
        scenario_payloads[spec.name] = {
            "description": spec.description,
            "axis_config": asdict(spec.axis_config),
            "backend": pid_result["backend"],
            "reference_segments": summarize_reference_segments(reference_bundle),
            "default_repo_params": {
                "b0": float(default_axis_after.b0),
                "omega_c": float(default_axis_after.omega_c),
                "k": float(default_axis_after.k),
                "r": float(default_axis_after.r),
            },
            "optimized_candidate": asdict(tuned_candidate),
            "metrics": {
                "pid_pos_att": pid_result["metrics"],
                "ladrc_x_pos_pid_att_default": default_result["metrics"],
                "ladrc_x_pos_pid_att_tuned": tuned_result["metrics"],
            },
            "figures": [str(path) for path in figure_paths],
        }
        write_summary_json(scenario_root / "summary.json", scenario_payloads[spec.name])

    write_metrics_csv(suite_root / "scenario_summary.csv", scenario_summary_rows)
    overview_figures = _plot_suite_overview(suite_root, scenario_summary_rows)
    all_figure_paths.extend([str(path) for path in overview_figures])

    recommendation = {
        "retune_needed": any(int(row["ladrc_beats_pid"]) == 0 for row in scenario_summary_rows),
        "best_scenario_for_ladrc": min(scenario_summary_rows, key=lambda row: float(row["ladrc_rmse"]))["scenario"],
        "ladrc_wins": [row["scenario"] for row in scenario_summary_rows if int(row["ladrc_beats_pid"]) == 1],
        "ladrc_losses": [row["scenario"] for row in scenario_summary_rows if int(row["ladrc_beats_pid"]) == 0],
    }
    payload = {
        "axis": axis,
        "suite_output_dir": str(suite_root),
        "scenario_summary": scenario_summary_rows,
        "scenarios": scenario_payloads,
        "recommendation": recommendation,
        "figures": all_figure_paths,
    }
    write_summary_json(suite_root / "summary.json", payload)
    (suite_root / "summary_readable.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(suite_root), "recommendation": recommendation}, ensure_ascii=False))


if __name__ == "__main__":
    main()
