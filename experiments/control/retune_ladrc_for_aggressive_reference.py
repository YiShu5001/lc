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


def _candidate_dict(candidate: Candidate) -> dict[str, float]:
    return {"b0": float(candidate.b0), "omega_c": float(candidate.omega_c), "k": float(candidate.k)}


def _build_aggressive_config(output_root: Path) -> PyBulletControlExperimentConfig:
    aggressive_x = AxisTrainingConfig(
        axis="x",
        initial_position=(0.0, 0.0, 1.0),
        fixed_axes=(0.0, 1.0),
        primary_speed_range=(0.18, 0.28),
        reverse_speed_range=(-0.28, -0.18),
        stage_duration_range=(0.9, 1.5),
        include_disturbance=False,
        disturbance_scale=0.0,
        stage_count=4,
    )
    return PyBulletControlExperimentConfig(
        gui=False,
        seed=7,
        duration_sec=8.0,
        eval_episodes=3,
        artifact=ArtifactConfig(output_root=str(output_root)),
        axis_configs=(
            aggressive_x,
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


def _evaluate_candidate(trainer: PyBulletAxisTrainer, candidate: Candidate, episodes: int = 2) -> dict[str, float]:
    result = trainer.evaluate_single_axis_ladrc_variant("x", _candidate_dict(candidate), difficulty="medium", episodes=episodes)
    row = dict(result["metrics"])
    row["name"] = candidate.name
    row["b0"] = candidate.b0
    row["omega_c"] = candidate.omega_c
    row["k"] = candidate.k
    return row


def _plot_result(output_root: Path, pid_rows: list[dict[str, float]], ladrc_rows: list[dict[str, float]]) -> list[Path]:
    figures_dir = ensure_dir(output_root / "figures")
    times = [row["time"] for row in pid_rows]
    reference = [row["target_x"] for row in pid_rows]
    pid_output = [row["x"] for row in pid_rows]
    ladrc_output = [row["x"] for row in ladrc_rows]

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
    ax.plot(times, ladrc_output, color="#2ca02c", linewidth=2.1, label="Retuned LADRC")
    ax.set_title("Aggressive Reference: PID vs Retuned LADRC")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.legend(frameon=False)
    fig.tight_layout()
    svg1 = figures_dir / "aggressive_pid_vs_retuned_ladrc_tracking.svg"
    png1 = figures_dir / "aggressive_pid_vs_retuned_ladrc_tracking.png"
    fig.savefig(svg1, bbox_inches="tight")
    fig.savefig(png1, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(times, np.asarray(reference) - np.asarray(pid_output), color="#d62728", linewidth=1.8, label="PID error")
    ax.plot(times, np.asarray(reference) - np.asarray(ladrc_output), color="#2ca02c", linewidth=2.0, label="Retuned LADRC error")
    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.set_title("Aggressive Reference Error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.legend(frameon=False)
    fig.tight_layout()
    svg2 = figures_dir / "aggressive_pid_vs_retuned_ladrc_error.svg"
    png2 = figures_dir / "aggressive_pid_vs_retuned_ladrc_error.png"
    fig.savefig(svg2, bbox_inches="tight")
    fig.savefig(png2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [svg1, png1, svg2, png2]


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = ensure_dir(Path("outputs") / "control_pybullet" / "aggressive_reference_retune" / "x" / stamp)
    cfg = _build_aggressive_config(output_root)
    trainer = PyBulletAxisTrainer(cfg)

    default_axis = create_controller_bundle("ladrc_x_pos_pid_att").parameter_set.axis_config("x")
    base_candidates = [
        Candidate("repo_default", float(default_axis.b0), float(default_axis.omega_c), float(default_axis.k)),
        Candidate("baseline_suite_best", 36.6, 0.8, 7.7),
        Candidate("steady_anchor", 30.5, 0.8, 7.0),
        Candidate("fast_anchor", 30.5, 1.5, 10.5),
        Candidate("fast_wc_high", 30.5, 2.2, 10.5),
        Candidate("fast_wc_mid", 30.5, 1.8, 9.0),
        Candidate("high_b0_fast", 45.0, 1.2, 8.0),
        Candidate("high_b0_mid", 60.0, 1.0, 7.0),
        Candidate("very_fast_low_k", 20.0, 3.0, 4.0),
    ]
    coarse_rows = [_evaluate_candidate(trainer, candidate, episodes=2) for candidate in base_candidates]
    top_rows = sorted(coarse_rows, key=lambda row: float(row["score"]))[:3]

    refine_candidates: list[Candidate] = []
    seen: set[tuple[float, float, float]] = set()
    b0_scales = (0.75, 0.9, 1.0, 1.15, 1.3)
    wc_scales = (0.7, 0.85, 1.0, 1.2, 1.5)
    k_scales = (0.75, 0.9, 1.0, 1.15, 1.3)
    for row in top_rows:
        for b0_scale in b0_scales:
            for wc_scale in wc_scales:
                for k_scale in k_scales:
                    candidate = Candidate(
                        name=f"refine_{row['name']}_b{b0_scale:.2f}_w{wc_scale:.2f}_k{k_scale:.2f}",
                        b0=float(max(float(row["b0"]) * b0_scale, 0.2)),
                        omega_c=float(max(float(row["omega_c"]) * wc_scale, 0.2)),
                        k=float(max(float(row["k"]) * k_scale, 0.2)),
                    )
                    key = (round(candidate.b0, 6), round(candidate.omega_c, 6), round(candidate.k, 6))
                    if key in seen:
                        continue
                    seen.add(key)
                    refine_candidates.append(candidate)
    refine_rows = [_evaluate_candidate(trainer, candidate, episodes=2) for candidate in refine_candidates]
    best_row = min(refine_rows + coarse_rows, key=lambda row: float(row["score"]))
    best_candidate = Candidate("retuned_aggressive_best", float(best_row["b0"]), float(best_row["omega_c"]), float(best_row["k"]))

    reference_bundle = build_xyz_reference_trajectory(
        cfg.axis_config("x"),
        cfg,
        rng=np.random.default_rng(cfg.seed + 701),
    )
    pid_controller = create_controller_bundle("pid_pos_att")
    ladrc_controller = create_controller_bundle("ladrc_x_pos_pid_att")
    ladrc_controller.set_axis_parameters("x", b0=best_candidate.b0, omega_c=best_candidate.omega_c, k=best_candidate.k)
    pid_result = run_controller_episode(cfg, pid_controller, reference_bundle)
    ladrc_result = run_controller_episode(cfg, ladrc_controller, reference_bundle)

    write_metrics_csv(output_root / "coarse_search.csv", coarse_rows)
    write_metrics_csv(output_root / "refine_search.csv", refine_rows)
    write_metrics_csv(
        output_root / "metrics.csv",
        [
            {"controller": "pid_pos_att", **pid_result["metrics"]},
            {"controller": "ladrc_x_pos_pid_att_retuned", **ladrc_result["metrics"]},
        ],
    )
    write_timeseries_csv(output_root / "pid_timeseries.csv", list(pid_result["timeseries"]))
    write_timeseries_csv(output_root / "retuned_ladrc_timeseries.csv", list(ladrc_result["timeseries"]))
    figures = _plot_result(output_root, list(pid_result["timeseries"]), list(ladrc_result["timeseries"]))

    payload = {
        "backend": pid_result["backend"],
        "reference_segments": summarize_reference_segments(reference_bundle),
        "best_candidate": asdict(best_candidate),
        "pid_metrics": pid_result["metrics"],
        "ladrc_metrics": ladrc_result["metrics"],
        "ladrc_beats_pid": bool(
            float(ladrc_result["metrics"]["rmse"]) < float(pid_result["metrics"]["rmse"])
            and float(ladrc_result["metrics"]["reward"]) >= float(pid_result["metrics"]["reward"])
        ),
        "figures": [str(path) for path in figures],
    }
    write_summary_json(output_root / "summary.json", payload)
    (output_root / "summary_readable.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_root), "best_candidate": asdict(best_candidate), "ladrc_beats_pid": payload["ladrc_beats_pid"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
