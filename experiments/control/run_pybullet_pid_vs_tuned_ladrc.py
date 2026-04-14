from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lc.common.io import ensure_dir
from lc.control.configs import ArtifactConfig, PyBulletControlExperimentConfig
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


def _plot_three_way_tracking(
    output_dir: Path,
    pid_rows: list[dict[str, float]],
    default_rows: list[dict[str, float]],
    tuned_rows: list[dict[str, float]],
) -> list[Path]:
    figures_dir = ensure_dir(output_dir / "figures")
    times = [row["time"] for row in pid_rows]
    reference = [row["target_x"] for row in pid_rows]
    pid_output = [row["x"] for row in pid_rows]
    default_output = [row["x"] for row in default_rows]
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
    ax.plot(times, default_output, color="#1f77b4", linewidth=1.9, label="Default LADRC")
    ax.plot(times, tuned_output, color="#2ca02c", linewidth=2.1, label="Tuned LADRC")
    ax.set_title("PyBullet X-Axis Tracking: PID vs LADRC")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    svg_path = figures_dir / "pid_default_tuned_ladrc_tracking.svg"
    png_path = figures_dir / "pid_default_tuned_ladrc_tracking.png"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(times, np.asarray(reference) - np.asarray(pid_output), color="#d62728", linewidth=1.8, label="PID error")
    ax.plot(times, np.asarray(reference) - np.asarray(default_output), color="#1f77b4", linewidth=1.8, label="Default LADRC error")
    ax.plot(times, np.asarray(reference) - np.asarray(tuned_output), color="#2ca02c", linewidth=2.0, label="Tuned LADRC error")
    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.set_title("PyBullet X-Axis Tracking Error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.legend(frameon=False)
    fig.tight_layout()

    err_svg_path = figures_dir / "pid_default_tuned_ladrc_error.svg"
    err_png_path = figures_dir / "pid_default_tuned_ladrc_error.png"
    fig.savefig(err_svg_path, bbox_inches="tight")
    fig.savefig(err_png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [svg_path, png_path, err_svg_path, err_png_path]


def main() -> None:
    axis = "x"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path("outputs") / "control_pybullet" / "pid_vs_tuned_ladrc" / axis / stamp
    ensure_dir(output_root)

    cfg = PyBulletControlExperimentConfig(
        gui=False,
        seed=7,
        duration_sec=8.0,
        eval_episodes=3,
        artifact=ArtifactConfig(output_root=str(output_root)),
    )
    trainer = PyBulletAxisTrainer(cfg)

    default_axis = create_controller_bundle("ladrc_x_pos_pid_att").parameter_set.axis_config(axis)
    seed_candidates = [
        Candidate("repo_default_x", default_axis.b0, default_axis.omega_c, default_axis.k),
        Candidate("summary_fast_anchor", 30.5, 1.5, 10.5),
        Candidate("summary_steady_anchor", 30.5, 0.8, 7.0),
        Candidate("summary_disturbed_fast", 1.0, 8.25, 4.0),
    ]

    search_rows = [
        _evaluate_candidate(trainer, axis, candidate, difficulty="medium", episodes=3) for candidate in seed_candidates
    ]
    best_seed = min(search_rows, key=lambda row: float(row["score"]))
    best_seed_candidate = Candidate(
        name=str(best_seed["name"]),
        b0=float(best_seed["b0"]),
        omega_c=float(best_seed["omega_c"]),
        k=float(best_seed["k"]),
    )

    local_scales = (0.88, 1.0, 1.12)
    refine_candidates: list[Candidate] = []
    seen: set[tuple[float, float, float]] = set()
    for b0_scale in local_scales:
        for wc_scale in local_scales:
            for k_scale in local_scales:
                candidate = Candidate(
                    name=f"refine_b{b0_scale:.2f}_w{wc_scale:.2f}_k{k_scale:.2f}",
                    b0=float(best_seed_candidate.b0 * b0_scale),
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

    write_timeseries_csv(output_root / "pid_timeseries.csv", list(pid_result["timeseries"]))
    write_timeseries_csv(output_root / "default_ladrc_timeseries.csv", list(default_result["timeseries"]))
    write_timeseries_csv(output_root / "tuned_ladrc_timeseries.csv", list(tuned_result["timeseries"]))
    write_metrics_csv(
        output_root / "metrics.csv",
        [
            {"controller": "pid_pos_att", **pid_result["metrics"]},
            {"controller": "ladrc_x_pos_pid_att_default", **default_result["metrics"]},
            {"controller": "ladrc_x_pos_pid_att_tuned", **tuned_result["metrics"]},
        ],
    )
    write_metrics_csv(output_root / "candidate_screening.csv", search_rows)
    write_metrics_csv(output_root / "candidate_refine.csv", refine_rows)
    write_metrics_csv(output_root / "candidate_validation.csv", validation_rows)

    figure_paths = _plot_three_way_tracking(
        output_root,
        list(pid_result["timeseries"]),
        list(default_result["timeseries"]),
        list(tuned_result["timeseries"]),
    )

    summary = {
        "axis": axis,
        "backend": pid_result["backend"],
        "reference_segments": summarize_reference_segments(reference_bundle),
        "default_repo_params": {
            "b0": float(default_axis.b0),
            "omega_c": float(default_axis.omega_c),
            "k": float(default_axis.k),
            "r": float(default_axis.r),
        },
        "seed_candidates": [asdict(candidate) for candidate in seed_candidates],
        "best_seed_candidate": asdict(best_seed_candidate),
        "optimized_candidate": asdict(tuned_candidate),
        "metrics": {
            "pid_pos_att": pid_result["metrics"],
            "ladrc_x_pos_pid_att_default": default_result["metrics"],
            "ladrc_x_pos_pid_att_tuned": tuned_result["metrics"],
        },
        "figures": [str(path) for path in figure_paths],
    }
    write_summary_json(output_root / "summary.json", summary)
    (output_root / "summary_readable.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_root), "backend": pid_result["backend"], "optimized_candidate": asdict(tuned_candidate)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
