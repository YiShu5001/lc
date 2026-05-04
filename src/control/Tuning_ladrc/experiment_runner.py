from __future__ import annotations

from pathlib import Path
from dataclasses import replace

from control.Tuning_ladrc.parameter_loader import build_single_axis_ladrc_bundle
from control.Tuning_ladrc.schemas import (
    AxisLADRCParameters,
    B0SweepResult,
    ManualTargetProfile,
    TuningCaseResult,
    WCSweepResult,
    XAxisDisturbedRescanResult,
    XAxisDisturbanceRefinedTuningResult,
    XAxisRBalanceResult,
    XAxisTaskTuningResult,
    XAxisRefinedTuningResult,
    XAxisSteadyTuningResult,
    ZAxisSpecializedTuningResult,
)
from control.Tuning_ladrc.parameter_loader import load_axis_parameter_file
from control.Tuning_ladrc.target_profiles import build_manual_reference_profile
from lc.control.configs import PyBulletControlExperimentConfig
from lc.control.controllers import create_controller_bundle
from lc.control.envs import run_controller_episode
from lc.control.io import write_metrics_csv, write_reference_csv, write_summary_json, write_timeseries_csv
from lc.control.plotting import plot_axis_error, plot_axis_tracking, plot_axis_velocity, plot_pid_vs_best_ladrc_response


def run_single_axis_manual_tuning_case(
    axis: str,
    parameter_file: str | Path,
    profile: ManualTargetProfile,
    config: PyBulletControlExperimentConfig | None = None,
    output_root: str | Path = "outputs/control_pybullet_manual_tuning",
) -> TuningCaseResult:
    cfg = config or PyBulletControlExperimentConfig(duration_sec=profile.total_duration)
    reference_bundle = build_manual_reference_profile(
        profile,
        control_dt=cfg.control_dt,
        step_count=cfg.step_count,
    )
    pid_controller = create_controller_bundle("pid_pos_att")
    ladrc_controller = build_single_axis_ladrc_bundle(axis, parameter_file)

    pid_result = run_controller_episode(cfg, pid_controller, reference_bundle)
    ladrc_result = run_controller_episode(cfg, ladrc_controller, reference_bundle)

    run_dir = Path(output_root) / axis / profile.mode
    run_dir.mkdir(parents=True, exist_ok=True)
    write_reference_csv(run_dir / "reference.csv", reference_bundle)
    write_timeseries_csv(run_dir / "pid_timeseries.csv", list(pid_result["timeseries"]))
    write_timeseries_csv(run_dir / "ladrc_timeseries.csv", list(ladrc_result["timeseries"]))
    write_metrics_csv(
        run_dir / "metrics.csv",
        [
            {"controller": "pid_pos_att", **pid_result["metrics"]},
            {"controller": f"ladrc_{axis}_pos_pid_att", **ladrc_result["metrics"]},
        ],
    )

    figures_dir = run_dir / "figures"
    plot_pid_vs_best_ladrc_response(list(pid_result["timeseries"]), list(ladrc_result["timeseries"]), axis, figures_dir)
    plot_axis_tracking(list(ladrc_result["timeseries"]), figures_dir)
    plot_axis_velocity(list(ladrc_result["timeseries"]), figures_dir)
    plot_axis_error(list(ladrc_result["timeseries"]), figures_dir)

    summary = {
        "axis": axis,
        "mode": profile.mode,
        "parameter_file": str(parameter_file),
        "pid_metrics": pid_result["metrics"],
        "ladrc_metrics": ladrc_result["metrics"],
    }
    write_summary_json(run_dir / "summary.json", summary)
    return TuningCaseResult(
        axis=axis,
        mode=profile.mode,
        parameter_file=str(parameter_file),
        output_dir=str(run_dir),
        pid_metrics=dict(pid_result["metrics"]),
        ladrc_metrics=dict(ladrc_result["metrics"]),
    )


def run_b0_step_sweep(
    axis: str,
    parameter_file: str | Path,
    *,
    b0_candidates: list[float],
    profile: ManualTargetProfile | None = None,
    config: PyBulletControlExperimentConfig | None = None,
    output_root: str | Path = "outputs/control_pybullet_manual_tuning",
) -> B0SweepResult:
    base_profile = profile or ManualTargetProfile(axis=axis, mode="hold_step_hold", step_value=0.12, total_duration=6.0)
    cfg = config or PyBulletControlExperimentConfig(duration_sec=base_profile.total_duration)
    reference_bundle = build_manual_reference_profile(
        base_profile,
        control_dt=cfg.control_dt,
        step_count=cfg.step_count,
    )
    pid_controller = create_controller_bundle("pid_pos_att")
    pid_result = run_controller_episode(cfg, pid_controller, reference_bundle)

    sweep_rows: list[dict[str, float]] = []
    best_b0 = float(b0_candidates[0])
    best_score = float("inf")
    run_dir = Path(output_root) / axis / f"{base_profile.mode}_b0_sweep"
    run_dir.mkdir(parents=True, exist_ok=True)

    for index, b0 in enumerate(b0_candidates):
        ladrc_controller = build_single_axis_ladrc_bundle(axis, parameter_file)
        snapshot = ladrc_controller.snapshot_params()
        current_wc = float(snapshot[f"{axis}_omega_c"])
        current_k = float(ladrc_controller.parameter_set.axis_config(axis).k)
        ladrc_controller.set_axis_parameters(axis, b0=float(b0), omega_c=current_wc, k=current_k)
        ladrc_result = run_controller_episode(cfg, ladrc_controller, reference_bundle)
        metrics = dict(ladrc_result["metrics"])
        score = _score_metrics(axis, list(ladrc_result["timeseries"]), metrics)
        row = {
            "candidate_index": float(index),
            "b0": float(b0),
            "score": float(score),
            **{key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))},
        }
        sweep_rows.append(row)
        candidate_dir = run_dir / f"candidate_{index:02d}_b0_{float(b0):.6g}"
        write_timeseries_csv(candidate_dir / "ladrc_timeseries.csv", list(ladrc_result["timeseries"]))
        if score < best_score:
            best_score = float(score)
            best_b0 = float(b0)
            plot_pid_vs_best_ladrc_response(
                list(pid_result["timeseries"]),
                list(ladrc_result["timeseries"]),
                axis,
                run_dir / "figures",
            )

    write_reference_csv(run_dir / "reference.csv", reference_bundle)
    write_timeseries_csv(run_dir / "pid_timeseries.csv", list(pid_result["timeseries"]))
    write_metrics_csv(run_dir / "b0_sweep_metrics.csv", sweep_rows)
    write_summary_json(
        run_dir / "summary.json",
        {
            "axis": axis,
            "mode": base_profile.mode,
            "parameter_file": str(parameter_file),
            "recommended_b0": float(best_b0),
            "pid_metrics": pid_result["metrics"],
            "b0_candidates": [float(value) for value in b0_candidates],
        },
    )
    return B0SweepResult(
        axis=axis,
        mode=base_profile.mode,
        parameter_file=str(parameter_file),
        output_dir=str(run_dir),
        sweep_rows=tuple(sweep_rows),
        recommended_b0=float(best_b0),
    )


def run_wc_step_sweep(
    axis: str,
    parameter_file: str | Path,
    *,
    fixed_b0: float,
    fixed_k: float = 4.0,
    wc_candidates: list[float],
    profile: ManualTargetProfile | None = None,
    config: PyBulletControlExperimentConfig | None = None,
    output_root: str | Path = "outputs/control_pybullet_manual_tuning",
) -> WCSweepResult:
    base_profile = profile or ManualTargetProfile(axis=axis, mode="hold_step_hold", step_value=0.12, total_duration=6.0)
    cfg = config or PyBulletControlExperimentConfig(duration_sec=base_profile.total_duration)
    reference_bundle = build_manual_reference_profile(
        base_profile,
        control_dt=cfg.control_dt,
        step_count=cfg.step_count,
    )
    pid_controller = create_controller_bundle("pid_pos_att")
    pid_result = run_controller_episode(cfg, pid_controller, reference_bundle)

    sweep_rows: list[dict[str, float]] = []
    best_wc = float(wc_candidates[0])
    best_score = float("inf")
    run_dir = Path(output_root) / axis / f"{base_profile.mode}_wc_sweep"
    run_dir.mkdir(parents=True, exist_ok=True)

    for index, wc in enumerate(wc_candidates):
        ladrc_controller = build_single_axis_ladrc_bundle(axis, parameter_file)
        ladrc_controller.set_axis_parameters(axis, b0=float(fixed_b0), omega_c=float(wc), k=float(fixed_k))
        ladrc_result = run_controller_episode(cfg, ladrc_controller, reference_bundle)
        metrics = dict(ladrc_result["metrics"])
        score = _score_metrics(axis, list(ladrc_result["timeseries"]), metrics)
        row = {
            "candidate_index": float(index),
            "b0": float(fixed_b0),
            "k": float(fixed_k),
            "wc": float(wc),
            "score": float(score),
            **{key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))},
        }
        sweep_rows.append(row)
        candidate_dir = run_dir / f"candidate_{index:02d}_wc_{float(wc):.6g}"
        write_timeseries_csv(candidate_dir / "ladrc_timeseries.csv", list(ladrc_result["timeseries"]))
        if score < best_score:
            best_score = float(score)
            best_wc = float(wc)
            plot_pid_vs_best_ladrc_response(
                list(pid_result["timeseries"]),
                list(ladrc_result["timeseries"]),
                axis,
                run_dir / "figures",
            )

    write_reference_csv(run_dir / "reference.csv", reference_bundle)
    write_timeseries_csv(run_dir / "pid_timeseries.csv", list(pid_result["timeseries"]))
    write_metrics_csv(run_dir / "wc_sweep_metrics.csv", sweep_rows)
    write_summary_json(
        run_dir / "summary.json",
        {
            "axis": axis,
            "mode": base_profile.mode,
            "parameter_file": str(parameter_file),
            "fixed_b0": float(fixed_b0),
            "fixed_k": float(fixed_k),
            "recommended_wc": float(best_wc),
            "pid_metrics": pid_result["metrics"],
            "wc_candidates": [float(value) for value in wc_candidates],
        },
    )
    return WCSweepResult(
        axis=axis,
        mode=base_profile.mode,
        parameter_file=str(parameter_file),
        output_dir=str(run_dir),
        fixed_b0=float(fixed_b0),
        fixed_k=float(fixed_k),
        sweep_rows=tuple(sweep_rows),
        recommended_wc=float(best_wc),
    )


def run_k_step_sweep(
    axis: str,
    parameter_file: str | Path,
    *,
    fixed_b0: float,
    fixed_wc: float,
    k_candidates: list[float],
    fixed_r: float = 10.0,
    profile: ManualTargetProfile | None = None,
    config: PyBulletControlExperimentConfig | None = None,
    output_root: str | Path = "outputs/control_pybullet_manual_tuning",
) -> WCSweepResult:
    base_profile = profile or ManualTargetProfile(axis=axis, mode="hold_step_hold", step_value=0.12, total_duration=6.0)
    cfg = config or PyBulletControlExperimentConfig(duration_sec=base_profile.total_duration)
    reference_bundle = build_manual_reference_profile(
        base_profile,
        control_dt=cfg.control_dt,
        step_count=cfg.step_count,
    )
    pid_controller = create_controller_bundle("pid_pos_att")
    pid_result = run_controller_episode(cfg, pid_controller, reference_bundle)

    sweep_rows: list[dict[str, float]] = []
    best_k = float(k_candidates[0])
    best_score = float("inf")
    run_dir = Path(output_root) / axis / f"{base_profile.mode}_k_sweep"
    run_dir.mkdir(parents=True, exist_ok=True)

    for index, k in enumerate(k_candidates):
        ladrc_controller = build_single_axis_ladrc_bundle(axis, parameter_file)
        ladrc_controller.set_axis_parameters(axis, b0=float(fixed_b0), omega_c=float(fixed_wc), k=float(k))
        ladrc_controller.parameter_set.axis_config(axis).r = float(fixed_r)
        if hasattr(ladrc_controller, "_sync_from_parameter_set"):
            ladrc_controller._sync_from_parameter_set()
        ladrc_result = run_controller_episode(cfg, ladrc_controller, reference_bundle)
        metrics = dict(ladrc_result["metrics"])
        score = _score_metrics(axis, list(ladrc_result["timeseries"]), metrics)
        row = {
            "candidate_index": float(index),
            "b0": float(fixed_b0),
            "wc": float(fixed_wc),
            "k": float(k),
            "r": float(fixed_r),
            "score": float(score),
            **{key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))},
        }
        sweep_rows.append(row)
        candidate_dir = run_dir / f"candidate_{index:02d}_k_{float(k):.6g}"
        write_timeseries_csv(candidate_dir / "ladrc_timeseries.csv", list(ladrc_result["timeseries"]))
        if score < best_score:
            best_score = float(score)
            best_k = float(k)
            plot_pid_vs_best_ladrc_response(
                list(pid_result["timeseries"]),
                list(ladrc_result["timeseries"]),
                axis,
                run_dir / "figures",
            )

    write_reference_csv(run_dir / "reference.csv", reference_bundle)
    write_timeseries_csv(run_dir / "pid_timeseries.csv", list(pid_result["timeseries"]))
    write_metrics_csv(run_dir / "k_sweep_metrics.csv", sweep_rows)
    write_summary_json(
        run_dir / "summary.json",
        {
            "axis": axis,
            "mode": base_profile.mode,
            "parameter_file": str(parameter_file),
            "fixed_b0": float(fixed_b0),
            "fixed_wc": float(fixed_wc),
            "fixed_r": float(fixed_r),
            "recommended_k": float(best_k),
            "pid_metrics": pid_result["metrics"],
            "k_candidates": [float(value) for value in k_candidates],
        },
    )
    return WCSweepResult(
        axis=axis,
        mode=base_profile.mode,
        parameter_file=str(parameter_file),
        output_dir=str(run_dir),
        fixed_b0=float(fixed_b0),
        fixed_k=float(best_k),
        sweep_rows=tuple(sweep_rows),
        recommended_wc=float(best_k),
    )


def run_z_axis_specialized_tuning(
    parameter_file: str | Path,
    *,
    config: PyBulletControlExperimentConfig | None = None,
    output_root: str | Path = "outputs/control_pybullet_manual_tuning/z_specialized",
) -> ZAxisSpecializedTuningResult:
    cfg = config or PyBulletControlExperimentConfig(duration_sec=6.0)
    z_profile = ManualTargetProfile(
        axis="z",
        mode="z_small_step",
        step_value=0.03,
        total_duration=6.0,
        segment_durations=(2.0, 2.0, 2.0),
        hover_reference=1.0,
    )
    b0_result = run_b0_step_sweep(
        axis="z",
        parameter_file=parameter_file,
        b0_candidates=[0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
        profile=z_profile,
        config=cfg,
        output_root=output_root,
    )
    wc_result = run_wc_step_sweep(
        axis="z",
        parameter_file=parameter_file,
        fixed_b0=b0_result.recommended_b0,
        fixed_k=4.0,
        wc_candidates=[0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0],
        profile=z_profile,
        config=cfg,
        output_root=output_root,
    )
    k_rows: list[dict[str, float]] = []
    best_k = 2.0
    best_score = float("inf")
    reference_bundle = build_manual_reference_profile(z_profile, control_dt=cfg.control_dt, step_count=cfg.step_count)
    for index, k in enumerate([2.0, 3.0, 4.0, 5.0, 6.0]):
        ladrc_controller = build_single_axis_ladrc_bundle("z", parameter_file)
        ladrc_controller.set_axis_parameters("z", b0=b0_result.recommended_b0, omega_c=wc_result.recommended_wc, k=k)
        ladrc_result = run_controller_episode(cfg, ladrc_controller, reference_bundle)
        metrics = dict(ladrc_result["metrics"])
        score = _score_metrics("z", list(ladrc_result["timeseries"]), metrics)
        row = {
            "candidate_index": float(index),
            "b0": float(b0_result.recommended_b0),
            "wc": float(wc_result.recommended_wc),
            "k": float(k),
            "score": float(score),
            **{key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))},
        }
        k_rows.append(row)
        if score < best_score:
            best_score = float(score)
            best_k = float(k)
    run_dir = Path(output_root) / "z"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(run_dir / "k_sweep_metrics.csv", k_rows)
    write_summary_json(
        run_dir / "summary.json",
        {
            "axis": "z",
            "profile": z_profile.mode,
            "recommended_b0": float(b0_result.recommended_b0),
            "recommended_wc": float(wc_result.recommended_wc),
            "recommended_k": float(best_k),
        },
    )
    return ZAxisSpecializedTuningResult(
        output_dir=str(run_dir),
        recommended_b0=float(b0_result.recommended_b0),
        recommended_wc=float(wc_result.recommended_wc),
        recommended_k=float(best_k),
        b0_rows=b0_result.sweep_rows,
        wc_rows=wc_result.sweep_rows,
        k_rows=tuple(k_rows),
    )


def run_x_axis_refined_tuning(
    parameter_file: str | Path,
    *,
    config: PyBulletControlExperimentConfig | None = None,
    output_root: str | Path = "outputs/control_pybullet_manual_tuning/x_retargeted",
) -> XAxisRefinedTuningResult:
    cfg = config or PyBulletControlExperimentConfig(duration_sec=6.0)
    tuning_profile = ManualTargetProfile(
        axis="x",
        mode="hold_step_hold",
        step_value=0.12,
        total_duration=6.0,
        segment_durations=(2.0, 2.0, 2.0),
    )
    validation_profile = ManualTargetProfile(
        axis="x",
        mode="hold_step_hold_reverse",
        step_value=0.12,
        reverse_step_value=-0.12,
        total_duration=8.0,
        segment_durations=(2.0, 2.0, 2.0, 2.0),
    )
    current_params = load_axis_parameter_file(parameter_file)["x"]
    stage_root = Path(output_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    pid_reference = _evaluate_candidate(
        axis="x",
        candidate_params=current_params,
        reference_params=current_params,
        profile=tuning_profile,
        config=cfg,
        output_dir=stage_root / "baseline_reference",
    )
    current_metrics = dict(pid_reference["ladrc_metrics"])
    pid_metrics = dict(pid_reference["pid_metrics"])

    stage_a_rows = _run_refined_stage(
        axis="x",
        reference_params=current_params,
        profile=tuning_profile,
        config=cfg,
        output_dir=stage_root / "a_k_r_sweep",
        b0_candidates=[float(current_params.b0)],
        wc_candidates=[float(current_params.wc)],
        k_candidates=[6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        r_candidates=[10.0, 15.0, 20.0, 25.0, 30.0],
        pid_metrics=pid_metrics,
        current_metrics=current_metrics,
        stage_name="a_k_r",
    )
    stage_a_best = _pick_best_stage_a(stage_a_rows)

    stage_b_rows = _run_refined_stage(
        axis="x",
        reference_params=current_params,
        profile=tuning_profile,
        config=cfg,
        output_dir=stage_root / "b_wc_sweep",
        b0_candidates=[float(current_params.b0)],
        wc_candidates=[1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55],
        k_candidates=[float(stage_a_best["k"])],
        r_candidates=[float(stage_a_best["r"])],
        pid_metrics=pid_metrics,
        current_metrics=current_metrics,
        stage_name="b_wc",
    )
    stage_b_best = _pick_best_stage_b(stage_b_rows)

    stage_c_rows = _run_refined_stage(
        axis="x",
        reference_params=current_params,
        profile=tuning_profile,
        config=cfg,
        output_dir=stage_root / "c_b0_sweep",
        b0_candidates=[29.0, 30.0, 31.0, 32.0, 33.0],
        wc_candidates=[float(stage_b_best["wc"])],
        k_candidates=[float(stage_b_best["k"])],
        r_candidates=[float(stage_b_best["r"])],
        pid_metrics=pid_metrics,
        current_metrics=current_metrics,
        stage_name="c_b0",
    )
    final_candidate = _pick_final_x_candidate(stage_c_rows, stage_b_best)

    comparison_against_pid = _build_comparison(final_candidate, pid_metrics)
    comparison_against_current = _build_comparison(final_candidate, current_metrics)

    validation = _evaluate_candidate(
        axis="x",
        candidate_params=_row_to_axis_params(final_candidate),
        reference_params=current_params,
        profile=validation_profile,
        config=cfg,
        output_dir=stage_root / "final_compare" / "reverse_validation",
    )
    validation_metrics = dict(validation["ladrc_metrics"])

    final_compare = _evaluate_candidate(
        axis="x",
        candidate_params=_row_to_axis_params(final_candidate),
        reference_params=current_params,
        profile=tuning_profile,
        config=cfg,
        output_dir=stage_root / "final_compare",
    )
    final_run_dir = Path(final_compare["output_dir"])
    write_summary_json(
        stage_root / "recommended_x_params.json",
        {
            "b0": float(final_candidate["b0"]),
            "wc": float(final_candidate["wc"]),
            "k": float(final_candidate["k"]),
            "r": float(final_candidate["r"]),
            "wo": float(final_candidate["wc"] * final_candidate["k"]),
            "beats_pid": bool(
                final_candidate["rmse"] <= pid_metrics["rmse"]
                or final_candidate["steady_state_error"] <= pid_metrics["steady_state_error"]
            ),
            "beats_current_ladrc": bool(
                final_candidate["rmse"] <= current_metrics["rmse"]
                and final_candidate["steady_state_error"] <= current_metrics["steady_state_error"]
                and final_candidate["overshoot"] < current_metrics["overshoot"]
                and final_candidate["control_variation"] < current_metrics["control_variation"]
            ),
        },
    )
    write_summary_json(stage_root / "comparison_against_pid.json", comparison_against_pid)
    write_summary_json(stage_root / "comparison_against_current_ladrc.json", comparison_against_current)
    write_summary_json(
        stage_root / "final_compare" / "summary.json",
        {
            "recommended_params": {
                "b0": float(final_candidate["b0"]),
                "wc": float(final_candidate["wc"]),
                "k": float(final_candidate["k"]),
                "r": float(final_candidate["r"]),
                "wo": float(final_candidate["wc"] * final_candidate["k"]),
            },
            "pid_metrics": pid_metrics,
            "current_ladrc_metrics": current_metrics,
            "new_ladrc_metrics": dict(final_compare["ladrc_metrics"]),
            "validation_profile_metrics": validation_metrics,
            "comparison_against_pid": comparison_against_pid,
            "comparison_against_current_ladrc": comparison_against_current,
        },
    )
    return XAxisRefinedTuningResult(
        output_dir=str(stage_root),
        stage_a_rows=tuple(stage_a_rows),
        stage_b_rows=tuple(stage_b_rows),
        stage_c_rows=tuple(stage_c_rows),
        recommended_params={
            "b0": float(final_candidate["b0"]),
            "wc": float(final_candidate["wc"]),
            "k": float(final_candidate["k"]),
            "r": float(final_candidate["r"]),
            "wo": float(final_candidate["wc"] * final_candidate["k"]),
            "beats_pid": bool(comparison_against_pid["beats_reference"]),
            "beats_current_ladrc": bool(comparison_against_current["beats_reference"]),
        },
        comparison_against_pid=comparison_against_pid,
        comparison_against_current_ladrc=comparison_against_current,
    )


def run_x_axis_steady_tuning(
    parameter_file: str | Path,
    *,
    config: PyBulletControlExperimentConfig | None = None,
    output_root: str | Path = "outputs/control_pybullet_manual_tuning/x_steady_tuning",
    wc_candidates: list[float] | None = None,
    k_candidates: list[float] | None = None,
    b0_candidates: list[float] | None = None,
) -> XAxisSteadyTuningResult:
    base_cfg = config or PyBulletControlExperimentConfig(duration_sec=6.0)
    disturbed_cfg = replace(
        base_cfg,
        duration_sec=6.0,
        axis_configs=tuple(
            replace(axis_cfg, include_disturbance=True, disturbance_scale=0.12, disturbance_axis_bias=1.0)
            if axis_cfg.axis == "x"
            else axis_cfg
            for axis_cfg in base_cfg.axis_configs
        ),
    )
    quiet_cfg = replace(
        base_cfg,
        duration_sec=6.0,
        axis_configs=tuple(
            replace(axis_cfg, include_disturbance=False, disturbance_scale=0.0)
            if axis_cfg.axis == "x"
            else axis_cfg
            for axis_cfg in base_cfg.axis_configs
        ),
    )
    fast_params = load_axis_parameter_file(parameter_file)["x"]
    tuning_profile = ManualTargetProfile(
        axis="x",
        mode="x_hold_disturbance_hold",
        total_duration=6.0,
        segment_durations=(2.0, 2.0, 2.0),
    )
    validation_profile = ManualTargetProfile(
        axis="x",
        mode="x_small_step_hold",
        step_value=0.03,
        total_duration=6.0,
        segment_durations=(2.0, 2.0, 2.0),
    )
    stage_root = Path(output_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    baseline = _evaluate_candidate(
        axis="x",
        candidate_params=fast_params,
        reference_params=fast_params,
        profile=tuning_profile,
        config=disturbed_cfg,
        output_dir=stage_root / "baseline_fast",
    )
    fast_metrics = _steady_metrics("x", list(baseline["timeseries"]), baseline["reference_bundle"])
    pid_metrics = _steady_metrics("x", list(baseline["pid_timeseries"]), baseline["reference_bundle"])

    stage_a_rows = _run_steady_stage(
        axis="x",
        reference_params=fast_params,
        profile=tuning_profile,
        config=disturbed_cfg,
        output_dir=stage_root / "a_wc_sweep",
        b0_candidates=[float(fast_params.b0)],
        wc_candidates=wc_candidates or [0.6, 0.8, 1.0, 1.1, 1.2, 1.3],
        k_candidates=[float(fast_params.k)],
        r_candidates=[float(fast_params.r)],
        fast_metrics=fast_metrics,
        pid_metrics=pid_metrics,
        stage_name="a_wc",
    )
    stage_a_best = _pick_best_steady_row(stage_a_rows)

    stage_b_rows = _run_steady_stage(
        axis="x",
        reference_params=fast_params,
        profile=tuning_profile,
        config=disturbed_cfg,
        output_dir=stage_root / "b_k_sweep",
        b0_candidates=[float(fast_params.b0)],
        wc_candidates=[float(stage_a_best["wc"])],
        k_candidates=k_candidates or [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        r_candidates=[float(fast_params.r)],
        fast_metrics=fast_metrics,
        pid_metrics=pid_metrics,
        stage_name="b_k",
    )
    stage_b_best = _pick_best_steady_row(stage_b_rows)

    stage_c_rows = _run_steady_stage(
        axis="x",
        reference_params=fast_params,
        profile=tuning_profile,
        config=disturbed_cfg,
        output_dir=stage_root / "c_b0_sweep",
        b0_candidates=b0_candidates or [28.0, 29.0, 30.0, 30.5, 31.0, 32.0, 33.0],
        wc_candidates=[float(stage_b_best["wc"])],
        k_candidates=[float(stage_b_best["k"])],
        r_candidates=[float(fast_params.r)],
        fast_metrics=fast_metrics,
        pid_metrics=pid_metrics,
        stage_name="c_b0",
    )
    final_candidate = _pick_best_steady_row(stage_c_rows)
    steady_params = _row_to_axis_params_with_axis(final_candidate, "x")

    steady_main = _evaluate_candidate(
        axis="x",
        candidate_params=steady_params,
        reference_params=fast_params,
        profile=tuning_profile,
        config=disturbed_cfg,
        output_dir=stage_root / "final_compare" / "steady_main",
    )
    fast_validation = _evaluate_candidate(
        axis="x",
        candidate_params=fast_params,
        reference_params=fast_params,
        profile=validation_profile,
        config=quiet_cfg,
        output_dir=stage_root / "final_compare" / "fast_validation",
    )
    steady_validation = _evaluate_candidate(
        axis="x",
        candidate_params=steady_params,
        reference_params=fast_params,
        profile=validation_profile,
        config=quiet_cfg,
        output_dir=stage_root / "final_compare" / "steady_validation",
    )
    steady_main_metrics = _steady_metrics("x", list(steady_main["timeseries"]), steady_main["reference_bundle"])
    compare_fast = _build_steady_comparison(steady_main_metrics, fast_metrics)
    compare_pid = _build_steady_comparison(steady_main_metrics, pid_metrics)
    rl_ranges = _derive_continuous_ranges(fast_params, steady_params, expansion_ratio=0.10)

    final_compare_dir = stage_root / "final_compare"
    _write_threeway_response_svg(
        final_compare_dir / "pid_vs_fast_vs_steady_response.svg",
        axis="x",
        pid_timeseries=list(steady_validation["pid_timeseries"]),
        fast_timeseries=list(fast_validation["timeseries"]),
        steady_timeseries=list(steady_validation["timeseries"]),
    )
    write_summary_json(
        stage_root / "recommended_x_steady_params.json",
        {
            "b0": float(steady_params.b0),
            "wc": float(steady_params.wc),
            "k": float(steady_params.k),
            "r": float(steady_params.r),
            "wo": float(steady_params.wo),
            "fast_x_params": {
                "b0": float(fast_params.b0),
                "wc": float(fast_params.wc),
                "k": float(fast_params.k),
                "r": float(fast_params.r),
                "wo": float(fast_params.wo),
            },
            "steady_x_params": {
                "b0": float(steady_params.b0),
                "wc": float(steady_params.wc),
                "k": float(steady_params.k),
                "r": float(steady_params.r),
                "wo": float(steady_params.wo),
            },
            "rl_ranges": rl_ranges,
            "beats_fast_x": bool(compare_fast["beats_reference"]),
            "beats_pid": bool(compare_pid["beats_reference"]),
        },
    )
    write_summary_json(stage_root / "comparison_against_fast_x.json", compare_fast)
    write_summary_json(stage_root / "comparison_against_pid.json", compare_pid)
    write_summary_json(
        final_compare_dir / "summary.json",
        {
            "fast_x_metrics": fast_metrics,
            "steady_x_metrics": steady_main_metrics,
            "pid_metrics": pid_metrics,
            "comparison_against_fast_x": compare_fast,
            "comparison_against_pid": compare_pid,
            "rl_ranges": rl_ranges,
        },
    )
    return XAxisSteadyTuningResult(
        output_dir=str(stage_root),
        stage_a_rows=tuple(stage_a_rows),
        stage_b_rows=tuple(stage_b_rows),
        stage_c_rows=tuple(stage_c_rows),
        fast_params={
            "b0": float(fast_params.b0),
            "wc": float(fast_params.wc),
            "k": float(fast_params.k),
            "r": float(fast_params.r),
            "wo": float(fast_params.wo),
        },
        steady_params={
            "b0": float(steady_params.b0),
            "wc": float(steady_params.wc),
            "k": float(steady_params.k),
            "r": float(steady_params.r),
            "wo": float(steady_params.wo),
        },
        rl_ranges=rl_ranges,
        comparison_against_fast_x=compare_fast,
        comparison_against_pid=compare_pid,
    )


def run_x_axis_disturbed_rescan(
    parameter_file: str | Path,
    *,
    config: PyBulletControlExperimentConfig | None = None,
    output_root: str | Path = "outputs/control_pybullet_manual_tuning/x_disturbed_rescan",
    b0_candidates: list[float] | None = None,
    wc_candidates: list[float] | None = None,
    k_candidates: list[float] | None = None,
) -> XAxisDisturbedRescanResult:
    base_cfg = config or PyBulletControlExperimentConfig(duration_sec=6.0)
    disturbed_cfg = replace(
        base_cfg,
        duration_sec=6.0,
        axis_configs=tuple(
            replace(axis_cfg, include_disturbance=True, disturbance_scale=0.12, disturbance_axis_bias=1.0)
            if axis_cfg.axis == "x"
            else axis_cfg
            for axis_cfg in base_cfg.axis_configs
        ),
    )
    profile = ManualTargetProfile(
        axis="x",
        mode="hold_step_hold",
        step_value=0.12,
        total_duration=6.0,
        segment_durations=(2.0, 2.0, 2.0),
    )
    b0_values = b0_candidates or [float(value) for value in range(1, 301, 10)]
    wc_values = wc_candidates or [float(value) for value in range(1, 51, 5)]
    k_values = k_candidates or [float(value) for value in range(1, 21)]

    b0_result = run_b0_step_sweep(
        axis="x",
        parameter_file=parameter_file,
        b0_candidates=b0_values,
        profile=profile,
        config=disturbed_cfg,
        output_root=Path(output_root),
    )
    wc_result = run_wc_step_sweep(
        axis="x",
        parameter_file=parameter_file,
        fixed_b0=b0_result.recommended_b0,
        fixed_k=4.0,
        wc_candidates=wc_values,
        profile=profile,
        config=disturbed_cfg,
        output_root=Path(output_root),
    )
    reference_bundle = build_manual_reference_profile(profile, control_dt=disturbed_cfg.control_dt, step_count=disturbed_cfg.step_count)
    pid_controller = create_controller_bundle("pid_pos_att")
    pid_result = run_controller_episode(disturbed_cfg, pid_controller, reference_bundle)
    run_dir = Path(output_root) / "x" / f"{profile.mode}_k_sweep"
    run_dir.mkdir(parents=True, exist_ok=True)
    k_rows: list[dict[str, float]] = []
    best_k = float(k_values[0])
    best_score = float("inf")
    for index, k in enumerate(k_values):
        ladrc_controller = build_single_axis_ladrc_bundle("x", parameter_file)
        ladrc_controller.set_axis_parameters("x", b0=float(b0_result.recommended_b0), omega_c=float(wc_result.recommended_wc), k=float(k))
        ladrc_result = run_controller_episode(disturbed_cfg, ladrc_controller, reference_bundle)
        metrics = dict(ladrc_result["metrics"])
        score = _score_metrics("x", list(ladrc_result["timeseries"]), metrics)
        row = {
            "candidate_index": float(index),
            "b0": float(b0_result.recommended_b0),
            "wc": float(wc_result.recommended_wc),
            "k": float(k),
            "score": float(score),
            **{key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))},
        }
        k_rows.append(row)
        candidate_dir = run_dir / f"candidate_{index:02d}_k_{float(k):.6g}"
        write_timeseries_csv(candidate_dir / "ladrc_timeseries.csv", list(ladrc_result["timeseries"]))
        if score < best_score:
            best_score = float(score)
            best_k = float(k)
            plot_pid_vs_best_ladrc_response(
                list(pid_result["timeseries"]),
                list(ladrc_result["timeseries"]),
                "x",
                run_dir / "figures",
            )
    write_reference_csv(run_dir / "reference.csv", reference_bundle)
    write_timeseries_csv(run_dir / "pid_timeseries.csv", list(pid_result["timeseries"]))
    write_metrics_csv(run_dir / "k_sweep_metrics.csv", k_rows)
    write_summary_json(
        Path(output_root) / "summary.json",
        {
            "axis": "x",
            "mode": profile.mode,
            "environment": "step_with_stable_disturbance",
            "recommended_b0": float(b0_result.recommended_b0),
            "recommended_wc": float(wc_result.recommended_wc),
            "recommended_k": float(best_k),
            "b0_scan_range": [float(b0_values[0]), float(b0_values[-1])],
            "wc_scan_range": [float(wc_values[0]), float(wc_values[-1])],
            "k_scan_range": [float(k_values[0]), float(k_values[-1])],
        },
    )
    return XAxisDisturbedRescanResult(
        output_dir=str(Path(output_root)),
        recommended_b0=float(b0_result.recommended_b0),
        recommended_wc=float(wc_result.recommended_wc),
        recommended_k=float(best_k),
        b0_rows=tuple(b0_result.sweep_rows),
        wc_rows=tuple(wc_result.sweep_rows),
        k_rows=tuple(k_rows),
    )


def run_x_axis_disturbed_threeway_compare(
    parameter_file: str | Path,
    *,
    current_params: AxisLADRCParameters | None = None,
    candidate_params: AxisLADRCParameters | None = None,
    config: PyBulletControlExperimentConfig | None = None,
    output_root: str | Path = "outputs/control_pybullet_manual_tuning/x_disturbed_threeway_compare",
) -> dict[str, object]:
    base_cfg = config or PyBulletControlExperimentConfig(duration_sec=6.0)
    disturbed_cfg = replace(
        base_cfg,
        duration_sec=6.0,
        axis_configs=tuple(
            replace(axis_cfg, include_disturbance=True, disturbance_scale=0.12, disturbance_axis_bias=1.0)
            if axis_cfg.axis == "x"
            else axis_cfg
            for axis_cfg in base_cfg.axis_configs
        ),
    )
    params = load_axis_parameter_file(parameter_file)
    current = current_params or params["x"]
    candidate = candidate_params or AxisLADRCParameters(axis="x", b0=1.0, wc=8.25, k=4.0, r=12.0)
    profile = ManualTargetProfile(
        axis="x",
        mode="hold_step_hold",
        step_value=0.12,
        total_duration=6.0,
        segment_durations=(2.0, 2.0, 2.0),
    )
    reference_bundle = build_manual_reference_profile(profile, control_dt=disturbed_cfg.control_dt, step_count=disturbed_cfg.step_count)
    pid_controller = create_controller_bundle("pid_pos_att")
    pid_result = run_controller_episode(disturbed_cfg, pid_controller, reference_bundle)

    current_controller = build_single_axis_ladrc_bundle("x", parameter_file)
    current_controller.set_axis_parameters("x", b0=float(current.b0), omega_c=float(current.wc), k=float(current.k))
    current_controller.parameter_set.axis_config("x").r = float(current.r)
    if hasattr(current_controller, "_sync_from_parameter_set"):
        current_controller._sync_from_parameter_set()
    current_result = run_controller_episode(disturbed_cfg, current_controller, reference_bundle)

    candidate_controller = build_single_axis_ladrc_bundle("x", parameter_file)
    candidate_controller.set_axis_parameters("x", b0=float(candidate.b0), omega_c=float(candidate.wc), k=float(candidate.k))
    candidate_controller.parameter_set.axis_config("x").r = float(candidate.r)
    if hasattr(candidate_controller, "_sync_from_parameter_set"):
        candidate_controller._sync_from_parameter_set()
    candidate_result = run_controller_episode(disturbed_cfg, candidate_controller, reference_bundle)

    run_dir = Path(output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_reference_csv(run_dir / "reference.csv", reference_bundle)
    write_timeseries_csv(run_dir / "pid_timeseries.csv", list(pid_result["timeseries"]))
    write_timeseries_csv(run_dir / "current_ladrc_timeseries.csv", list(current_result["timeseries"]))
    write_timeseries_csv(run_dir / "candidate_ladrc_timeseries.csv", list(candidate_result["timeseries"]))
    write_metrics_csv(
        run_dir / "metrics.csv",
        [
            {"controller": "pid_pos_att", **pid_result["metrics"]},
            {"controller": "current_ladrc_x", **current_result["metrics"]},
            {"controller": "disturbed_fast_ladrc_x", **candidate_result["metrics"]},
        ],
    )
    _write_threeway_response_svg(
        run_dir / "pid_current_candidate_response.svg",
        axis="x",
        pid_timeseries=list(pid_result["timeseries"]),
        fast_timeseries=list(current_result["timeseries"]),
        steady_timeseries=list(candidate_result["timeseries"]),
    )
    summary = {
        "profile": profile.mode,
        "environment": "step_with_stable_disturbance",
        "current_params": {
            "b0": float(current.b0),
            "wc": float(current.wc),
            "k": float(current.k),
            "r": float(current.r),
            "wo": float(current.wo),
        },
        "candidate_params": {
            "b0": float(candidate.b0),
            "wc": float(candidate.wc),
            "k": float(candidate.k),
            "r": float(candidate.r),
            "wo": float(candidate.wo),
        },
        "pid_metrics": pid_result["metrics"],
        "current_ladrc_metrics": current_result["metrics"],
        "candidate_ladrc_metrics": candidate_result["metrics"],
    }
    write_summary_json(run_dir / "summary.json", summary)
    return {
        "output_dir": str(run_dir),
        "summary": summary,
    }


def run_x_axis_r_balance_scan(
    parameter_file: str | Path,
    *,
    config: PyBulletControlExperimentConfig | None = None,
    output_root: str | Path = "outputs/control_pybullet_manual_tuning/x_r_balance",
    r_candidates: list[float] | None = None,
    fast_params: AxisLADRCParameters | None = None,
    steady_params: AxisLADRCParameters | None = None,
) -> XAxisRBalanceResult:
    cfg = config or PyBulletControlExperimentConfig(duration_sec=6.0, eval_episodes=1)
    run_dir = Path(output_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    params = load_axis_parameter_file(parameter_file)
    fast = fast_params or params["x"]
    steady = steady_params or AxisLADRCParameters(axis="x", b0=30.5, wc=0.8, k=7.0, r=12.0)
    candidates = [float(value) for value in (r_candidates or [4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0])]

    fast_profile = ManualTargetProfile(
        axis="x",
        mode="hold_step_hold",
        step_value=0.12,
        total_duration=6.0,
        segment_durations=(2.0, 2.0, 2.0),
    )
    steady_profile = ManualTargetProfile(
        axis="x",
        mode="x_hold_disturbance_hold",
        total_duration=6.0,
        segment_durations=(2.0, 2.0, 2.0),
    )
    validation_profile = ManualTargetProfile(
        axis="x",
        mode="x_small_step_hold",
        step_value=0.03,
        total_duration=6.0,
        segment_durations=(2.0, 2.0, 2.0),
    )

    rows: list[dict[str, float | bool]] = []
    candidate_results: dict[float, dict[str, object]] = {}
    baseline_row: dict[str, float | bool] | None = None
    for index, r_value in enumerate(candidates):
        fast_candidate = replace(fast, r=float(r_value))
        steady_candidate = replace(steady, r=float(r_value))

        fast_eval = _evaluate_candidate(
            axis="x",
            candidate_params=fast_candidate,
            reference_params=fast,
            profile=fast_profile,
            config=cfg,
            output_dir=run_dir / "fast_stage" / f"candidate_{index:03d}",
        )
        steady_eval = _evaluate_candidate(
            axis="x",
            candidate_params=steady_candidate,
            reference_params=steady,
            profile=steady_profile,
            config=cfg,
            output_dir=run_dir / "steady_stage" / f"candidate_{index:03d}",
        )
        validation_eval = _evaluate_candidate(
            axis="x",
            candidate_params=steady_candidate,
            reference_params=steady,
            profile=validation_profile,
            config=cfg,
            output_dir=run_dir / "validation_stage" / f"candidate_{index:03d}",
        )

        fast_metrics = dict(fast_eval["ladrc_metrics"])
        steady_metrics = _steady_metrics("x", list(steady_eval["timeseries"]), steady_eval["reference_bundle"])
        validation_metrics = _steady_metrics("x", list(validation_eval["timeseries"]), validation_eval["reference_bundle"])
        row: dict[str, float | bool] = {
            "candidate_index": float(index),
            "r": float(r_value),
            "fast_rmse": float(fast_metrics["rmse"]),
            "fast_steady_state_error": float(fast_metrics["steady_state_error"]),
            "fast_overshoot": float(fast_metrics["overshoot"]),
            "fast_control_variation": float(fast_metrics["control_variation"]),
            "steady_rmse": float(steady_metrics["rmse"]),
            "steady_state_error": float(steady_metrics["steady_state_error"]),
            "steady_overshoot": float(steady_metrics["overshoot"]),
            "steady_control_variation": float(steady_metrics["control_variation"]),
            "steady_recovery_time": float(steady_metrics["disturbance_recovery_time"]),
            "validation_rmse": float(validation_metrics["rmse"]),
            "validation_steady_state_error": float(validation_metrics["steady_state_error"]),
            "validation_overshoot": float(validation_metrics["overshoot"]),
            "validation_control_variation": float(validation_metrics["control_variation"]),
        }
        rows.append(row)
        candidate_results[float(r_value)] = {
            "fast_eval": fast_eval,
            "steady_eval": steady_eval,
        }
        if abs(float(r_value) - 12.0) <= 1.0e-9:
            baseline_row = row

    if baseline_row is None:
        baseline_row = rows[0]

    baseline_fast_rmse = max(float(baseline_row["fast_rmse"]), 1.0e-9)
    baseline_fast_overshoot = max(float(baseline_row["fast_overshoot"]), 1.0e-9)
    baseline_fast_cv = max(float(baseline_row["fast_control_variation"]), 1.0e-9)
    baseline_steady_sse = max(float(baseline_row["steady_state_error"]), 1.0e-9)
    baseline_steady_cv = max(float(baseline_row["steady_control_variation"]), 1.0e-9)
    baseline_steady_recovery = max(float(baseline_row["steady_recovery_time"]), 1.0)
    baseline_validation_sse = max(float(baseline_row["validation_steady_state_error"]), 1.0e-9)

    for row in rows:
        row["fast_rmse_ratio"] = float(row["fast_rmse"]) / baseline_fast_rmse
        row["fast_overshoot_ratio"] = float(row["fast_overshoot"]) / baseline_fast_overshoot
        row["fast_cv_ratio"] = float(row["fast_control_variation"]) / baseline_fast_cv
        row["steady_sse_ratio"] = float(row["steady_state_error"]) / baseline_steady_sse
        row["steady_cv_ratio"] = float(row["steady_control_variation"]) / baseline_steady_cv
        row["steady_recovery_ratio"] = float(row["steady_recovery_time"]) / baseline_steady_recovery
        row["validation_sse_ratio"] = float(row["validation_steady_state_error"]) / baseline_validation_sse
        row["passes_balance_gate"] = bool(
            float(row["fast_rmse"]) <= baseline_fast_rmse * 1.08
            and float(row["steady_state_error"]) <= baseline_steady_sse * 1.05
            and float(row["validation_steady_state_error"]) <= baseline_validation_sse * 1.05
        )
        row["balance_score"] = (
            0.30 * float(row["fast_rmse_ratio"])
            + 0.15 * float(row["fast_overshoot_ratio"])
            + 0.10 * float(row["fast_cv_ratio"])
            + 0.20 * float(row["steady_sse_ratio"])
            + 0.10 * float(row["steady_cv_ratio"])
            + 0.10 * float(row["steady_recovery_ratio"])
            + 0.05 * float(row["validation_sse_ratio"])
        )

    ranked_rows = sorted(
        rows,
        key=lambda row: (
            not bool(row["passes_balance_gate"]),
            float(row["balance_score"]),
            float(row["steady_state_error"]),
            float(row["fast_rmse"]),
        ),
    )
    best_row = ranked_rows[0]
    best_r = float(best_row["r"])
    best_fast = candidate_results[best_r]["fast_eval"]
    best_steady = candidate_results[best_r]["steady_eval"]

    _write_threeway_response_svg(
        run_dir / "figures" / "pid_vs_fast_vs_steady_r_balance.svg",
        axis="x",
        pid_timeseries=list(best_fast["pid_timeseries"]),
        fast_timeseries=list(best_fast["timeseries"]),
        steady_timeseries=list(best_steady["timeseries"]),
    )

    rl_ranges = {
        "b0": {"min": float(min(fast.b0, steady.b0)), "max": float(max(fast.b0, steady.b0))},
        "wc": {"min": float(min(fast.wc, steady.wc)), "max": float(max(fast.wc, steady.wc))},
        "k": {"min": float(min(fast.k, steady.k)), "max": float(max(fast.k, steady.k))},
        "r": {"min": float(min(fast.r, steady.r, best_r)), "max": float(max(fast.r, steady.r, best_r))},
    }

    write_metrics_csv(run_dir / "r_balance_metrics.csv", rows)
    write_summary_json(
        run_dir / "recommended_r_balance.json",
        {
            "recommended_r": float(best_r),
            "fast_x_params": {"b0": float(fast.b0), "wc": float(fast.wc), "k": float(fast.k), "r": float(best_r), "wo": float(fast.wo)},
            "steady_x_params": {"b0": float(steady.b0), "wc": float(steady.wc), "k": float(steady.k), "r": float(best_r), "wo": float(steady.wo)},
            "rl_ranges": rl_ranges,
            "baseline_r": float(baseline_row["r"]),
        },
    )
    write_summary_json(
        run_dir / "summary.json",
        {
            "recommended_r": float(best_r),
            "baseline_r": float(baseline_row["r"]),
            "best_row": best_row,
            "candidate_count": len(rows),
        },
    )
    return XAxisRBalanceResult(
        output_dir=str(run_dir),
        fast_params={"b0": float(fast.b0), "wc": float(fast.wc), "k": float(fast.k), "r": float(best_r), "wo": float(fast.wo)},
        steady_params={"b0": float(steady.b0), "wc": float(steady.wc), "k": float(steady.k), "r": float(best_r), "wo": float(steady.wo)},
        recommended_r=float(best_r),
        sweep_rows=tuple(rows),
        rl_ranges=rl_ranges,
    )


def run_x_axis_fast_task_tuning(
    parameter_file: str | Path,
    *,
    config: PyBulletControlExperimentConfig | None = None,
    output_root: str | Path = "outputs/control_pybullet_manual_tuning/x_fast_task",
    fixed_r: float = 10.0,
) -> XAxisTaskTuningResult:
    base_cfg = config or PyBulletControlExperimentConfig(duration_sec=8.0, eval_episodes=1)
    cfg = replace(base_cfg, duration_sec=8.0)
    profile = ManualTargetProfile(
        axis="x",
        mode="hold_step_hold",
        step_value=0.12,
        total_duration=8.0,
        segment_durations=(2.0, 3.0, 3.0),
    )
    run_root = Path(output_root)
    b0_result = run_b0_step_sweep(
        axis="x",
        parameter_file=parameter_file,
        b0_candidates=[float(value) for value in range(1, 301, 10)],
        profile=profile,
        config=cfg,
        output_root=run_root,
    )
    best_b0 = float(b0_result.recommended_b0)
    wc_result = run_wc_step_sweep(
        axis="x",
        parameter_file=parameter_file,
        fixed_b0=best_b0,
        fixed_k=4.0,
        wc_candidates=[float(value) for value in range(1, 21)],
        profile=profile,
        config=cfg,
        output_root=run_root,
    )
    best_wc = float(wc_result.recommended_wc)
    k_result = run_k_step_sweep(
        axis="x",
        parameter_file=parameter_file,
        fixed_b0=best_b0,
        fixed_wc=best_wc,
        k_candidates=[float(value) for value in range(1, 16)],
        fixed_r=fixed_r,
        profile=profile,
        config=cfg,
        output_root=run_root,
    )
    best_k = float(k_result.recommended_wc)
    local_rows, best_params = _run_x_task_local_refine(
        axis="x",
        parameter_file=parameter_file,
        profile=profile,
        config=cfg,
        output_dir=run_root / "local_refine",
        base_params=AxisLADRCParameters(axis="x", b0=best_b0, wc=best_wc, k=best_k, r=fixed_r),
        scorer=_fast_task_score,
    )
    final_eval = _evaluate_candidate(
        axis="x",
        candidate_params=best_params,
        reference_params=best_params,
        profile=profile,
        config=cfg,
        output_dir=run_root / "final_compare",
    )
    recommended = {
        "b0": float(best_params.b0),
        "wc": float(best_params.wc),
        "k": float(best_params.k),
        "r": float(best_params.r),
        "wo": float(best_params.wo),
    }
    write_summary_json(run_root / "recommended_params.json", recommended)
    write_summary_json(
        run_root / "summary.json",
        {
            "task_type": "fast_tracking",
            "recommended_params": recommended,
            "pid_metrics": final_eval["pid_metrics"],
            "ladrc_metrics": final_eval["ladrc_metrics"],
        },
    )
    return XAxisTaskTuningResult(
        output_dir=str(run_root),
        task_type="fast_tracking",
        recommended_params=recommended,
        b0_rows=tuple(b0_result.sweep_rows),
        wc_rows=tuple(wc_result.sweep_rows),
        k_rows=tuple(k_result.sweep_rows),
        local_rows=tuple(local_rows),
        pid_metrics=dict(final_eval["pid_metrics"]),
        ladrc_metrics=dict(final_eval["ladrc_metrics"]),
    )


def run_x_axis_disturbance_rejection_tuning(
    parameter_file: str | Path,
    *,
    config: PyBulletControlExperimentConfig | None = None,
    output_root: str | Path = "outputs/control_pybullet_manual_tuning/x_disturbance_rejection",
    fixed_r: float = 10.0,
) -> XAxisTaskTuningResult:
    base_cfg = config or PyBulletControlExperimentConfig(duration_sec=8.0, eval_episodes=1)
    disturbed_cfg = replace(
        base_cfg,
        duration_sec=8.0,
        axis_configs=tuple(
            replace(axis_cfg, include_disturbance=True, disturbance_scale=0.05, disturbance_axis_bias=1.0)
            if axis_cfg.axis == "x"
            else axis_cfg
            for axis_cfg in base_cfg.axis_configs
        ),
    )
    profile = ManualTargetProfile(
        axis="x",
        mode="x_hold_sine_disturbance_hold",
        total_duration=8.0,
        segment_durations=(2.0, 3.0, 3.0),
    )
    run_root = Path(output_root)
    b0_result = run_b0_step_sweep(
        axis="x",
        parameter_file=parameter_file,
        b0_candidates=[float(value) for value in range(1, 301, 10)],
        profile=profile,
        config=disturbed_cfg,
        output_root=run_root,
    )
    best_b0 = float(b0_result.recommended_b0)
    wc_result = run_wc_step_sweep(
        axis="x",
        parameter_file=parameter_file,
        fixed_b0=best_b0,
        fixed_k=4.0,
        wc_candidates=[float(value) for value in range(1, 21)],
        profile=profile,
        config=disturbed_cfg,
        output_root=run_root,
    )
    best_wc = float(wc_result.recommended_wc)
    k_result = run_k_step_sweep(
        axis="x",
        parameter_file=parameter_file,
        fixed_b0=best_b0,
        fixed_wc=best_wc,
        k_candidates=[float(value) for value in range(1, 16)],
        fixed_r=fixed_r,
        profile=profile,
        config=disturbed_cfg,
        output_root=run_root,
    )
    best_k = float(k_result.recommended_wc)
    local_rows, best_params = _run_x_task_local_refine(
        axis="x",
        parameter_file=parameter_file,
        profile=profile,
        config=disturbed_cfg,
        output_dir=run_root / "local_refine",
        base_params=AxisLADRCParameters(axis="x", b0=best_b0, wc=best_wc, k=best_k, r=fixed_r),
        scorer=_disturbance_task_score,
    )
    final_eval = _evaluate_candidate(
        axis="x",
        candidate_params=best_params,
        reference_params=best_params,
        profile=profile,
        config=disturbed_cfg,
        output_dir=run_root / "final_compare",
    )
    recommended = {
        "b0": float(best_params.b0),
        "wc": float(best_params.wc),
        "k": float(best_params.k),
        "r": float(best_params.r),
        "wo": float(best_params.wo),
    }
    write_summary_json(run_root / "recommended_params.json", recommended)
    write_summary_json(
        run_root / "summary.json",
        {
            "task_type": "disturbance_rejection",
            "recommended_params": recommended,
            "pid_metrics": final_eval["pid_metrics"],
            "ladrc_metrics": final_eval["ladrc_metrics"],
        },
    )
    return XAxisTaskTuningResult(
        output_dir=str(run_root),
        task_type="disturbance_rejection",
        recommended_params=recommended,
        b0_rows=tuple(b0_result.sweep_rows),
        wc_rows=tuple(wc_result.sweep_rows),
        k_rows=tuple(k_result.sweep_rows),
        local_rows=tuple(local_rows),
        pid_metrics=dict(final_eval["pid_metrics"]),
        ladrc_metrics=dict(final_eval["ladrc_metrics"]),
    )


def run_x_axis_disturbance_rejection_refined_tuning(
    parameter_file: str | Path,
    *,
    config: PyBulletControlExperimentConfig | None = None,
    output_root: str | Path = "outputs/control_pybullet_manual_tuning/x_disturbance_rejection_refined",
    fixed_r: float = 10.0,
    b0_candidates: list[float] | None = None,
    wc_candidates: list[float] | None = None,
    k_candidates: list[float] | None = None,
) -> XAxisDisturbanceRefinedTuningResult:
    base_cfg = config or PyBulletControlExperimentConfig(duration_sec=8.0, eval_episodes=1)
    disturbed_cfg = replace(
        base_cfg,
        duration_sec=8.0,
        axis_configs=tuple(
            replace(axis_cfg, include_disturbance=True, disturbance_scale=0.05, disturbance_axis_bias=1.0)
            if axis_cfg.axis == "x"
            else axis_cfg
            for axis_cfg in base_cfg.axis_configs
        ),
    )
    profile = ManualTargetProfile(
        axis="x",
        mode="x_hold_sine_disturbance_hold",
        total_duration=8.0,
        segment_durations=(2.0, 3.0, 3.0),
    )
    current_params = AxisLADRCParameters(axis="x", b0=0.8, wc=6.8, k=4.8, r=fixed_r)
    run_root = Path(output_root)
    run_root.mkdir(parents=True, exist_ok=True)

    current_eval = _evaluate_candidate(
        axis="x",
        candidate_params=current_params,
        reference_params=current_params,
        profile=profile,
        config=disturbed_cfg,
        output_dir=run_root / "baseline_current",
    )
    current_metrics = _merge_disturbance_metrics(
        list(current_eval["timeseries"]),
        current_eval["reference_bundle"],
        dict(current_eval["ladrc_metrics"]),
    )
    pid_metrics = _merge_disturbance_metrics(
        list(current_eval["pid_timeseries"]),
        current_eval["reference_bundle"],
        dict(current_eval["pid_metrics"]),
    )

    stage_a_rows = _run_disturbance_refined_stage(
        axis="x",
        reference_params=current_params,
        profile=profile,
        config=disturbed_cfg,
        output_dir=run_root / "a_b0_sweep",
        b0_candidates=b0_candidates or [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
        wc_candidates=[float(current_params.wc)],
        k_candidates=[float(current_params.k)],
        fixed_r=fixed_r,
        current_metrics=current_metrics,
        stage_name="a_b0",
    )
    stage_a_best = _pick_best_disturbance_refined(stage_a_rows)

    stage_b_rows = _run_disturbance_refined_stage(
        axis="x",
        reference_params=current_params,
        profile=profile,
        config=disturbed_cfg,
        output_dir=run_root / "b_wc_sweep",
        b0_candidates=[float(stage_a_best["b0"])],
        wc_candidates=wc_candidates or [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
        k_candidates=[float(current_params.k)],
        fixed_r=fixed_r,
        current_metrics=current_metrics,
        stage_name="b_wc",
    )
    stage_b_best = _pick_best_disturbance_refined(stage_b_rows)

    stage_c_rows = _run_disturbance_refined_stage(
        axis="x",
        reference_params=current_params,
        profile=profile,
        config=disturbed_cfg,
        output_dir=run_root / "c_k_sweep",
        b0_candidates=[float(stage_b_best["b0"])],
        wc_candidates=[float(stage_b_best["wc"])],
        k_candidates=k_candidates or [4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0],
        fixed_r=fixed_r,
        current_metrics=current_metrics,
        stage_name="c_k",
    )
    stage_c_best = _pick_best_disturbance_refined(stage_c_rows)

    local_rows, best_params = _run_x_task_local_refine(
        axis="x",
        parameter_file=parameter_file,
        profile=profile,
        config=disturbed_cfg,
        output_dir=run_root / "local_refine",
        base_params=AxisLADRCParameters(
            axis="x",
            b0=float(stage_c_best["b0"]),
            wc=float(stage_c_best["wc"]),
            k=float(stage_c_best["k"]),
            r=fixed_r,
        ),
        scorer=_disturbance_refined_score,
    )

    final_eval = _evaluate_candidate(
        axis="x",
        candidate_params=best_params,
        reference_params=current_params,
        profile=profile,
        config=disturbed_cfg,
        output_dir=run_root / "final_compare",
    )
    final_metrics = _merge_disturbance_metrics(
        list(final_eval["timeseries"]),
        final_eval["reference_bundle"],
        dict(final_eval["ladrc_metrics"]),
    )
    comparison_against_current = _build_refined_disturbance_comparison(final_metrics, current_metrics)
    comparison_against_pid = _build_refined_disturbance_comparison(final_metrics, pid_metrics)

    recommended = {
        "b0": float(best_params.b0),
        "wc": float(best_params.wc),
        "k": float(best_params.k),
        "r": float(best_params.r),
        "wo": float(best_params.wo),
    }
    write_summary_json(run_root / "recommended_params.json", recommended)
    write_summary_json(run_root / "comparison_against_current.json", comparison_against_current)
    write_summary_json(
        run_root / "summary.json",
        {
            "task_type": "disturbance_rejection_refined",
            "recommended_params": recommended,
            "pid_metrics": pid_metrics,
            "current_ladrc_metrics": current_metrics,
            "ladrc_metrics": final_metrics,
            "comparison_against_current": comparison_against_current,
            "comparison_against_pid": comparison_against_pid,
        },
    )
    return XAxisDisturbanceRefinedTuningResult(
        output_dir=str(run_root),
        recommended_params=recommended,
        b0_rows=tuple(stage_a_rows),
        wc_rows=tuple(stage_b_rows),
        k_rows=tuple(stage_c_rows),
        local_rows=tuple(local_rows),
        pid_metrics=pid_metrics,
        ladrc_metrics=final_metrics,
        comparison_against_current=comparison_against_current,
    )


def _score_metrics(axis: str, timeseries: list[dict[str, float]], metrics: dict[str, float]) -> float:
    if axis != "z":
        return (
            0.35 * float(metrics["steady_state_error"])
            + 0.25 * float(metrics["rmse"])
            + 0.15 * float(metrics["iae"])
            + 0.10 * float(metrics["overshoot"])
            + 0.10 * float(metrics["velocity_rmse"])
            + 0.05 * float(metrics["control_variation"])
        )
    z_values = [float(row["z"]) for row in timeseries]
    target_values = [float(row["target_z"]) for row in timeseries]
    hover_errors = [abs(z - target) for z, target in zip(z_values, target_values)]
    hover_error = sum(hover_errors) / max(len(hover_errors), 1)
    min_z_violation = max(0.0, 0.95 - min(z_values)) if z_values else 0.0
    if min_z_violation > 0.0:
        return 1.0e6 + min_z_violation * 1.0e4
    return (
        0.35 * hover_error
        + 0.25 * float(metrics["steady_state_error"])
        + 0.20 * float(metrics["rmse"])
        + 0.10 * float(metrics["overshoot"])
        + 0.10 * float(metrics["control_variation"])
    )


def _steady_metrics(axis: str, timeseries: list[dict[str, float]], reference_bundle) -> dict[str, float]:
    axis_key = axis
    target_key = f"target_{axis}"
    errors = [float(row[target_key] - row[axis_key]) for row in timeseries]
    rpm_mean = [float((row["rpm0"] + row["rpm1"] + row["rpm2"] + row["rpm3"]) / 4.0) for row in timeseries]
    stage_slices = reference_bundle.stage_slices or (slice(0, len(timeseries)),)
    final_slice = stage_slices[-1]
    final_errors = errors[final_slice]
    final_rpms = rpm_mean[final_slice]
    steady_state_error = float(sum(abs(value) for value in final_errors) / max(len(final_errors), 1))
    control_variation = 0.0
    if len(final_rpms) > 1:
        control_variation = float(
            sum(abs(final_rpms[index] - final_rpms[index - 1]) for index in range(1, len(final_rpms))) / (len(final_rpms) - 1)
        )
    overshoot = float(max(abs(value) for value in errors)) if errors else 0.0
    rmse = float((sum(value * value for value in errors) / max(len(errors), 1)) ** 0.5)
    disturbance_recovery_time = 0.0
    if len(stage_slices) >= 3 and final_slice.start < len(errors):
        threshold = max(steady_state_error * 1.25, 0.005)
        recovery_errors = [abs(value) for value in errors[final_slice]]
        recovery_index = len(recovery_errors) - 1
        for index, value in enumerate(recovery_errors):
            if value <= threshold:
                recovery_index = index
                break
        time_axis = [float(row["time"]) for row in timeseries]
        start_time = time_axis[final_slice.start]
        disturbance_recovery_time = float(time_axis[min(final_slice.start + recovery_index, len(time_axis) - 1)] - start_time)
    return {
        "rmse": rmse,
        "steady_state_error": steady_state_error,
        "overshoot": overshoot,
        "control_variation": control_variation,
        "disturbance_recovery_time": disturbance_recovery_time,
    }


def _final_stage_disturbance_metrics(axis: str, timeseries: list[dict[str, float]], reference_bundle) -> dict[str, float]:
    axis_key = axis
    target_key = f"target_{axis}"
    errors = [float(row[target_key] - row[axis_key]) for row in timeseries]
    stage_slices = reference_bundle.stage_slices or (slice(0, len(timeseries)),)
    final_slice = stage_slices[-1]
    final_errors = errors[final_slice]
    if not final_errors:
        return {
            "final_error_std": 0.0,
            "final_error_span": 0.0,
        }
    mean_error = sum(final_errors) / len(final_errors)
    variance = sum((value - mean_error) ** 2 for value in final_errors) / len(final_errors)
    return {
        "final_error_std": float(variance**0.5),
        "final_error_span": float(max(final_errors) - min(final_errors)),
    }


def _merge_disturbance_metrics(timeseries: list[dict[str, float]], reference_bundle, metrics: dict[str, float]) -> dict[str, float]:
    merged = dict(metrics)
    merged.update(_steady_metrics("x", timeseries, reference_bundle))
    merged.update(_final_stage_disturbance_metrics("x", timeseries, reference_bundle))
    return merged


def _evaluate_candidate(
    *,
    axis: str,
    candidate_params,
    reference_params,
    profile: ManualTargetProfile,
    config: PyBulletControlExperimentConfig,
    output_dir: Path,
) -> dict[str, object]:
    reference_bundle = build_manual_reference_profile(profile, control_dt=config.control_dt, step_count=config.step_count)
    pid_controller = create_controller_bundle("pid_pos_att")
    candidate_bundle = create_controller_bundle(f"ladrc_{axis}_pos_pid_att")
    candidate_bundle.set_axis_parameters(axis, b0=float(candidate_params.b0), omega_c=float(candidate_params.wc), k=float(candidate_params.k))
    candidate_bundle.parameter_set.axis_config(axis).r = float(candidate_params.r)
    if hasattr(candidate_bundle, "_sync_from_parameter_set"):
        candidate_bundle._sync_from_parameter_set()

    pid_result = run_controller_episode(config, pid_controller, reference_bundle)
    ladrc_result = run_controller_episode(config, candidate_bundle, reference_bundle)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_reference_csv(output_dir / "reference.csv", reference_bundle)
    write_timeseries_csv(output_dir / "pid_timeseries.csv", list(pid_result["timeseries"]))
    write_timeseries_csv(output_dir / "ladrc_timeseries.csv", list(ladrc_result["timeseries"]))
    write_metrics_csv(
        output_dir / "metrics.csv",
        [
            {"controller": "pid_pos_att", **pid_result["metrics"]},
            {"controller": f"ladrc_{axis}_pos_pid_att", **ladrc_result["metrics"]},
        ],
    )
    figures_dir = output_dir / "figures"
    plot_pid_vs_best_ladrc_response(list(pid_result["timeseries"]), list(ladrc_result["timeseries"]), axis, figures_dir)
    plot_axis_tracking(list(ladrc_result["timeseries"]), figures_dir)
    plot_axis_velocity(list(ladrc_result["timeseries"]), figures_dir)
    plot_axis_error(list(ladrc_result["timeseries"]), figures_dir)
    write_summary_json(
        output_dir / "summary.json",
        {
            "axis": axis,
            "mode": profile.mode,
            "candidate_params": {
                "b0": float(candidate_params.b0),
                "wc": float(candidate_params.wc),
                "k": float(candidate_params.k),
                "r": float(candidate_params.r),
                "wo": float(candidate_params.wo),
            },
            "reference_params": {
                "b0": float(reference_params.b0),
                "wc": float(reference_params.wc),
                "k": float(reference_params.k),
                "r": float(reference_params.r),
                "wo": float(reference_params.wo),
            },
            "pid_metrics": pid_result["metrics"],
            "ladrc_metrics": ladrc_result["metrics"],
        },
    )
    return {
        "output_dir": str(output_dir),
        "pid_metrics": dict(pid_result["metrics"]),
        "ladrc_metrics": dict(ladrc_result["metrics"]),
        "pid_timeseries": list(pid_result["timeseries"]),
        "timeseries": list(ladrc_result["timeseries"]),
        "reference_bundle": reference_bundle,
    }


def _run_x_task_local_refine(
    *,
    axis: str,
    parameter_file: str | Path,
    profile: ManualTargetProfile,
    config: PyBulletControlExperimentConfig,
    output_dir: Path,
    base_params: AxisLADRCParameters,
    scorer,
) -> tuple[list[dict[str, float]], AxisLADRCParameters]:
    output_dir.mkdir(parents=True, exist_ok=True)
    b0_candidates = _local_candidates(float(base_params.b0), 0.2, minimum=0.5)
    wc_candidates = _local_candidates(float(base_params.wc), 0.15, minimum=0.1)
    k_candidates = _local_candidates(float(base_params.k), 0.2, minimum=0.5)
    rows: list[dict[str, float]] = []
    best_score = float("inf")
    best_params = base_params
    best_eval: dict[str, object] | None = None
    index = 0
    for b0 in b0_candidates:
        for wc in wc_candidates:
            for k in k_candidates:
                candidate = AxisLADRCParameters(axis=axis, b0=float(b0), wc=float(wc), k=float(k), r=float(base_params.r))
                evaluated = _evaluate_candidate(
                    axis=axis,
                    candidate_params=candidate,
                    reference_params=base_params,
                    profile=profile,
                    config=config,
                    output_dir=output_dir / f"candidate_{index:03d}",
                )
                candidate_metrics = dict(evaluated["ladrc_metrics"])
                if axis == "x":
                    candidate_metrics = _merge_disturbance_metrics(
                        list(evaluated["timeseries"]),
                        evaluated["reference_bundle"],
                        candidate_metrics,
                    )
                score = float(scorer(list(evaluated["timeseries"]), evaluated["reference_bundle"], candidate_metrics))
                row = {
                    "candidate_index": float(index),
                    "b0": float(b0),
                    "wc": float(wc),
                    "k": float(k),
                    "r": float(base_params.r),
                    "score": score,
                    **{key: float(value) for key, value in candidate_metrics.items() if isinstance(value, (int, float))},
                }
                rows.append(row)
                if score < best_score:
                    best_score = score
                    best_params = candidate
                    best_eval = evaluated
                index += 1
    write_metrics_csv(output_dir / "local_refine_metrics.csv", rows)
    if best_eval is not None:
        plot_pid_vs_best_ladrc_response(
            list(best_eval["pid_timeseries"]),
            list(best_eval["timeseries"]),
            axis,
            output_dir / "figures",
        )
        write_summary_json(
            output_dir / "summary.json",
            {
                "recommended_params": {
                    "b0": float(best_params.b0),
                    "wc": float(best_params.wc),
                    "k": float(best_params.k),
                    "r": float(best_params.r),
                    "wo": float(best_params.wo),
                },
                "pid_metrics": best_eval["pid_metrics"],
                "ladrc_metrics": best_eval["ladrc_metrics"],
            },
        )
    return rows, best_params


def _local_candidates(value: float, span_ratio: float, *, minimum: float) -> list[float]:
    delta = max(abs(value) * span_ratio, minimum * 0.2)
    candidates = [max(minimum, value - delta), value, max(minimum, value + delta)]
    unique = sorted({round(float(candidate), 6) for candidate in candidates})
    return [float(candidate) for candidate in unique]


def _fast_task_score(timeseries: list[dict[str, float]], reference_bundle, metrics: dict[str, float]) -> float:
    steady_metrics = _steady_metrics("x", timeseries, reference_bundle)
    return (
        0.30 * float(metrics["rmse"])
        + 0.25 * float(steady_metrics["steady_state_error"])
        + 0.20 * float(metrics["overshoot"])
        + 0.15 * float(metrics["control_variation"])
        + 0.10 * float(metrics["settling_time"]) / max(len(timeseries), 1)
    )


def _disturbance_task_score(timeseries: list[dict[str, float]], reference_bundle, metrics: dict[str, float]) -> float:
    merged = _merge_disturbance_metrics(timeseries, reference_bundle, metrics)
    return _disturbance_refined_score(timeseries, reference_bundle, merged)


def _disturbance_refined_score(timeseries: list[dict[str, float]], reference_bundle, metrics: dict[str, float]) -> float:
    del timeseries, reference_bundle
    return (
        0.35 * float(metrics["steady_state_error"])
        + 0.25 * float(metrics["final_error_std"])
        + 0.20 * float(metrics["final_error_span"])
        + 0.15 * float(metrics["control_variation"]) / 1000.0
        + 0.05 * float(metrics["rmse"])
    )


def _run_disturbance_refined_stage(
    *,
    axis: str,
    reference_params,
    profile: ManualTargetProfile,
    config: PyBulletControlExperimentConfig,
    output_dir: Path,
    b0_candidates: list[float],
    wc_candidates: list[float],
    k_candidates: list[float],
    fixed_r: float,
    current_metrics: dict[str, float],
    stage_name: str,
) -> list[dict[str, float | bool | str]]:
    rows: list[dict[str, float | bool | str]] = []
    candidate_index = 0
    for b0 in b0_candidates:
        for wc in wc_candidates:
            for k in k_candidates:
                candidate = type(reference_params)(axis=axis, b0=float(b0), wc=float(wc), k=float(k), r=float(fixed_r))
                candidate_dir = output_dir / f"candidate_{candidate_index:03d}"
                evaluated = _evaluate_candidate(
                    axis=axis,
                    candidate_params=candidate,
                    reference_params=reference_params,
                    profile=profile,
                    config=config,
                    output_dir=candidate_dir,
                )
                metrics = _merge_disturbance_metrics(
                    list(evaluated["timeseries"]),
                    evaluated["reference_bundle"],
                    dict(evaluated["ladrc_metrics"]),
                )
                row: dict[str, float | bool | str] = {
                    "candidate_index": float(candidate_index),
                    "b0": float(b0),
                    "wc": float(wc),
                    "k": float(k),
                    "r": float(fixed_r),
                    "wo": float(wc * k),
                    "score": _disturbance_refined_score(list(evaluated["timeseries"]), evaluated["reference_bundle"], metrics),
                    **metrics,
                    "beats_current": (
                        float(metrics["steady_state_error"]) < float(current_metrics["steady_state_error"])
                        and float(metrics["control_variation"]) < float(current_metrics["control_variation"])
                        and float(metrics["final_error_std"]) < float(current_metrics["final_error_std"])
                    ),
                    "output_dir": str(candidate_dir),
                }
                rows.append(row)
                candidate_index += 1
    write_metrics_csv(output_dir / "search.csv", rows)
    top_candidates = sorted(rows, key=lambda row: (
        not bool(row["beats_current"]),
        float(row["steady_state_error"]),
        float(row["final_error_std"]),
        float(row["final_error_span"]),
        float(row["control_variation"]),
        float(row["rmse"]),
    ))[:10]
    write_summary_json(
        output_dir / "summary.json",
        {
            "stage": stage_name,
            "candidate_count": len(rows),
            "top_candidate": top_candidates[0] if top_candidates else None,
            "current_metrics": current_metrics,
        },
    )
    write_summary_json(output_dir / "top_candidates.json", {"top_candidates": top_candidates})
    return rows


def _pick_best_disturbance_refined(rows: list[dict[str, float | bool | str]]) -> dict[str, float | bool | str]:
    valid = [row for row in rows if bool(row["beats_current"])]
    if not valid:
        valid = rows
    return sorted(valid, key=lambda row: (
        float(row["steady_state_error"]),
        float(row["final_error_std"]),
        float(row["final_error_span"]),
        float(row["control_variation"]),
        float(row["rmse"]),
    ))[0]


def _build_refined_disturbance_comparison(
    candidate_metrics: dict[str, float],
    reference_metrics: dict[str, float],
) -> dict[str, float | bool]:
    return {
        "steady_state_error_delta": float(candidate_metrics["steady_state_error"]) - float(reference_metrics["steady_state_error"]),
        "final_error_std_delta": float(candidate_metrics["final_error_std"]) - float(reference_metrics["final_error_std"]),
        "final_error_span_delta": float(candidate_metrics["final_error_span"]) - float(reference_metrics["final_error_span"]),
        "control_variation_delta": float(candidate_metrics["control_variation"]) - float(reference_metrics["control_variation"]),
        "rmse_delta": float(candidate_metrics["rmse"]) - float(reference_metrics["rmse"]),
        "beats_reference": (
            float(candidate_metrics["steady_state_error"]) < float(reference_metrics["steady_state_error"])
            and float(candidate_metrics["final_error_std"]) < float(reference_metrics["final_error_std"])
            and float(candidate_metrics["control_variation"]) < float(reference_metrics["control_variation"])
        ),
    }


def _run_refined_stage(
    *,
    axis: str,
    reference_params,
    profile: ManualTargetProfile,
    config: PyBulletControlExperimentConfig,
    output_dir: Path,
    b0_candidates: list[float],
    wc_candidates: list[float],
    k_candidates: list[float],
    r_candidates: list[float],
    pid_metrics: dict[str, float],
    current_metrics: dict[str, float],
    stage_name: str,
) -> list[dict[str, float | bool | str]]:
    rows: list[dict[str, float | bool | str]] = []
    candidate_index = 0
    for b0 in b0_candidates:
        for wc in wc_candidates:
            for k in k_candidates:
                for r in r_candidates:
                    candidate = type(reference_params)(axis=axis, b0=float(b0), wc=float(wc), k=float(k), r=float(r))
                    candidate_dir = output_dir / f"candidate_{candidate_index:03d}"
                    evaluated = _evaluate_candidate(
                        axis=axis,
                        candidate_params=candidate,
                        reference_params=reference_params,
                        profile=profile,
                        config=config,
                        output_dir=candidate_dir,
                    )
                    metrics = dict(evaluated["ladrc_metrics"])
                    row: dict[str, float | bool | str] = {
                        "candidate_index": float(candidate_index),
                        "b0": float(b0),
                        "wc": float(wc),
                        "k": float(k),
                        "r": float(r),
                        "wo": float(wc * k),
                        "rmse": float(metrics["rmse"]),
                        "steady_state_error": float(metrics["steady_state_error"]),
                        "overshoot": float(metrics["overshoot"]),
                        "control_variation": float(metrics["control_variation"]),
                        "settling_time": float(metrics["settling_time"]),
                        "velocity_rmse": float(metrics["velocity_rmse"]),
                        "meets_stage_a_gate": _meets_stage_a_gate(metrics, current_metrics),
                        "meets_stage_b_gate": _meets_stage_b_gate(metrics, current_metrics),
                        "meets_stage_c_gate": _meets_stage_c_gate(metrics, current_metrics),
                        "meets_pid_priority": _meets_pid_priority(metrics, pid_metrics),
                        "output_dir": str(candidate_dir),
                    }
                    row["ranking_key_rmse"] = float(row["rmse"])
                    row["ranking_key_sse"] = float(row["steady_state_error"])
                    row["ranking_key_overshoot"] = float(row["overshoot"])
                    row["ranking_key_cv"] = float(row["control_variation"])
                    row["ranking_key_vrmse"] = float(row["velocity_rmse"])
                    rows.append(row)
                    candidate_index += 1

    write_metrics_csv(output_dir / "search.csv", rows)
    top_candidates = sorted(rows, key=_row_rank_key)[:10]
    write_summary_json(
        output_dir / "summary.json",
        {
            "stage": stage_name,
            "candidate_count": len(rows),
            "top_candidate": top_candidates[0] if top_candidates else None,
            "pid_metrics": pid_metrics,
            "current_ladrc_metrics": current_metrics,
        },
    )
    write_summary_json(output_dir / "top_candidates.json", {"top_candidates": top_candidates})
    return rows


def _row_rank_key(row: dict[str, float | bool | str]) -> tuple:
    gate_priority = not bool(row["meets_stage_c_gate"] or row["meets_stage_b_gate"] or row["meets_stage_a_gate"])
    pid_priority = not bool(row["meets_pid_priority"])
    return (
        gate_priority,
        pid_priority,
        float(row["ranking_key_rmse"]),
        float(row["ranking_key_sse"]),
        float(row["ranking_key_overshoot"]),
        float(row["ranking_key_cv"]),
        float(row["ranking_key_vrmse"]),
    )


def _meets_stage_a_gate(metrics: dict[str, float], current_metrics: dict[str, float]) -> bool:
    return (
        float(metrics["overshoot"]) < float(current_metrics["overshoot"])
        and float(metrics["control_variation"]) < float(current_metrics["control_variation"])
        and float(metrics["settling_time"]) <= float(current_metrics["settling_time"]) * 1.05
        and float(metrics["rmse"]) <= float(current_metrics["rmse"]) * 1.02
    )


def _meets_stage_b_gate(metrics: dict[str, float], current_metrics: dict[str, float]) -> bool:
    return (
        float(metrics["rmse"]) <= 0.02897
        and float(metrics["steady_state_error"]) <= 0.00445 * 1.10
        and float(metrics["overshoot"]) < float(current_metrics["overshoot"])
        and float(metrics["control_variation"]) <= float(current_metrics["control_variation"])
    )


def _meets_stage_c_gate(metrics: dict[str, float], current_metrics: dict[str, float]) -> bool:
    return (
        float(metrics["rmse"]) <= float(current_metrics["rmse"])
        and float(metrics["steady_state_error"]) <= float(current_metrics["steady_state_error"])
        and float(metrics["overshoot"]) < float(current_metrics["overshoot"])
        and float(metrics["control_variation"]) < float(current_metrics["control_variation"])
    )


def _meets_pid_priority(metrics: dict[str, float], pid_metrics: dict[str, float]) -> bool:
    return (
        float(metrics["rmse"]) <= float(pid_metrics["rmse"])
        and float(metrics["steady_state_error"]) <= float(pid_metrics["steady_state_error"])
        and float(metrics["control_variation"]) <= float(pid_metrics["control_variation"])
    )


def _pick_best_stage_a(rows: list[dict[str, float | bool | str]]) -> dict[str, float | bool | str]:
    valid = [row for row in rows if bool(row["meets_stage_a_gate"])]
    if not valid:
        valid = rows
    return sorted(valid, key=_row_rank_key)[0]


def _pick_best_stage_b(rows: list[dict[str, float | bool | str]]) -> dict[str, float | bool | str]:
    valid = [row for row in rows if bool(row["meets_stage_b_gate"])]
    if not valid:
        valid = [row for row in rows if bool(row["meets_stage_a_gate"])]
    if not valid:
        valid = rows
    return sorted(valid, key=_row_rank_key)[0]


def _pick_final_x_candidate(
    rows: list[dict[str, float | bool | str]],
    fallback: dict[str, float | bool | str],
) -> dict[str, float | bool | str]:
    valid = [row for row in rows if bool(row["meets_stage_c_gate"])]
    if valid:
        return sorted(valid, key=_row_rank_key)[0]

    backup = [
        row for row in rows
        if float(row["rmse"]) <= 0.02897
        and float(row["overshoot"]) < 0.02789
        and float(row["control_variation"]) < 1.91944
        and float(row["steady_state_error"]) <= 0.00445 * 1.05
    ]
    if backup:
        return sorted(backup, key=_row_rank_key)[0]
    return fallback


def _row_to_axis_params(row: dict[str, float | bool | str]):
    from control.Tuning_ladrc.schemas import AxisLADRCParameters

    return AxisLADRCParameters(
        axis="x",
        b0=float(row["b0"]),
        wc=float(row["wc"]),
        k=float(row["k"]),
        r=float(row["r"]),
    )


def _row_to_axis_params_with_axis(row: dict[str, float | bool | str], axis: str):
    from control.Tuning_ladrc.schemas import AxisLADRCParameters

    return AxisLADRCParameters(
        axis=axis,
        b0=float(row["b0"]),
        wc=float(row["wc"]),
        k=float(row["k"]),
        r=float(row["r"]),
    )


def _build_comparison(candidate_row: dict[str, float | bool | str], reference_metrics: dict[str, float]) -> dict[str, float | bool]:
    rmse = float(candidate_row["rmse"])
    sse = float(candidate_row["steady_state_error"])
    overshoot = float(candidate_row["overshoot"])
    cv = float(candidate_row["control_variation"])
    return {
        "rmse_delta": rmse - float(reference_metrics["rmse"]),
        "steady_state_error_delta": sse - float(reference_metrics["steady_state_error"]),
        "overshoot_delta": overshoot - float(reference_metrics["overshoot"]),
        "control_variation_delta": cv - float(reference_metrics["control_variation"]),
        "beats_reference": (
            rmse <= float(reference_metrics["rmse"])
            and sse <= float(reference_metrics["steady_state_error"])
            and overshoot < float(reference_metrics["overshoot"])
            and cv < float(reference_metrics["control_variation"])
        ),
    }


def _run_steady_stage(
    *,
    axis: str,
    reference_params,
    profile: ManualTargetProfile,
    config: PyBulletControlExperimentConfig,
    output_dir: Path,
    b0_candidates: list[float],
    wc_candidates: list[float],
    k_candidates: list[float],
    r_candidates: list[float],
    fast_metrics: dict[str, float],
    pid_metrics: dict[str, float],
    stage_name: str,
) -> list[dict[str, float | bool | str]]:
    rows: list[dict[str, float | bool | str]] = []
    candidate_index = 0
    for b0 in b0_candidates:
        for wc in wc_candidates:
            for k in k_candidates:
                for r in r_candidates:
                    candidate = type(reference_params)(axis=axis, b0=float(b0), wc=float(wc), k=float(k), r=float(r))
                    candidate_dir = output_dir / f"candidate_{candidate_index:03d}"
                    evaluated = _evaluate_candidate(
                        axis=axis,
                        candidate_params=candidate,
                        reference_params=reference_params,
                        profile=profile,
                        config=config,
                        output_dir=candidate_dir,
                    )
                    metrics = _steady_metrics(axis, list(evaluated["timeseries"]), evaluated["reference_bundle"])
                    row: dict[str, float | bool | str] = {
                        "candidate_index": float(candidate_index),
                        "b0": float(b0),
                        "wc": float(wc),
                        "k": float(k),
                        "r": float(r),
                        "wo": float(wc * k),
                        **metrics,
                        "passes_fast_gate": (
                            metrics["steady_state_error"] < float(fast_metrics["steady_state_error"])
                            and metrics["control_variation"] < float(fast_metrics["control_variation"])
                            and metrics["overshoot"] <= float(fast_metrics["overshoot"])
                            and metrics["disturbance_recovery_time"] <= float(fast_metrics["disturbance_recovery_time"]) * 1.10 + 1.0e-9
                        ),
                        "passes_pid_priority": (
                            metrics["steady_state_error"] <= float(pid_metrics["steady_state_error"])
                            or metrics["rmse"] <= float(pid_metrics["rmse"])
                        ),
                        "output_dir": str(candidate_dir),
                    }
                    rows.append(row)
                    candidate_index += 1
    write_metrics_csv(output_dir / "search.csv", rows)
    top_candidates = sorted(rows, key=_steady_rank_key)[:10]
    write_summary_json(
        output_dir / "summary.json",
        {
            "stage": stage_name,
            "candidate_count": len(rows),
            "top_candidate": top_candidates[0] if top_candidates else None,
            "fast_metrics": fast_metrics,
            "pid_metrics": pid_metrics,
        },
    )
    write_summary_json(output_dir / "top_candidates.json", {"top_candidates": top_candidates})
    return rows


def _steady_rank_key(row: dict[str, float | bool | str]) -> tuple:
    return (
        not bool(row["passes_fast_gate"]),
        float(row["steady_state_error"]),
        float(row["control_variation"]),
        float(row["disturbance_recovery_time"]),
        float(row["overshoot"]),
        float(row["rmse"]),
    )


def _pick_best_steady_row(rows: list[dict[str, float | bool | str]]) -> dict[str, float | bool | str]:
    valid = [row for row in rows if bool(row["passes_fast_gate"])]
    if not valid:
        valid = rows
    return sorted(valid, key=_steady_rank_key)[0]


def _build_steady_comparison(candidate_metrics: dict[str, float], reference_metrics: dict[str, float]) -> dict[str, float | bool]:
    return {
        "rmse_delta": float(candidate_metrics["rmse"] - reference_metrics["rmse"]),
        "steady_state_error_delta": float(candidate_metrics["steady_state_error"] - reference_metrics["steady_state_error"]),
        "overshoot_delta": float(candidate_metrics["overshoot"] - reference_metrics["overshoot"]),
        "control_variation_delta": float(candidate_metrics["control_variation"] - reference_metrics["control_variation"]),
        "disturbance_recovery_time_delta": float(candidate_metrics["disturbance_recovery_time"] - reference_metrics["disturbance_recovery_time"]),
        "beats_reference": (
            float(candidate_metrics["steady_state_error"]) < float(reference_metrics["steady_state_error"])
            and float(candidate_metrics["control_variation"]) < float(reference_metrics["control_variation"])
            and float(candidate_metrics["overshoot"]) <= float(reference_metrics["overshoot"])
            and float(candidate_metrics["disturbance_recovery_time"]) <= float(reference_metrics["disturbance_recovery_time"]) * 1.10 + 1.0e-9
        ),
    }


def _derive_continuous_ranges(fast_params, steady_params, *, expansion_ratio: float) -> dict[str, dict[str, float]]:
    def expand(a: float, b: float) -> tuple[float, float]:
        low = min(a, b)
        high = max(a, b)
        span = high - low
        if span <= 1.0e-9:
            span = max(abs(low), 1.0) * expansion_ratio
        margin = span * expansion_ratio
        return low - margin, high + margin

    b0_min, b0_max = expand(float(fast_params.b0), float(steady_params.b0))
    wc_min, wc_max = expand(float(fast_params.wc), float(steady_params.wc))
    k_min, k_max = expand(float(fast_params.k), float(steady_params.k))
    return {
        "b0": {"min": float(b0_min), "max": float(b0_max)},
        "wc": {"min": float(wc_min), "max": float(wc_max)},
        "k": {"min": float(k_min), "max": float(k_max)},
    }


def _write_threeway_response_svg(
    path: Path,
    *,
    axis: str,
    pid_timeseries: list[dict[str, float]],
    fast_timeseries: list[dict[str, float]],
    steady_timeseries: list[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 960
    height = 420
    margin = 50
    key = axis
    target_key = f"target_{axis}"
    series = {
        "reference": [(float(row["time"]), float(row[target_key])) for row in steady_timeseries],
        "pid": [(float(row["time"]), float(row[key])) for row in pid_timeseries],
        "fast": [(float(row["time"]), float(row[key])) for row in fast_timeseries],
        "steady": [(float(row["time"]), float(row[key])) for row in steady_timeseries],
    }
    all_points = [point for values in series.values() for point in values]
    if not all_points:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='960' height='420'></svg>", encoding="utf-8")
        return
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if abs(max_x - min_x) < 1.0e-9:
        max_x = min_x + 1.0
    if abs(max_y - min_y) < 1.0e-9:
        max_y = min_y + 1.0

    def to_svg_points(values: list[tuple[float, float]]) -> str:
        coords: list[str] = []
        for x_value, y_value in values:
            x_px = margin + (x_value - min_x) / (max_x - min_x) * (width - 2 * margin)
            y_px = height - margin - (y_value - min_y) / (max_y - min_y) * (height - 2 * margin)
            coords.append(f"{x_px:.2f},{y_px:.2f}")
        return " ".join(coords)

    colors = {"reference": "#111111", "pid": "#1f77b4", "fast": "#d62728", "steady": "#2ca02c"}
    lines = []
    for label, values in series.items():
        lines.append(f"<polyline fill='none' stroke='{colors[label]}' stroke-width='2' points='{to_svg_points(values)}' />")
    legend = []
    for index, label in enumerate(("reference", "pid", "fast", "steady")):
        y = 18 + index * 18
        legend.append(f"<line x1='720' y1='{y}' x2='745' y2='{y}' stroke='{colors[label]}' stroke-width='2' />")
        legend.append(f"<text x='752' y='{y + 4}' font-size='12'>{label}</text>")
    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' />"
        f"<text x='{margin}' y='24' font-size='16'>x-axis steady tuning comparison</text>"
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='#777' />"
        f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='#777' />"
        + "".join(lines)
        + "".join(legend)
        + "</svg>"
    )
    path.write_text(svg, encoding="utf-8")
