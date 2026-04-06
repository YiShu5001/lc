from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lc.common.utils import seed_everything
from lc.control.configs import AxisTuningResult, PyBulletControlExperimentConfig
from lc.control.controllers import create_controller_bundle
from lc.control.envs import run_controller_episode
from lc.control.io import (
    build_run_directory,
    export_legacy_logger_artifacts,
    write_metrics_csv,
    write_reference_csv,
    write_summary_json,
    write_timeseries_csv,
)
from lc.control.plotting import (
    plot_attitude_response,
    plot_axis_error,
    plot_axis_tracking,
    plot_axis_velocity,
    plot_control_effort,
    plot_controller_comparison,
    plot_metric_heatmap,
    plot_pid_vs_best_ladrc_response,
    plot_single_factor_sensitivity,
    plot_training_curves,
)
from lc.control.policies.stacking import stack_state
from lc.control.reference_generators import build_xyz_reference_trajectory, summarize_reference_segments
from lc.control.simulators import close_ctrl_aviary, create_ctrl_aviary, run_training_episode
from lc.rl.algorithms import MDDPGConfig, MDDPGPolicy


@dataclass
class PyBulletAxisTrainer:
    config: PyBulletControlExperimentConfig

    def train_axis(self, axis: str, policy_config: MDDPGConfig | None = None) -> dict[str, object]:
        seed_everything(self.config.seed)
        env = create_ctrl_aviary(self.config)
        history_rows: list[dict[str, float]] = []
        best_reward = float("-inf")
        best_checkpoint: dict[str, object] | None = None
        policy = MDDPGPolicy(
            policy_config
            or MDDPGConfig(
                state_dim=8,
                action_dim=3,
                stack_size=1,
                action_hold_steps=self.config.action_hold_steps,
                batch_size=self.config.batch_size,
            )
        )
        backend_name = env["backend"]
        try:
            for episode in range(self.config.train_episodes):
                rng = np.random.default_rng(self.config.seed + episode)
                reference_bundle = build_xyz_reference_trajectory(self.config.axis_config(axis), self.config, rng=rng)
                controller = create_controller_bundle(self.config.training_controller_variant)
                artifacts = run_training_episode(
                    env,
                    policy,
                    controller,
                    reference_bundle,
                    axis=axis,
                    action_hold_steps=self.config.action_hold_steps,
                    config=self.config,
                )
                backend_name = artifacts.backend
                losses = policy.update(self.config.updates_per_step)
                episode_reward = float(np.sum(artifacts.rewards))
                history_rows.append(
                    {
                        "episode": float(episode),
                        "reward": episode_reward,
                        "actor_loss": float(losses["actor_loss"]),
                        "critic_loss": float(losses["critic_loss"]),
                    }
                )
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    best_checkpoint = {
                        "policy_state": {
                            "actor": policy.actor.state_dict(),
                            "critic": policy.critic.state_dict(),
                        },
                        "parameter_snapshot": controller.snapshot_params(),
                        "backend": artifacts.backend,
                    }
        finally:
            close_ctrl_aviary(env)
        run_dir = build_run_directory(self.config, "train", axis, self.config.training_controller_variant)
        checkpoint_path = self.save_best_checkpoint(axis, best_checkpoint or {}, run_dir)
        write_metrics_csv(run_dir / "training_history.csv", history_rows)
        figure = plot_training_curves(history_rows, run_dir / "figures")
        reference_bundle = build_xyz_reference_trajectory(self.config.axis_config(axis), self.config, rng=np.random.default_rng(self.config.seed))
        write_reference_csv(run_dir / "reference.csv", reference_bundle)
        write_summary_json(
            run_dir / "summary.json",
            {
                "axis": axis,
                "controller_variant": self.config.training_controller_variant,
                "backend": backend_name,
                "checkpoint_path": str(checkpoint_path),
                "best_reward": best_reward,
                "reference_segments": summarize_reference_segments(reference_bundle),
            },
        )
        return {
            "axis": axis,
            "output_dir": str(run_dir),
            "history": history_rows,
            "backend": backend_name,
            "checkpoint_path": str(checkpoint_path),
            "figures": [str(figure)],
        }

    def evaluate_axis(self, axis: str, controller_variants: list[str] | None = None) -> dict[str, object]:
        variants = controller_variants or [variant.name for variant in self.config.controller_variants]
        metric_rows: list[dict[str, float]] = []
        controller_runs: dict[str, dict[str, object]] = {}
        reference_bundle = build_xyz_reference_trajectory(
            self.config.axis_config(axis),
            self.config,
            rng=np.random.default_rng(self.config.seed + 101),
        )
        backend_name = "fallback"
        for name in variants:
            checkpoint = self._load_checkpoint_if_exists(axis) if name != "pid_pos_att" else None
            controller = create_controller_bundle(name, checkpoint=checkpoint)
            result = run_controller_episode(self.config, controller, reference_bundle)
            backend_name = str(result.get("backend", backend_name))
            metrics = dict(result["metrics"])
            metrics["controller"] = name
            metrics["backend"] = backend_name
            metric_rows.append(metrics)
            controller_runs[name] = result
        run_dir = build_run_directory(self.config, "eval", axis, "comparison")
        write_reference_csv(run_dir / "reference.csv", reference_bundle)
        write_metrics_csv(run_dir / "metrics.csv", metric_rows)
        figure_paths = [str(plot_controller_comparison(metric_rows, run_dir / "figures"))]
        for controller_name, result in controller_runs.items():
            controller_dir = Path(run_dir) / controller_name
            rows = list(result["timeseries"])
            write_timeseries_csv(controller_dir / "timeseries.csv", rows)
            export_legacy_logger_artifacts(result["legacy_rows"], controller_dir, controller_name)
            figure_paths.extend(
                [
                    str(plot_axis_tracking(rows, controller_dir / "figures")),
                    str(plot_axis_velocity(rows, controller_dir / "figures")),
                    str(plot_axis_error(rows, controller_dir / "figures")),
                    str(plot_attitude_response(rows, controller_dir / "figures")),
                    str(plot_control_effort(rows, controller_dir / "figures")),
                ]
            )
        write_summary_json(
            run_dir / "summary.json",
            {
                "axis": axis,
                "backend": backend_name,
                "controllers": variants,
                "reference_segments": summarize_reference_segments(reference_bundle),
                "metrics": metric_rows,
            },
        )
        return {
            "axis": axis,
            "backend": backend_name,
            "output_dir": str(run_dir),
            "metrics": metric_rows,
            "figures": figure_paths,
        }

    def evaluate_single_axis_ladrc_variant(
        self,
        axis: str,
        params: dict[str, float],
        difficulty: str = "medium",
        episodes: int | None = None,
    ) -> dict[str, object]:
        cfg = self._with_difficulty(difficulty)
        eval_episodes = episodes or cfg.tuning.eval_episodes_per_candidate
        controller_name = f"ladrc_{axis}_pos_pid_att"
        metric_rows: list[dict[str, float]] = []
        best_result: dict[str, object] | None = None
        for episode in range(eval_episodes):
            reference_bundle = build_xyz_reference_trajectory(
                cfg.axis_config(axis),
                cfg,
                rng=np.random.default_rng(cfg.seed + 300 + episode),
            )
            controller = create_controller_bundle(controller_name)
            controller.set_axis_parameters(axis, b0=params["b0"], omega_c=params["omega_c"], k=params["k"])
            result = run_controller_episode(cfg, controller, reference_bundle)
            row = dict(result["metrics"])
            row["episode"] = float(episode)
            row["difficulty"] = difficulty
            row["axis"] = axis
            row["b0"] = float(params["b0"])
            row["omega_c"] = float(params["omega_c"])
            row["k"] = float(params["k"])
            row["score"] = self._ranking_score(row)
            metric_rows.append(row)
            if best_result is None:
                best_result = result
        averaged = self._average_metric_rows(metric_rows)
        averaged["b0"] = float(params["b0"])
        averaged["omega_c"] = float(params["omega_c"])
        averaged["k"] = float(params["k"])
        averaged["score"] = self._ranking_score(averaged)
        return {
            "metrics": averaged,
            "seed_rows": metric_rows,
            "result": best_result or {},
        }

    def tune_single_axis_ladrc(self, axis: str) -> AxisTuningResult:
        tuning_cfg = self.config.tuning
        pid_metrics = self._evaluate_pid_baseline(axis, tuning_cfg.tuning_difficulties)
        coarse_rows = self._run_search_stage(axis, tuning_cfg.tuning_difficulties, stage="coarse")
        top_rows = sorted(coarse_rows, key=lambda row: row["score"])[: max(tuning_cfg.top_k, 1)]
        fine_rows = self._run_search_stage(axis, tuning_cfg.tuning_difficulties, stage="fine", seeds=top_rows)
        best_row = min(fine_rows, key=lambda row: row["score"])
        sensitivity_rows = self._run_sensitivity(axis, best_row)
        rl_bounds = self._derive_rl_bounds(best_row, sensitivity_rows)
        validation_rows = self._validate_best_candidate(axis, best_row)

        run_dir = self._build_tuning_run_dir(axis)
        reference_bundle = build_xyz_reference_trajectory(
            self.config.axis_config(axis),
            self._with_difficulty(tuning_cfg.tuning_difficulties[0]),
            rng=np.random.default_rng(self.config.seed + 77),
        )
        write_reference_csv(run_dir / "reference.csv", reference_bundle)
        write_metrics_csv(run_dir / "coarse_search.csv", coarse_rows)
        write_metrics_csv(run_dir / "fine_search.csv", fine_rows)
        write_metrics_csv(run_dir / "validation_metrics.csv", validation_rows)
        for key, rows in sensitivity_rows.items():
            write_metrics_csv(run_dir / f"sensitivity_{key}.csv", list(rows))
        recommended_params = {
            "axis": axis,
            "controller_variant": f"ladrc_{axis}_pos_pid_att",
            "baseline": {
                "b0": float(best_row["b0"]),
                "wc": float(best_row["omega_c"]),
                "k": float(best_row["k"]),
                "omega_o": float(best_row["omega_c"] * best_row["k"]),
            },
            "pid_baseline_metrics": pid_metrics,
            "best_ladrc_metrics": {k: float(v) for k, v in best_row.items() if isinstance(v, (int, float))},
        }
        write_summary_json(run_dir / "recommended_params.json", recommended_params)
        write_summary_json(run_dir / "rl_bounds.json", {"axis": axis, "rl_bounds": rl_bounds})

        pid_result = self._collect_baseline_timeseries(axis, "pid_pos_att")
        best_result = self._collect_baseline_timeseries(
            axis,
            f"ladrc_{axis}_pos_pid_att",
            params={"b0": best_row["b0"], "omega_c": best_row["omega_c"], "k": best_row["k"]},
        )
        figures_dir = run_dir / "figures"
        figure_paths = [
            plot_pid_vs_best_ladrc_response(pid_result["timeseries"], best_result["timeseries"], axis, figures_dir),
            plot_metric_heatmap(coarse_rows, "omega_c", "k", "rmse", figures_dir, "rmse_heatmap_wc_k.png", f"RMSE Heatmap ({axis})"),
            plot_metric_heatmap(coarse_rows, "b0", "omega_c", "iae", figures_dir, "iae_heatmap_b0_wc.png", f"IAE Heatmap ({axis})"),
            plot_single_factor_sensitivity(list(sensitivity_rows["b0"]), "b0", "score", figures_dir, "single_factor_sensitivity_b0.png", f"Sensitivity b0 ({axis})"),
            plot_single_factor_sensitivity(list(sensitivity_rows["omega_c"]), "omega_c", "score", figures_dir, "single_factor_sensitivity_wc.png", f"Sensitivity wc ({axis})"),
            plot_single_factor_sensitivity(list(sensitivity_rows["k"]), "k", "score", figures_dir, "single_factor_sensitivity_k.png", f"Sensitivity k ({axis})"),
        ]
        write_summary_json(
            run_dir / "summary.json",
            {
                "axis": axis,
                "controller_variant": f"ladrc_{axis}_pos_pid_att",
                "baseline": recommended_params["baseline"],
                "rl_bounds": rl_bounds,
                "pid_metrics": pid_metrics,
                "best_metrics": {k: float(v) for k, v in best_row.items() if isinstance(v, (int, float))},
                "reference_segments": summarize_reference_segments(reference_bundle),
                "validation_rows": validation_rows,
                "figures": [str(path) for path in figure_paths],
            },
        )
        return AxisTuningResult(
            axis=axis,
            controller_variant=f"ladrc_{axis}_pos_pid_att",
            recommended_params=recommended_params["baseline"],
            rl_bounds=rl_bounds,
            coarse_rows=tuple(coarse_rows),
            fine_rows=tuple(fine_rows),
            sensitivity_rows={key: tuple(rows) for key, rows in sensitivity_rows.items()},
            pid_metrics=pid_metrics,
            best_metrics={k: float(v) for k, v in best_row.items() if isinstance(v, (int, float))},
            output_dir=str(run_dir),
        )

    def run_full_protocol(self) -> dict[str, object]:
        training = {axis: self.train_axis(axis) for axis in ("x", "y", "z")}
        evaluation = {axis: self.evaluate_axis(axis) for axis in ("x", "y", "z")}
        return {"training": training, "evaluation": evaluation}

    def save_best_checkpoint(self, axis: str, policy_state: dict[str, object], run_dir: str | Path) -> Path:
        target = Path(run_dir) / "checkpoints"
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{axis}_policy.pt"
        torch.save(policy_state, path)
        return path

    def _load_checkpoint_if_exists(self, axis: str) -> dict[str, object] | None:
        base = Path(self.config.artifact.output_root) / "train" / axis
        if not base.exists():
            return None
        candidates = sorted(base.glob("**/checkpoints/*.pt"))
        if not candidates:
            return None
        return torch.load(candidates[-1], map_location="cpu")

    def _evaluate_pid_baseline(self, axis: str, difficulties: tuple[str, ...]) -> dict[str, float]:
        rows = []
        for difficulty in difficulties:
            cfg = self._with_difficulty(difficulty)
            reference_bundle = build_xyz_reference_trajectory(
                cfg.axis_config(axis),
                cfg,
                rng=np.random.default_rng(cfg.seed + 201),
            )
            controller = create_controller_bundle("pid_pos_att")
            result = run_controller_episode(cfg, controller, reference_bundle)
            row = dict(result["metrics"])
            row["difficulty"] = difficulty
            rows.append(row)
        return self._average_metric_rows(rows)

    def _run_search_stage(
        self,
        axis: str,
        difficulties: tuple[str, ...],
        stage: str,
        seeds: list[dict[str, float]] | None = None,
    ) -> list[dict[str, float]]:
        tuning_cfg = self.config.tuning
        if stage == "coarse":
            candidates = self._coarse_candidates(axis)
        else:
            candidates = self._fine_candidates(axis, seeds or [])
        rows: list[dict[str, float]] = []
        for params in candidates:
            metric_rows = []
            for difficulty in difficulties:
                result = self.evaluate_single_axis_ladrc_variant(axis, params, difficulty=difficulty)
                row = dict(result["metrics"])
                row["difficulty"] = difficulty
                metric_rows.append(row)
            averaged = self._average_metric_rows(metric_rows)
            averaged["b0"] = float(params["b0"])
            averaged["omega_c"] = float(params["omega_c"])
            averaged["k"] = float(params["k"])
            averaged["score"] = self._ranking_score(averaged)
            rows.append(averaged)
        return rows

    def _run_sensitivity(self, axis: str, best_row: dict[str, float]) -> dict[str, list[dict[str, float]]]:
        sensitivity: dict[str, list[dict[str, float]]] = {"b0": [], "omega_c": [], "k": []}
        for factor in sensitivity:
            for scale in self.config.tuning.sensitivity_scales:
                params = {
                    "b0": float(best_row["b0"]),
                    "omega_c": float(best_row["omega_c"]),
                    "k": float(best_row["k"]),
                }
                params[factor] = float(params[factor] * scale)
                clipped_low, clipped_high = self.config.tuning.rl_bounds_clip["omega_c" if factor == "omega_c" else factor]
                params[factor] = float(np.clip(params[factor], clipped_low, clipped_high))
                result = self.evaluate_single_axis_ladrc_variant(axis, params, difficulty=self.config.tuning.tuning_difficulties[0])
                row = dict(result["metrics"])
                row["factor"] = factor
                row["scale"] = float(scale)
                row["b0"] = params["b0"]
                row["omega_c"] = params["omega_c"]
                row["k"] = params["k"]
                row["score"] = self._ranking_score(row)
                sensitivity[factor].append(row)
        return sensitivity

    def _derive_rl_bounds(self, best_row: dict[str, float], sensitivity_rows: dict[str, list[dict[str, float]]]) -> dict[str, float]:
        threshold = float(best_row["score"]) * (1.0 + self.config.tuning.acceptable_degradation_ratio)
        bounds = {}
        mapping = {"b0": "b0", "omega_c": "wc", "k": "k"}
        for factor, rows in sensitivity_rows.items():
            acceptable = [row[factor] for row in rows if float(row["score"]) <= threshold]
            if not acceptable:
                acceptable = [float(best_row[factor])]
            clip_low, clip_high = self.config.tuning.rl_bounds_clip["omega_c" if factor == "omega_c" else factor]
            bounds[f"{mapping[factor]}_min"] = float(np.clip(min(acceptable), clip_low, clip_high))
            bounds[f"{mapping[factor]}_max"] = float(np.clip(max(acceptable), clip_low, clip_high))
        return bounds

    def _validate_best_candidate(self, axis: str, best_row: dict[str, float]) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        for difficulty in self.config.tuning.validation_difficulties:
            result = self.evaluate_single_axis_ladrc_variant(
                axis,
                {"b0": best_row["b0"], "omega_c": best_row["omega_c"], "k": best_row["k"]},
                difficulty=difficulty,
            )
            row = dict(result["metrics"])
            row["difficulty"] = difficulty
            row["score"] = self._ranking_score(row)
            rows.append(row)
        return rows

    def _collect_baseline_timeseries(
        self,
        axis: str,
        controller_name: str,
        params: dict[str, float] | None = None,
    ) -> dict[str, object]:
        cfg = self._with_difficulty(self.config.tuning.tuning_difficulties[0])
        reference_bundle = build_xyz_reference_trajectory(
            cfg.axis_config(axis),
            cfg,
            rng=np.random.default_rng(cfg.seed + 401),
        )
        controller = create_controller_bundle(controller_name)
        if params is not None:
            controller.set_axis_parameters(axis, b0=params["b0"], omega_c=params["omega_c"], k=params["k"])
        return run_controller_episode(cfg, controller, reference_bundle)

    def _coarse_candidates(self, axis: str) -> list[dict[str, float]]:
        base = create_controller_bundle(f"ladrc_{axis}_pos_pid_att").parameter_set.axis_config(axis)
        return [
            {
                "b0": float(base.b0 * b0_scale),
                "omega_c": float(max(base.omega_c + wc_offset, 0.5)),
                "k": float(max(base.k + k_offset, 2.0)),
            }
            for b0_scale in self.config.tuning.coarse_b0_scales
            for wc_offset in self.config.tuning.coarse_wc_offsets
            for k_offset in self.config.tuning.coarse_k_offsets
        ]

    def _fine_candidates(self, axis: str, seeds: list[dict[str, float]]) -> list[dict[str, float]]:
        candidates: list[dict[str, float]] = []
        seen: set[tuple[float, float, float]] = set()
        if not seeds:
            return self._coarse_candidates(axis)
        for seed in seeds:
            for b0_scale in self.config.tuning.fine_b0_scales:
                for wc_offset in self.config.tuning.fine_wc_offsets:
                    for k_offset in self.config.tuning.fine_k_offsets:
                        candidate = (
                            round(float(seed["b0"] * b0_scale), 6),
                            round(float(max(seed["omega_c"] + wc_offset, 0.5)), 6),
                            round(float(max(seed["k"] + k_offset, 2.0)), 6),
                        )
                        if candidate in seen:
                            continue
                        seen.add(candidate)
                        candidates.append({"b0": candidate[0], "omega_c": candidate[1], "k": candidate[2]})
        return candidates

    def _ranking_score(self, row: dict[str, float]) -> float:
        score = 0.0
        for metric, weight in self.config.tuning.ranking_weights.items():
            score += float(weight) * float(row.get(metric, 0.0))
        return float(score)

    def _average_metric_rows(self, rows: list[dict[str, float]]) -> dict[str, float]:
        metrics = [key for key in rows[0].keys() if key not in {"difficulty", "episode", "axis", "backend", "controller"}]
        averaged: dict[str, float] = {}
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            averaged[metric] = float(np.mean(values))
        return averaged

    def _with_difficulty(self, difficulty: str) -> PyBulletControlExperimentConfig:
        axis_configs = []
        for axis_cfg in self.config.axis_configs:
            scale = {
                "easy": 0.75,
                "medium": 1.0,
                "hard": 1.2,
                "extreme": 1.35,
            }[difficulty]
            axis_configs.append(
                type(axis_cfg)(
                    axis=axis_cfg.axis,
                    initial_position=axis_cfg.initial_position,
                    fixed_axes=axis_cfg.fixed_axes,
                    primary_speed_range=(axis_cfg.primary_speed_range[0] * scale, axis_cfg.primary_speed_range[1] * scale),
                    reverse_speed_range=(axis_cfg.reverse_speed_range[0] * scale, axis_cfg.reverse_speed_range[1] * scale),
                    stage_duration_range=axis_cfg.stage_duration_range,
                    include_disturbance=axis_cfg.include_disturbance,
                    disturbance_scale=axis_cfg.disturbance_scale * scale,
                    disturbance_axis_bias=axis_cfg.disturbance_axis_bias,
                    stage_count=axis_cfg.stage_count,
                )
            )
        return PyBulletControlExperimentConfig(
            drone_model=self.config.drone_model,
            simulation_freq_hz=self.config.simulation_freq_hz,
            control_freq_hz=self.config.control_freq_hz,
            rl_freq_hz=self.config.rl_freq_hz,
            duration_sec=self.config.duration_sec,
            gui=self.config.gui,
            seed=self.config.seed,
            warmup_steps=self.config.warmup_steps,
            train_episodes=self.config.train_episodes,
            eval_episodes=self.config.eval_episodes,
            updates_per_step=self.config.updates_per_step,
            batch_size=self.config.batch_size,
            training_controller_variant=self.config.training_controller_variant,
            tuning=self.config.tuning,
            artifact=self.config.artifact,
            controller_variants=self.config.controller_variants,
            axis_configs=tuple(axis_configs),
        )

    def _build_tuning_run_dir(self, axis: str) -> Path:
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = Path("outputs") / "control_pybullet_tuning" / axis / stamp
        target.mkdir(parents=True, exist_ok=True)
        return target
