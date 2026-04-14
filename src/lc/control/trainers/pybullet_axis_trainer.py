from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lc.common.utils import seed_everything
from lc.control.configs import AxisTuningResult, PyBulletControlExperimentConfig
from lc.control.controllers import create_controller_bundle
from lc.control.envs import compute_episode_metrics, run_controller_episode
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
from lc.control.simulators import close_ctrl_aviary, create_ctrl_aviary, run_policy_episode, run_training_episode
from lc.control.simulators.pybullet_runner import (
    _apply_axis_action,
    _apply_real_disturbance,
    _build_axis_observation,
    _compute_axis_reward,
    _disturbance_vector,
    _ensure_real_env,
    _timeseries_row,
)
from lc.rl.algorithms import MDDPGConfig, MDDPGPolicy


@dataclass
class PyBulletAxisTrainer:
    config: PyBulletControlExperimentConfig

    def train_axis(
        self,
        axis: str,
        policy_config: MDDPGConfig | None = None,
        *,
        n_step: int | None = None,
        shared_value: int | None = None,
    ) -> dict[str, object]:
        seed_everything(self.config.seed)
        run_dir = build_run_directory(self.config, "train", axis, self.config.training_controller_variant)
        env = create_ctrl_aviary(self.config)
        history_rows: list[dict[str, float]] = []
        best_reward = float("-inf")
        best_eval_score = float("inf")
        best_checkpoint: dict[str, object] | None = None
        final_checkpoint: dict[str, object] | None = None
        eval_history_rows: list[dict[str, float]] = []
        best_eval_snapshot_paths: list[str] = []
        policy_cfg = policy_config or MDDPGConfig(
            state_dim=8,
            action_dim=4,
            stack_size=1,
            action_hold_steps=self.config.action_hold_steps,
            batch_size=self.config.batch_size,
        )
        effective_n_step = max(int(n_step if n_step is not None else getattr(policy_cfg, "stack_size", 1)), 1)
        policy = MDDPGPolicy(policy_cfg)
        backend_name = env["backend"]
        snapshot_paths: list[str] = []
        try:
            for episode in range(self.config.train_episodes):
                policy.set_exploration_noise(self._episode_exploration_noise(episode, self.config.train_episodes, policy_cfg))
                rng = np.random.default_rng(self.config.seed + episode)
                reference_bundle = build_xyz_reference_trajectory(self.config.axis_config(axis), self.config, rng=rng)
                controller = create_controller_bundle(self.config.training_controller_variant)
                artifacts = run_training_episode(
                    env,
                    policy,
                    controller,
                    reference_bundle,
                    axis=axis,
                    action_hold_steps=policy_cfg.action_hold_steps,
                    n_step=effective_n_step,
                    config=self.config,
                )
                backend_name = artifacts.backend
                losses = policy.update(self.config.updates_per_step)
                episode_reward = float(np.sum(artifacts.rewards))
                average_reward = float(np.mean([row["reward"] for row in history_rows] + [episode_reward]))
                history_rows.append(
                    {
                        "episode": float(episode + 1),
                        "reward": episode_reward,
                        "average_reward": average_reward,
                        "exploration_noise": float(policy._current_expl_noise),
                        "actor_loss": float(losses["actor_loss"]),
                        "critic_loss": float(losses["critic_loss"]),
                    }
                )
                if episode_reward > best_reward:
                    best_reward = episode_reward
                if self.config.snapshot_interval and (episode + 1) % self.config.snapshot_interval == 0:
                    snapshot_paths.extend(self._save_episode_snapshot(axis, run_dir, episode + 1, list(artifacts.timeseries)))
                    eval_metrics, eval_rows = self._evaluate_policy_deterministic(axis, policy)
                    eval_row = {
                        "episode": float(episode + 1),
                        "rmse": float(eval_metrics["rmse"]),
                        "mae": float(eval_metrics["mae"]),
                        "velocity_rmse": float(eval_metrics["velocity_rmse"]),
                        "reward": float(eval_metrics["reward"]),
                        "score": float(self._evaluation_score(eval_metrics)),
                    }
                    eval_history_rows.append(eval_row)
                    if eval_row["score"] < best_eval_score:
                        best_eval_score = float(eval_row["score"])
                        best_eval_snapshot_paths = self._save_best_eval_snapshot(axis, run_dir, episode + 1, eval_rows)
                        best_checkpoint = {
                            "policy_state": self._serialize_policy_state(policy),
                            "parameter_snapshot": controller.snapshot_params(),
                            "backend": artifacts.backend,
                            "best_eval_metrics": eval_row,
                        }
            final_checkpoint = {
                "policy_state": self._serialize_policy_state(policy),
                "parameter_snapshot": controller.snapshot_params(),
                "backend": backend_name,
            }
        finally:
            close_ctrl_aviary(env)
        checkpoint_path = self.save_checkpoint(axis, final_checkpoint or {}, run_dir, filename=f"{axis}_policy.pt")
        best_checkpoint_path = self.save_checkpoint(axis, best_checkpoint or final_checkpoint or {}, run_dir, filename=f"{axis}_policy_best.pt")
        write_metrics_csv(run_dir / "training_history.csv", history_rows)
        average_reward_rows = [
            {"episode": row["episode"], "average_reward": row["average_reward"], "reward": row["reward"]}
            for row in history_rows
        ]
        write_metrics_csv(run_dir / "average_reward.csv", average_reward_rows)
        write_metrics_csv(run_dir / "eval_history.csv", eval_history_rows)
        figure = plot_training_curves(history_rows, run_dir / "figures")
        reference_bundle = build_xyz_reference_trajectory(self.config.axis_config(axis), self.config, rng=np.random.default_rng(self.config.seed))
        write_reference_csv(run_dir / "reference.csv", reference_bundle)
        write_summary_json(
            run_dir / "summary.json",
            {
                "axis": axis,
                "controller_variant": self.config.training_controller_variant,
                "backend": backend_name,
                "shared_value": shared_value,
                "action_dim": int(policy_cfg.action_dim),
                "stack_size": int(policy_cfg.stack_size),
                "action_hold_steps": int(policy_cfg.action_hold_steps),
                "n_step": int(effective_n_step),
                "checkpoint_path": str(checkpoint_path),
                "best_checkpoint_path": str(best_checkpoint_path),
                "best_reward": best_reward,
                "best_eval_score": best_eval_score if np.isfinite(best_eval_score) else None,
                "best_eval_metrics": best_checkpoint.get("best_eval_metrics") if best_checkpoint else None,
                "average_reward": float(np.mean([row["reward"] for row in history_rows])) if history_rows else 0.0,
                "snapshot_interval": int(self.config.snapshot_interval),
                "snapshot_figures": snapshot_paths,
                "best_eval_snapshot_figures": best_eval_snapshot_paths,
                "eval_history_path": str(run_dir / "eval_history.csv"),
                "reference_segments": summarize_reference_segments(reference_bundle),
            },
        )
        return {
            "axis": axis,
            "output_dir": str(run_dir),
            "history": history_rows,
            "backend": backend_name,
            "checkpoint_path": str(checkpoint_path),
            "best_checkpoint_path": str(best_checkpoint_path),
            "average_reward": float(np.mean([row["reward"] for row in history_rows])) if history_rows else 0.0,
            "eval_history": eval_history_rows,
            "best_eval_metrics": best_checkpoint.get("best_eval_metrics") if best_checkpoint else None,
            "figures": [str(figure), *snapshot_paths, *best_eval_snapshot_paths],
        }

    def _evaluate_policy_deterministic(self, axis: str, policy: MDDPGPolicy) -> tuple[dict[str, float], list[dict[str, float]]]:
        eval_seeds = tuple(int(seed) for seed in (self.config.eval_seeds or (self.config.seed,)))
        metric_rows: list[dict[str, float]] = []
        representative_rows: list[dict[str, float]] = []
        backend_name = "fallback"
        for episode_seed in eval_seeds:
            env = create_ctrl_aviary(self.config)
            try:
                reference_bundle = build_xyz_reference_trajectory(
                    self.config.axis_config(axis),
                    self.config,
                    rng=np.random.default_rng(episode_seed),
                )
                controller = create_controller_bundle(self.config.training_controller_variant)
                artifacts = run_policy_episode(
                    env,
                    policy,
                    controller,
                    reference_bundle,
                    axis,
                    self.config,
                    explore=False,
                    store_transitions=False,
                    n_step=1,
                )
                metrics = compute_episode_metrics(artifacts.timeseries, axis)
                metrics["backend"] = env["backend"]
                metrics["eval_seed"] = float(episode_seed)
                metric_rows.append(metrics)
                if not representative_rows:
                    representative_rows = list(artifacts.timeseries)
                backend_name = env["backend"]
            finally:
                close_ctrl_aviary(env)
        if not metric_rows:
            return (
                {"rmse": float("inf"), "mae": float("inf"), "velocity_rmse": float("inf"), "reward": float("-inf"), "backend": backend_name},
                [],
            )
        averaged: dict[str, float] = {}
        for key in ("rmse", "mae", "velocity_rmse", "reward"):
            averaged[key] = float(np.mean([float(row[key]) for row in metric_rows]))
        averaged["backend"] = backend_name
        averaged["eval_seed_count"] = float(len(metric_rows))
        return averaged, representative_rows

    def _evaluation_score(self, metrics: dict[str, float]) -> float:
        return float(metrics["rmse"] + 0.35 * metrics["mae"] + 0.2 * metrics["velocity_rmse"])

    def _episode_exploration_noise(
        self,
        episode: int,
        train_episodes: int,
        policy_config: MDDPGConfig,
    ) -> float:
        if train_episodes <= 1:
            return float(policy_config.expl_noise_start)
        if policy_config.expl_noise_schedule == "three_phase":
            start_value = float(policy_config.expl_noise_start)
            mid_value = min(0.1, start_value)
            end_value = float(policy_config.expl_noise_end)
            first_end = max(train_episodes // 3, 1)
            second_end = max((2 * train_episodes) // 3, first_end + 1)
            if episode < first_end:
                progress = episode / max(first_end - 1, 1)
                return float(start_value + progress * (mid_value - start_value))
            if episode < second_end:
                return float(mid_value)
            progress = (episode - second_end) / max(train_episodes - second_end - 1, 1)
            return float(mid_value + progress * (end_value - mid_value))
        if policy_config.expl_noise_schedule != "linear":
            return float(policy_config.expl_noise_start)
        progress = episode / max(train_episodes - 1, 1)
        return float(policy_config.expl_noise_start + progress * (policy_config.expl_noise_end - policy_config.expl_noise_start))

    def _serialize_policy_state(self, policy: MDDPGPolicy) -> dict[str, object]:
        return {
            "actor": policy.actor.state_dict(),
            "critic": policy.critic.state_dict(),
            "actor_target": policy.actor_target.state_dict(),
            "critic_target": policy.critic_target.state_dict(),
            "config": dict(policy.config.__dict__),
            "normalizer": policy._normalizer.copy(),
            "last_action": policy._last_action.copy(),
            "hold_counter": int(policy._hold_counter),
            "current_expl_noise": float(policy._current_expl_noise),
        }

    def _save_episode_snapshot(self, axis: str, run_dir: Path, episode: int, rows: list[dict[str, float]]) -> list[str]:
        snapshot_dir = run_dir / "figures" / f"episode_{episode:03d}"
        projected_rows = self._project_axis_rows(rows, axis)
        return [
            str(plot_axis_tracking(projected_rows, snapshot_dir)),
            str(plot_axis_error(projected_rows, snapshot_dir)),
            str(plot_axis_velocity(projected_rows, snapshot_dir)),
        ]

    def _save_best_eval_snapshot(self, axis: str, run_dir: Path, episode: int, rows: list[dict[str, float]]) -> list[str]:
        snapshot_dir = run_dir / "figures" / "best_eval"
        projected_rows = self._project_axis_rows(rows, axis)
        return [
            str(plot_axis_tracking(projected_rows, snapshot_dir)),
            str(plot_axis_error(projected_rows, snapshot_dir)),
            str(plot_axis_velocity(projected_rows, snapshot_dir)),
        ]

    def _project_axis_rows(self, rows: list[dict[str, float]], axis: str) -> list[dict[str, float]]:
        projected: list[dict[str, float]] = []
        for row in rows:
            projected.append(
                {
                    "time": row["time"],
                    "x": row.get(axis, 0.0),
                    "y": 0.0,
                    "z": 0.0,
                    "vx": row.get(f"v{axis}", 0.0),
                    "vy": 0.0,
                    "vz": 0.0,
                    "target_x": row.get(f"target_{axis}", 0.0),
                    "target_y": 0.0,
                    "target_z": 0.0,
                    "target_vx": row.get(f"target_v{axis}", 0.0),
                    "target_vy": 0.0,
                    "target_vz": 0.0,
                }
            )
        return projected

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
            try:
                reference_bundle = build_xyz_reference_trajectory(
                    cfg.axis_config(axis),
                    cfg,
                    rng=np.random.default_rng(cfg.seed + 300 + episode),
                )
                controller = create_controller_bundle(controller_name)
                controller.set_axis_parameters(
                    axis,
                    b0=params["b0"],
                    omega_c=params["omega_c"],
                    k=params.get("k"),
                    omega_o=params.get("omega_o"),
                    r=params.get("r"),
                )
                result = run_controller_episode(cfg, controller, reference_bundle)
                row = dict(result["metrics"])
                row["stable"] = 1.0 if self._is_stable_result(row) else 0.0
                if not bool(row["stable"]):
                    row["score"] = float(self.config.tuning.instability_penalty)
                else:
                    row["score"] = self._ranking_score(row)
                if best_result is None:
                    best_result = result
            except Exception:
                row = self._unstable_metric_row()
                row["stable"] = 0.0
                row["score"] = float(self.config.tuning.instability_penalty)
            row["episode"] = float(episode)
            row["difficulty"] = difficulty
            row["axis"] = axis
            row["b0"] = float(params["b0"])
            row["omega_c"] = float(params["omega_c"])
            row["k"] = float(params.get("k", params.get("omega_o", 0.0) / max(float(params["omega_c"]), 1e-6)))
            row["omega_o"] = float(params.get("omega_o", row["k"] * float(params["omega_c"])))
            row["r"] = float(params.get("r", create_controller_bundle(controller_name).parameter_set.axis_config(axis).r))
            metric_rows.append(row)
        averaged = self._average_metric_rows(metric_rows)
        averaged["b0"] = float(params["b0"])
        averaged["omega_c"] = float(params["omega_c"])
        averaged["k"] = float(params.get("k", params.get("omega_o", 0.0) / max(float(params["omega_c"]), 1e-6)))
        averaged["omega_o"] = float(params.get("omega_o", averaged["k"] * float(params["omega_c"])))
        averaged["r"] = float(params.get("r", create_controller_bundle(controller_name).parameter_set.axis_config(axis).r))
        averaged["stable"] = float(np.mean([row.get("stable", 0.0) for row in metric_rows]))
        averaged["score"] = (
            self._ranking_score(averaged)
            if averaged["stable"] >= 0.5 and self._is_stable_result(averaged)
            else float(self.config.tuning.instability_penalty)
        )
        return {
            "metrics": averaged,
            "seed_rows": metric_rows,
            "result": best_result or {},
        }

    def tune_single_axis_ladrc(self, axis: str) -> AxisTuningResult:
        tuning_cfg = self.config.tuning
        pid_metrics = self._evaluate_pid_baseline(axis, tuning_cfg.tuning_difficulties)
        base = create_controller_bundle(f"ladrc_{axis}_pos_pid_att").parameter_set.axis_config(axis)
        sequential_rows = self._run_sequential_search(
            axis,
            {
                "b0": float(base.b0),
                "omega_c": float(base.omega_c),
                "k": float(base.k),
            },
        )
        stage_b0_rows = sequential_rows["b0"]
        stage_wc_rows = sequential_rows["omega_c"]
        stage_k_rows = sequential_rows["k"]
        local_rows = sequential_rows["local_refine"]
        coarse_rows = stage_b0_rows + stage_wc_rows + stage_k_rows
        fine_rows = local_rows
        best_row = min(local_rows or stage_k_rows or stage_wc_rows or stage_b0_rows, key=lambda row: row["score"])
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
        write_metrics_csv(run_dir / "b0_stage.csv", stage_b0_rows)
        write_metrics_csv(run_dir / "wc_stage.csv", stage_wc_rows)
        write_metrics_csv(run_dir / "k_stage.csv", stage_k_rows)
        write_metrics_csv(run_dir / "local_refine.csv", local_rows)
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
                "protocol": "sequential_b0_wc_k_then_local_refine",
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

    def save_checkpoint(
        self,
        axis: str,
        policy_state: dict[str, object],
        run_dir: str | Path,
        *,
        filename: str | None = None,
    ) -> Path:
        target = Path(run_dir) / "checkpoints"
        target.mkdir(parents=True, exist_ok=True)
        path = target / (filename or f"{axis}_policy.pt")
        torch.save(policy_state, path)
        return path

    def _load_checkpoint_if_exists(self, axis: str) -> dict[str, object] | None:
        base = Path(self.config.artifact.output_root)
        if not base.exists():
            return None
        preferred = sorted(base.glob(f"**/checkpoints/{axis}_policy.pt"))
        candidates = preferred or sorted(base.glob("**/checkpoints/*.pt"))
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

    def _run_sequential_search(self, axis: str, base_params: dict[str, float]) -> dict[str, list[dict[str, float]]]:
        current = dict(base_params)
        b0_rows = self._search_single_factor(
            axis,
            current,
            factor="b0",
            candidates=[float(base_params["b0"] * scale) for scale in self.config.tuning.sequential_b0_scales],
        )
        current["b0"] = min(b0_rows, key=lambda row: row["score"])["b0"]
        wc_rows = self._search_single_factor(
            axis,
            current,
            factor="omega_c",
            candidates=[float(max(base_params["omega_c"] * scale, 0.3)) for scale in self.config.tuning.sequential_wc_scales],
        )
        current["omega_c"] = min(wc_rows, key=lambda row: row["score"])["omega_c"]
        k_rows = self._search_single_factor(
            axis,
            current,
            factor="k",
            candidates=[float(max(base_params["k"] * scale, 0.5)) for scale in self.config.tuning.sequential_k_scales],
        )
        current["k"] = min(k_rows, key=lambda row: row["score"])["k"]
        local_rows = self._search_local_refine(axis, current)
        return {
            "b0": b0_rows,
            "omega_c": wc_rows,
            "k": k_rows,
            "local_refine": local_rows,
        }

    def _search_single_factor(
        self,
        axis: str,
        params: dict[str, float],
        factor: str,
        candidates: list[float],
    ) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        seen: set[float] = set()
        for candidate in candidates:
            rounded = round(float(candidate), 6)
            if rounded in seen:
                continue
            seen.add(rounded)
            trial = dict(params)
            trial[factor] = float(candidate)
            result = self.evaluate_single_axis_ladrc_variant(
                axis,
                trial,
                difficulty=self.config.tuning.tuning_difficulties[0],
            )
            row = dict(result["metrics"])
            row["axis"] = axis
            row["search_factor"] = factor
            row["b0"] = float(trial["b0"])
            row["omega_c"] = float(trial["omega_c"])
            row["k"] = float(trial["k"])
            row["score"] = float(row.get("score", self.config.tuning.instability_penalty))
            rows.append(row)
        return rows

    def _search_local_refine(self, axis: str, params: dict[str, float]) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        seen: set[tuple[float, float, float]] = set()
        for b0_scale in self.config.tuning.local_refine_scales:
            for wc_scale in self.config.tuning.local_refine_scales:
                for k_scale in self.config.tuning.local_refine_scales:
                    candidate = {
                        "b0": float(params["b0"] * b0_scale),
                        "omega_c": float(params["omega_c"] * wc_scale),
                        "k": float(params["k"] * k_scale),
                    }
                    key = (
                        round(candidate["b0"], 6),
                        round(candidate["omega_c"], 6),
                        round(candidate["k"], 6),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    result = self.evaluate_single_axis_ladrc_variant(
                        axis,
                        candidate,
                        difficulty=self.config.tuning.tuning_difficulties[0],
                    )
                    row = dict(result["metrics"])
                    row["axis"] = axis
                    row["search_factor"] = "local_refine"
                    row["b0"] = candidate["b0"]
                    row["omega_c"] = candidate["omega_c"]
                    row["k"] = candidate["k"]
                    row["score"] = float(row.get("score", self.config.tuning.instability_penalty))
                    rows.append(row)
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
            controller.set_axis_parameters(
                axis,
                b0=params["b0"],
                omega_c=params["omega_c"],
                k=params.get("k"),
                omega_o=params.get("omega_o"),
                r=params.get("r"),
            )
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

    def _is_stable_result(self, row: dict[str, float]) -> bool:
        if not np.isfinite(float(row.get("rmse", np.inf))):
            return False
        for metric, limit in self.config.tuning.stability_limits.items():
            value = float(row.get(metric, np.inf))
            if not np.isfinite(value) or value > float(limit):
                return False
        return True

    def _unstable_metric_row(self) -> dict[str, float]:
        penalty = float(self.config.tuning.instability_penalty)
        return {
            "mae": penalty,
            "rmse": penalty,
            "iae": penalty,
            "overshoot": penalty,
            "settling_time": penalty,
            "steady_state_error": penalty,
            "control_energy": penalty,
            "disturbance_recovery_time": penalty,
            "control_variation": penalty,
            "velocity_rmse": penalty,
            "reward": -penalty,
        }

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
                    fixed_stage_lengths=axis_cfg.fixed_stage_lengths,
                    fixed_stage_velocities=(
                        tuple(float(value) * scale for value in axis_cfg.fixed_stage_velocities)
                        if axis_cfg.fixed_stage_velocities is not None
                        else None
                    ),
                    disturbance_step_window=axis_cfg.disturbance_step_window,
                    disturbance_frequency_rad=axis_cfg.disturbance_frequency_rad,
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
