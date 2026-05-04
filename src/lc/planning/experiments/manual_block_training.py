from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from lc.envs.scenarios import build_planning_scenario
from lc.planning.configs import LocalRiskCriticConfig, PlanningNetworkConfig, TransformerActorConfig, build_planning_network_config
from lc.planning.envs import PlanningSwarmEnv
from lc.planning.experiments.a1_rule_teacher import convert_teacher_episode_to_transition_records, load_teacher_episodes_from_dir
from lc.planning.experiments.guidance_self_only_diagnostics import _action_field, _action_field_with_scene, _feature_probe, _obstacle_response_probe
from lc.planning.trainers import PlanningTrainer

REWARD_DECOMP_KEYS = (
    "target_reward",
    "avoidance_reward",
    "collaboration_reward",
    "recovery_reward",
    "smoothness_penalty",
    "consistency_penalty",
    "success_bonus",
    "progress_reward",
    "risk_drop_reward",
    "clearance_gain_reward",
    "detour_trend_reward",
    "near_collision_penalty",
    "severe_near_collision_penalty",
    "critical_collision_margin_penalty",
    "action_saturation_penalty",
    "action_change_penalty",
    "timeout_penalty",
)


class RollingSuccessPromotionController:
    def __init__(
        self,
        *,
        env_sequence: list[str],
        window_size: int = 20,
        min_successes: int = 15,
        max_duration_seconds: float = 30.0,
        max_episodes_per_env: int = 0,
    ) -> None:
        self.env_sequence = [str(item) for item in env_sequence]
        self.window_size = max(1, int(window_size))
        self.min_successes = max(1, int(min_successes))
        self.max_duration_seconds = float(max_duration_seconds)
        self.max_episodes_per_env = max(0, int(max_episodes_per_env))
        self.current_index = 0
        self.current_env = self.env_sequence[0] if self.env_sequence else ""
        self.window_rows: list[dict[str, Any]] = []
        self.transition_history: list[dict[str, Any]] = []
        self.env_step_counts: dict[str, int] = {env_name: 0 for env_name in self.env_sequence}
        self.env_episode_counts: dict[str, int] = {env_name: 0 for env_name in self.env_sequence}
        self.completed_envs: list[str] = []
        self.last_state: dict[str, Any] = {
            "promotion_ready": False,
            "promotion_triggered": False,
            "current_env": self.current_env,
            "next_env": "",
            "remaining_env_steps_budget": 0,
            "completed_envs": [],
            "rolling_window_size": 0,
            "rolling_qualified_success_count": 0,
            "qualified_success_ratio": 0.0,
        }

    def observe(self, payload: dict[str, Any]) -> None:
        metrics = dict(payload.get("episode_metrics", {}))
        trainer = payload.get("trainer")
        if trainer is None or not self.env_sequence:
            return
        env_name = str(metrics.get("curriculum_env", self.current_env))
        steps = int(round(float(metrics.get("episode_steps", 0.0))))
        duration_seconds = float(metrics.get("episode_duration_seconds", 0.0))
        success = float(metrics.get("episode_success", 0.0)) > 0.5
        qualified = success and duration_seconds <= self.max_duration_seconds
        self.current_env = env_name
        self.env_step_counts[env_name] = self.env_step_counts.get(env_name, 0) + steps
        self.env_episode_counts[env_name] = self.env_episode_counts.get(env_name, 0) + 1
        self.window_rows.append(
            {
                "episode_count": int(payload.get("episode_count", 0)),
                "env_name": env_name,
                "success": bool(success),
                "duration_seconds": duration_seconds,
                "qualified": bool(qualified),
            }
        )
        if len(self.window_rows) > self.window_size:
            self.window_rows = self.window_rows[-self.window_size :]
        qualified_count = sum(1 for row in self.window_rows if bool(row["qualified"]))
        max_episode_ready = (
            self.max_episodes_per_env > 0
            and self.env_episode_counts.get(env_name, 0) >= self.max_episodes_per_env
            and self.current_index < len(self.env_sequence) - 1
        )
        success_ready = len(self.window_rows) >= self.window_size and qualified_count >= self.min_successes
        promotion_ready = (success_ready or max_episode_ready) and self.current_index < len(self.env_sequence) - 1
        promotion_triggered = False
        next_env = ""
        promotion_reason = ""
        if promotion_ready:
            previous_env = env_name
            self.current_index += 1
            next_env = self.env_sequence[self.current_index]
            scenario = build_planning_scenario(curriculum_env=next_env)
            trainer.env.set_scenario(
                replace(
                    scenario,
                    max_obstacles=trainer.env.scenario.max_obstacles,
                    max_neighbors=trainer.env.scenario.max_neighbors,
                )
            )
            if previous_env not in self.completed_envs:
                self.completed_envs.append(previous_env)
            self.transition_history.append(
                {
                    "episode_count": int(payload.get("episode_count", 0)),
                    "from_env": previous_env,
                    "to_env": next_env,
                    "rolling_window_size": len(self.window_rows),
                    "rolling_qualified_success_count": qualified_count,
                    "qualified_success_ratio": float(qualified_count / max(1, len(self.window_rows))),
                    "promotion_reason": "success_window" if success_ready else "max_episodes",
                }
            )
            self.current_env = next_env
            self.window_rows.clear()
            promotion_triggered = True
            promotion_reason = "success_window" if success_ready else "max_episodes"
        training_counters = dict(payload.get("training_counters", {}))
        target_total_env_steps = int(training_counters.get("target_total_env_steps", 0))
        actual_total_env_steps = int(training_counters.get("actual_total_env_steps", 0))
        self.last_state = {
            "promotion_ready": bool(promotion_ready),
            "promotion_triggered": bool(promotion_triggered),
            "current_env": self.current_env,
            "next_env": next_env,
            "remaining_env_steps_budget": max(0, target_total_env_steps - actual_total_env_steps),
            "completed_envs": list(self.completed_envs),
            "rolling_window_size": len(self.window_rows),
            "rolling_qualified_success_count": sum(1 for row in self.window_rows if bool(row["qualified"])),
            "qualified_success_ratio": float(sum(1 for row in self.window_rows if bool(row["qualified"])) / max(1, len(self.window_rows))),
            "max_episodes_per_env": int(self.max_episodes_per_env),
            "current_env_episode_count": int(self.env_episode_counts.get(self.current_env, 0)),
            "promotion_reason": promotion_reason,
        }

    def build_transition_rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.transition_history]

    def build_env_step_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for env_name in self.env_sequence:
            rows.append(
                {
                    "env_name": env_name,
                    "episode_count": int(self.env_episode_counts.get(env_name, 0)),
                    "env_step_count": int(self.env_step_counts.get(env_name, 0)),
                }
            )
        return rows


class ReviewMixPromotionController:
    def __init__(
        self,
        *,
        env_sequence: list[str],
        window_size: int = 50,
        min_successes: int = 26,
        max_episodes_per_env: int = 800,
        current_sample_count: int = 9,
        previous_sample_count: int = 1,
        rollback_previous_failures: int = 3,
        max_collision_rate: float = 0.30,
    ) -> None:
        self.env_sequence = [str(item) for item in env_sequence]
        self.window_size = max(1, int(window_size))
        self.min_successes = max(1, int(min_successes))
        self.max_episodes_per_env = max(0, int(max_episodes_per_env))
        self.current_sample_count = max(1, int(current_sample_count))
        self.previous_sample_count = max(0, int(previous_sample_count))
        self.rollback_previous_failures = max(1, int(rollback_previous_failures))
        self.max_collision_rate = float(max_collision_rate)
        self.mix_cycle = self.current_sample_count + self.previous_sample_count
        self.current_index = 0
        self.current_env = self.env_sequence[0] if self.env_sequence else ""
        self.stage_episode_count = 0
        self.window_rows: list[dict[str, Any]] = []
        self.previous_review_rows: list[dict[str, Any]] = []
        self.degradation_warning_streak = 0
        self.transition_history: list[dict[str, Any]] = []
        self.env_step_counts: dict[str, int] = {env_name: 0 for env_name in self.env_sequence}
        self.env_episode_counts: dict[str, int] = {env_name: 0 for env_name in self.env_sequence}
        self.completed_envs: list[str] = []
        self.last_planned_env = self.current_env
        self.last_planned_env_role = "current"
        self.last_state: dict[str, Any] = {
            "promotion_ready": False,
            "promotion_triggered": False,
            "rollback_triggered": False,
            "current_env": self.current_env,
            "next_env": "",
            "previous_review_env": "",
            "rolling_window_size": 0,
            "rolling_success_count": 0,
            "rolling_success_ratio": 0.0,
            "rolling_collision_count": 0,
            "rolling_collision_rate": 0.0,
            "previous_review_count": 0,
            "previous_review_failure_count": 0,
            "degradation_warning": False,
            "degradation_warning_streak": 0,
            "promotion_reason": "",
        }

    def on_episode_start(self, payload: dict[str, Any]) -> None:
        trainer = payload.get("trainer")
        if trainer is None or not self.env_sequence:
            return
        env_name = self._select_episode_env()
        scenario = build_planning_scenario(curriculum_env=env_name)
        trainer.env.set_scenario(
            replace(
                scenario,
                max_obstacles=trainer.env.scenario.max_obstacles,
                max_neighbors=trainer.env.scenario.max_neighbors,
            )
        )
        self.last_planned_env = env_name
        self.last_planned_env_role = "previous_review" if env_name != self.current_env else "current"

    def observe(self, payload: dict[str, Any]) -> None:
        metrics = dict(payload.get("episode_metrics", {}))
        trainer = payload.get("trainer")
        if trainer is None or not self.env_sequence:
            return
        env_name = str(metrics.get("curriculum_env", self.last_planned_env or self.current_env))
        steps = int(round(float(metrics.get("episode_steps", 0.0))))
        success = float(metrics.get("episode_success", 0.0)) > 0.5
        collision = float(metrics.get("episode_collision", 0.0)) > 0.5
        previous_env = self._previous_env()
        is_previous_review = bool(previous_env and env_name == previous_env)
        self.stage_episode_count += 1
        self.env_step_counts[env_name] = self.env_step_counts.get(env_name, 0) + steps
        self.env_episode_counts[env_name] = self.env_episode_counts.get(env_name, 0) + 1
        self.window_rows.append(
            {
                "episode_count": int(payload.get("episode_count", 0)),
                "env_name": env_name,
                "success": bool(success),
                "collision": bool(collision),
                "sample_role": "previous_review" if is_previous_review else "current",
            }
        )
        if len(self.window_rows) > self.window_size:
            self.window_rows = self.window_rows[-self.window_size :]
        if is_previous_review:
            self.previous_review_rows.append(self.window_rows[-1])
            if len(self.previous_review_rows) > self.previous_sample_count * 5:
                self.previous_review_rows = self.previous_review_rows[-self.previous_sample_count * 5 :]
        self._maybe_transition(payload, trainer)

    def build_transition_rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.transition_history]

    def build_env_step_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "env_name": env_name,
                "episode_count": int(self.env_episode_counts.get(env_name, 0)),
                "env_step_count": int(self.env_step_counts.get(env_name, 0)),
            }
            for env_name in self.env_sequence
        ]

    def _select_episode_env(self) -> str:
        previous_env = self._previous_env()
        if previous_env and self.previous_sample_count > 0 and self.mix_cycle > 0:
            cycle_index = self.stage_episode_count % self.mix_cycle
            if cycle_index >= self.current_sample_count:
                return previous_env
        return self.current_env

    def _previous_env(self) -> str:
        if self.current_index <= 0:
            return ""
        return self.env_sequence[self.current_index - 1]

    def _set_current_env(self, trainer: Any, env_name: str) -> None:
        scenario = build_planning_scenario(curriculum_env=env_name)
        trainer.env.set_scenario(
            replace(
                scenario,
                max_obstacles=trainer.env.scenario.max_obstacles,
                max_neighbors=trainer.env.scenario.max_neighbors,
            )
        )

    def _maybe_transition(self, payload: dict[str, Any], trainer: Any) -> None:
        window = list(self.window_rows[-self.window_size :])
        success_count = sum(1 for row in window if bool(row["success"]))
        collision_count = sum(1 for row in window if bool(row.get("collision", False)))
        collision_rate = float(collision_count / max(1, len(window)))
        previous_env = self._previous_env()
        previous_rows = list(self.previous_review_rows[-5:])
        previous_failures = sum(1 for row in previous_rows if not bool(row["success"]))
        degradation_warning = self.current_index > 0 and len(previous_rows) >= 5 and previous_failures >= self.rollback_previous_failures
        if degradation_warning:
            self.degradation_warning_streak += 1
            self.previous_review_rows.clear()
        elif len(previous_rows) >= 5:
            self.degradation_warning_streak = 0
            self.previous_review_rows.clear()
        rollback_ready = self.current_index > 0 and self.degradation_warning_streak >= 2
        success_ready = (
            len(window) >= self.window_size
            and success_count >= self.min_successes
            and collision_rate < self.max_collision_rate
        )
        max_episode_ready = (
            self.max_episodes_per_env > 0
            and self.stage_episode_count >= self.max_episodes_per_env
            and self.current_index < len(self.env_sequence) - 1
        )
        transition_reason = ""
        promotion_triggered = False
        rollback_triggered = False
        next_env = ""
        previous_current = self.current_env
        if rollback_ready:
            self.current_index -= 1
            self.current_env = self.env_sequence[self.current_index]
            next_env = self.current_env
            transition_reason = "previous_review_failure"
            rollback_triggered = True
            self.degradation_warning_streak = 0
            self.previous_review_rows.clear()
        elif (success_ready or max_episode_ready) and self.current_index < len(self.env_sequence) - 1:
            self.current_index += 1
            self.current_env = self.env_sequence[self.current_index]
            next_env = self.current_env
            transition_reason = "success_window" if success_ready else "max_episodes"
            promotion_triggered = True
            if previous_current not in self.completed_envs:
                self.completed_envs.append(previous_current)
        if promotion_triggered or rollback_triggered:
            self._set_current_env(trainer, self.current_env)
            self.transition_history.append(
                {
                    "episode_count": int(payload.get("episode_count", 0)),
                    "from_env": previous_current,
                    "to_env": self.current_env,
                    "rolling_window_size": len(window),
                    "rolling_success_count": success_count,
                    "rolling_success_ratio": float(success_count / max(1, len(window))),
                    "rolling_collision_count": collision_count,
                    "rolling_collision_rate": collision_rate,
                    "previous_review_env": previous_env,
                    "previous_review_count": len(previous_rows),
                    "previous_review_failure_count": previous_failures,
                    "promotion_reason": transition_reason,
                    "rollback_triggered": float(rollback_triggered),
                }
            )
            self.window_rows.clear()
            self.previous_review_rows.clear()
            self.stage_episode_count = 0
        self.last_state = {
            "promotion_ready": bool(success_ready or max_episode_ready),
            "promotion_triggered": bool(promotion_triggered),
            "rollback_triggered": bool(rollback_triggered),
            "current_env": self.current_env,
            "next_env": next_env,
            "previous_review_env": previous_env,
            "completed_envs": list(self.completed_envs),
            "rolling_window_size": len(self.window_rows),
            "rolling_success_count": sum(1 for row in self.window_rows if bool(row["success"])),
            "rolling_success_ratio": float(sum(1 for row in self.window_rows if bool(row["success"])) / max(1, len(self.window_rows))),
            "rolling_collision_count": int(sum(1 for row in self.window_rows if bool(row.get("collision", False)))),
            "rolling_collision_rate": float(sum(1 for row in self.window_rows if bool(row.get("collision", False))) / max(1, len(self.window_rows))),
            "max_collision_rate": float(self.max_collision_rate),
            "previous_review_count": len(previous_rows),
            "previous_review_failure_count": previous_failures,
            "degradation_warning": bool(degradation_warning),
            "degradation_warning_streak": int(self.degradation_warning_streak),
            "current_previous_sample_ratio": f"{self.current_sample_count}:{self.previous_sample_count}",
            "promotion_reason": transition_reason,
        }


def _window_obstacle_probe(trainer: PlanningTrainer, *, stage_mode: str = "avoidance") -> dict[str, float]:
    actor = trainer.actor
    cfg = trainer.network_config
    base_state = np.array([0.6, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def obstacle_case(x: float | None, y: float | None) -> tuple[np.ndarray, np.ndarray]:
        obstacles = np.zeros((1, cfg.max_obstacles, cfg.obstacle_dim), dtype=np.float32)
        mask = np.zeros((1, cfg.max_obstacles), dtype=np.float32)
        if x is not None and y is not None:
            obstacles[0, 0] = np.array([x, y, 0.0, 0.0, 0.06], dtype=np.float32)
            mask[0, 0] = 1.0
        return obstacles, mask

    cases = {
        "clear": obstacle_case(None, None),
        "center": obstacle_case(0.45, 0.0),
        "left": obstacle_case(0.45, 0.10),
        "right": obstacle_case(0.45, -0.10),
    }
    actions: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, (obstacles, obstacle_mask) in cases.items():
            obs = {
                "self_state": torch.tensor(base_state, dtype=torch.float32).unsqueeze(0),
                "obstacles": torch.tensor(obstacles, dtype=torch.float32),
                "neighbors": torch.zeros((1, cfg.max_neighbors, cfg.neighbor_dim), dtype=torch.float32),
                "obstacle_mask": torch.tensor(obstacle_mask, dtype=torch.float32),
                "neighbor_mask": torch.zeros((1, cfg.max_neighbors), dtype=torch.float32),
            }
            _, action = actor(obs, stage_mode=stage_mode)
            actions[name] = action.squeeze(0).cpu()
    return {
        "obstacle_left_vs_right_action_delta": float(torch.norm(actions["left"] - actions["right"]).item()),
        "obstacle_center_vs_clear_action_delta": float(torch.norm(actions["center"] - actions["clear"]).item()),
    }


def _select_best_window_episode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    successes = [row for row in rows if bool(float(row["episode_metrics"].get("episode_success", 0.0)) > 0.5)]
    if successes:
        return min(
            successes,
            key=lambda row: (
                float(row["episode_metrics"].get("episode_duration_seconds", 1e9)),
                -float(row["episode_metrics"].get("reward", -1e9)),
                float(row["episode_metrics"].get("final_target_distance", 1e9)),
            ),
        )
    non_crash = [
        row
        for row in rows
        if float(row["episode_metrics"].get("episode_collision", 0.0)) <= 0.5
        and float(row["episode_metrics"].get("episode_out_of_bounds", 0.0)) <= 0.5
    ]
    if non_crash:
        return min(
            non_crash,
            key=lambda row: (
                float(row["episode_metrics"].get("final_target_distance", 1e9)),
                -float(row["episode_metrics"].get("reward", -1e9)),
            ),
        )
    return max(rows, key=lambda row: float(row["episode_metrics"].get("reward", -1e9)))


def _select_success_window_episode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if bool(float(row["episode_metrics"].get("episode_success", 0.0)) > 0.5)]
    if not successes:
        return {}
    return min(
        successes,
        key=lambda row: (
            float(row["episode_metrics"].get("episode_duration_seconds", 1e9)),
            -float(row["episode_metrics"].get("reward", -1e9)),
            float(row["episode_metrics"].get("final_target_distance", 1e9)),
        ),
    )


class WindowedRunMonitor:
    def __init__(
        self,
        *,
        run_dir: Path,
        write_every: int,
        route_save_every: int,
        window_summary_every: int,
        checkpoint_save_every: int = 0,
        promotion_controller: RollingSuccessPromotionController | ReviewMixPromotionController | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.write_every = int(write_every)
        self.route_save_every = int(route_save_every)
        self.window_summary_every = max(1, int(window_summary_every))
        self.checkpoint_save_every = max(0, int(checkpoint_save_every))
        self.promotion_controller = promotion_controller
        self.start_time = float(time.perf_counter())
        self.window_rows: list[dict[str, Any]] = []
        self.window_summaries: list[dict[str, Any]] = []
        self.completed_episode_metrics: list[dict[str, Any]] = []
        self.last_payload: dict[str, Any] | None = None
        self._episode_heartbeat_state: dict[int, dict[str, float]] = {}
        self._stall_status: dict[str, Any] = {
            "stall_warning": False,
            "stall_reason": "",
            "stall_details": {},
        }
        self._saved_early_events: set[str] = set()

    def on_episode_end(self, payload: dict[str, Any]) -> None:
        if self.promotion_controller is not None:
            self.promotion_controller.observe(payload)
        self.completed_episode_metrics.append(dict(payload.get("episode_metrics", {})))
        episode_index = int(payload.get("episode_index", -1))
        self._episode_heartbeat_state.pop(episode_index, None)
        self._update_stall_status(payload, is_heartbeat=False)
        self._maybe_save_early_event(payload)

    def on_progress(self, payload: dict[str, Any]) -> None:
        self.last_payload = dict(payload)
        is_heartbeat = bool(payload.get("heartbeat", False))
        self._update_stall_status(payload, is_heartbeat=is_heartbeat)
        episode_count = int(payload.get("episode_count", 0))
        total_episodes = int(payload.get("total_episodes", 0))
        if not is_heartbeat and self.route_save_every > 0 and episode_count > 0 and episode_count % self.route_save_every == 0:
            _save_episode_route_artifacts(
                run_dir=self.run_dir,
                episode_id=episode_count,
                episode_metrics=dict(payload.get("episode_metrics", {})),
                episode_trace=dict(payload.get("episode_trace", {})),
                reward_components=list(payload.get("reward_components", [])),
            )
        if not is_heartbeat:
            self._maybe_save_checkpoint(payload)
        if not is_heartbeat:
            self.window_rows.append(
                {
                    "episode_count": episode_count,
                    "episode_metrics": dict(payload.get("episode_metrics", {})),
                    "episode_trace": dict(payload.get("episode_trace", {})),
                    "reward_components": list(payload.get("reward_components", [])),
                    "replay_stats": dict(payload.get("replay_stats", {})),
                }
            )
            if len(self.window_rows) >= self.window_summary_every:
                self._finalize_window(trainer=payload.get("trainer"))
        should_write = episode_count == total_episodes or (self.write_every > 0 and episode_count % self.write_every == 0)
        if total_episodes <= 0:
            should_write = self.write_every > 0 and episode_count % max(1, self.write_every) == 0
        if is_heartbeat:
            should_write = True
        if should_write:
            self._write_live_files(payload)

    def _maybe_save_checkpoint(self, payload: dict[str, Any]) -> None:
        if self.checkpoint_save_every <= 0:
            return
        episode_count = int(payload.get("episode_count", 0))
        if episode_count <= 0 or episode_count % self.checkpoint_save_every != 0:
            return
        trainer = payload.get("trainer")
        if trainer is None:
            return
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"trainer_state_episode_{episode_count:04d}.pt"
        trainer.save_training_state(checkpoint_path, include_replay=True, include_optimizer=False)

    def finalize(self, payload: dict[str, Any]) -> None:
        if self.window_rows:
            self._finalize_window(trainer=payload.get("trainer"))
        self._write_live_files(payload)

    def _update_stall_status(self, payload: dict[str, Any], *, is_heartbeat: bool) -> None:
        episode_metrics = dict(payload.get("episode_metrics", {}))
        episode_index = int(payload.get("episode_index", -1))
        episode_steps = float(episode_metrics.get("episode_steps", 0.0))
        final_target_distance = float(episode_metrics.get("final_target_distance", 0.0))
        episode_success = float(episode_metrics.get("episode_success", 0.0)) > 0.5
        episode_collision = float(episode_metrics.get("episode_collision", 0.0)) > 0.5
        episode_out = float(episode_metrics.get("episode_out_of_bounds", 0.0)) > 0.5
        if is_heartbeat and episode_index >= 0:
            heartbeat_state = self._episode_heartbeat_state.setdefault(
                episode_index,
                {
                    "start_distance": final_target_distance,
                    "last_distance": final_target_distance,
                },
            )
            start_distance = float(heartbeat_state.get("start_distance", final_target_distance))
            previous_distance = float(heartbeat_state.get("last_distance", final_target_distance))
            heartbeat_state["last_distance"] = final_target_distance
            distance_drop = start_distance - final_target_distance
            heartbeat_improvement = previous_distance - final_target_distance
            if (
                episode_steps > 3000.0
                and not episode_success
                and not episode_collision
                and not episode_out
                and distance_drop < 0.10
            ):
                self._stall_status = {
                    "stall_warning": True,
                    "stall_reason": "long_episode_no_progress",
                    "stall_details": {
                        "episode_index": episode_index,
                        "episode_steps": episode_steps,
                        "distance_drop": distance_drop,
                        "final_target_distance": final_target_distance,
                    },
                }
                return
            if (
                not episode_success
                and not episode_collision
                and not episode_out
                and heartbeat_improvement < 0.05
                and episode_steps >= 2000.0
            ):
                self._stall_status = {
                    "stall_warning": True,
                    "stall_reason": "heartbeat_progress_flat",
                    "stall_details": {
                        "episode_index": episode_index,
                        "episode_steps": episode_steps,
                        "heartbeat_improvement": heartbeat_improvement,
                        "final_target_distance": final_target_distance,
                    },
                }
                return
        recent_rows = self.completed_episode_metrics[-20:]
        if recent_rows:
            success_count = sum(float(row.get("episode_success", 0.0)) > 0.5 for row in recent_rows)
            step_cap_count = sum(float(row.get("episode_step_cap_hit", 0.0)) > 0.5 for row in recent_rows)
            if success_count == 0 and step_cap_count >= 5:
                self._stall_status = {
                    "stall_warning": True,
                    "stall_reason": "recent_step_cap_cluster",
                    "stall_details": {
                        "recent_episode_count": len(recent_rows),
                        "recent_success_count": int(success_count),
                        "recent_step_cap_count": int(step_cap_count),
                    },
                }
                return
        self._stall_status = {
            "stall_warning": False,
            "stall_reason": "",
            "stall_details": {},
        }

    def _maybe_save_early_event(self, payload: dict[str, Any]) -> None:
        episode_metrics = dict(payload.get("episode_metrics", {}))
        trainer = payload.get("trainer")
        event_map = {
            "episode_step_cap_hit": float(episode_metrics.get("episode_step_cap_hit", 0.0)) > 0.5,
            "collision": float(episode_metrics.get("episode_collision", 0.0)) > 0.5,
            "out_of_bounds": float(episode_metrics.get("episode_out_of_bounds", 0.0)) > 0.5,
            "success": float(episode_metrics.get("episode_success", 0.0)) > 0.5,
        }
        for event_name, triggered in event_map.items():
            if not triggered or event_name in self._saved_early_events:
                continue
            event_dir = self.run_dir / "diagnostics" / "early_events"
            event_dir.mkdir(parents=True, exist_ok=True)
            episode_id = int(payload.get("episode_count", 0))
            stem = f"first_{event_name}_episode_{episode_id:04d}"
            _save_episode_route_artifacts(
                run_dir=event_dir,
                episode_id=episode_id,
                episode_metrics=episode_metrics,
                episode_trace=dict(payload.get("episode_trace", {})),
                reward_components=list(payload.get("reward_components", [])),
            )
            action_field = {}
            if trainer is not None:
                action_field = _action_field_with_scene(
                    trainer,
                    event_dir,
                    stem,
                    stage_mode=str(getattr(trainer, "current_stage_mode", "avoidance")),
                    scene=dict(payload.get("episode_trace", {})),
                    target_override=np.asarray(dict(payload.get("episode_trace", {})).get("target_position", [0.35, 0.0]), dtype=np.float32),
                )
            _write_json(
                event_dir / f"{stem}.json",
                {
                    "event_name": event_name,
                    "episode_id": episode_id,
                    "episode_metrics": _to_serializable(episode_metrics),
                    "reward_component_means": _reward_component_summary(list(payload.get("reward_components", []))),
                    "episode_trace": _to_serializable(dict(payload.get("episode_trace", {}))),
                    "action_field": _to_serializable(action_field),
                },
            )
            self._saved_early_events.add(event_name)

    def _finalize_window(self, *, trainer: Any) -> None:
        if not self.window_rows:
            return
        best = _select_best_window_episode(self.window_rows)
        best_episode_metrics = dict(best.get("episode_metrics", {}))
        best_episode_trace = dict(best.get("episode_trace", {}))
        best_reward_components = list(best.get("reward_components", []))
        best_payload = _episode_route_payload(
            episode_id=int(best.get("episode_count", 0)),
            episode_metrics=best_episode_metrics,
            episode_trace=best_episode_trace,
            reward_components=best_reward_components,
        )
        window_start_episode = int(self.window_rows[0]["episode_count"])
        window_end_episode = int(self.window_rows[-1]["episode_count"])
        env_names = [str(row["episode_metrics"].get("curriculum_env", "")) for row in self.window_rows]
        unique_envs = sorted(set(env_names))
        latest_replay_stats = dict(self.window_rows[-1].get("replay_stats", {}))
        full_window = len(self.window_rows) >= self.window_summary_every
        q_stats = _q_value_distribution(trainer, block_episode_count=window_end_episode, sample_window=50) if trainer is not None and full_window else {}
        probe_stats = (
            _window_obstacle_probe(trainer, stage_mode=str(getattr(trainer, "current_stage_mode", "avoidance")))
            if trainer is not None and full_window
            else {}
        )
        layout_split_stats = _window_layout_split_stats(self.window_rows)
        latest_episode_metrics = dict(self.window_rows[-1].get("episode_metrics", {}))
        summary = {
            "window_start_episode": window_start_episode,
            "window_end_episode": window_end_episode,
            "window_env": unique_envs[0] if len(unique_envs) == 1 else "mixed",
            "window_env_counts": {name: env_names.count(name) for name in unique_envs},
            "success_count": int(sum(float(row["episode_metrics"].get("episode_success", 0.0)) > 0.5 for row in self.window_rows)),
            "collision_count": int(sum(float(row["episode_metrics"].get("episode_collision", 0.0)) > 0.5 for row in self.window_rows)),
            "timeout_count": int(sum(float(row["episode_metrics"].get("episode_timeout", 0.0)) > 0.5 for row in self.window_rows)),
            "soft_timeout_count": int(sum(float(row["episode_metrics"].get("soft_timeout_episode", 0.0)) > 0.5 for row in self.window_rows)),
            "out_of_bounds_count": int(sum(float(row["episode_metrics"].get("episode_out_of_bounds", 0.0)) > 0.5 for row in self.window_rows)),
            "episode_step_cap_count": int(sum(float(row["episode_metrics"].get("episode_step_cap_hit", 0.0)) > 0.5 for row in self.window_rows)),
            "qualified_success_count": int(
                sum(
                    float(row["episode_metrics"].get("episode_success", 0.0)) > 0.5
                    and float(row["episode_metrics"].get("episode_duration_seconds", 0.0)) <= 30.0
                    for row in self.window_rows
                )
            ),
            "qualified_success_ratio": float(
                sum(
                    float(row["episode_metrics"].get("episode_success", 0.0)) > 0.5
                    and float(row["episode_metrics"].get("episode_duration_seconds", 0.0)) <= 30.0
                    for row in self.window_rows
                )
                / max(1, len(self.window_rows))
            ),
            "mean_episode_return": float(np.mean([float(row["episode_metrics"].get("reward", 0.0)) for row in self.window_rows])),
            "mean_final_target_distance": float(np.mean([float(row["episode_metrics"].get("final_target_distance", 0.0)) for row in self.window_rows])),
            "mean_episode_duration_seconds": float(np.mean([float(row["episode_metrics"].get("episode_duration_seconds", 0.0)) for row in self.window_rows])),
            "mean_steps_per_episode": float(np.mean([float(row["episode_metrics"].get("episode_steps", 0.0)) for row in self.window_rows])),
            "mean_action_abs": float(np.mean([float(row["episode_metrics"].get("action_abs_mean", 0.0)) for row in self.window_rows])),
            "mean_action_saturation_rate": float(np.mean([float(row["episode_metrics"].get("actor_output_saturation_rate", 0.0)) for row in self.window_rows])),
            "current_exploration_noise": float(latest_episode_metrics.get("current_exploration_noise", 0.0)),
            "exploration_stage_index": int(latest_episode_metrics.get("exploration_stage_index", 0)),
            "progress_reward_mean": float(np.mean([float(row["episode_metrics"].get("progress_reward", 0.0)) for row in self.window_rows])),
            "risk_drop_reward_mean": float(np.mean([float(row["episode_metrics"].get("risk_drop_reward", 0.0)) for row in self.window_rows])),
            "clearance_gain_reward_mean": float(np.mean([float(row["episode_metrics"].get("clearance_gain_reward", 0.0)) for row in self.window_rows])),
            "detour_trend_reward_mean": float(np.mean([float(row["episode_metrics"].get("detour_trend_reward", 0.0)) for row in self.window_rows])),
            "near_collision_penalty_mean": float(np.mean([float(row["episode_metrics"].get("near_collision_penalty", 0.0)) for row in self.window_rows])),
            "critical_collision_penalty_mean": float(np.mean([float(row["episode_metrics"].get("critical_collision_margin_penalty", 0.0)) for row in self.window_rows])),
            "timeout_penalty_mean": float(np.mean([float(row["episode_metrics"].get("timeout_penalty", 0.0)) for row in self.window_rows])),
            "td_error_mean": float(np.mean([float(row["episode_metrics"].get("td_error_mean", 0.0)) for row in self.window_rows])),
            "sampled_high_ratio": float(np.mean([float(dict(row["replay_stats"]).get("sampled_high_ratio", 0.0)) for row in self.window_rows])),
            "sampled_medium_ratio": float(np.mean([float(dict(row["replay_stats"]).get("sampled_medium_ratio", 0.0)) for row in self.window_rows])),
            "sampled_low_ratio": float(np.mean([float(dict(row["replay_stats"]).get("sampled_low_ratio", 0.0)) for row in self.window_rows])),
            "sampled_carryover_guidance_ratio": float(
                np.mean([float(dict(row["replay_stats"]).get("sampled_carryover_guidance_ratio", 0.0)) for row in self.window_rows])
            ),
            "sampled_local_avoidance_ratio": float(
                np.mean([float(dict(row["replay_stats"]).get("sampled_local_avoidance_ratio", 0.0)) for row in self.window_rows])
            ),
            "sampled_collision_ratio": float(np.mean([float(dict(row["replay_stats"]).get("sampled_collision_ratio", 0.0)) for row in self.window_rows])),
            "sampled_success_ratio": float(np.mean([float(dict(row["replay_stats"]).get("sampled_success_ratio", 0.0)) for row in self.window_rows])),
            "sampled_avoid_success_ratio": float(np.mean([float(dict(row["replay_stats"]).get("sampled_avoid_success_ratio", 0.0)) for row in self.window_rows])),
            "sampled_avoid_valuable_fail_ratio": float(np.mean([float(dict(row["replay_stats"]).get("sampled_avoid_valuable_fail_ratio", 0.0)) for row in self.window_rows])),
            "sampled_hard_negative_ratio": float(np.mean([float(dict(row["replay_stats"]).get("sampled_hard_negative_ratio", 0.0)) for row in self.window_rows])),
            "sampled_background_general_ratio": float(np.mean([float(dict(row["replay_stats"]).get("sampled_background_general_ratio", 0.0)) for row in self.window_rows])),
            "blocking_success_count": int(latest_replay_stats.get("blocking_success_count", 0)),
            "nonblocking_success_count": int(latest_replay_stats.get("nonblocking_success_count", 0)),
            "blocking_valuable_fail_count": int(latest_replay_stats.get("blocking_valuable_fail_count", 0)),
            "choose_side_count": int(latest_replay_stats.get("choose_side_count", 0)),
            "detour_progress_count": int(latest_replay_stats.get("detour_progress_count", 0)),
            "passed_block_count": int(latest_replay_stats.get("passed_block_count", 0)),
            "recover_to_goal_count": int(latest_replay_stats.get("recover_to_goal_count", 0)),
            "stall_slice_count": int(latest_replay_stats.get("stall_slice_count", 0)),
            "demo_success_ratio": float(latest_replay_stats.get("demo_success_ratio", 0.0)),
            "demo_valuable_fail_ratio": float(latest_replay_stats.get("demo_valuable_fail_ratio", 0.0)),
            "mean_phase_score": float(np.mean([float(dict(row["replay_stats"]).get("mean_phase_score", 0.0)) for row in self.window_rows])),
            "mean_episode_score": float(np.mean([float(dict(row["replay_stats"]).get("mean_episode_score", 0.0)) for row in self.window_rows])),
            "mean_n_step_return": float(np.mean([float(dict(row["replay_stats"]).get("mean_n_step_return", 0.0)) for row in self.window_rows])),
            "mean_tail_return": float(np.mean([float(dict(row["replay_stats"]).get("mean_tail_return", 0.0)) for row in self.window_rows])),
            "hard_negative_terminal_ratio": float(latest_replay_stats.get("hard_negative_terminal_ratio", 0.0)),
            "carryover_guidance_buffer_size": float(
                np.mean([float(dict(row["replay_stats"]).get("carryover_guidance_buffer_size", 0.0)) for row in self.window_rows])
            ),
            "carryover_guidance_priority_mean": float(
                np.mean([float(dict(row["replay_stats"]).get("carryover_guidance_priority_mean", 0.0)) for row in self.window_rows])
            ),
            "soft_timeout_steps_ratio": float(np.mean([float(row["episode_metrics"].get("soft_timeout_step_ratio", 0.0)) for row in self.window_rows])),
            "best_episode_id": int(best.get("episode_count", 0)),
            "best_episode_env": str(best_episode_metrics.get("curriculum_env", "")),
            "best_episode_success": bool(float(best_episode_metrics.get("episode_success", 0.0)) > 0.5),
            "best_episode_duration_seconds": float(best_episode_metrics.get("episode_duration_seconds", 0.0)),
            "best_episode_reward": float(best_episode_metrics.get("reward", 0.0)),
            **layout_split_stats,
            **q_stats,
            **probe_stats,
        }
        summary["stall_warning"] = bool(self._stall_status.get("stall_warning", False))
        summary["stall_reason"] = str(self._stall_status.get("stall_reason", ""))
        if self.promotion_controller is not None:
            summary.update(dict(self.promotion_controller.last_state))
        self.window_summaries.append(summary)
        window_best_dir = self.run_dir / "window_best"
        window_best_dir.mkdir(parents=True, exist_ok=True)
        best_payload.update(
            {
                "window_start_episode": window_start_episode,
                "window_end_episode": window_end_episode,
                "window_env": summary["window_env"],
                "window_summary": summary,
            }
        )
        stem = f"window_{window_end_episode:04d}"
        _write_json(window_best_dir / f"{stem}.json", best_payload)
        _save_episode_route_png(window_best_dir / f"{stem}.png", best_payload)
        success_field_svg_path = ""
        success_field_summary_path = ""
        success_field_payload_path = ""
        success_row = _select_success_window_episode(self.window_rows)
        if success_row and trainer is not None:
            success_episode_metrics = dict(success_row.get("episode_metrics", {}))
            success_episode_trace = dict(success_row.get("episode_trace", {}))
            success_field_dir = self.run_dir / "window_success_fields"
            success_field_dir.mkdir(parents=True, exist_ok=True)
            success_stem = f"{stem}_success"
            success_field_svg_path = str(success_field_dir / f"{success_stem}_action_field.svg")
            success_field_summary_path = str(success_field_dir / f"{success_stem}_action_field_summary.json")
            success_field_payload_path = str(success_field_dir / f"{success_stem}.json")
            target_position = np.asarray(success_episode_trace.get("target_position", [0.35, 0.0]), dtype=np.float32)
            success_action_field = _action_field_with_scene(
                trainer,
                success_field_dir,
                success_stem,
                stage_mode=str(getattr(trainer, "current_stage_mode", "avoidance")),
                scene=success_episode_trace,
                target_override=target_position,
            )
            _write_json(
                success_field_dir / f"{success_stem}.json",
                {
                    "window_start_episode": window_start_episode,
                    "window_end_episode": window_end_episode,
                    "episode_id": int(success_row.get("episode_count", 0)),
                    "curriculum_env": str(success_episode_metrics.get("curriculum_env", "")),
                    "episode_duration_seconds": float(success_episode_metrics.get("episode_duration_seconds", 0.0)),
                    "episode_reward": float(success_episode_metrics.get("reward", 0.0)),
                    "action_field": success_action_field,
                },
            )
        summary["success_action_field_svg_path"] = success_field_svg_path
        summary["success_action_field_summary_path"] = success_field_summary_path
        summary["success_action_field_payload_path"] = success_field_payload_path
        summary["has_success_action_field"] = bool(success_field_svg_path)
        self.window_rows.clear()

    def _write_live_files(self, payload: dict[str, Any]) -> None:
        history = list(payload.get("history", []))
        chunks = _chunk_summary(history, chunk_size=10)
        progress_payload = _build_live_progress_payload(
            callback_payload=payload,
            start_time=self.start_time,
            chunk_size=10,
        )
        progress_payload["window_summary_every"] = int(self.window_summary_every)
        progress_payload["window_summary_count"] = int(len(self.window_summaries))
        progress_payload["current_window_episode_count"] = int(len(self.window_rows))
        progress_payload["latest_window_summary"] = dict(self.window_summaries[-1]) if self.window_summaries else {}
        progress_payload["stall_warning"] = bool(self._stall_status.get("stall_warning", False))
        progress_payload["stall_reason"] = str(self._stall_status.get("stall_reason", ""))
        progress_payload["stall_details"] = _to_serializable(dict(self._stall_status.get("stall_details", {})))
        if self.promotion_controller is not None:
            progress_payload.update(dict(self.promotion_controller.last_state))
        _write_csv(self.run_dir / "live_block_history.csv", history)
        _write_csv(self.run_dir / "live_block_chunk_status.csv", chunks)
        _write_json(self.run_dir / "live_progress.json", progress_payload)
        _write_csv(self.run_dir / "live_window_summary.csv", self.window_summaries)
        _write_json(self.run_dir / "live_window_summary.json", {"windows": self.window_summaries})
        _write_json(
            self.run_dir / "stall_status.json",
            {
                "stall_warning": bool(self._stall_status.get("stall_warning", False)),
                "stall_reason": str(self._stall_status.get("stall_reason", "")),
                "stall_details": _to_serializable(dict(self._stall_status.get("stall_details", {}))),
            },
        )
        if self.promotion_controller is not None:
            _write_json(self.run_dir / "env_transition_history.json", {"transitions": self.promotion_controller.build_transition_rows()})
            _write_csv(self.run_dir / "env_step_summary.csv", self.promotion_controller.build_env_step_rows())


def _to_torch(value: Any) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(value, dict):
        return {key: _to_torch(item) for key, item in value.items()}
    array = np.asarray(value, dtype=np.float32)
    tensor = torch.from_numpy(array)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2 and array.shape[-1] in (2, 5):
        tensor = tensor.unsqueeze(0)
    return tensor


def _critic_profile_config(base: LocalRiskCriticConfig, profile: str) -> LocalRiskCriticConfig:
    profile_name = str(profile or "critic01_early_add_l1")
    profiles: dict[str, dict[str, Any]] = {
        "critic01_early_add_l1": {
            "num_layers": 1,
            "action_fusion_mode": "early_add",
            "q_head_include_action_feature": False,
            "q_head_include_raw_action": False,
        },
        "critic02_early_add_l2": {
            "num_layers": 2,
            "action_fusion_mode": "early_add",
            "q_head_include_action_feature": False,
            "q_head_include_raw_action": False,
        },
        "critic03_late_fusion_l1": {
            "num_layers": 1,
            "action_fusion_mode": "late_fusion",
            "q_head_include_action_feature": True,
            "q_head_include_raw_action": True,
        },
        "critic04_action_token_l1": {
            "num_layers": 1,
            "action_fusion_mode": "action_token",
            "q_head_include_action_feature": True,
            "q_head_include_raw_action": False,
        },
        "critic05_action_token_late_l2": {
            "num_layers": 2,
            "action_fusion_mode": "action_token",
            "q_head_include_action_feature": True,
            "q_head_include_raw_action": True,
        },
    }
    if profile_name not in profiles:
        msg = f"Unsupported critic_profile: {profile_name}"
        raise ValueError(msg)
    return replace(base, **profiles[profile_name])


def _build_network_config(
    *,
    q_head_dim: int = 512,
    mask_mode: str = "explicit",
    guidance_self_only: bool = True,
    transformer_num_avoidance_layers: int | None = None,
    transformer_num_collab_layers: int | None = None,
    critic_profile: str = "critic01_early_add_l1",
) -> PlanningNetworkConfig:
    cfg = build_planning_network_config("transformer_large")
    critic_cfg = _critic_profile_config(
        replace(cfg.local_risk_critic, q_head_dim=q_head_dim),
        critic_profile,
    )
    return replace(
        cfg,
        mask_mode=mask_mode,
        transformer=TransformerActorConfig(
            embed_dim=cfg.transformer.embed_dim,
            ff_dim=cfg.transformer.ff_dim,
            num_heads=cfg.transformer.num_heads,
            num_avoidance_layers=int(transformer_num_avoidance_layers)
            if transformer_num_avoidance_layers is not None
            else cfg.transformer.num_avoidance_layers,
            num_collab_layers=int(transformer_num_collab_layers)
            if transformer_num_collab_layers is not None
            else cfg.transformer.num_collab_layers,
            dropout=cfg.transformer.dropout,
            action_activation=cfg.transformer.action_activation,
            disable_collab_residual=cfg.transformer.disable_collab_residual,
            disable_explicit_mask=(mask_mode != "explicit"),
            guidance_self_only=guidance_self_only,
        ),
        local_risk_critic=critic_cfg,
    )


def _build_trainer(
    *,
    curriculum_env: str,
    mask_mode: str,
    guidance_self_only: bool,
    guidance_exploration_noise: float,
    exploration_noise: float,
    avoidance_noise_schedule_enabled: bool,
    avoidance_noise_stage_values: tuple[float, float, float],
    avoidance_noise_stage_ratios: tuple[float, float],
    actor_action_reg_weight: float,
    updates_per_step: float,
    critic_lr: float,
    q_head_dim: int,
    tau: float,
    timeout_seconds: float,
    target_hold_radius: float,
    transformer_num_avoidance_layers: int | None = None,
    transformer_num_collab_layers: int | None = None,
    critic_profile: str = "critic01_early_add_l1",
    guidance_replay_carryover_fraction: float = 0.0,
    guidance_replay_carryover_priority_scale: float = 0.35,
    guidance_replay_carryover_max_transitions: int = 0,
    guidance_replay_carryover_success_only: bool = True,
    replay_backend: str = "staged_pyramid",
    a1_skill_replay_config: str = "balanced",
    scenario_overrides: dict[str, Any] | None = None,
    env_overrides: dict[str, Any] | None = None,
    seed: int = 41,
) -> PlanningTrainer:
    cfg = _build_network_config(
        q_head_dim=q_head_dim,
        mask_mode=mask_mode,
        guidance_self_only=guidance_self_only,
        transformer_num_avoidance_layers=transformer_num_avoidance_layers,
        transformer_num_collab_layers=transformer_num_collab_layers,
        critic_profile=critic_profile,
    )
    scenario_kwargs: dict[str, Any] = {
        "max_obstacles": cfg.max_obstacles,
        "max_neighbors": cfg.max_neighbors,
        "timeout_seconds": timeout_seconds,
        "target_hold_radius": target_hold_radius,
    }
    if scenario_overrides:
        scenario_kwargs.update(dict(scenario_overrides))
    scenario = replace(
        build_planning_scenario(curriculum_env=curriculum_env),
        **scenario_kwargs,
    )
    env = PlanningSwarmEnv(
        scenario=scenario,
        self_dim=cfg.self_dim,
        obstacle_dim=cfg.obstacle_dim,
        neighbor_dim=cfg.neighbor_dim,
        action_limit=cfg.action_limit,
    )
    if env_overrides:
        for key, value in env_overrides.items():
            setattr(env, key, value)
        env.__post_init__()
    return PlanningTrainer(
        env=env,
        network_config=cfg,
        tau=tau,
        actor_lr=1e-4,
        critic_lr=critic_lr,
        batch_size=128,
        warmup_steps=600,
        updates_per_step=updates_per_step,
        exploration_noise=exploration_noise,
        guidance_exploration_noise=guidance_exploration_noise,
        avoidance_noise_schedule_enabled=avoidance_noise_schedule_enabled,
        avoidance_noise_stage_values=tuple(float(value) for value in avoidance_noise_stage_values),
        avoidance_noise_stage_ratios=tuple(float(value) for value in avoidance_noise_stage_ratios),
        guidance_replay_carryover_fraction=guidance_replay_carryover_fraction,
        guidance_replay_carryover_priority_scale=guidance_replay_carryover_priority_scale,
        guidance_replay_carryover_max_transitions=guidance_replay_carryover_max_transitions,
        guidance_replay_carryover_success_only=guidance_replay_carryover_success_only,
        replay_backend=replay_backend,
        a1_skill_replay_config=a1_skill_replay_config,
        actor_action_reg_weight=actor_action_reg_weight,
        seed=seed,
    )


def _default_a1_overrides() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {
            "num_obstacles": 1,
            "obstacle_count_range": (1, 1),
            "obstacle_size_small_range": (1, 1),
            "obstacle_size_medium_range": (0, 0),
            "obstacle_size_large_range": (0, 0),
            "obstacle_layout_modes": ("path_center_offset_nonblocking", "path_center_blocking"),
            "a1_direct_block_ratio": 0.4,
            "a1_blocking_layout": "path_center_blocking",
            "a1_nonblocking_layout": "path_center_offset_nonblocking",
            "target_clearance_radius": 0.30,
            "target_hold_radius": 0.10,
            "target_region_mode": "upper_right_square",
            "target_region_bounds": (0.70, 0.95, 0.55, 0.95),
        },
        {
            "small_obstacle_radius": 0.06,
            "action_hz": 48,
            "control_hz": 48,
            "delta_v_max": 9.6,
        },
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(_to_serializable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(target)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _reward_component_summary(reward_components: list[dict[str, Any]]) -> dict[str, float]:
    if not reward_components:
        return {}
    keys = sorted({str(key) for row in reward_components for key in row.keys()})
    return {
        key: float(np.mean([float(row.get(key, 0.0)) for row in reward_components]))
        for key in keys
    }


def _layout_group(layout_name: str) -> str:
    layout = str(layout_name)
    if "nonblocking" in layout:
        return "nonblocking"
    if "blocking" in layout:
        return "blocking"
    return "other"


def _window_layout_split_stats(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    stats: dict[str, float | int] = {}
    for group_name in ("blocking", "nonblocking"):
        group_rows = [
            row
            for row in rows
            if _layout_group(str(row["episode_metrics"].get("sampled_obstacle_layout", ""))) == group_name
        ]
        prefix = f"{group_name}_"
        stats[f"{prefix}episode_count"] = int(len(group_rows))
        stats[f"{prefix}success_count"] = int(
            sum(float(row["episode_metrics"].get("episode_success", 0.0)) > 0.5 for row in group_rows)
        )
        stats[f"{prefix}collision_count"] = int(
            sum(float(row["episode_metrics"].get("episode_collision", 0.0)) > 0.5 for row in group_rows)
        )
        stats[f"{prefix}out_of_bounds_count"] = int(
            sum(float(row["episode_metrics"].get("episode_out_of_bounds", 0.0)) > 0.5 for row in group_rows)
        )
        stats[f"{prefix}soft_timeout_count"] = int(
            sum(float(row["episode_metrics"].get("soft_timeout_episode", 0.0)) > 0.5 for row in group_rows)
        )
        stats[f"{prefix}mean_final_target_distance"] = (
            float(np.mean([float(row["episode_metrics"].get("final_target_distance", 0.0)) for row in group_rows]))
            if group_rows
            else 0.0
        )
    return stats


def _episode_route_payload(
    *,
    episode_id: int,
    episode_metrics: dict[str, Any],
    episode_trace: dict[str, Any],
    reward_components: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "episode_id": int(episode_id),
        "success": bool(float(episode_metrics.get("episode_success", 0.0)) > 0.5),
        "collision": bool(float(episode_metrics.get("episode_collision", 0.0)) > 0.5),
        "timeout": bool(float(episode_metrics.get("episode_timeout", 0.0)) > 0.5),
        "out_of_bounds": bool(float(episode_metrics.get("episode_out_of_bounds", 0.0)) > 0.5),
        "sampled_obstacle_layout": str(episode_metrics.get("sampled_obstacle_layout", "")),
        "trajectory": _to_serializable(episode_trace.get("trajectory", [])),
        "target_position": _to_serializable(episode_trace.get("target_position", [])),
        "target_trajectory": _to_serializable(episode_trace.get("target_trajectory", [])),
        "obstacle_positions_initial": _to_serializable(episode_trace.get("obstacle_positions_initial", [])),
        "obstacle_positions_final": _to_serializable(episode_trace.get("obstacle_positions_final", [])),
        "obstacle_radii_initial": _to_serializable(episode_trace.get("obstacle_radii_initial", [])),
        "obstacle_safe_radii_initial": _to_serializable(episode_trace.get("obstacle_safe_radii_initial", [])),
        "uav_collision_radius": float(episode_trace.get("uav_collision_radius", 0.10)),
        "reward_risk_margin": 0.05,
        "workspace_bounds": _to_serializable(episode_trace.get("workspace_bounds", [])),
        "reward_components": _to_serializable(reward_components),
        "reward_component_means": _reward_component_summary(reward_components),
        "final_target_distance": float(episode_metrics.get("final_target_distance", 0.0)),
        "min_obstacle_clearance": float(episode_metrics.get("min_obstacle_clearance", 10.0)),
        "velocity_delta_mean": float(episode_metrics.get("velocity_delta_mean", 0.0)),
        "velocity_delta_max": float(episode_metrics.get("velocity_delta_max", 0.0)),
        "acceleration_clip_rate": float(episode_metrics.get("acceleration_clip_rate", 0.0)),
    }


def _save_episode_route_png(path: Path, payload: dict[str, Any]) -> None:
    trajectory = np.asarray(payload.get("trajectory", []), dtype=float)
    target_trajectory = np.asarray(payload.get("target_trajectory", []), dtype=float)
    target_position = np.asarray(payload.get("target_position", []), dtype=float)
    obstacle_positions = np.asarray(payload.get("obstacle_positions_initial", []), dtype=float)
    obstacle_radii = np.asarray(payload.get("obstacle_radii_initial", []), dtype=float)
    uav_collision_radius = float(payload.get("uav_collision_radius", 0.10))
    reward_risk_margin = float(payload.get("reward_risk_margin", 0.0))
    workspace_bounds = payload.get("workspace_bounds", [])

    width = 960
    height = 960
    padding = 48
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    if len(workspace_bounds) == 4:
        xmin, xmax, ymin, ymax = [float(v) for v in workspace_bounds]
        if not (all(np.isfinite([xmin, xmax, ymin, ymax])) and xmin < xmax and ymin < ymax):
            xmin, xmax, ymin, ymax = -1.0, 1.0, -1.0, 1.0
    else:
        all_points = []
        if trajectory.size:
            all_points.extend(trajectory.tolist())
        if target_trajectory.size:
            all_points.extend(target_trajectory.tolist())
        if target_position.size == 2:
            all_points.append(target_position.tolist())
        if obstacle_positions.size:
            all_points.extend(obstacle_positions.tolist())
        if all_points:
            xs = [float(p[0]) for p in all_points if len(p) >= 2]
            ys = [float(p[1]) for p in all_points if len(p) >= 2]
            xmin, xmax = min(xs) - 0.2, max(xs) + 0.2
            ymin, ymax = min(ys) - 0.2, max(ys) + 0.2
        else:
            xmin, xmax, ymin, ymax = -1.0, 1.0, -1.0, 1.0

    x_span = max(1e-6, xmax - xmin)
    y_span = max(1e-6, ymax - ymin)

    def project(point: np.ndarray | list[float] | tuple[float, float]) -> tuple[float, float]:
        x = float(point[0])
        y = float(point[1])
        px = padding + (x - xmin) / x_span * (width - 2 * padding)
        py = height - padding - (y - ymin) / y_span * (height - 2 * padding)
        return px, py

    def scaled_radius(radius: float) -> float:
        pixels_per_unit = min((width - 2 * padding) / x_span, (height - 2 * padding) / y_span)
        return max(2.0, float(radius) * pixels_per_unit)

    draw.rectangle((padding, padding, width - padding, height - padding), outline=(180, 180, 180), width=2)

    for index, center in enumerate(obstacle_positions):
        if not np.isfinite(center).all():
            continue
        radius = float(obstacle_radii[index]) if index < len(obstacle_radii) else 0.0
        if not np.isfinite(radius) or radius <= 0.0:
            continue
        collision_boundary_radius = radius + uav_collision_radius
        risk_boundary_radius = collision_boundary_radius + reward_risk_margin
        cx, cy = project(center)
        rr = scaled_radius(radius)
        collision_r = scaled_radius(collision_boundary_radius)
        risk_r = scaled_radius(risk_boundary_radius)
        draw.ellipse((cx - risk_r, cy - risk_r, cx + risk_r, cy + risk_r), outline=(255, 140, 0), width=2)
        draw.ellipse((cx - collision_r, cy - collision_r, cx + collision_r, cy + collision_r), outline=(200, 50, 50), width=2)
        draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=(190, 40, 40), width=3)

    if target_trajectory.size and len(target_trajectory) >= 2:
        draw.line([project(point) for point in target_trajectory if np.isfinite(point).all()], fill=(255, 127, 14), width=2)
    if trajectory.size and len(trajectory) >= 2:
        draw.line([project(point) for point in trajectory if np.isfinite(point).all()], fill=(31, 119, 180), width=4)
    uav_r = scaled_radius(uav_collision_radius)
    if trajectory.size:
        for point in trajectory[1:-1]:
            if not np.isfinite(point).all():
                continue
            px, py = project(point)
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(80, 140, 220))
    if trajectory.size:
        sx, sy = project(trajectory[0])
        ex, ey = project(trajectory[-1])
        draw.ellipse((sx - 6, sy - 6, sx + 6, sy + 6), fill=(44, 160, 44))
        draw.ellipse((sx - uav_r, sy - uav_r, sx + uav_r, sy + uav_r), outline=(44, 160, 44), width=2)
        draw.ellipse((ex - 6, ey - 6, ex + 6, ey + 6), fill=(214, 39, 40))
        draw.ellipse((ex - uav_r, ey - uav_r, ex + uav_r, ey + uav_r), outline=(31, 119, 180), width=2)
    if target_position.size == 2 and np.isfinite(target_position).all():
        tx, ty = project(target_position)
        draw.line((tx - 8, ty, tx + 8, ty), fill=(148, 103, 189), width=3)
        draw.line((tx, ty - 8, tx, ty + 8), fill=(148, 103, 189), width=3)

    legend_x = padding + 10
    legend_y = padding + 10
    draw.rectangle((legend_x, legend_y, legend_x + 300, legend_y + 112), fill=(255, 255, 255), outline=(220, 220, 220))
    draw.line((legend_x + 12, legend_y + 20, legend_x + 42, legend_y + 20), fill=(31, 119, 180), width=4)
    draw.text((legend_x + 52, legend_y + 12), "UAV path", fill=(20, 20, 20))
    draw.ellipse((legend_x + 12, legend_y + 40, legend_x + 42, legend_y + 70), outline=(190, 40, 40), width=3)
    draw.text((legend_x + 52, legend_y + 42), "Obstacle body", fill=(20, 20, 20))
    draw.ellipse((legend_x + 12, legend_y + 72, legend_x + 42, legend_y + 102), outline=(200, 50, 50), width=2)
    draw.text((legend_x + 52, legend_y + 74), "Collision boundary", fill=(20, 20, 20))
    draw.ellipse((legend_x + 168, legend_y + 72, legend_x + 198, legend_y + 102), outline=(255, 140, 0), width=2)
    draw.text((legend_x + 208, legend_y + 74), "Reward risk boundary", fill=(20, 20, 20))

    image.save(path)


def _save_episode_route_artifacts(
    *,
    run_dir: Path,
    episode_id: int,
    episode_metrics: dict[str, Any],
    episode_trace: dict[str, Any],
    reward_components: list[dict[str, Any]],
) -> None:
    routes_dir = run_dir / "episode_routes"
    routes_dir.mkdir(parents=True, exist_ok=True)
    payload = _episode_route_payload(
        episode_id=episode_id,
        episode_metrics=episode_metrics,
        episode_trace=episode_trace,
        reward_components=reward_components,
    )
    stem = f"episode_{episode_id:04d}"
    _write_json(routes_dir / f"{stem}.json", payload)
    _save_episode_route_png(routes_dir / f"{stem}.png", payload)


def _chunk_summary(history: list[dict[str, object]], chunk_size: int = 10) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for start in range(0, len(history), chunk_size):
        rows = history[start : start + chunk_size]
        if not rows:
            continue
        chunks.append(
            {
                "episodes_total": start + len(rows),
                "successes_in_chunk": int(sum(1 for row in rows if float(row["success_rate"]) > 0.5)),
                "out_of_bounds_in_chunk": int(sum(1 for row in rows if str(row["failure_reason"]) == "out_of_bounds")),
                "timeout_in_chunk": int(sum(1 for row in rows if str(row["failure_reason"]) == "timeout")),
                "collision_in_chunk": int(sum(1 for row in rows if str(row["failure_reason"]) == "collision")),
                "chunk_mean_reward": float(sum(float(row["reward"]) for row in rows) / len(rows)),
                "chunk_mean_occupancy_error": float(sum(float(row["occupancy_error"]) for row in rows) / len(rows)),
                "action_abs_mean": float(sum(float(row["action_abs_mean"]) for row in rows) / len(rows)),
                "action_abs_max": float(max(float(row["action_abs_max"]) for row in rows)),
                "actor_output_saturation_rate": float(sum(float(row["actor_output_saturation_rate"]) for row in rows) / len(rows)),
            }
        )
    return chunks


def _tail_counts(chunks: list[dict[str, object]], tail_chunks: int = 5) -> dict[str, int]:
    tail = chunks[-tail_chunks:] if len(chunks) >= tail_chunks else chunks
    return {
        "tail_successes": int(sum(int(row["successes_in_chunk"]) for row in tail)),
        "tail_out_of_bounds": int(sum(int(row["out_of_bounds_in_chunk"]) for row in tail)),
        "tail_timeout": int(sum(int(row["timeout_in_chunk"]) for row in tail)),
        "tail_collision": int(sum(int(row["collision_in_chunk"]) for row in tail)),
    }


def _reward_decomposition_summary(history: list[dict[str, object]], *, tail_episodes: int = 50) -> dict[str, float]:
    tail = history[-tail_episodes:] if len(history) >= tail_episodes else history
    if not tail:
        return {key: 0.0 for key in REWARD_DECOMP_KEYS}
    return {
        key: float(sum(float(row.get(key, 0.0)) for row in tail) / len(tail))
        for key in REWARD_DECOMP_KEYS
    }


def _legacy_task_performance(
    trainer: PlanningTrainer,
    completed_stages: list[str],
    *,
    episodes: int = 5,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for env_name in completed_stages:
        metrics = trainer.evaluate_on_scenario(
            build_planning_scenario(curriculum_env=env_name),
            episodes=episodes,
        )
        results[env_name] = {
            "episodes": int(episodes),
            "success_rate": float(metrics.get("success_rate", 0.0)),
            "collision_rate": float(metrics.get("collision_rate", 0.0)),
            "timeout_rate": float(metrics.get("timeout_rate", 0.0)),
            "out_of_bounds_rate": float(metrics.get("out_of_bounds_rate", 0.0)),
            "mean_reward": float(metrics.get("reward", 0.0)),
            "mean_occupancy_error": float(metrics.get("occupancy_error", 0.0)),
        }
    return results


def _q_value_distribution(
    trainer: PlanningTrainer,
    *,
    block_episode_count: int,
    sample_window: int = 50,
) -> dict[str, float | str]:
    min_episode_id = max(1, trainer.episode_id_counter - sample_window + 1)
    rows = [
        payload
        for payload in trainer.replay_manager.buffer
        if payload is not None and int(payload.get("episode_id", -1)) >= min_episode_id
    ]
    if not rows:
        return {"q_value_mean": 0.0, "q_value_max": 0.0, "q_value_min": 0.0, "q_value_comment": "no_q_samples"}

    q_values: list[float] = []
    original_stage_mode = trainer.current_stage_mode
    with torch.no_grad():
        for payload in rows:
            obs = _to_torch(payload["obs"])
            flat_obs = torch.tensor(trainer._flatten_obs(payload["obs"]), dtype=torch.float32).unsqueeze(0)
            stage_name = str(payload.get("stage_name", trainer.current_stage_mode))
            trainer.current_stage_mode = stage_name
            action = trainer._td3_actor_forward(trainer.actor, obs, flat_obs)
            q1 = trainer._td3_critic_forward(trainer.critic_1, obs, flat_obs, action)
            q2 = trainer._td3_critic_forward(trainer.critic_2, obs, flat_obs, action)
            q_values.append(float(torch.minimum(q1, q2).item()))
    trainer.current_stage_mode = original_stage_mode

    q_mean = float(np.mean(q_values))
    q_max = float(np.max(q_values))
    q_min = float(np.min(q_values))
    if q_max > q_mean + 20.0 and q_min < q_mean - 20.0:
        comment = "wide_q_spread"
    elif q_min > 0.0 or q_max < 0.0:
        comment = "possible_q_bias"
    else:
        comment = "q_ordering_looks_reasonable"
    return {
        "q_value_mean": q_mean,
        "q_value_max": q_max,
        "q_value_min": q_min,
        "q_value_comment": comment,
    }


def _recommended_next_action(
    *,
    curriculum_env: str,
    tail_successes: int,
    tail_out_of_bounds: int,
    tail_timeout: int,
    tail_collision: int,
    legacy_task_performance: dict[str, dict[str, float]],
    obstacle_response_probe: dict[str, object] | None = None,
) -> str:
    legacy_drop = any(float(metrics.get("success_rate", 0.0)) < 0.8 for metrics in legacy_task_performance.values())
    if curriculum_env.startswith("avoidance"):
        if legacy_drop:
            return "reduce_avoidance_noise_or_replay_mix"
        obstacle_delta = float((obstacle_response_probe or {}).get("obstacle_left_vs_right_action_delta", 0.0))
        center_delta = float((obstacle_response_probe or {}).get("obstacle_center_vs_clear_action_delta", 0.0))
        if tail_collision >= max(tail_successes, 10) and tail_out_of_bounds == 0:
            return "refine_obstacle_response"
        if tail_successes >= 25 and obstacle_delta >= 0.2 and center_delta >= 0.2:
            return "continue_avoidance_block"
        if tail_successes < 15 and obstacle_delta < 0.1 and center_delta < 0.1:
            return "tighten_obstacle_geometry_or_reward"
        return "continue_avoidance_block"
    if legacy_drop:
        return "stabilize_before_stage_change"
    if tail_successes >= 30 and tail_out_of_bounds == 0 and tail_timeout <= 10:
        return "advance_to_next_stage"
    if tail_timeout > 20:
        return "keep_env_reduce_noise"
    return "keep_env_keep_noise"


def _build_live_progress_payload(
    *,
    callback_payload: dict[str, Any],
    start_time: float,
    chunk_size: int,
) -> dict[str, Any]:
    history = list(callback_payload.get("history", []))
    episode_count = int(callback_payload.get("episode_count", 0))
    total_episodes = int(callback_payload.get("total_episodes", 0))
    training_counters = dict(callback_payload.get("training_counters", {}))
    target_total_env_steps = int(training_counters.get("target_total_env_steps", 0))
    actual_total_env_steps = int(training_counters.get("actual_total_env_steps", 0))
    elapsed_seconds = max(0.0, float(time.perf_counter() - start_time))
    avg_seconds_per_episode = elapsed_seconds / max(1, episode_count)
    remaining_episodes = max(0, total_episodes - episode_count)
    eta_seconds = avg_seconds_per_episode * remaining_episodes
    if target_total_env_steps > 0 and actual_total_env_steps > 0:
        avg_seconds_per_step = elapsed_seconds / max(1, actual_total_env_steps)
        remaining_steps = max(0, target_total_env_steps - actual_total_env_steps)
        eta_seconds = avg_seconds_per_step * remaining_steps
    recent_rows = history[-chunk_size:] if history else []
    recent_episode_count = max(1, len(recent_rows))
    recent_reward_mean = float(sum(float(row.get("reward", 0.0)) for row in recent_rows) / recent_episode_count)
    recent_success_rate_mean = float(sum(float(row.get("success_rate", 0.0)) for row in recent_rows) / recent_episode_count)
    recent_collision_rate_mean = float(sum(float(row.get("collision_rate", 0.0)) for row in recent_rows) / recent_episode_count)
    recent_timeout_rate_mean = float(sum(float(row.get("timeout_rate", 0.0)) for row in recent_rows) / recent_episode_count)
    recent_out_of_bounds_rate_mean = float(sum(float(row.get("out_of_bounds_rate", 0.0)) for row in recent_rows) / recent_episode_count)
    reward_decomposition = _reward_decomposition_summary(history, tail_episodes=50)
    replay_stats = dict(callback_payload.get("replay_stats", {}))
    time_warning = ""
    if avg_seconds_per_episode > 15.0:
        time_warning = "slow_episode_runtime"
    if elapsed_seconds > 0.0 and episode_count >= chunk_size and eta_seconds > elapsed_seconds * 2.5:
        time_warning = "eta_much_larger_than_elapsed"
    return {
        "episode_count": episode_count,
        "total_episodes": total_episodes,
        "remaining_episodes": remaining_episodes,
        "target_total_env_steps": target_total_env_steps,
        "actual_total_env_steps": actual_total_env_steps,
        "elapsed_seconds": float(elapsed_seconds),
        "avg_seconds_per_episode": float(avg_seconds_per_episode),
        "eta_seconds": float(eta_seconds),
        "chunk_size": int(chunk_size),
        "recent_reward_mean": recent_reward_mean,
        "recent_success_rate_mean": recent_success_rate_mean,
        "recent_collision_rate_mean": recent_collision_rate_mean,
        "recent_timeout_rate_mean": recent_timeout_rate_mean,
        "recent_out_of_bounds_rate_mean": recent_out_of_bounds_rate_mean,
        "collision_rate": recent_collision_rate_mean,
        "timeout_rate": recent_timeout_rate_mean,
        "out_of_bounds_rate": recent_out_of_bounds_rate_mean,
        "soft_timeout_episode_count": int(sum(float(row.get("soft_timeout_episode", 0.0)) > 0.5 for row in recent_rows)),
        "episode_step_cap_count": int(sum(float(row.get("episode_step_cap_hit", 0.0)) > 0.5 for row in recent_rows)),
        "soft_timeout_steps_ratio": float(sum(float(row.get("soft_timeout_step_ratio", 0.0)) for row in recent_rows) / recent_episode_count),
        "risk_drop_reward_mean": float(reward_decomposition.get("risk_drop_reward", 0.0)),
        "clearance_gain_reward_mean": float(reward_decomposition.get("clearance_gain_reward", 0.0)),
        "detour_trend_reward_mean": float(reward_decomposition.get("detour_trend_reward", 0.0)),
        "near_collision_penalty_mean": float(reward_decomposition.get("near_collision_penalty", 0.0)),
        "critical_collision_penalty_mean": float(reward_decomposition.get("critical_collision_margin_penalty", 0.0)),
        "action_saturation_penalty_mean": float(reward_decomposition.get("action_saturation_penalty", 0.0)),
        "action_change_penalty_mean": float(reward_decomposition.get("action_change_penalty", 0.0)),
        "velocity_delta_mean": float(sum(float(row.get("velocity_delta_mean", 0.0)) for row in recent_rows) / recent_episode_count),
        "velocity_delta_max": float(max((float(row.get("velocity_delta_max", 0.0)) for row in recent_rows), default=0.0)),
        "acceleration_clip_rate": float(sum(float(row.get("acceleration_clip_rate", 0.0)) for row in recent_rows) / recent_episode_count),
        "current_exploration_noise": float(callback_payload.get("episode_metrics", {}).get("current_exploration_noise", 0.0)),
        "exploration_stage_index": int(callback_payload.get("episode_metrics", {}).get("exploration_stage_index", 0)),
        "reward_decomposition_tail50": reward_decomposition,
        "latest_episode_metrics": dict(callback_payload.get("episode_metrics", {})),
        "training_counters": training_counters,
        "replay_stats": replay_stats,
        "carryover_guidance_buffer_size": int(replay_stats.get("carryover_guidance_buffer_size", 0)),
        "sampled_carryover_guidance_ratio": float(replay_stats.get("sampled_carryover_guidance_ratio", 0.0)),
        "sampled_local_avoidance_ratio": float(replay_stats.get("sampled_local_avoidance_ratio", 0.0)),
        "carryover_guidance_priority_mean": float(replay_stats.get("carryover_guidance_priority_mean", 0.0)),
        "seed_demo_buffer_size": int(replay_stats.get("seed_demo_buffer_size", 0)),
        "sampled_seed_demo_ratio": float(replay_stats.get("sampled_seed_demo_ratio", 0.0)),
        "sampled_seed_success_ratio": float(replay_stats.get("sampled_seed_success_ratio", 0.0)),
        "sampled_seed_valuable_fail_ratio": float(replay_stats.get("sampled_seed_valuable_fail_ratio", 0.0)),
        "seed_left_detour_count": int(replay_stats.get("seed_left_detour_count", 0)),
        "seed_right_detour_count": int(replay_stats.get("seed_right_detour_count", 0)),
        "seed_left_vs_right_balance": float(replay_stats.get("seed_left_vs_right_balance", 0.0)),
        "time_warning": time_warning,
    }


def _build_progress_callback(
    *,
    run_dir: Path,
    write_every: int,
    route_save_every: int,
    chunk_size: int = 10,
) -> tuple[Any, dict[str, float]]:
    timing_state = {"start_time": float(time.perf_counter())}

    def _callback(payload: dict[str, Any]) -> None:
        is_heartbeat = bool(payload.get("heartbeat", False))
        episode_count = int(payload.get("episode_count", 0))
        total_episodes = int(payload.get("total_episodes", 0))
        if not is_heartbeat and route_save_every > 0 and episode_count > 0 and episode_count % route_save_every == 0:
            _save_episode_route_artifacts(
                run_dir=run_dir,
                episode_id=episode_count,
                episode_metrics=dict(payload.get("episode_metrics", {})),
                episode_trace=dict(payload.get("episode_trace", {})),
                reward_components=list(payload.get("reward_components", [])),
            )
        should_write = is_heartbeat or episode_count == total_episodes or (write_every > 0 and episode_count % write_every == 0)
        if total_episodes <= 0 and not is_heartbeat:
            should_write = write_every > 0 and episode_count % write_every == 0
        if not should_write:
            return
        history = list(payload.get("history", []))
        chunks = _chunk_summary(history, chunk_size=chunk_size)
        progress_payload = _build_live_progress_payload(
            callback_payload=payload,
            start_time=timing_state["start_time"],
            chunk_size=chunk_size,
        )
        _write_csv(run_dir / "live_block_history.csv", history)
        _write_csv(run_dir / "live_block_chunk_status.csv", chunks)
        _write_json(run_dir / "live_progress.json", progress_payload)

    return _callback, timing_state


def _import_guidance_carryover_from_checkpoint(
    trainer: PlanningTrainer,
    checkpoint_path: str | Path,
    *,
    fraction: float,
    priority_scale: float,
    max_transitions: int,
    success_only: bool,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    replay_state = checkpoint.get("replay_state", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(replay_state, dict) or not replay_state:
        return {
            "imported": False,
            "reason": "missing_replay_state",
            "selected": 0,
            "available": 0,
            "source_format": "missing",
        }
    result = trainer.replay_manager.import_guidance_carryover_from_replay_state(
        replay_state,
        fraction=float(fraction),
        priority_scale=float(priority_scale),
        max_transitions=int(max_transitions) if int(max_transitions) > 0 else None,
        success_only=bool(success_only),
    )
    return {
        "imported": bool(int(result.get("selected", 0)) > 0),
        "reason": "ok" if int(result.get("selected", 0)) > 0 else "no_eligible_guidance_samples",
        **dict(result),
    }


def _load_a1_seed_demo_records(
    trainer: PlanningTrainer,
    demo_dir: str | Path,
) -> dict[str, Any]:
    episodes = load_teacher_episodes_from_dir(demo_dir)
    left_records: list[Any] = []
    right_records: list[Any] = []
    neutral_records: list[Any] = []
    for episode in episodes:
        converted = convert_teacher_episode_to_transition_records(episode)
        side = str(episode.expected_detour_side)
        if side == "left":
            left_records.extend(converted)
        elif side == "right":
            right_records.extend(converted)
        else:
            neutral_records.extend(converted)
    if not hasattr(trainer.replay_manager, "avoidance_seed_demo"):
        records = [*left_records, *right_records, *neutral_records]
        result = trainer.replay_manager.load_seed_demo_records(records)
        return {
            "loaded": bool(int(result.get("inserted", 0)) > 0),
            "episode_count": int(len(episodes)),
            "record_count": int(len(records)),
            "available_left_record_count": int(len(left_records)),
            "available_right_record_count": int(len(right_records)),
            "selected_left_record_count": int(len(left_records)),
            "selected_right_record_count": int(len(right_records)),
            **dict(result),
        }
    capacity = int(getattr(trainer.replay_manager.avoidance_seed_demo, "capacity", 2000))
    half_capacity = max(1, capacity // 2)
    selected_left = left_records[:half_capacity]
    selected_right = right_records[:half_capacity]
    remaining_capacity = max(0, capacity - len(selected_left) - len(selected_right))
    records = [*selected_left, *selected_right, *neutral_records[:remaining_capacity]]
    result = trainer.replay_manager.load_seed_demo_records(records)
    return {
        "loaded": bool(int(result.get("inserted", 0)) > 0),
        "episode_count": int(len(episodes)),
        "record_count": int(len(records)),
        "available_left_record_count": int(len(left_records)),
        "available_right_record_count": int(len(right_records)),
        "selected_left_record_count": int(len(selected_left)),
        "selected_right_record_count": int(len(selected_right)),
        **dict(result),
    }


def _load_weights_with_actor_compatibility(
    trainer: PlanningTrainer,
    checkpoint_path: str | Path,
) -> None:
    payload = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    actor_result = trainer.actor.load_state_dict(payload["actor"], strict=False)
    trainer.target_actor.load_state_dict(payload["target_actor"], strict=False)
    trainer.critic_1.load_state_dict(payload["critic_1"])
    trainer.target_critic_1.load_state_dict(payload["target_critic_1"])
    trainer.critic_2.load_state_dict(payload["critic_2"])
    trainer.target_critic_2.load_state_dict(payload["target_critic_2"])
    trainer.global_env_steps = int(payload.get("global_env_steps", trainer.global_env_steps))
    trainer.update_step = int(payload.get("update_step", trainer.update_step))
    trainer.pending_updates = float(payload.get("pending_updates", trainer.pending_updates))
    trainer.episode_id_counter = int(payload.get("episode_id_counter", trainer.episode_id_counter))
    trainer.current_stage_mode = str(payload.get("current_stage_mode", trainer.current_stage_mode))
    trainer.retired_replay_stages = set(payload.get("retired_replay_stages", []))
    trainer._apply_stage_freeze(trainer.current_stage_mode)
    trainer._rebuild_optimizers()
    setattr(
        trainer,
        "_last_actor_compatibility_load",
        {
            "missing_keys": list(actor_result.missing_keys),
            "unexpected_keys": list(actor_result.unexpected_keys),
        },
    )


def run_manual_block(
    output_dir: str | Path,
    *,
    block_id: int = 1,
    run_name: str | None = None,
    curriculum_env: str = "guidance_G1",
    episodes_per_block: int = 300,
    env_step_budget: int | None = None,
    guidance_exploration_noise: float = 0.30,
    exploration_noise: float = 0.10,
    avoidance_noise_schedule_enabled: bool = False,
    avoidance_noise_stage_values: tuple[float, float, float] = (0.20, 0.12, 0.06),
    avoidance_noise_stage_ratios: tuple[float, float] = (0.30, 0.70),
    actor_action_reg_weight: float = 5e-3,
    updates_per_step: float = 0.12,
    mask_mode: str = "explicit",
    use_curriculum: bool = False,
    resume_from: str | Path | None = None,
    resume_weights: bool = True,
    resume_replay: bool = True,
    resume_optimizer: bool = False,
    completed_stages: list[str] | None = None,
    scenario_overrides: dict[str, Any] | None = None,
    env_overrides: dict[str, Any] | None = None,
    live_progress_every: int = 0,
    route_save_every: int = 0,
    window_summary_every: int = 20,
    checkpoint_save_every: int = 0,
    progress_heartbeat_steps: int = 0,
    transformer_num_avoidance_layers: int | None = None,
    transformer_num_collab_layers: int | None = None,
    critic_profile: str = "critic01_early_add_l1",
    rolling_promotion_sequence: list[str] | None = None,
    rolling_promotion_window: int = 20,
    rolling_promotion_min_successes: int = 15,
    rolling_promotion_max_duration_seconds: float = 30.0,
    rolling_promotion_max_episodes_per_env: int = 0,
    rolling_review_mix_enabled: bool = False,
    rolling_review_current_sample_count: int = 9,
    rolling_review_previous_sample_count: int = 1,
    rolling_review_rollback_previous_failures: int = 3,
    rolling_review_max_collision_rate: float = 0.30,
    guidance_replay_carryover_fraction: float = 0.0,
    guidance_replay_carryover_priority_scale: float = 0.35,
    guidance_replay_carryover_max_transitions: int = 0,
    guidance_replay_carryover_success_only: bool = True,
    load_a1_seed_demo: bool = False,
    a1_seed_demo_dir: str | None = None,
    a1_seed_demo_sample_ratio: float = 0.10,
    avoidance_local_sample_ratios: tuple[float, float, float] | None = None,
    replay_backend: str = "staged_pyramid",
    a1_skill_replay_config: str | None = None,
    a1_skill_demo_bootstrap_dir: str | None = None,
    a1_skill_demo_decay_enabled: bool = True,
    seed: int = 41,
) -> dict[str, Any]:
    completed_stages = list(completed_stages or [])
    effective_scenario_overrides = dict(scenario_overrides or {})
    effective_env_overrides = dict(env_overrides or {})
    skill_replay_enabled = replay_backend == "a1_skill" or str(curriculum_env).startswith("avoidance_A1_skill_replay")
    if skill_replay_enabled:
        replay_backend = "a1_skill"
        a1_skill_replay_config = a1_skill_replay_config or "balanced"
        transformer_num_avoidance_layers = 3 if transformer_num_avoidance_layers is None else transformer_num_avoidance_layers
        transformer_num_collab_layers = 1 if transformer_num_collab_layers is None else transformer_num_collab_layers
        guidance_replay_carryover_fraction = 0.0
        guidance_replay_carryover_priority_scale = 0.0
        guidance_replay_carryover_max_transitions = 0
        guidance_replay_carryover_success_only = False
        load_a1_seed_demo = False
        if a1_skill_demo_bootstrap_dir and not a1_seed_demo_dir:
            a1_seed_demo_dir = a1_skill_demo_bootstrap_dir
    if curriculum_env == "avoidance_A1" or str(curriculum_env).startswith("avoidance_A1_gmix_softtimeout") or skill_replay_enabled:
        default_scenario_overrides, default_env_overrides = _default_a1_overrides()
        if load_a1_seed_demo and "a1_direct_block_ratio" not in default_scenario_overrides:
            default_scenario_overrides["a1_direct_block_ratio"] = 0.60
        merged_scenario = dict(default_scenario_overrides)
        merged_scenario.update(effective_scenario_overrides)
        merged_env = dict(default_env_overrides)
        merged_env.update(effective_env_overrides)
        effective_scenario_overrides = merged_scenario
        effective_env_overrides = merged_env
        if abs(float(exploration_noise) - 0.10) < 1e-9:
            exploration_noise = 0.15
        if abs(float(actor_action_reg_weight) - 5e-3) < 1e-12:
            actor_action_reg_weight = 8e-3
    if str(curriculum_env).startswith("avoidance_A1_gmix_softtimeout") and not load_a1_seed_demo and not skill_replay_enabled:
        guidance_replay_carryover_fraction = 0.30
        guidance_replay_carryover_priority_scale = 0.25
        guidance_replay_carryover_max_transitions = 4000
        guidance_replay_carryover_success_only = True

    root = Path(output_dir)
    run_stem = run_name or f"block_{block_id:03d}_{curriculum_env}"
    run_dir = root / run_stem
    run_dir.mkdir(parents=True, exist_ok=True)

    trainer = _build_trainer(
        curriculum_env=curriculum_env,
        mask_mode=mask_mode,
        guidance_self_only=True,
        guidance_exploration_noise=guidance_exploration_noise,
        exploration_noise=exploration_noise,
        avoidance_noise_schedule_enabled=avoidance_noise_schedule_enabled,
        avoidance_noise_stage_values=avoidance_noise_stage_values,
        avoidance_noise_stage_ratios=avoidance_noise_stage_ratios,
        actor_action_reg_weight=actor_action_reg_weight,
        updates_per_step=updates_per_step,
        critic_lr=3e-4,
        q_head_dim=512,
        tau=0.02,
        timeout_seconds=8.0
        if curriculum_env.startswith("guidance")
        else (
            25.0
            if str(curriculum_env).startswith("avoidance_A1_skill_replay_timeout25")
            else (
                18.0
                if skill_replay_enabled
                else (60.0 if curriculum_env == "avoidance_A1" or str(curriculum_env).startswith("avoidance_A1_gmix_softtimeout") else 10.0)
            )
        ),
        target_hold_radius=0.10 if curriculum_env == "avoidance_A1" or str(curriculum_env).startswith("avoidance_A1_gmix_softtimeout") or skill_replay_enabled else 0.15,
        transformer_num_avoidance_layers=transformer_num_avoidance_layers,
        transformer_num_collab_layers=transformer_num_collab_layers,
        critic_profile=critic_profile,
        guidance_replay_carryover_fraction=guidance_replay_carryover_fraction,
        guidance_replay_carryover_priority_scale=guidance_replay_carryover_priority_scale,
        guidance_replay_carryover_max_transitions=guidance_replay_carryover_max_transitions,
        guidance_replay_carryover_success_only=guidance_replay_carryover_success_only,
        replay_backend=replay_backend,
        a1_skill_replay_config=str(a1_skill_replay_config or "balanced"),
        scenario_overrides=effective_scenario_overrides,
        env_overrides=effective_env_overrides,
        seed=seed,
    )

    replay_size_before = len(trainer.replay_manager)
    checkpoint_guidance_import: dict[str, Any] = {}
    seed_demo_import: dict[str, Any] = {}
    if resume_from:
        use_actor_compatibility_load = bool(
            resume_weights
            and transformer_num_avoidance_layers is not None
            and int(transformer_num_avoidance_layers) > 1
        )
        if use_actor_compatibility_load:
            _load_weights_with_actor_compatibility(trainer, Path(resume_from))
            if resume_replay:
                trainer.load_training_state(
                    Path(resume_from),
                    load_weights=False,
                    load_replay=True,
                    load_optimizer=resume_optimizer,
                )
        else:
            trainer.load_training_state(
                Path(resume_from),
                load_weights=resume_weights,
                load_replay=resume_replay,
                load_optimizer=resume_optimizer,
            )
        replay_size_before = len(trainer.replay_manager)
        should_import_checkpoint_guidance = (
            not resume_replay
            and curriculum_env.startswith("avoidance")
            and float(guidance_replay_carryover_fraction) > 0.0
        )
        if should_import_checkpoint_guidance:
            checkpoint_guidance_import = _import_guidance_carryover_from_checkpoint(
                trainer,
                resume_from,
                fraction=float(guidance_replay_carryover_fraction),
                priority_scale=float(guidance_replay_carryover_priority_scale),
                max_transitions=int(guidance_replay_carryover_max_transitions),
                success_only=bool(guidance_replay_carryover_success_only),
            )
            replay_size_before = len(trainer.replay_manager)
    if (
        load_a1_seed_demo
        and curriculum_env.startswith("avoidance")
        and str(curriculum_env).startswith("avoidance_A1_gmix_softtimeout")
        and a1_seed_demo_dir
    ):
        trainer.replay_manager.avoidance_seed_demo_ratio = float(a1_seed_demo_sample_ratio)
        if avoidance_local_sample_ratios is not None:
            trainer.replay_manager.avoidance_sample_ratios = tuple(float(value) for value in avoidance_local_sample_ratios)
        else:
            trainer.replay_manager.avoidance_sample_ratios = (0.20, 0.60, 0.20)
        seed_demo_import = _load_a1_seed_demo_records(trainer, a1_seed_demo_dir)
        replay_size_before = len(trainer.replay_manager)
    elif replay_backend == "a1_skill" and a1_seed_demo_dir:
        seed_demo_import = _load_a1_seed_demo_records(trainer, a1_seed_demo_dir)
        replay_size_before = len(trainer.replay_manager)

    progress_callback = None
    episode_end_callback = None
    episode_start_callback = None
    monitor: WindowedRunMonitor | None = None
    promotion_controller: RollingSuccessPromotionController | ReviewMixPromotionController | None = None
    timing_state = {"start_time": float(time.perf_counter())}
    if rolling_promotion_sequence:
        if rolling_review_mix_enabled:
            promotion_controller = ReviewMixPromotionController(
                env_sequence=list(rolling_promotion_sequence),
                window_size=int(rolling_promotion_window),
                min_successes=int(rolling_promotion_min_successes),
                max_episodes_per_env=int(rolling_promotion_max_episodes_per_env),
                current_sample_count=int(rolling_review_current_sample_count),
                previous_sample_count=int(rolling_review_previous_sample_count),
                rollback_previous_failures=int(rolling_review_rollback_previous_failures),
                max_collision_rate=float(rolling_review_max_collision_rate),
            )
            episode_start_callback = promotion_controller.on_episode_start
        else:
            promotion_controller = RollingSuccessPromotionController(
                env_sequence=list(rolling_promotion_sequence),
                window_size=int(rolling_promotion_window),
                min_successes=int(rolling_promotion_min_successes),
                max_duration_seconds=float(rolling_promotion_max_duration_seconds),
                max_episodes_per_env=int(rolling_promotion_max_episodes_per_env),
            )
        initial_env = str(rolling_promotion_sequence[0])
        if initial_env != trainer.env.scenario.curriculum_env:
            initial_scenario = build_planning_scenario(curriculum_env=initial_env)
            trainer.env.set_scenario(
                replace(
                    initial_scenario,
                    max_obstacles=trainer.env.scenario.max_obstacles,
                    max_neighbors=trainer.env.scenario.max_neighbors,
                )
            )
    if live_progress_every > 0 or route_save_every > 0 or window_summary_every > 0 or promotion_controller is not None:
        monitor = WindowedRunMonitor(
            run_dir=run_dir,
            write_every=int(live_progress_every if live_progress_every > 0 else max(1, window_summary_every)),
            route_save_every=int(route_save_every),
            window_summary_every=int(window_summary_every),
            checkpoint_save_every=int(checkpoint_save_every),
            promotion_controller=promotion_controller,
        )
        progress_callback = monitor.on_progress
        episode_end_callback = monitor.on_episode_end
        timing_state = {"start_time": monitor.start_time}

    summary = trainer.train(
        episodes=episodes_per_block,
        use_curriculum=use_curriculum,
        use_pyramid_per=True,
        use_uniform_replay=False,
        target_env_steps=env_step_budget,
        episode_start_callback=episode_start_callback,
        episode_end_callback=episode_end_callback,
        progress_callback=progress_callback,
        progress_heartbeat_steps=int(progress_heartbeat_steps),
    )
    elapsed_seconds = float(time.perf_counter() - timing_state["start_time"])
    chunks = _chunk_summary(summary["history"])
    if monitor is not None:
        final_payload = {
            "episode_index": int(max(0, len(summary["history"]) - 1)),
            "episode_count": int(len(summary["history"])),
            "total_episodes": int(episodes_per_block),
            "episode_metrics": dict(summary["final_metrics"]),
            "history": list(summary["history"]),
            "episode_trace": dict(summary.get("trajectory", {})),
            "reward_components": [],
            "replay_stats": dict(summary["replay_stats"]),
            "training_counters": dict(summary["training_counters"]),
            "trainer": trainer,
        }
        monitor.finalize(final_payload)
    diagnostics_dir = run_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    stage_mode = str(trainer.current_stage_mode)
    action_field = _action_field(trainer, diagnostics_dir, run_stem, stage_mode=stage_mode)
    feature_probe = _feature_probe(trainer, diagnostics_dir, run_stem, stage_mode=stage_mode)
    obstacle_response_probe = (
        _obstacle_response_probe(trainer, diagnostics_dir, run_stem, stage_mode=stage_mode)
        if curriculum_env.startswith("avoidance")
        else None
    )
    current_stage_eval = trainer.evaluate_on_scenario(
        build_planning_scenario(curriculum_env=curriculum_env),
        episodes=5,
    )
    legacy_eval = _legacy_task_performance(trainer, completed_stages, episodes=5)
    q_stats = _q_value_distribution(trainer, block_episode_count=episodes_per_block, sample_window=50)
    tail = _tail_counts(chunks, tail_chunks=5)
    reward_decomposition_tail50 = _reward_decomposition_summary(summary["history"], tail_episodes=50)
    recommended_next_action = _recommended_next_action(
        curriculum_env=curriculum_env,
        tail_successes=int(tail["tail_successes"]),
        tail_out_of_bounds=int(tail["tail_out_of_bounds"]),
        tail_timeout=int(tail["tail_timeout"]),
        tail_collision=int(tail["tail_collision"]),
        legacy_task_performance=legacy_eval,
        obstacle_response_probe=obstacle_response_probe,
    )

    block_summary = {
        "block_id": int(block_id),
        "run_name": run_stem,
        "curriculum_env": curriculum_env,
        "completed_stages": completed_stages,
        "episodes_per_block": int(episodes_per_block),
        "env_step_budget": int(env_step_budget) if env_step_budget is not None else 0,
        "elapsed_seconds": elapsed_seconds,
        "avg_seconds_per_episode": float(elapsed_seconds / max(1, summary["training_counters"].get("completed_episodes", episodes_per_block))),
        "mask_mode": mask_mode,
        "use_curriculum": bool(use_curriculum),
        "guidance_self_only": bool(trainer.network_config.transformer.guidance_self_only),
        "guidance_exploration_noise": float(guidance_exploration_noise),
        "exploration_noise": float(exploration_noise),
        "avoidance_noise_schedule_enabled": bool(avoidance_noise_schedule_enabled),
        "avoidance_noise_stage_values": [float(value) for value in avoidance_noise_stage_values],
        "avoidance_noise_stage_ratios": [float(value) for value in avoidance_noise_stage_ratios],
        "actor_action_reg_weight": float(actor_action_reg_weight),
        "updates_per_step": float(updates_per_step),
        "critic_lr": 3e-4,
        "q_head_dim": 512,
        "critic_profile": str(critic_profile),
        "tau": 0.02,
        "resume_from": str(resume_from) if resume_from else "",
        "scenario_overrides": dict(effective_scenario_overrides),
        "env_overrides": dict(effective_env_overrides),
        "live_progress_every": int(live_progress_every),
        "route_save_every": int(route_save_every),
        "window_summary_every": int(window_summary_every),
        "progress_heartbeat_steps": int(progress_heartbeat_steps),
        "transformer_num_avoidance_layers": int(trainer.network_config.transformer.num_avoidance_layers),
        "transformer_num_collab_layers": int(trainer.network_config.transformer.num_collab_layers),
        "guidance_replay_carryover_fraction": float(guidance_replay_carryover_fraction),
        "guidance_replay_carryover_priority_scale": float(guidance_replay_carryover_priority_scale),
        "guidance_replay_carryover_max_transitions": int(guidance_replay_carryover_max_transitions),
        "guidance_replay_carryover_success_only": bool(guidance_replay_carryover_success_only),
        "replay_backend": str(replay_backend),
        "a1_skill_replay_config": str(a1_skill_replay_config or ""),
        "a1_skill_demo_bootstrap_dir": str(a1_skill_demo_bootstrap_dir or ""),
        "a1_skill_demo_decay_enabled": bool(a1_skill_demo_decay_enabled),
        "load_a1_seed_demo": bool(load_a1_seed_demo),
        "a1_seed_demo_dir": str(a1_seed_demo_dir or ""),
        "a1_seed_demo_sample_ratio": float(a1_seed_demo_sample_ratio),
        "avoidance_local_sample_ratios": [
            float(value) for value in (
                avoidance_local_sample_ratios
                if avoidance_local_sample_ratios is not None
                else tuple(getattr(trainer.replay_manager, "avoidance_sample_ratios", (0.25, 0.50, 0.25)))
            )
        ],
        "seed": int(seed),
        "checkpoint_guidance_import": dict(checkpoint_guidance_import),
        "seed_demo_import": dict(seed_demo_import),
        "actor_compatibility_load": dict(getattr(trainer, "_last_actor_compatibility_load", {})),
        "replay_resume_enabled": bool(resume_replay),
        "replay_size_before_block": int(replay_size_before),
        "replay_size_after_block": int(len(trainer.replay_manager)),
        "replay_mode": str(summary["replay_stats"].get("mode", "")),
        "training_counters": summary["training_counters"],
        "final_metrics": summary["final_metrics"],
        "action_field": action_field,
        "feature_probe": feature_probe,
        "obstacle_response_probe": obstacle_response_probe,
        "current_stage_eval": current_stage_eval,
        "legacy_task_performance": legacy_eval,
        "reward_decomposition_tail50": reward_decomposition_tail50,
        "collision_rate": float(summary["final_metrics"].get("collision_rate", 0.0)),
        "timeout_rate": float(summary["final_metrics"].get("timeout_rate", 0.0)),
        "out_of_bounds_rate": float(summary["final_metrics"].get("out_of_bounds_rate", 0.0)),
        "risk_drop_reward_mean": float(reward_decomposition_tail50.get("risk_drop_reward", 0.0)),
        "clearance_gain_reward_mean": float(reward_decomposition_tail50.get("clearance_gain_reward", 0.0)),
        "detour_trend_reward_mean": float(reward_decomposition_tail50.get("detour_trend_reward", 0.0)),
        "near_collision_penalty_mean": float(reward_decomposition_tail50.get("near_collision_penalty", 0.0)),
        "critical_collision_penalty_mean": float(reward_decomposition_tail50.get("critical_collision_margin_penalty", 0.0)),
        "action_saturation_penalty_mean": float(reward_decomposition_tail50.get("action_saturation_penalty", 0.0)),
        "action_change_penalty_mean": float(reward_decomposition_tail50.get("action_change_penalty", 0.0)),
        "velocity_delta_mean": float(summary["final_metrics"].get("velocity_delta_mean", 0.0)),
        "velocity_delta_max": float(summary["final_metrics"].get("velocity_delta_max", 0.0)),
        "acceleration_clip_rate": float(summary["final_metrics"].get("acceleration_clip_rate", 0.0)),
        "window_summary_count": int(len(monitor.window_summaries)) if monitor is not None else 0,
        "latest_window_summary": dict(monitor.window_summaries[-1]) if monitor is not None and monitor.window_summaries else {},
        "checkpoint_save_every": int(checkpoint_save_every),
        "rolling_promotion_sequence": list(rolling_promotion_sequence or []),
        "rolling_promotion_window": int(rolling_promotion_window),
        "rolling_promotion_min_successes": int(rolling_promotion_min_successes),
        "rolling_promotion_max_duration_seconds": float(rolling_promotion_max_duration_seconds),
        "rolling_promotion_max_episodes_per_env": int(rolling_promotion_max_episodes_per_env),
        "rolling_review_mix_enabled": bool(rolling_review_mix_enabled),
        "rolling_review_current_sample_count": int(rolling_review_current_sample_count),
        "rolling_review_previous_sample_count": int(rolling_review_previous_sample_count),
        "rolling_review_rollback_previous_failures": int(rolling_review_rollback_previous_failures),
        "rolling_review_max_collision_rate": float(rolling_review_max_collision_rate),
        "env_transition_history": promotion_controller.build_transition_rows() if promotion_controller is not None else [],
        "env_step_summary": promotion_controller.build_env_step_rows() if promotion_controller is not None else [],
        "recommended_next_action": recommended_next_action,
        **q_stats,
        **tail,
    }
    if promotion_controller is not None:
        block_summary.update(dict(promotion_controller.last_state))

    _write_csv(run_dir / "block_history.csv", summary["history"])
    _write_csv(run_dir / "block_chunk_status.csv", chunks)
    (run_dir / "block_chunk_status.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "legacy_eval.json").write_text(json.dumps(legacy_eval, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "block_summary.json").write_text(json.dumps(block_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    checkpoint_path = trainer.save_training_state(
        run_dir / "trainer_state.pt",
        include_replay=True,
        include_optimizer=resume_optimizer,
    )
    block_summary["checkpoint_path"] = str(checkpoint_path)
    (run_dir / "block_summary.json").write_text(json.dumps(block_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return block_summary


if __name__ == "__main__":
    output = Path("outputs/planning/manual_block_training")
    result = run_manual_block(output, block_id=1, curriculum_env="guidance_G1", completed_stages=["guidance_G1"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
