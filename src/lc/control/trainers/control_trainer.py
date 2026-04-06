from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from lc.control.controllers import LADRCController, PIDController
from lc.control.envs import ControlTrackingEnv
from lc.control.policies import ControlLADRLAgent, stack_state
from lc.envs.metrics import compute_control_metrics


@dataclass
class ControlTrainer:
    """Chapter-3 trainer with axis-wise tuning, RL evaluation, and seed aggregation helpers."""

    env: ControlTrackingEnv
    stack_size: int = 4
    action_hold_steps: int = 4
    n_step: int = 4
    warmup_steps: int = 64
    batch_size: int = 32
    updates_per_step: int = 1
    tuned_ladrc_params: dict[str, dict[str, float]] = field(default_factory=dict)

    def evaluate_pid(self, episodes: int, axis: str | None = None, seed_offset: int = 0) -> dict[str, float]:
        return self._run_closed_loop(PIDController(), episodes, axis=axis, seed_offset=seed_offset)

    def evaluate_ladrc(
        self,
        episodes: int,
        axis: str | None = None,
        seed_offset: int = 0,
        params: dict[str, float] | None = None,
    ) -> dict[str, float]:
        controller = self.build_ladrc_controller(axis or self.env.axis, params=params)
        return self._run_closed_loop(controller, episodes, axis=axis, seed_offset=seed_offset)

    def evaluate_ddpg_ladrc(
        self,
        train_episodes: int,
        eval_episodes: int,
        axis: str | None = None,
        seed_offset: int = 0,
    ) -> dict[str, float]:
        return self.run_rl_experiment(train_episodes, eval_episodes, enhanced=False, axis=axis, seed_offset=seed_offset)["metrics"]

    def evaluate_mddpg_ladrc(
        self,
        train_episodes: int,
        eval_episodes: int,
        axis: str | None = None,
        seed_offset: int = 0,
    ) -> dict[str, float]:
        return self.run_rl_experiment(train_episodes, eval_episodes, enhanced=True, axis=axis, seed_offset=seed_offset)["metrics"]

    def tune_ladrc_axes(self, axes: tuple[str, ...], episodes: int) -> dict[str, dict[str, float]]:
        snapshots: dict[str, dict[str, float]] = {}
        for axis in axes:
            snapshots[axis] = self.tune_ladrc_axis(axis, episodes)
        self.tuned_ladrc_params.update(snapshots)
        return snapshots

    def tune_ladrc_axis(self, axis: str, episodes: int) -> dict[str, float]:
        candidates = self._candidate_grid(axis)
        best_metrics: dict[str, float] | None = None
        best_params: dict[str, float] | None = None
        for params in candidates:
            metrics = self.evaluate_ladrc(episodes=episodes, axis=axis, params=params)
            score = self._ladrc_score(metrics)
            if best_metrics is None or score < self._ladrc_score(best_metrics):
                best_metrics = metrics
                best_params = {**params, **metrics}
        if best_params is None:
            raise RuntimeError(f"Failed to tune LADRC axis {axis}")
        snapshot = {
            "axis": axis,
            "b0": float(best_params["b0"]),
            "omega_c": float(best_params["omega_c"]),
            "k": float(best_params["k"]),
            "omega_o": float(best_params["omega_c"] * best_params["k"]),
            "rmse": float(best_params["rmse"]),
            "iae": float(best_params["iae"]),
            "settling_time": float(best_params["settling_time"]),
            "control_variation": float(best_params["control_variation"]),
        }
        self.tuned_ladrc_params[axis] = snapshot
        return snapshot

    def build_ladrc_controller(self, axis: str, params: dict[str, float] | None = None) -> LADRCController:
        tuned = self.tuned_ladrc_params.get(axis, {})
        resolved = {
            "b0": float(params["b0"]) if params and "b0" in params else float(tuned.get("b0", self._default_axis_params(axis)["b0"])),
            "omega_c": float(params["omega_c"]) if params and "omega_c" in params else float(tuned.get("omega_c", self._default_axis_params(axis)["omega_c"])),
            "k": float(params["k"]) if params and "k" in params else float(tuned.get("k", self._default_axis_params(axis)["k"])),
        }
        return LADRCController(**resolved)

    def run_rl_experiment(
        self,
        train_episodes: int,
        eval_episodes: int,
        enhanced: bool,
        axis: str | None = None,
        seed_offset: int = 0,
        stack_size_override: int | None = None,
        action_hold_override: int | None = None,
        n_step_override: int | None = None,
    ) -> dict[str, object]:
        """Run one RL-LADRC experiment and return training/eval bundles."""
        method = "mddpg_ladrc" if enhanced else "ddpg_ladrc"
        agent, train_history = self._train_agent(
            train_episodes=train_episodes,
            enhanced=enhanced,
            axis=axis,
            seed_offset=seed_offset,
            stack_size_override=stack_size_override,
            action_hold_override=action_hold_override,
            n_step_override=n_step_override,
        )
        metrics, representative_trajectory = self._evaluate_agent(
            agent,
            eval_episodes,
            axis=axis,
            seed_offset=seed_offset + 1000,
            with_trajectory=True,
        )
        return {
            "method": method,
            "metrics": metrics,
            "train_history": train_history,
            "trajectory": representative_trajectory,
            "checkpoint": {
                "actor": agent.policy.actor.state_dict(),
                "critic": agent.policy.critic.state_dict(),
                "axis": axis or self.env.axis,
                "ladrc_baseline": self.tuned_ladrc_params.get(axis or self.env.axis, {}),
            },
        }

    def collect_controller_trajectory(
        self,
        controller: PIDController | LADRCController,
        axis: str | None = None,
        seed_offset: int = 0,
    ) -> dict[str, list[float]]:
        _, _, trajectory = self._run_controller_episode(controller, axis=axis, seed_offset=seed_offset)
        return trajectory

    def _run_closed_loop(
        self,
        controller: PIDController | LADRCController,
        episodes: int,
        axis: str | None = None,
        seed_offset: int = 0,
    ) -> dict[str, float]:
        metrics = []
        for episode in range(episodes):
            episode_metrics, _, _ = self._run_controller_episode(controller, axis=axis, seed_offset=seed_offset + episode)
            metrics.append(episode_metrics)
        return _average(metrics)

    def _train_agent(
        self,
        train_episodes: int,
        enhanced: bool,
        axis: str | None = None,
        seed_offset: int = 0,
        stack_size_override: int | None = None,
        action_hold_override: int | None = None,
        n_step_override: int | None = None,
    ) -> tuple[ControlLADRLAgent, list[dict[str, float]]]:
        axis_name = axis or self.env.axis
        stack_size = stack_size_override if stack_size_override is not None else (self.stack_size if enhanced else 1)
        hold_steps = action_hold_override if action_hold_override is not None else (self.action_hold_steps if enhanced else 1)
        n_step = n_step_override if n_step_override is not None else (self.n_step if enhanced else 1)
        initial_obs = self.env.reset(axis=axis_name, seed=self.env.seed + seed_offset)
        agent = ControlLADRLAgent(
            obs_dim=initial_obs.shape[0],
            stack_size=stack_size,
            action_hold_steps=hold_steps,
            n_step=n_step,
            batch_size=self.batch_size,
        )
        if axis_name in self.tuned_ladrc_params:
            tuned = self.tuned_ladrc_params[axis_name]
            agent.controller.base.set_parameters(b0=tuned["b0"], omega_c=tuned["omega_c"], k=tuned["k"])
        total_steps = 0
        gamma = agent.policy.config.gamma
        train_history: list[dict[str, float]] = []
        for episode in range(train_episodes):
            obs = self.env.reset(axis=axis_name, seed=self.env.seed + seed_offset + episode)
            history = [obs.copy()]
            agent.reset()
            if axis_name in self.tuned_ladrc_params:
                tuned = self.tuned_ladrc_params[axis_name]
                agent.controller.base.set_parameters(b0=tuned["b0"], omega_c=tuned["omega_c"], k=tuned["k"])
            rollout: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque()
            done = False
            episode_reward = 0.0
            losses = {"critic_loss": 0.0, "actor_loss": 0.0}
            while not done:
                stacked_state = stack_state(history, obs, stack_size)
                explore = total_steps >= self.warmup_steps
                if explore:
                    action = agent.act(stacked_state, explore=True)
                else:
                    action = np.random.uniform(-1.0, 1.0, size=3).astype(np.float32)
                agent.controller.adapt(action)
                control_signal = agent.controller.step(
                    self.env.reference,
                    self.env.state_axis,
                    1.0 / self.env.scenario.control_frequency_hz,
                )
                next_obs, reward, done, _ = self.env.step(control_signal)
                episode_reward += reward
                next_stacked = stack_state(history, next_obs, stack_size)
                rollout.append((stacked_state.copy(), action.copy(), reward, next_stacked.copy(), done))
                self._flush_n_step(agent, rollout, n_step, gamma, force=done)
                total_steps += 1
                if total_steps >= self.warmup_steps:
                    losses = agent.policy.update(self.updates_per_step)
                obs = next_obs
                history.append(obs.copy())
            episode_metrics = self._episode_metrics()
            train_history.append(
                {
                    "episode": float(episode),
                    "reward": float(episode_reward),
                    "mae": episode_metrics["mae"],
                    "rmse": episode_metrics["rmse"],
                    "iae": episode_metrics["iae"],
                    "steady_state_error": episode_metrics["steady_state_error"],
                    "actor_loss": float(losses["actor_loss"]),
                    "critic_loss": float(losses["critic_loss"]),
                }
            )
        return agent, train_history

    def _evaluate_agent(
        self,
        agent: ControlLADRLAgent,
        episodes: int,
        axis: str | None = None,
        seed_offset: int = 0,
        with_trajectory: bool = False,
    ) -> dict[str, float] | tuple[dict[str, float], dict[str, list[float]]]:
        metrics = []
        stack_size = agent.policy.config.stack_size
        representative_trajectory: dict[str, list[float]] | None = None
        axis_name = axis or self.env.axis
        for episode in range(episodes):
            obs = self.env.reset(axis=axis_name, seed=self.env.seed + seed_offset + episode)
            history = [obs.copy()]
            agent.reset()
            done = False
            total_reward = 0.0
            while not done:
                stacked = stack_state(history, obs, stack_size)
                action = agent.act(stacked, explore=False)
                agent.controller.adapt(action)
                control_signal = agent.controller.step(
                    self.env.reference,
                    self.env.state_axis,
                    1.0 / self.env.scenario.control_frequency_hz,
                )
                obs, reward, done, _ = self.env.step(control_signal)
                total_reward += reward
                history.append(obs.copy())
            episode_metrics = self._episode_metrics()
            episode_metrics["reward"] = float(total_reward)
            metrics.append(episode_metrics)
            if representative_trajectory is None:
                representative_trajectory = self._collect_trajectory()
        averaged = _average(metrics)
        if with_trajectory:
            return averaged, representative_trajectory or self._collect_trajectory()
        return averaged

    def _run_controller_episode(
        self,
        controller: PIDController | LADRCController,
        axis: str | None = None,
        seed_offset: int = 0,
    ) -> tuple[dict[str, float], float, dict[str, list[float]]]:
        axis_name = axis or self.env.axis
        self.env.reset(axis=axis_name, seed=self.env.seed + seed_offset)
        controller.reset()
        total_reward = 0.0
        done = False
        while not done:
            control_signal = controller.step(
                self.env.reference,
                self.env.state_axis,
                1.0 / self.env.scenario.control_frequency_hz,
            )
            _, reward, done, _ = self.env.step(control_signal)
            total_reward += reward
        episode_metrics = self._episode_metrics()
        episode_metrics["reward"] = float(total_reward)
        return episode_metrics, total_reward, self._collect_trajectory()

    def _episode_metrics(self) -> dict[str, float]:
        metrics = compute_control_metrics(self.env.errors, self.env.controls)
        if self.env.velocity_errors:
            metrics["velocity_rmse"] = float(np.sqrt(np.mean(np.asarray(self.env.velocity_errors, dtype=float) ** 2)))
        else:
            metrics["velocity_rmse"] = 0.0
        return metrics

    def _collect_trajectory(self) -> dict[str, list[float]]:
        return {
            "reference": list(self.env.references),
            "reference_velocity": list(self.env.reference_velocities),
            "error": list(self.env.errors),
            "velocity_error": list(self.env.velocity_errors),
            "output": list(self.env.outputs),
            "control": list(self.env.controls),
            "disturbance": list(self.env.disturbances),
        }

    def _flush_n_step(
        self,
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

    def _default_axis_params(self, axis: str) -> dict[str, float]:
        defaults = {
            "x": {"b0": 1.0, "omega_c": 5.5, "k": 3.0},
            "y": {"b0": 1.0, "omega_c": 5.5, "k": 3.2},
            "z": {"b0": 1.2, "omega_c": 4.5, "k": 3.8},
        }
        return defaults.get(axis, defaults["x"])

    def _candidate_grid(self, axis: str) -> list[dict[str, float]]:
        base = self._default_axis_params(axis)
        b0_values = [base["b0"] * scale for scale in (0.8, 1.0, 1.2)]
        wc_values = [base["omega_c"] + delta for delta in (-1.0, 0.0, 1.0)]
        k_values = [base["k"] + delta for delta in (-0.5, 0.0, 0.5)]
        return [
            {"b0": float(b0), "omega_c": float(max(wc, 1.5)), "k": float(max(k, 1.5))}
            for b0 in b0_values
            for wc in wc_values
            for k in k_values
        ]

    def _ladrc_score(self, metrics: dict[str, float]) -> float:
        return (
            0.45 * metrics["rmse"]
            + 0.25 * metrics["iae"]
            + 0.15 * metrics["steady_state_error"]
            + 0.10 * metrics["control_variation"]
            + 0.05 * metrics["settling_time"]
        )


def _average(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def save_checkpoint(path: str, checkpoint: dict[str, object]) -> None:
    """Save chapter-3 RL checkpoint."""
    torch.save(checkpoint, path)
