from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import torch
from torch import nn

from lc.envs.metrics import compute_planning_metrics
from lc.envs.scenarios import build_planning_scenario
from lc.planning.critics import StructuredCritic, StructuredCriticConfig
from lc.planning.curriculum import CurriculumScheduler
from lc.planning.envs import PlanningSwarmEnv
from lc.planning.memory import GuidanceReplayMemory, StagePyramidReplayMemory, summarize_stage_sources
from lc.planning.models import MultiUAVModel, MultiUAVModelConfig, SingleStreamMLPPolicy


@dataclass
class PlanningTrainer:
    env: PlanningSwarmEnv
    gamma: float = 0.98
    tau: float = 0.02
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    policy_noise: float = 0.15
    noise_clip: float = 0.25
    exploration_noise: float = 0.1
    policy_delay: int = 2
    batch_size: int = 32
    warmup_steps: int = 32
    replay_capacity: int = 512
    avoidance_old_fraction: float = 0.1
    cooperation_old_fraction: float = 0.2
    avoidance_sample_ratio: tuple[int, int, int] = (6, 3, 1)
    cooperation_sample_ratio: tuple[int, int, int] = (5, 3, 2)
    avoidance_priority_mode: str = "hybrid"
    cooperation_priority_mode: str = "hybrid"
    rare_priority_mode: str = "hybrid"
    seed: int = 7
    curriculum: CurriculumScheduler = field(default_factory=CurriculumScheduler)

    def __post_init__(self) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        actor_config = MultiUAVModelConfig(
            self_dim=self.env.self_dim,
            obstacle_dim=self.env.obstacle_dim,
            neighbor_dim=self.env.neighbor_dim,
            max_obstacles=self.env.scenario.max_obstacles,
            max_neighbors=self.env.scenario.max_neighbors,
        )
        critic_config = StructuredCriticConfig(
            self_dim=self.env.self_dim,
            obstacle_dim=self.env.obstacle_dim,
            neighbor_dim=self.env.neighbor_dim,
            action_dim=actor_config.action_dim,
            embed_dim=actor_config.embed_dim,
            ff_dim=actor_config.ff_dim,
            max_obstacles=self.env.scenario.max_obstacles,
            max_neighbors=self.env.scenario.max_neighbors,
        )
        input_dim = (
            actor_config.self_dim
            + actor_config.max_obstacles * actor_config.obstacle_dim
            + actor_config.max_neighbors * actor_config.neighbor_dim
        )
        self.actor = MultiUAVModel(actor_config)
        self.target_actor = MultiUAVModel(actor_config)
        self.target_actor.load_state_dict(self.actor.state_dict())
        actor_embeddings = self.actor.shared_embeddings()
        target_embeddings = self.target_actor.shared_embeddings()
        self.critic_1 = StructuredCritic(critic_config, **actor_embeddings)
        self.target_critic_1 = StructuredCritic(critic_config, **target_embeddings)
        self.critic_2 = StructuredCritic(critic_config, **actor_embeddings)
        self.target_critic_2 = StructuredCritic(critic_config, **target_embeddings)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict(), strict=False)
        self.target_critic_2.load_state_dict(self.critic_2.state_dict(), strict=False)
        self.mlp = SingleStreamMLPPolicy(input_dim=input_dim, action_dim=actor_config.action_dim)
        self.target_mlp = SingleStreamMLPPolicy(input_dim=input_dim, action_dim=actor_config.action_dim)
        self.target_mlp.load_state_dict(self.mlp.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        critic_params = _unique_parameters(list(self.critic_1.parameters()) + list(self.critic_2.parameters()))
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=self.critic_lr)
        self.mlp_optimizer = torch.optim.Adam(self.mlp.parameters(), lr=self.actor_lr)
        self.guidance_replay = GuidanceReplayMemory(capacity=self.replay_capacity, seed=self.seed)
        self.avoidance_replay = StagePyramidReplayMemory(
            stage_name="avoidance",
            capacity=self.replay_capacity,
            sample_ratio=self.avoidance_sample_ratio,
            secondary_priority_mode=self.avoidance_priority_mode,
            rare_priority_mode=self.rare_priority_mode,
            seed=self.seed + 1,
        )
        self.cooperation_replay = StagePyramidReplayMemory(
            stage_name="cooperation",
            capacity=self.replay_capacity,
            sample_ratio=self.cooperation_sample_ratio,
            secondary_priority_mode=self.cooperation_priority_mode,
            rare_priority_mode=self.rare_priority_mode,
            seed=self.seed + 2,
        )
        self.last_train_summary: dict[str, object] = {}
        self.update_step = 0
        self.previous_stage_name = self.env.scenario.stage_name

    def train(
        self,
        episodes: int,
        actor_variant: str = "task_decomposed",
        use_curriculum: bool = True,
        use_pyramid_per: bool = True,
        use_uniform_replay: bool = False,
    ) -> dict[str, object]:
        history: list[dict[str, float | int | str]] = []
        last_sample_sources: dict[str, int] = {}
        for episode in range(episodes):
            self._apply_episode_scenario(use_curriculum, episode, episodes)
            stage_name = self.curriculum.stage_name if use_curriculum else self.env.scenario.stage_name
            curriculum_env = self.curriculum.curriculum_env if use_curriculum else self.env.scenario.curriculum_env
            self._handle_stage_transition(stage_name)
            obs = self.env.reset()
            total_reward = 0.0
            actor_losses: list[float] = []
            critic_losses: list[float] = []
            reward_sums = {k: 0.0 for k in (
                "target_reward",
                "avoidance_reward",
                "collaboration_reward",
                "recovery_reward",
                "smoothness_penalty",
                "consistency_penalty",
                "success_bonus",
            )}
            steps = 0
            done = False
            prev_slot_error = float(np.linalg.norm(obs["self_state"][:2]))
            while not done:
                action, avoid_action = self._select_action(obs, actor_variant)
                next_obs, reward, done, info = self.env.step(action)
                total_reward += reward
                steps += 1
                for key in reward_sums:
                    reward_sums[key] += float(info.get(key, 0.0))
                next_slot_error = float(info["occupancy_error"])
                contribution = self._compute_contribution(stage_name, prev_slot_error, next_slot_error, info)
                previous_td_error = self._td_reference(stage_name)
                td_proxy = self._td_error_proxy(reward, info, contribution)
                transition = {
                    "obs": obs,
                    "action": np.asarray(action, dtype=np.float32),
                    "avoid_action": np.asarray(avoid_action, dtype=np.float32),
                    "reward": float(reward),
                    "next_obs": next_obs,
                    "done": float(done),
                    "risk": float(info["risk"]),
                    "collision": bool(info["collision"]),
                    "success": float(info["success"]),
                    "formation_error": float(info["formation_error"]),
                    "angle_error": float(info["angle_error"]),
                    "occupancy_error": float(info["occupancy_error"]),
                    "rare_event_score": float(info["rare_event_score"]),
                    "contribution": float(contribution),
                    "stage_name": stage_name,
                    "previous_td_error": float(previous_td_error),
                }
                self._push_transition(stage_name, transition, td_proxy, previous_td_error, use_pyramid_per and not use_uniform_replay)
                replay_ready = self._replay_ready(stage_name, use_pyramid_per and not use_uniform_replay)
                if replay_ready and (steps >= self.warmup_steps or self.update_step >= self.warmup_steps):
                    critic_loss, actor_loss, sample_sources = self._update_networks(
                        actor_variant=actor_variant,
                        stage_name=stage_name,
                        use_stage_replay=use_pyramid_per and not use_uniform_replay,
                    )
                    critic_losses.append(critic_loss)
                    if actor_loss is not None:
                        actor_losses.append(actor_loss)
                    last_sample_sources = sample_sources
                    self.update_step += 1
                prev_slot_error = next_slot_error
                obs = next_obs

            episode_metrics = compute_planning_metrics(
                collisions=self.env.collisions,
                occupancy_errors=self.env.occupancy_errors,
                formation_errors=self.env.formation_errors,
                success=self.env.success,
            )
            episode_metrics.update(
                {
                    "episode": float(episode),
                    "reward": float(total_reward),
                    "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
                    "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
                    "stage_index": float(self.env.scenario.stage_index),
                    "stage_name": stage_name,
                    "curriculum_env": curriculum_env,
                    "recovery_score": float(np.mean([max(0.0, 1.0 - risk) for risk in self.env.risk_history])) if self.env.risk_history else 0.0,
                }
            )
            for key, value in reward_sums.items():
                episode_metrics[key] = float(value / max(1, steps))
            history.append(episode_metrics)
            if use_curriculum:
                self.curriculum.update(
                    {
                        "reward": float(total_reward) / max(1, steps),
                        "success_rate": float(episode_metrics["success_rate"]),
                        "collision_rate": float(episode_metrics["collision_rate"]),
                        "occupancy_error": float(episode_metrics["occupancy_error"]),
                    }
                )

        summary = {
            "history": history,
            "final_metrics": history[-1] if history else {},
            "stage_history": list(self.curriculum.stage_history) if use_curriculum else [],
            "stage_averages": self.curriculum.get_stage_averages() if use_curriculum else {},
            "env_averages": self.curriculum.get_env_averages() if use_curriculum else {},
            "replay_stats": self._gather_replay_stats(last_sample_sources),
            "stage_transition_summary": self._summarize_stage_transitions(history),
            "trajectory": self.env.get_episode_trace(),
            "attention_proxy": self._build_attention_proxy(),
        }
        self.last_train_summary = summary
        return summary

    def evaluate_primary(self, episodes: int, difficulty: str | None = None, stage_index: int | None = None) -> dict[str, float]:
        return self._evaluate_actor("task_decomposed", episodes, difficulty, stage_index)

    def evaluate_mlp_baseline(self, episodes: int, difficulty: str | None = None, stage_index: int | None = None) -> dict[str, float]:
        return self._evaluate_actor("single_stream_mlp", episodes, difficulty, stage_index)

    def evaluate_on_scenario(self, scenario, episodes: int, actor_variant: str = "task_decomposed") -> dict[str, float]:
        original = self.env.scenario
        self.env.set_scenario(scenario)
        try:
            return self._evaluate_actor(actor_variant, episodes)
        finally:
            self.env.set_scenario(original)

    def _apply_episode_scenario(self, use_curriculum: bool, episode: int, total_episodes: int) -> None:
        if not use_curriculum:
            scenario = build_planning_scenario(curriculum_env=self.env.scenario.curriculum_env)
        else:
            scenario = build_planning_scenario(curriculum_env=self.curriculum.curriculum_env)
        self.env.set_scenario(
            replace(
                scenario,
                max_obstacles=self.env.scenario.max_obstacles,
                max_neighbors=self.env.scenario.max_neighbors,
            )
        )

    def _handle_stage_transition(self, stage_name: str) -> None:
        if stage_name == self.previous_stage_name:
            return
        if self.previous_stage_name == "guidance":
            self.guidance_replay.refresh_old_pool()
        elif self.previous_stage_name == "avoidance":
            self.avoidance_replay.refresh_old_pool()
        self.previous_stage_name = stage_name

    def _select_action(self, obs: dict[str, np.ndarray], actor_variant: str) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            if actor_variant == "single_stream_mlp":
                flat = torch.tensor(_flatten_obs(obs), dtype=torch.float32).unsqueeze(0)
                final_action = self.mlp(flat)
                avoid_action = final_action
            else:
                structured = _to_torch(obs)
                avoid_action, final_action = self.actor(structured)
        noise = self.rng.normal(0.0, self.exploration_noise, size=final_action.shape[-1]).astype(np.float32)
        action = np.clip(final_action.detach().cpu().numpy()[0] + noise, -1.0, 1.0)
        return action, avoid_action.detach().cpu().numpy()[0]

    def _push_transition(self, stage_name: str, transition: dict[str, object], td_proxy: float, previous_td_error: float, use_stage_replay: bool) -> None:
        if not use_stage_replay:
            self.guidance_replay.push(transition, priority=max(td_proxy + float(transition["reward"]), 1e-4))
            return
        if stage_name == "guidance":
            self.guidance_replay.push(transition, priority=max(td_proxy + float(transition["reward"]), 1e-4))
        elif stage_name == "avoidance":
            self.avoidance_replay.push(
                transition,
                td_error=td_proxy,
                previous_td_error=previous_td_error,
                contribution=float(transition["contribution"]),
                rare_event_score=float(transition["rare_event_score"]),
                success=bool(transition["success"]),
            )
        else:
            self.cooperation_replay.push(
                transition,
                td_error=td_proxy,
                previous_td_error=previous_td_error,
                contribution=float(transition["contribution"]),
                rare_event_score=float(transition["rare_event_score"]),
                success=bool(transition["success"]),
            )

    def _replay_ready(self, stage_name: str, use_stage_replay: bool) -> bool:
        if not use_stage_replay or stage_name == "guidance":
            return len(self.guidance_replay) >= self.batch_size
        if stage_name == "avoidance":
            return len(self.avoidance_replay) >= max(1, int(self.batch_size * 0.9))
        return len(self.cooperation_replay) >= max(1, int(self.batch_size * 0.8))

    def _update_networks(self, actor_variant: str, stage_name: str, use_stage_replay: bool) -> tuple[float, float | None, dict[str, int]]:
        batch_entries = self._sample_batch(stage_name, self.batch_size, use_stage_replay)
        if not batch_entries:
            return 0.0, None, {}
        batch = [entry["payload"] for entry in batch_entries]
        obs = _batch_obs([row["obs"] for row in batch])
        next_obs = _batch_obs([row["next_obs"] for row in batch])
        flat_obs = torch.tensor(np.stack([_flatten_obs(row["obs"]) for row in batch]), dtype=torch.float32)
        flat_next_obs = torch.tensor(np.stack([_flatten_obs(row["next_obs"]) for row in batch]), dtype=torch.float32)
        actions = torch.tensor(np.stack([row["action"] for row in batch]), dtype=torch.float32)
        rewards = torch.tensor([row["reward"] for row in batch], dtype=torch.float32).unsqueeze(-1)
        dones = torch.tensor([row["done"] for row in batch], dtype=torch.float32).unsqueeze(-1)
        is_weights = torch.tensor([entry["is_weight"] for entry in batch_entries], dtype=torch.float32).unsqueeze(-1)

        with torch.no_grad():
            if actor_variant == "single_stream_mlp":
                next_actions = self.target_mlp(flat_next_obs)
            else:
                _, next_actions = self.target_actor(next_obs)
            noise = torch.clamp(torch.randn_like(next_actions) * self.policy_noise, -self.noise_clip, self.noise_clip)
            next_actions = torch.clamp(next_actions + noise, -1.0, 1.0)
            target_q1 = self.target_critic_1(next_obs, next_actions)
            target_q2 = self.target_critic_2(next_obs, next_actions)
            target_q = rewards + self.gamma * (1.0 - dones) * torch.minimum(target_q1, target_q2)

        current_q1 = self.critic_1(obs, actions)
        current_q2 = self.critic_2(obs, actions)
        critic_loss = ((current_q1 - target_q).pow(2) * is_weights).mean() + ((current_q2 - target_q).pow(2) * is_weights).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        td_errors = torch.abs(current_q1.detach() - target_q).squeeze(-1).cpu().tolist()
        contributions = [float(row["contribution"]) for row in batch]
        self._update_replay_priorities(stage_name, batch_entries, td_errors, contributions)

        actor_loss: float | None = None
        if self.update_step % self.policy_delay == 0:
            if actor_variant == "single_stream_mlp":
                policy_actions = self.mlp(flat_obs)
                loss_tensor = -self.critic_1(obs, policy_actions).mean()
                self.mlp_optimizer.zero_grad()
                loss_tensor.backward()
                self.mlp_optimizer.step()
                self._soft_update(self.mlp, self.target_mlp)
            else:
                _, policy_actions = self.actor(obs)
                loss_tensor = -self.critic_1(obs, policy_actions).mean()
                self.actor_optimizer.zero_grad()
                loss_tensor.backward()
                self.actor_optimizer.step()
                self._soft_update(self.actor, self.target_actor)
            actor_loss = float(loss_tensor.item())
            self._soft_update(self.critic_1, self.target_critic_1)
            self._soft_update(self.critic_2, self.target_critic_2)
        return float(critic_loss.item()), actor_loss, summarize_stage_sources(batch_entries)

    def _sample_batch(self, stage_name: str, batch_size: int, use_stage_replay: bool) -> list[dict[str, object]]:
        if not use_stage_replay or stage_name == "guidance":
            return self.guidance_replay.sample_entries(batch_size)
        if stage_name == "avoidance":
            return self.avoidance_replay.sample_entries(
                batch_size,
                old_pool=self.guidance_replay.old_pool,
                old_fraction=self.avoidance_old_fraction,
            )
        return self.cooperation_replay.sample_entries(
            batch_size,
            old_pool=self.avoidance_replay.old_pool,
            old_fraction=self.cooperation_old_fraction,
        )

    def _update_replay_priorities(self, stage_name: str, sampled_entries: list[dict[str, object]], td_errors: list[float], contributions: list[float]) -> None:
        current_entries = []
        current_tds = []
        current_contribs = []
        for entry, td_error, contribution in zip(sampled_entries, td_errors, contributions):
            source = str(entry.get("source", ""))
            if source.startswith("guidance_old_pool") or source.startswith("avoidance_old_pool"):
                continue
            current_entries.append(entry)
            current_tds.append(td_error)
            current_contribs.append(contribution)
        if stage_name == "guidance":
            guidance_entries = [entry for entry in current_entries if str(entry.get("source", "")).startswith("guidance")]
            if guidance_entries:
                tree_indices = [int(entry["tree_index"]) for entry in guidance_entries]
                priorities = [max(td_error + contribution, 1e-4) for td_error, contribution in zip(current_tds[: len(guidance_entries)], current_contribs[: len(guidance_entries)])]
                self.guidance_replay.buffer.update_priorities(tree_indices, priorities)
        elif stage_name == "avoidance":
            self.avoidance_replay.update_priorities(current_entries, current_tds, current_contribs)
        else:
            self.cooperation_replay.update_priorities(current_entries, current_tds, current_contribs)

    def _evaluate_actor(
        self,
        actor_variant: str,
        episodes: int,
        difficulty: str | None = None,
        stage_index: int | None = None,
        curriculum_env: str | None = None,
    ) -> dict[str, float]:
        original = self.env.scenario
        if difficulty is not None or stage_index is not None or curriculum_env is not None:
            scenario = build_planning_scenario(
                difficulty or original.difficulty,
                stage_index=stage_index if stage_index is not None else original.stage_index,
                curriculum_env=curriculum_env,
            )
            self.env.set_scenario(replace(scenario, max_obstacles=original.max_obstacles, max_neighbors=original.max_neighbors))
        rows = []
        for _ in range(episodes):
            obs = self.env.reset()
            total_reward = 0.0
            done = False
            while not done:
                if actor_variant == "single_stream_mlp":
                    flat = torch.tensor(_flatten_obs(obs), dtype=torch.float32).unsqueeze(0)
                    action = self.mlp(flat).detach().cpu().numpy()[0]
                else:
                    structured = _to_torch(obs)
                    _, action_tensor = self.actor(structured)
                    action = action_tensor.detach().cpu().numpy()[0]
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
            metrics = compute_planning_metrics(
                collisions=self.env.collisions,
                occupancy_errors=self.env.occupancy_errors,
                formation_errors=self.env.formation_errors,
                success=self.env.success,
            )
            metrics["reward"] = float(total_reward)
            metrics["recovery_score"] = float(np.mean([max(0.0, 1.0 - risk) for risk in self.env.risk_history])) if self.env.risk_history else 0.0
            metrics["stage_index"] = float(self.env.scenario.stage_index)
            metrics["curriculum_env"] = self.env.scenario.curriculum_env
            for key in (
                "target_reward",
                "avoidance_reward",
                "collaboration_reward",
                "recovery_reward",
                "smoothness_penalty",
                "consistency_penalty",
                "success_bonus",
            ):
                metrics[key] = float(np.mean([row.get(key, 0.0) for row in self.env.reward_components])) if self.env.reward_components else 0.0
            rows.append(metrics)
        self.env.set_scenario(original)
        return _average(rows)

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def _compute_contribution(self, stage_name: str, previous_slot_error: float, next_slot_error: float, info: dict[str, float]) -> float:
        reach_gain = max(0.0, previous_slot_error - next_slot_error)
        if stage_name == "guidance":
            return reach_gain + 0.2 * float(info["success"])
        if stage_name == "avoidance":
            safety_recovery = max(0.0, 1.0 - float(info["risk"]))
            collision_margin = max(0.0, float(info["avoidance_reward"]))
            return 0.45 * reach_gain + 0.4 * safety_recovery + 0.15 * collision_margin
        formation_quality = max(0.0, 1.0 - float(info["formation_error"]))
        angular_quality = max(0.0, 1.0 - float(info["angle_error"]))
        safety_recovery = max(0.0, 1.0 - float(info["risk"]))
        return 0.25 * reach_gain + 0.4 * formation_quality + 0.25 * angular_quality + 0.1 * safety_recovery

    def _td_error_proxy(self, reward: float, info: dict[str, float], contribution: float) -> float:
        return (
            abs(float(reward)) * 0.25
            + float(info["risk"]) * 0.3
            + float(info["formation_error"]) * 0.15
            + float(info["angle_error"]) * 0.15
            + float(info["occupancy_error"]) * 0.1
            + contribution * 0.05
        )

    def _td_reference(self, stage_name: str) -> float:
        if stage_name == "avoidance" and self.avoidance_replay.td_history:
            return float(np.mean(self.avoidance_replay.td_history))
        if stage_name == "cooperation" and self.cooperation_replay.td_history:
            return float(np.mean(self.cooperation_replay.td_history))
        return 1.0

    def _gather_replay_stats(self, sample_sources: dict[str, int]) -> dict[str, object]:
        total_samples = max(1, sum(sample_sources.values()))
        return {
            "guidance": self.guidance_replay.stats(),
            "avoidance": self.avoidance_replay.stats(),
            "cooperation": self.cooperation_replay.stats(),
            "config": {
                "avoidance_old_fraction": self.avoidance_old_fraction,
                "cooperation_old_fraction": self.cooperation_old_fraction,
                "avoidance_sample_ratio": list(self.avoidance_sample_ratio),
                "cooperation_sample_ratio": list(self.cooperation_sample_ratio),
                "avoidance_priority_mode": self.avoidance_priority_mode,
                "cooperation_priority_mode": self.cooperation_priority_mode,
                "rare_priority_mode": self.rare_priority_mode,
            },
            "current_replay_type": self._current_replay_type(self.previous_stage_name),
            "last_sample_sources": sample_sources,
            "last_sample_source_fractions": {name: float(count / total_samples) for name, count in sample_sources.items()},
        }

    def _build_attention_proxy(self) -> dict[str, list[float]]:
        obstacle_attention = self.actor.last_attention.get("avoidance")
        collaboration_attention = self.actor.last_attention.get("collaboration")
        gate = self.actor.last_gate
        return {
            "obstacle_attention": obstacle_attention.mean(dim=1).squeeze(0).cpu().tolist() if obstacle_attention is not None and obstacle_attention.numel() else [0.0],
            "neighbor_attention": collaboration_attention.mean(dim=1).squeeze(0).cpu().tolist() if collaboration_attention is not None and collaboration_attention.numel() else [0.0],
            "gate_value": gate.reshape(-1).cpu().tolist() if gate is not None else [0.0],
        }

    def _current_replay_type(self, stage_name: str) -> str:
        if stage_name == "guidance":
            return "guidance_base_replay"
        if stage_name == "avoidance":
            return "avoidance_pyramid"
        return "cooperation_pyramid"

    def _summarize_stage_transitions(self, history: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
        if len(history) < 2:
            return []
        summary: list[dict[str, float | str]] = []
        previous = history[0]
        for current in history[1:]:
            prev_stage = str(previous.get("stage_name", ""))
            curr_stage = str(current.get("stage_name", ""))
            if prev_stage == curr_stage:
                previous = current
                continue
            summary.append(
                {
                    "from_stage": prev_stage,
                    "to_stage": curr_stage,
                    "from_env": str(previous.get("curriculum_env", "")),
                    "to_env": str(current.get("curriculum_env", "")),
                    "reward_delta": float(current.get("reward", 0.0)) - float(previous.get("reward", 0.0)),
                    "success_rate_delta": float(current.get("success_rate", 0.0)) - float(previous.get("success_rate", 0.0)),
                    "collision_rate_delta": float(current.get("collision_rate", 0.0)) - float(previous.get("collision_rate", 0.0)),
                    "occupancy_error_delta": float(current.get("occupancy_error", 0.0)) - float(previous.get("occupancy_error", 0.0)),
                }
            )
            previous = current
        return summary

def _to_torch(obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float32).unsqueeze(0) for key, value in obs.items()}


def _batch_obs(rows: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    return {
        "self_state": torch.tensor(np.stack([row["self_state"] for row in rows]), dtype=torch.float32),
        "obstacles": torch.tensor(np.stack([row["obstacles"] for row in rows]), dtype=torch.float32),
        "neighbors": torch.tensor(np.stack([row["neighbors"] for row in rows]), dtype=torch.float32),
    }


def _flatten_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([obs["self_state"], obs["obstacles"].reshape(-1), obs["neighbors"].reshape(-1)])


def _average(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    result: dict[str, float | str] = {}
    for key in rows[0].keys():
        values = [row[key] for row in rows]
        if all(isinstance(value, (int, float, np.floating)) for value in values):
            result[key] = float(np.mean(values))
        else:
            result[key] = str(values[-1])
    return result


def _unique_parameters(parameters: list[nn.Parameter]) -> list[nn.Parameter]:
    unique: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        identifier = id(parameter)
        if identifier in seen:
            continue
        seen.add(identifier)
        unique.append(parameter)
    return unique
