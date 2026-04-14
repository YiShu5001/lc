from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lc.envs.scenarios import build_planning_scenario
from lc.planning.configs import LocalRiskCriticConfig, PlanningNetworkConfig, TransformerActorConfig, build_planning_network_config
from lc.planning.envs import PlanningSwarmEnv
from lc.planning.experiments.guidance_self_only_diagnostics import _action_field, _feature_probe, _obstacle_response_probe
from lc.planning.trainers import PlanningTrainer


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


def _build_network_config(
    *,
    q_head_dim: int = 512,
    mask_mode: str = "explicit",
    guidance_self_only: bool = True,
) -> PlanningNetworkConfig:
    cfg = build_planning_network_config("transformer_large")
    return replace(
        cfg,
        mask_mode=mask_mode,
        transformer=TransformerActorConfig(
            embed_dim=cfg.transformer.embed_dim,
            ff_dim=cfg.transformer.ff_dim,
            num_heads=cfg.transformer.num_heads,
            num_avoidance_layers=cfg.transformer.num_avoidance_layers,
            num_collab_layers=cfg.transformer.num_collab_layers,
            dropout=cfg.transformer.dropout,
            action_activation=cfg.transformer.action_activation,
            disable_collab_residual=cfg.transformer.disable_collab_residual,
            disable_explicit_mask=(mask_mode != "explicit"),
            guidance_self_only=guidance_self_only,
        ),
        local_risk_critic=LocalRiskCriticConfig(
            embed_dim=cfg.local_risk_critic.embed_dim,
            ff_dim=cfg.local_risk_critic.ff_dim,
            q_head_dim=q_head_dim,
            num_heads=cfg.local_risk_critic.num_heads,
            num_layers=cfg.local_risk_critic.num_layers,
            neighbor_fusion_mode=cfg.local_risk_critic.neighbor_fusion_mode,
        ),
    )


def _build_trainer(
    *,
    curriculum_env: str,
    mask_mode: str,
    guidance_self_only: bool,
    guidance_exploration_noise: float,
    exploration_noise: float,
    actor_action_reg_weight: float,
    updates_per_step: float,
    critic_lr: float,
    q_head_dim: int,
    tau: float,
    timeout_seconds: float,
    target_hold_radius: float,
    scenario_overrides: dict[str, Any] | None = None,
    env_overrides: dict[str, Any] | None = None,
    seed: int = 41,
) -> PlanningTrainer:
    cfg = _build_network_config(
        q_head_dim=q_head_dim,
        mask_mode=mask_mode,
        guidance_self_only=guidance_self_only,
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
        actor_action_reg_weight=actor_action_reg_weight,
        seed=seed,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def run_manual_block(
    output_dir: str | Path,
    *,
    block_id: int = 1,
    curriculum_env: str = "guidance_G1",
    episodes_per_block: int = 300,
    guidance_exploration_noise: float = 0.30,
    exploration_noise: float = 0.10,
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
) -> dict[str, Any]:
    completed_stages = list(completed_stages or [])
    root = Path(output_dir)
    run_dir = root / f"block_{block_id:03d}_{curriculum_env}"
    run_dir.mkdir(parents=True, exist_ok=True)

    trainer = _build_trainer(
        curriculum_env=curriculum_env,
        mask_mode=mask_mode,
        guidance_self_only=True,
        guidance_exploration_noise=guidance_exploration_noise,
        exploration_noise=exploration_noise,
        actor_action_reg_weight=actor_action_reg_weight,
        updates_per_step=updates_per_step,
        critic_lr=3e-4,
        q_head_dim=512,
        tau=0.02,
        timeout_seconds=8.0 if curriculum_env.startswith("guidance") else 10.0,
        target_hold_radius=0.15,
        scenario_overrides=scenario_overrides,
        env_overrides=env_overrides,
    )

    replay_size_before = len(trainer.replay_manager)
    if resume_from:
        trainer.load_training_state(
            Path(resume_from),
            load_weights=resume_weights,
            load_replay=resume_replay,
            load_optimizer=resume_optimizer,
        )
        replay_size_before = len(trainer.replay_manager)

    summary = trainer.train(
        episodes=episodes_per_block,
        use_curriculum=use_curriculum,
        use_pyramid_per=False,
        use_uniform_replay=True,
    )
    chunks = _chunk_summary(summary["history"])
    diagnostics_dir = run_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    stage_mode = str(trainer.current_stage_mode)
    action_field = _action_field(trainer, diagnostics_dir, f"block_{block_id:03d}_{curriculum_env}", stage_mode=stage_mode)
    feature_probe = _feature_probe(trainer, diagnostics_dir, f"block_{block_id:03d}_{curriculum_env}", stage_mode=stage_mode)
    obstacle_response_probe = (
        _obstacle_response_probe(trainer, diagnostics_dir, f"block_{block_id:03d}_{curriculum_env}", stage_mode=stage_mode)
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
        "curriculum_env": curriculum_env,
        "completed_stages": completed_stages,
        "episodes_per_block": int(episodes_per_block),
        "mask_mode": mask_mode,
        "use_curriculum": bool(use_curriculum),
        "guidance_self_only": bool(trainer.network_config.transformer.guidance_self_only),
        "guidance_exploration_noise": float(guidance_exploration_noise),
        "exploration_noise": float(exploration_noise),
        "actor_action_reg_weight": float(actor_action_reg_weight),
        "updates_per_step": float(updates_per_step),
        "critic_lr": 3e-4,
        "q_head_dim": 512,
        "tau": 0.02,
        "resume_from": str(resume_from) if resume_from else "",
        "scenario_overrides": dict(scenario_overrides or {}),
        "env_overrides": dict(env_overrides or {}),
        "replay_resume_enabled": bool(resume_replay),
        "replay_size_before_block": int(replay_size_before),
        "replay_size_after_block": int(len(trainer.replay_manager)),
        "training_counters": summary["training_counters"],
        "final_metrics": summary["final_metrics"],
        "action_field": action_field,
        "feature_probe": feature_probe,
        "obstacle_response_probe": obstacle_response_probe,
        "current_stage_eval": current_stage_eval,
        "legacy_task_performance": legacy_eval,
        "recommended_next_action": recommended_next_action,
        **q_stats,
        **tail,
    }

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
