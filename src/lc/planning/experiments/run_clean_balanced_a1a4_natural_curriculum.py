from __future__ import annotations

import json
from pathlib import Path

from .a1_skill_replay_critic04_timeout25_7way import DEFAULT_CRITIC04_G2_CHECKPOINT, DEFAULT_DEMO_DIR
from .manual_block_training import run_manual_block


NATURAL_A1A4_SEQUENCE = [
    "avoidance_A1_static_single",
    "avoidance_A2_static_multi",
    "avoidance_A3_dynamic_few",
    "avoidance_A4_dynamic_multi_target",
]


def run_clean_balanced_a1a4_natural_curriculum(
    output_root: str | Path = "outputs/planning/manual_block_training_shadow_runs/clean_balanced_a1a4_natural_curriculum",
    *,
    episodes_per_stage_cap: int = 800,
    run_name: str = "clean_balanced_a1a4_natural_curriculum",
    review_mix_enabled: bool = False,
) -> dict[str, object]:
    total_episode_budget = int(episodes_per_stage_cap) * len(NATURAL_A1A4_SEQUENCE)
    return run_manual_block(
        output_root,
        run_name=run_name,
        block_id=1,
        curriculum_env=NATURAL_A1A4_SEQUENCE[0],
        episodes_per_block=total_episode_budget,
        env_step_budget=None,
        guidance_exploration_noise=0.10,
        exploration_noise=0.15,
        avoidance_noise_schedule_enabled=True,
        avoidance_noise_stage_values=(0.20, 0.12, 0.06),
        avoidance_noise_stage_ratios=(0.30, 0.70),
        actor_action_reg_weight=1.0e-2,
        updates_per_step=0.05,
        mask_mode="explicit",
        use_curriculum=False,
        resume_from=DEFAULT_CRITIC04_G2_CHECKPOINT,
        resume_weights=True,
        resume_replay=False,
        resume_optimizer=False,
        completed_stages=["guidance_G1", "guidance_G2"],
        live_progress_every=1,
        route_save_every=0,
        window_summary_every=20,
        checkpoint_save_every=50,
        progress_heartbeat_steps=2000,
        transformer_num_avoidance_layers=3,
        transformer_num_collab_layers=1,
        critic_profile="critic04_action_token_l1",
        replay_backend="a1_skill",
        a1_skill_replay_config="clean_balanced",
        a1_skill_demo_bootstrap_dir=str(DEFAULT_DEMO_DIR),
        a1_skill_demo_decay_enabled=True,
        rolling_promotion_sequence=NATURAL_A1A4_SEQUENCE,
        rolling_promotion_window=50,
        rolling_promotion_min_successes=31 if review_mix_enabled else 40,
        rolling_promotion_max_duration_seconds=1e9,
        rolling_promotion_max_episodes_per_env=int(episodes_per_stage_cap),
        rolling_review_mix_enabled=bool(review_mix_enabled),
        rolling_review_current_sample_count=9,
        rolling_review_previous_sample_count=1,
        rolling_review_rollback_previous_failures=3,
        rolling_review_max_collision_rate=0.30,
        scenario_overrides={
            "a1_direct_block_ratio": 0.60,
        },
        env_overrides={
            "delta_v_max": 9.6,
        },
        seed=41,
    )


def run_clean_balanced_a1a4_review_mix(
    output_root: str | Path = "outputs/planning/manual_block_training_shadow_runs/clean_balanced_a1a4_review_mix",
    *,
    episodes_per_stage_cap: int = 800,
) -> dict[str, object]:
    return run_clean_balanced_a1a4_natural_curriculum(
        output_root=output_root,
        episodes_per_stage_cap=episodes_per_stage_cap,
        run_name="clean_balanced_a1a4_review_mix",
        review_mix_enabled=True,
    )


def main() -> None:
    result = run_clean_balanced_a1a4_review_mix()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
