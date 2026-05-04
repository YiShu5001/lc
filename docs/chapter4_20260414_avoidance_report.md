# Chapter 4 Avoidance Report (2026-04-14)

## Summary

Today's work focused on Chapter 4 planning-stage training, especially the transition from stable `guidance_G1/G2` into `avoidance_A1`.

The main conclusions are:

- `guidance_G1` and `guidance_G2` are stable under the current transformer + TD3 mainline.
- The original `avoidance_A1` failures were partly caused by environment-generation defects and overly permissive geometry.
- After fixing obstacle sampling and tightening scene semantics, a simplified `avoidance_A1` became trainable.
- The current best `avoidance_A1` result is still the run from `block_014_avoidance_A1`.
- A replay-reset restart with the new `path_center_offset` geometry (`block_015`) did **not** improve behavior; it regressed into timeout-dominant behavior with weak obstacle response.

The practical recommendation for the next large server run is:

- do **not** continue from `block_015`
- restart from `block_014_avoidance_A1`
- keep the new diagnostics and environment fixes
- test a slightly higher `exploration_noise` such as `0.25` rather than pushing the current bad local optimum

## Code Changes

The main implementation changes completed today are:

- `src/lc/planning/envs/swarm.py`
  - fixed obstacle generation retry/fallback logic
  - fixed target-clearance semantics to use `center_distance - obstacle_radius`
  - added obstacle radius overrides on the environment instance
  - added `path_center` and `path_center_offset` layouts
- `src/lc/planning/experiments/guidance_self_only_diagnostics.py`
  - action-field plots now draw obstacles
  - added stage-aware force-field generation
  - added `obstacle_response_probe.json` for avoidance diagnostics
- `src/lc/planning/experiments/manual_block_training.py`
  - added block-level diagnostics for:
    - `legacy_task_performance`
    - Q-value distribution
    - obstacle-response probe
    - recommended next action
  - supports resume modes where weights and replay are restored independently
- `tests/test_planning_network_configs.py`
  - added regression coverage for:
    - target-obstacle clearance semantics
    - `path_center`
    - `path_center_offset`

## Experiment Record

### Stable guidance stages

- `block_001_guidance_G1`
  - tail: `35 success / 0 out_of_bounds / 15 timeout`
- `block_002_guidance_G1`
  - tail: `50 success / 0 / 0`
- `block_003_guidance_G1`
  - tail: `50 success / 0 / 0`
- `block_004_guidance_G2`
  - tail: `50 success / 0 / 0`
  - `legacy guidance_G1 = 1.0`

This established the current guidance baseline as stable.

### Early avoidance attempts

- `block_005_avoidance_A1`
  - tail: `1 success / 49 collision`
- `block_006_avoidance_A1`
  - changing the wrong noise knob had no real effect
- `block_007_avoidance_A1`
  - tail: `2 success / 48 collision`

These runs showed that the original `A1` setup remained collision-dominant.

### Simplified avoidance with environment fixes

- `block_008_avoidance_A1`
  - simplified A1
  - tail: `7 success / 41 collision / 2 timeout`
- `block_009_avoidance_A1`
  - regressed into a bad local optimum
- `block_011_avoidance_A1`
  - higher avoidance exploration + smaller action step
  - reduced collisions, but produced too many timeouts and damaged legacy guidance
- `block_012_avoidance_A1`
  - fixed target-obstacle clearance
  - tail: `40 success / 1 out_of_bounds / 1 timeout / 8 collision`
  - guidance legacy remained intact

This was the first genuinely strong `avoidance_A1` result.

### Current best avoidance baseline

- `block_014_avoidance_A1`
  - resumed from `block_004_guidance_G2`
  - `exploration_noise = 0.20`
  - `actor_action_reg_weight = 3e-3`
  - `action_hz = 48`
  - simplified A1 geometry
  - tail: `40 success / 0 out_of_bounds / 1 timeout / 9 collision`
  - `legacy G1 = 1.0`
  - `legacy G2 = 1.0`
  - `q_value_mean = 5.346`
  - `q_value_comment = q_ordering_looks_reasonable`

This is the current best avoidance checkpoint and should be treated as the correct restart point.

### Replay-reset avoidance restart

- `block_015_avoidance_A1`
  - resumed from `block_014_avoidance_A1`
  - restored weights only
  - new replay buffer (`resume_replay = false`)
  - new geometry:
    - `path_center_offset`
    - `small_obstacle_radius = 0.06`
    - `target_clearance_radius = 0.30`
  - tail: `8 success / 0 out_of_bounds / 41 timeout / 1 collision`
  - `legacy G1 = 0.0`
  - `legacy G2 = 0.6`
  - `q_value_mean = -4.911`
  - `obstacle_left_vs_right_action_delta = 0.025`
  - `obstacle_center_vs_clear_action_delta = 0.048`

This run is not a good continuation point.

## Interpretation

The current state is:

- the model can finish guidance reliably
- the model can learn a simplified avoidance stage
- but the current replay-reset + new-geometry restart did not yet learn obstacle-conditioned behavior

The most important diagnostic from `block_015` is not the timeout count by itself; it is the weak obstacle-response probe. The actor still reacts primarily to target direction and only weakly to obstacle relocation. That means the network has not yet learned robust obstacle-conditioned routing.

The current bottleneck is therefore:

- not raw control frequency
- not target-reaching semantics
- not critic divergence

The bottleneck is:

- obstacle token usage remains weak under the current restart setup

## Recommendation For Next Server Run

For the next large run on the school server:

1. restart from `block_014_avoidance_A1`, not `block_015`
2. keep the code changes from today
3. keep the new diagnostics:
   - `legacy_task_performance`
   - Q distribution
   - obstacle response probe
4. use a fresh replay again only if intentionally testing replay reset
5. prefer a slightly higher avoidance exploration setting than `0.20`, for example:
   - `exploration_noise = 0.25`

Recommended first server experiment:

- start checkpoint: `block_014_avoidance_A1/trainer_state.pt`
- `resume_weights = True`
- `resume_replay = False`
- `exploration_noise = 0.25`
- `actor_action_reg_weight = 3e-3`
- `updates_per_step = 0.12`
- `action_hz = 48`
- `control_hz = 48`

If this still produces weak obstacle-response probe values, the next step should be reward-side obstacle-direction shaping rather than more blind episode accumulation.
