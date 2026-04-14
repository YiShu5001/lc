from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlExperimentConfig:
    primary_method: str = "ddpg_ladrc"
    difficulty: str = "medium"
    difficulty_levels: tuple[str, ...] = ("easy", "medium", "hard", "extreme")
    axes: tuple[str, ...] = ("x", "y", "z")
    episode_length: int = 100
    episodes: int = 6
    seed: int = 7
    seed_runs: int = 3
    compare_episodes: int = 6
    train_episodes: int = 12
    warmup_steps: int = 32
    batch_size: int = 128
    updates_per_step: int = 1
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    hidden_dim: int = 512
    dropout_p: float = 0.2
    tau: float = 0.05
    soft_update_interval: int = 20
    enhanced_stack_size: int = 4
    enhanced_n_step: int = 4
    enhanced_action_hold_steps: int = 4
    mddpg_shared_values: tuple[int, ...] = (1, 3, 5, 7, 10)
    export_reference_preview: bool = True
    reference_profile_mode: str = "piecewise_constant_velocity"
    output_subdir: str | None = None
    snapshot_interval: int = 10
    exploration_noise_schedule: str = "fixed"
    exploration_noise_start: float = 0.1
    exploration_noise_end: float = 0.1


@dataclass(frozen=True)
class LADRCAnchorParameters:
    b0: float
    wc: float
    k: float
    r: float

    @property
    def wo(self) -> float:
        return float(self.wc * self.k)


@dataclass(frozen=True)
class LADRCActionBounds:
    r: tuple[float, float]
    b0: tuple[float, float]
    wc: tuple[float, float]
    k: tuple[float, float]
    delta_r: tuple[float, float]
    delta_b0: tuple[float, float]
    delta_wc: tuple[float, float]
    delta_k: tuple[float, float]
    train_anchor: LADRCAnchorParameters
    fast_anchor: LADRCAnchorParameters
    steady_anchor: LADRCAnchorParameters


_XY_FAST_ANCHOR = LADRCAnchorParameters(b0=30.5, wc=1.5, k=11.0, r=10.0)
_XY_STEADY_ANCHOR = LADRCAnchorParameters(b0=30.5, wc=1.5, k=11.0, r=10.0)
_XY_TRAIN_ANCHOR = LADRCAnchorParameters(b0=24.3, wc=2.95, k=7.415254237288136, r=63.0)
_Z_FAST_ANCHOR = LADRCAnchorParameters(b0=100.0, wc=0.05, k=0.5, r=2.0)
_Z_STEADY_ANCHOR = LADRCAnchorParameters(b0=100.0, wc=0.05, k=0.5, r=2.0)
_Z_TRAIN_ANCHOR = LADRCAnchorParameters(b0=100.0, wc=0.05, k=0.5, r=2.0)


def get_axis_ladrc_anchors(axis: str) -> tuple[LADRCAnchorParameters, LADRCAnchorParameters]:
    if axis in {"x", "y"}:
        return _XY_FAST_ANCHOR, _XY_STEADY_ANCHOR
    if axis == "z":
        return _Z_FAST_ANCHOR, _Z_STEADY_ANCHOR
    raise KeyError(f"Unsupported axis: {axis}")


def get_axis_ladrc_action_bounds(axis: str) -> LADRCActionBounds:
    fast_anchor, steady_anchor = get_axis_ladrc_anchors(axis)
    if axis in {"x", "y"}:
        return LADRCActionBounds(
            r=(58.0, 68.0),
            b0=(20.0, 30.0),
            wc=(2.4, 3.5),
            k=(5.8, 9.2),
            delta_r=(-5.0, 5.0),
            delta_b0=(-4.3, 5.7),
            delta_wc=(-0.55, 0.55),
            delta_k=(-1.615254237288136, 1.784745762711864),
            train_anchor=_XY_TRAIN_ANCHOR,
            fast_anchor=fast_anchor,
            steady_anchor=steady_anchor,
        )
    if axis == "z":
        return LADRCActionBounds(
            r=(1.0, 4.0),
            b0=(85.0, 115.0),
            wc=(0.03, 0.08),
            k=(0.4, 0.8),
            delta_r=(-1.0, 1.0),
            delta_b0=(-10.0, 10.0),
            delta_wc=(-0.02, 0.02),
            delta_k=(-0.2, 0.2),
            train_anchor=_Z_TRAIN_ANCHOR,
            fast_anchor=fast_anchor,
            steady_anchor=steady_anchor,
        )
    raise KeyError(f"Unsupported axis: {axis}")
