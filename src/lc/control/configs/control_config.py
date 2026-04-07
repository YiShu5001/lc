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
    batch_size: int = 16
    updates_per_step: int = 1
    enhanced_stack_size: int = 4
    enhanced_n_step: int = 4
    enhanced_action_hold_steps: int = 4
    mddpg_shared_values: tuple[int, ...] = (1, 3, 5, 7, 10)
    export_reference_preview: bool = True
    reference_profile_mode: str = "piecewise_constant_velocity"


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
    b0: tuple[float, float]
    wc: tuple[float, float]
    k: tuple[float, float]
    fixed_r: float
    fast_anchor: LADRCAnchorParameters
    steady_anchor: LADRCAnchorParameters


_XY_FAST_ANCHOR = LADRCAnchorParameters(b0=30.5, wc=1.5, k=11.0, r=10.0)
_XY_STEADY_ANCHOR = LADRCAnchorParameters(b0=4.0, wc=4.0, k=5.0, r=10.0)
_Z_FAST_ANCHOR = LADRCAnchorParameters(b0=100.0, wc=0.05, k=0.5, r=2.0)
_Z_STEADY_ANCHOR = LADRCAnchorParameters(b0=100.0, wc=0.05, k=0.5, r=2.0)


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
            b0=(1.35, 33.15),
            wc=(1.25, 4.25),
            k=(4.4, 11.6),
            fixed_r=10.0,
            fast_anchor=fast_anchor,
            steady_anchor=steady_anchor,
        )
    if axis == "z":
        return LADRCActionBounds(
            b0=(85.0, 115.0),
            wc=(0.03, 0.08),
            k=(0.4, 0.8),
            fixed_r=2.0,
            fast_anchor=fast_anchor,
            steady_anchor=steady_anchor,
        )
    raise KeyError(f"Unsupported axis: {axis}")
