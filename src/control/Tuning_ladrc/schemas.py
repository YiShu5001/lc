from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AxisLADRCParameters:
    axis: str
    b0: float
    wc: float
    k: float
    r: float = 30.0

    @property
    def wo(self) -> float:
        return float(self.wc * self.k)


@dataclass(frozen=True)
class ManualTargetProfile:
    axis: str
    mode: str = "hold_step_hold"
    initial_position: tuple[float, float, float] = (0.0, 0.0, 1.0)
    step_value: float = 0.15
    reverse_step_value: float | None = None
    fixed_axes: tuple[float, float] = (0.0, 1.0)
    segment_durations: tuple[float, ...] = (2.0, 2.0, 2.0)
    velocity_value: float = 0.12
    total_duration: float = 6.0
    hover_reference: float = 1.0


@dataclass(frozen=True)
class TuningCaseResult:
    axis: str
    mode: str
    parameter_file: str
    output_dir: str
    pid_metrics: dict[str, float] = field(default_factory=dict)
    ladrc_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class B0SweepResult:
    axis: str
    mode: str
    parameter_file: str
    output_dir: str
    sweep_rows: tuple[dict[str, float], ...] = field(default_factory=tuple)
    recommended_b0: float = 0.0


@dataclass(frozen=True)
class WCSweepResult:
    axis: str
    mode: str
    parameter_file: str
    output_dir: str
    fixed_b0: float = 0.0
    fixed_k: float = 4.0
    sweep_rows: tuple[dict[str, float], ...] = field(default_factory=tuple)
    recommended_wc: float = 0.0


@dataclass(frozen=True)
class ZAxisSpecializedTuningResult:
    output_dir: str
    recommended_b0: float
    recommended_wc: float
    recommended_k: float
    b0_rows: tuple[dict[str, float], ...] = field(default_factory=tuple)
    wc_rows: tuple[dict[str, float], ...] = field(default_factory=tuple)
    k_rows: tuple[dict[str, float], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class XAxisRefinedTuningResult:
    output_dir: str
    stage_a_rows: tuple[dict[str, float], ...] = field(default_factory=tuple)
    stage_b_rows: tuple[dict[str, float], ...] = field(default_factory=tuple)
    stage_c_rows: tuple[dict[str, float], ...] = field(default_factory=tuple)
    recommended_params: dict[str, float] = field(default_factory=dict)
    comparison_against_pid: dict[str, float | bool] = field(default_factory=dict)
    comparison_against_current_ladrc: dict[str, float | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class XAxisSteadyTuningResult:
    output_dir: str
    stage_a_rows: tuple[dict[str, float], ...] = field(default_factory=tuple)
    stage_b_rows: tuple[dict[str, float], ...] = field(default_factory=tuple)
    stage_c_rows: tuple[dict[str, float], ...] = field(default_factory=tuple)
    fast_params: dict[str, float] = field(default_factory=dict)
    steady_params: dict[str, float] = field(default_factory=dict)
    rl_ranges: dict[str, dict[str, float]] = field(default_factory=dict)
    comparison_against_fast_x: dict[str, float | bool] = field(default_factory=dict)
    comparison_against_pid: dict[str, float | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class XAxisDisturbedRescanResult:
    output_dir: str
    recommended_b0: float
    recommended_wc: float
    recommended_k: float
    b0_rows: tuple[dict[str, float], ...] = field(default_factory=tuple)
    wc_rows: tuple[dict[str, float], ...] = field(default_factory=tuple)
    k_rows: tuple[dict[str, float], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class XAxisRBalanceResult:
    output_dir: str
    fast_params: dict[str, float] = field(default_factory=dict)
    steady_params: dict[str, float] = field(default_factory=dict)
    recommended_r: float = 0.0
    sweep_rows: tuple[dict[str, float | bool], ...] = field(default_factory=tuple)
    rl_ranges: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class XAxisTaskTuningResult:
    output_dir: str
    task_type: str
    recommended_params: dict[str, float] = field(default_factory=dict)
    b0_rows: tuple[dict[str, float], ...] = field(default_factory=tuple)
    wc_rows: tuple[dict[str, float], ...] = field(default_factory=tuple)
    k_rows: tuple[dict[str, float], ...] = field(default_factory=tuple)
    local_rows: tuple[dict[str, float], ...] = field(default_factory=tuple)
    pid_metrics: dict[str, float] = field(default_factory=dict)
    ladrc_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class XAxisDisturbanceRefinedTuningResult:
    output_dir: str
    recommended_params: dict[str, float] = field(default_factory=dict)
    b0_rows: tuple[dict[str, float], ...] = field(default_factory=tuple)
    wc_rows: tuple[dict[str, float], ...] = field(default_factory=tuple)
    k_rows: tuple[dict[str, float], ...] = field(default_factory=tuple)
    local_rows: tuple[dict[str, float], ...] = field(default_factory=tuple)
    pid_metrics: dict[str, float] = field(default_factory=dict)
    ladrc_metrics: dict[str, float] = field(default_factory=dict)
    comparison_against_current: dict[str, float | bool] = field(default_factory=dict)
