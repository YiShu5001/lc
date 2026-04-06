from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ArtifactConfig:
    output_root: str = "outputs/control_pybullet"
    export_structured: bool = True
    export_legacy_logger: bool = True
    save_figures: bool = True


@dataclass(frozen=True)
class AxisTrainingConfig:
    axis: str
    initial_position: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fixed_axes: tuple[float, float] = (0.0, 1.0)
    primary_speed_range: tuple[float, float] = (0.3, 0.7)
    reverse_speed_range: tuple[float, float] = (-0.5, -0.2)
    stage_duration_range: tuple[float, float] = (0.8, 1.6)
    include_disturbance: bool = True
    disturbance_scale: float = 0.12
    disturbance_axis_bias: float = 1.0
    stage_count: int = 4


@dataclass(frozen=True)
class ControllerVariantConfig:
    name: str
    use_ladrc_position: bool
    use_ladrc_attitude: bool
    position_ladrc_axes: tuple[str, ...] = ()
    checkpoint_path: str | None = None


@dataclass(frozen=True)
class SingleAxisLADRCTuningConfig:
    coarse_b0_scales: tuple[float, ...] = (0.7, 0.85, 1.0, 1.15, 1.3)
    coarse_wc_offsets: tuple[float, ...] = (-1.4, -0.7, 0.0, 0.7, 1.4)
    coarse_k_offsets: tuple[float, ...] = (-0.8, -0.4, 0.0, 0.4, 0.8)
    fine_b0_scales: tuple[float, ...] = (0.9, 1.0, 1.1)
    fine_wc_offsets: tuple[float, ...] = (-0.35, 0.0, 0.35)
    fine_k_offsets: tuple[float, ...] = (-0.2, 0.0, 0.2)
    sensitivity_scales: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2)
    top_k: int = 5
    ranking_weights: dict[str, float] = field(
        default_factory=lambda: {
            "rmse": 0.35,
            "iae": 0.2,
            "steady_state_error": 0.15,
            "settling_time": 0.1,
            "control_variation": 0.1,
            "control_energy": 0.1,
        }
    )
    acceptable_degradation_ratio: float = 0.12
    tuning_difficulties: tuple[str, ...] = ("medium",)
    validation_difficulties: tuple[str, ...] = ("hard",)
    eval_episodes_per_candidate: int = 2
    rl_bounds_clip: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "b0": (0.2, 4.0),
            "omega_c": (0.5, 12.0),
            "k": (2.0, 6.0),
        }
    )


@dataclass(frozen=True)
class AxisTuningResult:
    axis: str
    controller_variant: str
    recommended_params: dict[str, float]
    rl_bounds: dict[str, float]
    coarse_rows: tuple[dict[str, float], ...]
    fine_rows: tuple[dict[str, float], ...]
    sensitivity_rows: dict[str, tuple[dict[str, float], ...]]
    pid_metrics: dict[str, float]
    best_metrics: dict[str, float]
    output_dir: str


@dataclass(frozen=True)
class PyBulletControlExperimentConfig:
    drone_model: str = "cf2x"
    simulation_freq_hz: int = 240
    control_freq_hz: int = 60
    rl_freq_hz: int = 10
    duration_sec: float = 5.0
    gui: bool = False
    seed: int = 7
    warmup_steps: int = 32
    train_episodes: int = 8
    eval_episodes: int = 3
    updates_per_step: int = 1
    batch_size: int = 32
    training_controller_variant: str = "ladrc_pos_pid_att"
    tuning: SingleAxisLADRCTuningConfig = field(default_factory=SingleAxisLADRCTuningConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)
    controller_variants: tuple[ControllerVariantConfig, ...] = field(
        default_factory=lambda: (
            ControllerVariantConfig("pid_pos_att", use_ladrc_position=False, use_ladrc_attitude=False),
            ControllerVariantConfig("ladrc_pos_pid_att", use_ladrc_position=True, use_ladrc_attitude=False),
            ControllerVariantConfig("ladrc_pos_att", use_ladrc_position=True, use_ladrc_attitude=True),
            ControllerVariantConfig("ladrc_x_pos_pid_att", use_ladrc_position=True, use_ladrc_attitude=False, position_ladrc_axes=("x",)),
            ControllerVariantConfig("ladrc_y_pos_pid_att", use_ladrc_position=True, use_ladrc_attitude=False, position_ladrc_axes=("y",)),
            ControllerVariantConfig("ladrc_z_pos_pid_att", use_ladrc_position=True, use_ladrc_attitude=False, position_ladrc_axes=("z",)),
        )
    )
    axis_configs: tuple[AxisTrainingConfig, ...] = field(
        default_factory=lambda: (
            AxisTrainingConfig(axis="x", initial_position=(0.0, 0.0, 1.0), fixed_axes=(0.0, 1.0)),
            AxisTrainingConfig(axis="y", initial_position=(0.0, 0.0, 1.0), fixed_axes=(0.0, 1.0)),
            AxisTrainingConfig(axis="z", initial_position=(0.0, 0.0, 1.0), fixed_axes=(0.0, 0.0)),
        )
    )

    @property
    def control_dt(self) -> float:
        return 1.0 / max(self.control_freq_hz, 1)

    @property
    def rl_dt(self) -> float:
        return 1.0 / max(self.rl_freq_hz, 1)

    @property
    def step_count(self) -> int:
        return max(int(self.duration_sec * self.control_freq_hz), 1)

    @property
    def rl_step_count(self) -> int:
        return max(int(self.duration_sec * self.rl_freq_hz), 1)

    @property
    def action_hold_steps(self) -> int:
        return max(self.control_freq_hz // max(self.rl_freq_hz, 1), 1)

    def axis_config(self, axis: str) -> AxisTrainingConfig:
        for config in self.axis_configs:
            if config.axis == axis:
                return config
        raise KeyError(f"Unsupported axis: {axis}")

    def controller_variant(self, name: str) -> ControllerVariantConfig:
        for variant in self.controller_variants:
            if variant.name == name:
                return variant
        raise KeyError(f"Unsupported controller variant: {name}")
