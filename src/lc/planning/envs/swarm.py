from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lc.envs.base import ActionSpec, BaseTaskEnv, ObservationSpec, TaskInfo
from lc.envs.scenarios import PlanningScenarioConfig
from lc.planning.rewards import compute_planning_reward


SMALL_OBSTACLE_RADIUS = 0.08
MEDIUM_OBSTACLE_RADIUS = 0.12
LARGE_OBSTACLE_RADIUS = 0.16


@dataclass
class PlanningSwarmEnv(BaseTaskEnv):
    scenario: PlanningScenarioConfig
    self_dim: int = 6
    obstacle_dim: int = 5
    neighbor_dim: int = 5
    action_limit: float = 0.8
    delta_v_max: float = 0.1
    obstacle_observation_limit: int = 4
    uav_collision_radius: float = 0.10
    obstacle_collision_radius: float = 0.18
    small_obstacle_radius: float = SMALL_OBSTACLE_RADIUS
    medium_obstacle_radius: float = MEDIUM_OBSTACLE_RADIUS
    large_obstacle_radius: float = LARGE_OBSTACLE_RADIUS
    physics_hz: int = 240
    control_hz: int = 48
    action_hz: int = 24
    horizon: int = 120
    step_dt: float = field(init=False, default=1.0 / 24.0)
    step_count: int = 0
    success: bool = False
    collisions: int = 0
    timed_out: bool = False
    out_of_bounds: bool = False
    failure_reason: str = "running"
    occupancy_errors: list[float] = field(default_factory=list)
    formation_errors: list[float] = field(default_factory=list)
    reward_components: list[dict[str, float]] = field(default_factory=list)
    risk_history: list[float] = field(default_factory=list)
    action_history: list[np.ndarray] = field(default_factory=list)
    trajectory: list[list[float]] = field(default_factory=list)
    target_trajectory: list[list[float]] = field(default_factory=list)
    _formation_history: list[float] = field(default_factory=list)
    _angle_history: list[float] = field(default_factory=list)
    _position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    _velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    _target_position: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], dtype=np.float32))
    _target_anchor: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], dtype=np.float32))
    _target_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    _target_phase: float = 0.0
    _obstacle_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    _obstacle_velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    _obstacle_radii: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _neighbor_positions_state: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    _neighbor_velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    _initial_obstacle_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    _initial_obstacle_radii: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _initial_neighbor_positions_state: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    _previous_action: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    _previous_slot_error: float = 0.0
    _previous_formation_error: float = 0.0
    _previous_angle_error: float = 0.0
    _hold_counter_steps: int = 0
    _encircle_counter_steps: int = 0
    _max_hold_counter_steps: int = 0
    _max_encircle_counter_steps: int = 0

    def __post_init__(self) -> None:
        if self.physics_hz <= 0 or self.control_hz <= 0 or self.action_hz <= 0:
            raise ValueError("physics_hz, control_hz, and action_hz must be positive")
        if self.physics_hz % self.control_hz != 0:
            raise ValueError("physics_hz must be divisible by control_hz")
        if self.control_hz % self.action_hz != 0:
            raise ValueError("control_hz must be divisible by action_hz")
        self.step_dt = 1.0 / float(self.action_hz)
        self.horizon = max(1, int(round(self.scenario.timeout_seconds / self.step_dt)))

    @property
    def physics_dt(self) -> float:
        return 1.0 / float(self.physics_hz)

    @property
    def control_dt(self) -> float:
        return 1.0 / float(self.control_hz)

    @property
    def action_dt(self) -> float:
        return 1.0 / float(self.action_hz)

    @property
    def physics_steps_per_control(self) -> int:
        return self.physics_hz // self.control_hz

    @property
    def control_steps_per_action(self) -> int:
        return self.control_hz // self.action_hz

    @property
    def obs_spec(self) -> ObservationSpec:
        flat_dim = (
            self.self_dim
            + self.scenario.max_obstacles * self.obstacle_dim
            + self.scenario.max_neighbors * self.neighbor_dim
            + self.scenario.max_obstacles
            + self.scenario.max_neighbors
        )
        return ObservationSpec(shape=(flat_dim,), keys=("self_state", "obstacles", "neighbors", "obstacle_mask", "neighbor_mask"))

    @property
    def action_spec(self) -> ActionSpec:
        return ActionSpec(shape=(2,), low=-self.action_limit, high=self.action_limit)

    @property
    def task_info(self) -> TaskInfo:
        return TaskInfo(
            name="planning_swarm_task",
            difficulty=self.scenario.difficulty,
            tags=(
                f"stage:{self.scenario.stage_name}",
                f"env:{self.scenario.curriculum_env}",
                f"uavs:{self.scenario.num_uavs}",
                f"obstacles:{self.scenario.num_obstacles}",
            ),
        )

    @property
    def hold_completion(self) -> float:
        required = max(1, int(round(self.scenario.target_hold_seconds / self.step_dt)))
        return min(1.0, self._max_hold_counter_steps / required)

    @property
    def encircle_completion(self) -> float:
        required = max(1, int(round(self.scenario.encircle_hold_seconds / self.step_dt)))
        return min(1.0, self._max_encircle_counter_steps / required)

    def reset(self) -> dict[str, np.ndarray]:
        self.step_dt = self.action_dt
        self.horizon = max(1, int(round(self.scenario.timeout_seconds / self.action_dt)))
        self.step_count = 0
        self.success = False
        self.collisions = 0
        self.timed_out = False
        self.out_of_bounds = False
        self.failure_reason = "running"
        self.occupancy_errors.clear()
        self.formation_errors.clear()
        self.reward_components.clear()
        self.risk_history.clear()
        self.action_history.clear()
        self.trajectory.clear()
        self.target_trajectory.clear()
        self._formation_history.clear()
        self._angle_history.clear()
        self._hold_counter_steps = 0
        self._encircle_counter_steps = 0
        self._max_hold_counter_steps = 0
        self._max_encircle_counter_steps = 0
        self._velocity = np.zeros(2, dtype=np.float32)
        self._previous_action = np.zeros(2, dtype=np.float32)

        self._initialize_scene()
        self._initial_obstacle_positions = self._obstacle_positions.copy()
        self._initial_obstacle_radii = self._obstacle_radii.copy()
        self._initial_neighbor_positions_state = self._neighbor_positions_state.copy()

        self._previous_slot_error = self._slot_error()
        self._previous_formation_error = self._compute_formation_error(np.zeros(2, dtype=np.float32))
        self._previous_angle_error = self._compute_angle_error()
        self.trajectory.append(self._position.tolist())
        self.target_trajectory.append(self._target_position.tolist())
        return self._make_observation()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict[str, float]]:
        self.step_count += 1
        desired_action = np.clip(np.asarray(action, dtype=np.float32), self.action_spec.low, self.action_spec.high)
        clipped = desired_action
        self._velocity = clipped.copy()
        for _ in range(self.control_steps_per_action):
            self._velocity = clipped.copy()
            for _ in range(self.physics_steps_per_control):
                self._position = self._position + self._velocity * self.physics_dt
                self._update_target_state(self.physics_dt)
                self._update_obstacle_state(self.physics_dt)
                self._update_neighbor_state(self.physics_dt)
        self.trajectory.append(self._position.tolist())
        self.target_trajectory.append(self._target_position.tolist())

        slot_error = self._slot_error()
        formation_error = self._compute_formation_error(clipped)
        angle_error = self._compute_angle_error()
        obstacle_distance, obstacle_radius, obstacle_clearance = self._nearest_obstacle()
        neighbor_distance, neighbor_clearance = self._nearest_neighbor_distance()
        boundary_distance = self._nearest_boundary_distance()
        self.out_of_bounds = self._out_of_bounds()
        collision = obstacle_distance <= obstacle_radius or neighbor_distance <= (2.0 * self.uav_collision_radius) or self.out_of_bounds
        self.collisions += int(collision)

        hold_satisfied = self._update_hold_state(collision)
        encircle_satisfied = self._update_encircle_state(collision)
        success = self._is_success(collision, hold_satisfied, encircle_satisfied)
        self.timed_out = self.step_count >= self.horizon and not success and not collision and not self.out_of_bounds
        done = bool(success or collision or self.out_of_bounds or self.timed_out)
        self.success = bool(success)
        self.failure_reason = self._failure_reason(success, collision)

        risk = self._compute_risk(slot_error, obstacle_clearance, neighbor_clearance)
        self.occupancy_errors.append(slot_error)
        self.formation_errors.append(formation_error)
        self._formation_history.append(formation_error)
        self.risk_history.append(risk)
        self.action_history.append(clipped.copy())
        self._angle_history.append(angle_error)

        reward_breakdown = compute_planning_reward(
            stage_name=self._stage_name(),
            occupancy_error=slot_error,
            previous_occupancy_error=self._previous_slot_error,
            formation_error=formation_error,
            angle_error=angle_error / np.pi,
            target_distance=float(np.linalg.norm(self._target_position - self._position)),
            obstacle_distance=obstacle_distance,
            neighbor_distance=neighbor_distance,
            boundary_distance=boundary_distance,
            obstacle_clearance=obstacle_clearance,
            neighbor_clearance=neighbor_clearance,
            obstacle_margin=max(0.0, obstacle_clearance),
            neighbor_margin=max(0.0, neighbor_clearance),
            collision=collision,
            out_of_bounds=self.out_of_bounds,
            action=clipped,
            previous_action=self._previous_action,
            safe_action=self._previous_action if self.action_history else clipped,
            success=success,
        )
        self.reward_components.append(reward_breakdown.to_dict())
        self._previous_action = clipped.copy()
        self._previous_slot_error = slot_error
        self._previous_formation_error = formation_error
        self._previous_angle_error = angle_error

        info = {
            "collision": float(collision),
            "timeout": float(self.timed_out),
            "out_of_bounds": float(self.out_of_bounds),
            "occupancy_error": slot_error,
            "formation_error": formation_error,
            "risk": risk,
            "recovery_score": float(max(0.0, 1.0 - risk)),
            "target_reward": reward_breakdown.target_reward,
            "avoidance_reward": reward_breakdown.avoidance_reward,
            "collaboration_reward": reward_breakdown.collaboration_reward,
            "recovery_reward": reward_breakdown.recovery_reward,
            "smoothness_penalty": reward_breakdown.smoothness_penalty,
            "consistency_penalty": reward_breakdown.consistency_penalty,
            "success_bonus": reward_breakdown.success_bonus,
            "angle_error": angle_error,
            "target_distance": float(np.linalg.norm(self._target_position - self._position)),
            "obstacle_center_distance": obstacle_distance,
            "obstacle_clearance": obstacle_clearance,
            "neighbor_center_distance": neighbor_distance,
            "neighbor_clearance": neighbor_clearance,
            "boundary_distance": boundary_distance,
            "hold_progress_seconds": float(self._hold_counter_steps * self.step_dt),
            "encircle_hold_progress_seconds": float(self._encircle_counter_steps * self.step_dt),
            "encircle_completion": float(self.encircle_completion),
            "hold_completion": float(self.hold_completion),
            "encircle_min_pair_distance": float(self._min_pair_distance()),
            "physics_hz": float(self.physics_hz),
            "control_hz": float(self.control_hz),
            "action_hz": float(self.action_hz),
            "stage_index": float(self.scenario.stage_index),
            "stage_name": self.scenario.stage_name,
            "curriculum_env": self.scenario.curriculum_env,
            "failure_reason": self.failure_reason,
            "success_mode": self.scenario.success_mode,
            "rare_event_score": float(self._rare_recovery_score(risk, slot_error, formation_error, angle_error, success)),
            "success": float(success),
        }
        return self._make_observation(), reward_breakdown.total_reward, done, info

    def set_scenario(self, scenario: PlanningScenarioConfig) -> None:
        self.scenario = scenario
        self.horizon = max(1, int(round(self.scenario.timeout_seconds / self.action_dt)))

    def get_episode_trace(self) -> dict[str, object]:
        snapshot = self.get_scene_snapshot()
        snapshot.update(
            {
                "trajectory": list(self.trajectory),
                "target_trajectory": list(self.target_trajectory),
                "risk_history": list(self.risk_history),
                "occupancy_errors": list(self.occupancy_errors),
                "formation_errors": list(self.formation_errors),
                "success": float(self.success),
                "collisions": float(self.collisions),
                "delta_v_max": float(self.delta_v_max),
                "uav_collision_radius": float(self.uav_collision_radius),
                "physics_hz": float(self.physics_hz),
                "control_hz": float(self.control_hz),
                "action_hz": float(self.action_hz),
            }
        )
        return snapshot

    def get_scene_snapshot(self) -> dict[str, object]:
        xmin, xmax, ymin, ymax = self._workspace_bounds()
        return {
            "stage_name": self.scenario.stage_name,
            "stage_label_zh": self.scenario.stage_label_zh,
            "curriculum_env": self.scenario.curriculum_env,
            "workspace_bounds": [xmin, xmax, ymin, ymax],
            "workspace_limit": float(max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))),
            "workspace_size": [self.scenario.workspace_size_x, self.scenario.workspace_size_y],
            "spawn_region": list(self.scenario.spawn_region),
            "spawn_region_label_zh": self.scenario.spawn_region_label_zh,
            "target_region_bounds": list(self.scenario.target_region_bounds),
            "target_region_mode": self.scenario.target_region_mode,
            "target_region_label_zh": self.scenario.target_region_label_zh,
            "trajectory": list(self.trajectory),
            "target_trajectory": list(self.target_trajectory),
            "target_position": self._target_position.tolist(),
            "uav_positions_initial": [self.trajectory[0], *self._initial_neighbor_positions_state.tolist()] if self.trajectory else [self._position.tolist()],
            "neighbor_positions_initial": self._initial_neighbor_positions_state.tolist(),
            "neighbor_positions_final": self._neighbor_positions_state.tolist(),
            "obstacle_positions_initial": self._initial_obstacle_positions.tolist(),
            "obstacle_positions_final": self._obstacle_positions.tolist(),
            "obstacle_radii_initial": self._initial_obstacle_radii.tolist(),
            "obstacle_safe_radii_initial": (self._initial_obstacle_radii + self.scenario.obstacle_safe_buffer).tolist(),
            "encircle_radius": float(self.scenario.encircle_radius),
            "target_is_dynamic": bool(self.scenario.target_is_dynamic),
            "target_trajectory_hint": self._build_target_trajectory_hint(),
            "success_mode": self.scenario.success_mode,
            "success": float(self.success),
            "collisions": float(self.collisions),
        }

    def _make_observation(self) -> dict[str, np.ndarray]:
        target_vector = self._target_position - self._position
        dist_left, dist_right, dist_bottom, dist_top = self._boundary_distances()
        wall_repulsion_x = max(0.0, 0.3 - dist_left) - max(0.0, 0.3 - dist_right)
        wall_repulsion_y = max(0.0, 0.3 - dist_bottom) - max(0.0, 0.3 - dist_top)
        distance_scale = max(self.scenario.workspace_size_x, self.scenario.workspace_size_y, 1e-6)
        self_state = np.array(
            [
                np.clip(target_vector[0] / distance_scale, -1.0, 1.0),
                np.clip(target_vector[1] / distance_scale, -1.0, 1.0),
                np.clip(self._velocity[0] / max(self.action_limit, 1e-6), -1.0, 1.0),
                np.clip(self._velocity[1] / max(self.action_limit, 1e-6), -1.0, 1.0),
                np.clip(wall_repulsion_x / 0.3, -1.0, 1.0),
                np.clip(wall_repulsion_y / 0.3, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        obstacles = np.zeros((self.scenario.max_obstacles, self.obstacle_dim), dtype=np.float32)
        neighbors = np.zeros((self.scenario.max_neighbors, self.neighbor_dim), dtype=np.float32)
        obstacle_mask = np.zeros(self.scenario.max_obstacles, dtype=np.float32)
        neighbor_mask = np.zeros(self.scenario.max_neighbors, dtype=np.float32)
        obstacle_indices = self._select_visible_obstacle_indices()
        for slot_index, obstacle_index in enumerate(obstacle_indices):
            relative = self._obstacle_positions[obstacle_index] - self._position
            relative_velocity = self._obstacle_velocities[obstacle_index] - self._velocity
            obstacles[slot_index] = np.array(
                [
                    relative[0],
                    relative[1],
                    relative_velocity[0],
                    relative_velocity[1],
                    self._obstacle_radii[obstacle_index],
                ],
                dtype=np.float32,
            )
            obstacle_mask[slot_index] = 1.0
        neighbor_indices = self._select_visible_neighbor_indices()
        for slot_index, neighbor_index in enumerate(neighbor_indices):
            relative = self._neighbor_positions_state[neighbor_index] - self._position
            relative_velocity = self._neighbor_velocities[neighbor_index] - self._velocity
            neighbors[slot_index] = np.array(
                [
                    relative[0],
                    relative[1],
                    relative_velocity[0],
                    relative_velocity[1],
                    self.uav_collision_radius,
                ],
                dtype=np.float32,
            )
            neighbor_mask[slot_index] = 1.0
        return {
            "self_state": self_state,
            "obstacles": obstacles,
            "neighbors": neighbors,
            "obstacle_mask": obstacle_mask,
            "neighbor_mask": neighbor_mask,
        }

    def _build_initial_positions(self) -> tuple[np.ndarray, np.ndarray]:
        if self.scenario.spawn_mode == "origin_single":
            return np.zeros(2, dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        if self.scenario.spawn_mode != "bottom_left_region_three_uavs":
            raise ValueError(f"Unsupported spawn_mode: {self.scenario.spawn_mode}")
        points = self._sample_points_in_rect(self.scenario.spawn_region, self.scenario.num_uavs, self.scenario.spawn_min_separation)
        return points[0], points[1:]

    def _build_target_position(self) -> np.ndarray:
        if self.scenario.target_region_mode == "workspace_random":
            xmin, xmax, ymin, ymax = self._workspace_bounds()
            margin = self.scenario.target_margin
            if self.scenario.stage_name != "guidance":
                x = float(np.random.uniform(xmin + margin, xmax - margin))
                y = float(np.random.uniform(ymin + margin, ymax - margin))
                return np.array([x, y], dtype=np.float32)
            origin = self._position.copy()
            min_distance = float(self.scenario.target_min_spawn_distance)
            max_distance = float(self.scenario.target_max_spawn_distance)
            for _ in range(64):
                x = float(np.random.uniform(xmin + margin, xmax - margin))
                y = float(np.random.uniform(ymin + margin, ymax - margin))
                candidate = np.array([x, y], dtype=np.float32)
                distance = float(np.linalg.norm(candidate - origin))
                if min_distance <= distance <= max_distance:
                    return candidate
            direction = np.array([1.0, 0.0], dtype=np.float32)
            candidate = origin + direction * np.clip(0.5 * (min_distance + max_distance), min_distance, max_distance)
            candidate[0] = float(np.clip(candidate[0], xmin + margin, xmax - margin))
            candidate[1] = float(np.clip(candidate[1], ymin + margin, ymax - margin))
            return candidate.astype(np.float32)
        if self.scenario.target_region_mode == "upper_right_triangle":
            left, right, bottom, top = self.scenario.target_region_bounds
            for _ in range(64):
                x = float(np.random.uniform(left, right))
                y = float(np.random.uniform(bottom, top))
                if (x - left) + (y - bottom) >= 0.7 * ((right - left) + (top - bottom)):
                    return np.array([x, y], dtype=np.float32)
            return np.array([0.75 * right, 0.75 * top], dtype=np.float32)
        if self.scenario.target_region_mode in {"upper_left_square", "upper_right_square"}:
            left, right, bottom, top = self.scenario.target_region_bounds
            x = float(np.random.uniform(left, right))
            y = float(np.random.uniform(bottom, top))
            return np.array([x, y], dtype=np.float32)
        raise ValueError(f"Unsupported target_region_mode: {self.scenario.target_region_mode}")

    def _build_target_velocity(self) -> np.ndarray:
        if not self.scenario.target_is_dynamic:
            return np.zeros(2, dtype=np.float32)
        heading = float(np.random.uniform(0.0, 2.0 * np.pi))
        speed = float(self.scenario.target_speed_scale)
        return np.array([speed * np.cos(heading), speed * np.sin(heading)], dtype=np.float32)

    def _build_obstacle_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = self._sample_obstacle_count()
        if count <= 0:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )
        radii = self._sample_obstacle_radii(count)
        layouts = self.scenario.obstacle_layout_modes or (self.scenario.obstacle_layout,)
        positions = np.zeros((count, 2), dtype=np.float32)
        for _ in range(80):
            layout = str(np.random.choice(layouts))
            positions = self._sample_obstacle_positions(layout, radii)
            if self._obstacles_valid(positions, radii):
                velocities = np.zeros((count, 2), dtype=np.float32)
                return positions.astype(np.float32), velocities, radii.astype(np.float32)
        for _ in range(256):
            positions = self._sample_random_obstacle_positions(radii)
            if self._obstacles_valid(positions, radii):
                velocities = np.zeros((count, 2), dtype=np.float32)
                return positions.astype(np.float32), velocities, radii.astype(np.float32)
        raise RuntimeError("Failed to sample a valid obstacle layout")

    def _sample_random_obstacle_positions(self, radii: np.ndarray) -> np.ndarray:
        xmin, xmax, ymin, ymax = self._workspace_bounds()
        positions: list[np.ndarray] = []
        for radius in radii:
            safe_radius = float(radius + self.scenario.obstacle_safe_buffer)
            candidate = np.array(
                [
                    np.random.uniform(xmin + safe_radius, xmax - safe_radius),
                    np.random.uniform(ymin + safe_radius, ymax - safe_radius),
                ],
                dtype=np.float32,
            )
            positions.append(candidate)
        return np.asarray(positions, dtype=np.float32)

    def _sample_obstacle_count(self) -> int:
        low, high = self.scenario.obstacle_count_range
        if high <= low:
            return int(low)
        return int(np.random.randint(low, high + 1))

    def _sample_obstacle_radii(self, count: int) -> np.ndarray:
        small = self._sample_count_range(self.scenario.obstacle_size_small_range)
        medium = self._sample_count_range(self.scenario.obstacle_size_medium_range)
        large = self._sample_count_range(self.scenario.obstacle_size_large_range)
        counts = [small, medium, large]
        radii_values = [self.small_obstacle_radius, self.medium_obstacle_radius, self.large_obstacle_radius]
        while sum(counts) < count:
            counts[int(np.argmin(counts))] += 1
        while sum(counts) > count:
            counts[int(np.argmax(counts))] -= 1
        radii: list[float] = []
        for radius, radius_count in zip(radii_values, counts):
            radii.extend([radius] * max(0, radius_count))
        np.random.shuffle(radii)
        return np.asarray(radii, dtype=np.float32)

    def _sample_count_range(self, bounds: tuple[int, int]) -> int:
        low, high = bounds
        if high <= low:
            return int(low)
        return int(np.random.randint(low, high + 1))

    def _sample_obstacle_positions(self, layout: str, radii: np.ndarray) -> np.ndarray:
        positions: list[np.ndarray] = []
        start = self._spawn_reference_point()
        target = self._target_position
        direction = target - start
        norm = max(float(np.linalg.norm(direction)), 1e-6)
        tangent = np.array([-direction[1], direction[0]], dtype=np.float32) / norm
        unit = direction / norm
        xmin, xmax, ymin, ymax = self._workspace_bounds()
        if layout == "path_offset":
            for index, radius in enumerate(radii):
                alpha = 0.35 + 0.4 * (index / max(len(radii) - 1, 1))
                offset = float(np.random.uniform(-0.22, 0.22))
                point = start + unit * alpha * norm + tangent * offset
                positions.append(self._clip_inside_workspace(point, float(radius)))
        elif layout == "path_center":
            for index, radius in enumerate(radii):
                alpha = 0.5 if len(radii) == 1 else 0.42 + 0.16 * (index / max(len(radii) - 1, 1))
                offset = float(np.random.uniform(-0.03, 0.03))
                point = start + unit * alpha * norm + tangent * offset
                positions.append(self._clip_inside_workspace(point, float(radius)))
        elif layout == "path_center_offset":
            for index, radius in enumerate(radii):
                alpha = 0.5 if len(radii) == 1 else 0.42 + 0.16 * (index / max(len(radii) - 1, 1))
                offset = float(np.random.uniform(-0.12, 0.12))
                point = start + unit * alpha * norm + tangent * offset
                positions.append(self._clip_inside_workspace(point, float(radius)))
        elif layout == "mid_left_block":
            center = 0.5 * (start + target) + tangent * 0.18
            positions = self._cluster_positions(center, radii, spread=0.28)
        elif layout == "mid_right_block":
            center = 0.5 * (start + target) - tangent * 0.18
            positions = self._cluster_positions(center, radii, spread=0.28)
        elif layout == "dual_side_block":
            midpoint = 0.5 * (start + target)
            half = max(1, len(radii) // 2)
            positions.extend(self._cluster_positions(midpoint + tangent * 0.28, radii[:half], spread=0.16))
            positions.extend(self._cluster_positions(midpoint - tangent * 0.28, radii[half:], spread=0.16))
        elif layout == "cluster_block":
            center = 0.55 * start + 0.45 * target
            positions = self._cluster_positions(center, radii, spread=0.22)
        else:
            for radius in radii:
                point = np.array(
                    [
                        np.random.uniform(xmin + radius, xmax - radius),
                        np.random.uniform(ymin + radius, ymax - radius),
                    ],
                    dtype=np.float32,
                )
                positions.append(point)
        return np.asarray(positions, dtype=np.float32)

    def _cluster_positions(self, center: np.ndarray, radii: np.ndarray, spread: float) -> list[np.ndarray]:
        positions: list[np.ndarray] = []
        for radius in radii:
            jitter = np.array([np.random.uniform(-spread, spread), np.random.uniform(-spread, spread)], dtype=np.float32)
            positions.append(self._clip_inside_workspace(center + jitter, float(radius)))
        return positions

    def _update_target_state(self, dt: float) -> None:
        if not self.scenario.target_is_dynamic:
            return
        self._target_phase += dt * 0.8
        desired_offset = np.array(
            [
                self.scenario.target_max_drift * np.cos(self._target_phase),
                0.65 * self.scenario.target_max_drift * np.sin(0.8 * self._target_phase),
            ],
            dtype=np.float32,
        )
        desired = self._target_anchor + desired_offset
        delta = desired - self._target_position
        distance = float(np.linalg.norm(delta))
        max_step = max(self.scenario.target_speed_scale * dt, 1e-6)
        if distance > max_step:
            delta = delta / max(distance, 1e-6) * max_step
        self._target_position = self._target_position + delta.astype(np.float32)
        self._target_position = self._clip_inside_target_region(self._target_position)

    def _update_obstacle_state(self, dt: float) -> None:
        if len(self._obstacle_positions) == 0 or not self.scenario.obstacle_is_dynamic:
            return
        self._obstacle_positions = self._obstacle_positions + self._obstacle_velocities * dt

    def _update_neighbor_state(self, dt: float) -> None:
        if len(self._neighbor_positions_state) == 0:
            return
        desired_slots = self._desired_neighbor_slots()
        for index, desired in enumerate(desired_slots):
            correction = desired - self._neighbor_positions_state[index]
            distance = float(np.linalg.norm(correction))
            if distance > 1e-6:
                velocity = correction / distance * min(0.5, distance * 0.9)
            else:
                velocity = np.zeros(2, dtype=np.float32)
            self._neighbor_velocities[index] = velocity.astype(np.float32)
            self._neighbor_positions_state[index] = self._neighbor_positions_state[index] + self._neighbor_velocities[index] * dt

    def _stage_name(self) -> str:
        return self.scenario.stage_name

    def _spawn_reference_point(self) -> np.ndarray:
        left, right, bottom, top = self.scenario.spawn_region
        return np.array([(left + right) / 2.0, (bottom + top) / 2.0], dtype=np.float32)

    def _desired_self_slot(self) -> np.ndarray:
        if self._stage_name() != "cooperation":
            return self._target_position
        return self._target_position + np.array([self.scenario.encircle_radius, 0.0], dtype=np.float32)

    def _desired_neighbor_slots(self) -> list[np.ndarray]:
        count = max(0, self.scenario.num_uavs - 1)
        if count == 0 or self._stage_name() != "cooperation":
            return []
        slots = []
        for index in range(1, self.scenario.num_uavs):
            angle = 2.0 * np.pi * index / max(self.scenario.num_uavs, 1)
            slots.append(
                self._target_position
                + self.scenario.encircle_radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            )
        return slots

    def _slot_error(self) -> float:
        if self._stage_name() == "cooperation":
            radius_error = abs(np.linalg.norm(self._position - self._target_position) - self.scenario.encircle_radius)
            return float(radius_error)
        return float(np.linalg.norm(self._target_position - self._position))

    def _nearest_obstacle(self) -> tuple[float, float, float]:
        if len(self._obstacle_positions) == 0:
            return 10.0, self.obstacle_collision_radius, 10.0
        distances = np.linalg.norm(self._obstacle_positions - self._position, axis=1)
        index = int(np.argmin(distances))
        collision_radius = float(self._obstacle_radii[index] + self.uav_collision_radius)
        clearance = float(distances[index] - collision_radius)
        return float(distances[index]), collision_radius, clearance

    def _select_visible_obstacle_indices(self) -> list[int]:
        if len(self._obstacle_positions) == 0:
            return []
        distances = np.linalg.norm(self._obstacle_positions - self._position, axis=1)
        count = min(len(self._obstacle_positions), self.scenario.max_obstacles, self.obstacle_observation_limit)
        return np.argsort(distances)[:count].tolist()

    def _select_visible_neighbor_indices(self) -> list[int]:
        if len(self._neighbor_positions_state) == 0:
            return []
        distances = np.linalg.norm(self._neighbor_positions_state - self._position, axis=1)
        count = min(len(self._neighbor_positions_state), self.scenario.max_neighbors)
        return np.argsort(distances)[:count].tolist()

    def _nearest_neighbor_distance(self) -> tuple[float, float]:
        if len(self._neighbor_positions_state) == 0:
            return 10.0, 10.0
        distances = np.linalg.norm(self._neighbor_positions_state - self._position, axis=1)
        center_distance = float(np.min(distances))
        clearance = center_distance - 2.0 * self.uav_collision_radius
        return center_distance, clearance

    def _boundary_distances(self) -> tuple[float, float, float, float]:
        xmin, xmax, ymin, ymax = self._workspace_bounds()
        return (
            float(self._position[0] - xmin),
            float(xmax - self._position[0]),
            float(self._position[1] - ymin),
            float(ymax - self._position[1]),
        )

    def _nearest_boundary_distance(self) -> float:
        return float(min(self._boundary_distances()))

    def _compute_formation_error(self, action: np.ndarray) -> float:
        if self._stage_name() != "cooperation":
            desired = self._target_position - self._position
            desired_norm = max(float(np.linalg.norm(desired)), 1e-6)
            action_norm = max(float(np.linalg.norm(action)), 1e-6)
            return float(0.5 * (1.0 - np.dot(desired / desired_norm, action / action_norm)))
        radii_errors = self._encircle_radius_errors()
        angle_error = self._compute_angle_error()
        spacing_error = max(0.0, self.scenario.encircle_min_spacing - self._min_pair_distance())
        return float(np.mean(radii_errors) + 0.2 * angle_error + spacing_error)

    def _encircle_radius_errors(self) -> list[float]:
        positions = [self._position, *list(self._neighbor_positions_state)]
        return [abs(float(np.linalg.norm(position - self._target_position)) - self.scenario.encircle_radius) for position in positions]

    def _compute_angle_error(self) -> float:
        if self.scenario.num_uavs <= 1 or self._stage_name() != "cooperation":
            return 0.0
        positions = [self._position, *list(self._neighbor_positions_state)]
        angles = []
        for position in positions:
            relative = position - self._target_position
            angles.append(float(np.arctan2(relative[1], relative[0])))
        angles = sorted(angles)
        gaps = np.diff(angles + [angles[0] + 2.0 * np.pi])
        target_gap = 2.0 * np.pi / max(self.scenario.num_uavs, 1)
        return float(np.mean(np.abs(gaps - target_gap)))

    def _update_hold_state(self, collision: bool) -> bool:
        if self.scenario.success_mode != "hold_at_target" or collision:
            self._hold_counter_steps = 0
            return False
        target_distance = float(np.linalg.norm(self._target_position - self._position))
        within = target_distance <= self.scenario.target_hold_radius
        self._hold_counter_steps = 1 if within else 0
        self._max_hold_counter_steps = max(self._max_hold_counter_steps, self._hold_counter_steps)
        return within

    def _update_encircle_state(self, collision: bool) -> bool:
        if self.scenario.success_mode != "encircle_target" or collision:
            self._encircle_counter_steps = 0
            return False
        radius_errors = self._encircle_radius_errors()
        angle_error_deg = float(np.degrees(self._compute_angle_error()))
        min_pair_distance = self._min_pair_distance()
        within_ring = all(error <= self.scenario.encircle_radius_tol for error in radius_errors)
        angle_ok = angle_error_deg <= self.scenario.encircle_angle_tol_deg
        spacing_ok = min_pair_distance >= self.scenario.encircle_min_spacing
        valid = within_ring and angle_ok and spacing_ok
        self._encircle_counter_steps = self._encircle_counter_steps + 1 if valid else 0
        self._max_encircle_counter_steps = max(self._max_encircle_counter_steps, self._encircle_counter_steps)
        required = max(1, int(round(self.scenario.encircle_hold_seconds / self.step_dt)))
        return self._encircle_counter_steps >= required

    def _is_success(self, collision: bool, hold_satisfied: bool, encircle_satisfied: bool) -> bool:
        if collision or self.out_of_bounds:
            return False
        if self.scenario.success_mode == "hold_at_target":
            return hold_satisfied
        return encircle_satisfied

    def _failure_reason(self, success: bool, collision: bool) -> str:
        if success:
            return "success"
        if collision and self.out_of_bounds:
            return "out_of_bounds"
        if collision:
            return "collision"
        if self.out_of_bounds:
            return "out_of_bounds"
        if self.timed_out:
            return "timeout"
        return "running"

    def _compute_risk(self, slot_error: float, obstacle_clearance: float, neighbor_clearance: float) -> float:
        return float(
            0.45 * max(0.0, 0.25 - obstacle_clearance)
            + 0.35 * max(0.0, self.scenario.encircle_min_spacing - neighbor_clearance)
            + 0.20 * min(1.0, slot_error)
        )

    def _out_of_bounds(self) -> bool:
        xmin, xmax, ymin, ymax = self._workspace_bounds()
        return bool(self._position[0] < xmin or self._position[0] > xmax or self._position[1] < ymin or self._position[1] > ymax)

    def _point_too_close_to_boundary(self, point: np.ndarray, margin: float) -> bool:
        xmin, xmax, ymin, ymax = self._workspace_bounds()
        return bool(
            point[0] < xmin + margin
            or point[0] > xmax - margin
            or point[1] < ymin + margin
            or point[1] > ymax - margin
        )

    def _out_of_workspace_point(self, point: np.ndarray) -> bool:
        xmin, xmax, ymin, ymax = self._workspace_bounds()
        return bool(point[0] < xmin or point[0] > xmax or point[1] < ymin or point[1] > ymax)

    def _workspace_bounds(self) -> tuple[float, float, float, float]:
        half_x = self.scenario.workspace_size_x / 2.0
        half_y = self.scenario.workspace_size_y / 2.0
        return -half_x, half_x, -half_y, half_y

    def _workspace_limit(self) -> float:
        xmin, xmax, ymin, ymax = self._workspace_bounds()
        return max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))

    def _min_pair_distance(self) -> float:
        positions = [self._position, *list(self._neighbor_positions_state)]
        if len(positions) < 2:
            return 10.0
        distances = []
        for index in range(len(positions)):
            for other_index in range(index + 1, len(positions)):
                distances.append(float(np.linalg.norm(positions[index] - positions[other_index])))
        return min(distances) if distances else 10.0

    def _obstacles_valid(self, positions: np.ndarray, radii: np.ndarray) -> bool:
        if len(positions) == 0:
            return True
        spawn_left, spawn_right, spawn_bottom, spawn_top = self.scenario.spawn_region
        target_left, target_right, target_bottom, target_top = self.scenario.target_region_bounds
        xmin, xmax, ymin, ymax = self._workspace_bounds()
        for index, position in enumerate(positions):
            safe_radius = float(radii[index] + self.scenario.obstacle_safe_buffer)
            x = float(position[0])
            y = float(position[1])
            if x - safe_radius < xmin or x + safe_radius > xmax or y - safe_radius < ymin or y + safe_radius > ymax:
                return False
            if spawn_left - safe_radius <= x <= spawn_right + safe_radius and spawn_bottom - safe_radius <= y <= spawn_top + safe_radius:
                return False
            if (
                self.scenario.target_region_mode != "workspace_random"
                and target_left - safe_radius <= x <= target_right + safe_radius
                and target_bottom - safe_radius <= y <= target_top + safe_radius
            ):
                return False
            target_clearance = float(np.linalg.norm(position - self._target_position) - radii[index])
            if target_clearance <= self.scenario.target_clearance_radius:
                return False
        for index in range(len(positions)):
            for other_index in range(index + 1, len(positions)):
                threshold = float(radii[index] + radii[other_index] + self.scenario.obstacle_safe_buffer)
                if np.linalg.norm(positions[index] - positions[other_index]) <= threshold:
                    return False
        if not self.scenario.path_feasibility_check:
            return True
        return self._coarse_path_exists(positions, radii)

    def _coarse_path_exists(self, positions: np.ndarray, radii: np.ndarray) -> bool:
        start = self._spawn_reference_point()
        target = self._target_position
        direction = target - start
        length = max(float(np.linalg.norm(direction)), 1e-6)
        tangent = np.array([-direction[1], direction[0]], dtype=np.float32) / length
        offsets = (0.0, 0.22, -0.22)
        for offset in offsets:
            blocked = False
            for alpha in np.linspace(0.1, 0.9, 12):
                point = start + direction * float(alpha) + tangent * float(offset)
                for position, radius in zip(positions, radii):
                    if np.linalg.norm(point - position) <= float(radius + self.scenario.obstacle_safe_buffer + self.uav_collision_radius):
                        blocked = True
                        break
                if blocked:
                    break
            if not blocked:
                return True
        return False

    def _sample_points_in_rect(self, rect: tuple[float, float, float, float], count: int, min_distance: float) -> np.ndarray:
        left, right, bottom, top = rect
        points: list[np.ndarray] = []
        for _ in range(256):
            candidate = np.array([np.random.uniform(left, right), np.random.uniform(bottom, top)], dtype=np.float32)
            if all(np.linalg.norm(candidate - point) >= min_distance for point in points):
                points.append(candidate)
                if len(points) == count:
                    return np.asarray(points, dtype=np.float32)
        if len(points) < count:
            while len(points) < count:
                points.append(np.array([left + 0.1 * len(points), bottom + 0.1 * len(points)], dtype=np.float32))
        return np.asarray(points, dtype=np.float32)

    def _clip_inside_workspace(self, point: np.ndarray, radius: float) -> np.ndarray:
        xmin, xmax, ymin, ymax = self._workspace_bounds()
        return np.array(
            [
                np.clip(point[0], xmin + radius, xmax - radius),
                np.clip(point[1], ymin + radius, ymax - radius),
            ],
            dtype=np.float32,
        )

    def _clip_inside_target_region(self, point: np.ndarray) -> np.ndarray:
        if self.scenario.target_region_mode == "workspace_random":
            xmin, xmax, ymin, ymax = self._workspace_bounds()
            margin = self.scenario.target_margin
            return np.array(
                [
                    np.clip(point[0], xmin + margin, xmax - margin),
                    np.clip(point[1], ymin + margin, ymax - margin),
                ],
                dtype=np.float32,
            )
        left, right, bottom, top = self.scenario.target_region_bounds
        clipped = np.array([np.clip(point[0], left, right), np.clip(point[1], bottom, top)], dtype=np.float32)
        if self.scenario.target_region_mode == "upper_right_triangle":
            if (clipped[0] - left) + (clipped[1] - bottom) < 0.7 * ((right - left) + (top - bottom)):
                clipped = np.array(
                    [
                        max(clipped[0], left + 0.55 * (right - left)),
                        max(clipped[1], bottom + 0.55 * (top - bottom)),
                    ],
                    dtype=np.float32,
                )
        return clipped

    def _rare_recovery_score(
        self,
        risk: float,
        slot_error: float,
        formation_error: float,
        angle_error: float,
        success: bool,
    ) -> float:
        if not success:
            return 0.0
        slot_change = abs(self._previous_slot_error - slot_error)
        formation_change = max(0.0, self._previous_formation_error - formation_error)
        angle_change = max(0.0, self._previous_angle_error - angle_error)
        return float(max(0.0, risk - 0.45) + slot_change + 0.5 * formation_change + 0.35 * angle_change)

    def _build_target_trajectory_hint(self) -> list[list[float]]:
        if not self.scenario.target_is_dynamic:
            return []
        points: list[np.ndarray] = [self._target_position.copy()]
        last_point = self._target_position.copy()
        for phase in np.linspace(0.35, 2.6, 7):
            offset = np.array(
                [
                    self.scenario.target_max_drift * np.cos(float(phase)),
                    0.65 * self.scenario.target_max_drift * np.sin(float(0.8 * phase)),
                ],
                dtype=np.float32,
            )
            candidate = self._clip_inside_target_region(self._target_position + offset)
            candidate = self._resolve_hint_candidate(last_point, candidate)
            points.append(candidate.copy())
            last_point = candidate
        return [point.tolist() for point in points]

    def _resolve_hint_candidate(self, previous: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        if self._hint_segment_is_safe(previous, candidate):
            return candidate
        for angle in np.linspace(0.35, 2.8, 8):
            rotated = self._target_position + self._rotate(candidate - self._target_position, float(angle))
            rotated = self._clip_inside_target_region(rotated)
            if self._hint_segment_is_safe(previous, rotated):
                return rotated
        midpoint = self._clip_inside_target_region((previous + candidate) / 2.0)
        if self._hint_segment_is_safe(previous, midpoint):
            return midpoint
        return previous.copy()

    def _hint_segment_is_safe(self, start: np.ndarray, end: np.ndarray) -> bool:
        if len(self._obstacle_positions) == 0:
            return True
        for obstacle, radius in zip(self._obstacle_positions, self._obstacle_radii):
            safe_radius = float(radius + self.scenario.obstacle_safe_buffer + 0.02)
            if self._segment_distance_to_point(start, end, obstacle) <= safe_radius:
                return False
        return True

    def _segment_distance_to_point(self, start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom <= 1e-8:
            return float(np.linalg.norm(point - start))
        t = float(np.dot(point - start, segment) / denom)
        t = max(0.0, min(1.0, t))
        projection = start + t * segment
        return float(np.linalg.norm(point - projection))

    def _rotate(self, vector: np.ndarray, angle: float) -> np.ndarray:
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]], dtype=np.float32)
    def _initialize_scene(self) -> None:
        for _ in range(64):
            position, neighbor_positions = self._build_initial_positions()
            self._position = position.astype(np.float32)
            self._neighbor_positions_state = neighbor_positions.astype(np.float32)
            self._neighbor_velocities = np.zeros_like(self._neighbor_positions_state, dtype=np.float32)
            self._target_position = self._build_target_position()
            self._target_anchor = self._target_position.copy()
            self._target_velocity = self._build_target_velocity()
            self._target_phase = float(np.random.uniform(0.0, 2.0 * np.pi))
            try:
                self._obstacle_positions, self._obstacle_velocities, self._obstacle_radii = self._build_obstacle_state()
            except RuntimeError:
                continue
            if self._initial_state_valid():
                return
        raise RuntimeError("Failed to sample a valid initial planning scene")

    def _initial_state_valid(self) -> bool:
        if self._out_of_workspace_point(self._position):
            return False
        if self._out_of_workspace_point(self._target_position):
            return False
        if not self._obstacles_valid(self._obstacle_positions, self._obstacle_radii):
            return False
        if self.scenario.stage_name == "guidance":
            if self._point_too_close_to_boundary(self._position, self.uav_collision_radius + self.scenario.workspace_safe_margin):
                return False
            if self._point_too_close_to_boundary(self._target_position, self.scenario.target_margin):
                return False
            target_distance = float(np.linalg.norm(self._target_position - self._position))
            if target_distance < self.scenario.target_min_spawn_distance or target_distance > self.scenario.target_max_spawn_distance:
                return False
        return True
