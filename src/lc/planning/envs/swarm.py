from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lc.envs.base import ActionSpec, BaseTaskEnv, ObservationSpec, TaskInfo
from lc.envs.scenarios import PlanningScenarioConfig
from lc.planning.rewards import compute_planning_reward


@dataclass
class PlanningSwarmEnv(BaseTaskEnv):
    scenario: PlanningScenarioConfig
    self_dim: int = 4
    obstacle_dim: int = 3
    neighbor_dim: int = 2
    horizon: int = 50
    step_dt: float = 0.1
    step_count: int = 0
    success: bool = False
    collisions: int = 0
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
    _target_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    _obstacle_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    _obstacle_velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    _neighbor_positions_state: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    _neighbor_velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    _previous_action: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    _previous_slot_error: float = 0.0
    _previous_formation_error: float = 0.0
    _previous_angle_error: float = 0.0
    _target_heading: float = 0.0

    @property
    def obs_spec(self) -> ObservationSpec:
        flat_dim = self.self_dim + self.scenario.max_obstacles * self.obstacle_dim + self.scenario.max_neighbors * self.neighbor_dim
        return ObservationSpec(shape=(flat_dim,), keys=("self_state", "obstacles", "neighbors"))

    @property
    def action_spec(self) -> ActionSpec:
        return ActionSpec(shape=(2,), low=-1.0, high=1.0)

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

    def reset(self) -> dict[str, np.ndarray]:
        self.step_count = 0
        self.success = False
        self.collisions = 0
        self.occupancy_errors.clear()
        self.formation_errors.clear()
        self.reward_components.clear()
        self.risk_history.clear()
        self.action_history.clear()
        self.trajectory.clear()
        self.target_trajectory.clear()
        self._formation_history.clear()
        self._angle_history.clear()
        self._position = np.zeros(2, dtype=np.float32)
        self._velocity = np.zeros(2, dtype=np.float32)
        self._target_position = self._build_target_position()
        self._target_heading = float(np.arctan2(self._target_position[1], self._target_position[0]))
        self._target_velocity = self._build_target_velocity()
        self._obstacle_positions, self._obstacle_velocities = self._build_obstacle_state()
        self._neighbor_positions_state, self._neighbor_velocities = self._build_neighbor_state()
        self._previous_action = np.zeros(2, dtype=np.float32)
        self._previous_slot_error = self._slot_error()
        self._previous_formation_error = self._compute_formation_error(np.zeros(2, dtype=np.float32))
        self._previous_angle_error = self._compute_angle_error()
        self.trajectory.append(self._position.tolist())
        self.target_trajectory.append(self._target_position.tolist())
        return self._make_observation()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict[str, float]]:
        self.step_count += 1
        clipped = np.clip(np.asarray(action, dtype=np.float32), self.action_spec.low, self.action_spec.high)
        speed = 0.22 + 0.12 * self.scenario.stage_index
        self._velocity = speed * clipped
        self._position = self._position + self._velocity * self.step_dt
        self._update_target_state()
        self._update_obstacle_state()
        self._update_neighbor_state()
        self.trajectory.append(self._position.tolist())
        self.target_trajectory.append(self._target_position.tolist())

        slot_error = self._slot_error()
        formation_error = self._compute_formation_error(clipped)
        obstacle_distance, obstacle_radius = self._nearest_obstacle()
        neighbor_distance = self._nearest_neighbor_distance()
        risk = float(
            0.45 * max(0.0, (obstacle_radius + 0.25 - obstacle_distance))
            + 0.35 * max(0.0, (0.35 - neighbor_distance))
            + 0.2 * min(1.0, slot_error)
        )
        collision = obstacle_distance <= obstacle_radius or neighbor_distance <= 0.12 or self._out_of_bounds()
        self.collisions += int(collision)
        self.occupancy_errors.append(slot_error)
        self.formation_errors.append(formation_error)
        self._formation_history.append(formation_error)
        self.risk_history.append(risk)
        self.action_history.append(clipped.copy())
        stage_name = self._stage_name()
        angle_error = self._compute_angle_error()
        self._angle_history.append(angle_error)
        success = self._is_success(stage_name, slot_error, formation_error, angle_error, collision)
        reward_breakdown = compute_planning_reward(
            stage_name=stage_name,
            occupancy_error=slot_error,
            previous_occupancy_error=self._previous_slot_error,
            formation_error=formation_error,
            angle_error=angle_error,
            obstacle_margin=max(0.0, obstacle_distance - obstacle_radius),
            neighbor_margin=max(0.0, neighbor_distance - 0.12),
            collision=collision,
            action=clipped,
            previous_action=self._previous_action,
            safe_action=self._previous_action if self.action_history else clipped,
            success=success and self.step_count >= self.horizon // 3,
        )
        self.reward_components.append(reward_breakdown.to_dict())
        self._previous_action = clipped.copy()
        self._previous_slot_error = slot_error
        self._previous_formation_error = formation_error
        self._previous_angle_error = angle_error
        done = collision or self.step_count >= self.horizon
        self.success = done and success
        info = {
            "collision": float(collision),
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
            "stage_index": float(self.scenario.stage_index),
            "stage_name": stage_name,
            "curriculum_env": self.scenario.curriculum_env,
            "rare_event_score": float(self._rare_recovery_score(risk, slot_error, formation_error, angle_error, success)),
            "success": float(success),
        }
        return self._make_observation(), reward_breakdown.total_reward, done, info

    def set_scenario(self, scenario: PlanningScenarioConfig) -> None:
        self.scenario = scenario

    def get_episode_trace(self) -> dict[str, list[float] | list[list[float]]]:
        return {
            "trajectory": list(self.trajectory),
            "target_trajectory": list(self.target_trajectory),
            "risk_history": list(self.risk_history),
            "occupancy_errors": list(self.occupancy_errors),
            "formation_errors": list(self.formation_errors),
        }

    def _make_observation(self) -> dict[str, np.ndarray]:
        target_vector = self._target_position - self._position
        self_state = np.array([target_vector[0], target_vector[1], self._velocity[0], self._velocity[1]], dtype=np.float32)
        obstacles = np.zeros((self.scenario.max_obstacles, self.obstacle_dim), dtype=np.float32)
        neighbors = np.zeros((self.scenario.max_neighbors, self.neighbor_dim), dtype=np.float32)
        for index in range(min(len(self._obstacle_positions), self.scenario.max_obstacles)):
            relative = self._obstacle_positions[index] - self._position
            obstacles[index] = np.array([relative[0], relative[1], 0.18], dtype=np.float32)
        for index in range(min(len(self._neighbor_positions_state), self.scenario.max_neighbors)):
            relative = self._neighbor_positions_state[index] - self._position
            neighbors[index] = np.array([relative[0], relative[1]], dtype=np.float32)
        return {"self_state": self_state, "obstacles": obstacles, "neighbors": neighbors}

    def _build_target_position(self) -> np.ndarray:
        band = self._sample_distance_band()
        ranges = {
            "near": (0.55, 0.85),
            "medium": (0.95, 1.35),
            "far": (1.45, 1.95),
        }
        low, high = ranges[band]
        distance = float(np.random.uniform(low, high) * self.scenario.world_scale)
        angle = float(np.random.uniform(-0.55, 0.55))
        return np.array([distance * np.cos(angle), distance * np.sin(angle)], dtype=np.float32)

    def _build_target_velocity(self) -> np.ndarray:
        if not self.scenario.target_is_dynamic:
            return np.zeros(2, dtype=np.float32)
        heading = self._target_heading + float(np.random.uniform(-0.25, 0.25))
        speed = self.scenario.target_speed_scale * 0.12
        return np.array([speed * np.cos(heading), speed * np.sin(heading)], dtype=np.float32)

    def _build_obstacle_state(self) -> tuple[np.ndarray, np.ndarray]:
        count = self.scenario.num_obstacles
        if count <= 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        world = self.scenario.world_scale
        positions: list[np.ndarray] = []
        velocities: list[np.ndarray] = []
        if self.scenario.obstacle_layout in {"sparse", "sparse_ring"}:
            for index in range(count):
                angle = 2.0 * np.pi * index / max(count, 1)
                radius = 0.6 * world + 0.15 * (index % 2)
                positions.append(np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float32))
        elif self.scenario.obstacle_layout in {"corridor", "dense_corridor", "clustered_corridor"}:
            for index in range(count):
                x = 0.4 * world + index * 0.28
                y = 0.22 * ((index % 3) - 1)
                positions.append(np.array([x, y], dtype=np.float32))
        else:
            for index in range(count):
                positions.append(
                    np.array(
                        [
                            np.random.uniform(0.3, 1.1) * world,
                            np.random.uniform(-0.6, 0.6) * world,
                        ],
                        dtype=np.float32,
                    )
                )
        for index in range(count):
            if not self.scenario.obstacle_is_dynamic:
                velocities.append(np.zeros(2, dtype=np.float32))
            else:
                direction = (-1.0) ** index
                velocities.append(np.array([0.0, direction * self.scenario.obstacle_speed_scale * 0.08], dtype=np.float32))
        return np.asarray(positions, dtype=np.float32), np.asarray(velocities, dtype=np.float32)

    def _build_neighbor_state(self) -> tuple[np.ndarray, np.ndarray]:
        count = max(0, self.scenario.num_uavs - 1)
        if count == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        positions = []
        velocities = []
        desired = self._desired_neighbor_slots()
        for slot in desired:
            jitter = np.array([np.random.uniform(-0.08, 0.08), np.random.uniform(-0.08, 0.08)], dtype=np.float32)
            positions.append(slot + jitter)
            velocities.append(np.zeros(2, dtype=np.float32))
        return np.asarray(positions, dtype=np.float32), np.asarray(velocities, dtype=np.float32)

    def _update_target_state(self) -> None:
        if not self.scenario.target_is_dynamic:
            return
        if self.scenario.target_motion == "linear":
            velocity = self._target_velocity
        elif self.scenario.target_motion == "curve":
            rotation = 0.08
            velocity = self._rotate(self._target_velocity, rotation)
            self._target_velocity = velocity
        else:
            if self.step_count % 8 == 0:
                delta = float(np.random.uniform(-0.7, 0.7))
                self._target_velocity = self._rotate(self._target_velocity, delta)
            velocity = self._target_velocity
        self._target_position = self._target_position + velocity * self.step_dt

    def _update_obstacle_state(self) -> None:
        if len(self._obstacle_positions) == 0 or not self.scenario.obstacle_is_dynamic:
            return
        world_limit = 1.4 * self.scenario.world_scale
        self._obstacle_positions = self._obstacle_positions + self._obstacle_velocities * self.step_dt
        for index in range(len(self._obstacle_positions)):
            for axis in range(2):
                if abs(self._obstacle_positions[index, axis]) > world_limit:
                    self._obstacle_velocities[index, axis] *= -1.0
                    self._obstacle_positions[index, axis] = np.clip(
                        self._obstacle_positions[index, axis],
                        -world_limit,
                        world_limit,
                    )

    def _update_neighbor_state(self) -> None:
        if len(self._neighbor_positions_state) == 0:
            return
        desired_slots = self._desired_neighbor_slots()
        for index, desired in enumerate(desired_slots):
            correction = desired - self._neighbor_positions_state[index]
            velocity = 0.55 * correction
            if self.scenario.obstacle_is_dynamic and len(self._obstacle_positions) > 0:
                nearest = self._obstacle_positions[np.argmin(np.linalg.norm(self._obstacle_positions - self._neighbor_positions_state[index], axis=1))]
                diff = self._neighbor_positions_state[index] - nearest
                distance = np.linalg.norm(diff)
                if distance < 0.38:
                    velocity += 0.08 * diff / max(distance, 1e-6)
            self._neighbor_velocities[index] = velocity.astype(np.float32)
            self._neighbor_positions_state[index] = self._neighbor_positions_state[index] + self._neighbor_velocities[index] * self.step_dt

    def _stage_name(self) -> str:
        return self.scenario.stage_name

    def _sample_distance_band(self) -> str:
        band = self.scenario.target_distance_band
        if band == "random":
            return str(np.random.choice(["near", "medium", "far"], p=[0.35, 0.4, 0.25]))
        return band

    def _desired_self_slot(self) -> np.ndarray:
        if self._stage_name() != "cooperation":
            return self._target_position
        radius = 0.5 + 0.04 * max(0, self.scenario.num_uavs - 3)
        return self._target_position + np.array([radius, 0.0], dtype=np.float32)

    def _desired_neighbor_slots(self) -> list[np.ndarray]:
        count = max(0, self.scenario.num_uavs - 1)
        if count == 0:
            return []
        if self._stage_name() != "cooperation":
            return []
        radius = 0.5 + 0.04 * max(0, self.scenario.num_uavs - 3)
        slots = []
        for index in range(1, self.scenario.num_uavs):
            angle = 2.0 * np.pi * index / max(self.scenario.num_uavs, 1)
            slots.append(self._target_position + radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32))
        return slots

    def _slot_error(self) -> float:
        return float(np.linalg.norm(self._desired_self_slot() - self._position))

    def _nearest_obstacle(self) -> tuple[float, float]:
        if len(self._obstacle_positions) == 0:
            return 10.0, 0.18
        distances = np.linalg.norm(self._obstacle_positions - self._position, axis=1)
        return float(np.min(distances)), 0.18

    def _nearest_neighbor_distance(self) -> float:
        if len(self._neighbor_positions_state) == 0:
            return 10.0
        distances = np.linalg.norm(self._neighbor_positions_state - self._position, axis=1)
        return float(np.min(distances))

    def _compute_formation_error(self, action: np.ndarray) -> float:
        if self._stage_name() != "cooperation":
            desired = self._target_position / max(np.linalg.norm(self._target_position), 1e-6)
            current = action / max(np.linalg.norm(action), 1e-6)
            return 0.4 * (1.0 - float(np.dot(desired, current)))
        slots = [self._desired_self_slot(), *self._desired_neighbor_slots()]
        positions = [self._position, *list(self._neighbor_positions_state)]
        if not slots or not positions:
            return 0.0
        errors = [np.linalg.norm(pos - slot) for pos, slot in zip(positions, slots)]
        return float(np.mean(errors) / max(self.scenario.world_scale, 1e-6))

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

    def _is_success(self, stage_name: str, slot_error: float, formation_error: float, angle_error: float, collision: bool) -> bool:
        if collision:
            return False
        if stage_name == "guidance":
            return slot_error <= 0.28
        if stage_name == "avoidance":
            return slot_error <= 0.34
        return slot_error <= 0.38 and formation_error <= 0.22 and angle_error <= 0.6

    def _out_of_bounds(self) -> bool:
        limit = 2.0 * self.scenario.world_scale
        return bool(np.any(np.abs(self._position) > limit))

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
        large_change = max(0.0, slot_change - 0.12)
        high_risk = max(0.0, risk - 0.45)
        cooperation_recovery = 0.6 * formation_change + 0.5 * angle_change
        return float(high_risk + large_change + cooperation_recovery)

    def _rotate(self, vector: np.ndarray, angle: float) -> np.ndarray:
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]], dtype=np.float32)
