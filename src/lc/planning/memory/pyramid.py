from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from random import Random
from typing import Any

import numpy as np


class SumTree:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.empty(capacity, dtype=object)
        self.write = 0
        self.size = 0

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data: object) -> int:
        data_index = self.write
        self.data[data_index] = data
        tree_index = data_index + self.capacity - 1
        self.update(tree_index, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return tree_index

    def update(self, tree_index: int, priority: float) -> None:
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get(self, value: float) -> tuple[int, float, object]:
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):
                data_index = parent - self.capacity + 1
                return parent, float(self.tree[parent]), self.data[data_index]
            if value <= self.tree[left]:
                parent = left
            else:
                value -= self.tree[left]
                parent = right


@dataclass
class PrioritizedReplayBuffer:
    capacity: int
    alpha: float = 0.6
    beta: float = 0.4
    beta_increment: float = 1e-3
    epsilon: float = 1e-5
    seed: int = 7

    def __post_init__(self) -> None:
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0
        self.rng = Random(self.seed)

    def add(self, sample: dict[str, Any], priority: float) -> int:
        adjusted = (abs(float(priority)) + self.epsilon) ** self.alpha
        self.max_priority = max(self.max_priority, adjusted)
        return self.tree.add(adjusted, sample)

    def sample(self, batch_size: int) -> list[dict[str, Any]]:
        if self.tree.size == 0 or batch_size <= 0:
            return []
        segment = self.tree.total / max(batch_size, 1)
        batch: list[dict[str, Any]] = []
        total = max(1, self.tree.size)
        self.beta = min(1.0, self.beta + self.beta_increment)
        for index in range(batch_size):
            start = segment * index
            end = segment * (index + 1)
            value = self.rng.uniform(start, end)
            tree_index, priority, payload = self.tree.get(value)
            probability = max(priority / max(self.tree.total, 1e-6), 1e-6)
            weight = (total * probability) ** (-self.beta)
            batch.append(
                {
                    "tree_index": tree_index,
                    "priority": priority,
                    "is_weight": weight,
                    "payload": payload,
                }
            )
        max_weight = max(item["is_weight"] for item in batch)
        for item in batch:
            item["is_weight"] = float(item["is_weight"] / max(max_weight, 1e-6))
        return batch

    def update_priorities(self, tree_indices: list[int], priorities: list[float]) -> None:
        for tree_index, priority in zip(tree_indices, priorities):
            adjusted = (abs(float(priority)) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, adjusted)
            self.tree.update(tree_index, adjusted)

    def export_payloads(self) -> list[dict[str, Any]]:
        return [row for row in self.tree.data[: self.tree.size] if row is not None]

    def priority_values(self) -> list[float]:
        if self.tree.size <= 0:
            return []
        start = self.capacity - 1
        end = start + self.tree.size
        return [float(value) for value in self.tree.tree[start:end]]

    def __len__(self) -> int:
        return self.tree.size


@dataclass
class GuidanceReplayMemory:
    capacity: int = 512
    old_pool_capacity: int = 128
    seed: int = 7

    def __post_init__(self) -> None:
        self.buffer = PrioritizedReplayBuffer(self.capacity, alpha=0.6, beta=0.4, seed=self.seed)
        self.old_pool: list[dict[str, Any]] = []
        self.sample_counts = {"current": 0, "old_pool": 0}
        self.rng = Random(self.seed + 17)

    def push(self, item: dict[str, Any], priority: float) -> None:
        self.buffer.add(item, priority)

    def sample_entries(self, batch_size: int) -> list[dict[str, Any]]:
        rows = self.buffer.sample(batch_size)
        self.sample_counts["current"] += len(rows)
        for row in rows:
            row["source"] = "guidance_current"
        return rows

    def refresh_old_pool(self) -> None:
        payloads = self.buffer.export_payloads()
        scored = sorted(
            payloads,
            key=lambda row: (
                float(row.get("success", 0.0)),
                float(row.get("reward", 0.0)),
                -float(row.get("occupancy_error", 1e9)),
            ),
            reverse=True,
        )
        self.old_pool = scored[: self.old_pool_capacity]

    def sample_old_pool(self, batch_size: int) -> list[dict[str, Any]]:
        if batch_size <= 0 or not self.old_pool:
            return []
        count = min(batch_size, len(self.old_pool))
        chosen = self.rng.sample(self.old_pool, k=count)
        self.sample_counts["old_pool"] += len(chosen)
        return [{"payload": row, "is_weight": 1.0, "source": "guidance_old_pool"} for row in chosen]

    def stats(self) -> dict[str, Any]:
        current_samples = self.sample_counts["current"]
        old_samples = self.sample_counts["old_pool"]
        total_samples = max(1, current_samples + old_samples)
        return {
            "buffer_size": len(self.buffer),
            "old_pool_size": len(self.old_pool),
            "sample_counts": dict(self.sample_counts),
            "old_pool_hit_rate": float(old_samples / total_samples),
            "current_hit_rate": float(current_samples / total_samples),
            "priority_mean": float(np.mean(self.buffer.priority_values())) if len(self.buffer) else 0.0,
        }

    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class StagePyramidReplayMemory:
    stage_name: str
    capacity: int = 512
    td_reference_window: int = 64
    old_pool_capacity: int = 128
    seed: int = 7
    layer_names: tuple[str, str, str] = field(init=False)
    sample_ratio: tuple[int, int, int] | None = None
    secondary_priority_mode: str = "hybrid"
    rare_priority_mode: str = "hybrid"

    def __post_init__(self) -> None:
        if self.stage_name == "avoidance":
            self.layer_names = ("td_layer", "filtered_layer", "rare_layer")
            default_ratio = (6, 3, 1)
        else:
            self.layer_names = ("td_layer", "contribution_layer", "rare_layer")
            default_ratio = (5, 3, 2)
        self.sample_ratio = self.sample_ratio or default_ratio
        self.buffers = {
            self.layer_names[0]: PrioritizedReplayBuffer(self.capacity, alpha=0.7, beta=0.4, seed=self.seed),
            self.layer_names[1]: PrioritizedReplayBuffer(self.capacity, alpha=0.6, beta=0.4, seed=self.seed + 1),
            self.layer_names[2]: PrioritizedReplayBuffer(self.capacity, alpha=0.8, beta=0.4, seed=self.seed + 2),
        }
        self.sample_counts = {name: 0 for name in self.layer_names}
        self.old_pool_sample_count = 0
        self.old_pool: list[dict[str, Any]] = []
        self.rng = Random(self.seed + 23)
        self.td_history: list[float] = []

    def push(
        self,
        item: dict[str, Any],
        *,
        td_error: float,
        previous_td_error: float,
        contribution: float,
        rare_event_score: float,
        success: bool,
    ) -> None:
        payload = {
            **item,
            "td_error": float(td_error),
            "previous_td_error": float(previous_td_error),
            "contribution": float(contribution),
            "rare_event_score": float(rare_event_score),
            "success": float(success),
        }
        self.buffers[self.layer_names[0]].add(payload, max(td_error, 1e-4))
        self.td_history.append(float(td_error))
        self.td_history = self.td_history[-self.td_reference_window :]

        if self.stage_name == "avoidance":
            if self._should_promote_to_filtered_layer(td_error, previous_td_error):
                filtered_priority = self._secondary_priority(td_error, contribution)
                self.buffers[self.layer_names[1]].add(payload, filtered_priority)
        else:
            contribution_priority = self._secondary_priority(td_error, contribution)
            self.buffers[self.layer_names[1]].add(payload, contribution_priority)

        if success and rare_event_score > 0.0:
            rare_priority = self._rare_priority(rare_event_score, td_error, contribution)
            self.buffers[self.layer_names[2]].add(payload, rare_priority)

    def sample_entries(self, batch_size: int, old_pool: list[dict[str, Any]] | None = None, old_fraction: float = 0.0) -> list[dict[str, Any]]:
        old_count = min(batch_size, int(round(batch_size * old_fraction)))
        current_count = max(0, batch_size - old_count)
        allocation = self._allocate_current(current_count)
        sampled: list[dict[str, Any]] = []
        for layer_name, count in allocation.items():
            rows = self.buffers[layer_name].sample(count)
            self.sample_counts[layer_name] += len(rows)
            for row in rows:
                row["layer"] = layer_name
                row["source"] = f"{self.stage_name}_current"
            sampled.extend(rows)
        if old_count > 0 and old_pool:
            chosen = self.rng.sample(old_pool, k=min(old_count, len(old_pool)))
            self.old_pool_sample_count += len(chosen)
            sampled.extend({"payload": row, "is_weight": 1.0, "source": f"{self.stage_name}_old_pool"} for row in chosen)
        return sampled

    def update_priorities(self, sampled_entries: list[dict[str, Any]], td_errors: list[float], contributions: list[float]) -> None:
        grouped_indices = {name: [] for name in self.layer_names}
        grouped_priorities = {name: [] for name in self.layer_names}
        for entry, td_error, contribution in zip(sampled_entries, td_errors, contributions):
            layer_name = str(entry.get("layer", ""))
            if layer_name not in grouped_indices:
                continue
            grouped_indices[layer_name].append(int(entry["tree_index"]))
            if layer_name == self.layer_names[0]:
                grouped_priorities[layer_name].append(max(td_error, 1e-4))
            elif layer_name == self.layer_names[1]:
                grouped_priorities[layer_name].append(self._secondary_priority(td_error, contribution))
            else:
                rare_score = float(entry["payload"].get("rare_event_score", 0.0))
                grouped_priorities[layer_name].append(self._rare_priority(rare_score, td_error, contribution))
        for layer_name in self.layer_names:
            self.buffers[layer_name].update_priorities(grouped_indices[layer_name], grouped_priorities[layer_name])

    def refresh_old_pool(self) -> None:
        secondary = self.buffers[self.layer_names[1]].export_payloads()
        tertiary = self.buffers[self.layer_names[2]].export_payloads()
        ranked = sorted(
            secondary + tertiary,
            key=lambda row: (
                float(row.get("success", 0.0)),
                float(row.get("contribution", 0.0)),
                float(row.get("rare_event_score", 0.0)),
            ),
            reverse=True,
        )
        self.old_pool = ranked[: self.old_pool_capacity]

    def stats(self) -> dict[str, Any]:
        sizes = {layer_name: len(buffer) for layer_name, buffer in self.buffers.items()}
        total = max(1, sum(sizes.values()))
        total_samples = max(1, sum(self.sample_counts.values()) + self.old_pool_sample_count)
        priority_means = {
            layer_name: float(np.mean(buffer.priority_values())) if len(buffer) else 0.0
            for layer_name, buffer in self.buffers.items()
        }
        return {
            "bucket_sizes": sizes,
            "bucket_fractions": {name: float(value / total) for name, value in sizes.items()},
            "sampling_counts": dict(self.sample_counts),
            "sampling_fractions": {name: float(value / total_samples) for name, value in self.sample_counts.items()},
            "old_pool_size": len(self.old_pool),
            "old_pool_sample_count": self.old_pool_sample_count,
            "old_pool_hit_rate": float(self.old_pool_sample_count / total_samples),
            "priority_means": priority_means,
            "sample_ratio": list(self.sample_ratio),
            "secondary_priority_mode": self.secondary_priority_mode,
            "rare_priority_mode": self.rare_priority_mode,
        }

    def __len__(self) -> int:
        return sum(len(buffer) for buffer in self.buffers.values())

    def _should_promote_to_filtered_layer(self, td_error: float, previous_td_error: float) -> bool:
        mean_td = float(np.mean(self.td_history)) if self.td_history else td_error
        threshold = max(1e-4, mean_td * 0.1)
        if td_error <= threshold or td_error <= previous_td_error * 0.2:
            return False
        upper_band = max(threshold * 2.0, mean_td * 0.85, previous_td_error * 0.85)
        return td_error <= upper_band

    def _allocate_current(self, batch_size: int) -> dict[str, int]:
        ratios = np.asarray(self.sample_ratio, dtype=float)
        ratios = ratios / max(ratios.sum(), 1.0)
        counts = np.floor(ratios * batch_size).astype(int)
        while counts.sum() < batch_size:
            counts[np.argmax(ratios - counts / max(batch_size, 1))] += 1
        return {layer_name: int(count) for layer_name, count in zip(self.layer_names, counts)}

    def _secondary_priority(self, td_error: float, contribution: float) -> float:
        mode = self.secondary_priority_mode
        if mode == "td_only":
            return max(td_error, 1e-4)
        if mode == "contribution_only":
            return max(contribution, 1e-4)
        if self.stage_name == "cooperation":
            return max(contribution + 0.25 * td_error, 1e-4)
        return max(0.55 * td_error + 0.45 * contribution, 1e-4)

    def _rare_priority(self, rare_event_score: float, td_error: float, contribution: float) -> float:
        if self.rare_priority_mode == "rare_only":
            return max(rare_event_score, 1e-4)
        return max(rare_event_score + 0.35 * contribution + 0.15 * td_error, 1e-4)


def summarize_stage_sources(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter()
    for row in rows:
        counter.update([str(row.get("source", "unknown"))])
    return dict(counter)
