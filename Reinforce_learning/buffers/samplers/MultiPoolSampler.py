from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from Reinforce_learning.buffers.BaseBuffer import BaseBuffer


class MultiPoolSampler:
    """Weighted sampler over multiple replay pools."""

    def __init__(self, pools: List[BaseBuffer], weights: Optional[List[float]] = None):
        self.pools = pools
        if weights is None:
            self.weights = [1.0 / len(pools)] * len(pools)
        else:
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]

    def sample(self, batch_size: int, use_priority: bool = True) -> Tuple:
        pool_sizes = [len(pool) for pool in self.pools]
        total_size = sum(pool_sizes)
        if total_size == 0:
            raise ValueError("No experiences are available across the replay pools.")

        requested = []
        remaining = batch_size
        for i, weight in enumerate(self.weights):
            if i == len(self.pools) - 1:
                size = min(pool_sizes[i], remaining)
            else:
                size = max(1, int(batch_size * weight))
                size = min(size, pool_sizes[i], remaining)
            requested.append(size)
            remaining -= size

        if remaining > 0:
            for i, pool_size in enumerate(pool_sizes):
                available = pool_size - requested[i]
                if available <= 0:
                    continue
                extra = min(available, remaining)
                requested[i] += extra
                remaining -= extra
                if remaining == 0:
                    break

        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []

        for pool, pool_batch_size in zip(self.pools, requested):
            if pool_batch_size <= 0 or len(pool) < pool_batch_size:
                continue

            result = pool.sample(pool_batch_size)
            if len(result) == 5:
                states, actions, rewards, next_states, dones = result
            elif len(result) == 7:
                states, actions, rewards, next_states, dones, _, _ = result
            else:
                continue

            all_states.append(states)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_next_states.append(next_states)
            all_dones.append(dones)

        if not all_states:
            raise ValueError("No replay pool could satisfy the requested sample size.")

        states = np.concatenate(all_states, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        rewards = np.concatenate(all_rewards, axis=0)
        next_states = np.concatenate(all_next_states, axis=0)
        dones = np.concatenate(all_dones, axis=0)
        return states, actions, rewards, next_states, dones

    def set_weights(self, weights: List[float]) -> None:
        if len(weights) != len(self.pools):
            raise ValueError(
                f"Weight count ({len(weights)}) does not match pool count ({len(self.pools)})."
            )

        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]
