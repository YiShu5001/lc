"""Compatibility bridge for the refactored multi-UAV planning model."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.lc.planning.models.multi_uav_model import (
    MultiUAVModel as RefactoredMultiUAVModel,
    MultiUAVModelConfig,
    SigmoidNormal,
)


class MultiUAVModel(RefactoredMultiUAVModel):
    """Accept both the new config-style constructor and common legacy kwargs."""

    def __init__(self, cfg: MultiUAVModelConfig | None = None, **legacy_kwargs: Any):
        config = cfg or MultiUAVModelConfig()

        legacy_to_config = {
            "self_state_dim": "self_dim",
            "state_dim": "self_dim",
            "obs_state_dim": "obstacle_dim",
            "obstacle_dim": "obstacle_dim",
            "neighbor_state_dim": "neighbor_dim",
            "node_dim": "neighbor_dim",
            "action_dim": "action_dim",
            "d_model": "embed_dim",
        }
        overrides = {
            target: legacy_kwargs[source]
            for source, target in legacy_to_config.items()
            if source in legacy_kwargs
        }
        if overrides:
            config = replace(config, **overrides)

        super().__init__(config)


__all__ = ["MultiUAVModel", "MultiUAVModelConfig", "SigmoidNormal"]
