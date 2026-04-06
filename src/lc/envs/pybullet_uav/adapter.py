from __future__ import annotations

from dataclasses import dataclass

from lc.envs.scenarios import ControlScenarioConfig, PlanningScenarioConfig


@dataclass(frozen=True)
class UAVTaskAdapter:
    """Lightweight record describing the chosen UAV backend and scenario."""

    backend: str
    scenario: ControlScenarioConfig | PlanningScenarioConfig

