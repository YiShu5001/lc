from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpisodeSummary:
    episode: int
    reward: float
    metrics: dict[str, float] = field(default_factory=dict)

