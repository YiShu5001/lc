from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ObservationSpec:
    shape: tuple[int, ...]
    description: str = ""
    keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionSpec:
    shape: tuple[int, ...]
    low: float
    high: float
    description: str = ""


@dataclass(frozen=True)
class TaskInfo:
    name: str
    stage: str = ""
    difficulty: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentArtifact:
    name: str
    path: Path
    kind: str
