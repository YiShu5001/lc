from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from lc.common.types import ActionSpec, ObservationSpec, TaskInfo


class BaseTaskEnv(ABC):
    @property
    @abstractmethod
    def obs_spec(self) -> ObservationSpec:
        raise NotImplementedError

    @property
    @abstractmethod
    def action_spec(self) -> ActionSpec:
        raise NotImplementedError

    @property
    @abstractmethod
    def task_info(self) -> TaskInfo:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        raise NotImplementedError

