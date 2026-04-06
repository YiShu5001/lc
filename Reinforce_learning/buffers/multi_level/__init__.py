"""Legacy multi-level buffer subpackage exports with lazy loading."""

from __future__ import annotations

from .BaseCoveragePool import BaseCoveragePool
from .DifficultyFocusPool import DifficultyFocusPool
from .KeyEventPool import KeyEventPool

__all__ = [
    "BaseCoveragePool",
    "DifficultyFocusPool",
    "KeyEventPool",
    "MultiLevelBuffer",
]


def __getattr__(name: str):
    if name == "MultiLevelBuffer":
        from .MultiLevelBuffer import MultiLevelBuffer

        return MultiLevelBuffer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
