from __future__ import annotations

from typing import Any


def render_report(title: str, payload: dict[str, Any]) -> str:
    lines = [f"# {title}", ""]
    for key, value in payload.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)

