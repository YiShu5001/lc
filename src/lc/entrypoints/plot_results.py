from __future__ import annotations

from pathlib import Path

from lc.analysis.compare import load_summary
from lc.analysis.reports import render_report


if __name__ == "__main__":
    control_summary = Path("outputs") / "control" / "medium" / "summary.json"
    planning_summary = Path("outputs") / "planning" / "medium" / "stage_1" / "summary.json"
    payload = {
        "control": load_summary(control_summary) if control_summary.exists() else "missing",
        "planning": load_summary(planning_summary) if planning_summary.exists() else "missing",
    }
    print(render_report("Experiment Summary", payload))

