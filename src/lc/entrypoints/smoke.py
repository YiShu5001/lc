from __future__ import annotations

from lc.control.experiments import run_control_comparison
from lc.integration.pipeline import run_bridge_experiment
from lc.planning.experiments import run_planning_comparison


if __name__ == "__main__":
    print({"control": run_control_comparison(), "planning": run_planning_comparison(), "bridge": run_bridge_experiment()})

